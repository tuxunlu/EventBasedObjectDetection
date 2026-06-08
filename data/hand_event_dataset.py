"""
Paired (event-voxel, SAM2-teacher-mask) dataset over the DVS_Actions
NatureRoboticsDataNew recordings.

Layout assumed under each sequence root:
    proc/
        events/{events_t, events_xy, events_p}.npy
        events/data.json
        flir/flir_t.npy
        flir/frame/{000000.png, ...}
        flir/data.json
        boundaries.json
        teacher_masks/{000000.png, ...}    # produced offline by sam2_pseudo_labels.py

Each (sequence, frame_index) becomes one sample. For frame `i`:
  - event window  = [flir_t[i] - dt/2, flir_t[i] + dt/2]
  - target mask   = teacher_masks/{i:06d}.png  (0 / 255 uint8)

Subject-disjoint splits use the leading subject token of the sequence name
(`sequence_<subj><session>_...` → `<subj>`). Default LOSO: a single held-out
subject for `validation` and `test`, the rest for `train`.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data as data

from data.event_representations import voxel_grid

SUBJECT_RE = re.compile(r"^sequence_([a-zA-Z]+)\d+_")


def _parse_subject(seq_name: str) -> str:
    m = SUBJECT_RE.match(seq_name)
    if not m:
        raise ValueError(f"Cannot parse subject from sequence name: {seq_name}")
    return m.group(1)


def _augment_pair(
    voxel: torch.Tensor,
    mask: torch.Tensor,
    cfg: Dict[str, Any],
    rng: random.Random,
    rgb: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Apply geometric augmentation jointly to voxel + mask (+ optional RGB).

    Voxel is resampled bilinearly (its signed polarity counts vary continuously);
    mask is resampled nearest-neighbour to stay binary; RGB (if present) is
    resampled bilinearly. All three share the same sampled transform parameters
    so the event input, the supervision, and the teacher input stay aligned.
    """
    from torchvision.transforms.v2 import functional as TF
    from torchvision.transforms.v2 import InterpolationMode

    if rng.random() < float(cfg.get("hflip_prob", 0.0)):
        voxel = TF.hflip(voxel)
        mask = TF.hflip(mask)
        if rgb is not None:
            rgb = TF.hflip(rgb)

    if rng.random() < float(cfg.get("affine_prob", 0.0)):
        rot_deg = float(cfg.get("rotate_deg", 0.0))
        scale_lo, scale_hi = cfg.get("scale_range", [1.0, 1.0])
        trans_frac = float(cfg.get("translate_frac", 0.0))
        h, w = int(mask.shape[-2]), int(mask.shape[-1])
        angle = rng.uniform(-rot_deg, rot_deg)
        scale = rng.uniform(float(scale_lo), float(scale_hi))
        tx = rng.uniform(-trans_frac, trans_frac) * w
        ty = rng.uniform(-trans_frac, trans_frac) * h
        voxel = TF.affine(
            voxel, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR, fill=0.0,
        )
        mask = TF.affine(
            mask.unsqueeze(0), angle=angle, translate=[tx, ty], scale=scale,
            shear=[0.0, 0.0], interpolation=InterpolationMode.NEAREST, fill=0.0,
        ).squeeze(0)
        if rgb is not None:
            rgb = TF.affine(
                rgb, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR, fill=0.0,
            )

    return voxel, mask, rgb


class HandEventDataset(data.Dataset):
    """Yields (event_voxel: (B,H,W), teacher_mask: (H,W), meta: dict).

    Parameters
    ----------
    root_dir : str
        Path containing the `sequence_*` recording folders.
    purpose : {"train", "validation", "test"}
        Selected by `DataInterface`. With the default LOSO split, the held-out
        subject populates both `validation` and `test`.
    voxel_bins : int
        Number of time bins in the voxel grid representation.
    window_ms : float
        Total event window centred on each FLIR frame timestamp, in
        milliseconds.
    image_height, image_width : int
        Sensor resolution. Default 480×640 matches the dataset.
    held_out_subject : Optional[str]
        Subject token to hold out for validation/test. If None, defaults to
        the alphabetically last subject.
    require_teacher : bool
        If True (default), only enumerate frames that have a cached
        teacher_masks/{i:06d}.png file. Set False when iterating the loader
        purely for input statistics.
    action_only : bool
        If True, restrict to frames inside the "action" segment of each
        clip's boundaries.json (drops the trailing "background" segment).
    mask_root : Optional[str]
        If set, look up teacher masks at <mask_root>/<sequence_name>/{i:06d}.png
        instead of <sequence>/proc/teacher_masks/. Use this when the dataset
        filesystem is read-only and masks live in scratch space.
    augmentation : Optional[Dict[str, Any]]
        Geometric augmentation config applied only when ``purpose == "train"``.
        Recognized keys (with defaults shown):

            enabled:        False
            hflip_prob:     0.0     # horizontal flip probability
            affine_prob:    0.0     # affine application probability
            rotate_deg:     0.0     # max |rotation| in degrees, sampled uniformly
            scale_range:    [1.0, 1.0]
            translate_frac: 0.0     # max |translation| as fraction of (H, W)

        Voxel resampling is bilinear, mask resampling is nearest-neighbour, so
        the binary supervision stays exact. Disabled (no-op) on val and test.
        RGB (if loaded) shares the same transform via bilinear interpolation.
    provide_rgb : bool
        If True, also load the paired FLIR PNG at ``proc/flir/frame/{i:06d}.png``
        and return it as a ``(3, H, W)`` float tensor in ``[0, 1]``. Used by
        the online feature-distillation path that runs SAM 2's image encoder on
        the RGB stream to align student event-domain features with the teacher.
        The sample becomes a 4-tuple ``(voxel, mask, rgb, meta)``; default 3-tuple
        contract is preserved when False.
    """

    POLARITY_MODE = "signed"  # voxel grid in [-1, 1] sense

    def __init__(
        self,
        root_dir: str,
        purpose: str = "train",
        voxel_bins: int = 5,
        window_ms: float = 36.0,
        image_height: int = 480,
        image_width: int = 640,
        held_out_subject: Optional[str] = None,
        require_teacher: bool = True,
        action_only: bool = False,
        mask_root: Optional[str] = None,
        augmentation: Optional[Dict[str, Any]] = None,
        provide_rgb: bool = False,
    ):
        super().__init__()
        self.root = Path(root_dir)
        if not self.root.is_dir():
            raise FileNotFoundError(f"root_dir does not exist: {self.root}")
        self.purpose = purpose
        self.voxel_bins = voxel_bins
        self.window_ms = window_ms
        self.image_height = image_height
        self.image_width = image_width
        self.require_teacher = require_teacher
        self.action_only = action_only
        self.mask_root = Path(mask_root) if mask_root else None

        # Augmentation only active on train; converted to a plain dict (the YAML
        # path may deliver an OmegaConf node, which doesn't accept .get cleanly
        # from worker processes).
        if augmentation is None:
            self.augmentation: Dict[str, Any] = {"enabled": False}
        else:
            self.augmentation = dict(augmentation)
        self._aug_active = (
            self.purpose == "train"
            and bool(self.augmentation.get("enabled", False))
        )
        self.provide_rgb = bool(provide_rgb)

        all_sequences = sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and p.name.startswith("sequence_")
        )
        if not all_sequences:
            raise RuntimeError(f"No sequence_* folders found under {self.root}")

        subjects = sorted({_parse_subject(p.name) for p in all_sequences})
        if held_out_subject is None:
            held_out_subject = subjects[-1]
        if held_out_subject not in subjects:
            raise ValueError(f"held_out_subject={held_out_subject!r} not in {subjects}")
        self.held_out_subject = held_out_subject

        if purpose == "train":
            keep = lambda s: _parse_subject(s.name) != held_out_subject
        elif purpose in ("validation", "test"):
            keep = lambda s: _parse_subject(s.name) == held_out_subject
        else:
            raise ValueError(f"Unknown purpose: {purpose}")
        self.sequences = [p for p in all_sequences if keep(p)]
        if not self.sequences:
            raise RuntimeError(
                f"No sequences left for purpose={purpose!r} with held_out={held_out_subject!r}"
            )

        # Lazy per-sequence handles, populated on first access in __getitem__.
        self._handles: Dict[str, Dict] = {}

        # Detect native sensor resolution from the first FLIR frame. When
        # image_height/image_width differ, event coordinates are rescaled
        # rather than clipped.
        self._native_h, self._native_w = self._detect_native_resolution()
        self._scale_x = self.image_width / self._native_w
        self._scale_y = self.image_height / self._native_h
        self._needs_rescale = (
            self._native_h != self.image_height
            or self._native_w != self.image_width
        )

        # Build the (seq_idx, frame_idx) index up front. Cheap — only reads
        # boundaries.json + checks for teacher mask files.
        self.index: List[Tuple[int, int]] = []
        for s_idx, seq in enumerate(self.sequences):
            n_frames = self._sequence_frame_count(seq)
            keep_frames = self._sequence_frame_mask(seq, n_frames)
            for f_idx in range(n_frames):
                if keep_frames[f_idx]:
                    self.index.append((s_idx, f_idx))
        if not self.index:
            raise RuntimeError(
                f"No usable frames; check teacher masks under {self.root}/*/proc/teacher_masks/"
            )

    # ------------------------------------------------------------------ utils

    def _detect_native_resolution(self) -> Tuple[int, int]:
        """Read a single FLIR PNG header to discover the sensor resolution."""
        for seq in self.sequences:
            frame_dir = seq / "proc" / "flir" / "frame"
            for p in sorted(frame_dir.glob("*.png"))[:1]:
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    return img.shape[0], img.shape[1]
        return self.image_height, self.image_width

    def _sequence_frame_count(self, seq_dir: Path) -> int:
        flir_t_path = seq_dir / "proc" / "flir" / "flir_t.npy"
        return int(np.load(flir_t_path, mmap_mode="r").shape[0])

    def _sequence_frame_mask(self, seq_dir: Path, n_frames: int) -> np.ndarray:
        keep = np.ones(n_frames, dtype=bool)

        if self.action_only:
            boundaries_path = seq_dir / "proc" / "boundaries.json"
            if boundaries_path.exists():
                boundaries = json.loads(boundaries_path.read_text())
                action_intervals = [
                    (b["flir_start_time"], b["flir_end_time"])
                    for b in boundaries if b["name"] != "background"
                ]
                if action_intervals:
                    flir_t = np.load(seq_dir / "proc" / "flir" / "flir_t.npy")
                    t0 = float(flir_t[0])
                    unit = self._detect_unit(flir_t)
                    t_rel = (flir_t.astype(np.float64) - t0) / unit
                    in_action = np.zeros(n_frames, dtype=bool)
                    for s, e in action_intervals:
                        in_action |= (t_rel >= s) & (t_rel < e)
                    keep &= in_action

        if self.require_teacher:
            mask_dir = self._mask_dir_for(seq_dir)
            if not mask_dir.exists():
                return np.zeros(n_frames, dtype=bool)
            present = np.array([
                (mask_dir / f"{i:06d}.png").exists() for i in range(n_frames)
            ])
            keep &= present

        return keep

    def _mask_dir_for(self, seq_dir: Path) -> Path:
        if self.mask_root is not None:
            return self.mask_root / seq_dir.name
        return seq_dir / "proc" / "teacher_masks"

    @staticmethod
    def _detect_unit(t: np.ndarray) -> float:
        """Heuristic: timestamp unit-per-second from total range."""
        rng = float(t[-1] - t[0])
        if rng > 1e9:
            return 1e9
        if rng > 1e6:
            return 1e6
        if rng > 1e3:
            return 1e3
        return 1.0

    def _get_handle(self, seq_dir: Path) -> Dict:
        key = str(seq_dir)
        handle = self._handles.get(key)
        if handle is not None:
            return handle
        proc = seq_dir / "proc"
        flir_t = np.load(proc / "flir" / "flir_t.npy")
        events_t = np.load(proc / "events" / "events_t.npy", mmap_mode="r")
        events_xy = np.load(proc / "events" / "events_xy.npy", mmap_mode="r")
        events_p = np.load(proc / "events" / "events_p.npy", mmap_mode="r")
        unit = self._detect_unit(events_t)
        handle = dict(
            flir_t=flir_t,
            events_t=events_t,
            events_xy=events_xy,
            events_p=events_p,
            unit=unit,
            mask_dir=self._mask_dir_for(seq_dir),
        )
        self._handles[key] = handle
        return handle

    # ------------------------------------------------------------------ data

    def __len__(self) -> int:
        return len(self.index)

    def _load_frame(self, seq_dir: Path, h: Dict, f_idx: int):
        """Load one frame's ``(voxel, mask, rgb, n_events, t_center)`` — no aug.

        Pure I/O + voxelization for a single ``(sequence, frame)``; the
        augmentation and meta-dict assembly live in ``__getitem__`` (single
        frame) / the clip dataset (shared-across-clip aug) so both can reuse
        this. ``rgb`` is ``None`` unless ``provide_rgb``.
        """
        t_center = float(h["flir_t"][f_idx])
        half = (self.window_ms / 1000.0) * h["unit"] / 2.0
        t_start = t_center - half
        t_end = t_center + half

        events_t = h["events_t"]
        lo = int(np.searchsorted(events_t, t_start, side="left"))
        hi = int(np.searchsorted(events_t, t_end, side="right"))
        t_slice = np.asarray(events_t[lo:hi])
        xy_slice = np.asarray(h["events_xy"][lo:hi])
        p_slice = np.asarray(h["events_p"][lo:hi])

        if self._needs_rescale and xy_slice.size > 0:
            xy_slice = xy_slice.astype(np.float32)
            xy_slice[:, 0] = xy_slice[:, 0] * self._scale_x
            xy_slice[:, 1] = xy_slice[:, 1] * self._scale_y

        voxel = voxel_grid(
            t=t_slice, xy=xy_slice, p=p_slice,
            t_start=t_start, t_end=t_end,
            bins=self.voxel_bins,
            height=self.image_height, width=self.image_width,
            device="cpu", signed=True,
        )

        mask_path = h["mask_dir"] / f"{f_idx:06d}.png"
        if self.require_teacher and not mask_path.exists():
            raise FileNotFoundError(f"Missing teacher mask {mask_path}")
        if mask_path.exists():
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise IOError(f"Could not read mask {mask_path}")
            if mask_img.shape != (self.image_height, self.image_width):
                mask_img = cv2.resize(mask_img,
                                      (self.image_width, self.image_height),
                                      interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy((mask_img > 127).astype(np.float32))
        else:
            mask = torch.zeros((self.image_height, self.image_width), dtype=torch.float32)

        rgb: Optional[torch.Tensor] = None
        if self.provide_rgb:
            rgb_path = seq_dir / "proc" / "flir" / "frame" / f"{f_idx:06d}.png"
            rgb_img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb_img is None:
                raise IOError(f"Could not read RGB frame {rgb_path}")
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            if rgb_img.shape[:2] != (self.image_height, self.image_width):
                rgb_img = cv2.resize(
                    rgb_img, (self.image_width, self.image_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            rgb = torch.from_numpy(rgb_img).permute(2, 0, 1).contiguous().float() / 255.0

        return voxel, mask, rgb, hi - lo, t_center

    def __getitem__(self, idx: int):
        s_idx, f_idx = self.index[idx]
        seq_dir = self.sequences[s_idx]
        h = self._get_handle(seq_dir)

        voxel, mask, rgb, n_events, t_center = self._load_frame(seq_dir, h, f_idx)

        if self._aug_active:
            # Per-sample RNG derived from idx + python's global seed (set per
            # worker by Lightning's seed_everything). Avoids cross-worker
            # correlation while staying reproducible at a fixed global seed.
            rng = random.Random((idx * 2654435761) ^ random.getrandbits(32))
            voxel, mask, rgb = _augment_pair(voxel, mask, self.augmentation, rng, rgb=rgb)

        meta = {
            "sequence": seq_dir.name,
            "frame_index": f_idx,
            "t_center": t_center,
            "n_events": n_events,
        }
        if self.provide_rgb:
            return voxel, mask, rgb, meta
        return voxel, mask, meta
