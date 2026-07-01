"""Per-event (sparse, 3D) segmentation dataset for **N-HOT3D**.

N-HOT3D is the neuromorphic (event-camera) version of Meta's HOT3D egocentric
hand-object recordings: a 346×260 event stream simulated/captured in the same FOV
as a 512×512 Aria RGB camera, shipped with **ground-truth per-frame hand masks**
(rendered from MANO). Unlike :class:`HandEventStreamDataset` over
NatureRoboticsDataNew — which distils SAM 2 *teacher* masks — the supervision here
is the dataset's own GT mask, so no teacher/SAM 2 is involved.

This class emits exactly the same per-sample contract as
:class:`HandEventStreamDataset` so it is a drop-in for the ``event_segmentation``
task (``EventSparseSeg`` + ``EventDistillationLoss`` + ``collate_sparse_events``):

    ``(coords[N,2] long (x,y), feats[N,C] float, times[N] float,
       labels[N] float, dense_mask[H,W] uint8, meta: dict)``

It **subclasses** ``HandEventStreamDataset`` purely to reuse the carefully tuned
per-event builders (``_build_events`` → features/labels/strided-subsample,
``_build_label_map`` → boundary trimap ignore, ``_empty_sample``, ``collate_fn``)
and the event-native ``EventAugmentor``. It does **not** call the parent
``__init__`` — N-HOT3D's storage, splitting and mask layout are entirely
different — and instead sets the handful of attributes those reused methods read.

On-disk layout (see ``docs`` / dataset notes)
---------------------------------------------
Root is the ``Aria/`` dir holding ``train/ valid/ test/`` (subject-disjoint splits
shipped with the dataset). ``purpose`` selects the split. Each sequence ``<TOKEN>``
(``P<subject>_<hash>``) is self-contained, but — because the dataset may still be
mid-copy — each modality lives in **either** the top-level ``<split>/<TOKEN>/`` dir
**or** the nested leaf ``<split>/<TOKEN>/<TOKEN>/`` dir. We resolve each modality to
whichever location actually has its files:

    <event_dir>/events_{i:010d}.h5      # /events (N,4) float64 = [t_sec, x, y, p]
    <mask_dir>/gt_hand_mask/gt_{i:010d}_{left,right}.jpg   # 512×512 JPEG-binary

One ``events_{i}.h5`` chunk holds one frame's events (~33 ms @ 30 fps). Frame index
``i`` is **1-based** and shared by events and masks. A frame is enumerated only when
both an event chunk and at least one (left/right) mask exist, so incomplete
sequences are tolerated.

Coordinate alignment
--------------------
Events are native ``event_width × event_height`` (default 346×260); masks/RGB are
``image_width × image_height`` (default 512×512) in the **same FOV**. Event ``(x,y)``
is rescaled by ``(image_w/event_w, image_h/event_h)`` so each event indexes the
resized mask — verified to land on the GT hand contour. ``label = mask_union[y, x]``
with the left/right masks OR-ed into one binary hand mask.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from data.event_augment import EventAugmentor
from data.hand_event_stream_dataset import HandEventStreamDataset

# events_0000000123.h5  (the merged `events.h5` has no index and is skipped)
_EVENT_RE = re.compile(r"^events_(\d{10})\.h5$")
# gt_0000000123_left.jpg / gt_0000000123_right.jpg
_MASK_RE = re.compile(r"^gt_(\d{10})_(left|right)\.jpg$")

#: split-name (purpose) -> on-disk subdirectory
_SPLIT_DIRS = {"train": "train", "validation": "valid", "test": "test"}


@dataclass
class _SeqEntry:
    """Resolved per-sequence handles built once at init (no event data loaded)."""
    token: str
    subject: str
    event_dir: Path
    mask_dir: Path
    index_set: set            # frame indices with both an event chunk and a mask

    @property
    def name(self) -> str:
        """Alias for ``token`` used by the dense validation-preview renderer
        (``model_interface._save_validation_previews`` reads
        ``val_set.sequences[s_idx].name``, matching the NatureRobotics datasets whose
        ``sequences`` are ``Path`` objects)."""
        return self.token


class NHOT3DEventDataset(HandEventStreamDataset):

    #: per-event features: signed polarity, normalized time-in-window (parent contract)
    NUM_FEATURES = 2

    def __init__(
        self,
        root_dir: str,
        purpose: str = "train",
        image_height: int = 512,
        image_width: int = 512,
        event_height: int = 260,
        event_width: int = 346,
        frames_per_sample: int = 1,
        max_events: int = 150000,
        require_mask: bool = True,
        subjects: Optional[List[str]] = None,
        max_sequences: Optional[int] = None,
        augmentation=None,
        boundary_ignore_px: int = 0,
        time_ignore_frac: float = 0.0,
        time_ignore_val: bool = False,
    ):
        # NOTE: intentionally NOT calling super().__init__ — the parent enumerates
        # NatureRoboticsDataNew (LOSO over ``sequence_*``, FLIR-PNG resolution
        # detection, big .npy event handles), none of which applies. We only reuse
        # the parent's per-event builders, so we set the attributes those read.
        torch.utils.data.Dataset.__init__(self)

        self.root = Path(root_dir)
        if not self.root.is_dir():
            raise FileNotFoundError(f"root_dir does not exist: {self.root}")
        self.purpose = purpose
        split_name = _SPLIT_DIRS.get(purpose)
        if split_name is None:
            raise ValueError(f"Unknown purpose {purpose!r}; expected one of {list(_SPLIT_DIRS)}")
        self.split_dir = self.root / split_name
        if not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"split dir for purpose={purpose!r} not found: {self.split_dir}"
            )

        # ---- output / native resolution and the event->mask rescale --------------
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.event_height = int(event_height)
        self.event_width = int(event_width)
        self._scale_x = self.image_width / self.event_width
        self._scale_y = self.image_height / self.event_height
        self._needs_rescale = (
            self.event_width != self.image_width
            or self.event_height != self.image_height
        )

        self.frames_per_sample = max(1, int(frames_per_sample))
        self.max_events = int(max_events)
        self.require_mask = bool(require_mask)

        # ---- label-refinement knobs read by the reused ``_build_events`` ---------
        # (boundary trimap ignore + window-edge ignore; train-only unless
        # ``time_ignore_val``). Mirrors HandEventStreamDataset semantics exactly.
        self.boundary_ignore_px = int(boundary_ignore_px)
        self.time_ignore_frac = float(time_ignore_frac)
        self.time_ignore_val = bool(time_ignore_val)
        self._label_refine_active = (
            (purpose == "train" or self.time_ignore_val)
            and (self.boundary_ignore_px > 0 or self.time_ignore_frac > 0.0)
        )

        # ---- event-native augmentation (train only) ------------------------------
        self._augmentor = EventAugmentor(dict(augmentation) if augmentation else None)
        self._event_aug_active = (purpose == "train" and self._augmentor.enabled)

        # ---- enumerate sequences + build the (seq, frame) index ------------------
        self._subjects = set(subjects) if subjects else None
        self.sequences: List[_SeqEntry] = []
        self.index: List[Tuple[int, int]] = []
        self._enumerate(max_sequences)

        if not self.index:
            raise RuntimeError(
                f"No usable (event-chunk + mask) frames under {self.split_dir} for "
                f"purpose={purpose!r}. The dataset may still be copying — check that "
                f"sequences contain events_*.h5 and gt_hand_mask/*.jpg (top-level or "
                f"nested <TOKEN>/ leaf dir)."
            )

    # ------------------------------------------------------------------ enumerate

    @staticmethod
    def _subject_of(token: str) -> str:
        return token.split("_", 1)[0]

    def _resolve_dir(self, base: Path, token: str, leaf: str, pattern: re.Pattern) -> Optional[Path]:
        """Return the first of ``base/<leaf>`` / ``base/<TOKEN>/<leaf>`` that holds a
        file matching ``pattern`` (handles the mid-copy "flattened vs nested" split).
        ``leaf == ""`` searches the dir itself (for event chunks)."""
        candidates = [base, base / token]
        for cand in candidates:
            d = cand / leaf if leaf else cand
            if not d.is_dir():
                continue
            try:
                for name in _iter_dir(d):
                    if pattern.match(name):
                        return d
            except OSError:
                continue
        return None

    def _scan_indices(self, d: Path, pattern: re.Pattern) -> Dict[int, Any]:
        """Map frame-index -> match info for every file in ``d`` matching ``pattern``."""
        out: Dict[int, Any] = {}
        for name in _iter_dir(d):
            m = pattern.match(name)
            if m:
                out.setdefault(int(m.group(1)), m)
        return out

    def _enumerate(self, max_sequences: Optional[int]) -> None:
        tokens = sorted(p.name for p in self.split_dir.iterdir() if p.is_dir())
        n_skipped = 0
        for token in tokens:
            if self._subjects is not None and self._subject_of(token) not in self._subjects:
                continue
            base = self.split_dir / token
            event_dir = self._resolve_dir(base, token, "", _EVENT_RE)
            mask_dir = self._resolve_dir(base, token, "gt_hand_mask", _MASK_RE)
            if event_dir is None or (self.require_mask and mask_dir is None):
                n_skipped += 1
                continue

            ev_idx = set(self._scan_indices(event_dir, _EVENT_RE).keys())
            mk_idx = (set(self._scan_indices(mask_dir, _MASK_RE).keys())
                      if mask_dir is not None else ev_idx)
            usable = sorted(ev_idx & mk_idx) if self.require_mask else sorted(ev_idx)
            if not usable:
                n_skipped += 1
                continue

            s_idx = len(self.sequences)
            self.sequences.append(_SeqEntry(
                token=token, subject=self._subject_of(token),
                event_dir=event_dir, mask_dir=mask_dir or event_dir,
                index_set=set(usable),
            ))
            for f in usable:
                self.index.append((s_idx, f))

            if max_sequences is not None and len(self.sequences) >= int(max_sequences):
                break

        self._n_skipped_sequences = n_skipped

    # ------------------------------------------------------------------ loading

    def _load_events_chunk(self, event_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
        """Read ``events_{i:010d}.h5`` -> ``(N, 4)`` float64 ``[t_sec, x, y, p]`` (or None)."""
        import h5py  # local import: keeps h5py optional for non-N-HOT3D paths
        path = event_dir / f"events_{frame_idx:010d}.h5"
        if not path.exists():
            return None
        with h5py.File(path, "r") as f:
            return np.asarray(f["events"][()], dtype=np.float64)

    def _load_union_mask(self, mask_dir: Path, frame_idx: int) -> torch.Tensor:
        """OR the left/right GT hand masks -> ``(H, W)`` float ``{0,1}`` at output res.

        Each side is a 512×512 JPEG-compressed binary mask (threshold ``>127`` to undo
        JPEG edge artefacts). A missing side is skipped; an all-zero union is a valid
        "no hand here" target (the dataset's null windows)."""
        H, W = self.image_height, self.image_width
        union = np.zeros((H, W), dtype=np.uint8)
        for side in ("left", "right"):
            p = mask_dir / f"gt_{frame_idx:010d}_{side}.jpg"
            if not p.exists():
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
            union |= (img > 127).astype(np.uint8)
        return torch.from_numpy(union.astype(np.float32))

    def _window_frames(self, seq: _SeqEntry, center: int) -> List[int]:
        """Contiguous frame indices around ``center`` for a multi-frame window
        (``frames_per_sample`` long), keeping only indices that exist in the seq."""
        if self.frames_per_sample <= 1:
            return [center]
        n_side = self.frames_per_sample // 2
        return [i for i in range(center - n_side, center + n_side + 1)
                if i in seq.index_set]

    # ------------------------------------------------------------------- sample

    def __getitem__(self, idx: int):
        s_idx, f_idx = self.index[idx]
        seq = self.sequences[s_idx]

        frames = self._window_frames(seq, f_idx)
        chunks = [c for c in (self._load_events_chunk(seq.event_dir, i) for i in frames)
                  if c is not None and c.size]
        mask = self._load_union_mask(seq.mask_dir, f_idx)  # supervision = center frame

        if not chunks:
            coords, feats, times, labels = self._empty_sample()
            t_center = 0.0
            n_events = 0
        else:
            ev = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
            t = ev[:, 0]
            xy = ev[:, 1:3]
            p = ev[:, 3]
            t_start = float(t.min())
            # +eps so the last event (t == t_max) survives ``_build_events``'s `< t_end`.
            t_end = float(t.max()) + 1e-6
            # Reuse the parent's vectorized per-event builder (rescale -> in-bounds
            # filter -> strided subsample -> [pol, t_norm] feats -> label sampling
            # incl. boundary/time ignore). Identical per-event contract by construction.
            coords, feats, times, labels = self._build_events(t, xy, p, t_start, t_end, mask)
            t_center = 0.5 * (t_start + t_end)
            n_events = int(ev.shape[0])

        if self._event_aug_active and coords.shape[0] > 0:
            seed = (idx * 2654435761) ^ int(torch.randint(0, 2 ** 62, (1,)).item())
            gen = torch.Generator().manual_seed(seed % (2 ** 63))
            coords, feats, times, labels = self._augmentor(
                coords, feats, times, labels, self.image_height, self.image_width, gen)

        dense_mask = (mask > 0.5).to(torch.uint8)

        meta = {
            "sequence": seq.token,
            "subject": seq.subject,
            "frame_index": f_idx,
            "t_center": t_center,
            "n_events": n_events,
            "n_kept": int(coords.shape[0]),
            "n_frames": len(frames),
        }
        return coords, feats, times, labels, dense_mask, meta


def _iter_dir(d: Path):
    """Names in ``d`` via os.scandir (one readdir batch — friendlier to NFS than glob)."""
    import os
    with os.scandir(d) as it:
        for entry in it:
            yield entry.name
