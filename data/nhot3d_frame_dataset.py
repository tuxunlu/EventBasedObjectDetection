"""Dense voxel-frame and temporal-clip N-HOT3D datasets for the DENSE event models.

:class:`~data.nhot3d_event_dataset.NHOT3DEventDataset` emits the SPARSE per-event
contract (``coords / feats / times / labels``) consumed by ``EventSparseSeg*`` and
``EventSSMSegStream`` (``task: event_segmentation``). The DENSE models instead consume
a Zhu-et-al **voxel grid** ``(voxel_bins, H, W)`` plus a dense ``(H, W)`` mask — exactly
like :class:`~data.hand_event_dataset.HandEventDataset` /
:class:`~data.hand_event_clip_dataset.HandEventClipDataset` over NatureRoboticsDataNew:

  * :class:`NHOT3DFrameDataset` → ``(voxel (C,H,W), mask (H,W), meta)`` for
    ``EventUnet`` / ``EventJEPAFrame`` (``task: segmentation``,
    ``model_interface._segmentation_step``).
  * :class:`NHOT3DClipDataset`  → ``(voxel (T,C,H,W), mask (T,H,W), meta)`` for
    ``EventTrackUnet`` (``task: tracking``, ``model_interface._tracking_step``); also
    exposes ``frame_sample`` for the stateful validation preview.

Both SUBCLASS ``NHOT3DEventDataset`` to reuse its carefully-resolved on-disk
enumeration (the mid-copy flattened-vs-nested layout handling), the subject-disjoint
``train/valid/test`` split, the per-frame ``events_*.h5`` chunk loader and the OR-ed
left/right GT-mask union. The ONLY thing they change is the per-sample REPRESENTATION:
sparse events → a dense voxel grid (events rescaled from the native 346×260 sensor to
the output ``image_*`` grid exactly as in the sparse path, then bilinearly binned by
:func:`data.event_representations.voxel_grid`).

Augmentation here is the DENSE geometric :func:`data.hand_event_dataset._augment_pair`
(voxel resampled bilinearly, mask nearest) — NOT the event-native ``EventAugmentor`` the
sparse dataset uses — so configs supply the dense aug keys (``hflip_prob`` /
``affine_prob`` / ``rotate_deg`` / ``scale_range`` / ``translate_frac``). GT masks stay
exact (nearest-neighbour) so the binary supervision is never blurred.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from data.event_representations import voxel_grid
from data.hand_event_dataset import _augment_pair
from data.nhot3d_event_dataset import NHOT3DEventDataset


class NHOT3DFrameDataset(NHOT3DEventDataset):
    """Dense voxel-frame N-HOT3D dataset (drop-in for the ``segmentation`` task).

    ``polarity_mode`` selects how event polarity enters the voxel grid:
    ``"signed"`` (default, matches pre-existing checkpoints) accumulates ON/OFF
    as +1/-1 into one channel per bin — opposite polarities at the same
    pixel/bin cancel; ``"two_channel"`` keeps ON and OFF counts in separate
    channel blocks (voxel ``(2*voxel_bins, H, W)``, ``[0:bins]`` = ON,
    ``[bins:]`` = OFF), preserving the leading/trailing-edge motion cue. The
    consuming model's ``in_channels`` must equal ``voxel_channels``.
    """

    # Shadow the inherited SPARSE ``collate_fn`` staticmethod (it unpacks the 6-tuple
    # per-event sample) with None so ``DataInterface``'s
    # ``getattr(ds, "collate_fn", None)`` falls back to PyTorch's default collate — the
    # dense ``(voxel, mask, meta)`` / clip ``(voxel_clip, mask_clip, meta)`` tuples batch
    # with the default collate exactly like the NatureRobotics dense datasets do.
    collate_fn = None

    def __init__(
        self,
        root_dir: str,
        purpose: str = "train",
        voxel_bins: int = 10,
        image_height: int = 256,
        image_width: int = 256,
        event_height: int = 260,
        event_width: int = 346,
        frames_per_sample: int = 1,
        max_events: int = 150000,
        require_mask: bool = True,
        subjects: Optional[List[str]] = None,
        max_sequences: Optional[int] = None,
        augmentation: Optional[Dict[str, Any]] = None,
        polarity_mode: str = "signed",
    ):
        # Reuse the NHOT3DEventDataset enumeration / chunk + mask loaders, but DISABLE
        # the event-native augmentor (``augmentation=None``) — dense aug is applied to
        # the voxel below — and the per-event label-refine knobs (boundary/time-ignore
        # are sparse-only concepts that don't apply to a dense voxel grid).
        super().__init__(
            root_dir=root_dir,
            purpose=purpose,
            image_height=image_height,
            image_width=image_width,
            event_height=event_height,
            event_width=event_width,
            frames_per_sample=frames_per_sample,
            max_events=max_events,
            require_mask=require_mask,
            subjects=subjects,
            max_sequences=max_sequences,
            augmentation=None,
            boundary_ignore_px=0,
            time_ignore_frac=0.0,
        )
        self.voxel_bins = int(voxel_bins)
        self.polarity_mode = str(polarity_mode).lower()
        if self.polarity_mode not in ("signed", "two_channel"):
            raise ValueError(
                f"polarity_mode must be 'signed' or 'two_channel', got {polarity_mode!r}"
            )
        if augmentation is None:
            self.augmentation: Dict[str, Any] = {"enabled": False}
        else:
            self.augmentation = dict(augmentation)
        self._dense_aug_active = (
            purpose == "train" and bool(self.augmentation.get("enabled", False))
        )

    # ------------------------------------------------------------------ per-frame

    @property
    def voxel_channels(self) -> int:
        """Channels of the emitted voxel = the model's ``in_channels``:
        ``voxel_bins``, doubled in ``two_channel`` polarity mode."""
        return self.voxel_bins * (2 if self.polarity_mode == "two_channel" else 1)

    def _build_voxel_mask(self, seq, f_idx: int):
        """One frame → ``(voxel (C,H,W), mask (H,W) float{0,1}, n_events, t_center)``.

        Mirrors ``NHOT3DEventDataset.__getitem__``'s windowing + spatial rescale, but
        bins the events into a voxel grid instead of keeping them sparse. An empty
        window yields an all-zero voxel (a valid "no events" frame).
        """
        frames = self._window_frames(seq, f_idx)
        chunks = [c for c in (self._load_events_chunk(seq.event_dir, i) for i in frames)
                  if c is not None and c.size]
        mask = self._load_union_mask(seq.mask_dir, f_idx)            # (H,W) float {0,1}
        H, W = self.image_height, self.image_width
        if not chunks:
            voxel = torch.zeros((self.voxel_channels, H, W), dtype=torch.float32)
            return voxel, mask, 0, 0.0

        ev = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        t = ev[:, 0]
        xy = ev[:, 1:3].astype(np.float32)
        p = ev[:, 3]
        if self._needs_rescale:
            xy[:, 0] *= self._scale_x
            xy[:, 1] *= self._scale_y
        t_start = float(t.min())
        t_end = float(t.max()) + 1e-6                                # +eps: keep last event
        voxel = voxel_grid(
            t=t, xy=xy, p=p, t_start=t_start, t_end=t_end,
            bins=self.voxel_bins, height=H, width=W, device="cpu", signed=True,
            split_polarity=self.polarity_mode == "two_channel",
        )
        return voxel, mask, int(ev.shape[0]), 0.5 * (t_start + t_end)

    def __getitem__(self, idx: int):
        s_idx, f_idx = self.index[idx]
        seq = self.sequences[s_idx]
        voxel, mask, n_events, t_center = self._build_voxel_mask(seq, f_idx)
        if self._dense_aug_active:
            rng = random.Random((idx * 2654435761) ^ random.getrandbits(32))
            voxel, mask, _ = _augment_pair(voxel, mask, self.augmentation, rng)
        meta = {
            "sequence": seq.token,
            "subject": seq.subject,
            "frame_index": f_idx,
            "t_center": t_center,
            "n_events": n_events,
        }
        return voxel, mask, meta


class NHOT3DClipDataset(NHOT3DFrameDataset):
    """Ordered ``clip_len``-frame N-HOT3D clips (drop-in for the ``tracking`` task).

    Re-groups the inherited per-frame index into windows of consecutive *kept* frames
    within a single sequence (clips never cross a sequence boundary, so the recurrent
    memory only ever propagates within one recording). A single clip-consistent spatial
    augmentation is applied to every frame so a random flip/affine doesn't break
    temporal coherence inside the clip.

    Extra parameters
    ----------------
    clip_len
        Consecutive kept frames per clip (the truncated-BPTT window).
    clip_stride
        Spacing between successive clip start positions (kept-frame units);
        ``clip_stride == clip_len`` → non-overlapping clips, ``1`` → maximally
        overlapping.
    """

    def __init__(
        self,
        root_dir: str,
        purpose: str = "train",
        voxel_bins: int = 5,
        image_height: int = 256,
        image_width: int = 256,
        event_height: int = 260,
        event_width: int = 346,
        max_events: int = 150000,
        require_mask: bool = True,
        subjects: Optional[List[str]] = None,
        max_sequences: Optional[int] = None,
        augmentation: Optional[Dict[str, Any]] = None,
        polarity_mode: str = "signed",
        clip_len: int = 8,
        clip_stride: int = 4,
    ):
        super().__init__(
            root_dir=root_dir,
            purpose=purpose,
            voxel_bins=voxel_bins,
            image_height=image_height,
            image_width=image_width,
            event_height=event_height,
            event_width=event_width,
            frames_per_sample=1,        # a clip stacks single-frame voxels over time
            max_events=max_events,
            require_mask=require_mask,
            subjects=subjects,
            max_sequences=max_sequences,
            augmentation=augmentation,
            polarity_mode=polarity_mode,
        )
        self.clip_len = int(clip_len)
        self.clip_stride = max(1, int(clip_stride))
        if self.clip_len < 1:
            raise ValueError(f"clip_len must be >= 1, got {self.clip_len}")

        # self.index is in (seq, frame) order; group kept frames per sequence, then cut
        # consecutive-frame clip windows. Windows never span two sequences.
        by_seq: Dict[int, List[int]] = defaultdict(list)
        for s_idx, f_idx in self.index:
            by_seq[s_idx].append(f_idx)
        self.clip_index: List[Tuple[int, List[int]]] = []
        for s_idx, frames in by_seq.items():
            frames = sorted(frames)
            if len(frames) < self.clip_len:
                continue  # sequence too short to form even one clip
            last_start = len(frames) - self.clip_len
            for start in range(0, last_start + 1, self.clip_stride):
                self.clip_index.append((s_idx, frames[start:start + self.clip_len]))
        if not self.clip_index:
            raise RuntimeError(
                f"No clips of length clip_len={self.clip_len} for purpose="
                f"{self.purpose!r}; longest sequence has "
                f"{max((len(v) for v in by_seq.values()), default=0)} kept frames. "
                f"Lower clip_len."
            )

    def __len__(self) -> int:
        return len(self.clip_index)

    def __getitem__(self, idx: int):
        s_idx, f_idxs = self.clip_index[idx]
        seq = self.sequences[s_idx]

        # One transform per clip: derive a single clip seed, then re-seed an identical
        # RNG for every frame so _augment_pair draws the SAME params each time.
        clip_seed = None
        if self._dense_aug_active:
            clip_seed = (idx * 2654435761) ^ random.getrandbits(32)

        voxels: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for f_idx in f_idxs:
            voxel, mask, _n, _tc = self._build_voxel_mask(seq, f_idx)
            if self._dense_aug_active:
                frame_rng = random.Random(clip_seed)
                voxel, mask, _ = _augment_pair(voxel, mask, self.augmentation, frame_rng)
            voxels.append(voxel)
            masks.append(mask)

        voxel_clip = torch.stack(voxels, dim=0)   # (T, C, H, W)
        mask_clip = torch.stack(masks, dim=0)     # (T, H, W)
        meta = {"sequence": seq.token, "frame_indices": list(f_idxs)}
        return voxel_clip, mask_clip, meta

    def frame_sample(self, pos: int):
        """Single (un-augmented) frame by inherited frame-index position.

        Used by the stateful validation preview to walk a held-out sequence one frame
        at a time. Returns ``(voxel (C,H,W), mask (H,W), meta)`` — mirrors the frame
        dataset's per-frame contract.
        """
        s_idx, f_idx = self.index[pos]
        seq = self.sequences[s_idx]
        voxel, mask, n_events, t_center = self._build_voxel_mask(seq, f_idx)
        meta = {"sequence": seq.token, "subject": seq.subject,
                "frame_index": f_idx, "t_center": t_center, "n_events": n_events}
        return voxel, mask, meta
