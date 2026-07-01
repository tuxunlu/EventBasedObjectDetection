"""Temporal-clip variant of :class:`HandEventDataset` for the tracking task.

Where ``HandEventDataset`` yields independent ``(voxel, mask, meta)`` frames,
this yields **ordered clips** of ``clip_len`` consecutive (kept) frames from a
single sequence:

    voxel : (T, C, H, W)      C == voxel_bins (2*voxel_bins when polarity_mode="two_channel")
    mask  : (T, H, W)
    meta  : {"sequence", "frame_indices", ...}

(default collate adds the batch dim → ``(N, T, C, H, W)`` / ``(N, T, H, W)``,
exactly what ``ModelInterface._tracking_step`` and ``EventTrackUnet`` expect.)

All the heavy lifting — sequence discovery, the LOSO subject split, the
``action_only`` / teacher-mask frame filtering, voxelization and mask loading —
is inherited from ``HandEventDataset`` unchanged; this subclass only:

1. re-groups the inherited per-frame ``self.index`` into clip windows
   (``self.clip_index``), and
2. overrides ``__len__`` / ``__getitem__`` to return a stacked clip, applying a
   single, *clip-consistent* spatial augmentation to every frame (so a random
   flip/affine doesn't break temporal coherence within a clip).

``frame_sample(pos)`` is also exposed so the validation-preview code can pull a
single frame (by the inherited frame-level ``self.index`` position) and run the
tracker statefully across a whole held-out sequence.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from data.hand_event_dataset import HandEventDataset, _augment_pair


class HandEventClipDataset(HandEventDataset):
    """``HandEventDataset`` that serves ordered ``clip_len``-frame clips.

    Extra parameters
    ----------------
    clip_len
        Number of consecutive kept frames per clip (the temporal window over
        which the recurrent memory is unrolled / BPTT'd). Keep modest (e.g.
        4-8) to bound memory and backprop depth.
    clip_stride
        Spacing between successive clip start positions (in kept-frame units).
        ``clip_stride == clip_len`` gives non-overlapping clips; ``1`` gives
        maximally overlapping clips (more, more-correlated training windows).
    """

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
        polarity_mode: str = "signed",
        clip_len: int = 8,
        clip_stride: int = 1,
    ):
        super().__init__(
            root_dir=root_dir,
            purpose=purpose,
            voxel_bins=voxel_bins,
            window_ms=window_ms,
            image_height=image_height,
            image_width=image_width,
            held_out_subject=held_out_subject,
            require_teacher=require_teacher,
            action_only=action_only,
            mask_root=mask_root,
            augmentation=augmentation,
            provide_rgb=provide_rgb,
            polarity_mode=polarity_mode,
        )
        self.clip_len = int(clip_len)
        self.clip_stride = max(1, int(clip_stride))
        if self.clip_len < 1:
            raise ValueError(f"clip_len must be >= 1, got {self.clip_len}")

        # Group the inherited frame index by sequence, preserving frame order
        # (self.index was built in (seq, frame) order). Each clip is a window of
        # consecutive *kept* frames within one sequence — windows never cross a
        # sequence boundary, so the memory only ever propagates within a clip of
        # the same recording.
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
        seq_dir = self.sequences[s_idx]
        h = self._get_handle(seq_dir)

        # One augmentation transform per clip: derive a single clip seed, then
        # re-seed an identical RNG for every frame so _augment_pair draws the
        # SAME flip/affine params each time — temporally consistent within the
        # clip, still random across clips and epochs.
        clip_seed = None
        if self._aug_active:
            clip_seed = (idx * 2654435761) ^ random.getrandbits(32)

        voxels: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        rgbs: List[torch.Tensor] = []
        for f_idx in f_idxs:
            voxel, mask, rgb, _n_events, _t_center = self._load_frame(seq_dir, h, f_idx)
            if self._aug_active:
                frame_rng = random.Random(clip_seed)
                voxel, mask, rgb = _augment_pair(
                    voxel, mask, self.augmentation, frame_rng, rgb=rgb
                )
            voxels.append(voxel)
            masks.append(mask)
            if rgb is not None:
                rgbs.append(rgb)

        voxel_clip = torch.stack(voxels, dim=0)   # (T, C, H, W)
        mask_clip = torch.stack(masks, dim=0)     # (T, H, W)
        meta = {"sequence": seq_dir.name, "frame_indices": list(f_idxs)}

        if self.provide_rgb:
            return voxel_clip, mask_clip, torch.stack(rgbs, dim=0), meta
        return voxel_clip, mask_clip, meta

    def frame_sample(self, pos: int):
        """Single (un-augmented) frame by inherited frame-index position.

        Used by the stateful validation preview to walk a held-out sequence one
        frame at a time. Returns ``(voxel(C,H,W), mask(H,W), meta)`` — or with an
        ``rgb(3,H,W)`` in the 4-tuple form when ``provide_rgb`` — mirroring the
        base dataset's per-frame contract.
        """
        s_idx, f_idx = self.index[pos]
        seq_dir = self.sequences[s_idx]
        h = self._get_handle(seq_dir)
        voxel, mask, rgb, n_events, t_center = self._load_frame(seq_dir, h, f_idx)
        meta = {"sequence": seq_dir.name, "frame_index": f_idx,
                "t_center": t_center, "n_events": n_events}
        if self.provide_rgb:
            return voxel, mask, rgb, meta
        return voxel, mask, meta
