"""Per-event (sparse) variant of :class:`HandEventDataset`.

Instead of a dense voxel grid, each sample is the set of **active event sites** in
the window centred on a FLIR frame, with a foreground/background label per site
sampled from the cached SAM 2 teacher mask (``label = mask[y, x]``). This is the
input/supervision for the event-native ``EventSparseSeg`` model.

Reuses the parent's subject-disjoint LOSO split, frame index, timestamp-unit
detection, handle caching, and teacher-mask lookup unchanged; only ``__getitem__``
differs. Geometric augmentation is intentionally disabled on this path (the
parent's augmenter warps dense voxels/masks, not raw coordinates — coord-space
augmentation is a Phase-2 addition).

Per-sample output (consumed by ``collate_sparse_events``):
    ``(coords[M, 2] long (x, y), feats[M, C] float, labels[M] float,
       dense_mask[H, W] uint8, meta: dict)``

Events sharing a pixel within the window are merged into one site (mean polarity,
mean normalized time, log event count). Because the teacher mask is per-pixel,
every event at a pixel shares the site's label; the true per-event stream output
is recovered at inference by broadcasting the site logit to its events.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch

from data.hand_event_dataset import HandEventDataset
from data.sparse_event_collate import collate_sparse_events


class HandEventStreamDataset(HandEventDataset):

    #: signed polarity, normalized time-in-window, log1p(event count)
    NUM_FEATURES = 3

    def __init__(
        self,
        root_dir: str,
        purpose: str = "train",
        window_ms: float = 36.0,
        image_height: int = 480,
        image_width: int = 640,
        held_out_subject=None,
        require_teacher: bool = True,
        action_only: bool = False,
        mask_root=None,
        max_events: int = 150000,
    ):
        # voxel_bins is irrelevant on the event path; pass a harmless 1. Augmentation
        # and RGB are disabled (see module docstring).
        super().__init__(
            root_dir=root_dir,
            purpose=purpose,
            voxel_bins=1,
            window_ms=window_ms,
            image_height=image_height,
            image_width=image_width,
            held_out_subject=held_out_subject,
            require_teacher=require_teacher,
            action_only=action_only,
            mask_root=mask_root,
            augmentation=None,
            provide_rgb=False,
        )
        self.max_events = int(max_events)

    # ------------------------------------------------------------------ helpers

    def _load_mask(self, handle: dict, f_idx: int) -> torch.Tensor:
        """Load the teacher mask as an ``(H, W)`` float tensor in ``{0, 1}``.

        Mirrors the parent's inline mask loading (kept here so this class needs no
        edit to ``HandEventDataset``).
        """
        H, W = self.image_height, self.image_width
        mask_path = handle["mask_dir"] / f"{f_idx:06d}.png"
        if self.require_teacher and not mask_path.exists():
            raise FileNotFoundError(f"Missing teacher mask {mask_path}")
        if mask_path.exists():
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise IOError(f"Could not read mask {mask_path}")
            if mask_img.shape != (H, W):
                mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
            return torch.from_numpy((mask_img > 127).astype(np.float32))
        return torch.zeros((H, W), dtype=torch.float32)

    def _empty_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((0, 2), dtype=torch.long),
            torch.zeros((0, self.NUM_FEATURES), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
        )

    def _build_sites(
        self,
        t: np.ndarray,
        xy: np.ndarray,
        p: np.ndarray,
        t_start: float,
        t_end: float,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized: raw events -> unique pixel sites with aggregated features + labels."""
        H, W = self.image_height, self.image_width
        if len(t) == 0:
            return self._empty_sample()

        xy_t = torch.as_tensor(np.asarray(xy), dtype=torch.long)
        x = xy_t[:, 0]
        y = xy_t[:, 1]
        t_t = torch.as_tensor(np.asarray(t), dtype=torch.float64)
        p_t = torch.as_tensor(np.asarray(p), dtype=torch.float32)
        pol = torch.where(p_t > 0, torch.ones_like(p_t), -torch.ones_like(p_t))

        in_b = ((x >= 0) & (x < W) & (y >= 0) & (y < H)
                & (t_t >= t_start) & (t_t < t_end))
        x, y, pol, t_t = x[in_b], y[in_b], pol[in_b], t_t[in_b]
        if x.numel() == 0:
            return self._empty_sample()

        # Bound compute/memory on bursty windows with a deterministic strided subsample.
        m = x.numel()
        if self.max_events > 0 and m > self.max_events:
            stride = (m + self.max_events - 1) // self.max_events
            sel = torch.arange(0, m, stride)
            x, y, pol, t_t = x[sel], y[sel], pol[sel], t_t[sel]

        denom = (t_end - t_start) if (t_end - t_start) > 0 else 1.0
        t_norm = ((t_t - t_start) / denom).clamp(0.0, 1.0).to(torch.float32)

        # Merge duplicate pixels into unique sites (spconv wants unique coords).
        lin = y * W + x
        uniq, inverse = torch.unique(lin, sorted=True, return_inverse=True)
        n_sites = uniq.numel()
        ones = torch.ones_like(pol)
        cnt = torch.zeros(n_sites, dtype=torch.float32).scatter_add_(0, inverse, ones)
        pol_sum = torch.zeros(n_sites, dtype=torch.float32).scatter_add_(0, inverse, pol)
        t_sum = torch.zeros(n_sites, dtype=torch.float32).scatter_add_(0, inverse, t_norm)
        cnt_safe = cnt.clamp(min=1.0)

        ux = (uniq % W).to(torch.long)
        uy = (uniq // W).to(torch.long)
        coords = torch.stack([ux, uy], dim=1)                       # (n_sites, 2) (x, y)
        feats = torch.stack([pol_sum / cnt_safe,                    # mean polarity
                             t_sum / cnt_safe,                      # mean time-in-window
                             torch.log1p(cnt)], dim=1)              # log event count
        labels = mask[uy, ux].to(torch.float32)                    # (n_sites,)
        return coords, feats, labels

    # ------------------------------------------------------------------- sample

    def __getitem__(self, idx: int):
        s_idx, f_idx = self.index[idx]
        seq_dir = self.sequences[s_idx]
        h = self._get_handle(seq_dir)

        t_center = float(h["flir_t"][f_idx])
        half = (self.window_ms / 1000.0) * h["unit"] / 2.0
        t_start = t_center - half
        t_end = t_center + half

        events_t = h["events_t"]
        lo = int(np.searchsorted(events_t, t_start, side="left"))
        hi = int(np.searchsorted(events_t, t_end, side="right"))
        t = np.asarray(events_t[lo:hi])
        xy = np.asarray(h["events_xy"][lo:hi])
        p = np.asarray(h["events_p"][lo:hi])

        mask = self._load_mask(h, f_idx)
        coords, feats, labels = self._build_sites(t, xy, p, t_start, t_end, mask)
        dense_mask = (mask > 0.5).to(torch.uint8)

        meta = {
            "sequence": seq_dir.name,
            "frame_index": f_idx,
            "t_center": t_center,
            "n_events": int(hi - lo),
            "n_sites": int(coords.shape[0]),
        }
        return coords, feats, labels, dense_mask, meta

    # ------------------------------------------------------------------ collate

    @staticmethod
    def collate_fn(samples):
        """Ragged collate selected by ``DataInterface`` via ``getattr(ds, 'collate_fn')``."""
        return collate_sparse_events(samples)
