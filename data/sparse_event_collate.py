"""Ragged batching for the per-event (sparse, 3D) segmentation path.

The default PyTorch collate cannot stack samples whose first dimension (the number
of events) varies. :func:`collate_sparse_events` concatenates the per-sample event
tensors into one flat batch and records a per-event batch index, following spconv's
batched-indices convention (see ``model/sparse_backend.py``).

Unlike the old per-pixel-merged path, samples here are **per event**: every event
keeps its own ``(x, y)``, normalized time, polarity feature and label, so the
``EventSparseSeg`` model can voxelize in 3D ``(t, y, x)`` itself and emit one logit
per event. ``times`` is the normalized event time in ``[0, 1]`` used for temporal
binning.

The returned :class:`SparseEventBatch` is a dataclass of plain tensors / scalars so
PyTorch-Lightning's automatic device transfer moves the tensors and leaves scalars
untouched; ``to()`` and ``pin_memory()`` are also provided for explicit control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class SparseEventBatch:
    """A batch of variable-length per-event samples.

    All event tensors are concatenated along the event axis; ``batch_idx`` maps each
    event back to its sample. ``dense_mask`` is the per-sample teacher mask (FLIR
    frame) kept for the warp-splat supervision and rasterized-IoU evaluation.
    """

    coords: torch.Tensor       # (N_total, 2) long  -- (x, y) per event
    feats: torch.Tensor        # (N_total, C) float -- per-event features
    times: torch.Tensor        # (N_total,)  float  -- normalized event time in [0,1]
    labels: torch.Tensor       # (N_total,)  float  -- per-event fg/bg target (mask[y,x])
    batch_idx: torch.Tensor    # (N_total,)  long   -- event -> sample index
    dense_mask: torch.Tensor   # (B, H, W)   uint8  -- teacher masks (FLIR frame)
    batch_size: int
    height: int
    width: int
    meta: Optional[List[Dict[str, Any]]] = None

    def to(self, device, non_blocking: bool = False) -> "SparseEventBatch":
        return SparseEventBatch(
            coords=self.coords.to(device, non_blocking=non_blocking),
            feats=self.feats.to(device, non_blocking=non_blocking),
            times=self.times.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
            batch_idx=self.batch_idx.to(device, non_blocking=non_blocking),
            dense_mask=self.dense_mask.to(device, non_blocking=non_blocking),
            batch_size=self.batch_size,
            height=self.height,
            width=self.width,
            meta=self.meta,
        )

    def pin_memory(self) -> "SparseEventBatch":
        return SparseEventBatch(
            coords=self.coords.pin_memory(),
            feats=self.feats.pin_memory(),
            times=self.times.pin_memory(),
            labels=self.labels.pin_memory(),
            batch_idx=self.batch_idx.pin_memory(),
            dense_mask=self.dense_mask.pin_memory(),
            batch_size=self.batch_size,
            height=self.height,
            width=self.width,
            meta=self.meta,
        )


def collate_sparse_events(
    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]],
) -> SparseEventBatch:
    """Collate ``(coords, feats, times, labels, dense_mask, meta)`` samples into a batch.

    Each sample's ``coords`` is ``(n_i, 2)`` long, ``feats`` ``(n_i, C)`` float,
    ``times`` ``(n_i,)`` float, ``labels`` ``(n_i,)`` float, ``dense_mask`` ``(H, W)``
    uint8, ``meta`` a dict. Samples with ``n_i == 0`` (empty windows) are tolerated.
    """
    if not samples:
        raise ValueError("collate_sparse_events received an empty sample list")

    coords_list: List[torch.Tensor] = []
    feats_list: List[torch.Tensor] = []
    times_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    batch_idx_list: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    meta: List[Dict[str, Any]] = []

    feat_dim = samples[0][1].shape[1] if samples[0][1].ndim == 2 else 1
    for b, (coords, feats, times, labels, dense_mask, m) in enumerate(samples):
        coords_list.append(coords)
        feats_list.append(feats)
        times_list.append(times)
        labels_list.append(labels)
        batch_idx_list.append(torch.full((coords.shape[0],), b, dtype=torch.long))
        masks.append(dense_mask)
        meta.append(m)

    dense_mask = torch.stack(masks, dim=0)
    H, W = int(dense_mask.shape[-2]), int(dense_mask.shape[-1])

    coords = (torch.cat(coords_list, dim=0)
              if coords_list else torch.zeros((0, 2), dtype=torch.long))
    feats = (torch.cat(feats_list, dim=0)
             if feats_list else torch.zeros((0, feat_dim), dtype=torch.float32))
    times = (torch.cat(times_list, dim=0)
             if times_list else torch.zeros((0,), dtype=torch.float32))
    labels = (torch.cat(labels_list, dim=0)
              if labels_list else torch.zeros((0,), dtype=torch.float32))
    batch_idx = (torch.cat(batch_idx_list, dim=0)
                 if batch_idx_list else torch.zeros((0,), dtype=torch.long))

    return SparseEventBatch(
        coords=coords, feats=feats, times=times, labels=labels, batch_idx=batch_idx,
        dense_mask=dense_mask, batch_size=len(samples), height=H, width=W, meta=meta,
    )
