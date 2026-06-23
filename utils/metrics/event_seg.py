"""Per-event segmentation metrics + rasterize-to-dense IoU.

Two families:

1. **Per-event** precision / recall / F1 / accuracy on the foreground class —
   the native metric for the sparse event-stream output.
2. **Rasterize-to-dense IoU** — scatter the per-event predictions back onto an
   ``(H, W)`` grid and reuse the existing dense ``binary_iou`` against the teacher
   mask, so the sparse model is directly comparable to the dense baseline's
   ``val_iou_epoch``.

All functions take flat per-site logits ``(N,)`` / ``(N, 1)`` and labels ``(N,)``
in ``{0, 1}`` and return 0-d tensors that slot into ``self.log(...)``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from utils.metrics.segmentation import binary_iou


def _binarize(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    if logits.dim() == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    return (torch.sigmoid(logits) > threshold).float()


def _counts(logits: torch.Tensor, labels: torch.Tensor, threshold: float):
    pred = _binarize(logits, threshold)
    labels = labels.float()
    # Drop trimap "ignore" events (label < 0). Without this, a -1 label makes
    # (pred*labels) negative and (pred*(1-labels)) inflated, so tp/fp/fn become
    # signed garbage and F1/precision/recall leave [0, 1]. Matches sweep_counts.
    keep = labels >= 0
    if keep.numel() and not bool(keep.all()):
        pred, labels = pred[keep], labels[keep]
    tp = (pred * labels).sum()
    fp = (pred * (1.0 - labels)).sum()
    fn = ((1.0 - pred) * labels).sum()
    tn = ((1.0 - pred) * (1.0 - labels)).sum()
    return tp, fp, fn, tn


def event_accuracy(logits: torch.Tensor, labels: torch.Tensor,
                   threshold: float = 0.5) -> torch.Tensor:
    tp, fp, fn, tn = _counts(logits, labels, threshold)
    total = tp + fp + fn + tn
    return (tp + tn) / total.clamp(min=1.0)


def event_precision(logits: torch.Tensor, labels: torch.Tensor,
                    threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    tp, fp, _fn, _tn = _counts(logits, labels, threshold)
    return tp / (tp + fp + eps)


def event_recall(logits: torch.Tensor, labels: torch.Tensor,
                 threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    tp, _fp, fn, _tn = _counts(logits, labels, threshold)
    return tp / (tp + fn + eps)


def event_f1(logits: torch.Tensor, labels: torch.Tensor,
             threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Foreground-class F1 over events. Returns a 0-d tensor."""
    tp, fp, fn, _tn = _counts(logits, labels, threshold)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


def sweep_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-threshold ``(tp, fp, fn)`` over events, vectorized across ``thresholds``.

    Returns three ``(len(thresholds),)`` tensors. Built for cheap accumulation over a
    whole validation epoch so the decision threshold can be calibrated post-hoc
    (the fixed-0.5 cut over-predicts when foreground events are inflated by the
    motion-smeared teacher labels). Events with ``label < 0`` (trimap "ignore") are
    dropped here too, so calibration matches the training supervision.
    """
    if logits.dim() == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    labels = labels.float()
    keep = labels >= 0
    if keep.numel() and not bool(keep.all()):
        logits, labels = logits[keep], labels[keep]
    probs = torch.sigmoid(logits)
    thr = thresholds.to(probs.device, probs.dtype).view(-1, 1)     # (Tt, 1)
    pred = (probs.view(1, -1) > thr).float()                       # (Tt, N)
    lab = labels.view(1, -1)
    tp = (pred * lab).sum(dim=1)
    fp = (pred * (1.0 - lab)).sum(dim=1)
    fn = ((1.0 - pred) * lab).sum(dim=1)
    return tp, fp, fn


def f1_from_counts(
    tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


def clean_dense_mask(
    mask: torch.Tensor,
    open_ksize: int = 3,
    keep_largest: bool = False,
    min_area: int = 0,
) -> torch.Tensor:
    """Post-hoc spatial-coherence clean of a binary ``(B, H, W)`` prediction.

    Drops isolated foreground speckle — the keyboard/sensor-noise false positives
    that survive a per-event classifier with no spatial-coherence prior:

      * ``open_ksize > 1``: morphological opening (erode then dilate) removes thin
        specks and 1-px noise.
      * ``min_area > 0``: connected components smaller than ``min_area`` px are removed.
      * ``keep_largest``: keep only the single largest component (the hand+arm blob).

    Returns a float ``(B, H, W)`` mask in ``{0, 1}`` on the input device. CPU/numpy
    (cv2) under the hood; meant for eval/inference, not the training loop.
    """
    import cv2
    import numpy as np

    dev = mask.device
    arr = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[None]
    out = np.zeros_like(arr)
    if open_ksize and open_ksize > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    else:
        kernel = None
    for b in range(arr.shape[0]):
        m = arr[b]
        if kernel is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        if keep_largest or min_area > 0:
            num, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if num > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]               # skip background (0)
                keep_ids = set()
                if keep_largest:
                    keep_ids.add(int(areas.argmax()) + 1)
                if min_area > 0:
                    keep_ids.update(int(i) + 1 for i, a in enumerate(areas) if a >= min_area)
                if not keep_largest and min_area <= 0:
                    keep_ids = set(range(1, num))
                m = np.isin(lbl, list(keep_ids)).astype(np.uint8)
            elif num <= 1:
                m = np.zeros_like(m)
        out[b] = m
    return torch.from_numpy(out.astype(np.float32)).to(dev)


def rasterize_events_to_logits(
    coords: torch.Tensor,
    logits: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    background_logit: float = -1e4,
) -> torch.Tensor:
    """Scatter per-event logits onto a dense ``(B, H, W)`` logit grid (max-pool per pixel).

    Pixels with no events get ``background_logit`` (≈ ``sigmoid -> 0``), so the
    inherent sparsity gap reads as background. ``max`` is the natural reduction
    for multiple events at a pixel: any confident-foreground event wins.
    """
    if logits.dim() == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    x = coords[:, 0].long()
    y = coords[:, 1].long()
    flat = torch.full(
        (batch_size * height * width,), float(background_logit),
        device=logits.device, dtype=logits.dtype,
    )
    if logits.numel() > 0:
        lin = (batch_idx.long() * height + y) * width + x
        flat = flat.scatter_reduce(0, lin, logits, reduce="amax", include_self=True)
    return flat.view(batch_size, height, width)


def event_pred_to_dense_iou(
    coords: torch.Tensor,
    logits: torch.Tensor,
    batch_idx: torch.Tensor,
    dense_mask: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Rasterize predictions and compute dense IoU vs the teacher mask.

    ``dense_mask`` is ``(B, H, W)`` in ``{0, 1}``. Reuses the dense ``binary_iou``
    (which applies ``sigmoid`` then thresholds at 0.5 ⇔ logit > 0), so the
    ``background_logit`` floor keeps no-event pixels as background. Returns a 0-d
    tensor directly comparable to the dense baseline's IoU.
    """
    raster_logits = rasterize_events_to_logits(
        coords, logits, batch_idx, batch_size, height, width,
    )
    # binary_iou thresholds sigmoid(logit) > threshold; align the rasterized
    # logit floor to that convention by passing the logit grid straight through.
    return binary_iou(raster_logits.unsqueeze(1), dense_mask.float(), threshold=threshold)


def event_pred_to_dense_iou_clean(
    coords: torch.Tensor,
    logits: torch.Tensor,
    batch_idx: torch.Tensor,
    dense_mask: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    threshold: float = 0.5,
    open_ksize: int = 3,
    keep_largest: bool = False,
    min_area: int = 0,
) -> torch.Tensor:
    """Like :func:`event_pred_to_dense_iou` but with a spatial-coherence clean of
    the rasterized prediction before scoring (see :func:`clean_dense_mask`). Reports
    the gain from dropping isolated false-positive speckle."""
    raster_logits = rasterize_events_to_logits(
        coords, logits, batch_idx, batch_size, height, width,
    )
    pred = (torch.sigmoid(raster_logits) > threshold).float()
    pred = clean_dense_mask(pred, open_ksize=open_ksize,
                            keep_largest=keep_largest, min_area=min_area)
    inter = (pred * dense_mask.float()).sum()
    union = pred.sum() + dense_mask.float().sum() - inter
    return inter / union.clamp(min=1.0)
