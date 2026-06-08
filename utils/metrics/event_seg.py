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

from typing import Dict

import torch

from utils.metrics.segmentation import binary_iou


def _binarize(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    if logits.dim() == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    return (torch.sigmoid(logits) > threshold).float()


def _counts(logits: torch.Tensor, labels: torch.Tensor, threshold: float):
    pred = _binarize(logits, threshold)
    labels = labels.float()
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
