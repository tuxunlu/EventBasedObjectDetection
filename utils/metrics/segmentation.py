"""Binary segmentation metrics for hand/arm mask distillation.

All functions are differentiable-shape-friendly: they accept raw logits
``(B, 1, H, W)`` or ``(B, H, W)`` and binary targets ``(B, H, W)`` in
``{0, 1}``. Outputs are scalar tensors so they slot directly into Lightning's
``self.log(...)`` calls.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _to_pred_mask(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    """Sigmoid + threshold; collapse channel dim if present."""
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    return (torch.sigmoid(logits) > threshold).float()


def _foreground_mask(target: torch.Tensor) -> torch.Tensor:
    """Per-sample boolean: True where the GT mask has any foreground pixel."""
    return target.sum(dim=(-2, -1)) > 0


def binary_iou(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    ignore_empty: bool = True,
) -> torch.Tensor:
    """Mean per-sample binary IoU. Returns a 0-d tensor.

    With ``ignore_empty=True`` (default), frames whose GT mask is all-zero
    (hand/arm absent) are dropped from the mean. Without this, an empty-GT +
    empty-pred frame scores a perfect ``1.0`` (``(0+eps)/(0+eps)``) and inflates
    the metric — a problem here because ``val_iou_epoch`` drives checkpoint
    selection. The exclusion is per-sample, so a mixed batch still scores its
    hand-present frames. If a batch happens to contain *only* empty-GT frames
    we fall back to the all-frame mean (a finite, correctness-conditioned value)
    rather than returning NaN, which keeps DDP epoch-aggregation well-defined.
    """
    pred = _to_pred_mask(logits, threshold)
    target = target.float()
    inter = (pred * target).sum(dim=(-2, -1))
    union = (pred + target - pred * target).sum(dim=(-2, -1))
    iou = (inter + eps) / (union + eps)
    if ignore_empty:
        valid = _foreground_mask(target)
        if valid.any():
            return iou[valid].mean()
    return iou.mean()


def binary_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    ignore_empty: bool = True,
) -> torch.Tensor:
    """Mean per-sample Dice coefficient. Useful as a complementary metric to IoU.

    ``ignore_empty`` behaves exactly as in :func:`binary_iou` — empty-GT frames
    are excluded from the mean to avoid the ``1.0`` self-inflation, with an
    all-frame fallback for a fully-empty batch.
    """
    pred = _to_pred_mask(logits, threshold)
    target = target.float()
    inter = (pred * target).sum(dim=(-2, -1))
    denom = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = (2 * inter + eps) / (denom + eps)
    if ignore_empty:
        valid = _foreground_mask(target)
        if valid.any():
            return dice[valid].mean()
    return dice.mean()


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    """1-pixel boundary via morphological gradient (dilation − erosion).

    ``mask`` is ``(B, H, W)`` in {0, 1}; output has the same shape.
    """
    m4 = mask.unsqueeze(1)
    eroded = -F.max_pool2d(-m4, kernel_size=3, stride=1, padding=1)
    dilated = F.max_pool2d(m4, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).squeeze(1).clamp_(0.0, 1.0)


def _dilate(mask: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 0:
        return mask
    m4 = mask.unsqueeze(1)
    return F.max_pool2d(m4, kernel_size=2 * d + 1, stride=1, padding=d).squeeze(1)


def boundary_f_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    d_tolerance: int = 2,
    threshold: float = 0.5,
    eps: float = 1e-6,
    ignore_empty: bool = True,
) -> torch.Tensor:
    """Boundary F-score with a ``d_tolerance``-pixel match radius.

    Following the standard contour-F formulation: a predicted boundary pixel
    counts as a hit if any GT boundary pixel lies within ``d_tolerance`` pixels
    (and symmetrically for recall). Returns a 0-d tensor — the mean F across
    the batch.

    ``ignore_empty=True`` (default) drops empty-GT frames from the mean, matching
    :func:`binary_iou` / :func:`binary_dice`, with an all-frame fallback for a
    fully-empty batch.
    """
    pred = _to_pred_mask(logits, threshold)
    target = target.float()
    p_bnd = _boundary(pred)
    g_bnd = _boundary(target)
    p_bnd_dil = _dilate(p_bnd, d_tolerance)
    g_bnd_dil = _dilate(g_bnd, d_tolerance)

    tp_precision = (p_bnd * g_bnd_dil).sum(dim=(-2, -1))
    tp_recall = (g_bnd * p_bnd_dil).sum(dim=(-2, -1))
    n_pred = p_bnd.sum(dim=(-2, -1))
    n_gt = g_bnd.sum(dim=(-2, -1))

    precision = tp_precision / (n_pred + eps)
    recall = tp_recall / (n_gt + eps)
    f = 2 * precision * recall / (precision + recall + eps)
    # A frame with neither a predicted nor a GT boundary is a perfect match
    # (the model correctly produced nothing): force F = 1.0. The raw formula
    # yields 0.0 there (both precision and recall are 0/eps), which is wrong —
    # this is the bug the comment used to claim was already handled.
    both_empty = (n_pred == 0) & (n_gt == 0)
    f = torch.where(both_empty, torch.ones_like(f), f)

    if ignore_empty:
        valid = _foreground_mask(target)
        if valid.any():
            return f[valid].mean()
    return f.mean()


def flicker_rate(masks_T_H_W: torch.Tensor) -> torch.Tensor:
    """Per-pixel fraction of state changes across a temporally ordered mask stack.

    Input: ``(T, H, W)`` binary masks for consecutive frames of ONE clip.
    Output: 0-d scalar — fraction of (pixel, transition) pairs that flipped.

    Not used inside the random-batch training loop; invoked from a separate
    sequence-aware evaluation pass over a held-out clip.
    """
    if masks_T_H_W.shape[0] < 2:
        return torch.zeros((), device=masks_T_H_W.device)
    diff = (masks_T_H_W[1:] != masks_T_H_W[:-1]).float()
    return diff.mean()


def static_region_flicker(
    masks_T_H_W: torch.Tensor,
    event_density_T_H_W: torch.Tensor,
    density_threshold: float = 0.0,
) -> torch.Tensor:
    """Flicker measured only in pixels with no events in the transition window.

    Captures the "mask should be persistent over stationary objects" property
    central to RQ2. Input shapes both ``(T, H, W)``.
    """
    if masks_T_H_W.shape[0] < 2:
        return torch.zeros((), device=masks_T_H_W.device)
    diff = (masks_T_H_W[1:] != masks_T_H_W[:-1]).float()
    static = (event_density_T_H_W[1:] <= density_threshold).float()
    denom = static.sum().clamp(min=1.0)
    return (diff * static).sum() / denom
