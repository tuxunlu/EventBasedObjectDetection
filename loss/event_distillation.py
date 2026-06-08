"""Per-event distillation loss for the sparse event-stream segmenter.

The student emits one logit per active event site; supervision is the teacher
(SAM 2) mask sampled at each site's pixel (``label = mask[y, x]``). This is the
sparse analogue of ``loss/distillation.py`` — that file operates on dense
``(B, 1, H, W)`` logits, this one on flat ``(N,)`` per-site logits.

    L = bce_weight   · BCE(site_logits, site_labels)          (with optional pos_weight)
      + focal_weight · Focal(site_logits, site_labels)        (optional, off by default)
      + dice_weight  · (1 - SoftDice over the event set)      (optional, off by default)

Foreground (hand/arm) events are typically a minority of the stream, so a
``pos_weight`` > 1 (or the focal term) is the main lever against class imbalance.
``forward`` returns a dict of sub-losses; the trainer logs each.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    """Accept ``(N,)`` or ``(N, 1)`` site logits, return ``(N,)``."""
    if logits.dim() == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    return logits


class EventDistillationLoss(nn.Module):
    """Composable per-event mask-distillation loss.

    Parameters
    ----------
    bce_weight
        Weight on the per-event binary cross-entropy. Default 1.0.
    pos_weight
        Optional positive-class weight for BCE. Set ``> 1`` when foreground events
        are a small fraction of the stream. ``None`` disables.
    focal_weight, focal_gamma, focal_alpha
        Optional focal term (Lin et al. 2017) for stronger imbalance control.
        ``focal_weight`` defaults to 0 (off).
    dice_weight
        Optional soft-Dice over the batch's event set (treats all sites as one
        "image"). Defaults to 0 (off). A cheap shape regularizer; per-sample Dice
        can be added later via ``batch_idx``.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        pos_weight: Optional[float] = None,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        dice_weight: float = 0.0,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.focal_weight = float(focal_weight)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.dice_weight = float(dice_weight)
        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
        else:
            self._pos_weight = None

    def _bce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, labels,
            pos_weight=self._pos_weight if self._pos_weight is not None else None,
            reduction="mean",
        )

    def _focal(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Numerically-stable focal loss on logits.
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p_t = p * labels + (1.0 - p) * (1.0 - labels)
        alpha_t = self.focal_alpha * labels + (1.0 - self.focal_alpha) * (1.0 - labels)
        loss = alpha_t * (1.0 - p_t).pow(self.focal_gamma) * ce
        return loss.mean()

    def _dice(self, logits: torch.Tensor, labels: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        inter = (probs * labels).sum()
        denom = probs.sum() + labels.sum()
        return 1.0 - (2.0 * inter + eps) / (denom + eps)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,  # reserved for per-sample Dice
    ) -> Dict[str, torch.Tensor]:
        logits = _flatten_logits(logits)
        labels = labels.float()

        terms: Dict[str, torch.Tensor] = {}
        if logits.numel() == 0:
            # Empty window: contribute a differentiable zero so the step is a no-op.
            zero = logits.sum()  # 0-d, carries grad graph if any
            terms["bce"] = zero
            terms["total"] = zero
            return terms

        terms["bce"] = self._bce(logits, labels)
        total = self.bce_weight * terms["bce"]

        if self.focal_weight > 0.0:
            terms["focal"] = self._focal(logits, labels)
            total = total + self.focal_weight * terms["focal"]

        if self.dice_weight > 0.0:
            terms["dice"] = self._dice(logits, labels)
            total = total + self.dice_weight * terms["dice"]

        terms["total"] = total
        return terms
