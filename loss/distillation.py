"""Mask-distillation loss for the event-only student.

Phase B uses only ``L_mask = BCE + λ_dice · Dice``. The class is designed to be
extended in Phase C with:

    L = L_mask
        + λ_motion    · L_motion       (event-density-weighted BCE)
        + λ_temporal  · L_temporal     (mask flicker penalty across windows)
        + λ_feat      · L_feat         (cosine alignment with a frozen RGB
                                        teacher feature map)

For Phase B the extra terms are accepted in ``__init__`` with default weight 0
so call sites already pass them and won't break when they are filled in. The
forward method returns a dict of sub-losses; the trainer logs each.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _binary_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss on sigmoid probabilities. Inputs match ``DistillationLoss.forward``."""
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    probs = torch.sigmoid(logits)
    target = target.float()
    inter = (probs * target).sum(dim=(-2, -1))
    denom = probs.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = (2 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


class DistillationLoss(nn.Module):
    """Composable mask-distillation loss.

    Parameters
    ----------
    bce_weight, dice_weight
        Weights on the per-pixel BCE and soft-Dice components of ``L_mask``.
        Both default to 1.0; cumulatively this is ``L_mask = BCE + Dice``.
    pos_weight
        Optional ``BCEWithLogitsLoss`` positive-class weight. Use when the
        hand/arm mask covers a small fraction of the frame and BCE alone is
        biased toward predicting all-zero. ``None`` disables.
    motion_weight, temporal_weight, feature_weight
        Phase-C placeholders. When > 0 and the corresponding auxiliary tensor
        is supplied in ``forward``, the term is added; otherwise it is silently
        zero so call sites can stay stable across phases.

    Returns (from ``forward``)
    --------------------------
    A dict with keys:
        ``"total"`` — the scalar loss to backprop.
        ``"bce"``, ``"dice"`` — sub-loss tensors for logging.
        ``"motion"``, ``"temporal"``, ``"feat"`` — present only when active.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: Optional[float] = None,
        motion_weight: float = 0.0,
        temporal_weight: float = 0.0,
        feature_weight: float = 0.0,
        motion_alpha: float = 4.0,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.motion_weight = float(motion_weight)
        self.temporal_weight = float(temporal_weight)
        self.feature_weight = float(feature_weight)
        self.motion_alpha = float(motion_alpha)

        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
        else:
            self._pos_weight = None

    def _bce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        return F.binary_cross_entropy_with_logits(
            logits, target.float(),
            pos_weight=self._pos_weight if self._pos_weight is not None else None,
            reduction="mean",
        )

    def _motion_bce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        event_density: torch.Tensor,
    ) -> torch.Tensor:
        """BCE reweighted per-pixel by ``(1 + α · event_density)``.

        ``event_density`` must be ``(B, H, W)``, non-negative, ideally
        normalized so its mean over the batch is O(1).
        """
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        per_pix = F.binary_cross_entropy_with_logits(
            logits, target.float(),
            pos_weight=self._pos_weight if self._pos_weight is not None else None,
            reduction="none",
        )
        weight = 1.0 + self.motion_alpha * event_density
        return (per_pix * weight).mean()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        event_density: Optional[torch.Tensor] = None,
        prev_logits: Optional[torch.Tensor] = None,
        static_mask: Optional[torch.Tensor] = None,
        student_feat: Optional[torch.Tensor] = None,
        teacher_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        terms: Dict[str, torch.Tensor] = {}

        terms["bce"] = self._bce(logits, target)
        terms["dice"] = _binary_dice_loss(logits, target)
        total = self.bce_weight * terms["bce"] + self.dice_weight * terms["dice"]

        if self.motion_weight > 0.0 and event_density is not None:
            terms["motion"] = self._motion_bce(logits, target, event_density)
            total = total + self.motion_weight * terms["motion"]

        if (
            self.temporal_weight > 0.0
            and prev_logits is not None
            and static_mask is not None
        ):
            cur = torch.sigmoid(logits.squeeze(1) if logits.dim() == 4 else logits)
            prv = torch.sigmoid(prev_logits.squeeze(1) if prev_logits.dim() == 4 else prev_logits)
            diff = (cur - prv).abs()
            terms["temporal"] = (diff * static_mask).mean()
            total = total + self.temporal_weight * terms["temporal"]

        if (
            self.feature_weight > 0.0
            and student_feat is not None
            and teacher_feat is not None
        ):
            sf = F.normalize(student_feat, dim=1)
            tf = F.normalize(teacher_feat, dim=1)
            terms["feat"] = (1.0 - (sf * tf).sum(dim=1)).mean()
            total = total + self.feature_weight * terms["feat"]

        terms["total"] = total
        return terms
