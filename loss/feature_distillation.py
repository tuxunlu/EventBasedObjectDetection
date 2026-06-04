"""Multi-level feature distillation loss for ``EventTinySeg``.

The student outputs ``(mask_logits, feature_dict)`` from
``EventTinySeg.forward_with_features``. The teacher supplies a pseudo-mask
(from SAM 2 + GroundingDINO, the existing Phase A pipeline) plus a
``teacher_features`` dict from the SAM ViT image encoder at three depths.

Loss:

    L =   λ_mask · ( BCE(s_mask, t_mask) + Dice(s_mask, t_mask) )
        + Σ_level λ_align[level] · ( 1 - cos( norm(W·s_level), norm(t_level) ) )

Each ``L_align`` is the standard cosine feature-alignment loss:

  1. Project student features through ``W`` (the 1×1 conv head living in
     ``EventTinySeg.distill_projections``).
  2. Bilinear-resize the projected student feature spatially to the teacher's
     token grid (teacher geometry is the geometry of truth).
  3. L2-normalize both along the channel dim → unit vectors per token.
  4. Loss = ``1 − mean_token( ⟨s_token, t_token⟩ )``.

Default level → SAM-layer pairing (ViT-B numbering):

    low  ← block  3 output  (early texture / edge structure)
    mid  ← block  7 output  (mid-level grouping)
    high ← block 11 output  OR  post-neck 256-d embedding

The pairing is enforced by the teacher feature extractor; this loss just
consumes a dict keyed by ``low/mid/high``.

For the mask supervision, the teacher mask is resized to the student's
output resolution (typically stride-4 binary map) with nearest-neighbour
interpolation, then the standard BCE + Dice from ``loss/distillation.py``
runs on the resized target. The output stride is much smaller than the
original SAM 2 mask resolution but adequate for the pre-filtering use case.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mask sub-loss (same shape as loss/distillation.py — reproduced here so this
# file stands alone, and so future per-level mask supervision can plug in).
# ---------------------------------------------------------------------------

def _bce_logits(logits: torch.Tensor, target: torch.Tensor,
                pos_weight: Optional[torch.Tensor]) -> torch.Tensor:
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    return F.binary_cross_entropy_with_logits(
        logits, target.float(),
        pos_weight=pos_weight, reduction="mean",
    )


def _soft_dice(logits: torch.Tensor, target: torch.Tensor,
               eps: float = 1e-6) -> torch.Tensor:
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    probs = torch.sigmoid(logits)
    target = target.float()
    inter = (probs * target).sum(dim=(-2, -1))
    denom = probs.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = (2 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


# ---------------------------------------------------------------------------
# Cosine feature-alignment kernel.
# ---------------------------------------------------------------------------

def cosine_align_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
) -> torch.Tensor:
    """Cosine alignment loss between a projected student feature map and a
    teacher feature map.

    ``student_feat``: ``(B, C, H_s, W_s)`` — projected by the student's 1×1
    head into the teacher's channel dim ``C``.
    ``teacher_feat``: ``(B, C, H_t, W_t)`` — at the teacher's native token grid.

    The student is bilinearly resized to ``(H_t, W_t)`` before alignment;
    both are L2-normalized along the channel dim. Returns ``1 − mean(cos)``.
    """
    if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
        student_feat = F.interpolate(
            student_feat, size=teacher_feat.shape[-2:],
            mode="bilinear", align_corners=False,
        )
    s = F.normalize(student_feat, dim=1, eps=1e-6)
    t = F.normalize(teacher_feat, dim=1, eps=1e-6)
    cos = (s * t).sum(dim=1)            # (B, H_t, W_t) per-token cosine
    return (1.0 - cos).mean()


# ---------------------------------------------------------------------------
# Composite loss module.
# ---------------------------------------------------------------------------

class FeatureDistillationLoss(nn.Module):
    """Mask supervision + multi-level cosine feature alignment.

    Parameters
    ----------
    mask_weight
        Weight on the mask sub-loss (BCE + Dice). Default 1.0.
    align_weights
        Per-level weight on the cosine alignment loss. Default
        ``{"low": 0.5, "mid": 1.0, "high": 1.0}``: deeper alignments are
        emphasized because the deeper student features carry more of the
        semantic burden the SAM teacher provides.
    pos_weight
        Optional positive-class weight passed to BCE. Useful if hand+arm
        pixels are a small fraction of the (downsampled) mask.
    mask_resize_mode
        ``"nearest"`` (default) keeps the teacher mask binary at the
        student's output resolution. Use ``"bilinear"`` if you intend to
        train with soft targets.
    """

    def __init__(
        self,
        mask_weight: float = 1.0,
        align_weights: Optional[Dict[str, float]] = None,
        pos_weight: Optional[float] = None,
        mask_resize_mode: str = "nearest",
    ):
        super().__init__()
        self.mask_weight = float(mask_weight)
        if align_weights is None:
            align_weights = {"low": 0.5, "mid": 1.0, "high": 1.0}
        self.align_weights = {k: float(v) for k, v in align_weights.items()}
        self.mask_resize_mode = mask_resize_mode
        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
        else:
            self._pos_weight = None

    # ------------------------------------------------------------------ pieces

    def _mask_loss(self, mask_logits: torch.Tensor,
                   teacher_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # teacher_mask: (B, H, W) at native resolution (e.g. 480x640 from
        # HandEventDataset). Resize to the student's output grid.
        if teacher_mask.shape[-2:] != mask_logits.shape[-2:]:
            mode = self.mask_resize_mode
            kw = {} if mode == "nearest" else {"align_corners": False}
            tgt = F.interpolate(
                teacher_mask.unsqueeze(1).float(),
                size=mask_logits.shape[-2:],
                mode=mode, **kw,
            ).squeeze(1)
            # Keep target strictly binary if we interpolated bilinearly.
            if mode != "nearest":
                tgt = (tgt > 0.5).float()
        else:
            tgt = teacher_mask.float()
        bce = _bce_logits(mask_logits, tgt, self._pos_weight)
        dice = _soft_dice(mask_logits, tgt)
        return {"bce": bce, "dice": dice, "total": bce + dice}

    def _align_loss(self, student_feats: Dict[str, torch.Tensor],
                    teacher_feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for level, w in self.align_weights.items():
            if w == 0.0:
                continue
            if level not in student_feats:
                raise KeyError(f"student feature missing level {level!r}")
            if level not in teacher_feats:
                raise KeyError(f"teacher feature missing level {level!r}")
            out[level] = cosine_align_loss(student_feats[level], teacher_feats[level])
        return out

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        mask_logits: torch.Tensor,
        student_feats: Dict[str, torch.Tensor],
        teacher_mask: torch.Tensor,
        teacher_feats: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        terms: Dict[str, torch.Tensor] = {}

        mask_terms = self._mask_loss(mask_logits, teacher_mask)
        terms["bce"] = mask_terms["bce"]
        terms["dice"] = mask_terms["dice"]
        terms["mask_total"] = mask_terms["total"]
        total = self.mask_weight * mask_terms["total"]

        align_terms = self._align_loss(student_feats, teacher_feats)
        for level, val in align_terms.items():
            terms[f"align_{level}"] = val
            total = total + self.align_weights[level] * val
        if align_terms:
            terms["align_total"] = sum(
                self.align_weights[k] * v for k, v in align_terms.items()
            )

        terms["total"] = total
        return terms
