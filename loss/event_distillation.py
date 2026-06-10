"""Per-event distillation/segmentation loss for the sparse event-stream segmenter.

:class:`EventDistillationLoss` is the **per-event** loss: each event carries a
foreground/background target (``label = mask[y,x]``) and the model emits one logit
per event. It is a rethink of the old BCE-only loss, designed for the two things
that actually break per-event hand segmentation:

* **Severe class imbalance** (foreground events are a minority) — handled by a
  per-sample **Lovász-hinge** (Berman et al. CVPR 2018), a direct IoU surrogate
  that is insensitive to the huge true-negative count, plus an optional
  **Focal-Tversky** term (Abraham & Khan ISBI 2019) with a tunable recall knob.
* **Label noise** (pseudo-masks are imperfect) — handled by **Symmetric
  Cross-Entropy** (Wang et al. ICCV 2019: ``SCE = α·CE + β·RCE``), which is provably
  more noise-tolerant than plain CE, and by an **ignore band** (``label < 0`` rows
  are dropped) so a boundary trimap can mask out the most-uncertain events.

We deliberately do **not** default to plain Focal loss: its ``(1-p_t)^γ``
hard-example weighting up-weights exactly the noisy-boundary events and *amplifies*
structured label noise.

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


# --------------------------------------------------------------------------- #
# Lovász-hinge (binary) — Berman, Triki, Blaschko, CVPR 2018.                  #
# Direct surrogate for the Jaccard (IoU) loss on a flat logit/label vector,    #
# which is exactly the shape of a sparse per-event prediction.                 #
# --------------------------------------------------------------------------- #
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovász hinge over a flat vector of per-event logits/labels in {0,1}."""
    if labels.numel() == 0:
        return logits.sum() * 0.0
    if labels.sum() == 0:                       # no foreground -> hinge is degenerate
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def _lovasz_hinge_per_sample(logits, labels, batch_idx) -> torch.Tensor:
    """Mean Lovász-hinge computed independently per sample (needs a full event set)."""
    if batch_idx is None:
        return lovasz_hinge_flat(logits, labels)
    losses = []
    for b in torch.unique(batch_idx):
        m = batch_idx == b
        losses.append(lovasz_hinge_flat(logits[m], labels[m]))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


class EventDistillationLoss(nn.Module):
    """Composable per-event mask-distillation loss (noise- and imbalance-aware).

    ``L = ce_weight·CE_or_SCE  +  lovasz_weight·Lovász-hinge  +  tversky_weight·FocalTversky``

    Parameters
    ----------
    bce_weight
        Weight on the pointwise cross-entropy term. (Name kept for config
        back-compat; it scales CE or, when ``sce_beta>0``, Symmetric-CE.)
    pos_weight
        Optional positive-class weight for the BCE/CE part. ``None`` disables.
    sce_beta, sce_alpha
        Symmetric-CE mix (Wang ICCV 2019). ``sce_beta>0`` turns the pointwise term
        into ``sce_alpha·CE + sce_beta·RCE`` (noise-robust). ``sce_beta=0`` (default)
        keeps plain (weighted) BCE so existing configs are unchanged.
    sce_clip
        Clamp for the reverse-CE log (Wang use ``A=-4`` ⇔ clip ≈ ``exp(-4)``).
    lovasz_weight
        Weight on the per-sample Lovász-hinge (direct IoU surrogate; the main
        imbalance lever). Computed per ``batch_idx`` group.
    tversky_weight, tversky_alpha, tversky_beta, tversky_gamma
        Optional Focal-Tversky term. ``beta>alpha`` favors recall (penalizes false
        negatives harder) — useful when the mask under-covers the thin arm.
    focal_weight, focal_gamma, focal_alpha
        Legacy focal term, kept for back-compat (default off). Prefer SCE+Lovász.
    label_smoothing
        Soften hard targets toward ``0.5`` by this amount (cheap noise robustness).
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        pos_weight: Optional[float] = None,
        sce_beta: float = 0.0,
        sce_alpha: float = 1.0,
        sce_clip: float = 1.8e-2,           # ≈ exp(-4)
        lovasz_weight: float = 0.0,
        tversky_weight: float = 0.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_gamma: float = 1.3333,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        dice_weight: float = 0.0,           # legacy alias for a global soft-dice
        label_smoothing: float = 0.0,
        ignore_negative: bool = True,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.sce_beta = float(sce_beta)
        self.sce_alpha = float(sce_alpha)
        self.sce_clip = float(sce_clip)
        self.lovasz_weight = float(lovasz_weight)
        self.tversky_weight = float(tversky_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.tversky_gamma = float(tversky_gamma)
        self.focal_weight = float(focal_weight)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.dice_weight = float(dice_weight)
        self.label_smoothing = float(label_smoothing)
        self.ignore_negative = bool(ignore_negative)
        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor(float(pos_weight)))
        else:
            self._pos_weight = None

    # --------------------------------------------------------------- terms
    def _ce(self, logits, labels):
        return F.binary_cross_entropy_with_logits(
            logits, labels,
            pos_weight=self._pos_weight if self._pos_weight is not None else None,
            reduction="mean",
        )

    def _rce(self, logits, labels):
        """Reverse cross-entropy: treat the prediction as 'truth', the label as 'pred'."""
        p = torch.sigmoid(logits).clamp(self.sce_clip, 1.0 - self.sce_clip)
        y = labels.clamp(self.sce_clip, 1.0 - self.sce_clip)
        return -(p * torch.log(y) + (1.0 - p) * torch.log(1.0 - y)).mean()

    def _focal(self, logits, labels):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p_t = p * labels + (1.0 - p) * (1.0 - labels)
        alpha_t = self.focal_alpha * labels + (1.0 - self.focal_alpha) * (1.0 - labels)
        return (alpha_t * (1.0 - p_t).pow(self.focal_gamma) * ce).mean()

    def _focal_tversky(self, logits, labels, eps=1e-6):
        p = torch.sigmoid(logits)
        tp = (p * labels).sum()
        fp = (p * (1.0 - labels)).sum()
        fn = ((1.0 - p) * labels).sum()
        ti = (tp + eps) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + eps)
        return (1.0 - ti).pow(self.tversky_gamma)

    def _dice(self, logits, labels, eps=1e-6):
        p = torch.sigmoid(logits)
        inter = (p * labels).sum()
        return 1.0 - (2.0 * inter + eps) / (p.sum() + labels.sum() + eps)

    # --------------------------------------------------------------- forward
    def forward(self, logits, labels, batch_idx=None) -> Dict[str, torch.Tensor]:
        logits = _flatten_logits(logits)
        labels = labels.float()

        if self.ignore_negative and (labels < 0).any():
            keep = labels >= 0
            logits, labels = logits[keep], labels[keep]
            batch_idx = batch_idx[keep] if batch_idx is not None else None

        terms: Dict[str, torch.Tensor] = {}
        if logits.numel() == 0:
            zero = logits.sum()                 # 0-d, carries grad graph if any
            terms["bce"] = zero
            terms["total"] = zero
            return terms

        tgt = labels
        if self.label_smoothing > 0.0:
            tgt = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        ce = self._ce(logits, tgt)
        terms["bce"] = ce
        if self.sce_beta > 0.0:
            rce = self._rce(logits, labels)
            terms["rce"] = rce
            pointwise = self.sce_alpha * ce + self.sce_beta * rce
        else:
            pointwise = ce
        total = self.bce_weight * pointwise

        if self.lovasz_weight > 0.0:
            lov = _lovasz_hinge_per_sample(logits, labels, batch_idx)
            terms["lovasz"] = lov
            total = total + self.lovasz_weight * lov

        if self.tversky_weight > 0.0:
            tv = self._focal_tversky(logits, labels)
            terms["tversky"] = tv
            total = total + self.tversky_weight * tv

        if self.focal_weight > 0.0:
            terms["focal"] = self._focal(logits, labels)
            total = total + self.focal_weight * terms["focal"]

        if self.dice_weight > 0.0:
            terms["dice"] = self._dice(logits, labels)
            total = total + self.dice_weight * terms["dice"]

        terms["total"] = total
        return terms
