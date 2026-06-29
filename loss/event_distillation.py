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

import math
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


def background_prototype_loss(emb, labels, batch_idx, margin: float = 1.0,
                              eps: float = 1e-6) -> torch.Tensor:
    """Discriminative background-anchored **null-state** loss (per-event embeddings).

    The global-context segmenter has no learned "no-hand" state, so on a quiet /
    handless window its additive prior still paints a blob (measured FP regression).
    This is the discriminative pull-push idea (De Brabandere et al., CVPR-W 2017,
    arXiv:1708.02551) used as an explicit null state:

      * **pull** every BACKGROUND event embedding toward its per-sample background
        prototype (the mean background embedding) — tightening one coherent
        background cluster;
      * **push** every FOREGROUND event embedding to at least ``margin`` away from
        that background prototype.

    On a window with **no hand**, every event is background → they collapse to a
    single tight cluster and there is no foreground prototype to drift toward, so the
    head is explicitly taught the null state instead of hallucinating a figure. The
    border-ownership literature is explicit that ownership presupposes a figure, so a
    boundary loss (≈0 gradient on an empty window) cannot do this — a background
    anchor can. Cost is O(N) + per-sample means; the "push" is the trivial binary
    case of the O(C²) inter-cluster term.

    ``emb`` is ``(N, D)``; ``labels`` ``(N,)`` in ``{0,1}`` (``<0`` = ignore, dropped);
    ``batch_idx`` ``(N,)`` groups events by sample so prototypes are per-window.
    """
    labels = labels.float()
    if labels.numel() and (labels < 0).any():
        keep = labels >= 0
        emb = emb[keep]; labels = labels[keep]
        batch_idx = batch_idx[keep] if batch_idx is not None else None
    if emb.shape[0] == 0:
        return emb.sum() * 0.0
    if batch_idx is None:
        batch_idx = torch.zeros(emb.shape[0], dtype=torch.long, device=emb.device)
    batch_idx = batch_idx.long()
    Bn = int(batch_idx.max().item()) + 1
    D = emb.shape[1]
    bg = labels < 0.5
    # Per-sample background prototype = mean background embedding (no-bg samples -> 0,
    # masked out below via has_bg so they contribute nothing).
    proto = emb.new_zeros(Bn, D)
    pc = emb.new_zeros(Bn, 1)
    if bool(bg.any()):
        bidx = batch_idx[bg]
        proto.index_add_(0, bidx, emb[bg])
        pc.index_add_(0, bidx, emb.new_ones(bidx.shape[0], 1))
    proto = proto / pc.clamp_min(1.0)
    proto_e = proto[batch_idx]
    dist = torch.linalg.vector_norm(emb - proto_e, dim=1)
    has_bg = pc[batch_idx, 0] > 0
    terms = []
    pull_m = bg & has_bg
    push_m = (~bg) & has_bg
    if bool(pull_m.any()):
        terms.append((dist[pull_m] ** 2).mean())
    if bool(push_m.any()):
        terms.append(((margin - dist[push_m]).clamp_min(0.0) ** 2).mean())
    if not terms:
        return emb.sum() * 0.0
    return torch.stack(terms).mean()


class EventDistillationLoss(nn.Module):
    """Composable per-event mask-distillation loss (noise- and imbalance-aware).

    A weighted sum of opt-in terms — a *pointwise* noise term (SCE / GCE / GJS) + a
    *set-level* imbalance term (Lovász-hinge / Focal-Tversky / NR-Dice / ASL):

    ``L = bce_weight·(SCE or CE) + gce_weight·GCE + gjs_weight·GJS``
    ``  + lovasz_weight·Lovász + nrdice_weight·NR-Dice + tversky_weight·FocalTversky``
    ``  + asl_weight·ASL + focal_weight·Focal + dice_weight·Dice``

    Set a term's weight to 0 to disable it. The four research terms (GCE, GJS,
    NR-Dice, ASL) are wired to the ``hand_event_seg_sparse_{gce,gjs,gjs_nrdice,
    gjs_asl}.yaml`` configs for separate A/B testing.

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

    Noise/imbalance research terms (each off by default; ``configs/
    hand_event_seg_sparse_{gce,gjs,gjs_nrdice,gjs_asl}.yaml`` enable them):

    gce_weight, gce_q, gce_truncate
        Generalized Cross-Entropy / L_q (Zhang & Sabuncu, NeurIPS 2018):
        ``(1 - p_t^q)/q``, ``q∈(0,1]`` interpolates CE(q→0)↔MAE(q=1). The
        lowest-risk noise-robust pointwise term; a strict generalization of bare
        RCE. ``gce_truncate=k>0`` clamps events with ``p_t<k`` to a constant
        (zero-gradient) — prunes likely-mislabeled events (truncated-GCE).
    gjs_weight, gjs_pi1
        Generalized Jensen-Shannon divergence, M=2 (Englesson & Azizpour,
        NeurIPS 2021): symmetric, bounded, *provably* interpolates CE↔MAE with one
        knob ``π1`` (→1 = more MAE-like = more noise-robust). The strongest single
        static noise-robust pointwise term; meant to *replace* the SCE block.
    nrdice_weight, nrdice_gamma
        Noise-Robust Dice (Wang et al., COPLE-Net, IEEE TMI 2020): a soft-Dice with
        an MAE-analog numerator ``Σ|p-g|^γ`` (γ∈(1,2]; γ=2 = standard Dice, γ→1 =
        max robustness). The only term that is overlap-based *and* noise-robust.
        Per-sample over ``batch_idx``.
    asl_weight, asl_gamma_pos, asl_gamma_neg, asl_clip
        Asymmetric Loss (Ben-Baruch et al., ICCV 2021): decoupled +/- focusing with
        a probability margin that hard-discards easy negatives. ``γ+=0`` never
        suppresses the rare foreground; ``γ-`` focuses the background flood; the
        ``asl_clip`` margin shifts/clamps negative probabilities. An imbalance lever
        that, unlike plain focal, does not starve the rare class.
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
        gce_weight: float = 0.0,
        gce_q: float = 0.7,
        gce_truncate: float = 0.0,
        gjs_weight: float = 0.0,
        gjs_pi1: float = 0.5,
        nrdice_weight: float = 0.0,
        nrdice_gamma: float = 1.5,
        asl_weight: float = 0.0,
        asl_gamma_pos: float = 0.0,
        asl_gamma_neg: float = 4.0,
        asl_clip: float = 0.05,
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
        self.gce_weight = float(gce_weight)
        self.gce_q = float(gce_q)
        self.gce_truncate = float(gce_truncate)
        self.gjs_weight = float(gjs_weight)
        self.gjs_pi1 = float(gjs_pi1)
        self.nrdice_weight = float(nrdice_weight)
        self.nrdice_gamma = float(nrdice_gamma)
        self.asl_weight = float(asl_weight)
        self.asl_gamma_pos = float(asl_gamma_pos)
        self.asl_gamma_neg = float(asl_gamma_neg)
        self.asl_clip = float(asl_clip)
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

    def _posweight(self, per_event, labels):
        """Apply the optional per-event class weight ``1+(pos_weight-1)·y`` and mean."""
        if self._pos_weight is not None:
            per_event = per_event * (1.0 + (self._pos_weight - 1.0) * labels)
        return per_event.mean()

    def _gce(self, logits, labels):
        """Generalized Cross-Entropy / L_q (Zhang & Sabuncu, NeurIPS 2018)."""
        eps = self.sce_clip
        p = torch.sigmoid(logits)
        p_t = (p * labels + (1.0 - p) * (1.0 - labels)).clamp_min(eps)
        q = max(self.gce_q, 1e-3)
        loss = (1.0 - p_t.pow(q)) / q
        if self.gce_truncate > 0.0:
            k = self.gce_truncate
            const = (1.0 - k ** q) / q
            loss = torch.where(p_t < k, torch.full_like(loss, const), loss)
        return self._posweight(loss, labels)

    def _gjs(self, logits, labels):
        """Generalized Jensen-Shannon (M=2) noisy-label loss (Englesson NeurIPS 2021).

        Interpolates CE↔MAE via ``pi1``; symmetric and bounded. For binary, the
        2-class distributions are ``P=[1-p, p]`` and ``Y=[1-y, y]``.
        """
        eps = self.sce_clip
        pi = min(max(self.gjs_pi1, 1e-3), 1.0 - 1e-3)
        p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
        P0, P1 = 1.0 - p, p
        Y0, Y1 = 1.0 - labels, labels
        m0 = (pi * Y0 + (1.0 - pi) * P0).clamp_min(eps)
        m1 = (pi * Y1 + (1.0 - pi) * P1).clamp_min(eps)

        def kl_y(Y, m):                          # Y·log(Y/m) with the 0·log0=0 rule
            return torch.where(Y > 0, Y * torch.log(Y.clamp_min(eps) / m),
                               torch.zeros_like(Y))
        kl_Y = kl_y(Y0, m0) + kl_y(Y1, m1)
        kl_P = P0 * torch.log(P0 / m0) + P1 * torch.log(P1 / m1)
        Z = -(1.0 - pi) * math.log(1.0 - pi)
        loss = (pi * kl_Y + (1.0 - pi) * kl_P) / Z
        return self._posweight(loss, labels)

    def _nr_dice(self, logits, labels, batch_idx, eps=1e-5):
        """Noise-Robust Dice (Wang et al., COPLE-Net, IEEE TMI 2020), per sample."""
        p = torch.sigmoid(logits)
        num_e = (p - labels).abs().pow(self.nrdice_gamma)
        den_e = p * p + labels * labels
        if batch_idx is None:
            return num_e.sum() / (den_e.sum() + eps)
        B = int(batch_idx.max().item()) + 1
        num = p.new_zeros(B).scatter_add_(0, batch_idx, num_e)
        den = p.new_zeros(B).scatter_add_(0, batch_idx, den_e) + eps
        present = torch.unique(batch_idx)
        return (num[present] / den[present]).mean()

    def _asl(self, logits, labels, eps=1e-6):
        """Asymmetric Loss (Ben-Baruch et al., ICCV 2021); per-event independent binary."""
        p = torch.sigmoid(logits)
        l_pos = (1.0 - p).clamp_min(0.0).pow(self.asl_gamma_pos) * torch.log(p.clamp_min(eps))
        p_m = (p - self.asl_clip).clamp_min(0.0)                  # probability-shifted neg
        l_neg = p_m.pow(self.asl_gamma_neg) * torch.log((1.0 - p_m).clamp_min(eps))
        loss = -(labels * l_pos + (1.0 - labels) * l_neg)
        return loss.mean()

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

        # Noise-robust pointwise alternatives (each replaces the SCE block when its
        # config sets bce_weight=0); see class docstring for the references.
        if self.gce_weight > 0.0:
            terms["gce"] = self._gce(logits, labels)
            total = total + self.gce_weight * terms["gce"]

        if self.gjs_weight > 0.0:
            terms["gjs"] = self._gjs(logits, labels)
            total = total + self.gjs_weight * terms["gjs"]

        if self.nrdice_weight > 0.0:
            terms["nrdice"] = self._nr_dice(logits, labels, batch_idx)
            total = total + self.nrdice_weight * terms["nrdice"]

        if self.asl_weight > 0.0:
            terms["asl"] = self._asl(logits, labels)
            total = total + self.asl_weight * terms["asl"]

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
