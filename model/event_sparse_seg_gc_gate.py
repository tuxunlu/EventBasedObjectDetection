"""Gated global-context segmenter — :class:`EventSparseSegGC` **+ a veto gate**.

Why this model (the second diagnosed regression)
------------------------------------------------
The first global-context run (project memory ``event-seg-gc-static-fp-overfit``)
fixed the motion-tail false positives but introduced a new one: on **static / quiet
windows** it paints a solid hand-shaped blob (~14× the baseline's quiet-frame FP).
Root cause: in :class:`EventSparseSegGC` the dense global context is purely
**additive** — it is residual-added onto the bottleneck voxels and gathered to the
per-event head, so it can only ever *add* foreground evidence, **never veto it**.
Combined with the absence of a learned "no-hand" state, the global "a hand exists
here" prior fires on any busy-but-handless scene.

The fix here (Fix B) turns the global context from a *generator* into a **gate**.
The model already trains a coarse occupancy head (SA-SSD style, He et al. CVPR
2020) that the teacher mask supervises to localize the hand. We:

  1. run that occupancy head at **inference** too (it was train-only before), and
  2. add a single global **"hand-present" scalar** read off the whole-frame context
     descriptor, and
  3. **multiplicatively gate** every per-event probability by (occupancy-at-its-cell
     × hand-present): ``p_final = σ(raw_logit) · σ(occ_cell) · σ(present)``.

A multiplicative gate is a true veto — if the occupancy head says "no hand at this
cell" or the global scalar says "no hand in this window", the final probability is
driven to ~0 **regardless of how hand-like the local event looks**. That is exactly
the lever the additive context lacked. The gate is applied end-to-end (train +
inference) so the raw head and the gate co-adapt, and the occupancy head keeps its
teacher supervision so the gate stays anchored to real hand occupancy rather than
collapsing to a no-op.

Pairs with Fix A (the ``hand_drop`` no-hand augmentation in
``data/event_augment.py``): A supplies the genuine no-hand windows that teach the
global "hand-present" scalar its negative class, so A+B is the intended combination.

Numerics
--------
The per-event loss consumes **logits**, so the gate is implemented in log-prob space
and re-expanded to a logit (stable, no ``logit(product)`` blow-up):

    log_g  = logσ(occ_cell) + logσ(present)          # ≤ 0, the multiplicative gate
    log_p  = logσ(raw_logit) + log_g                  # log of the gated probability
    logit  = log_p − log(1 − exp(log_p))              # back to a logit for the loss

``σ(logit) == σ(raw_logit)·σ(occ_cell)·σ(present)`` exactly. The gate biases are
initialized **open** (≈1) so optimization starts unvetoed and learns to close on
the no-hand windows Fix A provides.

Everything else — the sparse 3D U-Net, the dense bottleneck context, geometry/
density features, the regularizers — is inherited unchanged from
:class:`EventSparseSegGC`, so any delta vs. that model is attributable to the gate.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.event_sparse_seg_gc import EventSparseSegGC
from model.sparse_backend import build_sparse_tensor_3d

# Full base __init__ signature is replicated explicitly below (not absorbed into
# **kwargs): ModelInterface.filter_init_args introspects ``__init__`` to decide which
# YAML keys to pass, so any param hidden behind *args/**kwargs would be silently
# dropped and fall back to its default (e.g. ``algo: native`` -> the implicit-GEMM
# SIGFPE). Every config key therefore needs a named parameter here.


class EventSparseSegGCGate(EventSparseSegGC):
    """:class:`EventSparseSegGC` with a multiplicative occupancy + present-scalar gate.

    Adds two new parameters on top of the base model:

    present_init_bias
        Initial bias of the global "hand-present" scalar head, in logit units. A
        positive value (default ``3.0`` → σ ≈ 0.95) starts the present gate **open**
        so early training is effectively ungated.
    occ_init_bias
        Initial bias of the occupancy head's 1×1 conv (default ``2.0`` → σ ≈ 0.88).
        Starts the per-cell gate open; the teacher supervision then closes it on
        no-hand cells.

    The auxiliary occupancy head is **required** here (it drives the gate) and is
    forced on regardless of ``aux_shape_head``. ``num_classes`` must be 1.
    """

    def __init__(
        self,
        in_features: int = 2,
        stage_channels: Sequence[int] = (24, 32, 48, 64),
        time_bins: int = 6,
        num_classes: int = 1,
        head_hidden: int = 64,
        dropout: float = 0.2,
        drop_path: float = 0.1,
        norm: str = "ln",
        algo=None,
        coord_mode: str = "relative",
        density_features: bool = False,
        density_radius: int = 2,
        context_channels: int = 48,
        context_gather: bool = True,
        aux_shape_head: bool = True,
        aux_grid: int = 32,
        aux_shape_weight: float = 0.2,
        present_init_bias: float = 3.0,
        occ_init_bias: float = 2.0,
    ):
        # The occupancy head is the gate's source — force it on regardless of config.
        super().__init__(
            in_features=in_features, stage_channels=stage_channels, time_bins=time_bins,
            num_classes=num_classes, head_hidden=head_hidden, dropout=dropout,
            drop_path=drop_path, norm=norm, algo=algo, coord_mode=coord_mode,
            density_features=density_features, density_radius=density_radius,
            context_channels=context_channels, context_gather=context_gather,
            aux_shape_head=True, aux_grid=aux_grid, aux_shape_weight=aux_shape_weight,
        )
        if self.num_classes != 1:
            raise ValueError(
                "EventSparseSegGCGate implements a binary veto gate; num_classes must be 1")

        # Global "hand-present" scalar: one logit per sample off the whole-frame
        # context descriptor. Starts open (positive bias) so the gate does not
        # strangle gradients before the occupancy head has learned anything.
        self.present_head = nn.Linear(self.context_channels, 1)
        nn.init.zeros_(self.present_head.weight)
        nn.init.constant_(self.present_head.bias, float(present_init_bias))
        # Start the per-cell occupancy gate open too.
        with torch.no_grad():
            if self.aux_shape.bias is not None:
                self.aux_shape.bias.fill_(float(occ_init_bias))

    def forward(self, batch) -> torch.Tensor:
        self._aux_logits = None
        feats = batch.feats
        N = feats.shape[0]
        if N == 0:
            return feats.new_zeros((0,))

        T, H, W = self.time_bins, int(batch.height), int(batch.width)
        B = batch.batch_size
        x = batch.coords[:, 0]
        y = batch.coords[:, 1]
        times = batch.times
        t_bin = (times * T).floor().clamp_(0, T - 1)

        parts = [feats]
        gfeat = self._geom_feats(x, y, batch.batch_idx, B, H, W)
        if gfeat is not None:
            parts.append(gfeat.to(feats.dtype))
        dfeat = self._density_feats(x, y, times, t_bin, batch.batch_idx, B, T, H, W)
        if dfeat is not None:
            parts.append(dfeat.to(feats.dtype))
        feats = torch.cat(parts, dim=1) if len(parts) > 1 else feats

        vox_coords, vox_batch, vox_feats, inverse = self._voxelize(
            x, y, t_bin, batch.batch_idx, feats, T, H, W,
        )
        sp = build_sparse_tensor_3d(vox_coords, vox_feats, vox_batch, T, H, W, B)

        s0 = self.stem(sp)
        e1 = self.enc1(self.down1(s0))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))

        # ---- dense global-context bottleneck -------------------------------------
        c3 = e3.features.shape[1]
        dense, occ, Hd, Wd, col = self._build_dense(e3, c3, B)
        ctx_map = self.context(dense, occ)                   # (B, c_ctx, Hd, Wd)
        ctx_flat = ctx_map.permute(0, 2, 3, 1).reshape(B * Hd * Wd, self.context_channels)
        vox_ctx = ctx_flat[col]
        e3 = e3.replace_feature(
            e3.features + self.ctx_drop_path(self.ctx_to_c3(vox_ctx)))

        # ---- gate ingredients (computed every pass — the gate runs at inference) -
        # Per-cell occupancy logits (teacher-supervised in training via _aux_logits).
        G = self.aux_grid
        pooled = F.adaptive_avg_pool2d(ctx_map, (G, G))
        occ_logits = self.aux_shape(pooled)                  # (B, 1, G, G)
        if self.training:
            self._aux_logits = occ_logits                    # supervised in model_interface
        # Global "hand-present" scalar from the whole-frame (valid-cell) descriptor.
        denom = occ.flatten(2).sum(dim=2).clamp(min=1.0)     # (B, 1)
        gdesc = (ctx_map * occ).flatten(2).sum(dim=2) / denom  # (B, c_ctx)
        present_logit = self.present_head(gdesc)             # (B, 1)

        d2 = self.dec2(self._add(self.up3(e3), e2))
        d1 = self.dec1(self._add(self.up2(d2), e1))
        d0 = self.dec0(self._add(self.up1(d1), s0))

        of = d0.features
        oi = d0.indices.long()
        out_key = ((oi[:, 0] * T + oi[:, 1]) * H + oi[:, 2]) * W + oi[:, 3]
        in_key = ((vox_batch * T + vox_coords[:, 0]) * H + vox_coords[:, 1]) * W + vox_coords[:, 2]
        if of.shape[0] != in_key.shape[0]:
            raise RuntimeError(
                f"submanifold voxel-count invariant violated: model returned "
                f"{of.shape[0]} voxels for {in_key.shape[0]} input voxels")
        vox_out = torch.empty_like(of)
        vox_out[torch.argsort(in_key)] = of[torch.argsort(out_key)]

        ev_ctx = self.feat_drop(vox_out[inverse])
        head_parts = [ev_ctx, feats]
        if self.context_gather:
            egy = (y.long() * Hd // H).clamp(0, Hd - 1)
            egx = (x.long() * Wd // W).clamp(0, Wd - 1)
            ecol = (batch.batch_idx.long() * Hd + egy) * Wd + egx
            head_parts.append(ctx_flat[ecol])
        raw = self.head(torch.cat(head_parts, dim=1)).squeeze(-1)   # (N,) raw logits

        # ---- multiplicative veto gate --------------------------------------------
        # p_final = σ(raw) · σ(occ_at_cell) · σ(present);  return logit(p_final).
        bidx = batch.batch_idx.long()
        gy = (y.long() * G // H).clamp(0, G - 1)
        gx = (x.long() * G // W).clamp(0, G - 1)
        occ_ev = occ_logits[bidx, 0, gy, gx]                 # (N,)
        present_ev = present_logit[bidx, 0]                  # (N,)
        log_gate = F.logsigmoid(occ_ev) + F.logsigmoid(present_ev)   # ≤ 0
        log_p = F.logsigmoid(raw) + log_gate                 # log of gated probability
        log_p = log_p.clamp(max=-1e-6)                       # keep p < 1 for the logit
        gated = log_p - torch.log1p(-torch.exp(log_p))       # logit(p) == log p − log(1−p)
        return gated
