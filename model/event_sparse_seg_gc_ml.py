"""Multi-level global-context segmenter — dense context at **every encoder level**.

:class:`model.event_sparse_seg_gc.EventSparseSegGC` injects a dense global-context
module only at the deepest (spatially ``/8``) bottleneck. That gives a whole-frame
receptive field for the coarsest features, but the finer encoder levels (``/2`` and
``/4``) still reason with the purely-submanifold field — they never see across empty
space — so the global hand-blob shape only reaches them indirectly, after it has
been down/up-sampled through the bottleneck.

This model adds an independent :class:`_DenseContext` at the ``/2`` and ``/4``
encoder levels **as well as** the ``/8`` bottleneck, building a feature pyramid of
dense context. Each level's active voxels are time-collapsed into a small dense map,
a dense 2D context net (3×3 + dilated convs that *do* cross empty pixels, plus a
global-pool branch) processes it, and the result is projected back and residual-
added onto that level's voxels before the next downsample. Now every resolution of
the encoder — fine boundary features included — reasons with cross-empty-space
spatial context, not just the bottleneck. This is the standard "context at multiple
scales" recipe (PSPNet/HRNet-style multi-scale fusion) ported onto the sparse
backbone, and it directly targets the look-alike-background false positives that a
single coarse context map can miss at fine scale.

Cost note: the finer dense maps are larger (``/2`` ≈ 240×320, ``/4`` ≈ 120×160) but
still fixed-size — independent of event count — so inference stays event-rate-
proportional. The encoder-level context uses a narrower width
(``context_channels_enc``) than the bottleneck (``context_channels``) to keep the
finer, larger maps cheap.

The bottleneck context, the train-only auxiliary occupancy head, the per-event head
(which still gathers the bottleneck ``/8`` context), and every regularizer are
inherited unchanged from :class:`EventSparseSegGC`, so the only difference vs. that
model is the added encoder-level context — keeping the ablation attributable.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.event_sparse_seg_gc import EventSparseSegGC, _DenseContext, _DropPath
from model.sparse_backend import build_sparse_tensor_3d


class EventSparseSegGCML(EventSparseSegGC):
    """:class:`EventSparseSegGC` + dense global context at the ``/2`` and ``/4`` levels.

    Extra parameters on top of the base model:

    context_channels_enc
        Width of the dense context map at the finer encoder levels (``/2``, ``/4``).
        Narrower than the bottleneck ``context_channels`` to keep the larger,
        finer maps cheap. Default 32.
    context_levels
        Which encoder levels get an added dense-context module (besides the always-on
        ``/8`` bottleneck): ``3`` → both ``/2`` and ``/4``; ``2`` → only ``/4``.
        Default 3.
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
        context_channels_enc: int = 32,
        context_levels: int = 3,
    ):
        # Full base signature replicated explicitly (see EventSparseSegGCGate note):
        # ModelInterface.filter_init_args drops any YAML key without a named param,
        # so *args/**kwargs would silently lose ``algo: native`` etc.
        super().__init__(
            in_features=in_features, stage_channels=stage_channels, time_bins=time_bins,
            num_classes=num_classes, head_hidden=head_hidden, dropout=dropout,
            drop_path=drop_path, norm=norm, algo=algo, coord_mode=coord_mode,
            density_features=density_features, density_radius=density_radius,
            context_channels=context_channels, context_gather=context_gather,
            aux_shape_head=aux_shape_head, aux_grid=aux_grid, aux_shape_weight=aux_shape_weight,
        )
        _, c1, c2, _c3 = (int(c) for c in stage_channels)
        self.context_channels_enc = int(context_channels_enc)
        self.context_levels = int(context_levels)
        if self.context_levels not in (2, 3):
            raise ValueError("context_levels must be 2 (/4 only) or 3 (/2 and /4)")
        cce = self.context_channels_enc

        # /4 level (e2) — always added when multi-level is enabled.
        self.context2 = _DenseContext(c2, cce)
        self.ctx2_to_c2 = nn.Linear(cce, c2)
        self.ctx2_drop_path = _DropPath(drop_path)
        # /2 level (e1) — added only for the full 3-level pyramid.
        if self.context_levels >= 3:
            self.context1 = _DenseContext(c1, cce)
            self.ctx1_to_c1 = nn.Linear(cce, c1)
            self.ctx1_drop_path = _DropPath(drop_path)
        else:
            self.context1 = self.ctx1_to_c1 = self.ctx1_drop_path = None

    def _inject(self, sp, ctx_mod, proj, dp, B):
        """Build this level's dense context and residual-add it back onto the voxels.

        Returns the updated sparse tensor. ``col`` (per-voxel flat ``(b,y,x)`` column
        index) is row-aligned to ``sp.features`` exactly as in the bottleneck path.
        """
        c = sp.features.shape[1]
        dense, occ, Hd, Wd, col = self._build_dense(sp, c, B)
        ctx_map = ctx_mod(dense, occ)                        # (B, cce, Hd, Wd)
        flat = ctx_map.permute(0, 2, 3, 1).reshape(B * Hd * Wd, ctx_map.shape[1])
        return sp.replace_feature(sp.features + dp(proj(flat[col])))

    def forward(self, batch) -> torch.Tensor:
        self._aux_logits = None
        feats = batch.feats
        N = feats.shape[0]
        if N == 0:
            return feats.new_zeros((0,) if self.num_classes == 1 else (0, self.num_classes))

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
        # /2 dense context injection (full pyramid only).
        if self.context1 is not None:
            e1 = self._inject(e1, self.context1, self.ctx1_to_c1, self.ctx1_drop_path, B)
        e2 = self.enc2(self.down2(e1))
        # /4 dense context injection.
        e2 = self._inject(e2, self.context2, self.ctx2_to_c2, self.ctx2_drop_path, B)
        e3 = self.enc3(self.down3(e2))

        # ---- dense global-context bottleneck (/8, inherited) ---------------------
        c3 = e3.features.shape[1]
        dense, occ, Hd, Wd, col = self._build_dense(e3, c3, B)
        ctx_map = self.context(dense, occ)                   # (B, c_ctx, Hd, Wd)
        ctx_flat = ctx_map.permute(0, 2, 3, 1).reshape(B * Hd * Wd, self.context_channels)
        vox_ctx = ctx_flat[col]
        e3 = e3.replace_feature(
            e3.features + self.ctx_drop_path(self.ctx_to_c3(vox_ctx)))

        if self.aux_shape is not None and self.training:
            G = self.aux_grid
            pooled = F.adaptive_avg_pool2d(ctx_map, (G, G))
            self._aux_logits = self.aux_shape(pooled)

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
        logits = self.head(torch.cat(head_parts, dim=1))

        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits
