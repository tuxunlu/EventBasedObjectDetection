"""Event-native **3D** per-event segmenter: a submanifold sparse-conv U-Net over
the ``(t, y, x)`` event volume, with a per-event MLP head.

What changed vs. the old 2D version (and why the old one scored ~0.10 IoU)
-------------------------------------------------------------------------
The previous model collapsed every event at a pixel into a single 2D site
(``unique(y*W + x)``), discarding the **entire temporal axis** before a 2D
submanifold U-Net — so it was neither 3D nor truly per-event, and at this
sensor's ~1.2 events/pixel the collapse bought almost no dedup. This rewrite is
genuinely event-native in 3D:

* **3D voxelization.** Events are binned into ``(t_bin, y, x)`` voxels (``T`` time
  bins). Co-pixel events at different times are now *distinct* voxels, so motion
  is visible to the network — the moving hand/arm is a 3D space-time structure.
* **Anisotropic downsampling.** Strided ``SparseConv3d`` uses ``stride=(1, 2, 2)``
  — it grows the spatial receptive field (needed to segment a whole hand from
  sparse edges) while *preserving* temporal resolution. Collapsing ``t`` is the
  documented #1 event-cloud failure mode (PointNet++ dropping to ~54% on
  gestures, Ren 2024); we avoid it structurally.
* **Per-event output head.** Submanifold convs preserve the voxel set, so each
  voxel carries a context feature. A small MLP reads, *for every event*, its
  voxel's context feature ⊕ the event's own ``(polarity, exact normalized time,
  in-voxel time offset)`` and emits **one logit per event** — a true event-stream
  output, distinct even for events sharing a voxel. ``logits[i]`` pairs with
  ``batch.labels[i]`` regardless of any internal spconv reordering.

Architectural basis: submanifold sparse convolution (Graham & van der Maaten
2017; Graham et al. CVPR 2018 — SubMConv U-Net for per-point semantic seg). The
``(T,H,W)`` sparse-voxel design follows the 4D/3D sparse-conv line (Choy et al.
CVPR 2019). Cost scales with the number of active voxels (≈ #events), not the
pixel grid, so inference is cheap and event-rate-proportional; the synchronous
net can later be wrapped with AsyNet-style (Messikommer ECCV 2020) recursive
updates for an asynchronous per-event stream.

Shape contract
--------------
``forward(batch) -> logits``:
  * ``(N,)``            when ``num_classes == 1`` (default), or
  * ``(N, num_classes)`` otherwise,
where ``N == batch.coords.shape[0]`` is the number of **events** and row ``i``
corresponds to ``batch.coords[i]`` / ``batch.labels[i]``.

Expected ``batch`` fields (see ``data/sparse_event_collate.py``):
  ``coords (N,2) long`` (x, y); ``times (N,) float`` in ``[0,1]``;
  ``feats (N,F) float`` per-event features; ``batch_idx (N,) long``;
  ``height``, ``width``, ``batch_size``.

Backend: spconv v2 via ``model/sparse_backend.py`` (import-guarded).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sparse_backend import build_sparse_tensor_3d, require_spconv


def _resolve_algo(algo):
    """Map a friendly algo name (or ``None``/``ConvAlgo``) to an spconv ``ConvAlgo``.

    ``None`` keeps spconv's own default. ``"native"`` (gather + dense cuBLAS GEMM
    + scatter) sidesteps the cumm implicit-GEMM tile heuristics that SIGFPE under a
    CUDA runtime newer than the spconv wheel's build-time CUDA (the reason the old
    config pinned ``algo: native``).
    """
    if algo is None or not isinstance(algo, str):
        return algo
    from spconv.core import ConvAlgo
    table = {
        "native": ConvAlgo.Native,
        "implicit_gemm": ConvAlgo.MaskImplicitGemm,
        "mask_implicit_gemm": ConvAlgo.MaskImplicitGemm,
        "mask_split_implicit_gemm": ConvAlgo.MaskSplitImplicitGemm,
    }
    key = algo.strip().lower()
    if key not in table:
        raise ValueError(f"unknown algo {algo!r}; choose from {sorted(table)}")
    return table[key]


def _make_norm1d(c: int, kind: str) -> nn.Module:
    """Norm over the per-voxel feature vector ``(M, C)``.

    ``"ln"`` (LayerNorm over channels) is the LOSO-safe default: it normalizes each
    voxel independently, so the held-out subject's statistics never mix across the
    batch the way ``BatchNorm1d`` over voxels would.
    """
    if kind == "bn":
        return nn.BatchNorm1d(c)
    if kind == "ln":
        return nn.LayerNorm(c)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm kind: {kind!r} (expected 'bn', 'ln', or 'none')")


class _SubMBlock(nn.Module):
    """SubMConv3d -> norm -> ReLU. Preserves the active-voxel set (per-voxel refine)."""

    def __init__(self, c_in: int, c_out: int, indice_key: str, norm: str = "ln",
                 k: int = 3, algo=None):
        super().__init__()
        spconv = require_spconv()
        self.conv = spconv.SubMConv3d(c_in, c_out, kernel_size=k, bias=False,
                                      indice_key=indice_key, algo=algo)
        self.norm = _make_norm1d(c_out, norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return x.replace_feature(self.act(self.norm(x.features)))


class _DownBlock(nn.Module):
    """Strided SparseConv3d -> norm -> ReLU. Halves spatial res, keeps temporal res.

    ``stride=(1, 2, 2)`` over ``(t, y, x)`` — anisotropic on purpose (see module
    docstring). ``kernel=(1, 3, 3)`` so the downsampling conv is purely spatial
    (temporal mixing is done by the submanifold blocks, which keep ``t`` intact).
    """

    def __init__(self, c_in: int, c_out: int, indice_key: str, norm: str = "ln", algo=None):
        super().__init__()
        spconv = require_spconv()
        self.conv = spconv.SparseConv3d(c_in, c_out, kernel_size=(1, 3, 3),
                                        stride=(1, 2, 2), padding=(0, 1, 1),
                                        bias=False, indice_key=indice_key, algo=algo)
        self.norm = _make_norm1d(c_out, norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return x.replace_feature(self.act(self.norm(x.features)))


class _TemporalGRU(nn.Module):
    """Recurrent refinement along the temporal axis at each spatial column.

    The bottleneck sparse tensor has sites ``(b, t, y, x)`` at the *full* temporal
    resolution ``T`` — the encoder downsamples spatially only (``stride=(1, 2, 2)``),
    so ``t`` is intact at every level. The submanifold/strided convs mix ``t`` only
    within a small kernel window; this module instead lets the network integrate the
    *whole* motion history of a spatial location before decoding.

    For each spatial column ``(b, y, x)`` we scatter its occupied voxels into a dense
    ``(G, T, C)`` grid (``G`` = number of distinct columns, gaps zero-filled), sweep an
    ``nn.GRU`` along the ``T`` axis, and gather the refined per-voxel features back to
    the original voxel rows. The active-voxel SET is unchanged — we only rewrite each
    voxel's feature vector (residual add) — so every downstream submanifold/inverse-conv
    invariant in the U-Net still holds. ``T`` is small (6-10), so the dense pass is cheap.

    ``bidirectional`` (default) lets each time bin see future motion within the window —
    appropriate for offline window segmentation. Set it ``False`` for a causal pass if
    the model is later wrapped for streaming inference.
    """

    def __init__(self, channels: int, hidden: int | None = None,
                 bidirectional: bool = True, norm: str = "ln"):
        super().__init__()
        hidden = int(hidden) if hidden else channels
        self.gru = nn.GRU(channels, hidden, batch_first=True,
                          bidirectional=bidirectional)
        out = hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(out, channels)
        self.norm = _make_norm1d(channels, norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, T: int):
        feats = x.features                                 # (M, C)
        idx = x.indices.long()                             # columns [b, t, y, x]
        b, t, y, xx = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        # spatial_shape == [T, H_ds, W_ds]; key each spatial column (b, y, x).
        _, Hd, Wd = (int(s) for s in x.spatial_shape)
        col = (b * Hd + y) * Wd + xx
        uniq, g = torch.unique(col, return_inverse=True)
        G = uniq.numel()
        C = feats.shape[1]
        # (g, t) pairs are unique (voxels are unique in (b,t,y,x)), so direct
        # index-assignment scatters without collision; empty (g,t) slots stay zero.
        dense = feats.new_zeros(G, T, C)
        dense[g, t] = feats
        out, _ = self.gru(dense)                           # (G, T, out)
        out = self.proj(out)                               # (G, T, C)
        ref = self.act(self.norm(out[g, t]))               # (M, C), back to voxel rows
        return x.replace_feature(feats + ref)              # residual refine


class _UpBlock(nn.Module):
    """SparseInverseConv3d keyed to a prior ``_DownBlock`` -> norm -> ReLU. Restores sites."""

    def __init__(self, c_in: int, c_out: int, indice_key: str, norm: str = "ln", algo=None):
        super().__init__()
        spconv = require_spconv()
        self.conv = spconv.SparseInverseConv3d(c_in, c_out, kernel_size=(1, 3, 3),
                                               bias=False, indice_key=indice_key, algo=algo)
        self.norm = _make_norm1d(c_out, norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return x.replace_feature(self.act(self.norm(x.features)))


class EventSparseSeg(nn.Module):
    """3D submanifold sparse-conv U-Net + per-event head for per-event hand/arm seg.

    Parameters
    ----------
    in_features
        Number of per-event input feature channels (the dataset's feature builder;
        default 4 = signed polarity, normalized time-in-window, in-voxel time
        offset, log local count).
    stage_channels
        4-tuple ``(c0, c1, c2, c3)`` of channel widths (stem + three encoder
        levels). Default ``(24, 32, 48, 64)``.
    time_bins
        Number of temporal voxel bins ``T`` (the third spatial dim of the volume).
        Larger ``T`` = finer motion resolution at higher cost. Default 6.
    num_classes
        Output channels per event. ``1`` for a binary foreground logit.
    head_hidden
        Hidden width of the per-event MLP head. Default 64.
    dropout
        Dropout probability applied to (a) the per-event voxel-context feature
        before the head and (b) inside the head MLP. The model otherwise has no
        stochastic regularization, and the LOSO runs overfit the training subjects
        within ~3 epochs — a non-zero dropout (try 0.1-0.3) directly targets that.
        Default 0.0 (off).
    norm
        ``"ln"`` (default, LOSO-safe), ``"bn"``, or ``"none"``.
    algo
        Sparse-conv algorithm forwarded to every spconv layer (see ``_resolve_algo``).
        ``None`` keeps spconv's default; ``"native"`` avoids the implicit-GEMM SIGFPE
        when the wheel's build-time CUDA is older than the torch runtime.
    geom_features
        Append normalized ``(x, y)`` coordinates (each in ``[-1, 1]``) as two extra
        per-event channels. A weak spatial prior: the hand/arm is a connected region,
        while the distractor sources (keyboard edges, scattered sensor noise) are not.
        Computed inside ``forward`` from ``batch.coords`` so it is correct *after*
        event-stream augmentation. Default off (back-compat).
    density_features
        Append 3 neighborhood motion/density channels per event — ``log`` local
        event count, and the neighborhood mean and std of normalized event time —
        computed from a box-blurred dense ``(B, H, W)`` accumulation (radius
        ``density_radius``). These separate a coherent, dense, time-localized
        moving hand from sparse/isolated noise and from edges with a different
        temporal signature. The single feature the original 2-channel input lacked
        most (it could only see ``polarity`` and ``time``). Default off (back-compat).
    density_radius
        Half-width of the square neighborhood used by ``density_features`` (kernel
        size ``2*r + 1``). Default 2 (5x5).
    density_time_resolved
        Make the ``density_features`` channels **time-resolved**: accumulate and
        box-blur the local count / event-time mean+std *within each time bin*
        ``(b, t_bin, y, x)`` rather than collapsing over all ``t`` at a pixel. Without
        this, a pixel along the hand's trajectory carries both the hand's brief burst
        AND the static-source events that fired there at other moments, so its density
        looks "busy" regardless of *when* anything happened — which encodes the motion
        wake as hand-like and drives the trajectory false positives. Time-resolving it
        lets the model see that the wake's events fired at a different instant than the
        hand passage. Requires ``density_features``. Default off (back-compat).
    temporal_interp_head
        Have the per-event head gather its voxel-context feature by **interpolating
        between the event's own ``t_bin`` voxel and the next ``t_bin`` voxel** at the
        same ``(b, y, x)``, weighted by the event's fractional time, instead of reading
        the single bin's feature. At coarse ``T`` the context is shared by every event
        in a ``t_bin``, so two co-pixel events at different sub-bin moments (hand edge
        vs. a static-source event in the swept tube) get an identical context and the
        head cannot separate them; fractional interpolation gives each event a context
        reflecting its actual moment. Default off (back-compat).
    recurrent
        Insert a recurrent (GRU) module at the bottleneck that sweeps the temporal
        axis at each spatial column, integrating the whole window's motion history
        before decoding (see ``_TemporalGRU``). The encoder keeps ``t`` at full
        resolution, so this complements the convs' local temporal mixing with a
        long-range recurrent pass. Default off (back-compat).
    recurrent_hidden
        Hidden width of the bottleneck GRU. ``None`` (default) ties it to the
        bottleneck channel width ``c3``.
    recurrent_bidirectional
        Run the bottleneck GRU in both temporal directions (default ``True``, sees
        future motion within the window — fine for offline window segmentation). Set
        ``False`` for a causal pass suitable for streaming inference.
    """

    def __init__(
        self,
        in_features: int = 4,
        stage_channels: Sequence[int] = (24, 32, 48, 64),
        time_bins: int = 6,
        num_classes: int = 1,
        head_hidden: int = 64,
        dropout: float = 0.0,
        norm: str = "ln",
        algo=None,
        geom_features: bool = False,
        density_features: bool = False,
        density_radius: int = 2,
        density_time_resolved: bool = False,
        temporal_interp_head: bool = False,
        recurrent: bool = False,
        recurrent_hidden: int | None = None,
        recurrent_bidirectional: bool = True,
    ):
        super().__init__()
        require_spconv()  # fail fast with a clear install hint
        if len(stage_channels) != 4:
            raise ValueError("stage_channels must be a 4-tuple (c0, c1, c2, c3)")
        c0, c1, c2, c3 = (int(c) for c in stage_channels)
        self.in_features = int(in_features)
        self.time_bins = int(time_bins)
        self.num_classes = int(num_classes)
        self.geom_features = bool(geom_features)
        self.density_features = bool(density_features)
        self.density_radius = max(1, int(density_radius))
        self.density_time_resolved = bool(density_time_resolved)
        self.temporal_interp_head = bool(temporal_interp_head)
        # Extra per-event channels synthesized in forward (post-augmentation):
        # +2 for (x_norm, y_norm), +3 for (log-density, nbhd time mean, nbhd time std).
        self.n_extra = (2 if self.geom_features else 0) \
            + (3 if self.density_features else 0)
        # Effective per-event feature width seen by the voxel mean + the head.
        feat_dim = self.in_features + self.n_extra
        algo = _resolve_algo(algo)

        # Per-voxel input = mean of the voxel's per-event features, plus a log
        # event-count channel appended in forward -> feat_dim + 1.
        vox_in = feat_dim + 1

        # Stem keeps full (t, y, x) resolution (submanifold) -> sites == voxels.
        self.stem = _SubMBlock(vox_in, c0, indice_key="subm0", norm=norm, algo=algo)

        # Encoder: each level spatially downsamples (sp{n}) then refines (subm{n}).
        self.down1 = _DownBlock(c0, c1, indice_key="sp1", norm=norm, algo=algo)
        self.enc1 = _SubMBlock(c1, c1, indice_key="subm1", norm=norm, algo=algo)
        self.down2 = _DownBlock(c1, c2, indice_key="sp2", norm=norm, algo=algo)
        self.enc2 = _SubMBlock(c2, c2, indice_key="subm2", norm=norm, algo=algo)
        self.down3 = _DownBlock(c2, c3, indice_key="sp3", norm=norm, algo=algo)
        self.enc3 = _SubMBlock(c3, c3, indice_key="subm3", norm=norm, algo=algo)

        # Recurrent bottleneck: sweep the temporal axis at each spatial column,
        # integrating the full window's motion history before decoding (off by default).
        self.recurrent = _TemporalGRU(
            c3, hidden=recurrent_hidden,
            bidirectional=bool(recurrent_bidirectional), norm=norm,
        ) if recurrent else None

        # Decoder: inverse conv keyed to the matching down stage restores its exact
        # sites; skip-add then refine.
        self.up3 = _UpBlock(c3, c2, indice_key="sp3", norm=norm, algo=algo)
        self.dec2 = _SubMBlock(c2, c2, indice_key="subm2d", norm=norm, algo=algo)
        self.up2 = _UpBlock(c2, c1, indice_key="sp2", norm=norm, algo=algo)
        self.dec1 = _SubMBlock(c1, c1, indice_key="subm1d", norm=norm, algo=algo)
        self.up1 = _UpBlock(c1, c0, indice_key="sp1", norm=norm, algo=algo)
        self.dec0 = _SubMBlock(c0, c0, indice_key="subm0d", norm=norm, algo=algo)

        # Per-EVENT head: voxel context feature ⊕ the event's own raw features ->
        # MLP -> one logit per event. This is what makes the output a true
        # per-event stream (distinct logits for events sharing a voxel). Dropout
        # (input + hidden) is the model's main regularizer against the fast LOSO
        # overfitting; nn.Dropout(0.0) is a no-op so the default path is unchanged.
        self.feat_drop = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Linear(c0 + feat_dim, head_hidden),
            nn.LayerNorm(head_hidden) if norm == "ln" else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(head_hidden, num_classes),
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _add(a, b):
        """Add features of two sparse tensors sharing the same (inverse-restored) sites."""
        return a.replace_feature(a.features + b.features)

    def _voxelize(self, x, y, t_bin, batch_idx, feats, T, H, W):
        """Events -> unique ``(b, t_bin, y, x)`` voxels with mean features + a count.

        Returns ``(vox_coords_tyx (M,3), vox_batch (M,), vox_feats (M, F+1),
        inverse (N,))`` where ``inverse`` maps each event to its voxel row.
        """
        x = x.long(); y = y.long(); t_bin = t_bin.long(); b = batch_idx.long()
        key = ((b * T + t_bin) * H + y) * W + x          # unique per (b,t,y,x)
        uniq, inverse = torch.unique(key, sorted=True, return_inverse=True)
        M = uniq.numel()

        # scatter-mean per-event feats into voxels (+ log count channel).
        F = feats.shape[1]
        ones = torch.ones(inverse.shape[0], device=feats.device, dtype=feats.dtype)
        cnt = torch.zeros(M, device=feats.device, dtype=feats.dtype).scatter_add_(0, inverse, ones)
        idx_f = inverse.unsqueeze(1).expand(-1, F)
        summ = torch.zeros(M, F, device=feats.device, dtype=feats.dtype).scatter_add_(0, idx_f, feats)
        vox_feats = summ / cnt.clamp(min=1.0).unsqueeze(1)
        vox_feats = torch.cat([vox_feats, torch.log1p(cnt).unsqueeze(1)], dim=1)

        # decode the unique key back to (b, t, y, x).
        k = uniq
        vx = (k % W); k = torch.div(k, W, rounding_mode="floor")
        vy = (k % H); k = torch.div(k, H, rounding_mode="floor")
        vt = (k % T); k = torch.div(k, T, rounding_mode="floor")
        vb = k
        vox_coords = torch.stack([vt, vy, vx], dim=1).to(torch.int64)
        return vox_coords, vb.to(torch.int64), vox_feats, inverse

    def _extra_event_feats(self, x, y, times, t_bin, batch_idx, B, T, H, W):
        """Synthesize the optional geometry/motion-density per-event channels.

        Computed here (not in the dataset) so the features are derived from the
        *final*, augmentation-transformed events. Returns ``(N, n_extra)`` or
        ``None`` when no extra channels are enabled.
        """
        if self.n_extra == 0:
            return None
        x = x.long(); y = y.long(); b = batch_idx.long()
        parts = []
        if self.geom_features:
            xn = x.float() / max(W - 1, 1) * 2.0 - 1.0
            yn = y.float() / max(H - 1, 1) * 2.0 - 1.0
            parts.append(torch.stack([xn, yn], dim=1))
        if self.density_features:
            dev, dt = times.device, times.dtype
            r = self.density_radius
            k = 2 * r + 1
            if self.density_time_resolved:
                # Per-(b, t_bin, y, x): accumulate and spatially box-blur WITHIN each
                # time bin, so a trajectory pixel's static-source burst (a different
                # t_bin) no longer inflates the density/timing the hand passage reads.
                tb = t_bin.long()
                lin = ((b * T + tb) * H + y) * W + x
                n = B * T * H * W
                view_shape = (B * T, 1, H, W)
            else:
                lin = (b * H + y) * W + x                       # (N,) flat pixel id (all t)
                n = B * H * W
                view_shape = (B, 1, H, W)
            ones = torch.ones(times.shape[0], device=dev, dtype=dt)
            cnt = torch.zeros(n, device=dev, dtype=dt).scatter_add_(0, lin, ones)
            sumt = torch.zeros(n, device=dev, dtype=dt).scatter_add_(0, lin, times.to(dt))
            sumt2 = torch.zeros(n, device=dev, dtype=dt).scatter_add_(0, lin, (times * times).to(dt))

            def box_sum(g):                                    # neighborhood sum via avg-pool
                g = g.view(*view_shape)
                g = F.avg_pool2d(g, kernel_size=k, stride=1, padding=r) * float(k * k)
                return g.view(-1)

            bcnt = box_sum(cnt)
            bsumt = box_sum(sumt)
            bsumt2 = box_sum(sumt2)
            denom = bcnt.clamp(min=1.0)
            mean_t = bsumt / denom
            var_t = (bsumt2 / denom) - mean_t * mean_t
            std_t = var_t.clamp(min=0.0).sqrt()
            ev_dens = torch.log1p(bcnt)[lin]
            ev_mean = mean_t[lin]
            ev_std = std_t[lin]
            parts.append(torch.stack([ev_dens, ev_mean, ev_std], dim=1))
        return torch.cat(parts, dim=1).to(times.dtype if times.is_floating_point() else torch.float32)

    def forward(self, batch) -> torch.Tensor:
        feats = batch.feats
        N = feats.shape[0]
        if N == 0:
            return feats.new_zeros((0,) if self.num_classes == 1 else (0, self.num_classes))

        T, H, W = self.time_bins, int(batch.height), int(batch.width)
        x = batch.coords[:, 0]
        y = batch.coords[:, 1]
        times = batch.times
        t_bin = (times * T).floor().clamp_(0, T - 1)

        # Enrich the raw (polarity, time) per-event features with the optional
        # geometry/motion-density channels, then use the enriched set everywhere
        # (voxel mean + per-event head). This is the Tier-1 lever: the original
        # 2-channel input could not separate hand-motion from any other moving
        # edge; these channels give it spatial position + local density/timing.
        extra = self._extra_event_feats(x, y, times, t_bin, batch.batch_idx,
                                        batch.batch_size, T, H, W)
        if extra is not None:
            feats = torch.cat([feats, extra.to(feats.dtype)], dim=1)

        vox_coords, vox_batch, vox_feats, inverse = self._voxelize(
            x, y, t_bin, batch.batch_idx, feats, T, H, W,
        )

        sp = build_sparse_tensor_3d(vox_coords, vox_feats, vox_batch, T, H, W, batch.batch_size)

        s0 = self.stem(sp)
        e1 = self.enc1(self.down1(s0))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))
        if self.recurrent is not None:
            e3 = self.recurrent(e3, T)
        d2 = self.dec2(self._add(self.up3(e3), e2))
        d1 = self.dec1(self._add(self.up2(d2), e1))
        d0 = self.dec0(self._add(self.up1(d1), s0))

        # Re-align spconv's output rows to *our* voxel order (the order of `uniq`,
        # which `inverse` indexes into). Submanifold preserves the voxel set, so the
        # two index sets are a permutation of each other; match by coordinate key.
        of = d0.features                                   # (M, c0), spconv order
        oi = d0.indices.long()                             # columns [batch, t, y, x]
        out_key = ((oi[:, 0] * T + oi[:, 1]) * H + oi[:, 2]) * W + oi[:, 3]
        in_key = ((vox_batch * T + vox_coords[:, 0]) * H + vox_coords[:, 1]) * W + vox_coords[:, 2]
        if of.shape[0] != in_key.shape[0]:
            raise RuntimeError(
                f"submanifold voxel-count invariant violated: model returned "
                f"{of.shape[0]} voxels for {in_key.shape[0]} input voxels"
            )
        vox_out = torch.empty_like(of)
        vox_out[torch.argsort(in_key)] = of[torch.argsort(out_key)]

        # Per-event head: gather each event's voxel context feature, concat the
        # event's own raw features, MLP -> per-event logit. Dropout on the gathered
        # context regularizes the backbone features (the raw event feats pass through).
        if self.temporal_interp_head:
            # Interpolate context between the event's own t_bin voxel and the NEXT
            # t_bin voxel at the same (b, y, x), weighted by fractional time. Two
            # co-pixel events at different sub-bin moments then read DIFFERENT context
            # — the separation a single-bin gather (shared by the whole bin) can't make.
            # `in_key` is our sorted per-voxel key (== uniq), so searchsorted locates
            # the hi-neighbor voxel row when it is occupied.
            t_lo = t_bin                                   # floor, already in [0, T-1]
            frac = (times * T - t_lo).clamp(0.0, 1.0)
            t_hi = (t_lo + 1.0).clamp(max=float(T - 1)).long()
            bl = batch.batch_idx.long()
            key_hi = ((bl * T + t_hi) * H + y.long()) * W + x.long()
            pos = torch.searchsorted(in_key, key_hi).clamp(max=in_key.shape[0] - 1)
            valid = in_key[pos] == key_hi                  # hi voxel actually occupied?
            ctx_lo = vox_out[inverse]                      # (N, c0)
            ctx_hi = torch.where(valid.unsqueeze(1), vox_out[pos], ctx_lo)
            w = (frac * valid.to(frac.dtype)).unsqueeze(1)  # no hi neighbor -> pure lo
            ev_ctx = self.feat_drop((1.0 - w) * ctx_lo + w * ctx_hi)
        else:
            ev_ctx = self.feat_drop(vox_out[inverse])      # (N, c0)
        logits = self.head(torch.cat([ev_ctx, feats], dim=1))   # (N, num_classes)

        if self.num_classes == 1:
            return logits.squeeze(-1)                      # (N,)
        return logits                                      # (N, num_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(module: nn.Module) -> int:
    """Module-level convenience helper (mirrors ``model/event_unet.py``)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
