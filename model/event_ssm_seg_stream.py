"""Per-EVENT Koopman/SSM hand segmenter: an event stream in, a **filtered event
stream** out (one logit per event), with no dense-mask rasterization — so it cannot
exhibit the blocky/rectangular "frame artifact" a dense voxel->mask decoder produces.

This is the per-event evolution of :class:`model.event_ssm_seg.EventSSMSeg` (which was
dense ``voxel (B,K,H,W) -> mask (B,1,H,W)`` and only ~96k params). It keeps that
model's identity — a **structured-Koopman state-space model over time** plus a
**DMD-style static-vs-dynamic veto** — but rewires it onto the per-event
(``SparseEventBatch``) path used by :class:`model.event_sparse_seg_gc.EventSparseSegGC`,
and grows ~10x to ~960k params.

Why per-event removes the frame artifact
----------------------------------------
A dense voxel->mask model predicts on a coarse grid and upsamples, so its output is
quantized to grid cells (the rectangular blocks visible in the frame-based previews).
This model never rasterizes: every input event ``i`` keeps its own continuous
``(x, y)``, time and polarity, and the head emits ``logits[i]`` paired with
``batch.labels[i]``. Two events sharing a coarse cell still get **different** logits,
because the head reads each event's own continuous features (centroid-relative
coordinates, normal-flow motion, sub-cell offset, in-bin time) and a **bilinearly**
interpolated context (not a nearest-cell block). The visualization paints labels
directly on events — there is no grid to alias against.

The three reasoning components (and why each is here)
-----------------------------------------------------
1. **Koopman/SSM over time (temporal).** Events are scattered into a coarse
   ``(B, T, Hd, Wd)`` grid of ``T`` time-bin snapshots of learned per-event
   embeddings. A complex-DIAGONAL SSM (``_DiagSSM``, S5/S4D-style = a *structured
   Koopman operator*, SKOLR arXiv:2506.14113) evolves each spatial cell's ``T``-step
   sequence; the imaginary parts of its continuous eigenvalues are the learned
   **Koopman frequencies** (≈0 => static mode, nonzero => dynamic). Event-native SSMs
   (STREAM 2411.12603, Event-SSM 2404.18508, S5-for-events 2402.15584) are the SOTA
   constant-cost streaming primitive.
2. **DMD static veto (per-event).** The per-cell **dynamic energy** = temporal
   variance of the SSM output (the cheap DMD zero-mode / RPCA proxy, 1404.7592:
   static => constant output => ~0) is gathered per event and multiplicatively
   suppresses the logit where the surface is static/predictable — the direct fix for
   the static-window false positives.
3. **Dense global context (spatial).** Three independent measurement passes on this
   dataset found the dominant error (a "trajectory wake" of FPs) is a *global-spatial*
   deficit: locally a busy background event looks ~identical to a hand event, so only a
   whole-frame hand-blob signal can veto the look-alike background. A small dense 2D
   context net over the coarse agg map (3x3 + dilated convs that cross empty space, +
   a global-average-pool branch — the PSPNet/ASPP/SE recipe, ported in
   :class:`EventSparseSegGC`) supplies it, plus a per-window **presence gate** (the
   measured #1 FP-suppressor) and a train-only auxiliary occupancy head.

Per-event sharpness = local features (continuous), FP robustness = global context
(coarse). The two are factorized: the SSM is purely temporal (per cell), the context
net is purely spatial (per agg map) — and the head fuses them with the event's own
features so the decision boundary is continuous, never cell-quantized.

LOSO-safety: GroupNorm / LayerNorm only (BatchNorm running stats leak the held-out
subject). Cost is dominated by the fixed-size coarse grid, so inference stays
event-rate-proportional and streaming-friendly. No spconv (pure torch) — CPU-testable.

Shape contract (identical to ``EventSparseSeg`` / ``EventSparseSegGC``)
-----------------------------------------------------------------------
``forward(batch) -> logits``: ``(N,)`` when ``num_classes == 1`` (default), else
``(N, num_classes)``, with row ``i`` aligned to ``batch.coords[i]`` /
``batch.labels[i]``. Hooks for ``model_interface._event_segmentation_step`` (read-only):
``self._aux_logits (B,1,G,G)`` + ``aux_shape_weight``; ``self._presence_logit (B,)`` +
``presence_gate_weight`` / ``presence_min_fg``; ``self._event_embedding`` +
``null_loss_weight`` / ``null_margin``. Batch fields consumed: ``coords (N,2) (x,y)``,
``feats (N,F)``, ``times (N,) in [0,1]``, ``batch_idx (N,)``, ``batch_size``,
``height``, ``width`` (see ``data/sparse_event_collate.py``).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int, groups: int) -> nn.GroupNorm:
    """GroupNorm with a divisor-safe group count (LOSO-safe; no batch statistics)."""
    g = max(1, min(groups, c))
    while c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)


class _DiagSSM(nn.Module):
    """S5-style MIMO complex-DIAGONAL state-space model = a structured Koopman operator.

    ``h_k = Ā ⊙ h_{k-1} + B̄ x_k``  (state ``h ∈ C^N``);  ``y_k = Re(C h_k) + D x_k``.
    The continuous diagonal eigenvalues ``a = -exp(a_log_re) + i·a_im`` are stable
    (negative real part); ``Im(a) = a_im`` are the **Koopman frequencies** (≈0 => static
    mode, large => dynamic). Run as an explicit scan over the short ``T``-step sequence,
    so it is exact and parallel-over-cells. Real-valued parameters (the ``cfloat`` state
    is an intermediate only) -> DDP-safe.
    """

    def __init__(self, d_model: int, d_state: int = 128,
                 dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        self.d_model = int(d_model)
        self.N = int(d_state)
        self.a_log_re = nn.Parameter(torch.log(0.5 * torch.ones(self.N)))   # decay
        self.a_im = nn.Parameter(torch.linspace(0.0, math.pi, self.N))      # frequencies
        self.log_dt = nn.Parameter(
            torch.rand(self.N) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.B_re = nn.Parameter(torch.randn(self.N, d_model) / math.sqrt(d_model))
        self.B_im = nn.Parameter(torch.zeros(self.N, d_model))
        self.C_re = nn.Parameter(torch.randn(d_model, self.N) / math.sqrt(self.N))
        self.C_im = nn.Parameter(torch.randn(d_model, self.N) / math.sqrt(self.N))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """``x (M, L, d_model)`` real -> ``y (M, L, d_model)`` real, plus the frequencies."""
        M, L, Hc = x.shape
        dt = torch.exp(self.log_dt)                                  # (N,)
        a = -torch.exp(self.a_log_re) + 1j * self.a_im               # (N,) continuous
        Abar = torch.exp(a * dt)                                     # (N,) |.|<1
        Bbar = (self.B_re + 1j * self.B_im) * dt.unsqueeze(1)        # (N, Hc)
        C = self.C_re + 1j * self.C_im                               # (Hc, N)
        xc = x.to(torch.cfloat)
        h = torch.zeros(M, self.N, dtype=torch.cfloat, device=x.device)
        ys = []
        Bt = Bbar.transpose(0, 1)                                    # (Hc, N)
        Ct = C.transpose(0, 1)                                       # (N, Hc)
        for k in range(L):
            h = Abar.unsqueeze(0) * h + xc[:, k] @ Bt                # (M, N)
            ys.append((h @ Ct).real)                                # (M, Hc)
        y = torch.stack(ys, dim=1) + x * self.D                     # (M, L, Hc)
        return y, self.a_im


class _SSMBlock(nn.Module):
    """Residual SSM block: ``x + DiagSSM(LayerNorm(x))`` over the time axis."""

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = _DiagSSM(d_model, d_state)

    def forward(self, x):
        y, freqs = self.ssm(self.norm(x))
        return x + y, freqs


class _DenseContext(nn.Module):
    """Dense low-resolution global-context module (ported from ``EventSparseSegGC``).

    Operates on a small dense ``(B, c_in + 1, Hd, Wd)`` map (the coarse Koopman agg map
    plus a 1-channel occupancy mask). Its 3x3 + dilated convs propagate across empty
    pixels (the cross-empty-space mixing the per-cell SSM cannot do), and a parallel
    global-average-pool branch (over valid cells) adds whole-frame context. GroupNorm
    only. Output is a ``(B, c_ctx, Hd, Wd)`` context map. ``depth`` extra dilated convs
    widen the receptive field (and spend the 10x parameter budget).
    """

    def __init__(self, c_in: int, c_ctx: int, norm_groups: int = 8, depth: int = 1):
        super().__init__()
        g = max(1, min(norm_groups, c_ctx))
        self.conv1 = nn.Conv2d(c_in + 1, c_ctx, kernel_size=3, padding=1, bias=False)
        self.gn1 = _gn(c_ctx, norm_groups)
        # A stack of dilated convs widens the dense receptive field cheaply.
        self.dilated = nn.ModuleList()
        self.dilated_gn = nn.ModuleList()
        for i in range(max(1, depth)):
            d = 2 ** (i + 1)
            self.dilated.append(nn.Conv2d(c_ctx, c_ctx, kernel_size=3, padding=d,
                                          dilation=d, bias=False))
            self.dilated_gn.append(_gn(c_ctx, norm_groups))
        # Global-context branch: image-level descriptor broadcast back over the map.
        self.gpool_fc = nn.Conv2d(c_ctx, c_ctx, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(c_ctx * 2, c_ctx, kernel_size=3, padding=1, bias=False)
        self.gn3 = _gn(c_ctx, norm_groups)
        self.act = nn.ReLU(inplace=True)

    def forward(self, dense: torch.Tensor, occ: torch.Tensor) -> torch.Tensor:
        x = torch.cat([dense, occ], dim=1)
        x = self.act(self.gn1(self.conv1(x)))
        for conv, gn in zip(self.dilated, self.dilated_gn):
            x = self.act(gn(conv(x)))
        # whole-frame context: average over valid cells only, MLP, broadcast back.
        denom = occ.flatten(2).sum(dim=2).clamp(min=1.0)               # (B, 1)
        gvec = (x * occ).flatten(2).sum(dim=2) / denom                 # (B, c_ctx)
        gvec = self.gpool_fc(gvec[..., None, None])                    # (B, c_ctx, 1, 1)
        gvec = gvec.expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, gvec], dim=1)
        x = self.act(self.gn3(self.conv3(x)))
        return x


class EventSSMSegStream(nn.Module):
    """Per-event Koopman/SSM + DMD-veto + dense-global-context hand segmenter.

    Parameters
    ----------
    in_features
        Per-event input feature channels emitted by the dataset (2 = signed polarity,
        normalized time-in-window). Geometry / motion / sub-cell channels are
        synthesized inside ``forward`` from the post-augmentation events.
    time_bins
        Number of Koopman/SSM time-bin snapshots ``T`` over the window.
    down_factor
        Spatial downsample of the coarse grid the SSM + context run on (``Hd = ceil
        (H / down_factor)``). ``4`` (default) = 4x4-px cells (sharper context, more
        memory); ``8`` matches the ``EventSparseSegGC`` /8 bottleneck (cheaper).
    embed_dim
        Per-event embedding width ``d`` (= the SSM model dim and the agg-map width).
    d_state
        Complex-diagonal SSM modes ``N`` (Koopman eigenvalues).
    ssm_layers
        Number of stacked residual SSM blocks over the time axis.
    context_channels
        Width ``c_ctx`` of the dense global-context map.
    context_depth
        Number of dilated convs in the context net (widens the spatial field; spends
        the 10x parameter budget).
    head_hidden
        Hidden width of the per-event MLP head.
    coord_mode
        Geometry channels per event: ``relative`` (default, centroid-relative offset —
        LOSO-robust, no absolute work-location leak), ``absolute``, ``both``, ``none``.
    motion_features
        Append a per-event normal-flow MOTION descriptor (SAE plane fit: ``|∇t|``,
        planarity, log local count, + optional unit direction). The local "is this a
        coherently moving edge" cue. Default ``True``.
    motion_radius, motion_dir, motion_min_count
        Plane-fit neighborhood half-width, whether to add the 2 unit-direction
        channels, and the minimum neighborhood occupancy to trust the fit.
    dropout
        Dropout on the gathered per-event context and inside the head MLP.
    dmd_gate, dmd_gate_scale
        Enable the per-event DMD static veto (low dynamic energy -> suppress) and its
        scale on the logit.
    presence_gate, presence_gate_weight, presence_gate_scale, presence_min_fg
        Per-window presence gate (the null state): a single "is a moving hand present?"
        logit that log-sigmoid-gates every event logit (active train + eval, uses the
        predicted presence). ``_weight`` is read by ``model_interface`` for its BCE;
        ``_min_fg`` is the minimum foreground events to call a window present.
    aux_shape_head, aux_grid, aux_shape_weight
        Train-only coarse ``(B,1,G,G)`` occupancy head (SA-SSD style; zero inference
        cost) supervised by the teacher mask in ``model_interface``.
    null_loss_weight, null_margin
        Optional background-prototype null loss on the per-event embedding (read by
        ``model_interface``); default off.
    gn_groups
        GroupNorm group count for the dense path.
    """

    def __init__(
        self,
        in_features: int = 2,
        time_bins: int = 6,
        down_factor: int = 4,
        embed_dim: int = 128,
        d_state: int = 128,
        ssm_layers: int = 2,
        context_channels: int = 128,
        context_depth: int = 2,
        head_hidden: int = 128,
        num_classes: int = 1,
        coord_mode: str = "relative",
        motion_features: bool = True,
        motion_radius: int = 3,
        motion_dir: bool = False,
        motion_min_count: int = 6,
        density_features: bool = False,
        density_radius: int = 2,
        dropout: float = 0.2,
        dmd_gate: bool = True,
        dmd_gate_scale: float = 1.0,
        presence_gate: bool = True,
        presence_gate_weight: float = 0.3,
        presence_gate_scale: float = 1.0,
        presence_min_fg: int = 0,
        aux_shape_head: bool = True,
        aux_grid: int = 32,
        aux_shape_weight: float = 0.2,
        null_loss_weight: float = 0.0,
        null_margin: float = 1.0,
        gn_groups: int = 8,
        recency_context: bool = True,
        recency_init: float = 1.0,
        boundary_loss_weight: float = 1.0,
        boundary_loss_band: int = 0,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.time_bins = int(time_bins)
        self.down_factor = max(1, int(down_factor))
        self.embed_dim = int(embed_dim)
        self.d_state = int(d_state)
        self.num_classes = int(num_classes)
        self.coord_mode = str(coord_mode).strip().lower()
        if self.coord_mode not in ("relative", "absolute", "both", "none"):
            raise ValueError(
                f"coord_mode must be relative|absolute|both|none, got {coord_mode!r}")
        self.motion_features = bool(motion_features)
        self.motion_radius = max(1, int(motion_radius))
        self.motion_dir = bool(motion_dir)
        self.motion_min_count = max(3, int(motion_min_count))
        self.density_features = bool(density_features)
        self.density_radius = max(1, int(density_radius))
        self.dmd_gate = bool(dmd_gate)
        self.dmd_gate_scale = float(dmd_gate_scale)
        self.presence_gate = bool(presence_gate)
        self.presence_gate_weight = float(presence_gate_weight)
        self.presence_gate_scale = float(presence_gate_scale)
        self.presence_min_fg = int(presence_min_fg)
        self.aux_shape_head = bool(aux_shape_head)
        self.aux_grid = int(aux_grid)
        self.aux_shape_weight = float(aux_shape_weight)
        self.null_loss_weight = float(null_loss_weight)
        self.null_margin = float(null_margin)
        # Recency-weighted temporal readout: a moving edge's CURRENT position lives in
        # the latest time-bin, not the 36 ms mean — averaging smears a fast edge across
        # its whole swept band. A learned softmax over T (init biased to recent) lets
        # the localizer track "where the edge is now" while still able to flatten back
        # toward the mean if that helps. The SSM scan already integrates history into
        # each bin, so the late bins carry the past with a learned decay.
        self.recency_context = bool(recency_context)
        # Boundary-emphasis loss knobs (read by model_interface; default off). Up-weight
        # events within ``boundary_loss_band`` px of the (augmentation-aligned) teacher
        # boundary by ``boundary_loss_weight``. ONLY coherent on de-noised labels (the
        # loss docstring warns up-weighting noisy boundary events amplifies label noise),
        # so pair with boundary_ignore_px / nearest-frame labels.
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.boundary_loss_band = int(boundary_loss_band)
        self.context_channels = int(context_channels)
        # Hooks read by model_interface._event_segmentation_step (reset every forward).
        self._aux_logits = None
        self._presence_logit = None
        self._event_embedding = None
        self._dyn_energy = None        # diagnostic: per-cell DMD dynamic energy
        self._freqs = None             # diagnostic: learned Koopman frequencies

        d = self.embed_dim
        # ---- per-event feature widths -------------------------------------------------
        n_geom = {"relative": 2, "absolute": 2, "both": 4, "none": 0}[self.coord_mode]
        n_motion = (3 + (2 if self.motion_dir else 0)) if self.motion_features else 0
        n_density = 3 if self.density_features else 0
        # +2 sub-cell (x,y) offset, +1 in-bin time offset: continuous per-event channels
        # that make the head sharp WITHIN a coarse cell (the no-frame-artifact lever).
        n_subcell = 3
        self.n_geom, self.n_motion, self.n_density = n_geom, n_motion, n_density
        feat_dim = self.in_features + n_geom + n_motion + n_density + n_subcell
        self._feat_dim = feat_dim

        # ---- trainable submodules -----------------------------------------------------
        self.event_mlp = nn.Sequential(
            nn.Linear(feat_dim, d), nn.LayerNorm(d), nn.ReLU(inplace=True),
            nn.Linear(d, d),
        )
        self.ssm_blocks = nn.ModuleList(_SSMBlock(d, self.d_state)
                                        for _ in range(max(1, int(ssm_layers))))
        self.ssm_norm = nn.LayerNorm(d)
        # Learned per-time-bin readout weights (softmax'd in forward). Init as a ramp
        # toward the latest bin (recency_init>0) so the context starts motion-aware; the
        # net can flatten it to a uniform mean if that generalizes better. Only created
        # when active so an ablation (recency_context=False) leaves no unused DDP param.
        if self.recency_context:
            self.recency_logits = nn.Parameter(
                torch.linspace(0.0, float(recency_init), self.time_bins))
        else:
            self.register_parameter("recency_logits", None)
        self.context = _DenseContext(d, self.context_channels, gn_groups,
                                     depth=max(1, int(context_depth)))
        # DMD static-veto gate: maps per-event dynamic energy -> a logit shift.
        self.dyn_a = nn.Parameter(torch.tensor(4.0))
        self.dyn_b = nn.Parameter(torch.tensor(-2.0))

        # Per-event head: event embedding ⊕ time-specific SSM readout ⊕ bilinear global
        # context ⊕ dynamic energy ⊕ the event's own raw features -> one logit per event.
        head_in = d + d + self.context_channels + 1 + feat_dim
        self.feat_drop = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.LayerNorm(head_hidden),
            nn.ReLU(inplace=True), nn.Dropout(float(dropout)),
            nn.Linear(head_hidden, num_classes),
        )
        # Per-window presence head (the null state), read from the masked global-average
        # of the dense context map.
        self.presence_head = (nn.Sequential(
            nn.Linear(self.context_channels, self.context_channels),
            nn.ReLU(inplace=True), nn.Linear(self.context_channels, 1))
            if self.presence_gate else None)
        # Train-only coarse occupancy head off the dense context.
        self.aux_shape = (nn.Conv2d(self.context_channels, 1, kernel_size=1)
                          if self.aux_shape_head else None)

    # ------------------------------------------------------------------ feature helpers

    def _geom_feats(self, x, y, batch_idx, B, H, W):
        """Geometry channels per ``coord_mode`` (centroid-relative is LOSO-robust)."""
        if self.coord_mode == "none":
            return None
        xf = x.float(); yf = y.float()
        parts = []
        if self.coord_mode in ("absolute", "both"):
            xn = xf / max(W - 1, 1) * 2.0 - 1.0
            yn = yf / max(H - 1, 1) * 2.0 - 1.0
            parts.append(torch.stack([xn, yn], dim=1))
        if self.coord_mode in ("relative", "both"):
            b = batch_idx.long()
            cnt = torch.zeros(B, device=xf.device, dtype=xf.dtype).scatter_add_(
                0, b, torch.ones_like(xf))
            cx = torch.zeros(B, device=xf.device, dtype=xf.dtype).scatter_add_(0, b, xf)
            cy = torch.zeros(B, device=xf.device, dtype=xf.dtype).scatter_add_(0, b, yf)
            denom = cnt.clamp(min=1.0)
            cx = cx / denom; cy = cy / denom
            scale = 0.5 * float((H ** 2 + W ** 2) ** 0.5)
            dx = (xf - cx[b]) / scale
            dy = (yf - cy[b]) / scale
            parts.append(torch.stack([dx, dy], dim=1))
        return torch.cat(parts, dim=1)

    def _density_feats(self, x, y, times, t_bin, batch_idx, B, T, H, W):
        """3 neighborhood density/timing channels (time-resolved per ``(b,t_bin,y,x)``)."""
        if not self.density_features:
            return None
        x = x.long(); y = y.long(); b = batch_idx.long(); tb = t_bin.long()
        dev, dt = times.device, times.dtype
        r = self.density_radius
        k = 2 * r + 1
        lin = ((b * T + tb) * H + y) * W + x
        n = B * T * H * W
        view_shape = (B * T, 1, H, W)
        ones = torch.ones(times.shape[0], device=dev, dtype=dt)
        cnt = torch.zeros(n, device=dev, dtype=dt).scatter_add_(0, lin, ones)
        sumt = torch.zeros(n, device=dev, dtype=dt).scatter_add_(0, lin, times.to(dt))
        sumt2 = torch.zeros(n, device=dev, dtype=dt).scatter_add_(0, lin, (times * times).to(dt))

        def box_sum(g):
            g = g.view(*view_shape)
            g = F.avg_pool2d(g, kernel_size=k, stride=1, padding=r) * float(k * k)
            return g.view(-1)

        bcnt = box_sum(cnt); bsumt = box_sum(sumt); bsumt2 = box_sum(sumt2)
        denom = bcnt.clamp(min=1.0)
        mean_t = bsumt / denom
        std_t = ((bsumt2 / denom) - mean_t * mean_t).clamp(min=0.0).sqrt()
        return torch.stack([torch.log1p(bcnt)[lin], mean_t[lin], std_t[lin]], dim=1)

    def _motion_feats(self, x, y, times, batch_idx, B, H, W):
        """Per-event normal-flow MOTION descriptor (SAE plane fit). See ``EventSparseSegGC``.

        Channels: ``|∇t|`` (inverse normal-flow speed), ``planarity`` (R² of the local
        time-surface plane fit), ``log1p(count)``; optional unit flow direction. Computed
        whole-frame via fixed-kernel convs, gathered per event, under ``no_grad`` (a
        deterministic input feature). Separates a coherently moving edge from static
        scene / sensor noise.
        """
        if not self.motion_features:
            return None
        with torch.no_grad():
            dev = times.device
            ft = torch.float32
            b = batch_idx.long(); xi = x.long(); yi = y.long()
            r = self.motion_radius
            k = 2 * r + 1
            n = B * H * W
            lin = (b * H + yi) * W + xi
            t = times.to(ft)
            sae = torch.zeros(n, device=dev, dtype=ft).scatter_reduce_(
                0, lin, t, reduce="amax", include_self=True)
            cnt = torch.zeros(n, device=dev, dtype=ft).index_add_(
                0, lin, torch.ones_like(t))
            sae = sae.view(B, 1, H, W)
            m = (cnt.view(B, 1, H, W) > 0).to(ft)
            mt = sae * m
            mt2 = mt * sae
            off = (torch.arange(k, device=dev, dtype=ft) - r)
            dxk = off.view(1, 1, 1, k).expand(1, 1, k, k).contiguous()
            dyk = off.view(1, 1, k, 1).expand(1, 1, k, k).contiguous()
            onek = torch.ones(1, 1, k, k, device=dev, dtype=ft)
            dx2k = dxk * dxk; dy2k = dyk * dyk; dxyk = dxk * dyk

            def cv(src, ker):
                return F.conv2d(src, ker, padding=r).view(-1)[lin]

            S1 = cv(m, onek); Sx = cv(m, dxk); Sy = cv(m, dyk)
            Sxx = cv(m, dx2k); Syy = cv(m, dy2k); Sxy = cv(m, dxyk)
            St = cv(mt, onek); Stx = cv(mt, dxk); Sty = cv(mt, dyk); Stt = cv(mt2, onek)
            cnt_e = cv(m * cnt.view(B, 1, H, W), onek)

            lam = 1e-3
            Nn = t.shape[0]
            M3 = torch.zeros(Nn, 3, 3, device=dev, dtype=ft)
            M3[:, 0, 0] = Sxx + lam; M3[:, 0, 1] = Sxy; M3[:, 0, 2] = Sx
            M3[:, 1, 0] = Sxy; M3[:, 1, 1] = Syy + lam; M3[:, 1, 2] = Sy
            M3[:, 2, 0] = Sx;  M3[:, 2, 1] = Sy;  M3[:, 2, 2] = S1 + lam
            rhs = torch.stack([Stx, Sty, St], dim=1).unsqueeze(-1)
            abc = torch.linalg.solve(M3, rhs).squeeze(-1)
            a, bb, cc = abc[:, 0], abc[:, 1], abc[:, 2]
            grad_mag = torch.sqrt(a * a + bb * bb + 1e-12)
            resid = (Stt - a * Stx - bb * Sty - cc * St).clamp_min(0.0)
            mean_t = St / S1.clamp_min(1.0)
            total_var = (Stt - St * mean_t).clamp_min(0.0)
            r2 = 1.0 - resid / total_var.clamp_min(1e-6)
            planarity = torch.where(total_var > 1e-6, r2.clamp(0.0, 1.0),
                                    torch.ones_like(r2))
            valid = (S1 >= float(self.motion_min_count)).to(ft)
            grad_mag = grad_mag * valid
            planarity = planarity * valid
            chans = [grad_mag, planarity, torch.log1p(cnt_e)]
            if self.motion_dir:
                inv = 1.0 / grad_mag.clamp_min(1e-6)
                chans += [a * inv * valid, bb * inv * valid]
            out = torch.stack(chans, dim=1)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _bilinear_gather(flat, x, y, batch_idx, Hd, Wd, H, W):
        """Bilinearly sample a dense ``(B*Hd*Wd, C)`` map at each event's continuous
        full-res ``(x, y)`` — smooth, per-event-precise (no nearest-cell block sharing,
        the source of the rectangular artifact)."""
        b = batch_idx.long()
        fy = (y.float() + 0.5) * Hd / H - 0.5
        fx = (x.float() + 0.5) * Wd / W - 0.5
        y0 = torch.floor(fy); x0 = torch.floor(fx)
        wy = (fy - y0); wx = (fx - x0)
        y0 = y0.long(); x0 = x0.long(); y1 = y0 + 1; x1 = x0 + 1
        y0c = y0.clamp(0, Hd - 1); y1c = y1.clamp(0, Hd - 1)
        x0c = x0.clamp(0, Wd - 1); x1c = x1.clamp(0, Wd - 1)

        def at(yi, xi):
            return flat[(b * Hd + yi) * Wd + xi]
        c00 = at(y0c, x0c); c01 = at(y0c, x1c)
        c10 = at(y1c, x0c); c11 = at(y1c, x1c)
        wy = wy.unsqueeze(1); wx = wx.unsqueeze(1)
        top = c00 * (1 - wx) + c01 * wx
        bot = c10 * (1 - wx) + c11 * wx
        return top * (1 - wy) + bot * wy

    # ------------------------------------------------------------------ forward

    def forward(self, batch) -> torch.Tensor:
        self._aux_logits = None
        self._presence_logit = None
        self._event_embedding = None
        feats = batch.feats
        N = feats.shape[0]
        if N == 0:
            return feats.new_zeros((0,) if self.num_classes == 1 else (0, self.num_classes))

        T, H, W = self.time_bins, int(batch.height), int(batch.width)
        B = batch.batch_size
        df = self.down_factor
        Hd = (H + df - 1) // df
        Wd = (W + df - 1) // df
        x = batch.coords[:, 0]
        y = batch.coords[:, 1]
        times = batch.times
        t_bin = (times * T).floor().clamp_(0, T - 1).long()

        # ---- per-event feature enrichment (post-augmentation) -------------------------
        parts = [feats]
        gfeat = self._geom_feats(x, y, batch.batch_idx, B, H, W)
        if gfeat is not None:
            parts.append(gfeat.to(feats.dtype))
        dfeat = self._density_feats(x, y, times, t_bin, batch.batch_idx, B, T, H, W)
        if dfeat is not None:
            parts.append(dfeat.to(feats.dtype))
        mfeat = self._motion_feats(x, y, times, batch.batch_idx, B, H, W)
        if mfeat is not None:
            parts.append(mfeat.to(feats.dtype))
        # Sub-cell (x,y) offset within the coarse cell + in-bin time offset: continuous
        # per-event channels -> the head can separate co-cell events (no frame artifact).
        gy = (y.long() * Hd // H).clamp(0, Hd - 1)
        gx = (x.long() * Wd // W).clamp(0, Wd - 1)
        sub_x = (x.float() * Wd / max(W, 1) - gx.float()).clamp(0.0, 1.0)
        sub_y = (y.float() * Hd / max(H, 1) - gy.float()).clamp(0.0, 1.0)
        sub_t = (times * T - t_bin.float())
        parts.append(torch.stack([sub_x, sub_y, sub_t], dim=1).to(feats.dtype))
        feats = torch.cat(parts, dim=1)

        ev_embed = self.event_mlp(feats)                              # (N, d)
        d = ev_embed.shape[1]

        # ---- scatter to coarse Koopman snapshot grid (b, t_bin, gy, gx) ---------------
        cell = (batch.batch_idx.long() * Hd + gy) * Wd + gx           # (N,) in [0, B*Hd*Wd)
        n_cells = B * Hd * Wd
        flat_ct = cell * T + t_bin                                    # (N,) (cell, t) slot
        n_ct = n_cells * T
        snap_sum = ev_embed.new_zeros(n_ct, d).index_add_(0, flat_ct, ev_embed)
        snap_cnt = ev_embed.new_zeros(n_ct, 1).index_add_(
            0, flat_ct, ev_embed.new_ones(N, 1))
        snap = (snap_sum / snap_cnt.clamp(min=1.0)).view(n_cells, T, d)  # (n_cells, T, d)

        # ---- structured-Koopman SSM over time (per cell) ------------------------------
        seq = snap
        freqs = None
        for blk in self.ssm_blocks:
            seq, freqs = blk(seq)
        seq = self.ssm_norm(seq)                                     # (n_cells, T, d)
        if self.recency_context:
            # Recency-weighted readout: track the moving edge's CURRENT position instead
            # of smearing it across the window mean. Learned softmax over T (init recent).
            w = torch.softmax(self.recency_logits, dim=0)            # (T,)
            agg = (seq * w.view(1, T, 1)).sum(dim=1)                 # (n_cells, d)
        else:
            agg = seq.mean(dim=1)                                    # (n_cells, d) temporal readout
        # DMD dynamic energy: temporal variance of the SSM output (static -> ~0).
        dyn = seq.var(dim=1, unbiased=False).mean(dim=-1, keepdim=True)  # (n_cells, 1)
        self._dyn_energy, self._freqs = dyn, freqs

        # ---- dense global context (spatial) -------------------------------------------
        agg_map = agg.view(B, Hd, Wd, d).permute(0, 3, 1, 2).contiguous()    # (B,d,Hd,Wd)
        cell_cnt = snap_cnt.view(n_cells, T).sum(dim=1, keepdim=True)        # (n_cells,1)
        occ = (cell_cnt > 0).to(agg.dtype).view(B, Hd, Wd, 1).permute(0, 3, 1, 2).contiguous()
        ctx_map = self.context(agg_map, occ)                                 # (B,c_ctx,Hd,Wd)
        ctx_flat = ctx_map.permute(0, 2, 3, 1).reshape(n_cells, self.context_channels)

        # Per-window presence logit (masked global average of the context map).
        if self.presence_head is not None:
            denom = occ.flatten(2).sum(dim=2).clamp(min=1.0)                 # (B,1)
            gdesc = (ctx_map * occ).flatten(2).sum(dim=2) / denom            # (B,c_ctx)
            self._presence_logit = self.presence_head(gdesc).squeeze(-1)     # (B,)

        # Train-only coarse occupancy map (supervised in model_interface).
        if self.aux_shape is not None and self.training:
            G = self.aux_grid
            pooled = F.adaptive_avg_pool2d(ctx_map, (G, G))
            self._aux_logits = self.aux_shape(pooled)                        # (B,1,G,G)

        # ---- per-event head -----------------------------------------------------------
        # Time-specific SSM readout (the event's own t_bin state) -> per-event temporal
        # sharpness; smooth bilinear context + dynamic energy -> spatial sharpness.
        ssm_t_ev = seq.reshape(n_ct, d)[flat_ct]                             # (N,d)
        agg_ev = self.feat_drop(
            self._bilinear_gather(agg, x, y, batch.batch_idx, Hd, Wd, H, W))  # (N,d) -> via ctx
        ctx_ev = self.feat_drop(
            self._bilinear_gather(ctx_flat, x, y, batch.batch_idx, Hd, Wd, H, W))  # (N,c_ctx)
        dyn_ev = self._bilinear_gather(dyn, x, y, batch.batch_idx, Hd, Wd, H, W)   # (N,1)
        # agg_ev folds the (smoothed) temporal Koopman readout straight into the head in
        # addition to the time-specific ssm_t_ev; together they cover both the cell's
        # aggregate dynamics and the event's exact moment.
        head_in = torch.cat([ev_embed, ssm_t_ev + agg_ev, ctx_ev, dyn_ev, feats], dim=1)
        emb = self.head[:-1](head_in)
        logits = self.head[-1](emb)
        if self.training and self.null_loss_weight > 0.0:
            self._event_embedding = emb

        # DMD static veto (per-event): low dynamic energy (static/predictable) -> suppress.
        if self.dmd_gate:
            logits = logits + self.dmd_gate_scale * F.logsigmoid(self.dyn_a * dyn_ev + self.dyn_b)

        # Per-window presence gate (null state): presence->1 => +0; presence->0 => big
        # negative => all events in the window suppressed. Train + eval (predicted presence).
        if self._presence_logit is not None:
            gate = self.presence_gate_scale * F.logsigmoid(self._presence_logit)
            logits = logits + gate[batch.batch_idx.long()].view(logits.shape)

        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(module: nn.Module) -> int:
    """Module-level convenience helper (mirrors ``model/event_unet.py``)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
