"""EventStreamSegS7 — a TRUE event-by-event selective-SSM per-event hand segmenter.

This is the model designed in ``docs/deep_research_event_native_backbones_RECOVERED.md``
(Brief A). It is the per-event evolution of :class:`model.event_ssm_seg_stream.
EventSSMSegStream` — but where that baseline scatters events into a coarse
``(cells, T, d)`` grid and scans a **complex-diagonal SSM over the T time-bin
snapshots per cell**, this model scans the **raw time-ordered event stream one event
at a time** with a *selective* (input-dependent, S7/Mamba-style) diagonal SSM. That
removes the per-voxel-cell time quantization the baseline bakes in before its SSM ever
runs: every event keeps its own ``(x, y, t, p)`` and gets its own evolving state.

Why this should beat the ~0.786 baseline
-----------------------------------------
The baseline collapses all events sharing a coarse cell into ``T=6`` averaged
snapshots *before* the SSM, quantizing sub-cell time and forcing co-cell events to
share one temporal sequence. A true per-event scan gives every event its own state
with STREAM's exact Δt spacing (2411.12603) and S7 input-dependent selectivity
(2410.03464), while a **dual scan order** restores the 2D spatial structure a single
1D scan destroys, and the baseline's measured FP-suppressors (dense global context +
presence gate + DMD static veto) are ported unchanged.

The four ideas (grounded in the recovered research)
---------------------------------------------------
1. **Selective diagonal SSM (S7-style), MIMO** (:class:`_SelectiveSSM`). Stable
   reparameterized complex-diagonal ``A = -softplus(a_log) + i·a_im`` (S7), a
   **STREAM Δ-discretization** ``Δ_k = softplus(δ(x_k))·Δt_k`` that is input-dependent
   *and* physically time-aware, and an input gate on the drive — the "selective"
   mechanism. One shared ``C^N`` complex state per sequence (MIMO, like the baseline's
   ``_DiagSSM``) keeps it cheap.
2. **Segmented parallel scan** (:func:`_segmented_scan`). A Hillis-Steele associative
   scan over the flat event axis, masked by each event's within-sample position so the
   recurrence **resets at ``batch_idx`` boundaries** and never leaks across samples.
   Fully vectorized (no Python per-event loop → avoids the measured "Mamba-per-cell is
   9x slower" trap), numerically stable (|Ā|≤1, no divisions), and CPU-testable.
3. **Dual scan order.** A causal **TIME** scan (native time order = streaming-capable)
   plus a bidirectional **z-order / Morton** scan (spatially-adjacent events become
   sequence-adjacent) — the only way 2D locality re-enters a 1D scan (SMamba IPL-Scan
   analog). Both states scatter back to the original event rows.
4. **Ported FP-suppression + per-event head.** Fused per-event states are pooled into a
   coarse ``(B, d, Hd, Wd)`` map for the baseline's :class:`_DenseContext` +
   per-window presence gate + DMD static veto (the three measured FP-suppressors);
   the head reads ``[time-state, z-state, bilinear context, dynamic energy, raw
   feats]`` → one logit per event, row-aligned to ``batch.labels``.

Shape contract (identical to ``EventSSMSegStream``)
---------------------------------------------------
``forward(batch) -> (N,)`` logits (``num_classes == 1``), row ``i`` aligned to
``batch.coords[i]`` / ``batch.labels[i]``. Hooks read (read-only) by
``model_interface._event_segmentation_step``: ``_aux_logits (B,1,G,G)`` +
``aux_shape_weight``; ``_presence_logit (B,)`` + ``presence_gate_weight`` /
``presence_min_fg``; ``_event_embedding`` + ``null_loss_weight`` / ``null_margin``;
and the new ``_scan_loss`` scalar + ``scan_weight`` (dual-scan consistency; imitates
the existing ``_jepa_loss`` hook). ``boundary_loss_weight`` / ``boundary_loss_band``
are honoured by the trainer exactly as for the baseline. Diagnostics for the preview
tooling: ``_dyn_energy``, ``_freqs``.

DDP / LOSO safety: all parameters are real (the ``cfloat`` scan state is an
intermediate only); GroupNorm / LayerNorm only (no BatchNorm running stats). Pure
torch (no spconv). Event-pooling downsampling from the brief is left as a documented
future optimization (``event_pool`` knob, default off) so the first build stays
row-alignment-correct; the 4 blocks/scan run at full event resolution.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Reuse the baseline's dense global-context net and its GroupNorm helper verbatim.
from model.event_ssm_seg_stream import _gn, _DenseContext


# ======================================================================================
# Segmented associative scan (the core primitive)
# ======================================================================================
def _segmented_scan(Abar: torch.Tensor, u: torch.Tensor,
                    pos_in_seg: torch.Tensor, max_len: int) -> torch.Tensor:
    """Inclusive first-order linear scan ``h_k = Ā_k ⊙ h_{k-1} + u_k`` with per-segment
    reset, via a Hillis-Steele parallel scan.

    ``Abar (M, N)`` complex, ``u (M, N)`` complex are laid out in scan order (segments
    contiguous). ``pos_in_seg (M,)`` long is each element's 0-based position within its
    contiguous segment; the mask ``pos_in_seg >= d`` guarantees a doubling step never
    combines across a segment boundary (there are ``>= d`` predecessors in the same
    segment), so the recurrence resets at every boundary. Returns ``h (M, N)`` complex.

    The monoid is ``combine(left, right) = (right.a·left.a, right.a·left.b + right.b)``
    (composition of two steps of ``h_k = a_k h_{k-1} + b_k``). |Ā|≤1 so products never
    blow up and there are no divisions — numerically safe. O(M·log L) work, O(log L)
    depth. Verified against a sequential reference in the smoke test.
    """
    a = Abar
    b = u
    d = 1
    while d < max_len:
        zeros_a = a.new_zeros(d, a.shape[1])
        zeros_b = b.new_zeros(d, b.shape[1])
        a_prev = torch.cat([zeros_a, a[:-d]], dim=0)          # a[i-d]
        b_prev = torch.cat([zeros_b, b[:-d]], dim=0)          # b[i-d]
        mask = (pos_in_seg >= d).unsqueeze(1)                 # (M,1) bool
        b = torch.where(mask, a * b_prev + b, b)
        a = torch.where(mask, a * a_prev, a)
        d *= 2
    return b


# ======================================================================================
# Selective diagonal complex SSM (S7 / Mamba-style), MIMO
# ======================================================================================
class _SelectiveSSM(nn.Module):
    """Input-dependent (selective) complex-diagonal MIMO state-space model over a
    (segmented) event sequence.

    ``A = -softplus(a_log) + i·a_im`` (S7 stable reparam; ``Im(A)`` are the Koopman
    frequencies). Per event: ``Δ_k = softplus(δ(x_k))·Δt_k`` (STREAM coordinate/time
    difference discretization, input-dependent), ``Ā_k = exp(A·Δ_k)``, drive
    ``u_k = Δ_k·g(x_k)·(B x_k)`` with an input gate ``g`` (B-selectivity). Output
    ``y_k = Re(C h_k) + D x_k``. One shared ``C^N`` state (MIMO) — cheap, matches the
    baseline ``_DiagSSM`` cost. ``bidirectional`` runs a second scan on the
    within-segment-reversed sequence and sums (for the non-causal spatial order).
    """

    def __init__(self, d_model: int, d_state: int = 64, bidirectional: bool = False):
        super().__init__()
        self.d_model = int(d_model)
        self.N = int(d_state)
        self.bidirectional = bool(bidirectional)
        # Stable diagonal continuous eigenvalues (real part < 0 via -softplus).
        self.a_log = nn.Parameter(torch.log(torch.expm1(0.5 * torch.ones(self.N))))
        self.a_im = nn.Parameter(torch.linspace(0.0, math.pi, self.N))
        # MIMO input/output maps (real params; complex assembled in forward).
        self.B_re = nn.Parameter(torch.randn(self.N, d_model) / math.sqrt(d_model))
        self.B_im = nn.Parameter(torch.zeros(self.N, d_model))
        self.C_re = nn.Parameter(torch.randn(d_model, self.N) / math.sqrt(self.N))
        self.C_im = nn.Parameter(torch.randn(d_model, self.N) / math.sqrt(self.N))
        self.D = nn.Parameter(torch.ones(d_model))
        # Selectivity: input-dependent Δ (scalar/event) and input gate (per mode).
        self.delta_proj = nn.Linear(d_model, 1)
        self.gate_proj = nn.Linear(d_model, self.N)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.constant_(self.delta_proj.bias, 0.0)   # softplus(0)=~0.69 initial scale

    def _scan_once(self, x, dt, pos_in_seg, max_len, a_im, a_re):
        """One directional selective scan over the ordered sequence ``x (M,d)``."""
        A = torch.complex(a_re, a_im)                                   # (N,) continuous
        delta = F.softplus(self.delta_proj(x).squeeze(-1)) * dt         # (M,)
        Abar = torch.exp(A.unsqueeze(0) * delta.unsqueeze(1).to(A.dtype))  # (M,N)
        g = F.softplus(self.gate_proj(x))                              # (M,N) input gate
        xB = x @ self.B_re.t() + 1j * (x @ self.B_im.t())              # (M,N) complex
        u = (delta.unsqueeze(1) * g).to(xB.dtype) * xB                 # (M,N) drive
        h = _segmented_scan(Abar, u, pos_in_seg, max_len)             # (M,N) complex
        C = torch.complex(self.C_re, self.C_im)                        # (d,N)
        y = (h @ C.t()).real + x * self.D                              # (M,d)
        return y

    def forward(self, x, dt, pos_in_seg, seg_start, seg_len, max_len):
        a_re = -F.softplus(self.a_log)
        y = self._scan_once(x, dt, pos_in_seg, max_len, self.a_im, a_re)
        if self.bidirectional:
            # Reverse within each contiguous segment (an involution): position j -> len-1-j.
            rev = seg_start + (seg_len - 1 - pos_in_seg)
            y_b = self._scan_once(x[rev], dt[rev], pos_in_seg, max_len, self.a_im, a_re)
            y = y + y_b[rev]
        return y, self.a_im


class _SelectiveSSMBlock(nn.Module):
    """Residual selective-SSM block: ``x + out_proj(SSM(norm(x)) ⊙ SiLU(gate(x)))``."""

    def __init__(self, d_model: int, d_state: int, bidirectional: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = _SelectiveSSM(d_model, d_state, bidirectional=bidirectional)
        self.gate = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, dt, pos_in_seg, seg_start, seg_len, max_len):
        h = self.norm(x)
        y, freqs = self.ssm(h, dt, pos_in_seg, seg_start, seg_len, max_len)
        y = self.out_proj(y * F.silu(self.gate(h)))
        return x + y, freqs


# ======================================================================================
# z-order (Morton) helper
# ======================================================================================
def _morton2d(gx: torch.Tensor, gy: torch.Tensor, bits: int) -> torch.Tensor:
    """Interleave the low ``bits`` of ``gx``/``gy`` into a Morton (z-order) code so that
    spatially-adjacent cells receive nearby codes. Pure-torch bit twiddling (long)."""
    gx = gx.long().clamp_min(0)
    gy = gy.long().clamp_min(0)
    code = torch.zeros_like(gx)
    for i in range(bits):
        code |= ((gx >> i) & 1) << (2 * i)
        code |= ((gy >> i) & 1) << (2 * i + 1)
    return code


def _seg_layout(sorted_bidx: torch.Tensor, B: int):
    """From a scan-order sample index ``(M,)`` (segments contiguous) build
    ``pos_in_seg``, ``seg_start`` (first index of each element's segment) and
    ``seg_len`` (segment length per element), plus ``max_len``."""
    M = sorted_bidx.shape[0]
    device = sorted_bidx.device
    ar = torch.arange(M, device=device)
    if M == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, z, z, 1
    is_start = torch.ones(M, dtype=torch.bool, device=device)
    is_start[1:] = sorted_bidx[1:] != sorted_bidx[:-1]
    # seg_start[i] = index of the most recent segment start at or before i.
    start_idx = torch.where(is_start, ar, torch.zeros_like(ar))
    seg_start = torch.cummax(start_idx, dim=0).values
    pos_in_seg = ar - seg_start
    # seg_len via counts scattered by sample index, gathered back per element.
    counts = torch.zeros(B, dtype=torch.long, device=device).scatter_add_(
        0, sorted_bidx, torch.ones(M, dtype=torch.long, device=device))
    seg_len = counts[sorted_bidx]
    max_len = int(pos_in_seg.max().item()) + 1
    return pos_in_seg, seg_start, seg_len, max_len


class EventStreamSegS7(nn.Module):
    """True event-by-event dual-order selective-SSM per-event hand segmenter.

    Parameters (all defaulted → config-optional; see the recovered design brief)
    --------------------------------------------------------------------------
    in_features
        Per-event dataset channels (2 = signed polarity, normalized time).
    embed_dim
        Per-event / SSM model width ``d``.
    d_state
        Complex-diagonal SSM modes ``N`` per scan (S7 = 64 in the brief).
    ssm_layers
        Selective-SSM blocks per scan (brief = 4).
    dual_scan
        Add the bidirectional z-order (Morton) spatial scan alongside the causal time
        scan. ``False`` = time scan only (ablation).
    morton_bits
        Bits per axis for the z-order code (spatial quantization of the scan order).
    time_blocks
        Coarse time blocks the z-order scan is grouped within (order = sample, then
        time-block, then Morton) so nearby-in-time-and-space events are adjacent.
    fourier_bands
        Random-Fourier positional-encoding bands over ``(dx, dy, t)`` (0 = off).
    down_factor, context_channels, context_depth, head_hidden, gn_groups
        Dense global-context bottleneck (ported ``_DenseContext``): coarse grid ``Hd =
        ceil(H/down_factor)``, width, dilated-conv depth, head MLP width, GroupNorm.
    coord_mode, motion_features, motion_radius, motion_dir, motion_min_count,
    density_features, density_radius
        Per-event geometry / normal-flow motion / neighborhood-density channels (reused
        from the baseline verbatim).
    dropout, dmd_gate, dmd_gate_scale, presence_gate, presence_gate_weight,
    presence_gate_scale, presence_min_fg, aux_shape_head, aux_grid, aux_shape_weight,
    null_loss_weight, null_margin
        The baseline's FP-suppression + aux machinery, ported unchanged (see
        ``EventSSMSegStream``).
    scan_weight
        Dual-scan consistency penalty weight (read by ``model_interface`` via the
        ``_scan_loss`` hook; 0 = off). Only meaningful when ``dual_scan``.
    boundary_loss_weight, boundary_loss_band
        Honoured by the trainer to up-weight near-boundary events (default off).
    event_pool
        Event-pooling downsample factor between blocks (>1). NOT YET IMPLEMENTED — kept
        as a documented knob; must stay 1 (full event resolution) in this build.
    num_classes
        Output channels (1 = binary fg/bg).
    """

    def __init__(
        self,
        in_features: int = 2,
        embed_dim: int = 128,
        d_state: int = 64,
        ssm_layers: int = 4,
        dual_scan: bool = True,
        morton_bits: int = 10,
        time_blocks: int = 6,
        fourier_bands: int = 8,
        down_factor: int = 8,
        context_channels: int = 128,
        context_depth: int = 2,
        head_hidden: int = 128,
        num_classes: int = 1,
        coord_mode: str = "relative",
        motion_features: bool = True,
        motion_radius: int = 3,
        motion_dir: bool = False,
        motion_min_count: int = 6,
        density_features: bool = True,
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
        scan_weight: float = 0.1,
        boundary_loss_weight: float = 1.0,
        boundary_loss_band: int = 0,
        event_pool: int = 1,
        use_checkpoint: bool = False,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.embed_dim = int(embed_dim)
        self.d_state = int(d_state)
        self.num_classes = int(num_classes)
        self.dual_scan = bool(dual_scan)
        self.morton_bits = int(morton_bits)
        self.time_blocks = max(1, int(time_blocks))
        self.fourier_bands = int(fourier_bands)
        self.down_factor = max(1, int(down_factor))
        self.context_channels = int(context_channels)
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
        self.scan_weight = float(scan_weight)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.boundary_loss_band = int(boundary_loss_band)
        self.event_pool = max(1, int(event_pool))
        if self.event_pool != 1:
            raise NotImplementedError(
                "event_pool>1 (event-pooling downsample) is a documented future "
                "optimization; keep event_pool=1 in this build.")
        # Gradient-checkpoint the selective-SSM blocks: the segmented parallel scan
        # retains ~log2(N) complex (N, d_state) buffers per block, which dominates
        # memory on long event streams. Checkpointing recomputes each block's scan in
        # the backward pass instead of storing those intermediates -> ~O(blocks) memory
        # for one extra scan recompute. Train-time only; identical outputs/grads.
        self.use_checkpoint = bool(use_checkpoint)

        # Hooks read by model_interface (reset each forward).
        self._aux_logits = None
        self._presence_logit = None
        self._event_embedding = None
        self._scan_loss = None
        self._dyn_energy = None
        self._freqs = None

        d = self.embed_dim
        # ---- per-event feature widths (mirrors EventSSMSegStream) ---------------------
        n_geom = {"relative": 2, "absolute": 2, "both": 4, "none": 0}[self.coord_mode]
        n_motion = (3 + (2 if self.motion_dir else 0)) if self.motion_features else 0
        n_density = 3 if self.density_features else 0
        n_subcell = 3
        n_fourier = 2 * self.fourier_bands if self.fourier_bands > 0 else 0
        self.n_geom, self.n_motion, self.n_density = n_geom, n_motion, n_density
        feat_dim = (self.in_features + n_geom + n_motion + n_density
                    + n_subcell + n_fourier)
        self._feat_dim = feat_dim
        # Fixed random-Fourier projection over (dx, dy, t) — a buffer, not trained.
        if self.fourier_bands > 0:
            self.register_buffer(
                "fourier_W", torch.randn(3, self.fourier_bands) * 2.0 * math.pi)
        else:
            self.fourier_W = None

        # ---- trainable submodules -----------------------------------------------------
        self.event_mlp = nn.Sequential(
            nn.Linear(feat_dim, d), nn.LayerNorm(d), nn.ReLU(inplace=True),
            nn.Linear(d, d),
        )
        L = max(1, int(ssm_layers))
        self.time_blocks_mod = nn.ModuleList(
            _SelectiveSSMBlock(d, self.d_state, bidirectional=False) for _ in range(L))
        self.time_norm = nn.LayerNorm(d)
        if self.dual_scan:
            self.zorder_blocks = nn.ModuleList(
                _SelectiveSSMBlock(d, self.d_state, bidirectional=True) for _ in range(L))
            self.zorder_norm = nn.LayerNorm(d)
            # Shared probe for the dual-scan consistency aux. Only created when the term
            # is actually used (scan_weight>0), else it would be a dead parameter that
            # trips DDP's unused-parameter check. scan_weight=0 => no probe, no hook needed.
            self.scan_probe = (nn.Linear(d, 1) if self.scan_weight > 0.0 else None)
        else:
            self.zorder_blocks = None
            self.scan_probe = None

        self.context = _DenseContext(d, self.context_channels, gn_groups,
                                     depth=max(1, int(context_depth)))
        self.dyn_a = nn.Parameter(torch.tensor(4.0))
        self.dyn_b = nn.Parameter(torch.tensor(-2.0))

        # Per-event head: ev_embed ⊕ time-state ⊕ z-state ⊕ bilinear context ⊕ dynamic
        # energy ⊕ raw feats -> one logit per event.
        z_width = d if self.dual_scan else 0
        head_in = d + d + z_width + self.context_channels + 1 + feat_dim
        self.feat_drop = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.LayerNorm(head_hidden),
            nn.ReLU(inplace=True), nn.Dropout(float(dropout)),
            nn.Linear(head_hidden, num_classes),
        )
        self.presence_head = (nn.Sequential(
            nn.Linear(self.context_channels, self.context_channels),
            nn.ReLU(inplace=True), nn.Linear(self.context_channels, 1))
            if self.presence_gate else None)
        self.aux_shape = (nn.Conv2d(self.context_channels, 1, kernel_size=1)
                          if self.aux_shape_head else None)

    # ------------------------------------------------------------------ feature helpers
    # (copied verbatim from EventSSMSegStream to keep this model self-contained)

    def _geom_feats(self, x, y, batch_idx, B, H, W):
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

    # ------------------------------------------------------------------ scan orderings

    def _run_scan(self, blocks, norm, ev_embed, order, dt_ord, pos, seg_start,
                  seg_len, max_len):
        """Run a stack of selective-SSM blocks in a given event order and scatter the
        per-event output back to the ORIGINAL event rows (row-alignment preserved)."""
        x = ev_embed[order]
        freqs = None
        ckpt = self.use_checkpoint and self.training and x.requires_grad
        for blk in blocks:
            if ckpt:
                # Non-reentrant checkpoint handles the int max_len arg and tuple return;
                # recomputes the scan in backward instead of retaining its intermediates.
                x, freqs = checkpoint(
                    blk, x, dt_ord, pos, seg_start, seg_len, max_len,
                    use_reentrant=False)
            else:
                x, freqs = blk(x, dt_ord, pos, seg_start, seg_len, max_len)
        x = norm(x)
        out = torch.empty_like(x)
        out[order] = x                       # invert the permutation -> original rows
        return out, freqs

    # ------------------------------------------------------------------ forward

    def forward(self, batch) -> torch.Tensor:
        self._aux_logits = None
        self._presence_logit = None
        self._event_embedding = None
        self._scan_loss = None
        feats = batch.feats
        N = feats.shape[0]
        if N == 0:
            return feats.new_zeros((0,) if self.num_classes == 1 else (0, self.num_classes))

        T, H, W = self.time_blocks, int(batch.height), int(batch.width)
        B = batch.batch_size
        df = self.down_factor
        Hd = (H + df - 1) // df
        Wd = (W + df - 1) // df
        x = batch.coords[:, 0]
        y = batch.coords[:, 1]
        times = batch.times
        bidx = batch.batch_idx.long()
        t_bin = (times * T).floor().clamp_(0, T - 1).long()

        # ---- per-event feature enrichment (post-augmentation) -------------------------
        parts = [feats]
        gfeat = self._geom_feats(x, y, bidx, B, H, W)
        if gfeat is not None:
            parts.append(gfeat.to(feats.dtype))
        dfeat = self._density_feats(x, y, times, t_bin, bidx, B, T, H, W)
        if dfeat is not None:
            parts.append(dfeat.to(feats.dtype))
        mfeat = self._motion_feats(x, y, times, bidx, B, H, W)
        if mfeat is not None:
            parts.append(mfeat.to(feats.dtype))
        gy = (y.long() * Hd // H).clamp(0, Hd - 1)
        gx = (x.long() * Wd // W).clamp(0, Wd - 1)
        sub_x = (x.float() * Wd / max(W, 1) - gx.float()).clamp(0.0, 1.0)
        sub_y = (y.float() * Hd / max(H, 1) - gy.float()).clamp(0.0, 1.0)
        sub_t = (times * T - t_bin.float())
        parts.append(torch.stack([sub_x, sub_y, sub_t], dim=1).to(feats.dtype))
        if self.fourier_W is not None:
            if gfeat is not None and self.coord_mode in ("relative", "both"):
                dxy = gfeat[:, -2:]           # relative (dx,dy)
            else:
                dxy = torch.stack([x.float() / max(W - 1, 1) * 2 - 1,
                                   y.float() / max(H - 1, 1) * 2 - 1], dim=1)
            fin = torch.cat([dxy.to(feats.dtype),
                             times.unsqueeze(1).to(feats.dtype)], dim=1)   # (N,3)
            proj = fin @ self.fourier_W.to(feats.dtype)                    # (N,bands)
            parts.append(torch.cat([torch.sin(proj), torch.cos(proj)], dim=1))
        feats_full = torch.cat(parts, dim=1)
        ev_embed = self.event_mlp(feats_full)                             # (N,d)
        d = ev_embed.shape[1]

        # ---- TIME scan (native time order, causal) ------------------------------------
        # Order by (sample, time); double keeps samples separable at high event counts.
        # stable=True makes ties (equal times) resolve by original row -> deterministic
        # and, crucially, sample-local (no cross-sample leakage through the sort).
        key_t = bidx.double() + times.double().clamp(0.0, 1.0 - 1e-9)
        order_t = torch.argsort(key_t, stable=True)
        bidx_t = bidx[order_t]
        pos_t, ss_t, sl_t, ml_t = _seg_layout(bidx_t, B)
        times_t = times[order_t]
        dt_t = torch.zeros_like(times_t)
        dt_t[1:] = (times_t[1:] - times_t[:-1])
        dt_t = torch.where(pos_t > 0, dt_t, torch.zeros_like(dt_t))
        time_state, freqs = self._run_scan(
            self.time_blocks_mod, self.time_norm, ev_embed,
            order_t, dt_t, pos_t, ss_t, sl_t, ml_t)

        # ---- z-order (Morton) scan (spatial locality, bidirectional) ------------------
        # Morton code over a FINE (2^bits per axis) quantization of the full-res (x,y),
        # NOT the coarse context grid — fine codes keep spatially-adjacent events
        # sequence-adjacent and avoid the heavy ties a coarse grid would create.
        if self.dual_scan:
            mb = self.morton_bits
            gx_m = (x.long() * (1 << mb) // max(W, 1)).clamp(0, (1 << mb) - 1)
            gy_m = (y.long() * (1 << mb) // max(H, 1)).clamp(0, (1 << mb) - 1)
            morton = _morton2d(gx_m, gy_m, mb)
            n_mort = 1 << (2 * mb)
            key_z = (bidx * self.time_blocks + t_bin) * n_mort + morton
            order_z = torch.argsort(key_z, stable=True)
            bidx_z = bidx[order_z]
            pos_z, ss_z, sl_z, ml_z = _seg_layout(bidx_z, B)
            dt_z = torch.ones(order_z.shape[0], device=feats.device, dtype=times.dtype)
            z_state, _ = self._run_scan(
                self.zorder_blocks, self.zorder_norm, ev_embed,
                order_z, dt_z, pos_z, ss_z, sl_z, ml_z)
        else:
            z_state = None

        # ---- fuse per-event states, scatter to coarse grid for dense context ----------
        fused = time_state + z_state if z_state is not None else time_state
        cell = (bidx * Hd + gy) * Wd + gx                                 # (N,) cell id
        n_cells = B * Hd * Wd
        csum = fused.new_zeros(n_cells, d).index_add_(0, cell, fused)
        csum2 = fused.new_zeros(n_cells, d).index_add_(0, cell, fused * fused)
        ccnt = fused.new_zeros(n_cells, 1).index_add_(0, cell, fused.new_ones(N, 1))
        cmean = csum / ccnt.clamp(min=1.0)
        # DMD dynamic energy per cell = variance of fused states over its events.
        dyn = (csum2 / ccnt.clamp(min=1.0) - cmean * cmean).clamp(min=0.0).mean(
            dim=-1, keepdim=True)                                         # (n_cells,1)
        self._dyn_energy, self._freqs = dyn, freqs

        agg_map = cmean.view(B, Hd, Wd, d).permute(0, 3, 1, 2).contiguous()
        occ = (ccnt.view(B, Hd, Wd, 1) > 0).to(fused.dtype).permute(0, 3, 1, 2).contiguous()
        ctx_map = self.context(agg_map, occ)                              # (B,c_ctx,Hd,Wd)
        ctx_flat = ctx_map.permute(0, 2, 3, 1).reshape(n_cells, self.context_channels)

        if self.presence_head is not None:
            denom = occ.flatten(2).sum(dim=2).clamp(min=1.0)
            gdesc = (ctx_map * occ).flatten(2).sum(dim=2) / denom
            self._presence_logit = self.presence_head(gdesc).squeeze(-1)  # (B,)

        if self.aux_shape is not None and self.training:
            G = self.aux_grid
            pooled = F.adaptive_avg_pool2d(ctx_map, (G, G))
            self._aux_logits = self.aux_shape(pooled)                     # (B,1,G,G)

        # ---- per-event head -----------------------------------------------------------
        ctx_ev = self.feat_drop(
            self._bilinear_gather(ctx_flat, x, y, bidx, Hd, Wd, H, W))    # (N,c_ctx)
        dyn_ev = self._bilinear_gather(dyn, x, y, bidx, Hd, Wd, H, W)     # (N,1)
        head_parts = [ev_embed, time_state]
        if z_state is not None:
            head_parts.append(z_state)
        head_parts += [ctx_ev, dyn_ev, feats_full]
        head_in = torch.cat(head_parts, dim=1)
        emb = self.head[:-1](head_in)
        logits = self.head[-1](emb)
        if self.training and self.null_loss_weight > 0.0:
            self._event_embedding = emb

        # Dual-scan consistency: the two orders should predict the same per-event label.
        if self.training and self.scan_probe is not None and z_state is not None:
            p_t = self.scan_probe(time_state)
            p_z = self.scan_probe(z_state)
            self._scan_loss = F.mse_loss(p_t, p_z)

        # DMD static veto (per-event): low dynamic energy -> suppress.
        if self.dmd_gate:
            logits = logits + self.dmd_gate_scale * F.logsigmoid(
                self.dyn_a * dyn_ev + self.dyn_b)

        # Per-window presence gate (null state).
        if self._presence_logit is not None:
            gate = self.presence_gate_scale * F.logsigmoid(self._presence_logit)
            logits = logits + gate[bidx].view(logits.shape)

        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
