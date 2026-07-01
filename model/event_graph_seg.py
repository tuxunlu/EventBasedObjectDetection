"""Event-native per-event segmentation by a **graph neural network** — from scratch.

Why a graph net (the research that motivates this design)
---------------------------------------------------------
Events are a sparse spatiotemporal point set; the natural, cheapest representation
is a graph over events in ``(x, y, t)`` whose cost scales with the number of events
rather than the image area. The event-GNN literature converges on a small set of
choices that hit the low-power / low-latency envelope this task needs:

  * **Directed, causal edges** (``t_i < t_j``) keep the per-event update *constant-
    size with depth*, the property that makes asynchronous event GNNs fast — DAGr
    (Gehrig & Scaramuzza, *Nature* 2024; arXiv:2211.12324) and EvGNN (Yang, Kneip &
    Frenkel, arXiv:2404.19489).
  * **A max-relative operator** (``max_j Θᵀ(x_j − x_i, e_ij)``) is the FPGA-proven,
    INT8-friendly message function — EvGNN reaches **16 µs/event** with it (vs the
    far heavier B-spline kernels of AEGNN, arXiv:2203.17149). We use it, not Spline.
  * **Grid / hash graph construction**, never kNN or farthest-point sampling: a fixed
    voxel-offset neighborhood (Morton-/cell-list style, cf. PTv3 arXiv:2312.10035) is
    O(N) and amortizes the single most expensive part of a point GNN.
  * **Motion belongs on the temporal edges** (relative velocity ``Δp/Δt``;
    arXiv:2507.15150) — this is the literature's answer to *this project's own
    measurement* (memory: ``event-seg-motion-feature-nondiscriminative``) that a
    per-NODE hand-crafted motion descriptor is non-discriminative.

The closest published analogues bound what to expect: **GTNN** (arXiv:2404.10940) does
the *same* per-event binary moving/background task (kNN k=16, ~5.3 M params), and
**GMNN** (arXiv:2305.03640) does per-event panoptic seg at **3.9 M params, ~0.1 ms per
10 ms window** with **multi-scale neighborhoods** as its key accuracy lever. Notably
both found **polarity unhelpful** for per-event seg, so node features default to
geometry + time only (``use_polarity`` toggles it back on for ablation).

What is novel here (the contribution)
-------------------------------------
No published GNN couples an async/incremental backbone to a per-event *segmentation*
head — every per-event-seg GNN is static-window; every async backbone does
classification / detection / flow. This model is that bridge, and adds two task-
targeted modules:

  1. **CMR-Conv** — Causal Max-Relative graph convolution on a fixed grid-hash graph
     with **relative velocity on the temporal edges**. A novel *combination* (causal
     directedness + max-relative + motion-edges) aimed at per-event labeling.
  2. **Blob-Affinity Diffusion (BAD)** — the genuinely new module. This project
     measured that the *only* signal separating a dense moving hand-edge from
     locally-identical dense background is **membership in the one connected moving
     blob**. BAD operationalizes exactly that as a differentiable graph diffusion:
     on the coarse (whole-frame) graph it seeds a "blobness" score from learned
     feature + local density, then runs a few APPNP-style personalized-PageRank
     propagation steps (Klicpera et al., arXiv:1810.05997) so the score flows across
     the connected component and *decays across the empty gaps* that separate the
     hand from background clutter. The converged per-node membership is broadcast back
     to every event as a feature **and** an optional suppress-only veto — encoding the
     global-shape prior as a graph operation instead of hoping a conv receptive field
     learns it.

Three hard-won, *measured* levers from the spconv line are carried over unchanged so
this is a re-grounding, not a blank-slate gamble:
  * centroid-**relative** coordinates (LOSO-safe; absolute xy is a per-subject work-
    location fingerprint that leaks across the held-out split),
  * a **global-context token** (whole-frame descriptor broadcast to every event),
  * the **presence gate** (a per-window "is a hand present?" logit whose log-sigmoid
    gates every event logit — the validated fix that cut no-motion false positives
    2.16 % → 0.76 %; memory ``event-seg-presence-gate-fix``).

Deployment note (honest latency)
--------------------------------
No method has a *measured* ~1 ms result for dense per-event seg, and at this scene's
event rate (~1.35 M ev/s) pure per-event async cannot keep up; the realistic ~1 ms
target is **micro-window streaming** (1–5 ms slices labeled given a persistent global
state). This synchronous, full-window forward is the training / accuracy artifact;
its weights convert to the micro-window async runtime (train-dense/run-async, the
AEGNN / Messikommer arXiv:2003.09148 recipe). The model uses only gather / scatter /
MLP (no spconv) so it exports cleanly to ONNX / TensorRT.

Shape contract (identical to ``EventSparseSegGC``)
--------------------------------------------------
``forward(batch) -> logits``: ``(N,)`` when ``num_classes == 1`` (default), else
``(N, num_classes)``, row ``i`` aligned to ``batch.coords[i]`` / ``batch.labels[i]``.
Stashes ``_presence_logit`` (and ``_aux_logits`` when the aux head is on) for the
train-only auxiliary losses already implemented in ``model_interface``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt


# --------------------------------------------------------------------------- #
# Small shared pieces                                                          #
# --------------------------------------------------------------------------- #
class _DropPath(nn.Module):
    """Row-wise stochastic depth on a residual branch (Huang et al., ECCV 2016)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], 1).bernoulli_(keep).div_(keep)
        return x * mask


def _make_offsets(r_s: int, r_t: int, causal: bool) -> torch.Tensor:
    """Fixed neighbourhood offsets ``(K, 3)`` over ``(Δt_bin, Δgy, Δgx)``.

    Spatial offsets span ``[-r_s, r_s]²``; temporal offsets span ``[-r_t, 0]`` when
    ``causal`` (messages flow past→present only — the constant-depth-cost property)
    else ``[-r_t, r_t]``. The self offset ``(0, 0, 0)`` is always included so every
    node has at least one valid neighbour (itself) and the max-aggregate is defined.
    """
    ts = range(-r_t, 1) if causal else range(-r_t, r_t + 1)
    offs = [(dt, dy, dx)
            for dt in ts
            for dy in range(-r_s, r_s + 1)
            for dx in range(-r_s, r_s + 1)]
    return torch.tensor(offs, dtype=torch.long)


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather rows ``x[idx]`` via ``index_select`` — for a FAST backward.

    PyTorch's advanced-indexing backward (``x[long_idx]``) dispatches to
    ``index_put_(accumulate=True)`` / ``indexing_backward_kernel``, which a profiler
    showed costs **~83 % of this model's backward time** (a known gotcha for the
    many duplicate indices a neighbour gather produces). ``index_select`` returns the
    identical result with a dedicated ``index_add`` backward that is ~1–2 orders of
    magnitude faster. Supports ``x`` of shape ``(M,)`` or ``(M, C)`` and any-shape ``idx``.
    """
    out = x.index_select(0, idx.reshape(-1))
    if x.dim() == 1:
        return out.view(idx.shape)
    return out.view(*idx.shape, x.shape[-1])


class CMRConv(nn.Module):
    """Causal Max-Relative graph convolution with motion edge features.

    For node ``i`` with fixed-offset neighbours ``j`` (existing grid cells only)::

        m_ij = MLP([ h_j - h_i , e_ij ]) ,   e_ij = [Δy, Δx, Δt, v_y, v_x]
        out_i = max_j m_ij                  (max-relative aggregation)
        h_i'  = LN( ReLU( W_self h_i + out_i ) )   (+ residual if c_in == c_out)

    The edge geometry ``e_ij`` carries the **relative velocity** ``(Δp/Δt)`` between a
    node and its temporal neighbours — the motion cue, on the edges (arXiv:2507.15150).
    The neighbour set is fixed once per level (grid-hash), so all layers at a level
    reuse it. ``-inf`` masking of non-existent neighbours before the max keeps absent
    cells out of the aggregate (the self edge guarantees a finite result).
    """

    GEO = 5  # [Δy, Δx, Δt, v_y, v_x]

    def __init__(self, c_in: int, c_out: int, drop_path: float = 0.0):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Linear(c_in + self.GEO, c_out),
            nn.LayerNorm(c_out),
            nn.ReLU(inplace=True),
        )
        self.lin_self = nn.Linear(c_in, c_out)
        self.norm = nn.LayerNorm(c_out)
        self.act = nn.ReLU(inplace=True)
        self.residual = (c_in == c_out)
        self.drop_path = _DropPath(drop_path) if self.residual else None

    def forward(self, h, nbr_idx, nbr_mask, geom):
        # h: (M, C_in); nbr_idx/nbr_mask: (M, K); geom: (M, K, GEO)
        h_nbr = _gather(h, nbr_idx)                          # (M, K, C_in)
        rel = h_nbr - h.unsqueeze(1)                         # max-relative term
        msg = self.edge(torch.cat([rel, geom], dim=-1))      # (M, K, C_out)
        msg = msg.masked_fill(~nbr_mask.unsqueeze(-1), float("-inf"))
        agg = msg.amax(dim=1)                                # (M, C_out)
        agg = torch.nan_to_num(agg, neginf=0.0)              # safety (self keeps it finite)
        y = self.act(self.norm(self.lin_self(h) + agg))
        if self.residual:
            return h + self.drop_path(y)
        return y


# --------------------------------------------------------------------------- #
# The model                                                                    #
# --------------------------------------------------------------------------- #
class EventGraphSeg(nn.Module):
    """From-scratch event GNN for per-event hand segmentation (no spconv).

    Parameters
    ----------
    in_features
        Per-event channels emitted by the dataset (2 = signed polarity, norm. time).
    stage_channels
        Per-level node width, finest→coarsest. Its length sets the number of graph
        U-Net levels (default 4: base /2, then /4, /8, /16 of full resolution).
    time_bins
        Temporal voxel bins ``T``.
    base_stride
        Spatial stride of the finest (base) grid. 2 → a 240×320 base grid at 480×640.
    radius_s, radius_t
        Spatial / temporal neighbourhood half-widths for message passing.
    causal
        Directed past→present edges (``Δt ≤ 0``). Enables the constant-depth-cost
        async conversion later; harmless for synchronous training.
    layers_per_level
        CMR-Conv layers at each level (the first changes channels, the rest residual).
    num_classes
        Output channels per event (1 = binary foreground logit).
    head_hidden
        Hidden width of the per-event MLP head.
    use_polarity
        Include signed polarity in node features. Default ``False`` (GTNN/GMNN found
        it unhelpful for per-event seg); kept as an ablation toggle.
    dropout, drop_path
        Head dropout and residual stochastic-depth rate (LOSO-overfit regularizers).
    blob_diffusion
        Enable the novel Blob-Affinity Diffusion module. It runs on a dedicated
        **undirected, time-collapsed whole-frame** graph (the coarsest level pooled
        over time), not the causal backbone graph, so membership can flow across the
        full connected component and decay across the hand↔clutter gaps.
    blob_iters, blob_alpha
        APPNP propagation steps and teleport/retain coefficient ``α`` (membership =
        ``(1-α)·seed + α·neighbour_mean``, iterated ``blob_iters`` times). ``blob_iters``
        must be large enough to span the (small) coarse grid — default 8.
    blob_radius
        Undirected spatial radius of the whole-frame diffusion neighbourhood.
    blob_veto
        Also add a suppress-only ``log σ`` veto from the per-event membership to the
        logits (a spatially-precise shape veto), on top of feeding it to the head.
    global_token
        Broadcast the coarsest masked-mean descriptor to every event (global skip).
    presence_gate, presence_gate_weight, presence_gate_scale, presence_min_fg
        Per-window presence gate (the validated null-state fix). ``_presence_logit``
        is read by ``model_interface`` for the BCE; its ``log σ`` gates every event
        logit (suppress-only). Active at train and inference (uses predicted presence).
    aux_shape_head, aux_grid, aux_shape_weight
        Train-only coarse occupancy head off the coarsest level (``_aux_logits``,
        ``(B,1,G,G)``) for the structure-aware auxiliary loss in ``model_interface``.
    coord_scale_eps
        Numerical floor used in velocity normalisation.
    """

    def __init__(
        self,
        in_features: int = 2,
        stage_channels: Sequence[int] = (64, 96, 128, 192),
        time_bins: int = 3,
        base_stride: int = 4,
        radius_s: int = 1,
        radius_t: int = 1,
        causal: bool = True,
        layers_per_level: int = 2,
        num_classes: int = 1,
        head_hidden: int = 128,
        use_polarity: bool = False,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        grad_checkpoint: bool = True,
        blob_diffusion: bool = True,
        blob_iters: int = 8,
        blob_alpha: float = 0.8,
        blob_radius: int = 2,
        blob_veto: bool = True,
        global_token: bool = True,
        presence_gate: bool = True,
        presence_gate_weight: float = 0.3,
        presence_gate_scale: float = 1.0,
        presence_min_fg: int = 0,
        aux_shape_head: bool = True,
        aux_grid: int = 32,
        aux_shape_weight: float = 0.2,
        coord_scale_eps: float = 0.1,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.stage_channels = tuple(int(c) for c in stage_channels)
        self.n_levels = len(self.stage_channels)
        if self.n_levels < 1:
            raise ValueError("stage_channels must have >= 1 entry")
        self.time_bins = int(time_bins)
        self.base_stride = max(1, int(base_stride))
        self.radius_s = max(0, int(radius_s))
        self.radius_t = max(0, int(radius_t))
        self.causal = bool(causal)
        self.layers_per_level = max(1, int(layers_per_level))
        self.num_classes = int(num_classes)
        self.use_polarity = bool(use_polarity)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.blob_diffusion = bool(blob_diffusion)
        self.blob_iters = max(1, int(blob_iters))
        self.blob_alpha = float(blob_alpha)
        self.blob_radius = max(1, int(blob_radius))
        self.blob_veto = bool(blob_veto)
        self.global_token = bool(global_token)
        self.presence_gate = bool(presence_gate)
        self.presence_gate_weight = float(presence_gate_weight)
        self.presence_gate_scale = float(presence_gate_scale)
        self.presence_min_fg = int(presence_min_fg)
        self.aux_shape_head = bool(aux_shape_head)
        self.aux_grid = int(aux_grid)
        self.aux_shape_weight = float(aux_shape_weight)
        self.eps = float(coord_scale_eps)
        # Attributes read by model_interface (kept None when their loss is off).
        self._presence_logit = None
        self._aux_logits = None
        self._event_embedding = None   # this model does not use the null-prototype loss
        self.null_loss_weight = 0.0
        self._jepa_loss = None

        # Node-feature input width: [rel_dx, rel_dy, t] (+ polarity).
        self.node_in = 3 + (1 if self.use_polarity else 0)
        c0 = self.stage_channels[0]
        self.embed = nn.Sequential(
            nn.Linear(self.node_in, c0), nn.LayerNorm(c0), nn.ReLU(inplace=True),
        )

        # Fixed neighbourhood offsets (registered so device moves with the model).
        self.register_buffer("offsets", _make_offsets(self.radius_s, self.radius_t, self.causal),
                             persistent=False)

        # Encoder CMR-Conv stacks per level; a Linear pool-projection between levels.
        self.enc = nn.ModuleList()
        self.pool_proj = nn.ModuleList()
        for lvl in range(self.n_levels):
            c = self.stage_channels[lvl]
            c_prev = self.stage_channels[lvl - 1] if lvl > 0 else c0
            blocks = nn.ModuleList()
            blocks.append(CMRConv(c_prev if lvl == 0 else c, c, drop_path=drop_path))
            for _ in range(self.layers_per_level - 1):
                blocks.append(CMRConv(c, c, drop_path=drop_path))
            self.enc.append(blocks)
            # project finer features to this level's width before pooling-in (lvl>0).
            self.pool_proj.append(nn.Linear(c_prev, c) if lvl > 0 else nn.Identity())

        # Decoder: upsample coarse→fine with a skip add + one CMR-Conv refine.
        self.dec = nn.ModuleList()
        self.up_proj = nn.ModuleList()
        for lvl in range(self.n_levels - 1):           # levels 0..L-2 get a decoder
            c = self.stage_channels[lvl]
            c_coarse = self.stage_channels[lvl + 1]
            self.up_proj.append(nn.Linear(c_coarse, c))
            self.dec.append(CMRConv(c, c, drop_path=drop_path))

        c_top = self.stage_channels[-1]               # coarsest (global) width

        # Blob-Affinity Diffusion seed head (coarse-level scalar seed per node).
        if self.blob_diffusion:
            self.blob_seed = nn.Sequential(
                nn.Linear(c_top + 1, c_top), nn.ReLU(inplace=True),
                nn.Linear(c_top, 1),
            )
            # Dedicated UNDIRECTED, spatial-only (time-collapsed) neighbourhood for the
            # whole-frame diffusion — distinct from the backbone's (causal) offsets.
            self.register_buffer("blob_offsets",
                                 _make_offsets(self.blob_radius, 0, causal=False),
                                 persistent=False)
        else:
            self.blob_seed = None

        # Presence head on the coarse masked-mean descriptor (the null-state gate).
        if self.presence_gate:
            self.presence_head = nn.Sequential(
                nn.Linear(c_top, c_top), nn.ReLU(inplace=True), nn.Linear(c_top, 1),
            )
        else:
            self.presence_head = None

        # Train-only coarse occupancy head (structure-aware aux supervision).
        self.aux_head = nn.Linear(c_top, 1) if self.aux_shape_head else None

        # Per-event head: fine node ctx ⊕ raw event feats ⊕ PER-EVENT geometry ⊕ global
        # token ⊕ blob memb. The +node_in re-injects each event's own relative dx/dy/t
        # (mean-pooled away into the base voxel otherwise), so two events sharing a base
        # voxel stay separable at the head — restores sub-voxel boundary precision.
        head_in = c0 + self.in_features + self.node_in
        if self.global_token:
            head_in += c_top
        if self.blob_diffusion:
            head_in += 1
        self.feat_drop = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.LayerNorm(head_hidden),
            nn.ReLU(inplace=True), nn.Dropout(float(dropout)),
            nn.Linear(head_hidden, num_classes),
        )

    # ------------------------------------------------------------------ helpers
    def _node_features(self, x, y, times, batch_idx, feats, B, H, W):
        """Centroid-relative geometry + time (+ polarity) per event — LOSO-safe."""
        xf = x.float(); yf = y.float()
        b = batch_idx.long()
        cnt = torch.zeros(B, device=xf.device).scatter_add_(0, b, torch.ones_like(xf))
        cx = torch.zeros(B, device=xf.device).scatter_add_(0, b, xf) / cnt.clamp(min=1.0)
        cy = torch.zeros(B, device=xf.device).scatter_add_(0, b, yf) / cnt.clamp(min=1.0)
        scale = 0.5 * float((H ** 2 + W ** 2) ** 0.5)
        dx = (xf - cx[b]) / scale
        dy = (yf - cy[b]) / scale
        parts = [dx, dy, times.float()]
        if self.use_polarity:
            parts.append(feats[:, 0].float())           # signed polarity channel
        return torch.stack(parts, dim=1)

    @staticmethod
    def _pool(keys_coords, feats, pos):
        """Dedup integer ``(b, tb, gy, gx)`` rows → unique nodes with mean feat & pos.

        Returns ``(uniq_keys, coords(M,4), feat(M,C), pos(M,3), inverse(N,))`` where
        ``inverse`` maps each input row to its node row (in sorted-key order).
        """
        b, tb, gy, gx, key = keys_coords
        uniq, inverse = torch.unique(key, sorted=True, return_inverse=True)
        M = uniq.numel()
        C = feats.shape[1]
        ones = torch.ones(inverse.shape[0], device=feats.device)
        cnt = torch.zeros(M, device=feats.device).scatter_add_(0, inverse, ones).clamp(min=1.0)
        nf = torch.zeros(M, C, device=feats.device).index_add_(0, inverse, feats) / cnt.unsqueeze(1)
        npos = torch.zeros(M, 3, device=feats.device).index_add_(0, inverse, pos) / cnt.unsqueeze(1)
        # Recover each node's integer coords from its first occurrence: the smallest
        # input-row index per node, via an amin scatter (init = N, a safe sentinel).
        row = torch.arange(inverse.shape[0], device=feats.device)
        first = torch.full((M,), inverse.shape[0], dtype=torch.long, device=feats.device)
        first = first.scatter_reduce(0, inverse, row, reduce="amin", include_self=True)
        coords = torch.stack([b[first], tb[first], gy[first], gx[first]], dim=1)
        return uniq, coords, nf, npos, inverse

    def _neighbors(self, uniq, coords, T, Gh, Gw, offsets=None):
        """Fixed-offset grid neighbours via searchsorted → ``(nbr_idx, mask) (M,K)``.

        ``offsets`` defaults to the backbone's (possibly causal) neighbourhood; pass a
        different set (e.g. the undirected ``blob_offsets``) for other graphs.
        """
        b, tb, gy, gx = (coords[:, i] for i in range(4))
        off = (self.offsets if offsets is None else offsets).to(coords.device)
        dt, dy, dx = off[:, 0], off[:, 1], off[:, 2]                # (K,)
        ntb = tb[:, None] + dt[None]; ngy = gy[:, None] + dy[None]; ngx = gx[:, None] + dx[None]
        inb = (ntb >= 0) & (ntb < T) & (ngy >= 0) & (ngy < Gh) & (ngx >= 0) & (ngx < Gw)
        nkey = ((b[:, None] * T + ntb) * Gh + ngy) * Gw + ngx       # (M, K)
        nkey_c = nkey.clamp(min=0)
        pos = torch.searchsorted(uniq, nkey_c)
        M = uniq.numel()
        pos_c = pos.clamp(max=M - 1)
        exists = inb & (uniq[pos_c] == nkey)
        nbr_idx = pos_c.masked_fill(~exists, 0)                     # safe gather index
        return nbr_idx, exists

    def _edge_geom(self, pos, nbr_idx, nbr_mask, H, W):
        """Edge features ``[Δy, Δx, Δt, v_y, v_x]`` incl. relative velocity Δp/Δt."""
        pn = _gather(pos, nbr_idx)                                  # (M, K, 3): y,x,t
        d = pn - pos.unsqueeze(1)                                   # (M, K, 3)
        dy = (d[..., 0] / H).clamp(-1, 1)
        dx = (d[..., 1] / W).clamp(-1, 1)
        dt = d[..., 2].clamp(-1, 1)                                 # times in [0,1]
        # SIGN-PRESERVING velocity Δp/Δt: a |Δt| denominator would make the sign depend
        # only on the spatial offset, conflating temporally-opposite motions (the very
        # cue this feature exists to provide). Floor the MAGNITUDE of Δt at eps while
        # keeping its sign; the self edge (Δt=0, Δp=0) still gives v=0.
        sign = torch.where(dt < 0, -1.0, 1.0)
        denom = sign * dt.abs().clamp(min=self.eps)
        vy = (dy / denom).clamp(-1, 1)
        vx = (dx / denom).clamp(-1, 1)
        geom = torch.stack([dy, dx, dt, vy, vx], dim=-1)
        return geom * nbr_mask.unsqueeze(-1).float()

    def _blob_diffusion(self, feat, pos, coords, Gh, Gw):
        """Novel: diffuse a 'dominant moving-blob membership' over a WHOLE-FRAME graph.

        The membership prior must (a) flow across the entire connected component and
        (b) decay across the empty gaps separating the hand from background clutter —
        which requires an UNDIRECTED, single-time-slab graph with reach >= the frame.
        So we first **collapse the coarsest level over time** to 2-D supernodes
        ``(b, gy, gx)``, build a dedicated **undirected** neighbourhood (``blob_offsets``,
        spatial only), then run APPNP / personalized-PageRank::

            seed_i = σ( MLP([feat_i, log degree_i]) )
            m ← (1-α)·seed + α·mean_{j∈N(i)} m_j        (× blob_iters)

        ``blob_iters`` is sized to span the (small) coarse grid so the score genuinely
        reaches frame-scale. Returns membership per INPUT coarse node (broadcast back
        from the 2-D supernode it pooled into).
        """
        b = coords[:, 0]; gy = coords[:, 2]; gx = coords[:, 3]
        zeros = torch.zeros_like(b)
        key2d = (b * Gh + gy) * Gw + gx                            # time-collapsed (b,gy,gx)
        uniq2, coords2, feat2, pos2, inv2 = self._pool((b, zeros, gy, gx, key2d), feat, pos)
        nbr2, mask2 = self._neighbors(uniq2, coords2, 1, Gh, Gw, offsets=self.blob_offsets)
        deg = mask2.sum(dim=1, keepdim=True).float()
        seed = torch.sigmoid(self.blob_seed(torch.cat([feat2, torch.log1p(deg)], dim=-1))).squeeze(-1)
        m = seed
        a = self.blob_alpha
        maskf = mask2.float(); denom = maskf.sum(1).clamp(min=1.0)
        for _ in range(self.blob_iters):
            mn = (_gather(m, nbr2) * maskf).sum(1) / denom         # mean over undirected nbrs
            m = (1.0 - a) * seed + a * mn
        return _gather(m, inv2)                                    # 2-D supernode -> coarse node

    def _conv(self, blk, feat, nbr_idx, nbr_mask, geom):
        """Run a CMRConv, optionally gradient-checkpointed (train only).

        The ``(M, K, C)`` edge gather inside CMRConv dominates activation memory; at
        the default fine ``base_stride=4`` / B=8 it OOMs a 24 GB GPU without this.
        Checkpointing recomputes that gather in the backward pass instead of storing it
        — memory then scales with a single block's output ``(M, C)`` rather than the sum
        of all blocks' edge tensors, at ~30 % extra compute (measured: 7.9 GB / ~1.0 s
        per B=8 step, vs OOM). Off at eval (no autograd graph to trade).
        """
        if self.grad_checkpoint and self.training and feat.requires_grad:
            return _ckpt.checkpoint(blk, feat, nbr_idx, nbr_mask, geom, use_reentrant=False)
        return blk(feat, nbr_idx, nbr_mask, geom)

    def forward(self, batch) -> torch.Tensor:
        self._presence_logit = None
        self._aux_logits = None
        feats = batch.feats
        N = feats.shape[0]
        if N == 0:
            return feats.new_zeros((0,) if self.num_classes == 1 else (0, self.num_classes))

        T = self.time_bins
        H, W = int(batch.height), int(batch.width)
        B = batch.batch_size
        x = batch.coords[:, 0].long(); y = batch.coords[:, 1].long()
        times = batch.times
        t_bin = (times * T).floor().clamp_(0, T - 1).long()
        bidx = batch.batch_idx.long()

        node_in = self._node_features(x, y, times, bidx, feats, B, H, W)
        h0 = self.embed(node_in)                                   # (N, c0) per event

        # ---- base grid pooling (events -> base nodes) ----------------------------
        s = self.base_stride
        Gh0 = (H + s - 1) // s; Gw0 = (W + s - 1) // s
        gy0 = (y // s).clamp(0, Gh0 - 1); gx0 = (x // s).clamp(0, Gw0 - 1)
        key0 = ((bidx * T + t_bin) * Gh0 + gy0) * Gw0 + gx0
        pos_ev = torch.stack([y.float(), x.float(), times.float()], dim=1)
        uniq0, coords0, feat0, pos0, inv0 = self._pool(
            (bidx, t_bin, gy0, gx0, key0), h0, pos_ev)

        # ---- encoder: per-level message passing + grid pooling -------------------
        level_uniq: List[torch.Tensor] = []
        level_coords: List[torch.Tensor] = []
        level_pos: List[torch.Tensor] = []
        level_nbr: List[Tuple[torch.Tensor, torch.Tensor]] = []
        level_geom: List[torch.Tensor] = []
        level_feat: List[torch.Tensor] = []
        pool_inv: List[torch.Tensor] = []          # finer-node -> coarser-node (per pool)
        grids: List[Tuple[int, int]] = []

        uniq, coords, feat, pos = uniq0, coords0, feat0, pos0
        Gh, Gw = Gh0, Gw0
        for lvl in range(self.n_levels):
            if lvl > 0:
                feat = self.pool_proj[lvl](feat)                   # project to level width
            nbr_idx, nbr_mask = self._neighbors(uniq, coords, T, Gh, Gw)
            geom = self._edge_geom(pos, nbr_idx, nbr_mask, H, W)
            for blk in self.enc[lvl]:
                feat = self._conv(blk, feat, nbr_idx, nbr_mask, geom)
            level_uniq.append(uniq); level_coords.append(coords); level_pos.append(pos)
            level_nbr.append((nbr_idx, nbr_mask)); level_geom.append(geom)
            level_feat.append(feat); grids.append((Gh, Gw))

            if lvl < self.n_levels - 1:                            # pool to next level
                Ghn = (Gh + 1) // 2; Gwn = (Gw + 1) // 2
                b, tb, gy, gx = (coords[:, i] for i in range(4))
                ngy = (gy // 2).clamp(0, Ghn - 1); ngx = (gx // 2).clamp(0, Gwn - 1)
                nkey = ((b * T + tb) * Ghn + ngy) * Gwn + ngx
                uniq, coords, feat, pos, pinv = self._pool((b, tb, ngy, ngx, nkey), feat, pos)
                pool_inv.append(pinv)
                Gh, Gw = Ghn, Gwn

        # ---- coarsest-level global reasoning -------------------------------------
        top = self.n_levels - 1
        feat_top = level_feat[top]; coords_top = level_coords[top]; pos_top = level_pos[top]
        nbr_top, mask_top = level_nbr[top]
        b_top = coords_top[:, 0].long()
        c_top = feat_top.shape[1]

        # masked-mean global descriptor per sample (for presence + global token).
        denom = torch.zeros(B, 1, device=feat_top.device).index_add_(
            0, b_top, torch.ones(b_top.shape[0], 1, device=feat_top.device)).clamp(min=1.0)
        gdesc = torch.zeros(B, c_top, device=feat_top.device).index_add_(0, b_top, feat_top) / denom

        if self.presence_head is not None:
            self._presence_logit = self.presence_head(gdesc).squeeze(-1)   # (B,)

        # Novel Blob-Affinity Diffusion on a whole-frame (undirected, time-collapsed)
        # graph -> per-coarse-node membership in the dominant moving blob.
        if self.blob_seed is not None:
            Ght, Gwt = grids[top]
            blob_top = self._blob_diffusion(feat_top, pos_top, coords_top, Ght, Gwt)
        else:
            blob_top = None

        # Train-only coarse occupancy logits scattered into a (B,1,G,G) map.
        if self.aux_head is not None and self.training:
            G = self.aux_grid
            node_logit = self.aux_head(feat_top).squeeze(-1)               # (M_top,)
            gy = (pos_top[:, 0] * G / H).long().clamp(0, G - 1)
            gx = (pos_top[:, 1] * G / W).long().clamp(0, G - 1)
            cell = (b_top * G + gy) * G + gx
            nb = B * G * G
            num = torch.zeros(nb, device=feat_top.device).index_add_(0, cell, node_logit)
            cnt = torch.zeros(nb, device=feat_top.device).index_add_(
                0, cell, torch.ones_like(node_logit)).clamp(min=1.0)
            self._aux_logits = (num / cnt).view(B, 1, G, G)

        # ---- decoder: upsample coarse -> fine with skip add + refine -------------
        feat_dec = feat_top
        for lvl in range(self.n_levels - 2, -1, -1):
            up = _gather(self.up_proj[lvl](feat_dec), pool_inv[lvl])      # coarse->finer
            feat_dec = level_feat[lvl] + up                              # U-Net skip
            nbr_idx, nbr_mask = level_nbr[lvl]
            feat_dec = self._conv(self.dec[lvl], feat_dec, nbr_idx, nbr_mask, level_geom[lvl])

        # ---- per-event head ------------------------------------------------------
        ev_ctx = self.feat_drop(_gather(feat_dec, inv0))                 # base node -> event
        head_parts = [ev_ctx, feats, node_in]   # node_in = per-event geometry (re-injected)
        if self.global_token:
            head_parts.append(_gather(gdesc, bidx))                     # global skip
        if blob_top is not None:
            # map event -> base node -> ... -> coarsest node, then gather membership.
            idx = inv0
            for pinv in pool_inv:
                idx = pinv[idx]
            blob_ev = _gather(blob_top, idx).unsqueeze(1)               # (N,1)
            head_parts.append(blob_ev)
        logits = self.head(torch.cat(head_parts, dim=1))                # (N, num_classes)

        # Suppress-only blob veto (spatially-precise shape gate).
        if blob_top is not None and self.blob_veto:
            logits = logits + F.logsigmoid(
                (blob_ev * 6.0 - 3.0)).expand_as(logits)               # σ-centered veto

        # Per-window presence gate (null state): add log σ(presence) to every logit.
        if self._presence_logit is not None:
            gate = self.presence_gate_scale * F.logsigmoid(self._presence_logit)
            logits = logits + _gather(gate, bidx).view(logits.shape[0], -1)

        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
