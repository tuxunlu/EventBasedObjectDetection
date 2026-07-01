"""Manifold-JEPA: a from-scratch, low-power per-event segmenter whose core cue is the
**physics of the event manifold** learned by latent prediction.

Why this rewrite
----------------
Events fire at moving intensity edges, so a 1-D edge swept over time traces a **2-D
sheet (manifold) in (x,y,t)**. The sheet's local geometry IS the motion physics: the
tangent-plane orientation is the normal flow (a planar sheet = constant velocity), the
hand boundary is a **crease** (an orientation discontinuity), and an articulated hand
bends/accelerates the sheet. A Joint-Embedding Predictive Architecture (I-JEPA
2301.08243; V-JEPA 2404.08471 / V-JEPA 2 2506.09985) learns this by predicting the
**future time-slice latents from the past** — a constant-velocity background sheet is
predictable (low residual), the articulated hand is not (high residual = foreground).

The previous EventJEPASeg had three flaws this rewrite fixes:
  1. it masked *random spatial patches* (solvable by a static-appearance shortcut — the
     "hand exists here" prior behind static-window FPs)  ->  we mask along **time**
     (full-frame future prediction) so the pretext must model dynamics, not appearance;
  2. the prediction residual was *never wired into the head*  ->  the **directional
     residual** (angular mismatch between predicted and observed normal flow) and the
     **crease** are first-class per-event features feeding the segmentation head
     (the research lesson: residuals discriminate as *structure / orientation*, not as
     scalar magnitude — this project's measured null + Ye 2025 + ResFlow 2412.09105);
  3. ~262k params were a 4096-row positional table  ->  **factorized** PE (gy+gx+t).

Architecture (all pure ``nn``, no spconv, LayerNorm only — LOSO-safe):
  * spacetime **tubelet tokens** on a (t_bin, gy, gx) grid; each token = max⊕mean pool
    of its events' embeddings; per-event features = [pol, t] + centroid-relative xy
    (LOSO-safe) + intra-cell xy + intra-bin t + signed manifold descriptor
    [a, b, |∇t|, R², log n];
  * a **context encoder** (bidirectional transformer, the inference path), an EMA
    **target encoder** (train-only, stop-grad), and a shallow **predictor** with a
    strictly-causal time mask + an always-visible **anchor** state token (also the
    designated cross-window memory carry-slot);
  * the causal predictor + a small ``flow_head`` give, at BOTH train and inference, a
    per-token **directional residual** ``r_dir = 1 − cos(pred_flow, obs_flow)`` and a
    latent residual ``r_lat = ‖pred − ctx‖``, broadcast to events and fed to the head;
  * proven levers kept: **presence gate** (validated no-motion-FP suppressor), an aux
    occupancy head, centroid-relative coords.

Shape contract (identical to the sparse models): ``forward(batch) -> (N,)`` foreground
logits, row ``i`` aligned to ``batch.coords[i]`` / ``batch.labels[i]``. Train-only the
model stashes ``self._jepa_loss`` (×``jepa_weight``), ``self._presence_logit``
(×``presence_gate_weight``), ``self._aux_logits`` (×``aux_shape_weight``), and
``self._event_embedding`` (×``null_loss_weight``) for ``model_interface`` to read.
Inference drops the EMA target + JEPA loss; it keeps the (cheap) causal predictor pass.

The 144 ms / single-center-frame label-noise issue is handled OUTSIDE the model by the
dataset's ``time_ignore_frac`` trimap (supervise only the center band); the SSL /
residual / crease use the full window.
"""

from __future__ import annotations

import copy
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Signed manifold (normal-flow) descriptor on the Surface of Active Events     #
# --------------------------------------------------------------------------- #
def surface_manifold_features(x, y, times, batch_idx, B, H, W,
                              radius: int = 3, min_count: int = 6):
    """Per-event manifold descriptor from a local plane fit on the time surface.

    Fits ``t ≈ a·Δx + b·Δy + c`` in a ``(2r+1)²`` neighborhood of the most-recent-time
    surface by occupancy-weighted least squares (Benosman et al., IEEE TNNLS 2014),
    vectorized as fixed-kernel 2-D convs + a per-event 3×3 ridge solve. Returns 5
    **signed** channels: ``[a, b, |∇t|, R², log1p(count)]`` where ``(a, b) = (∂t/∂x,
    ∂t/∂y)`` is the tangent / normal-flow direction (the crease + directional-residual
    targets need the *sign*, not just magnitude), ``|∇t| = √(a²+b²)`` the inverse
    normal-flow speed, ``R²`` the plane-fit quality (a confidence weight, not a
    foreground score). All zeroed where the neighborhood is too sparse to constrain the
    plane. Computed under ``no_grad`` — a deterministic input feature.
    """
    with torch.no_grad():
        dev = times.device
        ft = torch.float32
        b = batch_idx.long(); xi = x.long(); yi = y.long()
        r = int(radius); k = 2 * r + 1
        n = B * H * W
        lin = (b * H + yi) * W + xi
        t = times.to(ft)
        sae = torch.zeros(n, device=dev, dtype=ft).scatter_reduce_(
            0, lin, t, reduce="amax", include_self=True)
        cnt = torch.zeros(n, device=dev, dtype=ft).index_add_(0, lin, torch.ones_like(t))
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
        planarity = torch.where(total_var > 1e-6, r2.clamp(0.0, 1.0), torch.ones_like(r2))
        valid = (S1 >= float(min_count)).to(ft)
        out = torch.stack([a * valid, bb * valid, grad_mag * valid,
                           planarity * valid, torch.log1p(cnt_e)], dim=1)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _mlp(sizes: Sequence[int], norm: bool = True, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _transformer(dim: int, depth: int, heads: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=dim * 2, dropout=dropout,
        activation="gelu", batch_first=True, norm_first=True)
    # enable_nested_tensor=False: the NestedTensor fast path mishandles fully-masked
    # rows; keep the plain padded path so masked rows stay isolated.
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


class EventJEPASeg(nn.Module):
    """From-scratch per-event segmenter: manifold tokens + a time-forward JEPA core.

    See the module docstring for the design. All loss weights are stored as attributes
    so ``model_interface`` can read them via ``getattr``; every ``__init__`` arg is
    config-bindable (no ``**kwargs``). The label-noise of a long window is handled by
    the dataset ``time_ignore_frac`` trimap, not here.
    """

    def __init__(
        self,
        in_features: int = 2,
        num_classes: int = 1,
        patch_size: int = 48,
        time_bins: int = 16,
        dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        pred_depth: int = 2,
        head_hidden: int = 64,
        dropout: float = 0.1,
        coord_mode: str = "relative",
        motion_radius: int = 3,
        motion_min_count: int = 6,
        crease_feature: bool = True,
        residual_feature: bool = True,
        dir_residual: bool = True,
        flow_loss_weight: float = 0.5,
        jepa_weight: float = 1.0,
        jepa_mask_ratio: float = 0.5,
        mask_mode: str = "future_all",
        ema_momentum: float = 0.996,
        var_weight: float = 1.0,
        jepa_warmup_epochs: int = 10,
        shear_consistency: bool = False,
        shear_max: float = 0.15,
        shear_weight: float = 0.1,
        presence_gate: bool = True,
        presence_gate_weight: float = 0.3,
        presence_gate_scale: float = 1.0,
        presence_min_fg: int = 0,
        aux_shape_head: bool = True,
        aux_grid: int = 32,
        aux_shape_weight: float = 0.2,
        null_loss_weight: float = 0.0,
        null_margin: float = 1.0,
        max_gy: int = 24,
        max_gx: int = 24,
        max_t: int = 24,
        max_tokens: int = 8192,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.patch_size = int(patch_size)
        self.time_bins = int(time_bins)
        self.dim = int(dim)
        self.coord_mode = str(coord_mode).strip().lower()
        self.motion_radius = int(motion_radius)
        self.motion_min_count = int(motion_min_count)
        self.crease_feature = bool(crease_feature)
        self.residual_feature = bool(residual_feature)
        self.dir_residual = bool(dir_residual)
        self.flow_loss_weight = float(flow_loss_weight)
        self.jepa_weight = float(jepa_weight)
        self.jepa_mask_ratio = float(jepa_mask_ratio)
        self.mask_mode = str(mask_mode).strip().lower()
        self.ema_momentum = float(ema_momentum)
        self.var_weight = float(var_weight)
        self.jepa_warmup_epochs = int(jepa_warmup_epochs)
        self.shear_consistency = bool(shear_consistency)
        self.shear_max = float(shear_max)
        self.shear_weight = float(shear_weight)
        self.presence_gate = bool(presence_gate)
        self.presence_gate_weight = float(presence_gate_weight)
        self.presence_gate_scale = float(presence_gate_scale)
        self.presence_min_fg = int(presence_min_fg)
        self.aux_shape_head = bool(aux_shape_head)
        self.aux_grid = int(aux_grid)
        self.aux_shape_weight = float(aux_shape_weight)
        self.null_loss_weight = float(null_loss_weight)
        self.null_margin = float(null_margin)
        self.max_gy = int(max_gy)
        self.max_gx = int(max_gx)
        self.max_t = int(max_t)
        self.max_tokens = int(max_tokens)
        self.eps = 1e-6

        # model_interface containers (reset every forward).
        self._jepa_loss = None
        self._presence_logit = None
        self._aux_logits = None
        self._event_embedding = None

        d = self.dim
        # Per-event input: [pol, t](in_features) + rel-xy(2) + intra-cell xy(2)
        # + intra-bin t(1) + manifold descriptor(5).
        self.n_event_feats = self.in_features + 2 + 2 + 1 + 5
        self.event_mlp = _mlp([self.n_event_feats, d, d], norm=True, dropout=dropout)
        self.token_proj = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d))

        # Factorized positional encoding (replaces the dense table).
        self.pe_gy = nn.Embedding(self.max_gy, d)
        self.pe_gx = nn.Embedding(self.max_gx, d)
        self.pe_t = nn.Embedding(self.max_t, d)
        for emb in (self.pe_gy, self.pe_gx, self.pe_t):
            nn.init.trunc_normal_(emb.weight, std=0.02)
        # Anchor / state token: always-visible key (no all-`-inf` attention rows) and
        # the designated cross-window memory carry-slot (constant for now).
        self.anchor = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.anchor, std=0.02)

        self.ctx_tf = _transformer(d, depth, heads, dropout)
        self.target_tf = copy.deepcopy(self.ctx_tf)       # EMA target (frozen)
        for p in self.target_tf.parameters():
            p.requires_grad_(False)
        self.pred_tf = _transformer(d, pred_depth, heads, dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pred_in = nn.Linear(d, d)
        self.pred_out = nn.Linear(d, d)
        # Predicts the future normal-flow direction (the directional-residual readout).
        self.flow_head = nn.Linear(d, 2)

        self.presence_head = (nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True),
                                            nn.Linear(d, 1)) if self.presence_gate else None)
        self.aux_head = nn.Linear(d, 1) if self.aux_shape_head else None

        # Per-event head: event embed ⊕ its token's context ⊕ residual(3) ⊕ crease(1).
        self.feat_drop = nn.Dropout(dropout)
        head_in = 2 * d + 3 + 1
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.LayerNorm(head_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_hidden, num_classes))

        self._mask_cache = {}   # (Ntok, S) -> causal attn mask (CPU)

    # ------------------------------------------------------------------ helpers
    def _grid(self, H, W):
        ps = self.patch_size
        Hs = (H + ps - 1) // ps
        Ws = (W + ps - 1) // ps
        return Hs, Ws, self.time_bins * Hs * Ws

    @staticmethod
    def _scatter_mean(vals, idx, M):
        """Mean of ``vals (N,C)`` into ``M`` rows by ``idx (N,)`` -> ``(M,C)``."""
        C = vals.shape[1]
        out = vals.new_zeros(M, C).index_add_(0, idx, vals)
        cnt = vals.new_zeros(M, 1).index_add_(0, idx, vals.new_ones(vals.shape[0], 1))
        return out / cnt.clamp_min(1.0)

    def _event_feats(self, x, y, times, feats, t_bin, batch_idx, B, H, W):
        """12-dim per-event feature vector (relative coords + manifold)."""
        xf = x.float(); yf = y.float()
        b = batch_idx.long()
        cnt = torch.zeros(B, device=xf.device, dtype=xf.dtype).scatter_add_(0, b, torch.ones_like(xf))
        cx = torch.zeros(B, device=xf.device, dtype=xf.dtype).scatter_add_(0, b, xf) / cnt.clamp_min(1.0)
        cy = torch.zeros(B, device=xf.device, dtype=xf.dtype).scatter_add_(0, b, yf) / cnt.clamp_min(1.0)
        scale = 0.5 * float((H ** 2 + W ** 2) ** 0.5)
        rel = torch.stack([(xf - cx[b]) / scale, (yf - cy[b]) / scale], dim=1)
        ps = self.patch_size
        intra = torch.stack([(xf % ps) / ps, (yf % ps) / ps], dim=1)
        intra_t = (times.float() * self.time_bins - t_bin.float()).unsqueeze(1)
        mg = surface_manifold_features(x, y, times, batch_idx, B, H, W,
                                       self.motion_radius, self.motion_min_count)
        return torch.cat([feats[:, :self.in_features].float(), rel, intra, intra_t,
                          mg.to(xf.dtype)], dim=1), mg

    def _tokens(self, ev_embed, gtok, B, Ntok):
        """Pool per-event embeddings into tubelet tokens (max ⊕ mean)."""
        d = ev_embed.shape[1]
        idx = gtok[:, None].expand(-1, d)
        tmax = ev_embed.new_full((B * Ntok, d), -1e4).scatter_reduce_(
            0, idx, ev_embed, reduce="amax", include_self=True)
        tsum = ev_embed.new_zeros(B * Ntok, d).scatter_add_(0, idx, ev_embed)
        cnt = ev_embed.new_zeros(B * Ntok, 1).index_add_(
            0, gtok, ev_embed.new_ones(ev_embed.shape[0], 1))
        occ = (cnt[:, 0] > 0)
        tmax = torch.where(occ[:, None], tmax, torch.zeros_like(tmax))
        tmean = tsum / cnt.clamp_min(1.0)
        tok = self.token_proj(torch.cat([tmax, tmean], dim=1))
        tok = torch.where(occ[:, None], tok, torch.zeros_like(tok))
        return tok.view(B, Ntok, d), occ.view(B, Ntok)

    def _pos_emb(self, Hs, Ws, device):
        """Factorized PE for every (t_bin, gy, gx) token -> ``(1, Ntok, d)``."""
        T = self.time_bins
        t_ix = torch.arange(T, device=device).view(T, 1, 1).expand(T, Hs, Ws).reshape(-1)
        gy_ix = torch.arange(Hs, device=device).view(1, Hs, 1).expand(T, Hs, Ws).reshape(-1)
        gx_ix = torch.arange(Ws, device=device).view(1, 1, Ws).expand(T, Hs, Ws).reshape(-1)
        pe = self.pe_t(t_ix) + self.pe_gy(gy_ix) + self.pe_gx(gx_ix)
        return pe.unsqueeze(0)

    def _grid_t(self, Hs, Ws, device):
        """Time-bin index of every token -> ``(Ntok,)`` (encoded in the token id)."""
        S = Hs * Ws
        return (torch.arange(self.time_bins * S, device=device) // S)

    def _causal_mask(self, grid_t):
        """Additive ``(1+Ntok, 1+Ntok)`` mask: token i attends token j iff
        ``grid_t[j] < grid_t[i]`` (strictly earlier bin); the anchor (row/col 0) is
        always visible and attends only itself. Guarantees every row has a valid key."""
        Ntok = grid_t.numel()
        key = (Ntok, int(grid_t.max().item()) if Ntok else 0)
        cached = self._mask_cache.get(key)
        if cached is None or cached.shape[0] != Ntok + 1:
            tt = grid_t[:, None].gt(grid_t[None, :])            # (Ntok,Ntok) i attends j
            allow = torch.zeros(Ntok + 1, Ntok + 1, dtype=torch.bool)
            allow[0, 0] = True                                  # anchor -> self
            allow[1:, 0] = True                                 # tokens -> anchor
            allow[1:, 1:] = tt.cpu()
            cached = ~allow                                     # bool: True = NOT allowed
            self._mask_cache[key] = cached
        return cached.to(grid_t.device)

    @staticmethod
    def _safe_pad(pad):
        allmask = pad.all(dim=1)
        if bool(allmask.any()):
            pad = pad.clone()
            pad[allmask, 0] = False
        return pad

    def _encode(self, tf, tok, pe, occ, attn_mask=None):
        """Run a transformer over [anchor, tok+pe]; the anchor col is never padded."""
        B, Ntok, d = tok.shape
        x = torch.cat([self.anchor.expand(B, 1, d), tok + pe], dim=1)
        pad = torch.cat([occ.new_zeros(B, 1, dtype=torch.bool), ~occ], dim=1)
        out = tf(x, mask=attn_mask, src_key_padding_mask=self._safe_pad(pad))
        return out[:, 0], out[:, 1:]                            # anchor(B,d), tokens(B,Ntok,d)

    @torch.no_grad()
    def _ema_update(self):
        m = self.ema_momentum
        for pc, pt in zip(self.ctx_tf.parameters(), self.target_tf.parameters()):
            pt.mul_(m).add_(pc.detach(), alpha=1.0 - m)

    def _crease(self, tok_flow, occ, B, Hs, Ws):
        """Orientation-discontinuity (motion-boundary) score on the token grid."""
        with torch.no_grad():
            T = self.time_bins
            flow = tok_flow.view(B * T, Hs, Ws, 2).permute(0, 3, 1, 2)          # (BT,2,Hs,Ws)
            occm = occ.view(B * T, 1, Hs, Ws).float()
            k = torch.ones(1, 1, 3, 3, device=tok_flow.device)
            wflow = (flow * occm)
            kf = k.expand(2, 1, 3, 3)
            nbr_sum = F.conv2d(wflow, kf, padding=1, groups=2) - wflow            # exclude center
            nbr_cnt = (F.conv2d(occm, k, padding=1) - occm).clamp_min(1.0)
            mean_nbr = nbr_sum / nbr_cnt
            crease = ((flow - mean_nbr) * occm).pow(2).sum(1, keepdim=True).sqrt()  # (BT,1,Hs,Ws)
            crease = crease.permute(0, 2, 3, 1).reshape(B, T * Hs * Ws)
            return torch.nan_to_num(crease)

    def _residual(self, tok, occ, ctx, tok_flow, grid_t, pe):
        """Causal directional + latent residual per token -> ``(r_dir, r_lat) (B,Ntok)``."""
        B, Ntok, d = tok.shape
        cmask = self._causal_mask(grid_t)
        pin = self.pred_in(self.mask_token.expand(B, Ntok, d))
        _, pred = self._encode(self.pred_tf, pin, pe, occ, attn_mask=cmask)
        pred = self.pred_out(pred)
        # latent residual (secondary) — ctx detached so the head can't corrupt the rep.
        r_lat = torch.linalg.vector_norm(pred - ctx.detach(), dim=-1)
        # directional residual (primary) — angular mismatch of predicted vs observed flow.
        pf = self.flow_head(pred)
        pf = pf / pf.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        of_n = tok_flow.norm(dim=-1, keepdim=True)
        of = tok_flow / of_n.clamp_min(self.eps)
        r_dir = (1.0 - (pf * of.detach()).sum(-1)) * (of_n.squeeze(-1) > self.eps).float()
        return torch.nan_to_num(r_dir), torch.nan_to_num(r_lat)

    def _res_map(self, r_dir, r_lat, occ):
        """Standardized per-token residual channel map ``(B, Ntok, 3)``."""
        occf = occ.float()
        denom = occf.sum(1, keepdim=True).clamp_min(1.0)
        mean = (r_lat * occf).sum(1, keepdim=True) / denom
        var = (((r_lat - mean) ** 2) * occf).sum(1, keepdim=True) / denom
        z = (r_lat - mean.detach()) / (var.detach().sqrt() + 1e-4)
        return torch.stack([r_dir, torch.log1p(r_lat.clamp_min(0.0)), z], dim=-1)  # (B,Ntok,3)

    def _bilinear_sample(self, flat, x, y, t_bin, bidx, Hs, Ws):
        """Bilinearly sample a per-token map ``flat (B*T*Hs*Ws, C)`` at each event's
        CONTINUOUS spatial position within its time bin — smooth context blended across
        the 4 neighbour cells. Replaces the nearest-token gather so per-event predictions
        vary smoothly inside a patch instead of snapping to the coarse grid (kills the
        square-boundary artifact). Index math matches ``gtok`` exactly."""
        ps = self.patch_size
        T = self.time_bins
        fy = y.float() / ps - 0.5
        fx = x.float() / ps - 0.5
        y0 = torch.floor(fy); x0 = torch.floor(fx)
        wy = (fy - y0).clamp(0.0, 1.0).unsqueeze(1)
        wx = (fx - x0).clamp(0.0, 1.0).unsqueeze(1)
        y0 = y0.long(); x0 = x0.long()
        y0c = y0.clamp(0, Hs - 1); y1c = (y0 + 1).clamp(0, Hs - 1)
        x0c = x0.clamp(0, Ws - 1); x1c = (x0 + 1).clamp(0, Ws - 1)
        base = (bidx * T + t_bin) * Hs
        at = lambda yi, xi: flat.index_select(0, (base + yi) * Ws + xi)
        c00 = at(y0c, x0c); c01 = at(y0c, x1c); c10 = at(y1c, x0c); c11 = at(y1c, x1c)
        top = c00 * (1 - wx) + c01 * wx
        bot = c10 * (1 - wx) + c11 * wx
        return top * (1 - wy) + bot * wy

    # ------------------------------------------------------------------ pretext
    def _sample_targets(self, occ, grid_t):
        """Target token mask. ``future_all``: all occupied tokens in bins >= a random
        present split bin (full-frame future prediction). ``tube_time``: same but only a
        random ~half of the columns. ``space``: random occupied patches (the old mode)."""
        B, Ntok = occ.shape
        tgt = torch.zeros_like(occ)
        S = Ntok // self.time_bins
        for b in range(B):
            occb = occ[b]
            idx = occb.nonzero(as_tuple=False).flatten()
            if idx.numel() < 2:
                continue
            if self.mask_mode in ("future_all", "tube_time"):
                bins = torch.unique(grid_t[idx])
                if bins.numel() < 2:
                    continue
                cand = bins[bins > bins.min()]
                s = int(cand[torch.randint(len(cand), (1,), device=occ.device)].item())
                m = occb & (grid_t >= s)
                if self.mask_mode == "tube_time":
                    col = (torch.arange(Ntok, device=occ.device) % S)
                    keep_col = torch.rand(S, device=occ.device) < 0.5
                    m = m & keep_col[col]
                if m.any() and (occb & ~m).any():
                    tgt[b] = m
            else:   # space
                n = idx.numel()
                n_tgt = max(1, min(n - 1, int(round(self.jepa_mask_ratio * n))))
                perm = torch.randperm(n, device=occ.device)[:n_tgt]
                tgt[b, idx[perm]] = True
        return tgt

    def _touch_predictor(self):
        """Zero-weight pass touching every predictor param (DDP bucket parity)."""
        z = self.pred_out(self.pred_tf(self.pred_in(self.mask_token) + self.anchor))
        return z.sum() * 0.0 + self.flow_head(z).sum() * 0.0

    def _jepa(self, tok, occ, pe, grid_t, tok_flow):
        """Train-only full-frame future latent + flow prediction loss."""
        B, Ntok, d = tok.shape
        tgt_mask = self._sample_targets(occ, grid_t)
        if not bool(tgt_mask.any()):
            return self._touch_predictor()
        ctx_occ = occ & ~tgt_mask                              # context = visible past
        _, ctx = self._encode(self.ctx_tf, tok, pe, ctx_occ)
        pin = torch.where(tgt_mask[..., None], self.mask_token.expand(B, Ntok, d),
                          self.pred_in(ctx))
        _, pred = self._encode(self.pred_tf, pin, pe, occ)
        pred = self.pred_out(pred)
        with torch.no_grad():
            _, tgt = self._encode(self.target_tf, tok, pe, occ)
        sel = tgt_mask
        p = pred[sel]; zt = tgt[sel].detach()
        loss = F.smooth_l1_loss(p, zt)
        # directional-flow prediction (structured residual the head will reuse).
        if self.flow_loss_weight > 0.0:
            pf = self.flow_head(pred)[sel]
            pf = pf / pf.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            of_n = tok_flow.norm(dim=-1, keepdim=True)
            of = (tok_flow / of_n.clamp_min(self.eps))[sel]
            w = (of_n[sel].squeeze(-1) > self.eps).float()
            if w.sum() > 0:
                loss = loss + self.flow_loss_weight * ((1.0 - (pf * of.detach()).sum(-1)) * w).sum() / w.sum().clamp_min(1.0)
        if self.var_weight > 0.0 and p.shape[0] >= 2:
            std = torch.sqrt(p.var(dim=0, unbiased=False) + 1e-4)
            loss = loss + self.var_weight * F.relu(1.0 - std).mean()
        return loss

    # ------------------------------------------------------------------ forward
    def _featurize(self, x, y, times, feats, batch_idx, B, H, W):
        """Shared front-end: events -> (ev_embed, gtok, tok, occ, tok_flow, pe, grid_t)."""
        Hs, Ws, Ntok = self._grid(H, W)
        if Ntok > self.max_tokens:
            raise RuntimeError(f"{Ntok} tokens exceed max_tokens={self.max_tokens}")
        t_bin = (times.float() * self.time_bins).floor().clamp(0, self.time_bins - 1).long()
        ev, mg = self._event_feats(x, y, times, feats, t_bin, batch_idx, B, H, W)
        ev_embed = self.event_mlp(ev)
        gy = (y.long() // self.patch_size).clamp(0, Hs - 1)
        gx = (x.long() // self.patch_size).clamp(0, Ws - 1)
        gtok = batch_idx.long() * Ntok + (t_bin * Hs + gy) * Ws + gx
        tok, occ = self._tokens(ev_embed, gtok, B, Ntok)
        tok_flow = self._scatter_mean(mg[:, :2], gtok, B * Ntok).view(B, Ntok, 2)
        pe = self._pos_emb(Hs, Ws, x.device)
        grid_t = self._grid_t(Hs, Ws, x.device)
        return dict(ev_embed=ev_embed, gtok=gtok, tok=tok, occ=occ, tok_flow=tok_flow,
                    pe=pe, grid_t=grid_t, Ntok=Ntok, Hs=Hs, Ws=Ws, t_bin=t_bin)

    def forward(self, batch) -> torch.Tensor:
        self._jepa_loss = self._presence_logit = self._aux_logits = self._event_embedding = None
        self._res_ev = self._crease_ev = None      # diagnostics (kill-switch / monitoring)
        feats = batch.feats
        N = feats.shape[0]
        if N == 0:
            return feats.new_zeros((0,) if self.num_classes == 1 else (0, self.num_classes))
        H, W = int(batch.height), int(batch.width)
        B = batch.batch_size
        x = batch.coords[:, 0]; y = batch.coords[:, 1]; times = batch.times
        bidx = batch.batch_idx.long()

        f = self._featurize(x, y, times, feats, bidx, B, H, W)
        ev_embed, gtok, tok, occ = f["ev_embed"], f["gtok"], f["tok"], f["occ"]
        tok_flow, pe, grid_t, Ntok, Hs, Ws = (f["tok_flow"], f["pe"], f["grid_t"],
                                              f["Ntok"], f["Hs"], f["Ws"])
        t_bin = f["t_bin"]

        # JEPA pretext (train only).
        if self.training and self.jepa_weight > 0.0:
            self._jepa_loss = self._jepa(tok, occ, pe, grid_t, tok_flow)
            self._ema_update()

        # Context encode (the inference path).
        ctx_anchor, ctx = self._encode(self.ctx_tf, tok, pe, occ)

        # Causal directional + latent residual (train AND inference) — bilinearly
        # sampled at each event so the residual varies smoothly within a patch.
        if self.residual_feature:
            r_dir, r_lat = self._residual(tok, occ, ctx, tok_flow, grid_t, pe)
            if not self.dir_residual:
                r_dir = torch.zeros_like(r_dir)
            res_map = self._res_map(r_dir, r_lat, occ)                       # (B,Ntok,3)
            res_ev = self._bilinear_sample(res_map.reshape(B * Ntok, 3), x, y, t_bin, bidx, Hs, Ws)
        else:
            res_ev = ev_embed.new_zeros(N, 3)

        # Crease feature (bilinearly sampled).
        if self.crease_feature:
            crease = self._crease(tok_flow, occ, B, Hs, Ws)                  # (B,Ntok)
            crease_ev = self._bilinear_sample(crease.reshape(B * Ntok, 1), x, y, t_bin, bidx, Hs, Ws)
        else:
            crease_ev = ev_embed.new_zeros(N, 1)
        self._res_ev, self._crease_ev = res_ev, crease_ev

        # Presence gate.
        if self.presence_head is not None:
            occf = occ.float()
            gdesc = (ctx * occf.unsqueeze(-1)).sum(1) / occf.sum(1, keepdim=True).clamp_min(1.0)
            self._presence_logit = self.presence_head(gdesc).squeeze(-1)

        # Aux occupancy head (train only).
        if self.aux_head is not None and self.training:
            occf = occ.float().view(B, self.time_bins, Hs, Ws)
            al = self.aux_head(ctx).squeeze(-1).view(B, self.time_bins, Hs, Ws)
            num = (al * occf).sum(1)
            den = occf.sum(1).clamp_min(1.0)
            amap = (num / den).unsqueeze(1)                    # (B,1,Hs,Ws)
            self._aux_logits = F.interpolate(amap, size=(self.aux_grid, self.aux_grid),
                                             mode="bilinear", align_corners=False)

        # Per-event head — bilinear context (not nearest-token) for sub-patch boundaries.
        ev_ctx = self.feat_drop(self._bilinear_sample(
            ctx.reshape(B * Ntok, self.dim), x, y, t_bin, bidx, Hs, Ws))
        head_in = torch.cat([ev_embed, ev_ctx, res_ev, crease_ev], dim=1)
        emb = self.head[:-1](head_in)
        logits = self.head[-1](emb)
        if self.training and self.null_loss_weight > 0.0:
            self._event_embedding = emb

        # Shear-invariance consistency (train only; OFF in headline -> ablation #4).
        if self.training and self.shear_consistency and self.shear_weight > 0.0 \
                and self.residual_feature:
            self._jepa_loss = (self._jepa_loss if self._jepa_loss is not None else logits.sum() * 0.0) \
                + self._shear_loss(x, y, times, feats, bidx, B, H, W, res_ev)

        # Presence gate application (logits is (N, num_classes) here).
        if self._presence_logit is not None:
            gate = self.presence_gate_scale * F.logsigmoid(self._presence_logit)
            logits = logits + gate[bidx].unsqueeze(-1)

        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits

    def _shear_loss(self, x, y, times, feats, bidx, B, H, W, res_ev):
        """Ego-motion shear-consistency: residual should be invariant to a random global
        constant-flow shear (Huber on the per-event directional residual)."""
        tt = (times.float() - 0.5)
        vx = (torch.rand(B, device=x.device) * 2 - 1) * self.shear_max * W
        vy = (torch.rand(B, device=x.device) * 2 - 1) * self.shear_max * H
        xs = (x.float() + vx[bidx] * tt).clamp(0, W - 1)
        ys = (y.float() + vy[bidx] * tt).clamp(0, H - 1)
        g = self._featurize(xs, ys, times, feats, bidx, B, H, W)
        _, ctx_s = self._encode(self.ctx_tf, g["tok"], g["pe"], g["occ"])
        r_dir_s, r_lat_s = self._residual(g["tok"], g["occ"], ctx_s, g["tok_flow"],
                                          g["grid_t"], g["pe"])
        res_map_s = self._res_map(r_dir_s, r_lat_s, g["occ"])
        res_s = self._bilinear_sample(res_map_s.reshape(B * g["Ntok"], 3),
                                      xs, ys, g["t_bin"], bidx, g["Hs"], g["Ws"])
        return self.shear_weight * F.smooth_l1_loss(res_s, res_ev.detach())

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
