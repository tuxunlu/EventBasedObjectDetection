"""Frame-based I-JEPA segmenter: dense event-voxel -> dense mask, with a train-only
I-JEPA pretext on the encoder bottleneck.

This is the DENSE (frame) analogue of the per-event ``EventJEPASeg``
(``model/event_jepa_seg.py``). It is an ``EventUnet`` (voxel -> ``(B,1,H,W)`` mask
logits) plus a Joint-Embedding Predictive Architecture pretext (I-JEPA, Assran et al.
CVPR 2023, arXiv:2301.08243) applied to the **encoder bottleneck feature map**:

  * a frozen EMA **target encoder** (a deepcopy of ``stem`` + ``encoder``) embeds the
    voxel under stop-grad;
  * a fraction of the bottleneck **patch tokens** are masked as targets;
  * a small **predictor** transformer fills mask tokens at the masked positions and
    predicts the target encoder's latents there (smooth-L1 + a VICReg variance hinge
    against collapse).

Why feature-space masking: a conv encoder can't cleanly mask *input* patches without
edge artifacts, so I-JEPA on dense images masks the encoder feature grid. The pretext
is TRAIN-ONLY and stashed on ``self._jepa_loss`` (read by ``model_interface`` via
``jepa_weight``); at inference the predictor + EMA target are unused, so the inference
path is exactly an ``EventUnet`` (zero added cost).

Shape contract (identical to ``EventUnet``): ``forward(voxel) -> (N, num_classes, H, W)``
mask logits. GroupNorm only (no BatchNorm — running stats leak the held-out LOSO
subject).

CAVEAT (honest): on a single voxel frame, random *spatial*-patch masking is solvable by
a static-appearance "hand exists here" prior — the exact shortcut behind the measured
static-window FPs that motivated masking along TIME in the per-event model. A frame
model has no time axis at the bottleneck, so this pretext may not suppress static FPs;
block-masking / higher ``var_weight`` / the recurrent ``EventTrackUnet`` base (masking a
post-memory bottleneck) are the mitigations.
"""

from __future__ import annotations

import copy
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.event_unet import EventUnet


def _transformer(dim: int, depth: int, heads: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=dim * 2, dropout=dropout,
        activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


class EventJEPAFrame(EventUnet):
    """``EventUnet`` voxel->mask segmenter + a train-only I-JEPA bottleneck pretext."""

    def __init__(
        self,
        in_channels: int = 5,
        encoder_channels: Sequence[int] = (48, 96, 128, 160),
        num_classes: int = 1,
        norm: str = "gn",
        gn_groups: int = 8,
        use_se: bool = True,
        use_context: bool = True,
        context_dilations: Sequence[int] = (1, 2, 4),
        # ---- JEPA pretext (train-only) ----
        jepa_weight: float = 1.0,
        jepa_patch: int = 1,
        jepa_mask_ratio: float = 0.6,
        jepa_pred_dim: int = 160,
        jepa_pred_depth: int = 2,
        jepa_pred_heads: int = 4,
        jepa_dropout: float = 0.0,
        ema_momentum: float = 0.996,
        var_weight: float = 1.0,
        max_grid: int = 96,
    ):
        super().__init__(in_channels=in_channels, encoder_channels=encoder_channels,
                         num_classes=num_classes, norm=norm, gn_groups=gn_groups,
                         use_se=use_se, use_context=use_context,
                         context_dilations=context_dilations)
        self.jepa_weight = float(jepa_weight)
        self.jepa_patch = max(1, int(jepa_patch))
        self.jepa_mask_ratio = float(jepa_mask_ratio)
        self.ema_momentum = float(ema_momentum)
        self.var_weight = float(var_weight)
        self.max_grid = int(max_grid)
        self._jepa_loss = None

        cb = int(encoder_channels[-1])               # bottleneck channels
        pd = int(jepa_pred_dim)

        # EMA target encoder: frozen deepcopy of {stem, encoder} (the tap is
        # PRE-context, so the target mirrors stem+encoder only).
        self.target_stem = copy.deepcopy(self.stem)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in chain(self.target_stem.parameters(), self.target_encoder.parameters()):
            p.requires_grad_(False)

        # Predictor (train-only): project bottleneck tokens -> predictor dim, a tiny
        # transformer over the patch grid, project back to bottleneck dim.
        self.in_proj = nn.Linear(cb, pd)
        self.out_proj = nn.Linear(pd, cb)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pd))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pe_row = nn.Embedding(self.max_grid, pd)
        self.pe_col = nn.Embedding(self.max_grid, pd)
        for emb in (self.pe_row, self.pe_col):
            nn.init.trunc_normal_(emb.weight, std=0.02)
        self.predictor = _transformer(pd, jepa_pred_depth, jepa_pred_heads, jepa_dropout)

    # ------------------------------------------------------------------ helpers
    def _online_params(self):
        return chain(self.stem.parameters(), self.encoder.parameters())

    def _target_params(self):
        return chain(self.target_stem.parameters(), self.target_encoder.parameters())

    @torch.no_grad()
    def _ema_update(self):
        m = self.ema_momentum
        for pc, pt in zip(self._online_params(), self._target_params()):
            pt.mul_(m).add_(pc.detach(), alpha=1.0 - m)

    @torch.no_grad()
    def _target_encode(self, x):
        """Bottleneck feature (pre-context) from the frozen EMA encoder."""
        h = self.target_stem(x)
        for stage in self.target_encoder:
            h = stage(h)
        return h

    @staticmethod
    def _patchify(feat, pp):
        """``(B,C,Hb,Wb)`` -> tokens ``(B, Hp*Wp, C)`` (avg-pool by ``pp`` if >1)."""
        if pp > 1:
            feat = F.avg_pool2d(feat, pp)
        B, C, Hp, Wp = feat.shape
        return feat.flatten(2).transpose(1, 2), Hp, Wp        # (B, Np, C)

    def _pos_emb(self, Hp, Wp, device):
        r = torch.arange(Hp, device=device).clamp(max=self.max_grid - 1)
        c = torch.arange(Wp, device=device).clamp(max=self.max_grid - 1)
        pe = self.pe_row(r)[:, None, :] + self.pe_col(c)[None, :, :]   # (Hp,Wp,pd)
        return pe.reshape(1, Hp * Wp, -1)

    def _sample_mask(self, B, Np, device):
        m = torch.zeros(B, Np, dtype=torch.bool, device=device)
        if Np < 2:
            return m
        n_tgt = max(1, min(Np - 1, int(round(self.jepa_mask_ratio * Np))))
        for b in range(B):
            idx = torch.randperm(Np, device=device)[:n_tgt]
            m[b, idx] = True
        return m

    def _touch_predictor(self):
        """Zero-weight pass touching every predictor-side param (DDP bucket parity)."""
        x = self.out_proj(self.predictor(self.in_proj(self.mask_token)))
        return x.sum() * 0.0 + self.pe_row.weight.sum() * 0.0 + self.pe_col.weight.sum() * 0.0

    def _jepa(self, feat, voxel):
        """Train-only I-JEPA latent-prediction loss on the bottleneck feature map."""
        pp = self.jepa_patch
        Z, Hp, Wp = self._patchify(feat, pp)                  # (B, Np, Cb)
        B, Np, Cb = Z.shape
        with torch.no_grad():
            Zt, _, _ = self._patchify(self._target_encode(voxel), pp)
            Zt = Zt.detach()
        m = self._sample_mask(B, Np, feat.device)
        if not bool(m.any()):
            return self._touch_predictor()
        pe = self._pos_emb(Hp, Wp, feat.device)
        ctx = self.in_proj(Z)
        pin = torch.where(m.unsqueeze(-1), self.mask_token.expand(B, Np, -1), ctx) + pe
        pred = self.out_proj(self.predictor(pin))             # (B, Np, Cb)
        p = pred[m]
        zt = Zt[m].detach()
        loss = F.smooth_l1_loss(p, zt)
        if self.var_weight > 0.0 and p.shape[0] >= 2:
            std = torch.sqrt(p.var(dim=0, unbiased=False) + 1e-4)
            loss = loss + self.var_weight * F.relu(1.0 - std).mean()
        return loss

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """voxel ``(N, C, H, W)`` -> mask logits ``(N, num_classes, H, W)``."""
        self._jepa_loss = None
        h = self.stem(x)
        skips = []
        for stage in self.encoder:
            h = stage(h)
            skips.append(h)
        # JEPA taps the bottleneck BEFORE the context module (train only).
        if self.training and self.jepa_weight > 0.0:
            self._jepa_loss = self._jepa(skips[-1], x)
            self._ema_update()
        # Inference path is exactly EventUnet: context on the bottleneck + decode.
        skips[-1] = self.context(skips[-1])
        return self._decode(skips)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
