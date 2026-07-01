"""Koopman/SSM event segmenter: event-voxel time-bins as a Koopman snapshot sequence,
streamed by a structured-Koopman state-space model, with a DMD-style static-vs-dynamic
veto. Dense voxel -> (B,1,H,W) mask.

Grounded in the deep-research synthesis (docs/manifold_dynamics_research.md):
  * SKOLR (arXiv 2506.14113, ICML 2025) — a structured Koopman operator is EQUIVALENT to
    a linear-RNN/SSM update; so the SSM here IS the (finite-dim) Koopman approximation.
  * STREAM (2411.12603) / Event-SSM (2404.18508) — event-native SSMs are SOTA + O(N)/
    constant-cost streaming; an S5-style complex-diagonal SSM is the cleanest realization.
  * DMD background/foreground (1404.7592) + multi-resolution DMD (1506.00564) — near-zero-
    frequency "zero-modes" = static/low-rank background (vetoable); modes bounded away from
    the origin = dynamic foreground. Single-level DMD ~= robust PCA.

Design: each event-voxel of shape ``(B, K, H, W)`` is K time-binned snapshots. A shared
spatial stem encodes each slice to ``(K, c, h, w)``; a complex-DIAGONAL SSM (``_DiagSSM``,
S5/S4D-style) evolves each spatial location's K-step sequence — the imaginary parts of its
continuous eigenvalues are the learned **Koopman frequencies** (near-zero => static mode,
nonzero => dynamic). The per-location **dynamic energy** (temporal variance of the SSM
output, the cheap DMD zero-mode/RPCA proxy: static => constant output => ~0) feeds a
multiplicative **static veto** on the mask logits, directly attacking the static-window-FP
failure. A small decoder upsamples to a full-resolution mask. GroupNorm only (no BatchNorm
— running stats leak the held-out LOSO subject). Streaming, magnitude-free (the gate keys
on temporal STRUCTURE/frequency, not displacement). NOT yet trained.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int, groups: int) -> nn.GroupNorm:
    g = max(1, min(groups, c))
    while c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)


class _DiagSSM(nn.Module):
    """S5-style MIMO complex-DIAGONAL state-space model = a structured Koopman operator.

    ``h_k = Ā ⊙ h_{k-1} + B̄ x_k``  (state ``h ∈ C^N``);  ``y_k = Re(C h_k) + D x_k``.
    The continuous diagonal eigenvalues ``a = -exp(a_log_re) + i·a_im`` are stable
    (negative real part); ``Im(a) = a_im`` are the **Koopman frequencies** (≈0 => static
    mode, large => dynamic). Run as an explicit scan over the short K-step sequence.
    """

    def __init__(self, d_model: int, d_state: int = 64,
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
        M, L, H = x.shape
        dt = torch.exp(self.log_dt)                                  # (N,)
        a = -torch.exp(self.a_log_re) + 1j * self.a_im               # (N,) continuous
        Abar = torch.exp(a * dt)                                     # (N,) |.|<1
        Bbar = (self.B_re + 1j * self.B_im) * dt.unsqueeze(1)        # (N, H)
        C = self.C_re + 1j * self.C_im                               # (H, N)
        xc = x.to(torch.cfloat)
        h = torch.zeros(M, self.N, dtype=torch.cfloat, device=x.device)
        ys = []
        Bt = Bbar.transpose(0, 1)                                    # (H, N)
        Ct = C.transpose(0, 1)                                       # (N, H)
        for k in range(L):
            h = Abar.unsqueeze(0) * h + xc[:, k] @ Bt                # (M, N)
            ys.append((h @ Ct).real)                                # (M, H)
        y = torch.stack(ys, dim=1) + x * self.D                     # (M, L, H)
        return y, self.a_im


class EventSSMSeg(nn.Module):
    """Event-voxel -> Koopman/SSM over time-bins -> DMD static veto -> mask logits."""

    def __init__(
        self,
        stem_channels: Sequence[int] = (32, 64),
        d_state: int = 64,
        dec_channels: Sequence[int] = (64, 32),
        num_classes: int = 1,
        gn_groups: int = 8,
        dmd_gate: bool = True,
        dmd_gate_scale: float = 1.0,
        presence_gate: bool = True,
        presence_gate_scale: float = 1.0,
    ):
        super().__init__()
        c1, c2 = int(stem_channels[0]), int(stem_channels[1])
        self.c2 = c2
        self.num_classes = int(num_classes)
        self.dmd_gate = bool(dmd_gate)
        self.dmd_gate_scale = float(dmd_gate_scale)
        self.presence_gate = bool(presence_gate)
        self.presence_gate_scale = float(presence_gate_scale)
        self._dyn_energy = None
        self._freqs = None

        # Shared per-slice spatial stem: 1ch snapshot -> c2 features at /4.
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=2, padding=1, bias=False), _gn(c1, gn_groups), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), _gn(c2, gn_groups), nn.ReLU(inplace=True),
        )
        # Structured-Koopman SSM over the K time-bins (per spatial location).
        self.ssm = _DiagSSM(c2, d_state)
        self.ssm_norm = nn.LayerNorm(c2)

        # DMD static-veto gate: maps per-location dynamic energy -> logit shift.
        self.dyn_a = nn.Parameter(torch.tensor(4.0))
        self.dyn_b = nn.Parameter(torch.tensor(-2.0))

        # Decoder: /4 -> full res. Input = aggregated SSM state (+1 dynamic-energy ch).
        din = c2 + (1 if self.dmd_gate else 0)
        d0, d1 = int(dec_channels[0]), int(dec_channels[1])
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(din, d0, 3, padding=1, bias=False), _gn(d0, gn_groups), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(d0, d1, 3, padding=1, bias=False), _gn(d1, gn_groups), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(d1, num_classes, 1)
        self.presence_head = (nn.Sequential(nn.Linear(c2, c2), nn.ReLU(inplace=True),
                                            nn.Linear(c2, 1)) if self.presence_gate else None)

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        """``voxel (B, K, H, W)`` (K = time-bins) -> mask logits ``(B, num_classes, H, W)``."""
        B, K, H, W = voxel.shape
        x = voxel.reshape(B * K, 1, H, W)
        f = self.stem(x)                                            # (B*K, c2, h, w)
        c2, h, w = f.shape[1], f.shape[2], f.shape[3]
        # per-location K-step sequence -> SSM (the Koopman operator)
        seq = f.reshape(B, K, c2, h, w).permute(0, 3, 4, 1, 2).reshape(B * h * w, K, c2)
        y, freqs = self.ssm(seq)                                    # (B*hw, K, c2)
        y = self.ssm_norm(y)
        agg = y.mean(dim=1)                                         # (B*hw, c2) temporal readout
        # DMD-style dynamic energy: temporal variance of the SSM output (static->~0).
        dyn = y.var(dim=1, unbiased=False).mean(dim=-1, keepdim=True)   # (B*hw, 1)
        agg = agg.reshape(B, h, w, c2).permute(0, 3, 1, 2).contiguous()    # (B, c2, h, w)
        dyn = dyn.reshape(B, 1, h, w)
        self._dyn_energy, self._freqs = dyn, freqs

        feat = torch.cat([agg, dyn], dim=1) if self.dmd_gate else agg
        d = self.dec(feat)
        if d.shape[-2:] != (H, W):                                 # guard odd sizes
            d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        logits = self.head(d)                                      # (B, num_classes, H, W)

        # DMD static veto: low dynamic energy (static/predictable) -> suppress.
        if self.dmd_gate:
            dyn_up = F.interpolate(dyn, size=(H, W), mode="bilinear", align_corners=False)
            logits = logits + self.dmd_gate_scale * F.logsigmoid(self.dyn_a * dyn_up + self.dyn_b)

        # Per-window presence gate (global readout).
        if self.presence_head is not None:
            g = agg.mean(dim=(2, 3))                               # (B, c2)
            pl = self.presence_head(g)                             # (B, 1)
            logits = logits + self.presence_gate_scale * F.logsigmoid(pl).view(B, 1, 1, 1)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
