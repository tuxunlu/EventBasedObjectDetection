"""CPU smoke test for ``model.event_jepa_frame.EventJEPAFrame`` (dense voxel->mask
EventUnet + train-only I-JEPA bottleneck pretext). No spconv, no GPU.

Run: EventBasedObjectDetection/bin/python tools/smoke_test_event_jepa_frame.py
"""

from __future__ import annotations

import sys

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from model.event_jepa_frame import EventJEPAFrame

FAILED = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def main():
    torch.manual_seed(0)
    B, C, H, W = 2, 5, 120, 160
    model = EventJEPAFrame(in_channels=C, encoder_channels=(48, 96, 128, 160), norm="gn")
    npar = model.count_parameters()
    infer = sum(p.numel() for n, p in model.named_parameters()
                if not n.startswith(("predictor", "in_proj", "out_proj", "mask_token",
                                     "pe_row", "pe_col", "target_")))
    print(f"== build ==  total {npar:,} | inference-only {infer:,}")
    _check("param budget sane (<3M)", npar < 3_000_000, f"{npar:,}")

    x = torch.randn(B, C, H, W)

    print("== 1. eval: voxel -> mask logits, no JEPA in path ==")
    model.eval()
    with torch.no_grad():
        y = model(x)
    _check("mask logits (B,1,H,W)", tuple(y.shape) == (B, 1, H, W), str(tuple(y.shape)))
    _check("logits finite", torch.isfinite(y).all().item())
    _check("_jepa_loss None at eval", model._jepa_loss is None)

    print("== 2. train: _jepa_loss is a finite scalar with grad ==")
    model.train()
    y = model(x)
    _check("train mask shape (B,1,H,W)", tuple(y.shape) == (B, 1, H, W))
    jl = model._jepa_loss
    _check("_jepa_loss finite scalar requires_grad",
           jl is not None and jl.dim() == 0 and torch.isfinite(jl).item() and jl.requires_grad,
           None if jl is None else f"{jl.item():.4f}")

    print("== 3. backward reaches encoder/decoder/predictor, NOT the frozen EMA target ==")
    ema0 = next(model._target_params()).detach().clone()
    (y.float().pow(2).mean() + model.jepa_weight * jl).backward()
    _check("head (decoder) got grad", model.head.weight.grad is not None
           and model.head.weight.grad.abs().sum() > 0)
    _check("encoder got grad", any(p.grad is not None and p.grad.abs().sum() > 0
                                   for p in model.encoder.parameters()))
    _check("predictor mask_token got grad",
           model.mask_token.grad is not None and model.mask_token.grad.abs().sum() > 0)
    _check("predictor transformer got grad",
           any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters()))
    _check("EMA target frozen (no grad)", all(p.grad is None for p in model._target_params()))

    print("== 4. EMA moves target toward online encoder ==")
    # perturb the online encoder so the EMA has something to track
    with torch.no_grad():
        for p in model._online_params():
            p.add_(torch.randn_like(p) * 0.05)
    onl = [p.detach().clone() for p in model._online_params()]
    tgt0 = [p.detach().clone() for p in model._target_params()]
    model._ema_update()
    tgt1 = [p.detach().clone() for p in model._target_params()]
    moved = sum((a - b).abs().sum().item() for a, b in zip(tgt1, tgt0))
    d0 = sum((t - o).abs().sum().item() for t, o in zip(tgt0, onl))
    d1 = sum((t - o).abs().sum().item() for t, o in zip(tgt1, onl))
    _check("EMA target moved", moved > 0, f"Δ={moved:.3e}")
    _check("EMA moved toward online", d1 < d0, f"d0={d0:.3e} d1={d1:.3e}")
    _check("EMA target changed at all (sanity vs ema0)",
           not torch.equal(ema0, next(model._target_params())))

    print("== 5. NO BatchNorm anywhere (LOSO-safe) ==")
    _check("no BatchNorm modules", not any(isinstance(m, _BatchNorm) for m in model.modules()))

    print("== 6. robustness: jepa_weight=0 + odd-ish sizes + a tiny patch grid ==")
    m0 = EventJEPAFrame(in_channels=C, jepa_weight=0.0).train()
    with torch.no_grad():
        o0 = m0(torch.randn(1, C, 96, 96))
    _check("jepa_weight=0 trains (no _jepa_loss)", o0.shape == (1, 1, 96, 96) and m0._jepa_loss is None)
    m2 = EventJEPAFrame(in_channels=C, jepa_patch=2).train()
    o2 = m2(torch.randn(2, C, 120, 160))
    _check("jepa_patch=2 ok", o2.shape == (2, 1, 120, 160) and m2._jepa_loss is not None
           and torch.isfinite(m2._jepa_loss).item())

    print()
    if FAILED:
        print(f"SMOKE TEST FAILED ({len(FAILED)}): {FAILED}")
        return 1
    print("SMOKE TEST PASSED — all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
