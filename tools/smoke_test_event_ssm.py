"""CPU smoke test for ``model.event_ssm_seg.EventSSMSeg`` (Koopman/SSM over voxel
time-bins + DMD static-vs-dynamic veto). No spconv, no GPU.

Run: EventBasedObjectDetection/bin/python tools/smoke_test_event_ssm.py
"""

from __future__ import annotations

import sys

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from model.event_ssm_seg import EventSSMSeg

FAILED = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def main():
    torch.manual_seed(0)
    B, K, H, W = 2, 16, 120, 160
    model = EventSSMSeg(stem_channels=(32, 64), d_state=64, dec_channels=(64, 32))
    npar = model.count_parameters()
    print(f"== build ==  params {npar:,}")
    _check("param budget sane (<5M)", npar < 5_000_000, f"{npar:,}")

    print("== 1. voxel -> mask logits contract ==")
    vox = torch.randn(B, K, H, W)
    model.eval()
    with torch.no_grad():
        y = model(vox)
    _check("mask logits (B,1,H,W)", tuple(y.shape) == (B, 1, H, W), str(tuple(y.shape)))
    _check("logits finite", torch.isfinite(y).all().item())
    _check("dyn-energy + freqs exposed", model._dyn_energy is not None and model._freqs is not None
           and tuple(model._freqs.shape) == (64,))

    print("== 2. gradient flow into all params (esp. SSM, gate, presence) ==")
    model.train()
    y = model(vox)
    mask = (torch.rand(B, 1, H, W) > 0.6).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y, mask)
    loss.backward()
    n_tr = sum(1 for p in model.parameters() if p.requires_grad)
    n_gr = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None
               and torch.isfinite(p.grad).all())
    miss = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    _check("all params finite grad", n_gr == n_tr, f"{n_gr}/{n_tr}; missing={miss[:5]}")
    _check("SSM eigen params got grad", model.ssm.a_im.grad is not None
           and model.ssm.a_im.grad.abs().sum() > 0 and model.ssm.B_re.grad.abs().sum() > 0)
    _check("DMD gate params got grad", model.dyn_a.grad is not None and model.dyn_a.grad.abs() >= 0
           and model.dyn_b.grad is not None)
    _check("presence head got grad", model.presence_head[0].weight.grad is not None
           and model.presence_head[0].weight.grad.abs().sum() > 0)

    print("== 3. DMD static-veto works: dynamic voxel >> dynamic-energy than static ==")
    model.eval()
    with torch.no_grad():
        slc = torch.randn(B, 1, H, W)
        static_vox = slc.repeat(1, K, 1, 1)                  # same slice every bin -> static
        dynamic_vox = torch.randn(B, K, H, W)                # changes every bin -> dynamic
        model(static_vox); e_static = model._dyn_energy.mean().item()
        model(dynamic_vox); e_dyn = model._dyn_energy.mean().item()
    _check("dynamic energy(dynamic) > dynamic energy(static)", e_dyn > e_static,
           f"static={e_static:.4f} dynamic={e_dyn:.4f}")

    print("== 4. NO BatchNorm (LOSO-safe) ==")
    _check("no BatchNorm modules", not any(isinstance(m, _BatchNorm) for m in model.modules()))

    print("== 5. robustness: K=1, odd sizes, gates off ==")
    with torch.no_grad():
        o1 = model(torch.randn(1, 1, 96, 96))                # K=1 (var unbiased=False -> 0, no NaN)
    _check("K=1 ok (no NaN)", o1.shape == (1, 1, 96, 96) and torch.isfinite(o1).all().item())
    m_off = EventSSMSeg(dmd_gate=False, presence_gate=False).train()
    o_off = m_off(torch.randn(2, 8, 104, 152))
    _check("gates-off + odd size ok", o_off.shape == (2, 1, 104, 152) and torch.isfinite(o_off).all().item())

    print()
    if FAILED:
        print(f"SMOKE TEST FAILED ({len(FAILED)}): {FAILED}")
        return 1
    print("SMOKE TEST PASSED — all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
