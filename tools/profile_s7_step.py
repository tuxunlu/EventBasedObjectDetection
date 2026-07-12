"""Decompose an EventStreamSegS7 training step to find where the 131 min/epoch goes
(research-plan R0). Times forward and forward+backward at a realistic event count/batch,
and attributes cost by ABLATION:

  * full            — the training config (dual_scan + checkpoint)
  * no_dual_scan    — full minus the z-order scan   -> (full - this) = z-scan cost
  * no_checkpoint   — full minus grad-checkpoint     -> (full - this) = recompute cost (uses more mem)
  * real_diagonal   — use_complex=False (the SSD-target form) -> complex-vs-real cost
  * compile_scan    — torch.compile the whole model  -> the "free" fused-kernel win

Also prints a torch.profiler top-op table (argsort / cat / where / mm shares).

Run on ONE gpu (won't disturb training on another node):
    CUDA_VISIBLE_DEVICES=0 EventBasedObjectDetection/bin/python tools/profile_s7_step.py --device cuda
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from data.sparse_event_collate import collate_sparse_events
from model.event_stream_seg_s7 import EventStreamSegS7

# model_init_args mirroring configs/hand_event_seg_stream_s7.yaml (the current train config)
BASE = dict(in_features=2, embed_dim=128, d_state=64, ssm_layers=2, dual_scan=True,
            morton_bits=10, time_blocks=6, fourier_bands=8, down_factor=8,
            context_channels=128, context_depth=2, head_hidden=128, coord_mode="relative",
            motion_features=True, density_features=True, dmd_gate=True, presence_gate=True,
            aux_shape_head=True, scan_weight=0.1, use_checkpoint=True, use_complex=True)


def _synth_sample(n, H, W, gen):
    x = torch.randint(0, W, (n,), generator=gen).long()
    y = torch.randint(0, H, (n,), generator=gen).long()
    coords = torch.stack([x, y], 1)
    times = torch.rand(n, generator=gen)
    pol = torch.where(torch.rand(n, generator=gen) > 0.5, torch.ones(n), -torch.ones(n))
    feats = torch.stack([pol, times], 1)
    labels = (x < W // 2).float()
    dense = torch.zeros(H, W, dtype=torch.uint8); dense[y, x] = labels.to(torch.uint8)
    return coords, feats, times, labels, dense, {"n_events": n}


def _batch(n, H, W, B, gen, device):
    b = collate_sparse_events([_synth_sample(n, H, W, gen) for _ in range(B)])
    for name in ("coords", "feats", "times", "labels", "batch_idx", "dense"):
        v = getattr(b, name, None)
        if torch.is_tensor(v):
            setattr(b, name, v.to(device))
    return b


def _time(model, batch, device, reps, warmup, backward):
    cuda = device.type == "cuda"
    def one():
        model.zero_grad(set_to_none=True)
        out = model(batch)
        if backward:
            F.binary_cross_entropy_with_logits(
                out, (batch.labels > 0.5).float()).backward()
    for _ in range(warmup):
        one()
    if cuda:
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            one()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        one()
    return (time.perf_counter() - t0) * 1e3 / reps


def _build(device, **over):
    kw = dict(BASE); kw.update(over)
    return EventStreamSegS7(**kw).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--events", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--profiler", action="store_true", help="also dump a torch.profiler op table")
    ap.add_argument("--try-compile", action="store_true",
                    help="attempt torch.compile (SLOW/may hang: the scan's data-dependent "
                         "length triggers endless recompiles; complex also fails on aten::_conj)")
    args = ap.parse_args()

    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    gen = torch.Generator().manual_seed(0)
    batch = _batch(args.events, args.height, args.width, args.batch, gen, dev)
    print(f"device={dev}  events={args.events:,}  batch={args.batch}  "
          f"img={args.height}x{args.width}  reps={args.reps}\n")

    variants = [
        ("full (train cfg)",   {}),
        ("no_dual_scan",       {"dual_scan": False}),
        ("no_checkpoint",      {"use_checkpoint": False}),
        ("real_diagonal",      {"use_complex": False}),
        ("real + no_dual",     {"use_complex": False, "dual_scan": False}),
    ]
    print(f"{'variant':20s} | {'fwd ms':>9} | {'fwd+bwd ms':>11} | {'params':>11}")
    print("-" * 62)
    results = {}
    for name, over in variants:
        try:
            m = _build(dev, **over).train()
            fwd = _time(m, batch, dev, args.reps, args.warmup, backward=False)
            fb = _time(m, batch, dev, args.reps, args.warmup, backward=True)
            results[name] = (fwd, fb)
            print(f"{name:20s} | {fwd:>9.2f} | {fb:>11.2f} | {m.count_parameters():>11,}")
            del m
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as ex:
            print(f"{name:20s} | ERR: {str(ex)[:60]}")

    # torch.compile — try on BOTH the complex (full) and the real-diagonal model.
    # inductor cannot lower complex ops (aten::_conj), so complex FAILS; real *may* compile
    # but the scan's data-dependent length triggers heavy recompilation. Off by default.
    for cname, over in ((("compile (full)", {}), ("compile (real)", {"use_complex": False}))
                        if args.try_compile else ()):
        try:
            m = _build(dev, **over).train()
            mc = torch.compile(m)
            fb = _time(mc, batch, dev, max(6, args.reps // 2), max(3, args.warmup), backward=True)
            results[cname] = (float("nan"), fb)
            print(f"{cname:20s} | {'--':>9} | {fb:>11.2f} | (torch.compile)")
            del m, mc
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as ex:  # noqa: BLE001 — report honestly, keep going
            print(f"{cname:20s} | compile FAILED: {str(ex)[:55]}")

    # Attribution
    if "full (train cfg)" in results:
        f_fb = results["full (train cfg)"][1]
        print("\nAttribution (fwd+bwd deltas vs full):")
        def delta(name):
            return f_fb - results[name][1] if name in results else float("nan")
        print(f"  z-order scan  (full - no_dual_scan)   ≈ {delta('no_dual_scan'):7.2f} ms")
        print(f"  ckpt recompute(full - no_checkpoint)  ≈ {delta('no_checkpoint'):7.2f} ms "
              f"(negative = checkpoint SAVES time here, trades memory)")
        print(f"  complex-vs-real(full - real_diagonal) ≈ {delta('real_diagonal'):7.2f} ms")
        if "compile (full)" in results:
            print(f"  torch.compile win (full - compile)    ≈ "
                  f"{f_fb - results['compile (full)'][1]:7.2f} ms")

    if args.profiler and dev.type == "cuda":
        from torch.profiler import profile, ProfilerActivity
        m = _build(dev, **{}).train()
        for _ in range(3):
            m.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(m(batch), (batch.labels > 0.5).float()).backward()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            m.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(m(batch), (batch.labels > 0.5).float()).backward()
            torch.cuda.synchronize()
        print("\nTop CUDA ops (self time):")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
