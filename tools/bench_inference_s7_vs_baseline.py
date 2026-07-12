"""Forward-only inference benchmark: EventStreamSegS7 (per-event selective SSM) vs
EventSSMSegStream (per-cell grid SSM, the run_ssm_stream_all / _edge baseline).

Measures latency (ms per forward) and throughput (events/sec) at matched event
counts, batch size, image size. Uses the same synthetic SparseEventBatch builder as
the S7 smoke test, so no dataset / disk is touched.

Run (needs ONE free GPU; contends with training if all 4 are busy):
    EventBasedObjectDetection/bin/python tools/bench_inference_s7_vs_baseline.py --device cuda
CPU fallback (ratio is indicative, absolute ms is not GPU-representative):
    EventBasedObjectDetection/bin/python tools/bench_inference_s7_vs_baseline.py --device cpu --events 5000
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import time

import torch
import yaml

from data.sparse_event_collate import collate_sparse_events


def _synth_sample(n_events, H, W, gen):
    x = torch.randint(0, W, (n_events,), generator=gen).long()
    y = torch.randint(0, H, (n_events,), generator=gen).long()
    coords = torch.stack([x, y], dim=1)
    times = torch.rand(n_events, generator=gen)
    pol = torch.where(torch.rand(n_events, generator=gen) > 0.5,
                      torch.ones(n_events), -torch.ones(n_events))
    feats = torch.stack([pol, times], dim=1)
    labels = (x < W // 2).float()
    dense = torch.zeros(H, W, dtype=torch.uint8)
    dense[y, x] = labels.to(torch.uint8)
    return coords, feats, times, labels, dense, {"n_events": n_events, "n_kept": n_events}


def _batch(n_events, H, W, B, gen):
    return collate_sparse_events([_synth_sample(n_events, H, W, gen) for _ in range(B)])


def _to_device(batch, device):
    for name in ("coords", "feats", "times", "labels", "batch_idx", "dense"):
        v = getattr(batch, name, None)
        if torch.is_tensor(v):
            setattr(batch, name, v.to(device))
    return batch


def _build_model(config_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    m = cfg["MODEL"]
    mod = importlib.import_module(f"model.{m['file_name']}")
    cls = getattr(mod, m["class_name"])
    init = dict(m.get("model_init_args") or {})
    # keep only kwargs the constructor actually accepts (mirrors repo filter_init_args)
    accepted = set(inspect.signature(cls.__init__).parameters)
    init = {k: v for k, v in init.items() if k in accepted}
    # inference: no gradient-checkpointing (train-only) and eval mode
    if "use_checkpoint" in accepted:
        init["use_checkpoint"] = False
    model = cls(**init).to(device).eval()
    return model, cls.__name__, sum(p.numel() for p in model.parameters())


@torch.no_grad()
def _time_forward(model, batch, device, reps, warmup):
    cuda = device.type == "cuda"
    for _ in range(warmup):
        model(batch)
    if cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            model(batch)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps          # ms per forward
    t0 = time.perf_counter()
    for _ in range(reps):
        model(batch)
    return (time.perf_counter() - t0) * 1e3 / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--events", type=int, nargs="+", default=[5000, 15000, 30000])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--s7", default="configs/hand_event_seg_stream_s7.yaml")
    ap.add_argument("--baseline", default="configs/run_ssm_stream_all_edge.yaml")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    gen = torch.Generator().manual_seed(0)

    s7, s7_name, s7_np = _build_model(args.s7, device)
    base, base_name, base_np = _build_model(args.baseline, device)
    print(f"device={device}  batch={args.batch}  img={args.height}x{args.width}  "
          f"reps={args.reps} (warmup {args.warmup})")
    print(f"  {s7_name:20s} params {s7_np:,}")
    print(f"  {base_name:20s} params {base_np:,}\n")

    hdr = f"{'events/sample':>13} | {s7_name+' ms':>22} | {base_name+' ms':>22} | {'S7/base':>8}"
    print(hdr); print("-" * len(hdr))
    for n in args.events:
        batch = _to_device(_batch(n, args.height, args.width, args.batch, gen), device)
        try:
            t_s7 = _time_forward(s7, batch, device, args.reps, args.warmup)
        except RuntimeError as e:
            t_s7 = float("nan"); print(f"  S7 OOM/err at {n}: {str(e)[:80]}")
        try:
            t_base = _time_forward(base, batch, device, args.reps, args.warmup)
        except RuntimeError as e:
            t_base = float("nan"); print(f"  base OOM/err at {n}: {str(e)[:80]}")
        ev_s7 = args.batch * n / (t_s7 / 1e3) if t_s7 == t_s7 else float("nan")
        ev_base = args.batch * n / (t_base / 1e3) if t_base == t_base else float("nan")
        ratio = t_s7 / t_base if (t_s7 == t_s7 and t_base == t_base and t_base > 0) else float("nan")
        print(f"{n:>13,} | {t_s7:>10.2f} ({ev_s7/1e6:>5.2f} Mev/s) | "
              f"{t_base:>10.2f} ({ev_base/1e6:>5.2f} Mev/s) | {ratio:>7.1f}x")


if __name__ == "__main__":
    main()
