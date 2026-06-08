"""Standalone smoke test for the event-native sparse segmentation path.

Runs the full wiring on a tiny SYNTHETIC batch — no real dataset and no SAM 2
needed — to verify the contract before launching a training run:

  * collate -> model -> loss -> backward executes,
  * the submanifold invariant holds (output sites == input event sites),
  * per-event logits are row-aligned to the labels they were built from,
  * the parameter budget is within target (<= 300K by default),
  * (optional) a rough forward latency / FPS estimate at a realistic event count.

Requires torch + spconv (this is what actually exercises the CUDA sparse ops), so
run it on the GPU box, not in a CPU-only sandbox:

    python tools/smoke_test_event_sparse.py
    python tools/smoke_test_event_sparse.py --events 50000 --bench
"""

from __future__ import annotations

import argparse
import time

import torch

from data.sparse_event_collate import collate_sparse_events
from loss.event_distillation import EventDistillationLoss
from model.event_sparse_seg import EventSparseSeg
from utils.metrics.event_seg import event_f1, event_pred_to_dense_iou


def _synth_sample(n_events: int, H: int, W: int, in_features: int, gen: torch.Generator):
    """One synthetic sample of unique pixel sites, mimicking HandEventStreamDataset."""
    n_pixels = H * W
    n = min(n_events, n_pixels)
    lin = torch.randperm(n_pixels, generator=gen)[:n]          # unique pixels
    x = (lin % W).long()
    y = (lin // W).long()
    coords = torch.stack([x, y], dim=1)
    feats = torch.randn(n, in_features, generator=gen)
    # A spatially coherent-ish target: foreground in the left half.
    labels = (x < W // 2).float()
    dense = torch.zeros(H, W, dtype=torch.uint8)
    dense[y, x] = labels.to(torch.uint8)
    meta = {"n_events": n, "n_sites": n}
    return coords, feats, labels, dense, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--events", type=int, default=2000, help="events per sample")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--in_features", type=int, default=3)
    ap.add_argument("--max_params", type=int, default=300_000)
    ap.add_argument("--bench", action="store_true", help="time forward passes")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("spconv needs a CUDA device; run this on the GPU box.")
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)

    samples = [
        _synth_sample(args.events, args.height, args.width, args.in_features, gen)
        for _ in range(args.batch)
    ]
    batch = collate_sparse_events(samples).to(device)
    n_sites = batch.coords.shape[0]
    print(f"[data ] batch_size={batch.batch_size} total_sites={n_sites} "
          f"feat_dim={batch.feats.shape[1]}")

    model = EventSparseSeg(in_features=args.in_features, num_classes=1).to(device)
    n_params = model.count_parameters()
    print(f"[model] params={n_params:,}  (budget <= {args.max_params:,})")
    assert n_params <= args.max_params, "parameter budget exceeded"

    # Forward / backward.
    logits = model(batch)
    assert logits.shape[0] == n_sites, (
        f"submanifold invariant violated: {logits.shape[0]} logits for {n_sites} sites"
    )
    loss_fn = EventDistillationLoss(pos_weight=2.0)
    terms = loss_fn(logits, batch.labels, batch_idx=batch.batch_idx)
    terms["total"].backward()
    n_with_grad = sum(int(p.grad is not None) for p in model.parameters() if p.requires_grad)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[bwd  ] loss={terms['total'].item():.4f}  params_with_grad="
          f"{n_with_grad}/{n_trainable}")
    assert n_with_grad == n_trainable, "some parameters received no gradient"

    # Metrics smoke.
    with torch.no_grad():
        f1 = event_f1(logits.detach(), batch.labels)
        iou = event_pred_to_dense_iou(
            batch.coords, logits.detach(), batch.batch_idx,
            batch.dense_mask, batch.batch_size, batch.height, batch.width,
        )
    print(f"[eval ] event_f1={f1.item():.4f}  rasterized_iou={iou.item():.4f}")

    if args.bench:
        model.eval()
        with torch.no_grad():
            for _ in range(10):                       # warmup
                model(batch)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            iters = 100
            for _ in range(iters):
                model(batch)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / iters * 1e3
        print(f"[bench] {ms:.3f} ms/forward  ({1e3 / ms:.1f} FPS) at "
              f"{args.events} events x {args.batch} samples")

    print("[ok   ] smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
