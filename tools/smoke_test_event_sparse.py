"""Standalone smoke test for the event-native 3D sparse segmentation path.

Runs the full wiring on a tiny SYNTHETIC batch — no real dataset and no SAM 2
needed — to verify the contract before launching a training run:

  * collate -> 3D model -> per-event logits, one per EVENT (not per pixel),
  * row-alignment of logits to the events/labels they were built from,
  * the per-event loss (SCE + per-sample Lovász) runs + backward, all params get grad,
  * the parameter budget is within target,
  * (optional) a rough forward latency / FPS estimate at a realistic event count.

Requires torch + spconv (this exercises the CUDA sparse ops), so run it on the GPU
box, not in a CPU-only sandbox:

    python tools/smoke_test_event_sparse.py
    python tools/smoke_test_event_sparse.py --events 40000 --bench
"""

from __future__ import annotations

import argparse
import time

import torch

from data.sparse_event_collate import collate_sparse_events
from loss.event_distillation import EventDistillationLoss, background_prototype_loss
from model.event_sparse_seg import EventSparseSeg
from model.event_sparse_seg_gc import EventSparseSegGC
from model.event_sparse_seg_gc_gate import EventSparseSegGCGate
from model.event_sparse_seg_gc_ml import EventSparseSegGCML
from model.event_jepa_seg import EventJEPASeg


def _synth_sample(n_events: int, H: int, W: int, gen: torch.Generator):
    """One synthetic sample of individual events, mimicking HandEventStreamDataset.

    Returns ``(coords (N,2), feats (N,2)=[pol,t_norm], times (N,), labels (N,),
    dense_mask (H,W), meta)``.
    """
    x = torch.randint(0, W, (n_events,), generator=gen).long()
    y = torch.randint(0, H, (n_events,), generator=gen).long()
    coords = torch.stack([x, y], dim=1)
    times = torch.rand(n_events, generator=gen)                 # [0,1)
    pol = torch.where(torch.rand(n_events, generator=gen) > 0.5,
                      torch.ones(n_events), -torch.ones(n_events))
    feats = torch.stack([pol, times], dim=1)                    # (N,2)
    labels = (x < W // 2).float()                               # foreground = left half
    dense = torch.zeros(H, W, dtype=torch.uint8)
    dense[y, x] = labels.to(torch.uint8)
    meta = {"n_events": n_events, "n_kept": n_events}
    return coords, feats, times, labels, dense, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--events", type=int, default=4000, help="events per sample")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--in_features", type=int, default=2)
    ap.add_argument("--time_bins", type=int, default=6)
    ap.add_argument("--max_params", type=int, default=1_200_000)
    ap.add_argument("--model", choices=("ess", "gc", "gc_gate", "gc_ml", "jepa"), default="ess",
                    help="ess = EventSparseSeg (submanifold U-Net); "
                         "gc = EventSparseSegGC (+ dense global-context bottleneck); "
                         "gc_gate = +multiplicative occupancy/present veto gate (A+B); "
                         "gc_ml = +dense context at the /2 and /4 encoder levels; "
                         "jepa = EventJEPASeg (from-scratch motion-token + JEPA core, no spconv).")
    ap.add_argument("--coord_mode", default="relative",
                    help="EventSparseSegGC geometry mode: relative|absolute|both|none.")
    ap.add_argument("--algo", default="implicit_gemm",
                    help="spconv conv algo: 'native' avoids the implicit-GEMM SIGFPE "
                         "under a CUDA runtime newer than the spconv wheel.")
    ap.add_argument("--motion", action="store_true",
                    help="(model=gc) enable motion_features + the background-prototype "
                         "null loss (exercises _motion_feats and _event_embedding).")
    ap.add_argument("--motion_dir", action="store_true",
                    help="(model=gc --motion) add the 2 normal-flow direction channels.")
    ap.add_argument("--geom", action="store_true",
                    help="enable EventSparseSeg.geom_features (+normalized x,y).")
    ap.add_argument("--density", action="store_true",
                    help="enable EventSparseSeg.density_features (+local density/timing).")
    ap.add_argument("--recurrent", action="store_true",
                    help="enable the bottleneck temporal GRU (EventSparseSeg.recurrent).")
    ap.add_argument("--density_time_resolved", action="store_true",
                    help="time-resolve the density features (per t_bin, anti-smear).")
    ap.add_argument("--temporal_interp_head", action="store_true",
                    help="interpolate per-event head context across adjacent t_bins.")
    ap.add_argument("--bench", action="store_true", help="time forward passes")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("spconv needs a CUDA device; run this on the GPU box.")
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)

    samples = [_synth_sample(args.events, args.height, args.width, gen)
               for _ in range(args.batch)]
    batch = collate_sparse_events(samples).to(device)
    n_ev = batch.coords.shape[0]
    print(f"[data ] batch_size={batch.batch_size} total_events={n_ev} "
          f"feat_dim={batch.feats.shape[1]}")

    if args.model in ("gc", "gc_gate", "gc_ml"):
        gc_cls = {"gc": EventSparseSegGC, "gc_gate": EventSparseSegGCGate,
                  "gc_ml": EventSparseSegGCML}[args.model]
        gc_kwargs = dict(in_features=args.in_features, time_bins=args.time_bins,
                         num_classes=1, algo=args.algo, coord_mode=args.coord_mode,
                         density_features=args.density)
        if args.model == "gc" and args.motion:
            gc_kwargs.update(motion_features=True, motion_dir=args.motion_dir,
                             null_loss_weight=0.5, null_margin=1.0)
        model = gc_cls(**gc_kwargs).to(device)
        print(f"[model] {gc_cls.__name__} coord_mode={args.coord_mode} "
              f"density_features={args.density} context_channels={model.context_channels} "
              f"aux_shape_head={model.aux_shape_head} n_extra={model.n_extra} "
              f"motion_features={getattr(model, 'motion_features', False)}")
    elif args.model == "jepa":
        model = EventJEPASeg(in_features=args.in_features, num_classes=1,
                             jepa_weight=1.0).to(device)
        Hs, Ws, Ntok = model._grid(args.height, args.width)
        print(f"[model] EventJEPASeg patch_size={model.patch_size} dim={model.dim} "
              f"tokens={Ntok} ({Hs}x{Ws}) jepa_weight={model.jepa_weight}")
    else:
        model = EventSparseSeg(in_features=args.in_features, time_bins=args.time_bins,
                               num_classes=1, algo=args.algo,
                               geom_features=args.geom, density_features=args.density,
                               density_time_resolved=args.density_time_resolved,
                               temporal_interp_head=args.temporal_interp_head,
                               recurrent=args.recurrent).to(device)
        print(f"[model] EventSparseSeg geom_features={args.geom} "
              f"density_features={args.density} "
              f"density_time_resolved={args.density_time_resolved} "
              f"temporal_interp_head={args.temporal_interp_head} "
              f"recurrent={args.recurrent} n_extra={model.n_extra}")
    n_params = model.count_parameters()
    print(f"[model] params={n_params:,}  (budget <= {args.max_params:,})")
    assert n_params <= args.max_params, "parameter budget exceeded"

    # Forward -> one logit per EVENT.
    logits = model(batch)
    assert logits.shape[0] == n_ev, (
        f"per-event contract violated: {logits.shape[0]} logits for {n_ev} events"
    )

    # --- per-event loss (SCE + per-sample Lovász) fwd/bwd ---
    loss_fn = EventDistillationLoss(pos_weight=2.0, sce_beta=1.0, sce_alpha=0.1,
                                    lovasz_weight=1.0)
    terms = loss_fn(logits, batch.labels, batch_idx=batch.batch_idx)
    total = terms["total"]
    # The GC model's train-only aux occupancy head only gets gradient when its logits
    # are in the loss (ModelInterface supervises it against the teacher mask). Add a
    # crude aux term here so the all-params-receive-grad check covers the aux head.
    aux_logits = getattr(model, "_aux_logits", None)
    if aux_logits is not None:
        total = total + 0.2 * aux_logits.float().mean()
    # Background-prototype null loss (exercises the exposed per-event embedding).
    emb = getattr(model, "_event_embedding", None)
    if emb is not None:
        nullv = background_prototype_loss(emb, batch.labels, batch.batch_idx, margin=1.0)
        total = total + 0.5 * nullv
        print(f"[null ] embedding={tuple(emb.shape)} null_loss={nullv.item():.4f}")
    elif args.model == "gc" and args.motion:
        raise AssertionError("--motion set but model._event_embedding was not exposed")
    # JEPA latent-prediction pretext loss (EventJEPASeg, train-only).
    jl = getattr(model, "_jepa_loss", None)
    if jl is not None:
        total = total + float(getattr(model, "jepa_weight", 1.0)) * jl
        print(f"[jepa ] jepa_loss={jl.item():.4f}")
    elif args.model == "jepa":
        raise AssertionError("--model jepa but model._jepa_loss was not exposed")
    total.backward()
    n_grad = sum(int(p.grad is not None) for p in model.parameters() if p.requires_grad)
    n_train = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[loss ] total={terms['total'].item():.4f} bce={terms['bce'].item():.4f} "
          f"lovasz={terms.get('lovasz', torch.tensor(0.)).item():.4f} "
          f"params_with_grad={n_grad}/{n_train}")
    assert n_grad == n_train, "some model parameters received no gradient"

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
