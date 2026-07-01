"""CPU smoke test for ``model.event_graph_seg.EventGraphSeg`` (no spconv, no GPU).

Validates the things that actually break a from-scratch event GNN:
  1. forward shape contract: ``(N,)`` logits, row-aligned to events; finite.
  2. the per-event ALIGNMENT invariant — output for event ``i`` must depend only on
     event ``i`` (its voxel + global context), so permuting the event order within a
     sample must permute the logits identically (the logits[i] <-> labels[i] contract).
  3. auxiliary attributes for ``model_interface``: ``_presence_logit (B,)``,
     ``_aux_logits (B,1,G,G)`` in train mode (absent in eval).
  4. gradient flow into every submodule, esp. the novel Blob-Affinity Diffusion seed
     and the presence head.
  5. robustness: a batch with an empty sample; a fully-empty batch.
  6. parameter count + a rough CPU forward time at a realistic event count.

Run: EventBasedObjectDetection/bin/python tools/smoke_test_event_graph.py
"""

from __future__ import annotations

import sys
import time

import torch

from data.sparse_event_collate import SparseEventBatch
from model.event_graph_seg import EventGraphSeg

H, W = 480, 640
FAILED = []


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def make_batch(counts, seed=0, device="cpu"):
    """Synthetic ``SparseEventBatch`` with ``counts[b]`` events in sample ``b``."""
    g = torch.Generator().manual_seed(seed)
    coords, feats, times, labels, bidx = [], [], [], [], []
    for b, n in enumerate(counts):
        if n == 0:
            continue
        x = torch.randint(0, W, (n,), generator=g)
        y = torch.randint(0, H, (n,), generator=g)
        t = torch.rand(n, generator=g)
        pol = (torch.randint(0, 2, (n,), generator=g).float() * 2 - 1)
        coords.append(torch.stack([x, y], dim=1))
        feats.append(torch.stack([pol, t], dim=1))
        times.append(t)
        labels.append((torch.rand(n, generator=g) > 0.6).float())
        bidx.append(torch.full((n,), b, dtype=torch.long))
    B = len(counts)
    cat = lambda xs, shp, dt: (torch.cat(xs, 0) if xs else torch.zeros(shp, dtype=dt))
    return SparseEventBatch(
        coords=cat(coords, (0, 2), torch.long).to(device),
        feats=cat(feats, (0, 2), torch.float32).to(device),
        times=cat(times, (0,), torch.float32).to(device),
        labels=cat(labels, (0,), torch.float32).to(device),
        batch_idx=cat(bidx, (0,), torch.long).to(device),
        dense_mask=torch.zeros(B, H, W, dtype=torch.uint8, device=device),
        batch_size=B, height=H, width=W, meta=None,
    )


def build(**over):
    # mirror the config defaults (configs/hand_event_seg_graph.yaml)
    kw = dict(in_features=2, stage_channels=(64, 96, 128, 192), time_bins=3,
              base_stride=4, radius_s=1, radius_t=1, causal=True, layers_per_level=2,
              blob_diffusion=True, blob_iters=8, blob_radius=2, blob_veto=True,
              presence_gate=True, aux_shape_head=True, global_token=True)
    kw.update(over)
    return EventGraphSeg(**kw)


def main():
    torch.manual_seed(0)

    print("== build ==")
    model = build()
    nparam = model.count_parameters()
    print(f"  EventGraphSeg params: {nparam:,}  ({nparam/1e6:.2f} M)")
    _check("param count in 0.3-8M range", 3e5 <= nparam <= 8e6, f"{nparam:,}")

    print("== 1/2/3. forward (train) + shape contract + aux attrs ==")
    batch = make_batch([1500, 2200], seed=1)
    model.train()
    logits = model(batch)
    N = batch.feats.shape[0]
    _check("logits shape (N,)", tuple(logits.shape) == (N,), str(tuple(logits.shape)))
    _check("logits finite", torch.isfinite(logits).all().item())
    pl = model._presence_logit
    _check("_presence_logit shape (B,)", pl is not None and tuple(pl.shape) == (2,),
           None if pl is None else str(tuple(pl.shape)))
    al = model._aux_logits
    _check("_aux_logits shape (B,1,G,G)", al is not None and al.shape[0] == 2 and al.shape[1] == 1
           and al.shape[2] == al.shape[3], None if al is None else str(tuple(al.shape)))
    _check("_aux_logits finite", al is not None and torch.isfinite(al).all().item())

    print("== alignment invariant: permuting events permutes logits ==")
    model.eval()
    with torch.no_grad():
        base = make_batch([2000], seed=7)
        l0 = model(base)
        perm = torch.randperm(base.feats.shape[0])
        pb = SparseEventBatch(
            coords=base.coords[perm], feats=base.feats[perm], times=base.times[perm],
            labels=base.labels[perm], batch_idx=base.batch_idx[perm],
            dense_mask=base.dense_mask, batch_size=1, height=H, width=W, meta=None)
        l1 = model(pb)
        max_dev = (l1 - l0[perm]).abs().max().item()
    _check("logits permutation-equivariant", max_dev < 1e-3, f"max|Δ|={max_dev:.2e}")

    print("== 4. gradient flow (train) ==")
    model.train()
    batch = make_batch([1800, 1200], seed=3)
    logits = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.labels)
    if model._presence_logit is not None:
        tgt = torch.tensor([1.0, 0.0])
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(model._presence_logit, tgt)
    if model._aux_logits is not None:
        loss = loss + model._aux_logits.mean()
    loss.backward()
    n_params = sum(1 for _ in model.parameters())
    with_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    none_grad = [n for n, p in model.named_parameters() if p.grad is None]
    _check("all params received finite grad", with_grad == n_params,
           f"{with_grad}/{n_params}; missing={none_grad[:6]}")
    # the novel modules specifically must be in the graph
    seed_grad = model.blob_seed[0].weight.grad
    pres_grad = model.presence_head[0].weight.grad
    _check("Blob-Affinity-Diffusion seed got grad", seed_grad is not None and seed_grad.abs().sum() > 0)
    _check("presence head got grad", pres_grad is not None and pres_grad.abs().sum() > 0)

    print("== 5. robustness: empty sample + fully-empty batch ==")
    model.eval()
    with torch.no_grad():
        b_mixed = make_batch([0, 1000, 0], seed=5)
        lm = model(b_mixed)
        _check("mixed batch (some empty samples) ok",
               tuple(lm.shape) == (b_mixed.feats.shape[0],) and torch.isfinite(lm).all().item())
        b_empty = make_batch([0, 0], seed=6)
        le = model(b_empty)
        _check("fully-empty batch -> (0,) logits", tuple(le.shape) == (0,))

    print("== 6. ablations construct & run (no-blob / no-presence / polarity / undirected) ==")
    for tag, over in [("no_blob", dict(blob_diffusion=False)),
                      ("no_presence", dict(presence_gate=False)),
                      ("polarity", dict(use_polarity=True)),
                      ("undirected", dict(causal=False)),
                      ("3level", dict(stage_channels=(48, 72, 128)))]:
        try:
            m = build(**over); m.eval()
            with torch.no_grad():
                o = m(make_batch([1200, 900], seed=11))
            ok = torch.isfinite(o).all().item()
        except Exception as e:                                   # noqa: BLE001
            ok = False; tag = f"{tag} ({type(e).__name__}: {e})"
        _check(f"ablation {tag}", ok)

    print("== 7. rough CPU forward time @ realistic event count ==")
    model.eval()
    big = make_batch([24000, 24000], seed=9)
    with torch.no_grad():
        model(big)                                              # warmup
        t0 = time.perf_counter()
        for _ in range(3):
            model(big)
        dt = (time.perf_counter() - t0) / 3
    print(f"  ~{big.feats.shape[0]} events, CPU forward: {dt*1e3:.1f} ms/window "
          f"(GPU/INT8 will be far lower; CPU is not the target)")

    print()
    if FAILED:
        print(f"SMOKE TEST FAILED ({len(FAILED)}): {FAILED}")
        return 1
    print("SMOKE TEST PASSED — all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
