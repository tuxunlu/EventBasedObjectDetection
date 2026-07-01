"""CPU smoke test for the rewritten ``model.event_jepa_seg.EventJEPASeg``
(Manifold-JEPA: time-forward latent prediction + directional residual + crease).

No spconv, no GPU, no real dataset. Validates the things that actually break this
model (plan §6):
  1. per-event ``(N,)`` contract + finiteness; train-only attrs populate.
  2. the ALIGNMENT invariant — permuting events permutes logits (logits[i]<->event i).
  3. gradient flow into EVERY trainable param, esp. the predictor, ``flow_head``,
     ``anchor``, ``mask_token`` and the manifold-input columns of ``event_mlp``.
  4. the causal directional residual is LIVE at inference (perturb ``pred_tf`` -> logits
     change) and finite; manifold path drives crease/residual.
  5. EMA moves the target encoder toward the context encoder.
  6. eval drops the JEPA/aux branch but keeps the residual (predictor) pass.
  7. DDP-amplified rare-batch NaN cases: empty / 1-event / all-one-t_bin / fully-empty /
     empty-SSL-target all stay finite (no softmax(all -inf), no var-of-1 NaN).
  8. parameter budget; ablations construct & run.
  9. discriminability probe (informational kill-switch): crease / r_dir on hand vs
     no-hand events (untrained -> expected to overlap; the real gate runs post-train).

Run:  EventBasedObjectDetection/bin/python tools/smoke_test_event_jepa.py
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

from data.sparse_event_collate import SparseEventBatch
from model.event_jepa_seg import EventJEPASeg

H, W = 480, 640
FAILED = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def make_batch(counts, seed=0, t_fixed=None, hand=False, device="cpu"):
    """Synthetic SparseEventBatch. ``t_fixed`` pins all times (all-one-t_bin case);
    ``hand`` plants a coherent moving cluster (label 1) in a sea of scattered bg."""
    g = torch.Generator().manual_seed(seed)
    coords, feats, times, labels, bidx = [], [], [], [], []
    for b, n in enumerate(counts):
        if n == 0:
            continue
        if hand:
            nh = n // 3
            t = torch.rand(n, generator=g)
            # hand: compact cluster that translates with time (a gliding sheet).
            cx = 200 + 120 * t[:nh]; cy = 240 + 60 * t[:nh]
            hx = (cx + torch.randn(nh, generator=g) * 12).clamp(0, W - 1)
            hy = (cy + torch.randn(nh, generator=g) * 12).clamp(0, H - 1)
            bx = torch.randint(0, W, (n - nh,), generator=g).float()
            by = torch.randint(0, H, (n - nh,), generator=g).float()
            x = torch.cat([hx, bx]); y = torch.cat([hy, by])
            lab = torch.cat([torch.ones(nh), torch.zeros(n - nh)])
        else:
            x = torch.randint(0, W, (n,), generator=g).float()
            y = torch.randint(0, H, (n,), generator=g).float()
            t = torch.rand(n, generator=g)
            lab = (torch.rand(n, generator=g) > 0.6).float()
        if t_fixed is not None:
            t = torch.full((n,), float(t_fixed))
        pol = (torch.randint(0, 2, (n,), generator=g).float() * 2 - 1)
        coords.append(torch.stack([x.long().clamp(0, W - 1), y.long().clamp(0, H - 1)], 1))
        feats.append(torch.stack([pol, t], dim=1))
        times.append(t); labels.append(lab)
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
        batch_size=B, height=H, width=W, meta=None)


def build(**over):
    # Mirror config hyperparams; reduce the grid for CPU speed (param count is grid-
    # independent thanks to factorized PE, so the budget check stays faithful).
    kw = dict(in_features=2, num_classes=1, patch_size=64, time_bins=4, dim=64,
              depth=4, heads=4, pred_depth=2, mask_mode="future_all",
              crease_feature=True, residual_feature=True, dir_residual=True,
              presence_gate=True, aux_shape_head=True, jepa_weight=0.5)
    kw.update(over)
    return EventJEPASeg(**kw)


def main():
    torch.manual_seed(0)

    print("== build ==")
    model = build()
    npar = model.count_parameters()
    print(f"  EventJEPASeg trainable params: {npar:,} ({npar/1e6:.3f} M)")
    # Lenient cap: capacity is a config knob (dim/depth/head_hidden); allow up to ~8M so
    # 10x-scale variants pass the smoke. (Generalization, not capacity, is the LOSO limiter.)
    _check("param budget <= 8M", npar <= 8_000_000, f"{npar:,}")

    print("== 1. forward (train) + contract + train-only attrs ==")
    model.train()
    batch = make_batch([1500, 2200], seed=1)
    logits = model(batch)
    N = batch.feats.shape[0]
    _check("logits shape (N,)", tuple(logits.shape) == (N,), str(tuple(logits.shape)))
    _check("logits finite", torch.isfinite(logits).all().item())
    jl = model._jepa_loss
    _check("_jepa_loss scalar finite", jl is not None and jl.dim() == 0 and torch.isfinite(jl).item(),
           None if jl is None else f"{jl.item():.4f}")
    pl = model._presence_logit
    _check("_presence_logit (B,) finite", pl is not None and tuple(pl.shape) == (2,) and torch.isfinite(pl).all())
    al = model._aux_logits
    _check("_aux_logits (B,1,G,G) finite",
           al is not None and al.shape[0] == 2 and al.shape[1] == 1 and al.shape[2] == al.shape[3]
           and torch.isfinite(al).all(), None if al is None else str(tuple(al.shape)))

    print("== 2. alignment: permuting events permutes logits ==")
    model.eval()
    with torch.no_grad():
        base = make_batch([2000], seed=7)
        l0 = model(base)
        perm = torch.randperm(base.feats.shape[0])
        pb = SparseEventBatch(coords=base.coords[perm], feats=base.feats[perm],
                              times=base.times[perm], labels=base.labels[perm],
                              batch_idx=base.batch_idx[perm], dense_mask=base.dense_mask,
                              batch_size=1, height=H, width=W, meta=None)
        l1 = model(pb)
        dev = (l1 - l0[perm]).abs().max().item()
    _check("logits permutation-equivariant", dev < 1e-4, f"max|Δ|={dev:.2e}")

    print("== 3. gradient flow into all params (+ predictor/flow/manifold) ==")
    model.train()
    # dense coherent (hand) clusters so the local plane-fit produces non-zero flow —
    # otherwise flow_head correctly receives zero grad on sparse random events.
    batch = make_batch([5000, 5000], seed=3, hand=True)
    logits = model(batch)
    valid = batch.labels >= 0
    loss = F.binary_cross_entropy_with_logits(logits[valid], batch.labels[valid].clamp(0, 1))
    loss = loss + model._jepa_loss
    tgt = torch.tensor([1.0, 0.0])
    loss = loss + F.binary_cross_entropy_with_logits(model._presence_logit, tgt)
    loss = loss + model._aux_logits.mean()
    loss.backward()
    n_tr = sum(1 for p in model.parameters() if p.requires_grad)
    n_gr = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None
               and torch.isfinite(p.grad).all())
    miss = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    _check("all params received finite grad", n_gr == n_tr, f"{n_gr}/{n_tr}; missing={miss[:4]}")
    fh = model.flow_head.weight.grad
    _check("flow_head got grad", fh is not None and fh.abs().sum() > 0)
    ag = model.anchor.grad
    _check("anchor got grad", ag is not None and ag.abs().sum() > 0)
    mg_cols = model.event_mlp[0].weight.grad[:, -5:]      # manifold input columns
    _check("event_mlp manifold cols got grad", mg_cols.abs().sum() > 0)

    print("== 4. residual is live at inference + finite ==")
    model.eval()
    bb = make_batch([1600], seed=4)
    with torch.no_grad():
        la = model(bb)
        r_dir = model._res_ev[:, 0]
        fin = torch.isfinite(model._res_ev).all().item() and torch.isfinite(model._crease_ev).all().item()
        # perturb the predictor -> the inference residual (and thus logits) must move.
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        for n, p in model.named_parameters():
            if n.startswith("pred_tf") or n.startswith("flow_head"):
                p.add_(torch.randn_like(p) * 0.1)
        lb = model(bb)
        model.load_state_dict(sd)
    _check("residual/crease finite", fin)
    _check("residual live (predictor perturb changes logits)", (la - lb).abs().max().item() > 1e-4,
           f"Δ={ (la-lb).abs().max().item():.3e}")

    print("== 5. EMA moves target toward context ==")
    model.train()
    # simulate an optimizer step on the context encoder (target starts == ctx via the
    # deepcopy, so without this the EMA has nothing to track), then check the target
    # tracks it via _ema_update.
    with torch.no_grad():
        for p in model.ctx_tf.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ctx_ps = [p.detach().clone() for p in model.ctx_tf.parameters()]
    tgt0 = [p.detach().clone() for p in model.target_tf.parameters()]
    d0 = sum((t - c).abs().sum().item() for t, c in zip(tgt0, ctx_ps))
    model._ema_update()
    tgt1 = [p.detach().clone() for p in model.target_tf.parameters()]
    moved = sum((a - b).abs().sum().item() for a, b in zip(tgt1, tgt0))
    d1 = sum((t - c).abs().sum().item() for t, c in zip(tgt1, ctx_ps))
    _check("EMA target moved", moved > 0, f"Δ={moved:.3e}")
    _check("EMA moved toward ctx", d1 < d0, f"d0={d0:.4e} d1={d1:.4e}")

    print("== 6. eval drops JEPA/aux, keeps residual ==")
    model.eval()
    with torch.no_grad():
        model(make_batch([1000], seed=9))
    _check("_jepa_loss None at eval", model._jepa_loss is None)
    _check("_aux_logits None at eval", model._aux_logits is None)
    _check("_presence_logit kept at eval", model._presence_logit is not None)

    print("== 7. DDP-NaN rare-batch cases ==")
    model.train()
    cases = {
        "empty-batch": make_batch([0, 0], seed=1),
        "single-event": make_batch([1, 1500], seed=2),
        "all-one-t_bin": make_batch([1500, 1500], seed=3, t_fixed=0.5),
        "mixed-empty": make_batch([0, 1000, 0], seed=4),
        "empty-SSL-target": make_batch([800, 600], seed=5, t_fixed=0.4),
    }
    for tag, bt in cases.items():
        try:
            out = model(bt)
            ok = torch.isfinite(out).all().item() if out.numel() else True
            jlc = model._jepa_loss
            ok = ok and (jlc is None or torch.isfinite(jlc).item())
            if tag == "empty-batch":
                ok = ok and tuple(out.shape) == (0,)
        except Exception as e:                                # noqa: BLE001
            ok = False; tag = f"{tag} ({type(e).__name__}: {e})"
        _check(f"NaN-case {tag}", ok)

    print("== 8. ablations construct & run ==")
    for tag, over in [("no_crease", dict(crease_feature=False)),
                      ("no_residual", dict(residual_feature=False)),
                      ("mag_residual", dict(dir_residual=False)),
                      ("mask_space", dict(mask_mode="space")),
                      ("tube_time", dict(mask_mode="tube_time")),
                      ("no_presence", dict(presence_gate=False)),
                      ("jepa0", dict(jepa_weight=0.0)),
                      ("shear_on", dict(shear_consistency=True))]:
        try:
            m = build(**over); m.train()
            o = m(make_batch([1000, 700], seed=11))
            ok = torch.isfinite(o).all().item()
            if over.get("shear_consistency"):
                ok = ok and m._jepa_loss is not None and torch.isfinite(m._jepa_loss).item()
        except Exception as e:                                # noqa: BLE001
            ok = False; tag = f"{tag} ({type(e).__name__}: {e})"
        _check(f"ablation {tag}", ok)

    print("== 9. discriminability probe (informational; untrained) ==")
    model.eval()
    with torch.no_grad():
        hb = make_batch([6000], seed=21, hand=True)
        model(hb)
        lab = hb.labels > 0.5
        cre = model._crease_ev.squeeze(1)
        rdir = model._res_ev[:, 0]
        def stat(v):
            return (v[lab].mean().item(), v[~lab].mean().item(), v.std().item())
        ch, cb, cs = stat(cre); rh, rb, rs = stat(rdir)
    print(f"  crease  hand={ch:.4f} bg={cb:.4f} std={cs:.4f}")
    print(f"  r_dir   hand={rh:.4f} bg={rb:.4f} std={rs:.4f}")
    print("  (untrained: overlap expected; the kill-switch fires on a TRAINED ckpt)")

    print("== 10. CPU forward timing ==")
    model.eval()
    big = make_batch([12000, 12000], seed=31)
    with torch.no_grad():
        model(big)
        t0 = time.perf_counter()
        for _ in range(3):
            model(big)
        dt = (time.perf_counter() - t0) / 3
    print(f"  ~{big.feats.shape[0]} events, CPU forward: {dt*1e3:.0f} ms (GPU far lower)")

    print()
    if FAILED:
        print(f"SMOKE TEST FAILED ({len(FAILED)}): {FAILED}")
        return 1
    print("SMOKE TEST PASSED — all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
