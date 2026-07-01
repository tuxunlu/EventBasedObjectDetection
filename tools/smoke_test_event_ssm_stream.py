"""CPU smoke test for ``model.event_ssm_seg_stream.EventSSMSegStream`` — the per-EVENT
Koopman/SSM segmenter (event stream in, filtered event stream out). No spconv, no GPU.

Verifies the contract before a training run:
  * per-event output (one logit per EVENT, row-aligned to labels),
  * NO frame artifact: two events sharing a coarse cell get DIFFERENT logits,
  * grad into every param (esp. the SSM complex eigen-params, DMD gate, presence, aux),
  * DMD dynamic-energy veto live + Koopman frequencies exposed,
  * model_interface hooks (_presence_logit, _aux_logits, _event_embedding),
  * GroupNorm/LayerNorm only (LOSO-safe), and empty / 1-event / odd-size robustness,
  * the ~10x parameter budget (~960k).

Run: EventBasedObjectDetection/bin/python tools/smoke_test_event_ssm_stream.py
"""

from __future__ import annotations

import sys

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from data.sparse_event_collate import collate_sparse_events
from loss.event_distillation import EventDistillationLoss, background_prototype_loss
from model.event_ssm_seg_stream import EventSSMSegStream

FAILED = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def _synth_sample(n_events, H, W, gen):
    """One synthetic per-event sample (coords, feats=[pol,t_norm], times, labels, mask, meta)."""
    x = torch.randint(0, W, (n_events,), generator=gen).long()
    y = torch.randint(0, H, (n_events,), generator=gen).long()
    coords = torch.stack([x, y], dim=1)
    times = torch.rand(n_events, generator=gen)
    pol = torch.where(torch.rand(n_events, generator=gen) > 0.5,
                      torch.ones(n_events), -torch.ones(n_events))
    feats = torch.stack([pol, times], dim=1)
    labels = (x < W // 2).float()                              # fg = left half
    dense = torch.zeros(H, W, dtype=torch.uint8)
    dense[y, x] = labels.to(torch.uint8)
    return coords, feats, times, labels, dense, {"n_events": n_events, "n_kept": n_events}


def _batch(n_events, H, W, B, gen):
    return collate_sparse_events([_synth_sample(n_events, H, W, gen) for _ in range(B)])


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    B, H, W = 2, 120, 160
    model = EventSSMSegStream()                                # all defaults
    npar = model.count_parameters()
    print(f"== build ==  params {npar:,}  feat_dim {model._feat_dim}")
    _check("~10x param budget (0.80M–1.15M)", 800_000 <= npar <= 1_150_000, f"{npar:,}")

    print("== 1. per-event contract ==")
    batch = _batch(3000, H, W, B, gen)
    n_ev = batch.coords.shape[0]
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    _check("one logit per EVENT (N,)", tuple(logits.shape) == (n_ev,),
           f"{tuple(logits.shape)} vs N={n_ev}")
    _check("logits finite", torch.isfinite(logits).all().item())
    _check("Koopman freqs + dyn-energy exposed",
           model._freqs is not None and tuple(model._freqs.shape) == (model.d_state,)
           and model._dyn_energy is not None)
    _check("presence logit (B,)", model._presence_logit is not None
           and tuple(model._presence_logit.shape) == (B,))

    print("== 2. NO frame artifact: co-cell events get DIFFERENT logits ==")
    # Two events in the SAME coarse cell (down_factor=4 -> 10//4 == 11//4 == 2) but
    # different sub-pixel position, time and polarity. A frame/grid model would give them
    # one shared cell value; a true per-event filter must separate them.
    df = model.down_factor
    cox = torch.tensor([[10, 10], [11, 11]], dtype=torch.long)
    cof = torch.tensor([[1.0, 0.20], [-1.0, 0.85]], dtype=torch.float32)
    cot = torch.tensor([0.20, 0.85], dtype=torch.float32)
    col = torch.tensor([1.0, 0.0], dtype=torch.float32)
    com = torch.zeros(H, W, dtype=torch.uint8)
    same_cell = (10 // df == 11 // df)
    cob = collate_sparse_events([(cox, cof, cot, col, com, {})])
    with torch.no_grad():
        co_logits = model(cob)
    _check("both events fall in one coarse cell", same_cell, f"down_factor={df}")
    _check("co-cell events -> distinct logits (no cell quantization)",
           (co_logits[0] - co_logits[1]).abs().item() > 1e-4,
           f"Δlogit={ (co_logits[0]-co_logits[1]).abs().item():.4f}")

    print("== 3. gradient flow into all params (SSM eigen, DMD gate, presence, aux) ==")
    model.train()
    batch = _batch(3000, H, W, B, gen)
    logits = model(batch)
    loss_fn = EventDistillationLoss(pos_weight=2.0, gjs_weight=1.0, gjs_pi1=0.5,
                                    bce_weight=0.0, lovasz_weight=1.0)
    terms = loss_fn(logits, batch.labels, batch_idx=batch.batch_idx)
    total = terms["total"]
    # aux occupancy + presence get grad via their model_interface losses; add stand-ins.
    if model._aux_logits is not None:
        total = total + 0.2 * model._aux_logits.float().mean()
    if model._presence_logit is not None:
        total = total + 0.3 * model._presence_logit.float().mean()
    total.backward()
    n_tr = sum(1 for p in model.parameters() if p.requires_grad)
    n_gr = sum(1 for n, p in model.named_parameters()
               if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all())
    miss = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    _check("all params finite grad", n_gr == n_tr, f"{n_gr}/{n_tr}; missing={miss[:6]}")
    ssm0 = model.ssm_blocks[0].ssm
    _check("SSM eigen params got grad",
           ssm0.a_im.grad is not None and ssm0.a_im.grad.abs().sum() > 0
           and ssm0.B_re.grad.abs().sum() > 0)
    _check("DMD gate params got grad",
           model.dyn_a.grad is not None and model.dyn_b.grad is not None)
    _check("aux head (B,1,G,G)", model._aux_logits is not None
           and tuple(model._aux_logits.shape) == (B, 1, model.aux_grid, model.aux_grid))

    print("== 4. DMD static veto is live (perturbing the gate changes logits) ==")
    model.eval()
    batch = _batch(3000, H, W, B, gen)
    with torch.no_grad():
        base = model(batch).clone()
        e = model._dyn_energy
        finite_nonneg = torch.isfinite(e).all().item() and (e >= 0).all().item()
        saved = model.dyn_a.data.clone()
        model.dyn_a.data.fill_(-8.0)                            # strong static veto
        vetoed = model(batch)
        model.dyn_a.data.copy_(saved)
    _check("dynamic energy finite & >= 0", finite_nonneg)
    _check("veto shifts logits (and only downward)",
           (vetoed <= base + 1e-5).all().item()
           and (base - vetoed).abs().max().item() > 1e-4,
           f"max|Δ|={(base-vetoed).abs().max().item():.4f}")

    print("== 5. background-prototype null loss path (optional embedding hook) ==")
    m2 = EventSSMSegStream(null_loss_weight=0.5).train()
    b2 = _batch(2000, H, W, B, gen)
    out2 = m2(b2)
    emb = m2._event_embedding
    _check("null-loss embedding exposed in train", emb is not None
           and emb.shape[0] == out2.shape[0])
    if emb is not None:
        nullv = background_prototype_loss(emb, b2.labels, b2.batch_idx, margin=1.0)
        _check("null loss finite", torch.isfinite(nullv).item())

    print("== 6. NO BatchNorm (LOSO-safe) ==")
    _check("no BatchNorm modules", not any(isinstance(m, _BatchNorm) for m in model.modules()))

    print("== 7. robustness: empty, 1-event, odd sizes ==")
    empty = collate_sparse_events([(torch.zeros(0, 2, dtype=torch.long),
                                     torch.zeros(0, 2), torch.zeros(0), torch.zeros(0),
                                     torch.zeros(H, W, dtype=torch.uint8), {})])
    with torch.no_grad():
        oe = model(empty)
    _check("empty batch -> (0,)", tuple(oe.shape) == (0,))
    one = _batch(1, 97, 131, 1, gen)
    with torch.no_grad():
        o1 = model(one)
    _check("1-event odd-size ok", tuple(o1.shape) == (1,) and torch.isfinite(o1).all().item())
    m_small = EventSSMSegStream(time_bins=2, down_factor=8).eval()
    with torch.no_grad():
        os = m_small(_batch(1500, 103, 149, 2, gen))
    _check("time_bins=2 / down_factor=8 ok", torch.isfinite(os).all().item())

    print("== 8. recency-weighted context (edge-sharpening readout) ==")
    # recency_context=True (default) registers a learnable per-T readout weight (softmax'd
    # in forward) that tracks the moving edge's current bin instead of the window mean; it
    # gets gradient (covered by the all-params check in section 3). recency_context=False
    # registers NO parameter -> no unused-DDP-param under strategy:auto.
    _check("recency_logits present, len == time_bins",
           model.recency_logits is not None
           and tuple(model.recency_logits.shape) == (model.time_bins,))
    m_off = EventSSMSegStream(recency_context=False)
    _check("recency off -> no recency param (DDP-safe ablation)",
           m_off.recency_logits is None)
    _check("recency model has +T params vs off",
           model.count_parameters() - m_off.count_parameters() == model.time_bins)

    print("== 9. per-event loss weighting (boundary emphasis) ==")
    lf = EventDistillationLoss(bce_weight=0.0, pos_weight=2.0, gjs_weight=1.0,
                               gjs_pi1=0.5, lovasz_weight=1.0)
    g = torch.randn(2000); lb = (torch.rand(2000) > 0.6).float()
    bi = torch.randint(0, 4, (2000,))
    t_none = lf(g, lb, batch_idx=bi)["total"]
    t_ones = lf(g, lb, batch_idx=bi, weight=torch.ones(2000))["total"]
    _check("weight=None == weight=ones (back-compat)",
           (t_none - t_ones).abs().item() < 1e-5,
           f"Δ={(t_none - t_ones).abs().item():.2e}")
    w = 1.0 + 4.0 * (lb > 0.5).float()
    t_w = lf(g, lb, batch_idx=bi, weight=w)["gjs"]
    t_b = lf(g, lb, batch_idx=bi)["gjs"]
    _check("non-uniform weight changes the pointwise term",
           (t_w - t_b).abs().item() > 1e-3)
    lb2 = lb.clone(); lb2[:50] = -1.0                     # ignore rows + weight must align
    _check("ignore (-1) rows + weight -> finite",
           torch.isfinite(lf(g, lb2, batch_idx=bi, weight=w)["total"]).item())

    print()
    if FAILED:
        print(f"SMOKE TEST FAILED ({len(FAILED)}): {FAILED}")
        return 1
    print("SMOKE TEST PASSED — all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
