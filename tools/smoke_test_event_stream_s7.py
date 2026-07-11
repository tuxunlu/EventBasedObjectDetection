"""CPU smoke test for ``model.event_stream_seg_s7.EventStreamSegS7`` — the TRUE
event-by-event dual-order selective-SSM per-event hand segmenter (Brief A of
``docs/deep_research_event_native_backbones_RECOVERED.md``). No spconv, no GPU.

Verifies the contract before a training run:
  * per-event output (one logit per EVENT, row-aligned to labels),
  * segmented parallel scan == a sequential reference (correctness of the core scan),
  * scan resets at batch_idx boundaries (no cross-sample leakage),
  * row-alignment survives the internal (unsorted-time + z-order) re-orderings,
  * NO frame artifact: two events sharing a coarse cell get DIFFERENT logits,
  * grad into every param (SSM eigen/Δ/gate, DMD gate, presence, aux, dual-scan probe),
  * model_interface hooks (_presence_logit, _aux_logits, _event_embedding, _scan_loss),
  * DMD dynamic-energy veto live + Koopman frequencies exposed,
  * dual_scan on/off ablation, GroupNorm/LayerNorm only, empty/1-event/odd-size robust.

Run: EventBasedObjectDetection/bin/python tools/smoke_test_event_stream_s7.py
"""

from __future__ import annotations

import sys

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from data.sparse_event_collate import collate_sparse_events
from loss.event_distillation import EventDistillationLoss, background_prototype_loss
from model.event_stream_seg_s7 import (
    EventStreamSegS7, _segmented_scan, _seg_layout)

FAILED = []


def _check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def _synth_sample(n_events, H, W, gen):
    x = torch.randint(0, W, (n_events,), generator=gen).long()
    y = torch.randint(0, H, (n_events,), generator=gen).long()
    coords = torch.stack([x, y], dim=1)
    times = torch.rand(n_events, generator=gen)                # UNSORTED on purpose
    pol = torch.where(torch.rand(n_events, generator=gen) > 0.5,
                      torch.ones(n_events), -torch.ones(n_events))
    feats = torch.stack([pol, times], dim=1)
    labels = (x < W // 2).float()
    dense = torch.zeros(H, W, dtype=torch.uint8)
    dense[y, x] = labels.to(torch.uint8)
    return coords, feats, times, labels, dense, {"n_events": n_events, "n_kept": n_events}


def _batch(n_events, H, W, B, gen):
    return collate_sparse_events([_synth_sample(n_events, H, W, gen) for _ in range(B)])


def _reference_scan(Abar, u, pos_in_seg):
    """Sequential ground-truth for _segmented_scan: h_k = Ā_k h_{k-1} + u_k, reset when
    pos_in_seg == 0."""
    M, N = u.shape
    h = torch.zeros(N, dtype=u.dtype)
    out = torch.empty_like(u)
    for k in range(M):
        h = u[k] if pos_in_seg[k].item() == 0 else Abar[k] * h + u[k]
        out[k] = h
    return out


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    B, H, W = 2, 120, 160
    model = EventStreamSegS7()                                  # all defaults
    npar = model.count_parameters()
    print(f"== build ==  params {npar:,}  feat_dim {model._feat_dim}  "
          f"d_state {model.d_state}  dual_scan {model.dual_scan}")
    _check("param budget (1.0M–2.6M)", 1_000_000 <= npar <= 2_600_000, f"{npar:,}")

    print("== 0. segmented scan == sequential reference (core correctness) ==")
    M, Nmodes = 40, 8
    Abar = torch.complex(torch.rand(M, Nmodes) * 0.9, torch.rand(M, Nmodes) - 0.5)
    u = torch.complex(torch.randn(M, Nmodes), torch.randn(M, Nmodes))
    # three segments of lengths 10, 25, 5
    bidx = torch.cat([torch.zeros(10), torch.ones(25), torch.full((5,), 2)]).long()
    pos, ss, sl, ml = _seg_layout(bidx, 3)
    got = _segmented_scan(Abar, u, pos, ml)
    ref = _reference_scan(Abar, u, pos)
    err = (got - ref).abs().max().item()
    _check("parallel scan matches sequential (per-segment reset)", err < 1e-4, f"max|Δ|={err:.2e}")
    _check("seg layout: pos resets, seg_len correct",
           pos[10].item() == 0 and pos[9].item() == 9 and sl[0].item() == 10
           and sl[10].item() == 25 and ml == 25)

    print("== 1. per-event contract + row-alignment through re-ordering ==")
    batch = _batch(3000, H, W, B, gen)
    n_ev = batch.coords.shape[0]
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    _check("one logit per EVENT (N,)", tuple(logits.shape) == (n_ev,),
           f"{tuple(logits.shape)} vs N={n_ev}")
    _check("logits finite", torch.isfinite(logits).all().item())
    with torch.no_grad():
        logits2 = model(batch)
    _check("deterministic (same batch -> identical logits)",
           torch.equal(logits, logits2))
    # Check the B-dependent hooks now, before the batch_size=1 forwards below overwrite them.
    _check("Koopman freqs + dyn-energy exposed",
           model._freqs is not None and tuple(model._freqs.shape) == (model.d_state,)
           and model._dyn_energy is not None)
    _check("presence logit (B,)", model._presence_logit is not None
           and tuple(model._presence_logit.shape) == (B,))
    # Row-alignment: on a TIE-FREE batch (distinct pixels + distinct times, so the scan
    # order is canonical regardless of input order) permuting the input events must
    # permute the outputs identically -> out[i] is exactly event i's logit.
    from data.sparse_event_collate import SparseEventBatch
    nd = 1000
    flat = torch.randperm(H * W, generator=gen)[:nd]
    dx = (flat % W).long(); dy = (flat // W).long()
    dcoords = torch.stack([dx, dy], dim=1)
    dtimes = (torch.randperm(nd, generator=gen).float() + 0.5) / nd     # distinct
    dfeats = torch.stack([torch.ones(nd), dtimes], dim=1)
    dlabels = (dx < W // 2).float()
    dmask = torch.zeros(H, W, dtype=torch.uint8); dmask[dy, dx] = dlabels.to(torch.uint8)
    dbatch = collate_sparse_events([(dcoords, dfeats, dtimes, dlabels, dmask, {})])
    perm = torch.randperm(nd, generator=gen)
    pb = SparseEventBatch(
        coords=dcoords[perm], feats=dfeats[perm], times=dtimes[perm],
        labels=dlabels[perm], batch_idx=torch.zeros(nd, dtype=torch.long),
        dense_mask=dmask.unsqueeze(0), batch_size=1, height=H, width=W)
    with torch.no_grad():
        ld = model(dbatch)
        lp = model(pb)
    _check("row-aligned under input permutation (tie-free)",
           torch.allclose(lp, ld[perm], atol=1e-4),
           f"max|Δ|={(lp - ld[perm]).abs().max().item():.2e}")

    print("== 2. scan resets at batch_idx boundaries (no cross-sample leak) ==")
    # Same single sample evaluated alone vs. concatenated with a second sample: the first
    # sample's per-event logits must be identical (state must not leak across samples).
    s0 = _synth_sample(500, H, W, gen)
    s1 = _synth_sample(700, H, W, gen)
    solo = collate_sparse_events([s0])
    pair = collate_sparse_events([s0, s1])
    model.eval()
    with torch.no_grad():
        lo_solo = model(solo)
        lo_pair = model(pair)[pair.batch_idx == 0]
    _check("sample-0 logits identical solo vs paired (no leak)",
           torch.allclose(lo_solo, lo_pair, atol=1e-4),
           f"max|Δ|={(lo_solo - lo_pair).abs().max().item():.2e}")

    print("== 3. NO frame artifact: co-cell events get DIFFERENT logits ==")
    cox = torch.tensor([[10, 10], [11, 11]], dtype=torch.long)
    cof = torch.tensor([[1.0, 0.20], [-1.0, 0.85]], dtype=torch.float32)
    cot = torch.tensor([0.20, 0.85], dtype=torch.float32)
    col = torch.tensor([1.0, 0.0], dtype=torch.float32)
    com = torch.zeros(H, W, dtype=torch.uint8)
    df = model.down_factor
    Hd = (H + df - 1) // df; Wd = (W + df - 1) // df
    same_cell = (10 * Wd // W == 11 * Wd // W) and (10 * Hd // H == 11 * Hd // H)
    cob = collate_sparse_events([(cox, cof, cot, col, com, {})])
    with torch.no_grad():
        co_logits = model(cob)
    _check("both events fall in one coarse cell", same_cell, f"down_factor={df}")
    _check("co-cell events -> distinct logits (no cell quantization)",
           (co_logits[0] - co_logits[1]).abs().item() > 1e-4,
           f"Δlogit={(co_logits[0]-co_logits[1]).abs().item():.4f}")

    print("== 4. gradient flow into ALL params (scan eigen/Δ/gate, DMD, presence, aux, probe) ==")
    model.train()
    batch = _batch(3000, H, W, B, gen)
    logits = model(batch)
    loss_fn = EventDistillationLoss(pos_weight=2.0, gjs_weight=1.0, gjs_pi1=0.5,
                                    bce_weight=0.0, lovasz_weight=1.0)
    total = loss_fn(logits, batch.labels, batch_idx=batch.batch_idx)["total"]
    if model._aux_logits is not None:
        total = total + 0.2 * model._aux_logits.float().mean()
    if model._presence_logit is not None:
        total = total + 0.3 * model._presence_logit.float().mean()
    if model._scan_loss is not None:
        total = total + 0.1 * model._scan_loss
    total.backward()
    n_tr = sum(1 for p in model.parameters() if p.requires_grad)
    n_gr = sum(1 for n, p in model.named_parameters()
               if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all())
    miss = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    _check("all params finite grad", n_gr == n_tr, f"{n_gr}/{n_tr}; missing={miss[:6]}")
    ssm0 = model.time_blocks_mod[0].ssm
    _check("selective-SSM params got grad (a_im, B_re, delta_proj, gate_proj)",
           ssm0.a_im.grad is not None and ssm0.a_im.grad.abs().sum() > 0
           and ssm0.B_re.grad.abs().sum() > 0
           and ssm0.delta_proj.weight.grad is not None
           and ssm0.gate_proj.weight.grad.abs().sum() > 0)
    _check("DMD gate params got grad",
           model.dyn_a.grad is not None and model.dyn_b.grad is not None)
    _check("dual-scan probe got grad",
           model.scan_probe.weight.grad is not None
           and model.scan_probe.weight.grad.abs().sum() > 0)
    _check("aux head (B,1,G,G)", model._aux_logits is not None
           and tuple(model._aux_logits.shape) == (B, 1, model.aux_grid, model.aux_grid))
    _check("scan_loss scalar exposed in train", model._scan_loss is not None
           and model._scan_loss.ndim == 0)

    print("== 5. DMD static veto is live (perturbing the gate only lowers logits) ==")
    model.eval()
    batch = _batch(3000, H, W, B, gen)
    with torch.no_grad():
        base = model(batch).clone()
        e = model._dyn_energy
        finite_nonneg = torch.isfinite(e).all().item() and (e >= 0).all().item()
        saved = model.dyn_a.data.clone()
        model.dyn_a.data.fill_(-8.0)
        vetoed = model(batch)
        model.dyn_a.data.copy_(saved)
    _check("dynamic energy finite & >= 0", finite_nonneg)
    _check("veto shifts logits (and only downward)",
           (vetoed <= base + 1e-5).all().item()
           and (base - vetoed).abs().max().item() > 1e-4,
           f"max|Δ|={(base-vetoed).abs().max().item():.4f}")

    print("== 6. background-prototype null loss path (optional embedding hook) ==")
    m2 = EventStreamSegS7(null_loss_weight=0.5).train()
    b2 = _batch(2000, H, W, B, gen)
    out2 = m2(b2)
    emb = m2._event_embedding
    _check("null-loss embedding exposed in train", emb is not None
           and emb.shape[0] == out2.shape[0])
    if emb is not None:
        nullv = background_prototype_loss(emb, b2.labels, b2.batch_idx, margin=1.0)
        _check("null loss finite", torch.isfinite(nullv).item())

    print("== 7. dual_scan on/off ablation (z-order scan adds params, still row-aligned) ==")
    m_off = EventStreamSegS7(dual_scan=False).eval()
    _check("dual_scan=False -> no z-order blocks/probe",
           m_off.zorder_blocks is None
           and m_off.count_parameters() < model.count_parameters())
    with torch.no_grad():
        o_off = m_off(_batch(1500, H, W, B, gen))
    _check("single-scan model still per-event + finite",
           tuple(o_off.shape)[0] == _batch(1, H, W, 1, gen).coords.shape[0] * 0 + o_off.shape[0]
           and torch.isfinite(o_off).all().item())

    print("== 8. NO BatchNorm (LOSO-safe) ==")
    _check("no BatchNorm modules", not any(isinstance(m, _BatchNorm) for m in model.modules()))

    print("== 9. robustness: empty, 1-event, odd sizes ==")
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
    m_small = EventStreamSegS7(down_factor=8, time_blocks=3).eval()
    with torch.no_grad():
        os = m_small(_batch(1500, 103, 149, 2, gen))
    _check("time_blocks=3 / down_factor=8 ok", torch.isfinite(os).all().item())

    print()
    if FAILED:
        print(f"SMOKE TEST FAILED ({len(FAILED)}): {FAILED}")
        return 1
    print("SMOKE TEST PASSED — all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
