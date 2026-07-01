"""Smoke test for the N-HOT3D per-event sparse segmentation dataloader.

Unlike ``tools/smoke_test_event_sparse.py`` (synthetic events, GPU/spconv required),
this exercises the *real* :class:`NHOT3DEventDataset` against the on-disk dataset and
verifies the :class:`SparseEventBatch` contract end-to-end on CPU:

  * dataset enumerates sequences + (event-chunk, mask) frame pairs,
  * each sample is row-aligned (coords/feats/times/labels share N), in-bounds,
    times in [0,1], features [pol, t_norm],
  * ``collate_sparse_events`` produces a valid batch,
  * per-event labels are exactly ``dense_mask[y, x]`` (label/row alignment), and
    foreground events all fall inside the GT mask (the event->mask rescale is correct).

If a CUDA device + spconv are available it additionally pushes one real batch through
``EventSparseSeg`` + ``EventDistillationLoss`` + backward, proving the loader is a
drop-in for the ``event_segmentation`` task.

    python tools/smoke_test_nhot3d.py
    python tools/smoke_test_nhot3d.py --purpose validation --batch 4 --model
"""

from __future__ import annotations

import argparse

import torch

from data.nhot3d_event_dataset import NHOT3DEventDataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/fs/nexus-projects/DVS_Actions/N-HOT3D/Aria",
                    help="N-HOT3D Aria dir holding train/ valid/ test/")
    ap.add_argument("--purpose", default="train",
                    choices=("train", "validation", "test"))
    ap.add_argument("--batch", type=int, default=3, help="samples to collate")
    ap.add_argument("--max_sequences", type=int, default=2,
                    help="cap enumerated sequences for a quick test")
    ap.add_argument("--frames_per_sample", type=int, default=1)
    ap.add_argument("--boundary_ignore_px", type=int, default=0)
    ap.add_argument("--model", action="store_true",
                    help="also run EventSparseSeg + loss + backward (needs CUDA + spconv)")
    args = ap.parse_args()

    ds = NHOT3DEventDataset(
        root_dir=args.root, purpose=args.purpose,
        max_sequences=args.max_sequences,
        frames_per_sample=args.frames_per_sample,
        boundary_ignore_px=args.boundary_ignore_px,
    )
    print(f"[data ] purpose={args.purpose} sequences={len(ds.sequences)} "
          f"frames={len(ds)} skipped_seqs={ds._n_skipped_sequences} "
          f"res={ds.image_width}x{ds.image_height} (event {ds.event_width}x{ds.event_height})")
    for s in ds.sequences[:4]:
        print(f"        {s.token} subj={s.subject} frames={len(s.index_set)} "
              f"ev_dir=...{str(s.event_dir)[-40:]}")

    # ---- per-sample contract + label/mask alignment --------------------------
    idxs = torch.linspace(0, len(ds) - 1, steps=min(args.batch, len(ds))).long().tolist()
    samples = []
    fg_inside = fg_total = 0
    for i in idxs:
        coords, feats, times, labels, dense_mask, meta = ds[i]
        N = coords.shape[0]
        assert feats.shape == (N, 2) and times.shape == (N,) and labels.shape == (N,), \
            f"row misalignment at i={i}"
        if N:
            assert coords[:, 0].min() >= 0 and coords[:, 0].max() < ds.image_width
            assert coords[:, 1].min() >= 0 and coords[:, 1].max() < ds.image_height
            assert 0.0 <= float(times.min()) and float(times.max()) <= 1.0001
            assert set(torch.unique(feats[:, 0]).tolist()) <= {-1.0, 1.0}, "polarity not signed"
            # label == dense_mask[y, x] for every non-ignored event (the core contract)
            keep = labels >= 0
            x = coords[keep, 0].long(); y = coords[keep, 1].long()
            mask_at = dense_mask[y, x].float()
            assert torch.equal(mask_at, labels[keep]), "label != dense_mask[y,x] (misaligned)"
            fg = labels == 1
            fg_inside += int(dense_mask[coords[fg, 1].long(), coords[fg, 0].long()].sum())
            fg_total += int(fg.sum())
        samples.append((coords, feats, times, labels, dense_mask, meta))
        print(f"[samp ] i={i:6d} seq={meta['sequence']} f={meta['frame_index']:4d} "
              f"N={N:6d} fg={int((labels==1).sum())} ign={int((labels<0).sum())} "
              f"maskpx={int(dense_mask.sum())} nf={meta['n_frames']}")
    assert fg_total == 0 or fg_inside == fg_total, "foreground events fall outside the mask"
    print(f"[align] foreground-events-inside-mask: {fg_inside}/{fg_total} (expect all)")

    batch = ds.collate_fn(samples)
    assert batch.coords.shape[0] == batch.feats.shape[0] == batch.labels.shape[0]
    assert batch.batch_idx.max() < batch.batch_size
    assert batch.dense_mask.shape[0] == batch.batch_size
    print(f"[coll ] batch_size={batch.batch_size} total_events={batch.coords.shape[0]} "
          f"feat_dim={batch.feats.shape[1]} dense_mask={tuple(batch.dense_mask.shape)}")

    # ---- optional: real model + loss + backward (GPU/spconv) -----------------
    if args.model:
        if not torch.cuda.is_available():
            print("[model] skipped: no CUDA device (spconv needs the GPU box)")
        else:
            from loss.event_distillation import EventDistillationLoss
            from model.event_sparse_seg import EventSparseSeg
            dev = torch.device("cuda")
            b = batch.to(dev)
            model = EventSparseSeg(in_features=2, num_classes=1).to(dev)
            logits = model(b)
            assert logits.shape[0] == b.coords.shape[0], "per-event contract violated"
            loss_fn = EventDistillationLoss(pos_weight=2.0, sce_beta=1.0,
                                            sce_alpha=0.1, lovasz_weight=1.0)
            terms = loss_fn(logits, b.labels, batch_idx=b.batch_idx)
            terms["total"].backward()
            n_grad = sum(int(p.grad is not None) for p in model.parameters() if p.requires_grad)
            n_train = sum(1 for p in model.parameters() if p.requires_grad)
            print(f"[model] EventSparseSeg logits={tuple(logits.shape)} "
                  f"loss={terms['total'].item():.4f} grad={n_grad}/{n_train}")
            assert n_grad == n_train

    print("[ok   ] N-HOT3D dataloader smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
