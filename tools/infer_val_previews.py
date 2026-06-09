"""Render N random validation sequences through a trained checkpoint → zip.

Standalone (non-Lightning-Trainer) inference: load a ``ModelInterface``
checkpoint, pick a random subset of held-out validation sequences, run the model
*statefully* over each whole sequence, write one Events | Teacher (GT) | Student
triptych MP4 per sequence (reusing ``utils/viz/val_preview.py``), then bundle all
the MP4s into a single .zip.

This mirrors ``ModelInterface._save_validation_previews`` but, instead of one
video per DDP rank, renders an arbitrary count on a single device — meant to be
launched on a GPU compute node (NOT a login node).

Example
-------
    python tools/infer_val_previews.py \
        --config_path configs/hand_tracking.yaml \
        --ckpt 'lightning_logs/.../best-epoch=037-val_iou_epoch=0.8724.ckpt' \
        --num_videos 100 --seed 42 \
        --out_zip val_previews_100.zip
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import time
import zipfile
from math import ceil
from pathlib import Path

# Allow `python tools/infer_val_previews.py` from the repo root: put the repo
# root (this file's parent's parent) on sys.path so the top-level packages
# (configs, data, model, utils) import the same way they do for main.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from configs.config_schema import load_config_with_schema
from data.hand_event_clip_dataset import HandEventClipDataset  # noqa: F401 (import side: registers module)
from data_interface import DataInterface
from model_interface import ModelInterface
from utils.metrics.segmentation import binary_iou
from utils.viz.val_preview import (
    events_panel_from_voxel,
    infer_fps,
    write_panels_video,
)

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config_path", required=True,
                   help="Config YAML used to train the checkpoint (e.g. configs/hand_tracking.yaml).")
    p.add_argument("--ckpt", required=True, help="Path to the .ckpt file.")
    p.add_argument("--num_videos", type=int, default=100,
                   help="How many random validation sequences to render (capped at the number available).")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for the random sequence selection (default: TRAINING.seed from config).")
    p.add_argument("--max_frames", type=int, default=600,
                   help="Per-video frame cap; longer sequences are temporally strided down to this many frames.")
    p.add_argument("--out_zip", default="val_previews.zip", help="Output zip path.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Inference device (cuda|cpu).")
    return p.parse_args()


def _sync(device):
    """Block until queued GPU work is done — required for honest timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_parameters(net):
    """(total, trainable) parameter counts for an nn.Module."""
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_flops(net, sample_voxel, device, streaming):
    """Per-frame MACs for one ``step``/forward via torch's FlopCounterMode.

    Returns the multiply-accumulate count (MACs) for a single frame at the real
    input resolution, or ``None`` if no FLOP counter is available. FLOPs are
    conventionally ~2x MACs (one multiply + one add per MAC).
    """
    inp = sample_voxel.unsqueeze(0).to(device)

    def _run():
        if streaming:
            net.step(inp, None)
        else:
            net(inp)

    try:
        from torch.utils.flop_counter import FlopCounterMode
        fcm = FlopCounterMode(display=False)
        with fcm:
            _run()
        return int(fcm.get_total_flops())
    except Exception as exc:  # noqa: BLE001
        print(f"[stats] FLOP counter unavailable ({type(exc).__name__}: {exc})")
        return None


@torch.no_grad()
def report_complexity_and_warmup(model, sample_voxel, device, streaming, warmup=5):
    """Print params + per-frame FLOPs, then warm the kernels for fair timing."""
    total, trainable = count_parameters(model.model)
    print(f"[stats] parameters: {total:,} total ({total / 1e6:.3f} M), "
          f"{trainable:,} trainable")

    macs = measure_flops(model.model, sample_voxel, device, streaming)
    if macs is not None:
        c, h, w = tuple(sample_voxel.shape)
        print(f"[stats] compute @ input {c}x{h}x{w} (1 frame): "
              f"{macs / 1e9:.3f} GMACs  ≈ {2 * macs / 1e9:.3f} GFLOPs")

    # Warm up so cuDNN autotune / lazy CUDA init don't pollute the speed numbers.
    inp = sample_voxel.unsqueeze(0).to(device)
    state = None
    for _ in range(max(0, warmup)):
        if streaming:
            _, state = model.model.step(inp, state)
        else:
            model.model(inp)
    _sync(device)


@torch.no_grad()
def render_sequence(model, val_set, s_idx, max_frames, device):
    """Run one held-out sequence statefully → (event_panels, gt, pred, cropped, ious, name, fps, inf_time).

    ``cropped`` is the events panel masked by the model's own predicted mask
    (events outside the prediction blacked out) — i.e. what the event stream
    would look like if cropped to the predicted hand/arm region.
    """
    # Frame positions (in the inherited frame-level index) belonging to this
    # sequence, already in (seq, frame) order.
    positions = [i for i, (si, _f) in enumerate(val_set.index) if si == s_idx]
    if not positions:
        return None
    if len(positions) > max_frames:
        stride = ceil(len(positions) / max_frames)
        positions = positions[::stride]

    frame_getter = getattr(val_set, "frame_sample", None) or val_set.__getitem__
    streaming = hasattr(model.model, "step")

    event_panels, gt_masks, pred_masks, cropped_panels, ious = [], [], [], [], []
    inf_time = 0.0  # seconds spent strictly inside the model (excludes I/O + viz)
    state = None
    for pos in positions:
        sample = frame_getter(pos)
        voxel, mask = sample[0], sample[1]  # (C,H,W), (H,W)
        inp = voxel.unsqueeze(0).to(device)
        # Time only the forward/step, with a CUDA sync on either side so async
        # GPU dispatch doesn't make the model look faster than it is.
        _sync(device)
        t0 = time.perf_counter()
        if streaming:
            logits, state = model.model.step(inp, state)
        else:
            logits = model.model(inp)
        _sync(device)
        inf_time += time.perf_counter() - t0
        prob = torch.sigmoid(logits)[0, 0]
        pred = (prob > 0.5).float()

        ev_panel = events_panel_from_voxel(voxel)          # (H, W, 3) BGR
        event_panels.append(ev_panel)
        gt_masks.append((mask.detach().cpu().numpy() * 255).astype("uint8"))
        pred_np = pred.cpu().numpy().astype("uint8")        # (H, W) at logits res
        pred_masks.append(pred_np * 255)

        # Crop the original event panel to the predicted region: keep event
        # colors where the prediction is foreground, black out the rest. Resize
        # the mask to the panel grid (nearest) in case the model output stride
        # differs from the event-panel resolution.
        ph, pw = ev_panel.shape[:2]
        keep = pred_np if pred_np.shape == (ph, pw) else cv2.resize(
            pred_np, (pw, ph), interpolation=cv2.INTER_NEAREST)
        cropped = np.where(keep[:, :, None].astype(bool), ev_panel, 0).astype(np.uint8)
        cropped_panels.append(cropped)

        if float(mask.sum()) > 0:
            ious.append(float(binary_iou(logits.detach(), mask.unsqueeze(0).to(device))))
        else:
            ious.append(None)

    name = val_set.sequences[s_idx].name
    try:
        handle = val_set._get_handle(val_set.sequences[s_idx])
        fps = infer_fps(handle.get("flir_t"))
    except Exception:  # noqa: BLE001
        fps = 30.0
    return event_panels, gt_masks, pred_masks, cropped_panels, ious, name, fps, inf_time


def main():
    args = parse_args()

    cfg, _tracker = load_config_with_schema(args.config_path)
    seed = args.seed if args.seed is not None else int(getattr(cfg.TRAINING, "seed", 0))

    # Build only the validation split (DataInterface builds all three; that's fine
    # and cheap — the index build is metadata-only).
    dm = DataInterface(data_cfg=cfg.DATA)
    val_set = dm.validation_set

    device = torch.device(args.device)
    model = ModelInterface.load_from_checkpoint(
        args.ckpt,
        map_location=device,
        model_cfg=cfg.MODEL,
        optimizer_cfg=cfg.OPTIMIZER,
        scheduler_cfg=cfg.SCHEDULER,
        training_cfg=cfg.TRAINING,
        data_cfg=cfg.DATA,
    )
    model.eval()
    model.model.to(device)

    # Params + per-frame FLOPs, plus a kernel warmup, measured on a real sample.
    sample0 = (getattr(val_set, "frame_sample", None) or val_set.__getitem__)(0)
    streaming = hasattr(model.model, "step")
    report_complexity_and_warmup(model, sample0[0], device, streaming)

    n_seq = len(val_set.sequences)
    ids = list(range(n_seq))
    random.Random(seed).shuffle(ids)
    k = min(args.num_videos, n_seq)
    chosen = ids[:k]
    print(f"[infer] {n_seq} validation sequences available; rendering {k} "
          f"(seed={seed}) on {device}")
    if k < args.num_videos:
        print(f"[infer] NOTE: requested {args.num_videos} but only {n_seq} "
              f"validation sequences exist; rendering all {k}.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="val_previews_"))
    written = []
    total_frames = 0
    total_inf_time = 0.0  # seconds of pure model time, summed over all frames
    for rank_i, s_idx in enumerate(chosen):
        out = render_sequence(model, val_set, s_idx, args.max_frames, device)
        if out is None:
            print(f"[infer] ({rank_i + 1}/{k}) seq#{s_idx}: no frames, skipped")
            continue
        event_panels, gt_masks, pred_masks, cropped_panels, ious, name, fps, inf_time = out
        total_frames += len(event_panels)
        total_inf_time += inf_time
        out_path = tmp_dir / f"{name}.mp4"
        write_panels_video(
            [event_panels, gt_masks, pred_masks, cropped_panels],
            ["Events", "Teacher (GT)", "Prediction", "Cropped Events"],
            out_path, fps=fps, title=name, per_frame_iou=ious, iou_panel=2,
        )
        written.append(out_path)
        per_frame_ms = 1e3 * inf_time / max(1, len(event_panels))
        print(f"[infer] ({rank_i + 1}/{k}) wrote {name}.mp4 "
              f"({len(event_panels)} frames, {fps:.1f} fps; "
              f"{per_frame_ms:.2f} ms/frame model)")

    # Inference-speed summary: throughput is over pure model time only (data I/O,
    # event-panel rendering, and video encoding are excluded).
    if total_frames > 0 and total_inf_time > 0:
        print(f"[stats] inference speed: {total_frames} frames in "
              f"{total_inf_time:.2f}s model time → "
              f"{total_frames / total_inf_time:.1f} frames/s, "
              f"{1e3 * total_inf_time / total_frames:.2f} ms/frame "
              f"(batch=1, {device.type})")

    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in written:
            zf.write(p, arcname=p.name)
    print(f"[infer] zipped {len(written)} videos → {out_zip.resolve()}")


if __name__ == "__main__":
    main()
