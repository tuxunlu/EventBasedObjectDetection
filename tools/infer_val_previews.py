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
from data.sparse_event_collate import collate_sparse_events
from data_interface import DataInterface
from model_interface import ModelInterface
from utils.metrics.event_seg import event_pred_to_dense_iou, rasterize_events_to_logits
from utils.metrics.segmentation import binary_iou
from utils.viz.val_preview import (
    events_panel_from_sites,
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


def _is_sparse_sample(sample) -> bool:
    """Detect the per-event sparse path by its dataset sample signature.

    The sparse stream dataset yields ``(coords, feats, times, labels, dense_mask,
    meta)`` (a 6-tuple ending in a metadata dict), whereas the dense voxel path
    yields ``(voxel, mask, ...)``. This lets the same tool drive both the dense
    ``EventUNet``-style models and the event-native ``EventSparseSeg`` model.
    """
    return (
        isinstance(sample, (tuple, list))
        and len(sample) == 6
        and isinstance(sample[-1], dict)
    )


def _sparse_batch_from_sample(sample, device):
    """Collate one ``(coords, feats, times, labels, dense_mask, meta)`` sample
    into a single-element ``SparseEventBatch`` on ``device`` (what the model wants)."""
    coords, feats, times, labels, dense_mask, meta = sample
    return collate_sparse_events(
        [(coords, feats, times, labels, dense_mask, meta)]
    ).to(device)


def _spconv_conv_types():
    """The spconv base conv class (all SubM/Sparse/Inverse convs subclass it)."""
    try:
        import spconv.pytorch as spconv
        return (spconv.SparseConvolution,)
    except Exception:  # noqa: BLE001 - spconv missing (dense-only environments)
        return tuple()


def _fmt_macs(macs):
    """Human-readable MAC/FLOP count (K/M/G/T)."""
    for unit, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if macs >= scale:
            return f"{macs / scale:.3f} {unit}"
    return f"{macs:.0f} "


class SparseOpProfiler:
    """Hook-based MAC counter for the event-native sparse model.

    ``torch.utils.flop_counter`` does not understand spconv's custom autograd ops,
    so it silently reports 0 for the submanifold/sparse convolutions that dominate
    this model. This profiler instead hooks every conv (spconv) and ``nn.Linear``
    and counts the *actual* work for the events in the batch:

      * **Exact** (``exact=True``): MACs = ``(#valid kernel input-output pairs)`` ×
        ``C_in`` × ``C_out``, read from spconv's per-offset ``indice_pair_num`` in the
        output tensor's ``indice_dict``. This is the true submanifold cost (only the
        kernel taps that land on an *existing* active site are computed), so it is
        strictly less than the dense-kernel upper bound. Requires a small GPU→CPU
        reduction per layer, so it is used only on a single representative frame.
      * **Upper bound** (``exact=False``): MACs = ``n_out`` × ``kernel_volume`` ×
        ``C_in`` × ``C_out`` — computed from tensor *shapes* only (no GPU sync), cheap
        enough to record on every timed frame.

    ``nn.Linear`` (the per-event head) is exact in both modes (``N`` × ``in`` × ``out``).
    """

    def __init__(self, module):
        import torch.nn as nn
        self._nn_linear = nn.Linear
        self.module = module
        self.conv_types = _spconv_conv_types()
        self.handles = []
        self.records = []          # per-layer dicts for the current frame
        self.exact = False

    @staticmethod
    def _kvol(kernel_size):
        if hasattr(kernel_size, "__len__"):
            vol = 1
            for s in kernel_size:
                vol *= int(s)
            return vol
        return int(kernel_size) ** 3

    def _exact_pairs(self, m, out):
        """Sum of valid kernel input-output connections for this conv, or ``None``."""
        try:
            key = getattr(m, "indice_key", None)
            data = out.indice_dict.get(key) if key is not None else None
            if data is None:
                return None
            for attr in ("indice_pair_num", "pair_num", "indice_pairs_num"):
                v = getattr(data, attr, None)
                if v is not None and hasattr(v, "sum"):
                    return int(v.sum().item())
        except Exception:  # noqa: BLE001 - spconv internals vary by version
            pass
        return None

    def _conv_hook(self, m, inp, out):
        try:
            x = inp[0]
            rec = {
                "kind": "conv", "name": m.__class__.__name__,
                "indice_key": getattr(m, "indice_key", None),
                "c_in": int(x.features.shape[1]),
                "c_out": int(out.features.shape[1]),
                "n_out": int(out.features.shape[0]),
                "kvol": self._kvol(m.kernel_size),
                "exact_pairs": self._exact_pairs(m, out) if self.exact else None,
            }
            self.records.append(rec)
        except Exception:  # noqa: BLE001
            pass

    def _linear_hook(self, m, inp, out):
        try:
            x = inp[0]
            self.records.append({
                "kind": "linear", "name": "Linear", "indice_key": None,
                "c_in": int(x.shape[-1]), "c_out": int(out.shape[-1]),
                "n_out": int(x.shape[0]) if x.dim() >= 2 else 1,
                "kvol": 1, "exact_pairs": None,
            })
        except Exception:  # noqa: BLE001
            pass

    def attach(self):
        for mod in self.module.modules():
            if self.conv_types and isinstance(mod, self.conv_types):
                self.handles.append(mod.register_forward_hook(self._conv_hook))
            elif isinstance(mod, self._nn_linear):
                self.handles.append(mod.register_forward_hook(self._linear_hook))
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self):
        self.records = []

    def frame_macs(self):
        """``(upper_macs, exact_macs_or_None)`` for the records of the current frame."""
        upper = 0
        exact = 0
        have_exact = True
        for r in self.records:
            dense = r["n_out"] * r["kvol"] * r["c_in"] * r["c_out"]
            upper += dense
            if r["kind"] == "linear":
                exact += dense
            elif r["exact_pairs"] is not None:
                exact += r["exact_pairs"] * r["c_in"] * r["c_out"]
            else:
                have_exact = False
        return upper, (exact if have_exact else None)

    def print_breakdown(self):
        """Print a per-layer MAC table for the current (single) frame's records."""
        if not self.records:
            print("[flops] no conv/linear layers captured (profiler saw nothing).")
            return
        print("[flops] per-layer cost (one frame):")
        print(f"        {'layer':<22}{'sites/N':>10}{'Cin->Cout':>12}{'k^3':>5}"
              f"{'exactMAC':>12}{'upperMAC':>12}")
        for r in self.records:
            dense = r["n_out"] * r["kvol"] * r["c_in"] * r["c_out"]
            if r["kind"] == "linear":
                ex = dense
            elif r["exact_pairs"] is not None:
                ex = r["exact_pairs"] * r["c_in"] * r["c_out"]
            else:
                ex = None
            tag = f"{r['name']}:{r['indice_key']}" if r["indice_key"] else r["name"]
            ex_s = _fmt_macs(ex) if ex is not None else "   n/a"
            print(f"        {tag:<22}{r['n_out']:>10,}{r['c_in']:>6}->{r['c_out']:<5}"
                  f"{r['kvol']:>5}{ex_s:>12}{_fmt_macs(dense):>12}")
        upper, exact = self.frame_macs()
        if exact is not None:
            print(f"[flops] frame TOTAL: exact {_fmt_macs(exact)}MAC "
                  f"({_fmt_macs(2 * exact)}FLOP) | dense-upper {_fmt_macs(upper)}MAC "
                  f"({100.0 * exact / max(upper, 1):.1f}% of upper bound)")
        else:
            print(f"[flops] frame TOTAL: dense-upper {_fmt_macs(upper)}MAC "
                  f"({_fmt_macs(2 * upper)}FLOP); exact pair-counts unavailable.")


@torch.no_grad()
def measure_flops(net, model_input, streaming):
    """Per-frame MACs for one ``step``/forward via torch's FlopCounterMode.

    ``model_input`` is the already-prepared input the model consumes — a dense
    ``(1, C, H, W)`` voxel tensor, or a ``SparseEventBatch`` for the event-native
    model. Returns the multiply-accumulate count (MACs) for a single frame, or
    ``None`` if no FLOP counter is available (spconv submanifold convs are not
    covered by ``FlopCounterMode``). FLOPs are conventionally ~2x MACs.
    """
    def _run():
        if streaming:
            net.step(model_input, None)
        else:
            net(model_input)

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
def report_complexity_and_warmup(model, sample, device, sparse, streaming, warmup=5):
    """Print params + a rigorous per-frame compute cost, then warm the kernels.

    ``sample`` is a raw dataset sample (dense ``(voxel, mask, ...)`` or the sparse
    6-tuple); the right input is built per ``sparse`` so both paths share this.

    Returns a stats dict consumed by ``main`` to scale per-frame cost over the run:
        ``{"per_frame_macs": int|None,        # representative-frame MACs
           "exact_ratio": float|None}``       # sparse exact/upper-bound MAC ratio
    """
    total, trainable = count_parameters(model.model)
    print(f"[stats] parameters: {total:,} total ({total / 1e6:.3f} M), "
          f"{trainable:,} trainable")

    stats = {"per_frame_macs": None, "exact_ratio": None}

    if sparse:
        model_input = _sparse_batch_from_sample(sample, device)
        n_ev = int(model_input.feats.shape[0])
        # Rigorous spconv-aware accounting (FlopCounterMode is blind to spconv ops).
        prof = SparseOpProfiler(model.model).attach()
        prof.exact = True
        prof.reset()
        model.model(model_input)              # populates indice_dict + records
        prof.detach()
        print(f"[flops] representative frame: {n_ev:,} events on a "
              f"{model_input.height}x{model_input.width} grid")
        prof.print_breakdown()
        upper, exact = prof.frame_macs()
        stats["per_frame_macs"] = exact if exact is not None else upper
        stats["exact_ratio"] = (exact / upper) if (exact is not None and upper > 0) else None
    else:
        voxel = sample[0]
        model_input = voxel.unsqueeze(0).to(device)
        macs = measure_flops(model.model, model_input, streaming)
        if macs is not None:
            c, h, w = tuple(voxel.shape)
            print(f"[flops] compute @ input {c}x{h}x{w} (1 frame): "
                  f"{macs / 1e9:.3f} GMACs  ≈ {2 * macs / 1e9:.3f} GFLOPs")
            stats["per_frame_macs"] = macs

    # Warm up so cuDNN autotune / lazy CUDA init don't pollute the speed numbers.
    # The sparse model is single-shot (no streaming ``step``); reusing one sample
    # batch is enough to trigger spconv's lazy algorithm selection.
    state = None
    for _ in range(max(0, warmup)):
        if streaming:
            _, state = model.model.step(model_input, state)
        else:
            model.model(model_input)
    _sync(device)
    return stats


def _sequence_positions(val_set, s_idx, max_frames):
    """Frame positions for sequence ``s_idx`` (in (seq, frame) order), strided to cap."""
    positions = [i for i, (si, _f) in enumerate(val_set.index) if si == s_idx]
    if not positions:
        return None
    if len(positions) > max_frames:
        stride = ceil(len(positions) / max_frames)
        positions = positions[::stride]
    return positions


def _sequence_fps(val_set, s_idx):
    try:
        handle = val_set._get_handle(val_set.sequences[s_idx])
        return infer_fps(handle.get("flir_t"))
    except Exception:  # noqa: BLE001
        return 30.0


@torch.no_grad()
def render_sequence(model, val_set, s_idx, max_frames, device, sparse):
    """Dispatch to the dense voxel or sparse event-stream renderer for one sequence."""
    if sparse:
        return _render_sequence_sparse(model, val_set, s_idx, max_frames, device)
    return _render_sequence_dense(model, val_set, s_idx, max_frames, device)


@torch.no_grad()
def _render_sequence_sparse(model, val_set, s_idx, max_frames, device):
    """Run one held-out sequence through the event-native ``EventSparseSeg`` model.

    The model emits one logit per event; for the dense triptych we rasterize those
    per-event predictions back onto an ``(H, W)`` grid (max-pool per pixel, same as
    training's ``event_pred_to_dense_iou``), so the panels and IoU line up with the
    dense baseline. ``cropped`` keeps event colors only inside the predicted mask.
    """
    positions = _sequence_positions(val_set, s_idx, max_frames)
    if positions is None:
        return None
    frame_getter = getattr(val_set, "frame_sample", None) or val_set.__getitem__

    event_panels, gt_masks, pred_masks, cropped_panels, ious = [], [], [], [], []
    latencies, frame_macs, n_events = [], [], []
    inf_time = 0.0
    # Shape-only MAC profiler (no GPU sync) attached for the whole sequence; per
    # frame we read the upper-bound MAC count, which tracks the active-voxel count.
    prof = SparseOpProfiler(model.model).attach()
    prof.exact = False
    try:
        for pos in positions:
            coords, feats, times, labels, dense_mask, meta = frame_getter(pos)
            H, W = int(dense_mask.shape[-2]), int(dense_mask.shape[-1])
            ev_panel = events_panel_from_sites(coords, feats, H, W)   # (H, W, 3) BGR
            gt_np = (dense_mask.detach().cpu().numpy() > 0).astype("uint8")

            if coords.shape[0] == 0:
                # Empty window: nothing to predict, emit an all-background frame.
                event_panels.append(ev_panel)
                gt_masks.append(gt_np * 255)
                pred_masks.append(np.zeros((H, W), dtype="uint8"))
                cropped_panels.append(np.full_like(ev_panel, 128))
                ious.append(None)
                continue

            batch = _sparse_batch_from_sample(
                (coords, feats, times, labels, dense_mask, meta), device)
            prof.reset()
            _sync(device)
            t0 = time.perf_counter()
            logits = model.model(batch)                          # (N,) per-event
            _sync(device)
            dt = time.perf_counter() - t0
            inf_time += dt
            latencies.append(dt)
            upper, _exact = prof.frame_macs()
            frame_macs.append(upper)
            n_events.append(int(batch.feats.shape[0]))

            # Rasterize per-event logits -> dense (1, H, W) logit grid -> mask.
            raster = rasterize_events_to_logits(
                batch.coords, logits.detach(), batch.batch_idx, 1, H, W)
            pred_np = (torch.sigmoid(raster[0]) > 0.5).cpu().numpy().astype("uint8")

            event_panels.append(ev_panel)
            gt_masks.append(gt_np * 255)
            pred_masks.append(pred_np * 255)

            ph, pw = ev_panel.shape[:2]
            keep = pred_np if pred_np.shape == (ph, pw) else cv2.resize(
                pred_np, (pw, ph), interpolation=cv2.INTER_NEAREST)
            cropped = np.where(keep[:, :, None].astype(bool), ev_panel, 128).astype(np.uint8)
            cropped_panels.append(cropped)

            if float(dense_mask.sum()) > 0:
                ious.append(float(event_pred_to_dense_iou(
                    batch.coords, logits.detach(), batch.batch_idx,
                    batch.dense_mask, 1, H, W)))
            else:
                ious.append(None)
    finally:
        prof.detach()

    name = val_set.sequences[s_idx].name
    fps = _sequence_fps(val_set, s_idx)
    cost = {"latencies": latencies, "frame_macs": frame_macs, "n_events": n_events}
    return (event_panels, gt_masks, pred_masks, cropped_panels, ious, name, fps,
            inf_time, cost)


@torch.no_grad()
def _render_sequence_dense(model, val_set, s_idx, max_frames, device):
    """Run one held-out sequence statefully → (event_panels, gt, pred, cropped, ious, name, fps, inf_time).

    ``cropped`` is the events panel masked by the model's own predicted mask
    (events outside the prediction blacked out) — i.e. what the event stream
    would look like if cropped to the predicted hand/arm region.
    """
    # Frame positions (in the inherited frame-level index) belonging to this
    # sequence, already in (seq, frame) order.
    positions = _sequence_positions(val_set, s_idx, max_frames)
    if positions is None:
        return None

    frame_getter = getattr(val_set, "frame_sample", None) or val_set.__getitem__
    streaming = hasattr(model.model, "step")

    event_panels, gt_masks, pred_masks, cropped_panels, ious = [], [], [], [], []
    latencies = []
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
        dt = time.perf_counter() - t0
        inf_time += dt
        latencies.append(dt)
        prob = torch.sigmoid(logits)[0, 0]
        pred = (prob > 0.5).float()

        ev_panel = events_panel_from_voxel(voxel)          # (H, W, 3) BGR
        event_panels.append(ev_panel)
        gt_masks.append((mask.detach().cpu().numpy() * 255).astype("uint8"))
        pred_np = pred.cpu().numpy().astype("uint8")        # (H, W) at logits res
        pred_masks.append(pred_np * 255)

        # Crop the original event panel to the predicted region: keep event
        # colors where the prediction is foreground, fill the rest with the same
        # gray (128) the Events panel uses as its background so the two panels
        # match visually. Resize the mask to the panel grid (nearest) in case the
        # model output stride differs from the event-panel resolution.
        ph, pw = ev_panel.shape[:2]
        keep = pred_np if pred_np.shape == (ph, pw) else cv2.resize(
            pred_np, (pw, ph), interpolation=cv2.INTER_NEAREST)
        cropped = np.where(keep[:, :, None].astype(bool), ev_panel, 128).astype(np.uint8)
        cropped_panels.append(cropped)

        if float(mask.sum()) > 0:
            ious.append(float(binary_iou(logits.detach(), mask.unsqueeze(0).to(device))))
        else:
            ious.append(None)

    name = val_set.sequences[s_idx].name
    fps = _sequence_fps(val_set, s_idx)
    # Dense model has a fixed per-frame cost; main fills MACs from the FLOP report.
    cost = {"latencies": latencies, "frame_macs": [], "n_events": []}
    return (event_panels, gt_masks, pred_masks, cropped_panels, ious, name, fps,
            inf_time, cost)


def _event_throughput_epms(n_events, latencies_sec, total_inf_time_sec):
    """Events processed per millisecond of pure model time.

    Returns ``(aggregate_epms, per_frame_epms_or_None)`` where ``aggregate_epms``
    is ``sum(events) / (total_model_time_ms)`` and ``per_frame_epms`` is an array of
    ``events_i / latency_ms_i`` when ``n_events`` and ``latencies_sec`` are paired
    (same length, one entry per timed forward).
    """
    if not n_events or total_inf_time_sec <= 0:
        return None, None
    ev = np.asarray(n_events, dtype=np.float64)
    total_ev = float(ev.sum())
    aggregate = total_ev / (total_inf_time_sec * 1e3)
    per_frame = None
    if latencies_sec and len(latencies_sec) == len(n_events):
        lat_ms = np.asarray(latencies_sec, dtype=np.float64) * 1e3
        with np.errstate(divide="ignore", invalid="ignore"):
            per_frame = ev / np.maximum(lat_ms, 1e-9)
    return aggregate, per_frame


def _report_run_cost(total_frames, total_inf_time, latencies, frame_macs,
                     n_events, cost_stats, sparse, device):
    """Print the rigorous end-of-run cost + speed summary.

    Throughput/latency are over pure model time only (data I/O, event-panel
    rendering, and video encoding are excluded). MAC/FLOP totals are the dense-kernel
    upper bound for the sparse path, scaled by the representative-frame exact ratio
    for an exact estimate; for the dense path they are the FlopCounterMode per-frame
    count times the frame total.
    """
    if total_frames <= 0 or total_inf_time <= 0:
        return

    lat_ms = np.asarray(latencies, dtype=np.float64) * 1e3
    print("[stats] ============ inference cost & speed ============")
    print(f"[stats] frames: {total_frames}   pure model time: {total_inf_time:.2f}s "
          f"(batch=1, {device.type})")
    print(f"[stats] throughput: {total_frames / total_inf_time:.1f} frames/s  "
          f"({1e3 * total_inf_time / total_frames:.2f} ms/frame mean)")
    if lat_ms.size:
        p50, p90, p99 = np.percentile(lat_ms, [50, 90, 99])
        print(f"[stats] latency/frame ms: mean {lat_ms.mean():.2f}  p50 {p50:.2f}  "
              f"p90 {p90:.2f}  p99 {p99:.2f}  min {lat_ms.min():.2f}  max {lat_ms.max():.2f}")

    # ---- number of computations (MACs / FLOPs) ----
    exact_ratio = cost_stats.get("exact_ratio")
    if sparse and frame_macs:
        fm = np.asarray(frame_macs, dtype=np.float64)        # per-frame upper-bound MACs
        total_upper = float(fm.sum())
        avg_upper = float(fm.mean())
        print(f"[stats] compute/frame (upper bound): mean {_fmt_macs(avg_upper)}MAC  "
              f"min {_fmt_macs(fm.min())}MAC  max {_fmt_macs(fm.max())}MAC")
        if n_events:
            ev = np.asarray(n_events, dtype=np.float64)
            print(f"[stats] events/frame: mean {ev.mean():,.0f}  "
                  f"min {int(ev.min()):,}  max {int(ev.max()):,}  "
                  f"(MACs scale ~linearly with active events)")
            agg_epms, pf_epms = _event_throughput_epms(n_events, latencies, total_inf_time)
            if agg_epms is not None:
                print(f"[stats] event throughput: {agg_epms:,.1f} events/ms "
                      f"({agg_epms * 1e3:,.0f} events/s, "
                      f"{int(ev.sum()):,} events in {total_inf_time * 1e3:.1f} ms model time)")
            if pf_epms is not None and pf_epms.size:
                p50, p90, p99 = np.percentile(pf_epms, [50, 90, 99])
                print(f"[stats] events/ms per frame: mean {pf_epms.mean():,.1f}  "
                      f"p50 {p50:,.1f}  p90 {p90:,.1f}  p99 {p99:,.1f}  "
                      f"min {pf_epms.min():,.1f}  max {pf_epms.max():,.1f}")
        if exact_ratio is not None:
            total_exact = total_upper * exact_ratio
            print(f"[stats] total compute (exact est, ratio {exact_ratio:.2f}): "
                  f"{_fmt_macs(total_exact)}MAC  ({_fmt_macs(2 * total_exact)}FLOP) "
                  f"over {total_frames} frames")
            ach = (2 * total_exact) / total_inf_time
            print(f"[stats] achieved compute rate: {_fmt_macs(ach)}FLOP/s (exact est)")
        else:
            print(f"[stats] total compute (upper bound): {_fmt_macs(total_upper)}MAC  "
                  f"({_fmt_macs(2 * total_upper)}FLOP) over {total_frames} frames")
            ach = (2 * total_upper) / total_inf_time
            print(f"[stats] achieved compute rate: {_fmt_macs(ach)}FLOP/s (upper bound)")
    else:
        per_frame = cost_stats.get("per_frame_macs")
        if per_frame:
            total_macs = float(per_frame) * total_frames
            print(f"[stats] compute/frame: {_fmt_macs(per_frame)}MAC "
                  f"({_fmt_macs(2 * per_frame)}FLOP, fixed grid)")
            print(f"[stats] total compute: {_fmt_macs(total_macs)}MAC "
                  f"({_fmt_macs(2 * total_macs)}FLOP) over {total_frames} frames")
            print(f"[stats] achieved compute rate: "
                  f"{_fmt_macs(2 * total_macs / total_inf_time)}FLOP/s")
    print("[stats] ================================================")


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
    sparse = _is_sparse_sample(sample0)
    # The event-native sparse model is single-shot (no streaming ``step``).
    streaming = hasattr(model.model, "step") and not sparse
    print(f"[infer] input path: {'sparse event-stream' if sparse else 'dense voxel'}")
    cost_stats = report_complexity_and_warmup(model, sample0, device, sparse, streaming)

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
    all_latencies = []    # per-frame model seconds, across every sequence
    all_frame_macs = []   # per-frame upper-bound MACs (sparse path only)
    all_n_events = []     # per-frame active-event count (sparse path only)
    for rank_i, s_idx in enumerate(chosen):
        out = render_sequence(model, val_set, s_idx, args.max_frames, device, sparse)
        if out is None:
            print(f"[infer] ({rank_i + 1}/{k}) seq#{s_idx}: no frames, skipped")
            continue
        (event_panels, gt_masks, pred_masks, cropped_panels, ious, name, fps,
         inf_time, cost) = out
        total_frames += len(event_panels)
        total_inf_time += inf_time
        all_latencies.extend(cost.get("latencies", []))
        all_frame_macs.extend(cost.get("frame_macs", []))
        all_n_events.extend(cost.get("n_events", []))
        out_path = tmp_dir / f"{name}.mp4"
        write_panels_video(
            [event_panels, gt_masks, pred_masks, cropped_panels],
            ["Events", "Teacher (GT)", "Prediction", "Cropped Events"],
            out_path, fps=fps, title=name, per_frame_iou=ious, iou_panel=2,
        )
        written.append(out_path)
        per_frame_ms = 1e3 * inf_time / max(1, len(event_panels))
        seq_line = (f"[infer] ({rank_i + 1}/{k}) wrote {name}.mp4 "
                    f"({len(event_panels)} frames, {fps:.1f} fps; "
                    f"{per_frame_ms:.2f} ms/frame model)")
        if sparse and cost.get("n_events"):
            seq_ev = sum(cost["n_events"])
            seq_epms = seq_ev / max(inf_time * 1e3, 1e-9)
            seq_line += f"; {seq_epms:,.1f} events/ms ({seq_ev:,} events)"
        print(seq_line)

    _report_run_cost(total_frames, total_inf_time, all_latencies, all_frame_macs,
                     all_n_events, cost_stats, sparse, device)

    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in written:
            zf.write(p, arcname=p.name)
    print(f"[infer] zipped {len(written)} videos → {out_zip.resolve()}")


if __name__ == "__main__":
    main()
