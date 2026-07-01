"""Render N random validation sequences through a trained checkpoint → zip.

Standalone (non-Lightning-Trainer) inference: load a ``ModelInterface``
checkpoint, pick a random subset of held-out validation sequences, run the model
over each whole sequence, and write one MP4 per sequence (then bundle all the
MP4s into a single .zip).

Each output frame is a **2x2 panel grid stacked over a 3D (x, y, t) event-volume
row**:

    +----------------+----------------+
    |   RGB (FLIR)   |    GT Mask     |
    +----------------+----------------+
    |     Events     |   Prediction   |
    +----------------+----------------+
    |  (x,y,t) all | GT-filtered | pred-filtered |   <- event volumes

The top RGB/GT row is in the FLIR camera frame; the Events/Prediction row and the
(x, y, t) volumes are in the event-sensor frame (see ``_load_rgb_frame`` for the
small inter-sensor parallax this implies). The volume row shows the per-event
point cloud of the window, then the same cloud keeping only GT-foreground events,
then only prediction-foreground events. Rendered with a dependency-free cv2/numpy
3D projector (no matplotlib).

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
from utils.metrics.event_seg import (
    event_f1,
    event_pred_to_dense_iou,
    rasterize_events_to_logits,
)
from utils.metrics.segmentation import binary_iou
from utils.viz.val_preview import (
    events_panel_from_sites,
    events_panel_from_voxel,
    infer_fps,
    mask_to_bgr,
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
    p.add_argument("--vol_dlen", type=float, default=1.4,
                   help="(x,y,t) volume: length of the time axis in the oblique "
                        "projection (>1 stretches the tube along t; try 1.0-2.5).")
    p.add_argument("--vol_alpha", type=float, default=37.0,
                   help="(x,y,t) volume: oblique angle in degrees for the receding "
                        "time axis (smaller = flatter/wider).")
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


# ----------------------------------------------------------------------------
# Layout: a 2x2 panel grid (RGB | GT mask / Events | Prediction) stacked over a
# row of three 3D (x, y, t) event volumes (all | GT-filtered | pred-filtered).
# ----------------------------------------------------------------------------
#
# Replaces the old single-row "Events | GT | Pred | Cropped" strip. The volumes
# use a dependency-free cv2/numpy orthographic projection on purpose: matplotlib
# is not installed in this repo's venv and the viz path is deliberately
# cv2/numpy/torch-only (see utils/viz/val_preview.py).

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GRID_GAP = 8          # px gap between the four grid panels (and grid <-> volume)


def _put_label(img, text, org=(8, 22), scale=0.6, color=(255, 255, 255)):
    """Caption with a translucent dark backing + black halo — legible on any
    background (incl. the bright white prediction/GT mask panels)."""
    (tw, th), bl = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = org
    x0, y0 = max(x - 4, 0), max(y - th - 4, 0)
    x1, y1 = min(x + tw + 4, img.shape[1]), min(y + bl + 2, img.shape[0])
    roi = img[y0:y1, x0:x1]
    if roi.size:                                   # darken behind the text to ~35%
        cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0, roi)
    cv2.putText(img, text, org, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, 1, cv2.LINE_AA)


def _load_rgb_frame(seq_dir, frame_index, h, w):
    """FLIR RGB frame PNG for ``frame_index`` → BGR ``(h, w, 3)`` uint8 panel.

    The FLIR stream is a real 8-bit RGB machine-vision camera (``proc/flir/frame/
    NNNNNN.png``), indexed by the same ``frame_index`` that selects the teacher
    mask. Mirrors ``tools/visualize_rgb_events.load_frame`` (16-bit / grayscale
    handling). Returns a gray "n/a" placeholder when the frame can't be read so a
    missing RGB never aborts the video.

    NOTE: the FLIR and Prophesee event cameras are distinct sensors with their own
    intrinsics; events are *not* warped into the FLIR frame, so the RGB/GT-mask
    (FLIR) top row and the events/prediction (event sensor) bottom row carry a
    small depth-dependent parallax — an existing property of the dataset, not
    this visualization.
    """
    if frame_index is not None:
        path = Path(seq_dir) / "proc" / "flir" / "frame" / f"{int(frame_index):06d}.png"
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.dtype == np.uint16:
                img = (img.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            return img
    ph = np.full((h, w, 3), 40, np.uint8)
    _put_label(ph, "RGB (n/a)", (8, h // 2), 0.7, (160, 160, 160))
    return ph


class Volume3DRenderer:
    """Dependency-free (cv2/numpy) 3D ``(x, y, t)`` event-volume renderer.

    Uses an **oblique (cabinet) projection**: the front ``t=0`` face is an
    undistorted, fronto-parallel ``x``–``y`` frame (so you read it like a rendered
    frame, ``x`` →, ``y`` ↓), and each later time-slice is shifted diagonally by
    ``nt * dlen`` along the oblique angle, so the window's events stack *back* along
    ``t`` and the temporal progression of the stream is visible. ``dlen`` sets how
    long the ``t`` axis appears (raise it to stretch the tube); ``alpha`` is the
    oblique angle. Draws a wireframe box + colored x/y/t axes, depth-sorts points
    back→front, shades them by time (recent/front bright, older/back dim), and
    colors each by polarity (BGR red = +, blue = −, matching the 2D event panels).
    One :meth:`render` produces three subsets side by side — **all events |
    GT-mask-filtered | prediction-filtered** — sharing one camera. ``t`` is the
    in-window normalized time ``∈[0, 1]``.
    """

    def __init__(self, total_w, sub_h=440, sep_w=3, alpha=37.0, dlen=1.4,
                 max_points=6000, point_px=1):
        self.sep_w = int(sep_w)
        self.sub_w = max(64, (total_w - 2 * self.sep_w) // 3)
        self.sub_h = int(sub_h)
        self.out_w = 3 * self.sub_w + 2 * self.sep_w
        self.max_points = int(max_points)
        self.point_px = int(point_px)
        # Oblique per-unit-time screen shift (cabinet projection): +t recedes
        # up-and-to-the-right by (shift_u, shift_v).
        a = np.radians(alpha)
        self.shift_u = np.cos(a) * float(dlen)
        self.shift_v = np.sin(a) * float(dlen)
        # 8 box corners (nx, ny, nt): nx,ny in {-0.5, 0.5}, nt in {0, 1} (front=0).
        g = np.array(np.meshgrid([-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0],
                                 indexing="ij")).reshape(3, -1).T
        self.corners = g.astype(np.float64)
        self.edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
                      if int((g[i] != g[j]).sum()) == 1]
        # Axes out of the front-top-left corner; t recedes along the oblique.
        self.axis_origin = np.array([-0.5, -0.5, 0.0])
        self.axes_def = {  # name: (end, label_anchor, BGR color)
            "x": (np.array([0.5, -0.5, 0.0]), np.array([0.60, -0.5, 0.0]), (80, 80, 255)),
            "y": (np.array([-0.5, 0.5, 0.0]), np.array([-0.5, 0.62, 0.0]), (90, 230, 90)),
            "t": (np.array([-0.5, -0.5, 1.0]), np.array([-0.5, -0.5, 1.08]), (230, 230, 60)),
        }
        self._fit_scale()

    def _to_px(self, P):
        """(nx, ny, nt) → (px, py, nt); nt doubles as the front=0 depth key."""
        nt = P[:, 2]
        u = P[:, 0] + nt * self.shift_u
        v = P[:, 1] - nt * self.shift_v          # +t recedes up-and-to-the-right
        return self.ou + self.scale * u, self.ov + self.scale * v, nt

    def _fit_scale(self):
        px = self.corners[:, 0] + self.corners[:, 2] * self.shift_u
        py = self.corners[:, 1] - self.corners[:, 2] * self.shift_v
        margin = 0.12
        self.scale = min((1 - 2 * margin) * self.sub_w / (np.ptp(px) + 1e-9),
                         (1 - 2 * margin) * self.sub_h / (np.ptp(py) + 1e-9))
        self.ou = self.sub_w / 2 - self.scale * (px.min() + px.max()) / 2
        self.ov = self.sub_h / 2 - self.scale * (py.min() + py.max()) / 2

    @staticmethod
    def _norm(x, y, t, W, H):
        nx = np.clip(x / max(W, 1), 0.0, 1.0) - 0.5
        ny = np.clip(y / max(H, 1), 0.0, 1.0) - 0.5
        nt = np.clip(t, 0.0, 1.0)                # front face at t=0 (not centered)
        return np.stack([nx, ny, nt], axis=1)

    def _draw_box(self, canvas, front):
        px, py, _ = self._to_px(self.corners)
        nt = self.corners[:, 2]
        col = (115, 115, 115) if front else (60, 60, 60)
        for i, j in self.edges:
            if (((nt[i] + nt[j]) / 2) <= 0.5) != front:   # t=0 face is nearer
                continue
            cv2.line(canvas, (int(px[i]), int(py[i])), (int(px[j]), int(py[j])),
                     col, 1, cv2.LINE_AA)

    def _draw_axes(self, canvas):
        ox, oy, _ = self._to_px(self.axis_origin[None, :])
        for name, (end, anchor, color) in self.axes_def.items():
            ex, ey, _ = self._to_px(end[None, :])
            cv2.line(canvas, (int(ox[0]), int(oy[0])), (int(ex[0]), int(ey[0])),
                     color, 1, cv2.LINE_AA)
            lx, ly, _ = self._to_px(anchor[None, :])
            cv2.putText(canvas, name, (int(lx[0]) - 4, int(ly[0]) + 4),
                        _FONT, 0.5, color, 1, cv2.LINE_AA)

    def _draw_points(self, canvas, P, pol):
        if len(P) == 0:
            return
        px, py, nt = self._to_px(P)
        order = np.argsort(-nt)                      # back (t=1) first, front last
        px, py, nt, pol = px[order], py[order], nt[order], pol[order]
        xi = np.round(px).astype(np.int64)
        yi = np.round(py).astype(np.int64)
        inb = (xi >= 0) & (xi < self.sub_w) & (yi >= 0) & (yi < self.sub_h)
        xi, yi, nt, pol = xi[inb], yi[inb], nt[inb], pol[inb]
        if not len(xi):
            return
        shade = (0.4 + 0.6 * (1.0 - nt))[:, None]   # recent/front bright, older dim
        red = np.array([60.0, 60.0, 255.0])         # BGR positive polarity
        blue = np.array([255.0, 130.0, 60.0])       # BGR negative polarity
        col = (np.where(pol[:, None] > 0, red, blue) * shade).astype(np.uint8)
        r = self.point_px
        for dy in range(0, r + 1):
            yy = np.clip(yi + dy, 0, self.sub_h - 1)
            for dx in range(0, r + 1):
                xx = np.clip(xi + dx, 0, self.sub_w - 1)
                canvas[yy, xx] = col

    def _subset_idx(self, keep, seed):
        idx = np.flatnonzero(keep)
        if len(idx) > self.max_points:
            idx = np.random.default_rng(seed).choice(idx, self.max_points, replace=False)
        return idx

    def _panel(self, P, pol, keep, title, count, seed):
        canvas = np.zeros((self.sub_h, self.sub_w, 3), np.uint8)
        self._draw_box(canvas, front=False)
        idx = self._subset_idx(keep, seed)
        self._draw_points(canvas, P[idx], pol[idx])
        self._draw_box(canvas, front=True)
        self._draw_axes(canvas)
        _put_label(canvas, title, (8, 22), 0.6)
        _put_label(canvas, f"n={count:,}", (8, self.sub_h - 10), 0.45, (180, 180, 180))
        return canvas

    def render(self, x, y, t, pol, keep_gt, keep_pred, W, H, seed=0):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        t = np.asarray(t, np.float64)
        pol = np.asarray(pol)
        keep_gt = np.asarray(keep_gt, bool)
        keep_pred = np.asarray(keep_pred, bool)
        n = len(x)
        P = self._norm(x, y, t, W, H) if n else np.zeros((0, 3))
        allk = np.ones(n, bool)
        panels = [
            self._panel(P, pol, allk, "All events", n, seed),
            self._panel(P, pol, keep_gt, "GT-filtered", int(keep_gt.sum()), seed),
            self._panel(P, pol, keep_pred, "Pred-filtered", int(keep_pred.sum()), seed),
        ]
        sep = np.full((self.sub_h, self.sep_w, 3), 255, np.uint8)
        return np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)


def _grid_2x2(tl, tr, bl, br):
    """Stack four equal-size BGR panels into a 2x2 grid with thin gaps."""
    h, _w = tl.shape[:2]
    vsep = np.zeros((h, _GRID_GAP, 3), np.uint8)
    top = np.concatenate([tl, vsep, tr], axis=1)
    bot = np.concatenate([bl, vsep, br], axis=1)
    hsep = np.zeros((_GRID_GAP, top.shape[1], 3), np.uint8)
    return np.concatenate([top, hsep, bot], axis=0)


def _compose_frame(rgb, gt_bgr, ev, pred_bgr, vol_row, name, i, n, iou, f1):
    """Build one composite frame: labeled 2x2 grid over the (x,y,t) volume row."""
    tl = rgb.copy(); _put_label(tl, "RGB")
    tr = gt_bgr.copy(); _put_label(tr, "GT Mask")
    bl = ev.copy(); _put_label(bl, "Events")
    br = pred_bgr.copy()
    plabel = "Prediction"
    if f1 is not None:
        plabel += f"  F1={f1:.3f}"
    if iou is not None:
        plabel += f"  IoU={iou:.2f}"
    _put_label(br, plabel)

    grid = _grid_2x2(tl, tr, bl, br)
    gw = grid.shape[1]
    if vol_row.shape[1] != gw:
        vol_row = cv2.resize(vol_row, (gw, vol_row.shape[0]))
    gap = np.zeros((_GRID_GAP, gw, 3), np.uint8)
    strip = np.zeros((34, gw, 3), np.uint8)
    _put_label(strip, name, (10, 24), 0.55, (200, 200, 200))
    cnt = f"frame {i}/{max(n - 1, 0)}"
    (tw, _), _ = cv2.getTextSize(cnt, _FONT, 0.55, 1)
    _put_label(strip, cnt, (gw - tw - 10, 24), 0.55, (200, 200, 200))
    return np.concatenate([grid, gap, vol_row, strip], axis=0)


def _open_writer(out_path, w, h, fps):
    fps = max(1.0, float(fps))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {out_path}")
    return writer


@torch.no_grad()
def _iter_payloads_sparse(model, val_set, seq_dir, frame_getter, positions, device):
    """Yield one per-frame payload dict for the event-native ``EventSparseSeg`` model.

    Each payload carries the four 2x2-grid panels (rgb / gt / events / pred), the
    per-event ``(x, y, t)`` volume arrays with GT/prediction keep-masks, the per-
    frame IoU + per-event F1, and the timing/MAC/event counts for the cost report.
    The dense prediction mask is the per-event logits rasterized back to ``(H, W)``
    (max-pool per pixel, same as training's ``event_pred_to_dense_iou``); the 3D
    pred-filter uses the matching per-event sigmoid > 0.5 so the panels agree.
    """
    prof = SparseOpProfiler(model.model).attach()
    prof.exact = False
    try:
        for pos in positions:
            coords, feats, times, labels, dense_mask, meta = frame_getter(pos)
            H, W = int(dense_mask.shape[-2]), int(dense_mask.shape[-1])
            ev_panel = events_panel_from_sites(coords, feats, H, W)   # (H, W, 3) BGR
            gt_np = (dense_mask.detach().cpu().numpy() > 0).astype("uint8")
            gt_bgr = mask_to_bgr(gt_np * 255, H, W)
            rgb = _load_rgb_frame(seq_dir, (meta or {}).get("frame_index"), H, W)

            if coords.shape[0] == 0:
                # Empty window: nothing to predict — emit a background frame.
                empty = np.zeros(0)
                vol = dict(x=empty, y=empty, t=empty, pol=empty,
                           keep_gt=np.zeros(0, bool), keep_pred=np.zeros(0, bool),
                           W=W, H=H)
                yield dict(rgb=rgb, gt=gt_bgr, events=ev_panel,
                           pred=mask_to_bgr(np.zeros((H, W), "uint8"), H, W), vol=vol,
                           iou=None, f1=None, latency=0.0, macs=None, nev=None,
                           counted=False)
                continue

            batch = _sparse_batch_from_sample(
                (coords, feats, times, labels, dense_mask, meta), device)
            prof.reset()
            _sync(device)
            t0 = time.perf_counter()
            logits = model.model(batch)                          # (N,) per event
            _sync(device)
            dt = time.perf_counter() - t0
            upper, _exact = prof.frame_macs()

            logits = logits.detach()
            ev_prob = torch.sigmoid(logits.squeeze(-1) if logits.dim() == 2 else logits)
            raster = rasterize_events_to_logits(
                batch.coords, logits, batch.batch_idx, 1, H, W)
            pred_np = (torch.sigmoid(raster[0]) > 0.5).cpu().numpy().astype("uint8")
            pred_bgr = mask_to_bgr(pred_np * 255, H, W)

            vol = dict(
                x=batch.coords[:, 0].cpu().numpy(),
                y=batch.coords[:, 1].cpu().numpy(),
                t=batch.times.detach().cpu().numpy(),
                pol=batch.feats[:, 0].detach().cpu().numpy(),
                keep_gt=(batch.labels.detach().cpu().numpy() > 0.5),
                keep_pred=(ev_prob.cpu().numpy() > 0.5),
                W=W, H=H,
            )
            iou = None
            if float(dense_mask.sum()) > 0:
                iou = float(event_pred_to_dense_iou(
                    batch.coords, logits, batch.batch_idx, batch.dense_mask, 1, H, W))
            f1 = float(event_f1(logits, batch.labels))
            yield dict(rgb=rgb, gt=gt_bgr, events=ev_panel, pred=pred_bgr, vol=vol,
                       iou=iou, f1=f1, latency=dt, macs=upper,
                       nev=int(batch.feats.shape[0]), counted=True)
    finally:
        prof.detach()


@torch.no_grad()
def _bin_subtime(c, y, x):
    """Deterministic per-cell offset in ``[0, 1)`` used to spread voxel-binned
    events across the width of their time bin.

    A dense voxel has already collapsed every event's timestamp into one of
    ``n_bins`` temporal bins, so placing each point at the bin center renders the
    ``(x, y, t)`` volume as ``n_bins`` discrete planes with empty gaps. Offsetting
    each cell within its bin turns those planes into a continuous tube. The offset
    is a spatial hash of the cell index ``(bin, y, x)`` — not random — so a static
    scene's points keep the same sub-bin time across frames (no shimmer).
    """
    c = c.astype(np.uint64)
    y = y.astype(np.uint64)
    x = x.astype(np.uint64)
    h = ((c * np.uint64(73856093)) ^ (y * np.uint64(19349663))
         ^ (x * np.uint64(83492791)))
    return (h % np.uint64(1000003)).astype(np.float64) / 1000003.0


def _dense_volume(get_events, pos, voxel, gt_np, pred, W, H,
                  split_polarity=False):
    """Build the ``(x, y, t)`` volume dict for one dense-model frame.

    Prefers the dataset's TRUE per-event timestamps (``get_events(pos)`` →
    ``frame_events``): the same windowed events the voxel was built from, with
    continuous time and sub-pixel coords. Datasets that don't expose it fall back
    to the voxel's nonzero cells, spreading each across its time bin
    (``_bin_subtime``) so the tube is still continuous rather than ``n_bins``
    discrete planes. GT/pred membership is a nearest-pixel lookup into the dense
    masks. Returns empty arrays when the window has no events.

    ``split_polarity``: set True for the two-channel-per-polarity voxel layout
    (dataset ``polarity_mode="two_channel"``) — the first half of the channels
    holds ON counts, the second half OFF counts, so cell polarity comes from
    the channel block (all values are non-negative) and the temporal bin is
    the channel index modulo the per-polarity bin count.
    """
    if get_events is not None:
        ev = get_events(pos)
        x, y, t, pol = ev["x"], ev["y"], ev["t"], ev["p"]
        if len(t):
            xi = np.clip(np.round(x).astype(np.int64), 0, W - 1)
            yi = np.clip(np.round(y).astype(np.int64), 0, H - 1)
            return dict(x=x, y=y, t=t, pol=pol,
                        keep_gt=gt_np[yi, xi] > 0, keep_pred=pred[yi, xi] > 0,
                        W=W, H=H)
    else:
        v = voxel.detach().cpu()
        if v.dim() == 4:
            v = v[0]
        nz = (v != 0).nonzero(as_tuple=False).numpy()           # (M, 3) -> (c, y, x)
        if len(nz):
            cc, yy, xx = nz[:, 0], nz[:, 1], nz[:, 2]
            n_bins = int(v.shape[0])
            vals = v[cc, yy, xx].numpy()
            if split_polarity:
                n_bins //= 2
                pol = np.where(cc < n_bins, 1.0, -1.0)
                cc = cc % n_bins
            else:
                pol = np.sign(vals)
            # Spread each cell across its bin's time slab ``[c, c+1)/n_bins`` so
            # the volume is a continuous tube instead of ``n_bins`` stacked planes
            # (see ``_bin_subtime``). t stays in ``[0, 1)``.
            t = (cc.astype(np.float64) + _bin_subtime(cc, yy, xx)) / n_bins
            return dict(x=xx.astype(np.float64), y=yy.astype(np.float64),
                        t=t, pol=pol,
                        keep_gt=gt_np[yy, xx] > 0, keep_pred=pred[yy, xx] > 0, W=W, H=H)
    empty = np.zeros(0)
    return dict(x=empty, y=empty, t=empty, pol=empty,
                keep_gt=np.zeros(0, bool), keep_pred=np.zeros(0, bool), W=W, H=H)


def _iter_payloads_dense(model, val_set, seq_dir, frame_getter, positions, device):
    """Yield per-frame payloads for the dense voxel models.

    The ``(x, y, t)`` volume uses the dataset's TRUE per-event timestamps when
    available (``frame_events``), else reconstructs continuous points from the
    voxel bins; see ``_dense_volume``. RGB is loaded from the sample ``meta``
    frame index when present, else a placeholder.
    """
    streaming = hasattr(model.model, "step")
    get_events = getattr(val_set, "frame_events", None)
    split_pol = getattr(val_set, "polarity_mode", "signed") == "two_channel"
    state = None
    for pos in positions:
        sample = frame_getter(pos)
        voxel, mask = sample[0], sample[1]                       # (C,H,W), (H,W)
        meta = sample[-1] if isinstance(sample[-1], dict) else {}
        H, W = int(mask.shape[-2]), int(mask.shape[-1])
        inp = voxel.unsqueeze(0).to(device)
        _sync(device)
        t0 = time.perf_counter()
        if streaming:
            logits, state = model.model.step(inp, state)
        else:
            logits = model.model(inp)
        _sync(device)
        dt = time.perf_counter() - t0

        pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype("uint8")
        if pred.shape != (H, W):
            pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
        ev_panel = events_panel_from_voxel(voxel, split_polarity=split_pol)  # (H, W, 3) BGR
        gt_np = (mask.detach().cpu().numpy() > 0).astype("uint8")
        gt_bgr = mask_to_bgr(gt_np * 255, H, W)
        pred_bgr = mask_to_bgr(pred * 255, H, W)
        rgb = _load_rgb_frame(seq_dir, meta.get("frame_index"), H, W)

        vol = _dense_volume(get_events, pos, voxel, gt_np, pred, W, H,
                            split_polarity=split_pol)
        iou = None
        if float(mask.sum()) > 0:
            iou = float(binary_iou(logits.detach(), mask.unsqueeze(0).to(device)))
        yield dict(rgb=rgb, gt=gt_bgr, events=ev_panel, pred=pred_bgr, vol=vol,
                   iou=iou, f1=None, latency=dt, macs=None, nev=None, counted=True)


@torch.no_grad()
def render_and_write_sequence(model, val_set, s_idx, max_frames, device, sparse,
                              vol_renderer, out_path):
    """Stream one held-out sequence to ``out_path`` as the 2x2-grid + (x,y,t)
    volume video. Frames are composed and written one at a time (no whole-sequence
    panel buffering), so memory stays flat regardless of length/resolution.

    Returns ``(name, n_frames, ious, fps, inf_time, cost)`` for the run cost report,
    or ``None`` if the sequence has no frames.
    """
    positions = _sequence_positions(val_set, s_idx, max_frames)
    if positions is None:
        return None
    name = val_set.sequences[s_idx].name
    fps = _sequence_fps(val_set, s_idx)
    seq_dir = val_set.sequences[s_idx]
    frame_getter = getattr(val_set, "frame_sample", None) or val_set.__getitem__
    payloads = (_iter_payloads_sparse if sparse else _iter_payloads_dense)(
        model, val_set, seq_dir, frame_getter, positions, device)

    writer = None
    ious, latencies, frame_macs, n_events = [], [], [], []
    inf_time, n_written, n = 0.0, 0, len(positions)
    try:
        for i, pl in enumerate(payloads):
            vol_row = vol_renderer.render(seed=i, **pl["vol"])
            frame = _compose_frame(pl["rgb"], pl["gt"], pl["events"], pl["pred"],
                                   vol_row, name, i, n, pl["iou"], pl["f1"])
            if writer is None:
                writer = _open_writer(out_path, frame.shape[1], frame.shape[0], fps)
            writer.write(frame)
            n_written += 1
            ious.append(pl["iou"])
            if pl.get("counted", True):
                inf_time += pl["latency"]
                latencies.append(pl["latency"])
                if pl["macs"] is not None:
                    frame_macs.append(pl["macs"])
                if pl["nev"] is not None:
                    n_events.append(pl["nev"])
    finally:
        if writer is not None:
            writer.release()
    cost = {"latencies": latencies, "frame_macs": frame_macs, "n_events": n_events}
    return name, n_written, ious, fps, inf_time, cost


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


def _load_model_interface(ckpt_path, cfg, device):
    """Build a ``ModelInterface`` from the config and load checkpoint weights,
    tolerating stale diagnostic buffers.

    Some models stash diagnostics on themselves during the forward pass (e.g.
    ``EventSSMSegStream._freqs`` / ``._dyn_energy`` — learned Koopman frequencies
    and DMD energies). Older checkpoints persisted those in the state_dict; the
    current code keeps them as plain attributes, so they show up as harmless
    *unexpected* keys. We drop them. We still refuse to load when keys are
    *missing*, since that means the architecture built from the config does not
    match the checkpoint and predictions would be silently wrong.
    """
    model = ModelInterface(
        model_cfg=cfg.MODEL,
        optimizer_cfg=cfg.OPTIMIZER,
        scheduler_cfg=cfg.SCHEDULER,
        training_cfg=cfg.TRAINING,
        data_cfg=cfg.DATA,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"checkpoint is missing {len(missing)} weight(s) the model expects "
            f"(config/architecture mismatch). First few: {list(missing)[:8]}"
        )
    if unexpected:
        print(f"[infer] ignoring {len(unexpected)} unexpected checkpoint key(s) "
              f"(stale diagnostic buffers): {list(unexpected)[:8]}")
    return model


def main():
    args = parse_args()

    cfg, _tracker = load_config_with_schema(args.config_path)
    seed = args.seed if args.seed is not None else int(getattr(cfg.TRAINING, "seed", 0))

    # Build only the validation split (DataInterface builds all three; that's fine
    # and cheap — the index build is metadata-only).
    dm = DataInterface(data_cfg=cfg.DATA)
    val_set = dm.validation_set

    device = torch.device(args.device)
    model = _load_model_interface(args.ckpt, cfg, device)
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

    # Panel resolution → 2x2 grid width → one shared (x, y, t) volume renderer
    # (reused across every sequence; the matplotlib-free cv2 projector is cheap to
    # build but cheaper to keep around).
    if sparse:
        _h0, _w0 = int(sample0[4].shape[-2]), int(sample0[4].shape[-1])
    else:
        _h0, _w0 = int(sample0[1].shape[-2]), int(sample0[1].shape[-1])
    vol_renderer = Volume3DRenderer(total_w=2 * _w0 + _GRID_GAP,
                                    alpha=args.vol_alpha, dlen=args.vol_dlen)

    tmp_dir = Path(tempfile.mkdtemp(prefix="val_previews_"))
    written = []
    total_frames = 0
    total_inf_time = 0.0  # seconds of pure model time, summed over all frames
    all_latencies = []    # per-frame model seconds, across every sequence
    all_frame_macs = []   # per-frame upper-bound MACs (sparse path only)
    all_n_events = []     # per-frame active-event count (sparse path only)
    for rank_i, s_idx in enumerate(chosen):
        out_path = tmp_dir / f"{val_set.sequences[s_idx].name}.mp4"
        out = render_and_write_sequence(
            model, val_set, s_idx, args.max_frames, device, sparse,
            vol_renderer, out_path)
        if out is None:
            print(f"[infer] ({rank_i + 1}/{k}) seq#{s_idx}: no frames, skipped")
            continue
        name, n_frames, ious, fps, inf_time, cost = out
        total_frames += n_frames
        total_inf_time += inf_time
        all_latencies.extend(cost.get("latencies", []))
        all_frame_macs.extend(cost.get("frame_macs", []))
        all_n_events.extend(cost.get("n_events", []))
        written.append(out_path)
        per_frame_ms = 1e3 * inf_time / max(1, n_frames)
        seq_line = (f"[infer] ({rank_i + 1}/{k}) wrote {name}.mp4 "
                    f"({n_frames} frames, {fps:.1f} fps; "
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
