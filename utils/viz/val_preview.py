"""Qualitative validation preview videos for the event→mask student.

Mirrors the three-panel MP4 style of ``data/sam2_pseudo_labels.py``'s
``write_preview_video``, but the panels are **Events | Teacher (GT) Mask |
Student Prediction** so you can eyeball — over a held-out sequence — how well
the distilled event-only student tracks the cached SAM 2 target.

Self-contained on purpose: only ``cv2`` / ``numpy`` / ``torch`` are imported,
so pulling this in from ``ModelInterface`` never drags in SAM 2 or
GroundingDINO (which ``data/sam2_pseudo_labels.py`` imports lazily for the
offline teacher run).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch


# Same palette as data/sam2_pseudo_labels.py::_render_event_panel and
# tools/visualize_rgb_events.py (gray bg, red=positive, blue=negative).
_GRAY = 128


def events_panel_from_voxel(voxel: torch.Tensor) -> np.ndarray:
    """Signed voxel ``(B, H, W)`` → BGR panel: gray bg, red/blue net polarity.

    The voxel bins are summed to a single signed accumulation map; a pixel is
    drawn red where the net event polarity over the window is positive and blue
    where it is negative. This visualizes exactly the input the student saw
    (rather than re-reading raw events), so the panel and the prediction are
    always aligned.
    """
    acc = voxel.sum(dim=0).detach().cpu().numpy()  # (H, W), signed
    h, w = acc.shape
    img = np.full((h, w, 3), _GRAY, dtype=np.uint8)
    img[acc > 0] = (0, 0, 255)   # BGR red  = positive polarity
    img[acc < 0] = (255, 0, 0)   # BGR blue = negative polarity
    return img


def mask_to_bgr(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Binary/uint8 mask → 3-channel BGR, resized (nearest) to ``(h, w)``."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def infer_fps(flir_t: Optional[np.ndarray], default: float = 30.0) -> float:
    """Frames-per-second from a FLIR timestamp array (unit auto-detected)."""
    if flir_t is None or np.size(flir_t) < 2:
        return default
    rng = float(flir_t[-1] - flir_t[0])
    if rng <= 0:
        return default
    unit = 1e9 if rng > 1e9 else 1e6 if rng > 1e6 else 1e3 if rng > 1e3 else 1.0
    duration = rng / unit
    n = int(np.size(flir_t))
    return (n - 1) / duration if duration > 0 else default


def write_triptych_video(
    event_panels: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    out_path: Path,
    fps: float = 30.0,
    title: str = "",
    per_frame_iou: Optional[Sequence[Optional[float]]] = None,
) -> None:
    """Compose an Events | Teacher (GT) | Prediction MP4 with panel labels.

    All three sequences must be the same length and the panels the same H×W
    (the GT/pred masks are uint8 0/255 single-channel; the event panels are
    BGR). ``per_frame_iou`` (optional) is overlaid on the prediction panel.
    """
    n = min(len(event_panels), len(gt_masks), len(pred_masks))
    if n == 0:
        return
    h, w = event_panels[0].shape[:2]

    gap = 10
    label_h = 40
    out_w = w * 3 + 2 * gap
    out_h = h + label_h
    # Guard against a degenerate inferred fps (e.g. odd timestamp units) that
    # would write an unplayable file.
    fps = max(1.0, float(fps))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {out_path}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        for i in range(n):
            gt_bgr = mask_to_bgr(gt_masks[i], h, w)
            pred_bgr = mask_to_bgr(pred_masks[i], h, w)

            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            canvas[:h, :w] = event_panels[i]
            canvas[:h, w + gap : 2 * w + gap] = gt_bgr
            canvas[:h, 2 * w + 2 * gap : 3 * w + 2 * gap] = pred_bgr

            cv2.putText(canvas, "Events", (10, h + 28), font, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Teacher (GT)", (w + gap + 10, h + 28), font, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
            pred_label = "Prediction"
            if per_frame_iou is not None and per_frame_iou[i] is not None:
                pred_label = f"Prediction  IoU={per_frame_iou[i]:.2f}"
            cv2.putText(canvas, pred_label, (2 * w + 2 * gap + 10, h + 28), font,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)

            meta = f"{title}  frame {i}/{n - 1}" if title else f"frame {i}/{n - 1}"
            (tw, _), _ = cv2.getTextSize(meta, font, 0.55, 1)
            cv2.putText(canvas, meta, (out_w - tw - 10, h + 28), font, 0.55,
                        (200, 200, 200), 1, cv2.LINE_AA)

            writer.write(canvas)
    finally:
        writer.release()
