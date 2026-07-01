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


def events_panel_from_voxel(voxel: torch.Tensor,
                            split_polarity: bool = False) -> np.ndarray:
    """Voxel ``(B, H, W)`` → BGR panel: gray bg, red/blue net polarity.

    The voxel bins are summed to a single signed accumulation map; a pixel is
    drawn red where the net event polarity over the window is positive and blue
    where it is negative. This visualizes exactly the input the student saw
    (rather than re-reading raw events), so the panel and the prediction are
    always aligned.

    ``split_polarity``: set True for the two-channel-per-polarity voxel layout
    (dataset ``polarity_mode="two_channel"``): channels ``[0:B/2]`` hold ON
    counts and ``[B/2:]`` OFF counts, so the net map is their difference
    (the default single-block sum would always be non-negative).
    """
    v = voxel.detach().cpu()
    if split_polarity:
        half = v.shape[0] // 2
        acc = (v[:half].sum(dim=0) - v[half:].sum(dim=0)).numpy()
    else:
        acc = v.sum(dim=0).numpy()  # (H, W), signed
    h, w = acc.shape
    img = np.full((h, w, 3), _GRAY, dtype=np.uint8)
    img[acc > 0] = (0, 0, 255)   # BGR red  = positive polarity
    img[acc < 0] = (255, 0, 0)   # BGR blue = negative polarity
    return img


def events_panel_from_sites(coords: torch.Tensor, feats: torch.Tensor,
                            h: int, w: int) -> np.ndarray:
    """Sparse per-event sites → BGR panel: gray bg, red/blue per net polarity.

    The event-native counterpart of :func:`events_panel_from_voxel`: instead of a
    dense voxel grid, the active sites ``coords`` ``(M, 2)`` ``(x, y)`` are painted
    onto a gray canvas, red where the site's mean signed polarity ``feats[:, 0]``
    is positive and blue where negative. Visualizes exactly the sparse input the
    EventSparseSeg model saw, so the panel and the prediction stay aligned.
    """
    img = np.full((h, w, 3), _GRAY, dtype=np.uint8)
    if coords.numel() == 0:
        return img
    x = coords[:, 0].long().cpu().numpy()
    y = coords[:, 1].long().cpu().numpy()
    pol = feats[:, 0].detach().cpu().numpy()
    in_b = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x, y, pol = x[in_b], y[in_b], pol[in_b]
    img[y[pol > 0], x[pol > 0]] = (0, 0, 255)   # BGR red  = positive polarity
    img[y[pol < 0], x[pol < 0]] = (255, 0, 0)   # BGR blue = negative polarity
    return img


def _paint_sites(canvas: np.ndarray, x: np.ndarray, y: np.ndarray,
                 color, r: int = 0) -> None:
    """Paint events at ``(x, y)`` with ``color`` (optional 1-px dilation for visibility)."""
    h, w = canvas.shape[:2]
    canvas[y, x] = color
    if r:
        for dx in (-r, 0, r):
            for dy in (-r, 0, r):
                canvas[np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)] = color


def event_pred_triptych(
    coords: torch.Tensor,
    feats: torch.Tensor,
    labels: torch.Tensor,
    pred: torch.Tensor,
    dense_mask: torch.Tensor,
    h: int,
    w: int,
    tag: str = "",
    f1: Optional[float] = None,
) -> np.ndarray:
    """Per-event 3-panel BGR frame (the val-preview analog of tools/viz_per_event_labels_video.py).

    Panels, painted directly on the event sites (event-native, no rasterization):
      1. **Prediction** — green where the model predicts foreground, gray where bg.
      2. **GT label**   — green where ``label == 1`` (``mask[y,x]``), gray where bg.
      3. **Context**    — events (gray) + the GT FLIR mask OUTLINE (yellow).

    ``pred`` and ``labels`` are per-event ``(N,)`` in ``{0,1}``; ``dense_mask`` is the
    ``(H, W)`` teacher mask used only for the outline in panel 3.
    """
    x = coords[:, 0].long().cpu().numpy()
    y = coords[:, 1].long().cpu().numpy()
    inb = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x, y = x[inb], y[inb]
    lab = (labels.detach().cpu().numpy()[inb] >= 0.5)
    prd = (pred.detach().cpu().numpy()[inb] >= 0.5)
    mask = (dense_mask.detach().cpu().numpy() > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    font, GREEN, GRAY, YELLOW, WHITE = cv2.FONT_HERSHEY_SIMPLEX, (0, 255, 0), (90, 90, 90), (0, 255, 255), (255, 255, 255)

    p1 = np.zeros((h, w, 3), np.uint8)
    _paint_sites(p1, x[~prd], y[~prd], GRAY)
    _paint_sites(p1, x[prd], y[prd], GREEN, r=1)
    head = f"prediction (green=fg {prd.mean() * 100:.1f}%)" if prd.size else "prediction (no events)"
    if f1 is not None:
        head += f"  F1={f1:.3f}"
    cv2.putText(p1, head, (8, 22), font, 0.55, WHITE, 1, cv2.LINE_AA)

    p2 = np.zeros((h, w, 3), np.uint8)
    _paint_sites(p2, x[~lab], y[~lab], GRAY)
    _paint_sites(p2, x[lab], y[lab], GREEN, r=1)
    cv2.putText(p2, f"GT label (green=fg {lab.mean() * 100:.1f}%)" if lab.size else "GT label",
                (8, 22), font, 0.55, WHITE, 1, cv2.LINE_AA)

    p3 = np.zeros((h, w, 3), np.uint8)
    _paint_sites(p3, x, y, (120, 120, 120))
    cv2.drawContours(p3, contours, -1, YELLOW, 2)
    cv2.putText(p3, "events (gray) + GT mask outline (yellow)", (8, 22), font, 0.55, WHITE, 1, cv2.LINE_AA)
    if tag:
        cv2.putText(p3, tag, (8, h - 12), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    sep = np.full((h, 4, 3), 255, np.uint8)
    return np.concatenate([p1, sep, p2, sep, p3], axis=1)


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


def write_panels_video(
    panel_seqs: Sequence[Sequence[np.ndarray]],
    labels: Sequence[str],
    out_path: Path,
    fps: float = 30.0,
    title: str = "",
    per_frame_iou: Optional[Sequence[Optional[float]]] = None,
    iou_panel: Optional[int] = None,
) -> None:
    """Compose ``K`` labeled panels side by side into a single MP4.

    ``panel_seqs`` is a list of ``K`` equal-length sequences; each frame element
    is either a single-channel mask (uint8 0/255, converted with ``mask_to_bgr``)
    or an already-BGR ``(H, W, 3)`` panel. ``labels`` gives the ``K`` panel
    captions. When ``per_frame_iou`` and ``iou_panel`` are supplied, the IoU is
    appended to ``labels[iou_panel]`` per frame. Panel geometry, the two-row
    label strip, and the title/frame-counter are shared with the (now thin)
    ``write_triptych_video`` wrapper.
    """
    k = len(panel_seqs)
    if k == 0:
        return
    n = min(len(s) for s in panel_seqs)
    if n == 0:
        return
    h, w = panel_seqs[0][0].shape[:2]

    gap = 10
    # Two stacked text rows under the panels: row 1 holds the per-panel labels,
    # row 2 holds the long sequence title and the frame counter. Keeping them on
    # separate rows is what stops the right-aligned title from being drawn on top
    # of the panel labels.
    row_h = 28
    label_h = 2 * row_h
    out_w = w * k + (k - 1) * gap
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
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            row1_y = h + 22
            for p in range(k):
                frame = panel_seqs[p][i]
                bgr = frame if frame.ndim == 3 else mask_to_bgr(frame, h, w)
                x0 = p * (w + gap)
                canvas[:h, x0 : x0 + w] = bgr

                label = labels[p]
                if (iou_panel is not None and p == iou_panel
                        and per_frame_iou is not None and per_frame_iou[i] is not None):
                    label = f"{label}  IoU={per_frame_iou[i]:.2f}"
                cv2.putText(canvas, label, (x0 + 10, row1_y), font, 0.7,
                            (255, 255, 255), 1, cv2.LINE_AA)

            # Row 2: sequence title (left) and frame counter (right), on their
            # own row so the long title never overlaps the panel labels above.
            row2_y = h + 22 + row_h
            if title:
                cv2.putText(canvas, title, (10, row2_y), font, 0.55,
                            (200, 200, 200), 1, cv2.LINE_AA)
            frame_meta = f"frame {i}/{n - 1}"
            (tw, _), _ = cv2.getTextSize(frame_meta, font, 0.55, 1)
            cv2.putText(canvas, frame_meta, (out_w - tw - 10, row2_y), font, 0.55,
                        (200, 200, 200), 1, cv2.LINE_AA)

            writer.write(canvas)
    finally:
        writer.release()


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

    Thin wrapper over ``write_panels_video``. All three sequences must be the
    same length and the panels the same H×W (the GT/pred masks are uint8 0/255
    single-channel; the event panels are BGR). ``per_frame_iou`` (optional) is
    overlaid on the prediction panel.
    """
    write_panels_video(
        [event_panels, gt_masks, pred_masks],
        ["Events", "Teacher (GT)", "Prediction"],
        out_path, fps=fps, title=title,
        per_frame_iou=per_frame_iou, iou_panel=2,
    )
