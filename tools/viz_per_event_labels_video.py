"""Render the per-event GROUND-TRUTH label visualization as a video over a sequence.

Three panels per frame, same as tools/viz_per_event_labels.py but across all
action frames of one sequence (in time order) so the depth-dependent parallax
between the events and the FLIR mask is visible as a wobble:

  (1) events colored by per-event label  label=mask[y,x]  (GREEN=fg, GRAY=bg)
  (2) GT FLIR mask (green fill) + all events (red)
  (3) events (gray) + mask OUTLINE (yellow)  <- does the boundary track the hand?

Output: <repo>/per_event_labels_preview.mp4 (override with --out).
Usage:
    PYTHONPATH=<repo> python tools/viz_per_event_labels_video.py
    PYTHONPATH=<repo> python tools/viz_per_event_labels_video.py --sequence sequence_..._bottle
"""
import argparse
from collections import Counter

import numpy as np
import cv2

from data.hand_event_stream_dataset import HandEventStreamDataset

ROOT = "/fs/nexus-projects/DVS_Actions/NatureRoboticsDataNew"
MASK = "/fs/nexus-projects/DVS_Actions/NatureRoboticsDataNewTeacherMasks"


def paint(canvas, x, y, color, r=0):
    canvas[y, x] = color
    if r:
        for dx in (-r, 0, r):
            for dy in (-r, 0, r):
                xi = np.clip(x + dx, 0, canvas.shape[1] - 1)
                yi = np.clip(y + dy, 0, canvas.shape[0] - 1)
                canvas[yi, xi] = color


def render(coords, labels, dense, H, W, tag):
    x = coords[:, 0].numpy(); y = coords[:, 1].numpy()
    lab = labels.numpy(); mask = dense.numpy().astype(bool)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    fg = lab >= 0.5

    p1 = np.zeros((H, W, 3), np.uint8)
    paint(p1, x[~fg], y[~fg], (90, 90, 90))
    paint(p1, x[fg], y[fg], (0, 255, 0), r=1)
    cv2.putText(p1, f"per-event GT label (green=fg {fg.mean()*100:.1f}%)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    p2 = np.zeros((H, W, 3), np.uint8)
    p2[mask] = (0, 110, 0)
    paint(p2, x, y, (0, 0, 255))
    cv2.putText(p2, "GT FLIR mask (green) + events (red)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    p3 = np.zeros((H, W, 3), np.uint8)
    paint(p3, x, y, (120, 120, 120))
    cv2.drawContours(p3, contours, -1, (0, 255, 255), 2)
    cv2.putText(p3, "events (gray) + mask outline (yellow)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(p3, tag, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    sep = np.full((H, 4, 3), 255, np.uint8)
    return np.concatenate([p1, sep, p2, sep, p3], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default=None, help="sequence folder name; default = most action frames")
    ap.add_argument("--out", default=f"{ROOT.rsplit('/',2)[0]}")  # placeholder, set below
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--max_frames", type=int, default=300)
    args = ap.parse_args()
    out_path = "/fs/nexus-scratch/tuxunlu/git/EventBasedObjectDetection/per_event_labels_preview.mp4"

    ds = HandEventStreamDataset(
        root_dir=ROOT, purpose="train", window_ms=36.0,
        image_height=480, image_width=640, held_out_subject="tuxun",
        require_teacher=True, action_only=True, mask_root=MASK, max_events=150000,
    )
    H, W = 480, 640

    # group dataset positions by sequence, pick the requested one or the longest.
    by_seq = {}
    for pos, (s_idx, f_idx) in enumerate(ds.index):
        by_seq.setdefault(s_idx, []).append((f_idx, pos))
    if args.sequence is not None:
        s_idx = next(i for i, s in enumerate(ds.sequences) if s.name == args.sequence)
    else:
        s_idx = max(by_seq, key=lambda k: len(by_seq[k]))
    positions = [p for _, p in sorted(by_seq[s_idx])][: args.max_frames]
    seq_name = ds.sequences[s_idx].name
    print(f"[seq] {seq_name}  frames={len(positions)}")

    writer = None
    fg_hist = []
    for k, pos in enumerate(positions):
        coords, feats, times, labels, dense, meta = ds[pos]
        if coords.shape[0] == 0:
            continue
        frame = render(coords, labels, dense, H, W,
                       f"{seq_name}  f{meta['frame_index']:04d}  ({k+1}/{len(positions)})")
        fg_hist.append(float((labels >= 0.5).float().mean()))
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, args.fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    if writer is not None:
        writer.release()
    print(f"[ok] wrote {out_path}  ({len(positions)} frames @ {args.fps:.0f}fps, "
          f"mean fg={np.mean(fg_hist)*100:.1f}%)")


if __name__ == "__main__":
    main()
