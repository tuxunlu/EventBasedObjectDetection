"""
Side-by-side visualisation of FLIR RGB frames and Prophesee events.

For each FLIR frame timestamp, accumulates events in a ±window/2 window and
renders them as a polarity image next to the FLIR frame. Writes an MP4 video.

Usage:
    python tools/visualize_rgb_events.py \
        /fs/nexus-projects/DVS_Actions/NatureRoboticsDataNew/sequence_haowen1_SIDE_DYNAMIC_LIGHT_bottle \
        --output viz_bottle.mp4 --window_ms 33 --fps 30
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def detect_unit_per_second(t_range: float) -> float:
    """Heuristic: guess timestamp unit from total range, assuming ~seconds-scale clips."""
    if t_range > 1e9:
        return 1e9  # nanoseconds
    if t_range > 1e6:
        return 1e6  # microseconds
    if t_range > 1e3:
        return 1e3  # milliseconds
    return 1.0      # seconds


def load_frame(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read frame: {path}")
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def render_events(xy: np.ndarray, p: np.ndarray, h: int, w: int) -> np.ndarray:
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    if len(xy) == 0:
        return img
    x = np.clip(xy[:, 0].astype(np.int32), 0, w - 1)
    y = np.clip(xy[:, 1].astype(np.int32), 0, h - 1)
    pos = p > 0
    img[y[~pos], x[~pos]] = (255, 0, 0)  # BGR blue = negative polarity
    img[y[pos], x[pos]] = (0, 0, 255)    # BGR red  = positive polarity
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_dir", type=str,
                        help="Path to sequence_<...> root containing proc/")
    parser.add_argument("--output", type=str, default="rgb_events_viz.mp4")
    parser.add_argument("--window_ms", type=float, default=33.0,
                        help="Event accumulation window centred on each frame, in ms")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Playback FPS for output video")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit number of frames to render (for quick checks)")
    parser.add_argument("--start_frame", type=int, default=0)
    args = parser.parse_args()

    seq = Path(args.sequence_dir)
    proc = seq / "proc"
    flir_dir = proc / "flir"
    events_dir = proc / "events"
    if not proc.is_dir():
        raise FileNotFoundError(f"No proc/ under {seq}")

    flir_t = np.load(flir_dir / "flir_t.npy")
    frame_files = sorted((flir_dir / "frame").glob("*.png"))
    if len(frame_files) != len(flir_t):
        print(f"[warn] {len(frame_files)} png frames vs {len(flir_t)} timestamps; "
              f"using min length")
    n_total = min(len(frame_files), len(flir_t))

    events_t = np.load(events_dir / "events_t.npy", mmap_mode="r")
    events_xy = np.load(events_dir / "events_xy.npy", mmap_mode="r")
    events_p = np.load(events_dir / "events_p.npy", mmap_mode="r")

    t_range = float(events_t[-1] - events_t[0])
    unit_per_second = detect_unit_per_second(t_range)
    window_units = (args.window_ms / 1000.0) * unit_per_second

    print(f"sequence       : {seq.name}")
    print(f"frames         : {n_total} (FLIR)")
    print(f"events         : {len(events_t):,}")
    print(f"flir_t range   : {flir_t[0]:.3f} .. {flir_t[-1]:.3f}")
    print(f"events_t range : {events_t[0]:.3f} .. {events_t[-1]:.3f}")
    print(f"detected unit  : {unit_per_second:g} (=1 second)")
    print(f"event window   : ±{args.window_ms/2:.1f} ms ({window_units:g} units)")

    boundaries_path = proc / "boundaries.json"
    boundaries = json.loads(boundaries_path.read_text()) if boundaries_path.exists() else []

    sample = load_frame(frame_files[0])
    h, w, _ = sample.shape
    print(f"resolution     : {w}x{h}")

    n_start = max(0, args.start_frame)
    n_end = n_total if args.max_frames is None else min(n_total, n_start + args.max_frames)

    gap = 10
    label_h = 40
    out_w = w * 2 + gap
    out_h = h + label_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {args.output}")

    for i in tqdm(range(n_start, n_end), desc="rendering"):
        frame = load_frame(frame_files[i])
        t_center = float(flir_t[i])
        idx_lo = int(np.searchsorted(events_t, t_center - window_units / 2, side="left"))
        idx_hi = int(np.searchsorted(events_t, t_center + window_units / 2, side="right"))
        xy_slice = np.asarray(events_xy[idx_lo:idx_hi])
        p_slice = np.asarray(events_p[idx_lo:idx_hi])
        ev_img = render_events(xy_slice, p_slice, h, w)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:h, :w] = frame
        canvas[:h, w + gap : w + gap + w] = ev_img

        t_seconds = (t_center - float(flir_t[0])) / unit_per_second
        action = "?"
        for b in boundaries:
            if b["flir_start_time"] <= t_seconds < b["flir_end_time"]:
                action = b["name"]
                break

        cv2.putText(canvas, "FLIR RGB", (10, h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Events (+/-{args.window_ms/2:.1f} ms, n={idx_hi - idx_lo})",
                    (w + gap + 10, h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        meta = f"frame {i}/{n_total - 1}  t={t_seconds:.2f}s  action={action}"
        (tw, _), _ = cv2.getTextSize(meta, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(canvas, meta, (out_w - tw - 10, h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(canvas)

    writer.release()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
