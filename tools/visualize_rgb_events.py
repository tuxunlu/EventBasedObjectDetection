"""Side-by-side visualisation of RGB frames, events, and GT hand masks.

Two dataset layouts are auto-detected from the sequence directory:

1. **NatureRoboticsDataNew** (a ``proc/`` dir): renders the FLIR RGB frame next to
   Prophesee events accumulated in a ±window/2 window around each frame timestamp.
   If cached ``proc/teacher_masks/`` (SAM 2) exist they are overlaid as the hand mask.

2. **N-HOT3D** (``events_*.h5`` + ``gt_hand_mask/``, possibly in a nested ``<TOKEN>/``
   leaf dir): renders, per frame ``i``, three panels — the Aria RGB frame, the event
   polarity image, and the events with the **ground-truth left/right hand masks**
   overlaid — so you can eyeball the per-event supervision the sparse model receives.
   Events (native 346×260) are rescaled into the 512×512 mask/RGB frame.

Writes an MP4 video.

Usage:
    # NatureRoboticsDataNew
    python tools/visualize_rgb_events.py \
        /fs/nexus-projects/DVS_Actions/NatureRoboticsDataNew/sequence_haowen1_..._bottle \
        --output viz_bottle.mp4 --window_ms 33 --fps 30

    # N-HOT3D (RGB auto-located from the parallel RGB/ tree; override with --rgb_root)
    python tools/visualize_rgb_events.py \
        /fs/nexus-projects/DVS_Actions/N-HOT3D/Aria/train/P0001_15c4300c \
        --output viz_P0001.mp4 --fps 30 --max_frames 300
"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# N-HOT3D file patterns + native event sensor resolution (see data/nhot3d_event_dataset.py)
_EVENT_RE = re.compile(r"^events_(\d{10})\.h5$")
_MASK_RE = re.compile(r"^gt_(\d{10})_(left|right)\.jpg$")
NHOT3D_EVENT_W, NHOT3D_EVENT_H = 346, 260
NHOT3D_OUT = 512

# Mask overlay colours (BGR): left = green, right = magenta.
_MASK_COLOR = {"left": (0, 200, 0), "right": (200, 0, 200)}


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


def overlay_masks(img: np.ndarray, masks: dict, alpha: float = 0.4) -> np.ndarray:
    """Translucent fill + contour for each named binary mask in ``masks`` (BGR colours
    from ``_MASK_COLOR``). ``masks`` maps side -> uint8 ``{0,1}`` array at ``img`` size."""
    out = img.copy()
    for side, m in masks.items():
        if m is None or not m.any():
            continue
        color = _MASK_COLOR.get(side, (0, 200, 200))
        fill = np.zeros_like(img)
        fill[m > 0] = color
        sel = m > 0
        out[sel] = (out[sel].astype(np.float32) * (1 - alpha)
                    + fill[sel].astype(np.float32) * alpha).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 1, cv2.LINE_AA)
    return out


# ===================================================================== N-HOT3D

def _resolve_nhot3d_dir(base: Path, token: str, leaf: str, pattern: re.Pattern):
    """First of ``base/<leaf>`` / ``base/<TOKEN>/<leaf>`` holding a file matching
    ``pattern`` (handles the mid-copy flattened-vs-nested split). ``leaf==''`` = base."""
    for cand in (base, base / token):
        d = cand / leaf if leaf else cand
        if not d.is_dir():
            continue
        try:
            for name in os.listdir(d):
                if pattern.match(name):
                    return d
        except OSError:
            continue
    return None


def _looks_like_nhot3d(seq: Path) -> bool:
    token = seq.name
    return _resolve_nhot3d_dir(seq, token, "", _EVENT_RE) is not None and \
        _resolve_nhot3d_dir(seq, token, "gt_hand_mask", _MASK_RE) is not None


def _default_rgb_root(seq: Path) -> Path:
    """``.../N-HOT3D/Aria/<split>/<TOKEN>`` -> ``.../N-HOT3D/RGB/<TOKEN>_rgb/<TOKEN>/rgb_images``."""
    token = seq.name
    # seq.parents[2] == the N-HOT3D root (… / Aria / <split> / <TOKEN>)
    nhot3d_root = seq.parents[2]
    return nhot3d_root / "RGB" / f"{token}_rgb" / token / "rgb_images"


def _scan_indices(d: Path, pattern: re.Pattern) -> set:
    out = set()
    if d is None:
        return out
    for name in os.listdir(d):
        m = pattern.match(name)
        if m:
            out.add(int(m.group(1)))
    return out


def _load_h5_events(path: Path) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        return np.asarray(f["events"][()], dtype=np.float64)


def _load_side_mask(mask_dir: Path, i: int, side: str, out: int) -> np.ndarray:
    p = mask_dir / f"gt_{i:010d}_{side}.jpg"
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape != (out, out):
        img = cv2.resize(img, (out, out), interpolation=cv2.INTER_NEAREST)
    return (img > 127).astype(np.uint8)


def visualize_nhot3d(args) -> None:
    seq = Path(args.sequence_dir)
    token = seq.name
    event_dir = _resolve_nhot3d_dir(seq, token, "", _EVENT_RE)
    mask_dir = _resolve_nhot3d_dir(seq, token, "gt_hand_mask", _MASK_RE)
    rgb_dir = Path(args.rgb_root) if args.rgb_root else _default_rgb_root(seq)
    out = NHOT3D_OUT

    frames = sorted(_scan_indices(event_dir, _EVENT_RE) & _scan_indices(mask_dir, _MASK_RE))
    if not frames:
        raise RuntimeError(f"No (event-chunk, mask) frames under {seq}")

    print(f"sequence    : {token}  (N-HOT3D)")
    print(f"event_dir   : {event_dir}")
    print(f"mask_dir    : {mask_dir}")
    print(f"rgb_dir     : {rgb_dir}  (exists={rgb_dir.is_dir()})")
    print(f"frames      : {len(frames)}  (idx {frames[0]}..{frames[-1]})")
    print(f"event scale : {args.event_width}x{args.event_height} -> {out}x{out}")

    n_start = max(0, args.start_frame)
    sel = frames[n_start:] if args.max_frames is None else frames[n_start:n_start + args.max_frames]

    gap, label_h, n_panels = 10, 58, 2  # 2-row label strip (panel titles + per-frame meta)
    out_w = out * n_panels + gap * (n_panels - 1)
    out_h = out + label_h
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {args.output}")

    for i in tqdm(sel, desc="rendering"):
        # RGB (may be absent — N-HOT3D RGB starts at frame ~50)
        rgb_path = rgb_dir / f"rgb_{i:010d}.jpg"
        rgb = load_frame(rgb_path) if rgb_path.exists() else np.zeros((out, out, 3), np.uint8)
        if rgb.shape[:2] != (out, out):
            rgb = cv2.resize(rgb, (out, out))

        # events -> polarity image. Render at NATIVE sensor resolution then upscale
        # with nearest-neighbour, so each native pixel becomes a contiguous block.
        # Plotting upscaled-then-rounded coords as single pixels instead leaves a
        # regular moiré grid (only 346/260 distinct cols/rows reach the 512 grid).
        ev = _load_h5_events(event_dir / f"events_{i:010d}.h5")
        xy = np.empty((0, 2)); pol = np.empty((0,))
        if ev.size:
            xy = np.stack([ev[:, 1], ev[:, 2]], axis=1)  # native (x,y), no rescale
            pol = np.where(ev[:, 3] > 0, 1.0, -1.0)
        ev_img = render_events(xy, pol, args.event_height, args.event_width)
        ev_img = cv2.resize(ev_img, (out, out), interpolation=cv2.INTER_NEAREST)

        masks = {s: _load_side_mask(mask_dir, i, s, out) for s in ("left", "right")}
        n_mask = int(sum(int(m.sum()) for m in masks.values() if m is not None))
        present = "+".join(s[0].upper() for s, m in masks.items() if m is not None and m.any()) or "none"

        p_rgb = overlay_masks(rgb, masks, alpha=args.mask_alpha)
        p_evt = ev_img

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for k, panel in enumerate((p_rgb, p_evt)):
            x0 = k * (out + gap)
            canvas[:out, x0:x0 + out] = panel
        labels = ["RGB + GT mask", f"Events (n={len(xy)})"]
        for k, lab in enumerate(labels):
            cv2.putText(canvas, lab, (k * (out + gap) + 8, out + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        meta = f"{token}  frame {i}/{frames[-1]}  hands={present}  mask_px={n_mask}"
        cv2.putText(canvas, meta, (8, out + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        writer.write(canvas)

    writer.release()
    print(f"wrote {args.output}")


# ========================================================== NatureRoboticsDataNew

def visualize_natrobotics(args) -> None:
    seq = Path(args.sequence_dir)
    proc = seq / "proc"
    flir_dir = proc / "flir"
    events_dir = proc / "events"
    mask_dir = proc / "teacher_masks"  # optional SAM 2 hand masks

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

    has_masks = mask_dir.is_dir()
    print(f"sequence       : {seq.name}  (NatureRoboticsDataNew)")
    print(f"frames         : {n_total} (FLIR)")
    print(f"events         : {len(events_t):,}")
    print(f"detected unit  : {unit_per_second:g} (=1 second)")
    print(f"event window   : ±{args.window_ms/2:.1f} ms ({window_units:g} units)")
    print(f"teacher masks  : {'yes (overlaid)' if has_masks else 'none'}")

    boundaries_path = proc / "boundaries.json"
    boundaries = json.loads(boundaries_path.read_text()) if boundaries_path.exists() else []

    sample = load_frame(frame_files[0])
    h, w, _ = sample.shape
    print(f"resolution     : {w}x{h}")

    n_start = max(0, args.start_frame)
    n_end = n_total if args.max_frames is None else min(n_total, n_start + args.max_frames)

    gap, label_h = 10, 40
    out_w = w * 2 + gap
    out_h = h + label_h
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (out_w, out_h))
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

        # Optional hand-mask overlay (cached SAM 2 teacher mask for this frame).
        mask = None
        if has_masks:
            mp = mask_dir / f"{i:06d}.png"
            if mp.exists():
                mimg = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if mimg is not None:
                    if mimg.shape != (h, w):
                        mimg = cv2.resize(mimg, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = (mimg > 127).astype(np.uint8)
        if mask is not None:
            frame = overlay_masks(frame, {"left": mask}, alpha=args.mask_alpha)
            ev_img = overlay_masks(ev_img, {"left": mask}, alpha=args.mask_alpha)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:h, :w] = frame
        canvas[:h, w + gap: w + gap + w] = ev_img

        t_seconds = (t_center - float(flir_t[0])) / unit_per_second
        action = "?"
        for b in boundaries:
            if b["flir_start_time"] <= t_seconds < b["flir_end_time"]:
                action = b["name"]
                break

        left_label = "FLIR RGB" + (" + hand mask" if mask is not None else "")
        cv2.putText(canvas, left_label, (10, h + 28),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequence_dir", type=str,
                        help="NatureRobotics sequence_<...> (with proc/) OR an N-HOT3D "
                             "Aria sequence dir (with events_*.h5 + gt_hand_mask/).")
    parser.add_argument("--output", type=str, default="rgb_events_viz.mp4")
    parser.add_argument("--window_ms", type=float, default=33.0,
                        help="[NatureRobotics] event window centred on each frame, in ms")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Playback FPS for output video")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit number of frames to render (for quick checks)")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--mask_alpha", type=float, default=0.4,
                        help="Hand-mask overlay opacity")
    parser.add_argument("--dataset", choices=("auto", "nhot3d", "natrobotics"),
                        default="auto", help="Force a layout instead of auto-detecting")
    # N-HOT3D-specific
    parser.add_argument("--rgb_root", type=str, default=None,
                        help="[N-HOT3D] override the auto-located rgb_images dir")
    parser.add_argument("--event_width", type=int, default=NHOT3D_EVENT_W,
                        help="[N-HOT3D] native event sensor width (rescaled to 512)")
    parser.add_argument("--event_height", type=int, default=NHOT3D_EVENT_H,
                        help="[N-HOT3D] native event sensor height (rescaled to 512)")
    args = parser.parse_args()

    seq = Path(args.sequence_dir)
    if not seq.is_dir():
        raise FileNotFoundError(f"sequence_dir does not exist: {seq}")

    kind = args.dataset
    if kind == "auto":
        if (seq / "proc").is_dir():
            kind = "natrobotics"
        elif _looks_like_nhot3d(seq):
            kind = "nhot3d"
        else:
            raise SystemExit(
                f"Could not detect dataset layout under {seq}. Expected a NatureRobotics "
                f"proc/ dir or N-HOT3D events_*.h5 + gt_hand_mask/. Force with --dataset.")

    if kind == "nhot3d":
        visualize_nhot3d(args)
    else:
        visualize_natrobotics(args)


if __name__ == "__main__":
    main()
