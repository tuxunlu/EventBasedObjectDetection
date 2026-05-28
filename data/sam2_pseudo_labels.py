"""
Generate per-frame hand+arm masks for one or more recordings using
GroundingDINO (to seed) + SAM 2 video predictor (to propagate).

Outputs are written to <mask_root>/<sequence_name>/{frame_index:06d}.png
(if --mask_root is given) or <sequence>/proc/teacher_masks/{frame_index:06d}.png
otherwise, as binary 0/255 uint8 masks, one per FLIR frame.

By default the FLIR RGB frames are fed to GroundingDINO + SAM 2 (the
in-distribution choice). Pass --input_modality events to instead render a
polarity image per FLIR timestamp (gray bg, red=positive, blue=negative; see
tools/visualize_rgb_events.py) and feed those event frames to both models.
Event-mode masks are written to a sibling '<...>_events/' directory so the
RGB and event runs don't clobber each other.

Usage
-----
Single sequence:
    python data/sam2_pseudo_labels.py \
        --sequence /fs/nexus-projects/DVS_Actions/NatureRoboticsDataNew/sequence_haowen1_SIDE_DYNAMIC_LIGHT_bottle \
        --sam2_checkpoint sam2_hiera_large.pt \
        --sam2_config sam2_hiera_l.yaml \
        --mask_root /fs/nexus-scratch/tuxunlu/teacher_masks \
        --hf_cache /fs/nexus-scratch/tuxunlu/hf_cache

Batch all sequences under a root:
    python data/sam2_pseudo_labels.py \
        --root /fs/nexus-projects/DVS_Actions/NatureRoboticsDataNew \
        --sam2_checkpoint ... --sam2_config ... \
        --mask_root ... --hf_cache ...

Dependencies (must be installed in the active env):
    - sam2 (Meta repo: https://github.com/facebookresearch/sam2)
    - transformers + torch (GroundingDINO via IDEA-Research/grounding-dino-tiny)
    - opencv-python, pillow, numpy, tqdm
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_frame_rgb(path: Path) -> np.ndarray:
    """Read FLIR PNG → uint8 RGB ndarray (H, W, 3)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read {path}")
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


def materialise_jpeg_dir(frame_files: Sequence[Path], out_dir: Path) -> None:
    """SAM 2's video predictor expects sequentially named JPEGs; materialise."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(frame_files):
        rgb = load_frame_rgb(src)
        Image.fromarray(rgb).save(out_dir / f"{i:06d}.jpg", quality=95)


class GroundingDinoDetector:
    """Load GroundingDINO once, reuse across many frames/sequences."""

    def __init__(
        self,
        device: torch.device,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.device = device
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            .to(device).eval()
        )

    @torch.no_grad()
    def __call__(
        self,
        frame_rgb: np.ndarray,
        prompt: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> List[Tuple[List[float], str, float]]:
        image = Image.fromarray(frame_rgb)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=box_threshold, text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=box_threshold, text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]
        out = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            out.append((box.tolist(), str(label), float(score)))
        return out


def build_sam2_predictor(checkpoint: Path, config: str, device: torch.device):
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(config, str(checkpoint), device=device)


def _candidate_seed_indices(
    n_frames: int,
    n_probes: int,
    action_range: Optional[Tuple[int, int]] = None,
) -> List[int]:
    """Strided sample of probe indices. Prefers the action segment if known."""
    if action_range is not None:
        lo, hi = action_range
        lo = max(0, lo)
        hi = min(n_frames, hi)
        if hi <= lo:
            lo, hi = 0, n_frames
    else:
        lo, hi = 0, n_frames
    if n_probes >= (hi - lo):
        candidates = list(range(lo, hi))
    else:
        candidates = [int(round(lo + i * (hi - lo - 1) / (n_probes - 1))) for i in range(n_probes)]
    return sorted(set(candidates))


def _action_frame_range(
    boundaries_path: Path, flir_t: np.ndarray
) -> Optional[Tuple[int, int]]:
    """Map boundaries.json (in seconds) to FLIR frame indices for the action segment."""
    if not boundaries_path.exists():
        return None
    boundaries = json.loads(boundaries_path.read_text())
    action_intervals = [
        (b["flir_start_time"], b["flir_end_time"])
        for b in boundaries if b.get("name") != "background"
    ]
    if not action_intervals:
        return None
    rng = float(flir_t[-1] - flir_t[0])
    unit = 1e9 if rng > 1e9 else 1e6 if rng > 1e6 else 1e3 if rng > 1e3 else 1.0
    t0 = float(flir_t[0])
    t_rel = (flir_t.astype(np.float64) - t0) / unit
    lo_i = hi_i = None
    for s, e in action_intervals:
        in_seg = (t_rel >= s) & (t_rel < e)
        if not in_seg.any():
            continue
        idx = np.nonzero(in_seg)[0]
        seg_lo, seg_hi = int(idx[0]), int(idx[-1]) + 1
        lo_i = seg_lo if lo_i is None else min(lo_i, seg_lo)
        hi_i = seg_hi if hi_i is None else max(hi_i, seg_hi)
    if lo_i is None or hi_i is None:
        return None
    return (lo_i, hi_i)


def _box_area(xyxy: Sequence[float]) -> float:
    x0, y0, x1, y1 = xyxy
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _box_center(xyxy: Sequence[float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = xyxy
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def keep_foreground_boxes(
    boxes: Sequence[Tuple[List[float], str, float]],
    img_h: int,
    img_w: int,
    min_area_frac: float = 0.01,
    associate_arm_to_hand: bool = True,
) -> List[Tuple[List[float], str, float]]:
    """Suppress background instances so SAM 2 only seeds on the foreground hand+arm.

    Rules:
      1. Drop any box smaller than ``min_area_frac`` of the image (background people
         hands are tiny in pixel terms).
      2. Keep at most ONE box per phrase, picking the largest (foreground == closer
         to camera == bigger).
      3. If both "hand" and "arm" survive, optionally keep only the arm closest to
         the kept hand center — guards against the rare case where the largest hand
         and the largest arm belong to different people.
    """
    img_area = float(img_h * img_w)
    floor = min_area_frac * img_area
    by_phrase: Dict[str, Tuple[List[float], str, float]] = {}
    for xyxy, label, score in boxes:
        if _box_area(xyxy) < floor:
            continue
        key = label.strip().lower()
        prev = by_phrase.get(key)
        if prev is None or _box_area(xyxy) > _box_area(prev[0]):
            by_phrase[key] = (xyxy, label, score)

    if not associate_arm_to_hand or len(by_phrase) <= 1:
        return list(by_phrase.values())

    # If we have both a hand and an arm survivor, sanity-check they belong together
    # by checking center proximity relative to image diagonal.
    hand_key = next((k for k in by_phrase if "hand" in k), None)
    arm_key = next((k for k in by_phrase if "arm" in k and k != hand_key), None)
    if hand_key is None or arm_key is None:
        return list(by_phrase.values())
    hx, hy = _box_center(by_phrase[hand_key][0])
    ax, ay = _box_center(by_phrase[arm_key][0])
    diag = float(np.hypot(img_h, img_w))
    if np.hypot(ax - hx, ay - hy) > 0.5 * diag:
        # Arm is far from hand — likely a different person. Drop the arm.
        by_phrase.pop(arm_key)
    return list(by_phrase.values())


def find_best_seed(
    detector: "GroundingDinoDetector",
    frame_files: Sequence[Path],
    prompt: str,
    candidates: Sequence[int],
    min_score: float = 0.30,
    min_area_frac: float = 0.01,
) -> Tuple[Optional[int], List[Tuple[List[float], str, float]]]:
    """Scan candidate frames; return (seed_idx, foreground_boxes), or (None, [])."""
    best_idx: Optional[int] = None
    best_boxes: List[Tuple[List[float], str, float]] = []
    best_rank = 0.0
    for idx in candidates:
        rgb = load_frame_rgb(frame_files[idx])
        raw_boxes = detector(rgb, prompt=prompt, box_threshold=min_score)
        if not raw_boxes:
            continue
        h, w = rgb.shape[:2]
        fg = keep_foreground_boxes(raw_boxes, h, w, min_area_frac=min_area_frac)
        if not fg:
            continue
        # Rank frames by sum of (score * sqrt(area_frac)) — favours frames where the
        # foreground hand/arm is BOTH confident AND large.
        img_area = float(h * w)
        rank = sum(s * float(np.sqrt(_box_area(b) / img_area)) for b, _l, s in fg)
        if rank > best_rank:
            best_idx = idx
            best_boxes = fg
            best_rank = rank
    return best_idx, best_boxes


def run_sam2_video(
    predictor,
    frames_dir: Path,
    boxes: Sequence[Tuple[List[float], str, float]],
    seed_frame_idx: int,
    num_frames: int,
) -> Dict[int, np.ndarray]:
    """Seed at `seed_frame_idx`, propagate forward AND backward, return per-frame masks."""
    state = predictor.init_state(video_path=str(frames_dir))
    predictor.reset_state(state)

    for obj_id, (xyxy, _label, _score) in enumerate(boxes, start=1):
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=seed_frame_idx,
            obj_id=obj_id,
            box=np.array(xyxy, dtype=np.float32),
        )

    masks: Dict[int, np.ndarray] = {}

    def _consume(reverse: bool):
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(
            state, reverse=reverse
        ):
            m = (mask_logits > 0.0).any(dim=0).squeeze(0).cpu().numpy().astype(bool)
            masks[int(frame_idx)] = m

    _consume(reverse=False)
    if seed_frame_idx > 0:
        _consume(reverse=True)

    return masks


def _render_event_panel(xy: np.ndarray, p: np.ndarray, h: int, w: int) -> np.ndarray:
    """Gray background with red (positive) and blue (negative) event pixels."""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    if len(xy) == 0:
        return img
    x = np.clip(xy[:, 0].astype(np.int32), 0, w - 1)
    y = np.clip(xy[:, 1].astype(np.int32), 0, h - 1)
    pos = p > 0
    img[y[~pos], x[~pos]] = (255, 0, 0)   # BGR blue = negative polarity
    img[y[pos], x[pos]] = (0, 0, 255)     # BGR red  = positive polarity
    return img


def _detect_unit_per_second(t_range: float) -> float:
    if t_range > 1e9:
        return 1e9
    if t_range > 1e6:
        return 1e6
    if t_range > 1e3:
        return 1e3
    return 1.0


def _build_event_input_dir(
    proc_dir: Path,
    flir_t: np.ndarray,
    h: int,
    w: int,
    window_ms: float,
    out_dir: Path,
) -> List[Path]:
    """Render one event polarity image per FLIR timestamp and save as JPEG.

    Output JPEGs are named ``{idx:05d}.jpg`` so the SAM 2 video predictor
    indexes them in the same order as the FLIR frames. Colormap matches
    tools/visualize_rgb_events.py (gray bg, red=positive, blue=negative).
    """
    events_t = np.load(proc_dir / "events" / "events_t.npy", mmap_mode="r")
    events_xy = np.load(proc_dir / "events" / "events_xy.npy", mmap_mode="r")
    events_p = np.load(proc_dir / "events" / "events_p.npy", mmap_mode="r")
    unit = _detect_unit_per_second(float(events_t[-1] - events_t[0]))
    window_units = (window_ms / 1000.0) * unit

    paths: List[Path] = []
    for i in range(len(flir_t)):
        t = float(flir_t[i])
        lo = int(np.searchsorted(events_t, t - window_units / 2, side="left"))
        hi = int(np.searchsorted(events_t, t + window_units / 2, side="right"))
        img = _render_event_panel(
            np.asarray(events_xy[lo:hi]),
            np.asarray(events_p[lo:hi]),
            h, w,
        )
        out_path = out_dir / f"{i:05d}.jpg"
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(out_path)
    return paths


def write_preview_video(
    frame_files: Sequence[Path],
    mask_dir: Path,
    proc_dir: Path,
    out_path: Path,
    fps: float = 30.0,
    window_ms: float = 33.0,
    seed_frame_idx: Optional[int] = None,
    seed_panel: str = "rgb",
) -> None:
    """Three-panel MP4: FLIR RGB | Events (±window/2) | Teacher Mask, with labels.

    ``seed_panel`` controls which panel the "SEED" badge is drawn on — set to
    ``"events"`` when the teacher pipeline was fed event frames so the badge
    lands on the panel GroundingDINO actually saw.
    """
    h, w = load_frame_rgb(frame_files[0]).shape[:2]

    flir_t = np.load(proc_dir / "flir" / "flir_t.npy")
    events_t = np.load(proc_dir / "events" / "events_t.npy", mmap_mode="r")
    events_xy = np.load(proc_dir / "events" / "events_xy.npy", mmap_mode="r")
    events_p = np.load(proc_dir / "events" / "events_p.npy", mmap_mode="r")

    unit = _detect_unit_per_second(float(events_t[-1] - events_t[0]))
    window_units = (window_ms / 1000.0) * unit

    gap = 10
    label_h = 40
    out_w = w * 3 + 2 * gap
    out_h = h + label_h
    n = min(len(frame_files), len(flir_t))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {out_path}")
    try:
        for i in range(n):
            # --- Panel 1: FLIR RGB ---
            rgb = load_frame_rgb(frame_files[i])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # --- Panel 2: Events in ±window/2 around this FLIR timestamp ---
            t_center = float(flir_t[i])
            lo = int(np.searchsorted(events_t, t_center - window_units / 2, side="left"))
            hi = int(np.searchsorted(events_t, t_center + window_units / 2, side="right"))
            ev_img = _render_event_panel(
                np.asarray(events_xy[lo:hi]),
                np.asarray(events_p[lo:hi]),
                h, w,
            )

            # --- Panel 3: Teacher mask ---
            mask_path = mask_dir / f"{i:06d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
            if mask is None:
                mask = np.zeros((h, w), dtype=np.uint8)
            elif mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # --- Compose ---
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            canvas[:h, :w] = bgr
            canvas[:h, w + gap : 2 * w + gap] = ev_img
            canvas[:h, 2 * w + 2 * gap : 3 * w + 2 * gap] = mask_bgr

            # Seed-frame badge on whichever panel the model actually saw.
            if seed_frame_idx is not None and i == seed_frame_idx:
                seed_x = (w + gap + 8) if seed_panel == "events" else 8
                cv2.putText(canvas, "SEED", (seed_x, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Labels (panel titles + global frame meta on the right).
            cv2.putText(canvas, "FLIR RGB", (10, h + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Events (+/-{window_ms/2:.1f} ms, n={hi - lo})",
                        (w + gap + 10, h + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Teacher Mask", (2 * w + 2 * gap + 10, h + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            meta = f"frame {i}/{n - 1}"
            (tw, _), _ = cv2.getTextSize(meta, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(canvas, meta, (out_w - tw - 10, h + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            writer.write(canvas)
    finally:
        writer.release()


def _infer_fps(flir_t: np.ndarray, default: float = 30.0) -> float:
    if flir_t.size < 2:
        return default
    rng = float(flir_t[-1] - flir_t[0])
    if rng <= 0:
        return default
    unit = 1e9 if rng > 1e9 else 1e6 if rng > 1e6 else 1e3 if rng > 1e3 else 1.0
    duration = rng / unit
    if duration <= 0:
        return default
    return float(flir_t.size - 1) / duration


def process_sequence(
    sequence_dir: Path,
    detector: GroundingDinoDetector,
    sam2_predictor,
    prompt: str,
    overwrite: bool,
    mask_root: Path | None = None,
    n_probes: int = 8,
    min_score: float = 0.30,
    min_area_frac: float = 0.01,
    write_preview: bool = False,
    preview_fps: Optional[float] = None,
    preview_window_ms: float = 33.0,
    input_modality: str = "rgb",
    input_window_ms: float = 33.0,
) -> None:
    if input_modality not in ("rgb", "events"):
        raise ValueError(f"input_modality must be 'rgb' or 'events', got {input_modality!r}")
    proc = sequence_dir / "proc"
    if not proc.is_dir():
        raise FileNotFoundError(f"No proc/ under {sequence_dir}")
    flir_dir = proc / "flir" / "frame"
    frame_files = sorted(flir_dir.glob("*.png"))
    if not frame_files:
        raise RuntimeError(f"No PNG frames under {flir_dir}")

    # Event-mode masks go to a sibling directory so the two pipelines don't clobber.
    mask_subdir_suffix = "_events" if input_modality == "events" else ""
    if mask_root is not None:
        mask_dir = mask_root / (sequence_dir.name + mask_subdir_suffix)
    else:
        mask_dir = proc / ("teacher_masks" + mask_subdir_suffix)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(mask_dir.iterdir()):
        print(f"[skip] {sequence_dir.name} (masks already exist at {mask_dir})")
        return

    flir_t = np.load(proc / "flir" / "flir_t.npy")
    action_range = _action_frame_range(proc / "boundaries.json", flir_t)
    h, w = load_frame_rgb(frame_files[0]).shape[:2]

    with tempfile.TemporaryDirectory(prefix="sam2_input_") as tmp:
        tmp_dir = Path(tmp)

        if input_modality == "events":
            print(f"[prep] {sequence_dir.name}: rendering {len(flir_t)} event input frames "
                  f"(window +/-{input_window_ms/2:.1f} ms)")
            input_files = _build_event_input_dir(
                proc, flir_t, h, w, input_window_ms, tmp_dir,
            )
        else:
            print(f"[prep] {sequence_dir.name}: materialising {len(frame_files)} JPEGs")
            materialise_jpeg_dir(frame_files, tmp_dir)
            input_files = sorted(tmp_dir.glob("*.jpg"))
        n_input = len(input_files)

        candidates = _candidate_seed_indices(n_input, n_probes, action_range)
        print(f"[seed] {sequence_dir.name}: scanning {len(candidates)} candidate frame(s) "
              f"(modality={input_modality}, action_range={action_range})")
        seed_idx, boxes = find_best_seed(
            detector, input_files, prompt,
            candidates=candidates, min_score=min_score, min_area_frac=min_area_frac,
        )
        if seed_idx is None:
            # Fall back to a wider scan over the whole clip with a relaxed threshold.
            print(f"[seed] {sequence_dir.name}: action-segment scan empty, retrying "
                  f"full-clip with min_score=0.20")
            fallback = _candidate_seed_indices(n_input, 2 * n_probes, None)
            seed_idx, boxes = find_best_seed(
                detector, input_files, prompt,
                candidates=fallback, min_score=0.20, min_area_frac=min_area_frac,
            )
        if seed_idx is None or not boxes:
            print(f"[warn] {sequence_dir.name}: GroundingDINO found nothing for {prompt!r} "
                  f"across {len(candidates)} probes ({input_modality} input); skipping.")
            return
        print(f"[seed] {sequence_dir.name}: chose frame {seed_idx}")
        for xyxy, label, score in boxes:
            print(f"        box {label!r} {score:.2f} {xyxy}")

        print(f"[sam2] {sequence_dir.name}: propagating masks "
              f"(seed={seed_idx}, {n_input} frames, bidirectional, input={input_modality})")
        masks = run_sam2_video(sam2_predictor, tmp_dir, boxes, seed_idx, n_input)
        zeros = np.zeros((h, w), dtype=bool)
        for i in tqdm(range(n_input), desc=sequence_dir.name):
            mask = masks.get(i, zeros)
            cv2.imwrite(str(mask_dir / f"{i:06d}.png"), (mask.astype(np.uint8) * 255))

    if write_preview:
        fps = preview_fps if preview_fps is not None else _infer_fps(flir_t)
        preview_path = mask_dir.parent / f"{mask_dir.name}_preview.mp4"
        print(f"[viz ] {sequence_dir.name}: writing {preview_path.name} "
              f"at {fps:.2f} fps (event window +/-{preview_window_ms/2:.1f} ms)")
        try:
            write_preview_video(
                frame_files, mask_dir, proc, preview_path,
                fps=fps, window_ms=preview_window_ms, seed_frame_idx=seed_idx,
                seed_panel="events" if input_modality == "events" else "rgb",
            )
        except Exception as e:
            print(f"[warn] {sequence_dir.name}: preview video failed: {e}")


def _set_hf_cache(path: Path) -> None:
    """Redirect HuggingFace caches to `path`. Must run BEFORE transformers import."""
    path = path.expanduser()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(path)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(path / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(path / "hub")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--sequence", type=str, help="Path to single sequence_<...> root")
    g.add_argument("--root", type=str,
                   help="Root directory containing many sequence_<...> folders")
    parser.add_argument("--sam2_checkpoint", type=str, required=True,
                        help="Path to SAM 2 .pt checkpoint")
    parser.add_argument("--sam2_config", type=str, required=True,
                        help="SAM 2 config name (e.g. sam2_hiera_l.yaml)")
    parser.add_argument("--prompt", type=str, default="hand. arm.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run on sequences whose mask dir already contains files")
    parser.add_argument("--shard", type=str, default=None,
                        help="Process only shard 'i/N' (0-indexed) of the sorted "
                             "sequence list. Use for multi-GPU runs: launch N "
                             "processes with --shard 0/N, 1/N, ..., (N-1)/N, "
                             "each pinned to a distinct GPU via "
                             "CUDA_VISIBLE_DEVICES. Avoids the race where "
                             "concurrent workers both pick up the same sequence.")
    parser.add_argument("--glob", type=str, default="sequence_*",
                        help="Glob filter when using --root")
    parser.add_argument("--mask_root", type=str, default=None,
                        help="Write masks under <mask_root>/<sequence_name>/{i:06d}.png "
                             "instead of <sequence>/proc/teacher_masks/. Use when the "
                             "dataset filesystem is read-only.")
    parser.add_argument("--hf_cache", type=str, default=None,
                        help="Override HuggingFace cache location (set HF_HOME). "
                             "Required when the default ~/.cache is on a small NFS volume.")
    parser.add_argument("--gd_model_id", type=str,
                        default="IDEA-Research/grounding-dino-tiny",
                        help="GroundingDINO HF model id.")
    parser.add_argument("--seed_probes", type=int, default=8,
                        help="Number of frames to scan looking for a hand/arm seed. "
                             "Probes prefer the action segment from boundaries.json.")
    parser.add_argument("--seed_min_score", type=float, default=0.30,
                        help="GroundingDINO box score threshold for the seed scan.")
    parser.add_argument("--seed_min_area_frac", type=float, default=0.01,
                        help="Minimum box area as a fraction of the image (filters background "
                             "people; foreground hand/arm is much larger in pixel terms). "
                             "Per phrase ('hand', 'arm'), only the LARGEST surviving box is "
                             "used to seed SAM 2.")
    parser.add_argument("--no_preview", action="store_true",
                        help="Skip the side-by-side preview MP4.")
    parser.add_argument("--preview_fps", type=float, default=None,
                        help="Override preview MP4 fps (default: inferred from flir_t).")
    parser.add_argument("--preview_window_ms", type=float, default=33.0,
                        help="Event accumulation window (ms) centred on each FLIR "
                             "timestamp for the preview's events panel.")
    parser.add_argument("--input_modality", type=str, default="rgb",
                        choices=["rgb", "events"],
                        help="What to feed GroundingDINO + SAM 2. 'rgb' uses FLIR "
                             "PNGs (default, in-distribution for both models); "
                             "'events' renders a polarity image per FLIR timestamp "
                             "and feeds those instead. Event-mode masks are written "
                             "to a sibling '<...>_events/' directory.")
    parser.add_argument("--input_window_ms", type=float, default=33.0,
                        help="Event accumulation window (ms) for the model-input "
                             "frames when --input_modality events.")
    args = parser.parse_args(argv)

    if args.hf_cache:
        _set_hf_cache(Path(args.hf_cache))
        print(f"[env] HF_HOME = {os.environ['HF_HOME']}")

    mask_root = Path(args.mask_root).expanduser() if args.mask_root else None
    if mask_root is not None:
        mask_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    sam2_checkpoint = Path(args.sam2_checkpoint).expanduser()
    if not sam2_checkpoint.exists():
        print(f"[fatal] SAM 2 checkpoint not found: {sam2_checkpoint}", file=sys.stderr)
        return 2

    if args.sequence:
        sequences = [Path(args.sequence)]
    else:
        root = Path(args.root)
        sequences = sorted(p for p in root.glob(args.glob) if p.is_dir())
    if not sequences:
        print("[fatal] no sequences matched", file=sys.stderr)
        return 1

    shard_tag = ""
    if args.shard is not None:
        try:
            shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            parser.error(f"--shard expects 'i/N' (integers), got {args.shard!r}")
        if not (shard_n > 0 and 0 <= shard_i < shard_n):
            parser.error(f"--shard {args.shard!r}: require 0 <= i < N and N > 0")
        sequences = [s for k, s in enumerate(sequences) if k % shard_n == shard_i]
        shard_tag = f" [shard {shard_i}/{shard_n}]"
        if not sequences:
            print(f"[fatal] shard {args.shard} is empty", file=sys.stderr)
            return 1

    print(f"[load] GroundingDINO ({args.gd_model_id}) on {device}")
    detector = GroundingDinoDetector(device=device, model_id=args.gd_model_id)
    print(f"[load] SAM 2 ({sam2_checkpoint.name}, cfg={args.sam2_config}) on {device}")
    sam2_predictor = build_sam2_predictor(sam2_checkpoint, args.sam2_config, device)

    print(f"[run]{shard_tag} {len(sequences)} sequence(s)")
    failed = 0
    for seq in sequences:
        try:
            process_sequence(
                sequence_dir=seq,
                detector=detector,
                sam2_predictor=sam2_predictor,
                prompt=args.prompt,
                overwrite=args.overwrite,
                mask_root=mask_root,
                n_probes=args.seed_probes,
                min_score=args.seed_min_score,
                min_area_frac=args.seed_min_area_frac,
                write_preview=not args.no_preview,
                preview_fps=args.preview_fps,
                preview_window_ms=args.preview_window_ms,
                input_modality=args.input_modality,
                input_window_ms=args.input_window_ms,
            )
        except OSError as e:
            if e.errno == errno.ENOSPC:
                print(f"[fatal] disk full on {seq.name}: {e}. Stopping batch.",
                      file=sys.stderr)
                return 28
            print(f"[error] {seq.name}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
        except Exception as e:
            print(f"[error] {seq.name}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
    if failed:
        print(f"[done] {failed} sequence(s) failed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
