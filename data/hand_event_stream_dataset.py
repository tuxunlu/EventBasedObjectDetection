"""Per-event (sparse, 3D) variant of :class:`HandEventDataset`.

Instead of a dense voxel grid, each sample is the set of **individual events** in
the window centred on a FLIR frame — every event keeps its own ``(x, y)``, time and
polarity, with a foreground/background label per event sampled from the cached
SAM 2 teacher mask. This is the input/supervision for the event-native 3D
``EventSparseSeg`` model, which voxelizes ``(t, y, x)`` itself and emits one logit
per event.

Why per-event (not per-pixel-merged) any more
---------------------------------------------
The old version merged all events at a pixel into one 2D site, discarding the
temporal axis before a 2D U-Net. At this sensor's ~1.2 events/pixel that bought
almost no dedup while throwing away motion — the very signal that separates a
moving hand from background. We now keep every event so the model can reason in 3D.

The label
---------
``label = mask[y, x]`` is sampled from the cached SAM 2 mask at each event's pixel.
These per-event labels are the direct supervision target for ``EventSparseSeg``
(one logit per event), via ``loss/event_distillation.EventDistillationLoss``.

Reuses the parent's subject-disjoint LOSO split, frame index, timestamp-unit
detection, handle caching, teacher-mask lookup, and event-coordinate rescaling
unchanged; only ``__getitem__`` differs. Geometric augmentation is disabled on this
path (the parent's augmenter warps dense voxels/masks, not raw coordinates).

Per-sample output (consumed by ``collate_sparse_events``):
    ``(coords[N, 2] long (x, y), feats[N, C] float, times[N] float,
       labels[N] float, dense_mask[H, W] uint8, meta: dict)``
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch

from data.event_augment import EventAugmentor
from data.hand_event_dataset import HandEventDataset
from data.sparse_event_collate import collate_sparse_events


class HandEventStreamDataset(HandEventDataset):

    #: per-event features: signed polarity, normalized time-in-window
    NUM_FEATURES = 2

    def __init__(
        self,
        root_dir: str,
        purpose: str = "train",
        window_ms: float = 36.0,
        image_height: int = 480,
        image_width: int = 640,
        held_out_subject=None,
        require_teacher: bool = True,
        action_only: bool = False,
        mask_root=None,
        max_events: int = 150000,
        augmentation=None,
        boundary_ignore_px: int = 0,
        time_ignore_frac: float = 0.0,
        time_ignore_val: bool = False,
        label_mode: str = "center",
        max_label_frames: int = 5,
        split_mode: str = "loso",
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        split_seed: int = 0,
    ):
        # voxel_bins is irrelevant on the event path; pass a harmless 1. The parent's
        # dense voxel/mask augmenter is left OFF (augmentation=None below) — this path
        # uses the event-native EventAugmentor instead (operates on raw events).
        super().__init__(
            root_dir=root_dir,
            purpose=purpose,
            voxel_bins=1,
            window_ms=window_ms,
            image_height=image_height,
            image_width=image_width,
            held_out_subject=held_out_subject,
            require_teacher=require_teacher,
            action_only=action_only,
            mask_root=mask_root,
            augmentation=None,
            provide_rgb=False,
            split_mode=split_mode,
            val_frac=val_frac,
            test_frac=test_frac,
            split_seed=split_seed,
        )
        self.max_events = int(max_events)
        # Event-native augmentation (train only). dict() defends against an OmegaConf
        # node arriving from the YAML path (same guard as the parent augmenter).
        self._augmentor = EventAugmentor(dict(augmentation) if augmentation else None)
        self._event_aug_active = (purpose == "train" and self._augmentor.enabled)

        # --- Tier-2 supervision cleanup (TRAIN ONLY) ------------------------------
        # The per-event label is sampled from a SINGLE FLIR-frame SAM 2 mask, but the
        # event window spans ``window_ms`` during which the hand moves. Two sources of
        # structured label noise follow, both of which teach a dilated/fuzzy boundary:
        #   * boundary_ignore_px: the mask edge is the least trustworthy region (a
        #     few px of motion smear + teacher slop). Events in a morphological band
        #     of this half-width around the mask edge are marked label = -1 ("ignore"),
        #     which EventDistillationLoss drops (ignore_negative).
        #   * time_ignore_frac: events far from the window centre (where the mask was
        #     captured) are the most spatially misaligned with it. Events in the first
        #     / last ``time_ignore_frac`` of the normalized window are marked -1.
        # Applied to the supervision target only; val/test normally keep clean {0,1}
        # labels so the logged F1/IoU stay honest, and ``dense_mask`` is always clean.
        # EXCEPTION (``time_ignore_val=True``): with a LONG window (e.g. 144ms) the
        # single center-frame mask mislabels off-center events, so scoring them is NOT
        # honest — applying the center-band ignore at val/test too makes the metric
        # honest (the -1 events are dropped by event_f1/precision/recall, which filter
        # ``labels >= 0``). Keep ``boundary_ignore_px=0`` at val to leave boundary F1
        # untouched. SSL/residual still see all events (labels are a loss/metric concern).
        self.boundary_ignore_px = int(boundary_ignore_px)
        self.time_ignore_frac = float(time_ignore_frac)
        self.time_ignore_val = bool(time_ignore_val)
        self._label_refine_active = (
            (purpose == "train" or self.time_ignore_val)
            and (self.boundary_ignore_px > 0 or self.time_ignore_frac > 0.0)
        )

        # --- Motion-compensated (nearest-frame) labels --------------------------------
        # ``center`` (default): every event's label = the SINGLE center-frame SAM 2 mask
        # ``mask[y,x]`` — over a ``window_ms`` window the moving hand edge sweeps across
        # pixels, so events at the window extremes are mislabeled (the root cause of the
        # fuzzy edge under fast motion). ``nearest_frame``: each event is labeled by the
        # teacher mask of the FLIR frame NEAREST its OWN timestamp (a discrete
        # motion-compensation), de-smearing the boundary supervision. Applied at all
        # stages so train and val targets stay consistent; falls back to the center frame
        # for events whose nearest neighbor frame has no cached mask. ``max_label_frames``
        # bounds the per-window mask loads (closest-to-center kept).
        self.label_mode = str(label_mode).strip().lower()
        if self.label_mode not in ("center", "nearest_frame"):
            raise ValueError(
                f"label_mode must be center|nearest_frame, got {label_mode!r}")
        self._max_label_frames = max(1, int(max_label_frames))

    # ------------------------------------------------------------------ helpers

    def _load_mask(self, handle: dict, f_idx: int) -> torch.Tensor:
        """Load the teacher mask as an ``(H, W)`` float tensor in ``{0, 1}``.

        Mirrors the parent's inline mask loading (kept here so this class needs no
        edit to ``HandEventDataset``).
        """
        H, W = self.image_height, self.image_width
        mask_path = handle["mask_dir"] / f"{f_idx:06d}.png"
        if self.require_teacher and not mask_path.exists():
            raise FileNotFoundError(f"Missing teacher mask {mask_path}")
        if mask_path.exists():
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise IOError(f"Could not read mask {mask_path}")
            if mask_img.shape != (H, W):
                mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
            return torch.from_numpy((mask_img > 127).astype(np.float32))
        return torch.zeros((H, W), dtype=torch.float32)

    def _build_label_map(self, mask: torch.Tensor) -> torch.Tensor:
        """Mask ``(H,W)`` in ``{0,1}`` -> a per-pixel target map in ``{0, 1, -1}``.

        With ``boundary_ignore_px > 0`` the morphological boundary band
        ``dilate(mask) \\ erode(mask)`` is set to ``-1`` (ignored in the loss). Without
        it, the map is just the binary mask. Train-only (gated by the caller).
        """
        if self.boundary_ignore_px <= 0:
            return mask
        m = (mask.numpy() > 0.5).astype(np.uint8)
        k = self.boundary_ignore_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        dil = cv2.dilate(m, kernel)
        ero = cv2.erode(m, kernel)
        band = (dil > 0) & (ero == 0)
        label_map = m.astype(np.float32)
        label_map[band] = -1.0
        return torch.from_numpy(label_map)

    def _nearest_frame_masks(self, h, f_idx, t_center, t_start, t_end, center_mask):
        """For ``label_mode='nearest_frame'``: stack the (trimap) teacher masks of the
        FLIR frames within the window so each event can be labeled by its nearest-in-time
        frame. Returns ``(frame_masks (K,H,W) float, frame_times (K,) float64)`` or
        ``(None, None)`` when nearest-frame labeling is off or only the center frame has a
        cached mask (then the caller uses the plain single-frame path)."""
        if self.label_mode != "nearest_frame":
            return None, None
        flir_t = np.asarray(h["flir_t"])
        j_lo = int(np.searchsorted(flir_t, t_start, side="left"))
        j_hi = int(np.searchsorted(flir_t, t_end, side="right"))
        cand = sorted((set(range(j_lo, j_hi)) | {f_idx}))
        cand = [c for c in cand if 0 <= c < len(flir_t)]
        if len(cand) > self._max_label_frames:                       # keep closest to centre
            cand = sorted(sorted(cand, key=lambda c: abs(float(flir_t[c]) - t_center))
                          [:self._max_label_frames])
        maps, times = [], []
        for j in cand:
            if j == f_idx:
                mj = center_mask
            else:
                if not (h["mask_dir"] / f"{j:06d}.png").exists():
                    continue
                mj = self._load_mask(h, j)
            maps.append(self._build_label_map(mj) if self._label_refine_active else mj)
            times.append(float(flir_t[j]))
        if len(maps) <= 1:                                           # only centre available
            return None, None
        return torch.stack(maps, dim=0), np.asarray(times, dtype=np.float64)

    def _empty_sample(self):
        return (
            torch.zeros((0, 2), dtype=torch.long),
            torch.zeros((0, self.NUM_FEATURES), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
        )

    def _build_events(
        self,
        t: np.ndarray,
        xy: np.ndarray,
        p: np.ndarray,
        t_start: float,
        t_end: float,
        mask: torch.Tensor,
        frame_masks: torch.Tensor = None,
        frame_times: np.ndarray = None,
    ):
        """Vectorized: raw events -> per-event (coords, feats, times, labels). No merge.

        Event coordinates are rescaled to the configured ``(image_height,
        image_width)`` the same way the parent voxel path rescales them, so events
        and the (resized) teacher mask share one pixel grid.
        """
        H, W = self.image_height, self.image_width
        if len(t) == 0:
            return self._empty_sample()

        xy_np = np.asarray(xy)
        if self._needs_rescale and xy_np.size:
            xy_np = xy_np.astype(np.float32)
            xy_np[:, 0] = xy_np[:, 0] * self._scale_x
            xy_np[:, 1] = xy_np[:, 1] * self._scale_y
        x = torch.as_tensor(xy_np[:, 0]).floor().to(torch.long)
        y = torch.as_tensor(xy_np[:, 1]).floor().to(torch.long)
        t_t = torch.as_tensor(np.asarray(t), dtype=torch.float64)
        p_t = torch.as_tensor(np.asarray(p), dtype=torch.float32)
        pol = torch.where(p_t > 0, torch.ones_like(p_t), -torch.ones_like(p_t))

        in_b = ((x >= 0) & (x < W) & (y >= 0) & (y < H)
                & (t_t >= t_start) & (t_t < t_end))
        x, y, pol, t_t = x[in_b], y[in_b], pol[in_b], t_t[in_b]
        if x.numel() == 0:
            return self._empty_sample()

        # Bound compute/memory on bursty windows with a deterministic strided subsample.
        m = x.numel()
        if self.max_events > 0 and m > self.max_events:
            stride = (m + self.max_events - 1) // self.max_events
            sel = torch.arange(0, m, stride)
            x, y, pol, t_t = x[sel], y[sel], pol[sel], t_t[sel]

        denom = (t_end - t_start) if (t_end - t_start) > 0 else 1.0
        t_norm = ((t_t - t_start) / denom).clamp(0.0, 1.0).to(torch.float32)

        coords = torch.stack([x, y], dim=1)                         # (N, 2) (x, y)
        feats = torch.stack([pol, t_norm], dim=1)                   # (N, 2) per event

        # Per-event target.
        #  * nearest_frame (frame_masks given): label each event from the FLIR frame
        #    nearest its OWN timestamp — discrete motion-compensation that de-smears the
        #    moving boundary. ``frame_masks`` already carry the boundary trimap when
        #    refinement is active.
        #  * else with label refinement on (train only): sample from the single-frame
        #    trimap label map (mask edge -> -1).
        #  * else: the raw single-frame mask.
        # In all refined cases, additionally ignore events far from the window centre.
        if frame_masks is not None:
            ftt = torch.as_tensor(np.asarray(frame_times), dtype=torch.float64)   # (K,)
            choice = (t_t.view(-1, 1) - ftt.view(1, -1)).abs().argmin(dim=1)      # (N,)
            labels = frame_masks[choice, y, x].to(torch.float32)                  # (N,)
            if self._label_refine_active and self.time_ignore_frac > 0.0:
                f = min(self.time_ignore_frac, 0.49)
                far = (t_norm < f) | (t_norm > 1.0 - f)
                labels = labels.masked_fill(far, -1.0)
        elif self._label_refine_active:
            label_map = self._build_label_map(mask)
            labels = label_map[y, x].to(torch.float32)
            if self.time_ignore_frac > 0.0:
                f = min(self.time_ignore_frac, 0.49)
                far = (t_norm < f) | (t_norm > 1.0 - f)
                labels = labels.masked_fill(far, -1.0)
        else:
            labels = mask[y, x].to(torch.float32)                   # (N,)
        return coords, feats, t_norm, labels

    # ------------------------------------------------------------------- sample

    def __getitem__(self, idx: int):
        s_idx, f_idx = self.index[idx]
        seq_dir = self.sequences[s_idx]
        h = self._get_handle(seq_dir)

        t_center = float(h["flir_t"][f_idx])
        half = (self.window_ms / 1000.0) * h["unit"] / 2.0
        t_start = t_center - half
        t_end = t_center + half

        events_t = h["events_t"]
        lo = int(np.searchsorted(events_t, t_start, side="left"))
        hi = int(np.searchsorted(events_t, t_end, side="right"))
        t = np.asarray(events_t[lo:hi])
        xy = np.asarray(h["events_xy"][lo:hi])
        p = np.asarray(h["events_p"][lo:hi])

        mask = self._load_mask(h, f_idx)
        frame_masks, frame_times = self._nearest_frame_masks(
            h, f_idx, t_center, t_start, t_end, mask)
        coords, feats, times, labels = self._build_events(
            t, xy, p, t_start, t_end, mask, frame_masks, frame_times)

        if self._event_aug_active and coords.shape[0] > 0:
            # Per-sample generator: idx mixes in the worker's running global RNG so
            # the augmentation varies across epochs but is reproducible at a fixed
            # global seed (mirrors the dense path's per-sample RNG derivation).
            seed = (idx * 2654435761) ^ int(torch.randint(0, 2 ** 62, (1,)).item())
            gen = torch.Generator().manual_seed(seed % (2 ** 63))
            coords, feats, times, labels = self._augmentor(
                coords, feats, times, labels, self.image_height, self.image_width, gen)

        dense_mask = (mask > 0.5).to(torch.uint8)

        meta = {
            "sequence": seq_dir.name,
            "frame_index": f_idx,
            "t_center": t_center,
            "n_events": int(hi - lo),
            "n_kept": int(coords.shape[0]),
        }
        return coords, feats, times, labels, dense_mask, meta

    # ------------------------------------------------------------------ collate

    @staticmethod
    def collate_fn(samples):
        """Ragged collate selected by ``DataInterface`` via ``getattr(ds, 'collate_fn')``."""
        return collate_sparse_events(samples)
