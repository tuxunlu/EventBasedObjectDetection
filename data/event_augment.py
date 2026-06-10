"""Event-native augmentation for the per-event (sparse) segmentation path.

The dense voxel path augments by warping a ``(C,H,W)`` voxel + its mask
(``data/hand_event_dataset._augment_pair``). That does not apply here: a sample is
a *set of individual events* ``(x, y, t, polarity)`` with a per-event label, not a
grid. :class:`EventAugmentor` therefore transforms the raw event arrays directly,
and — crucially — **every per-event label travels with its event**, so the
supervision stays valid under every transform (an event that moves keeps its
``mask[y,x]`` label; events pushed out of frame are dropped together with their
labels; injected sensor-noise events are labeled background).

Why this matters here: the LOSO split holds out a whole subject, so the train→val
gap is largely cross-subject (different hand size/pose/position, motion speed →
event density, and noise floor). The pipeline below targets exactly those degrees
of freedom — geometric (pose/size/position), density (drop/sample), and noise —
plus event-specific temporal/polarity transforms.

Pipeline order (each step gated by its own probability):
  1. geometric:   h/v-flip, affine (rotate/scale/translate), sub-pixel jitter
  2. bounds:      round to pixels, drop out-of-frame events (+ their labels)
  3. density:     random event drop (speed/density invariance), area drop (occlusion)
  4. temporal:    time-crop (+renormalize), time-reverse (with polarity flip),
                  polarity flip, time jitter
  5. noise:       inject background sensor-noise events (label 0)

All randomness comes from a per-sample ``torch.Generator`` for reproducibility.
``feats[:,0]`` is signed polarity and ``feats[:,1]`` is normalized time; the
augmentor keeps ``feats[:,1] == times`` consistent (the model rebins time from it).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch


def _coin(p: float, gen: torch.Generator) -> bool:
    return p > 0.0 and float(torch.rand((), generator=gen)) < p


def _uniform(lo: float, hi: float, gen: torch.Generator) -> float:
    return lo + float(torch.rand((), generator=gen)) * (hi - lo)


class EventAugmentor:
    """Config-driven event-stream augmentation (train only).

    Recognized config keys (all default to a no-op):

        enabled:            False
        hflip_prob:         0.0     # mirror x  (hand can come from either side)
        vflip_prob:         0.0     # mirror y  (use sparingly — changes up/down)
        affine_prob:        0.0     # apply rotate+scale+translate as one transform
        rotate_deg:         0.0     # max |rotation| degrees, sampled uniformly
        scale_range:        [1,1]   # uniform zoom range (hand size varies by subject)
        translate_frac:     0.0     # max |translation| as a fraction of (W,H)
        jitter_px:          0.0     # per-event Gaussian coord jitter std (px)
        event_drop_prob:    0.0     # prob of random event dropout (EventDrop)
        event_drop_frac:    [0,0]   # [min,max] fraction of events to drop
        upsample_prob:      0.0     # prob of event-rate UPSAMPLE (the inverse of drop)
        upsample_range:     [1,1]   # [min,max] density multiplier r (duplicate r-1 frac)
        area_drop_prob:     0.0     # prob of dropping all events in a random box
        area_drop_frac:     0.0     # box side as a fraction of (W,H)
        time_crop_prob:     0.0     # prob of keeping only a temporal sub-window
        time_crop_min:      1.0     # min retained temporal fraction
        time_reverse_prob:  0.0     # reverse time AND flip polarity (physical)
        polarity_flip_prob: 0.0     # flip polarity only
        time_jitter:        0.0     # Gaussian jitter std on normalized time
        noise_frac:         0.0     # add this fraction of background-noise events
        min_events:         64      # never drop a sample below this many events
    """

    def __init__(self, cfg: Optional[Dict[str, Any]]):
        c = dict(cfg or {})
        self.enabled = bool(c.get("enabled", False))
        self.hflip_prob = float(c.get("hflip_prob", 0.0))
        self.vflip_prob = float(c.get("vflip_prob", 0.0))
        self.affine_prob = float(c.get("affine_prob", 0.0))
        self.rotate_deg = float(c.get("rotate_deg", 0.0))
        sr = c.get("scale_range", [1.0, 1.0])
        self.scale_range = (float(sr[0]), float(sr[1]))
        self.translate_frac = float(c.get("translate_frac", 0.0))
        self.jitter_px = float(c.get("jitter_px", 0.0))
        self.event_drop_prob = float(c.get("event_drop_prob", 0.0))
        edf = c.get("event_drop_frac", [0.0, 0.0])
        self.event_drop_frac = (float(edf[0]), float(edf[1]))
        self.upsample_prob = float(c.get("upsample_prob", 0.0))
        ur = c.get("upsample_range", [1.0, 1.0])
        self.upsample_range = (float(ur[0]), float(ur[1]))
        self.area_drop_prob = float(c.get("area_drop_prob", 0.0))
        self.area_drop_frac = float(c.get("area_drop_frac", 0.0))
        self.time_crop_prob = float(c.get("time_crop_prob", 0.0))
        self.time_crop_min = float(c.get("time_crop_min", 1.0))
        self.time_reverse_prob = float(c.get("time_reverse_prob", 0.0))
        self.polarity_flip_prob = float(c.get("polarity_flip_prob", 0.0))
        self.time_jitter = float(c.get("time_jitter", 0.0))
        self.noise_frac = float(c.get("noise_frac", 0.0))
        self.min_events = int(c.get("min_events", 64))

    # ------------------------------------------------------------------ apply
    def __call__(
        self,
        coords: torch.Tensor,    # (N,2) long (x,y)
        feats: torch.Tensor,     # (N,2) float [polarity, t_norm]
        times: torch.Tensor,     # (N,)  float in [0,1]  (== feats[:,1])
        labels: torch.Tensor,    # (N,)  float
        height: int,
        width: int,
        gen: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enabled or coords.shape[0] == 0:
            return coords, feats, times, labels

        H, W = int(height), int(width)
        x = coords[:, 0].float()
        y = coords[:, 1].float()
        pol = feats[:, 0].clone()
        t = times.clone().float()
        lab = labels.clone().float()

        # 1. geometric ---------------------------------------------------------
        if _coin(self.hflip_prob, gen):
            x = (W - 1) - x
        if _coin(self.vflip_prob, gen):
            y = (H - 1) - y
        if _coin(self.affine_prob, gen):
            ang = math.radians(_uniform(-self.rotate_deg, self.rotate_deg, gen))
            s = _uniform(self.scale_range[0], self.scale_range[1], gen)
            tx = _uniform(-self.translate_frac, self.translate_frac, gen) * W
            ty = _uniform(-self.translate_frac, self.translate_frac, gen) * H
            cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
            ca, sa = math.cos(ang), math.sin(ang)
            xr, yr = x - cx, y - cy
            x = cx + tx + s * (ca * xr - sa * yr)
            y = cy + ty + s * (sa * xr + ca * yr)
        if self.jitter_px > 0.0:
            x = x + torch.randn(x.shape, generator=gen) * self.jitter_px
            y = y + torch.randn(y.shape, generator=gen) * self.jitter_px

        # 2. bounds: round to pixels, drop out-of-frame (labels go with them) --
        xi = x.round().long()
        yi = y.round().long()
        inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        xi, yi, pol, t, lab = xi[inb], yi[inb], pol[inb], t[inb], lab[inb]

        # 3. density: random drop, event-rate upsample, then area (occlusion) drop
        n = xi.shape[0]
        if n > self.min_events and _coin(self.event_drop_prob, gen):
            frac = _uniform(self.event_drop_frac[0], self.event_drop_frac[1], gen)
            keep_n = max(self.min_events, int(round(n * (1.0 - frac))))
            perm = torch.randperm(n, generator=gen)[:keep_n]
            xi, yi, pol, t, lab = xi[perm], yi[perm], pol[perm], t[perm], lab[perm]
        # Event-rate UPSAMPLE: duplicate sampled events with ≤1px / small-Δt jitter
        # (so they are *distinct* voxels, not deduped) and inherit their labels — the
        # high-event-rate counterpart to drop. Covers the low-rate-train→high-rate-test
        # half of the cross-subject density gap that EventDrop alone cannot.
        if xi.shape[0] > 0 and _coin(self.upsample_prob, gen):
            r = _uniform(self.upsample_range[0], self.upsample_range[1], gen)
            m = int(round((r - 1.0) * xi.shape[0]))
            if m > 0:
                src = torch.randint(0, xi.shape[0], (m,), generator=gen)
                dx = torch.randint(-1, 2, (m,), generator=gen)
                dy = torch.randint(-1, 2, (m,), generator=gen)
                ux = (xi[src] + dx).clamp(0, W - 1)
                uy = (yi[src] + dy).clamp(0, H - 1)
                ut = (t[src] + torch.randn(m, generator=gen) * 0.01).clamp(0.0, 1.0)
                xi = torch.cat([xi, ux]); yi = torch.cat([yi, uy])
                pol = torch.cat([pol, pol[src]]); t = torch.cat([t, ut])
                lab = torch.cat([lab, lab[src]])           # duplicate inherits its label
        if xi.shape[0] > self.min_events and _coin(self.area_drop_prob, gen):
            bw, bh = int(self.area_drop_frac * W), int(self.area_drop_frac * H)
            if bw > 0 and bh > 0:
                bx = int(_uniform(0, max(W - bw, 1), gen))
                by = int(_uniform(0, max(H - bh, 1), gen))
                inbox = (xi >= bx) & (xi < bx + bw) & (yi >= by) & (yi < by + bh)
                keep = ~inbox
                if int(keep.sum()) >= self.min_events:
                    xi, yi, pol, t, lab = xi[keep], yi[keep], pol[keep], t[keep], lab[keep]

        # 4. temporal / polarity ----------------------------------------------
        if _coin(self.time_crop_prob, gen):
            length = _uniform(self.time_crop_min, 1.0, gen)
            start = _uniform(0.0, 1.0 - length, gen)
            sel = (t >= start) & (t < start + length)
            if int(sel.sum()) >= self.min_events:
                xi, yi, pol, lab = xi[sel], yi[sel], pol[sel], lab[sel]
                t = ((t[sel] - start) / max(length, 1e-6)).clamp(0.0, 1.0)
        if _coin(self.time_reverse_prob, gen):
            t = 1.0 - t
            pol = -pol                      # reversing motion flips event polarity
        if _coin(self.polarity_flip_prob, gen):
            pol = -pol
        if self.time_jitter > 0.0:
            t = (t + torch.randn(t.shape, generator=gen) * self.time_jitter).clamp(0.0, 1.0)

        # 5. background-noise injection (labeled background) -------------------
        if self.noise_frac > 0.0 and xi.shape[0] > 0:
            m = int(self.noise_frac * xi.shape[0])
            if m > 0:
                nx = torch.randint(0, W, (m,), generator=gen)
                ny = torch.randint(0, H, (m,), generator=gen)
                nt = torch.rand(m, generator=gen)
                npol = torch.where(torch.rand(m, generator=gen) > 0.5,
                                   torch.ones(m), -torch.ones(m))
                xi = torch.cat([xi, nx])
                yi = torch.cat([yi, ny])
                t = torch.cat([t, nt])
                pol = torch.cat([pol, npol])
                lab = torch.cat([lab, torch.zeros(m)])      # noise = background

        coords = torch.stack([xi.long(), yi.long()], dim=1)
        t = t.float()
        feats = torch.stack([pol.float(), t], dim=1)
        return coords, feats, t, lab.float()
