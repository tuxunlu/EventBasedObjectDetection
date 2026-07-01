"""
Differentiable event-stream → dense-tensor converters.

All builders take 1-D arrays/tensors of timestamps (t), pixel coords (xy), and
polarity (p), plus a time window [t_start, t_end) and an (H, W) frame size,
and return a dense torch.Tensor suitable as CNN input.

Conventions
-----------
- Polarity p is taken as boolean-ish: > 0 → positive, otherwise negative.
- Coordinates xy are (x_col, y_row) and clipped to [0, W) x [0, H).
- Output dtype is float32. Inputs may be numpy or torch tensors.
- All builders are vectorised — no Python-level loops over events.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_tensor(x: ArrayLike, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _prepare(
    t: ArrayLike, xy: ArrayLike, p: ArrayLike, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t_t = _to_tensor(t, torch.float64, device)  # high precision for time math
    xy_t = _to_tensor(xy, torch.int64, device)
    x = xy_t[..., 0]
    y = xy_t[..., 1]
    p_t = _to_tensor(p, torch.float32, device)
    pol = torch.where(p_t > 0, torch.ones_like(p_t), -torch.ones_like(p_t))
    return t_t, x, y, pol


def voxel_grid(
    t: ArrayLike,
    xy: ArrayLike,
    p: ArrayLike,
    t_start: float,
    t_end: float,
    bins: int,
    height: int,
    width: int,
    device: torch.device | str = "cpu",
    signed: bool = True,
    split_polarity: bool = False,
) -> torch.Tensor:
    """Zhu et al. 2019 voxel grid with bilinear temporal interpolation.

    Parameters
    ----------
    split_polarity
        If True, keep ON and OFF events in separate channel blocks instead of
        accumulating them into one signed value per (bin, pixel) — where
        opposite polarities cancel and the ON/OFF leading/trailing-edge
        structure (the motion-direction cue) is lost. Output becomes
        ``(2*bins, height, width)``: channels ``[0:bins]`` hold positive-event
        counts, ``[bins:2*bins]`` negative-event counts, each bilinearly
        interpolated across time bins. ``signed`` is ignored in this mode
        (both blocks are non-negative counts).

    Returns
    -------
    Tensor of shape (bins, height, width) — or (2*bins, height, width) when
    ``split_polarity`` — float32. Each event contributes its polarity (signed)
    or 1.0 (unsigned / split) bilinearly to the two surrounding time bins.
    """
    device = torch.device(device)
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start})")
    n_ch = 2 * bins if split_polarity else bins

    t_t, x, y, pol = _prepare(t, xy, p, device)
    in_bounds = (
        (t_t >= t_start) & (t_t < t_end)
        & (x >= 0) & (x < width)
        & (y >= 0) & (y < height)
    )
    t_t = t_t[in_bounds]
    x = x[in_bounds]
    y = y[in_bounds]
    pol = pol[in_bounds]
    if split_polarity or not signed:
        val = torch.ones_like(pol)
    else:
        val = pol

    if t_t.numel() == 0:
        return torch.zeros((n_ch, height, width), dtype=torch.float32, device=device)

    # Normalise time to [0, bins-1] for bilinear interpolation across bins.
    t_norm = ((t_t - t_start) / (t_end - t_start) * (bins - 1)).to(torch.float32)
    t_lo = torch.clamp(t_norm.floor().long(), 0, bins - 1)
    t_hi = torch.clamp(t_lo + 1, 0, bins - 1)
    w_hi = t_norm - t_lo.to(torch.float32)
    w_lo = 1.0 - w_hi

    if split_polarity:
        # Route each event to its polarity's channel block: ON → [0:bins],
        # OFF → [bins:2*bins].
        ch_off = torch.where(pol > 0, torch.zeros_like(t_lo),
                             torch.full_like(t_lo, bins))
        c_lo = t_lo + ch_off
        c_hi = t_hi + ch_off
    else:
        c_lo, c_hi = t_lo, t_hi

    grid = torch.zeros((n_ch, height, width), dtype=torch.float32, device=device)
    flat = grid.view(n_ch, -1)

    lin = (y * width + x).long()
    flat.index_put_((c_lo, lin), val * w_lo, accumulate=True)
    flat.index_put_((c_hi, lin), val * w_hi, accumulate=True)

    return grid


def time_surface(
    t: ArrayLike,
    xy: ArrayLike,
    p: ArrayLike,
    t_start: float,
    t_end: float,
    height: int,
    width: int,
    tau: float | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Two-channel exponential time-surface, one channel per polarity.

    Each output pixel = exp(-(t_end - t_last_event_at_pixel) / tau),
    computed separately for positive and negative polarity. tau defaults
    to (t_end - t_start) / 3.

    Returns Tensor (2, H, W), float32, in [0, 1].
    """
    device = torch.device(device)
    if tau is None:
        tau = (t_end - t_start) / 3.0
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    t_t, x, y, pol = _prepare(t, xy, p, device)
    in_bounds = (
        (t_t >= t_start) & (t_t < t_end)
        & (x >= 0) & (x < width)
        & (y >= 0) & (y < height)
    )
    t_t = t_t[in_bounds]
    x = x[in_bounds]
    y = y[in_bounds]
    pol = pol[in_bounds]

    last = torch.full((2, height, width), -float("inf"), dtype=torch.float64, device=device)
    for c in (0, 1):
        m = (pol > 0) if c == 1 else (pol <= 0)
        if not m.any():
            continue
        tc = t_t[m]
        xc = x[m]
        yc = y[m]
        flat_c = last[c].view(-1)
        lin_c = (yc * width + xc).long()
        flat_c.scatter_reduce_(0, lin_c, tc, reduce="amax", include_self=True)

    age = (t_end - last).clamp(min=0.0)
    surface = torch.exp(-age / tau).to(torch.float32)
    surface = torch.where(torch.isfinite(age), surface, torch.zeros_like(surface))
    return surface


def event_density_map(
    t: ArrayLike,
    xy: ArrayLike,
    t_start: float,
    t_end: float,
    height: int,
    width: int,
    blur_sigma: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-pixel event count in [t_start, t_end), optionally Gaussian-blurred.

    Used as the per-pixel weight in motion-confidence-aware distillation loss.
    Returns Tensor (H, W), float32, normalised to [0, 1] by max-value.
    """
    device = torch.device(device)
    t_t = _to_tensor(t, torch.float64, device)
    xy_t = _to_tensor(xy, torch.int64, device)
    x = xy_t[..., 0]
    y = xy_t[..., 1]
    mask = (t_t >= t_start) & (t_t < t_end) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[mask]
    y = y[mask]

    density = torch.zeros((height, width), dtype=torch.float32, device=device)
    if x.numel() == 0:
        return density
    flat = density.view(-1)
    lin = (y * width + x).long()
    flat.index_put_((lin,), torch.ones_like(lin, dtype=torch.float32), accumulate=True)

    if blur_sigma > 0:
        k = max(3, int(2 * round(3 * blur_sigma) + 1))
        coords = torch.arange(k, dtype=torch.float32, device=device) - (k - 1) / 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * blur_sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        density = torch.nn.functional.conv2d(
            density[None, None], kernel_2d[None, None],
            padding=k // 2
        )[0, 0]

    m = density.max()
    if m > 0:
        density = density / m
    return density
