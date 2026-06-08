"""Sparse-convolution backend adapter for the event-native models.

Isolates the choice of sparse-conv library (spconv v2 is primary; MinkowskiEngine /
TorchSparse are possible fallbacks) behind a tiny surface, so model code stays
backend-agnostic and — crucially — so the rest of the repository imports cleanly
even when no sparse-conv library (or even torch) is installed. The heavy import is
deferred to model *instantiation* time via :func:`require_spconv`.

spconv coordinate convention
----------------------------
``spconv.pytorch.SparseConvTensor(features, indices, spatial_shape, batch_size)``
expects ``indices`` as an ``int32`` tensor of shape ``(N, ndim + 1)`` whose first
column is the batch index and whose remaining columns are spatial coordinates in
the SAME order as ``spatial_shape``. We use ``spatial_shape = [H, W]`` for 2D, so
the index columns are ``[batch, row(y), col(x)]``.

The event pipeline stores coordinates as ``(x, y)`` (matching ``events_xy`` and
``mask[y, x]`` indexing). :func:`build_sparse_tensor` performs the single
``(x, y) -> (y, x)`` swap, so coordinate ordering is handled in exactly one place
(a deliberate guard against the kind of silent geometry bug that anisotropic
resizes or axis swaps tend to introduce).
"""

from __future__ import annotations

import torch

# Cached handle to the imported ``spconv.pytorch`` module (populated lazily).
_SPCONV = None


def require_spconv():
    """Import and return ``spconv.pytorch``; raise a clear, actionable error if absent.

    Called from the model's ``__init__``/``forward`` rather than at module import
    so that importing this file (and therefore the ``model`` package) never fails
    just because spconv is missing.
    """
    global _SPCONV
    if _SPCONV is not None:
        return _SPCONV
    try:
        import spconv.pytorch as spconv  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "EventSparseSeg requires the `spconv` library (v2). Install the wheel "
            "matching your CUDA toolkit, e.g. `pip install spconv-cu120` for CUDA "
            "12.x or `pip install spconv-cu118` for CUDA 11.8. On Jetson/Orin a "
            f"source build is typically required. Original import error: {exc!r}"
        ) from exc
    _SPCONV = spconv
    return spconv


def build_sparse_tensor(
    coords_xy: torch.Tensor,
    feats: torch.Tensor,
    batch_idx: torch.Tensor,
    height: int,
    width: int,
    batch_size: int,
):
    """Assemble an spconv ``SparseConvTensor`` from (x, y) coords + per-site features.

    Parameters
    ----------
    coords_xy : ``(N, 2)`` int tensor of ``(x, y)`` pixel coordinates (one row per site).
    feats     : ``(N, C)`` float tensor of per-site features.
    batch_idx : ``(N,)`` int tensor mapping each site to its batch element.
    height, width : spatial extent (``spatial_shape = [H, W]``).
    batch_size : number of samples in the batch.

    Returns an ``spconv.pytorch.SparseConvTensor`` on ``feats.device``.
    """
    spconv = require_spconv()
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]
    # spconv index order matches spatial_shape == [H, W]  ->  [batch, y, x].
    indices = torch.stack([batch_idx, y, x], dim=1).to(torch.int32).contiguous()
    return spconv.SparseConvTensor(
        features=feats,
        indices=indices,
        spatial_shape=[int(height), int(width)],
        batch_size=int(batch_size),
    )
