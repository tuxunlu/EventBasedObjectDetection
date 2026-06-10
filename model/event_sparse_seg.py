"""Event-native **3D** per-event segmenter: a submanifold sparse-conv U-Net over
the ``(t, y, x)`` event volume, with a per-event MLP head.

What changed vs. the old 2D version (and why the old one scored ~0.10 IoU)
-------------------------------------------------------------------------
The previous model collapsed every event at a pixel into a single 2D site
(``unique(y*W + x)``), discarding the **entire temporal axis** before a 2D
submanifold U-Net — so it was neither 3D nor truly per-event, and at this
sensor's ~1.2 events/pixel the collapse bought almost no dedup. This rewrite is
genuinely event-native in 3D:

* **3D voxelization.** Events are binned into ``(t_bin, y, x)`` voxels (``T`` time
  bins). Co-pixel events at different times are now *distinct* voxels, so motion
  is visible to the network — the moving hand/arm is a 3D space-time structure.
* **Anisotropic downsampling.** Strided ``SparseConv3d`` uses ``stride=(1, 2, 2)``
  — it grows the spatial receptive field (needed to segment a whole hand from
  sparse edges) while *preserving* temporal resolution. Collapsing ``t`` is the
  documented #1 event-cloud failure mode (PointNet++ dropping to ~54% on
  gestures, Ren 2024); we avoid it structurally.
* **Per-event output head.** Submanifold convs preserve the voxel set, so each
  voxel carries a context feature. A small MLP reads, *for every event*, its
  voxel's context feature ⊕ the event's own ``(polarity, exact normalized time,
  in-voxel time offset)`` and emits **one logit per event** — a true event-stream
  output, distinct even for events sharing a voxel. ``logits[i]`` pairs with
  ``batch.labels[i]`` regardless of any internal spconv reordering.

Architectural basis: submanifold sparse convolution (Graham & van der Maaten
2017; Graham et al. CVPR 2018 — SubMConv U-Net for per-point semantic seg). The
``(T,H,W)`` sparse-voxel design follows the 4D/3D sparse-conv line (Choy et al.
CVPR 2019). Cost scales with the number of active voxels (≈ #events), not the
pixel grid, so inference is cheap and event-rate-proportional; the synchronous
net can later be wrapped with AsyNet-style (Messikommer ECCV 2020) recursive
updates for an asynchronous per-event stream.

Shape contract
--------------
``forward(batch) -> logits``:
  * ``(N,)``            when ``num_classes == 1`` (default), or
  * ``(N, num_classes)`` otherwise,
where ``N == batch.coords.shape[0]`` is the number of **events** and row ``i``
corresponds to ``batch.coords[i]`` / ``batch.labels[i]``.

Expected ``batch`` fields (see ``data/sparse_event_collate.py``):
  ``coords (N,2) long`` (x, y); ``times (N,) float`` in ``[0,1]``;
  ``feats (N,F) float`` per-event features; ``batch_idx (N,) long``;
  ``height``, ``width``, ``batch_size``.

Backend: spconv v2 via ``model/sparse_backend.py`` (import-guarded).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from model.sparse_backend import build_sparse_tensor_3d, require_spconv


def _resolve_algo(algo):
    """Map a friendly algo name (or ``None``/``ConvAlgo``) to an spconv ``ConvAlgo``.

    ``None`` keeps spconv's own default. ``"native"`` (gather + dense cuBLAS GEMM
    + scatter) sidesteps the cumm implicit-GEMM tile heuristics that SIGFPE under a
    CUDA runtime newer than the spconv wheel's build-time CUDA (the reason the old
    config pinned ``algo: native``).
    """
    if algo is None or not isinstance(algo, str):
        return algo
    from spconv.core import ConvAlgo
    table = {
        "native": ConvAlgo.Native,
        "implicit_gemm": ConvAlgo.MaskImplicitGemm,
        "mask_implicit_gemm": ConvAlgo.MaskImplicitGemm,
        "mask_split_implicit_gemm": ConvAlgo.MaskSplitImplicitGemm,
    }
    key = algo.strip().lower()
    if key not in table:
        raise ValueError(f"unknown algo {algo!r}; choose from {sorted(table)}")
    return table[key]


def _make_norm1d(c: int, kind: str) -> nn.Module:
    """Norm over the per-voxel feature vector ``(M, C)``.

    ``"ln"`` (LayerNorm over channels) is the LOSO-safe default: it normalizes each
    voxel independently, so the held-out subject's statistics never mix across the
    batch the way ``BatchNorm1d`` over voxels would.
    """
    if kind == "bn":
        return nn.BatchNorm1d(c)
    if kind == "ln":
        return nn.LayerNorm(c)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm kind: {kind!r} (expected 'bn', 'ln', or 'none')")


class _SubMBlock(nn.Module):
    """SubMConv3d -> norm -> ReLU. Preserves the active-voxel set (per-voxel refine)."""

    def __init__(self, c_in: int, c_out: int, indice_key: str, norm: str = "ln",
                 k: int = 3, algo=None):
        super().__init__()
        spconv = require_spconv()
        self.conv = spconv.SubMConv3d(c_in, c_out, kernel_size=k, bias=False,
                                      indice_key=indice_key, algo=algo)
        self.norm = _make_norm1d(c_out, norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return x.replace_feature(self.act(self.norm(x.features)))


class _DownBlock(nn.Module):
    """Strided SparseConv3d -> norm -> ReLU. Halves spatial res, keeps temporal res.

    ``stride=(1, 2, 2)`` over ``(t, y, x)`` — anisotropic on purpose (see module
    docstring). ``kernel=(1, 3, 3)`` so the downsampling conv is purely spatial
    (temporal mixing is done by the submanifold blocks, which keep ``t`` intact).
    """

    def __init__(self, c_in: int, c_out: int, indice_key: str, norm: str = "ln", algo=None):
        super().__init__()
        spconv = require_spconv()
        self.conv = spconv.SparseConv3d(c_in, c_out, kernel_size=(1, 3, 3),
                                        stride=(1, 2, 2), padding=(0, 1, 1),
                                        bias=False, indice_key=indice_key, algo=algo)
        self.norm = _make_norm1d(c_out, norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return x.replace_feature(self.act(self.norm(x.features)))


class _UpBlock(nn.Module):
    """SparseInverseConv3d keyed to a prior ``_DownBlock`` -> norm -> ReLU. Restores sites."""

    def __init__(self, c_in: int, c_out: int, indice_key: str, norm: str = "ln", algo=None):
        super().__init__()
        spconv = require_spconv()
        self.conv = spconv.SparseInverseConv3d(c_in, c_out, kernel_size=(1, 3, 3),
                                               bias=False, indice_key=indice_key, algo=algo)
        self.norm = _make_norm1d(c_out, norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return x.replace_feature(self.act(self.norm(x.features)))


class EventSparseSeg(nn.Module):
    """3D submanifold sparse-conv U-Net + per-event head for per-event hand/arm seg.

    Parameters
    ----------
    in_features
        Number of per-event input feature channels (the dataset's feature builder;
        default 4 = signed polarity, normalized time-in-window, in-voxel time
        offset, log local count).
    stage_channels
        4-tuple ``(c0, c1, c2, c3)`` of channel widths (stem + three encoder
        levels). Default ``(24, 32, 48, 64)``.
    time_bins
        Number of temporal voxel bins ``T`` (the third spatial dim of the volume).
        Larger ``T`` = finer motion resolution at higher cost. Default 6.
    num_classes
        Output channels per event. ``1`` for a binary foreground logit.
    head_hidden
        Hidden width of the per-event MLP head. Default 64.
    norm
        ``"ln"`` (default, LOSO-safe), ``"bn"``, or ``"none"``.
    algo
        Sparse-conv algorithm forwarded to every spconv layer (see ``_resolve_algo``).
        ``None`` keeps spconv's default; ``"native"`` avoids the implicit-GEMM SIGFPE
        when the wheel's build-time CUDA is older than the torch runtime.
    """

    def __init__(
        self,
        in_features: int = 4,
        stage_channels: Sequence[int] = (24, 32, 48, 64),
        time_bins: int = 6,
        num_classes: int = 1,
        head_hidden: int = 64,
        norm: str = "ln",
        algo=None,
    ):
        super().__init__()
        require_spconv()  # fail fast with a clear install hint
        if len(stage_channels) != 4:
            raise ValueError("stage_channels must be a 4-tuple (c0, c1, c2, c3)")
        c0, c1, c2, c3 = (int(c) for c in stage_channels)
        self.in_features = int(in_features)
        self.time_bins = int(time_bins)
        self.num_classes = int(num_classes)
        algo = _resolve_algo(algo)

        # Per-voxel input = mean of the voxel's per-event features, plus a log
        # event-count channel appended in forward -> in_features + 1.
        vox_in = self.in_features + 1

        # Stem keeps full (t, y, x) resolution (submanifold) -> sites == voxels.
        self.stem = _SubMBlock(vox_in, c0, indice_key="subm0", norm=norm, algo=algo)

        # Encoder: each level spatially downsamples (sp{n}) then refines (subm{n}).
        self.down1 = _DownBlock(c0, c1, indice_key="sp1", norm=norm, algo=algo)
        self.enc1 = _SubMBlock(c1, c1, indice_key="subm1", norm=norm, algo=algo)
        self.down2 = _DownBlock(c1, c2, indice_key="sp2", norm=norm, algo=algo)
        self.enc2 = _SubMBlock(c2, c2, indice_key="subm2", norm=norm, algo=algo)
        self.down3 = _DownBlock(c2, c3, indice_key="sp3", norm=norm, algo=algo)
        self.enc3 = _SubMBlock(c3, c3, indice_key="subm3", norm=norm, algo=algo)

        # Decoder: inverse conv keyed to the matching down stage restores its exact
        # sites; skip-add then refine.
        self.up3 = _UpBlock(c3, c2, indice_key="sp3", norm=norm, algo=algo)
        self.dec2 = _SubMBlock(c2, c2, indice_key="subm2d", norm=norm, algo=algo)
        self.up2 = _UpBlock(c2, c1, indice_key="sp2", norm=norm, algo=algo)
        self.dec1 = _SubMBlock(c1, c1, indice_key="subm1d", norm=norm, algo=algo)
        self.up1 = _UpBlock(c1, c0, indice_key="sp1", norm=norm, algo=algo)
        self.dec0 = _SubMBlock(c0, c0, indice_key="subm0d", norm=norm, algo=algo)

        # Per-EVENT head: voxel context feature ⊕ the event's own raw features ->
        # MLP -> one logit per event. This is what makes the output a true
        # per-event stream (distinct logits for events sharing a voxel).
        self.head = nn.Sequential(
            nn.Linear(c0 + self.in_features, head_hidden),
            nn.LayerNorm(head_hidden) if norm == "ln" else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, num_classes),
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _add(a, b):
        """Add features of two sparse tensors sharing the same (inverse-restored) sites."""
        return a.replace_feature(a.features + b.features)

    def _voxelize(self, x, y, t_bin, batch_idx, feats, T, H, W):
        """Events -> unique ``(b, t_bin, y, x)`` voxels with mean features + a count.

        Returns ``(vox_coords_tyx (M,3), vox_batch (M,), vox_feats (M, F+1),
        inverse (N,))`` where ``inverse`` maps each event to its voxel row.
        """
        x = x.long(); y = y.long(); t_bin = t_bin.long(); b = batch_idx.long()
        key = ((b * T + t_bin) * H + y) * W + x          # unique per (b,t,y,x)
        uniq, inverse = torch.unique(key, sorted=True, return_inverse=True)
        M = uniq.numel()

        # scatter-mean per-event feats into voxels (+ log count channel).
        F = feats.shape[1]
        ones = torch.ones(inverse.shape[0], device=feats.device, dtype=feats.dtype)
        cnt = torch.zeros(M, device=feats.device, dtype=feats.dtype).scatter_add_(0, inverse, ones)
        idx_f = inverse.unsqueeze(1).expand(-1, F)
        summ = torch.zeros(M, F, device=feats.device, dtype=feats.dtype).scatter_add_(0, idx_f, feats)
        vox_feats = summ / cnt.clamp(min=1.0).unsqueeze(1)
        vox_feats = torch.cat([vox_feats, torch.log1p(cnt).unsqueeze(1)], dim=1)

        # decode the unique key back to (b, t, y, x).
        k = uniq
        vx = (k % W); k = torch.div(k, W, rounding_mode="floor")
        vy = (k % H); k = torch.div(k, H, rounding_mode="floor")
        vt = (k % T); k = torch.div(k, T, rounding_mode="floor")
        vb = k
        vox_coords = torch.stack([vt, vy, vx], dim=1).to(torch.int64)
        return vox_coords, vb.to(torch.int64), vox_feats, inverse

    def forward(self, batch) -> torch.Tensor:
        feats = batch.feats
        N = feats.shape[0]
        if N == 0:
            return feats.new_zeros((0,) if self.num_classes == 1 else (0, self.num_classes))

        T, H, W = self.time_bins, int(batch.height), int(batch.width)
        x = batch.coords[:, 0]
        y = batch.coords[:, 1]
        times = batch.times
        t_bin = (times * T).floor().clamp_(0, T - 1)

        vox_coords, vox_batch, vox_feats, inverse = self._voxelize(
            x, y, t_bin, batch.batch_idx, feats, T, H, W,
        )

        sp = build_sparse_tensor_3d(vox_coords, vox_feats, vox_batch, T, H, W, batch.batch_size)

        s0 = self.stem(sp)
        e1 = self.enc1(self.down1(s0))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))
        d2 = self.dec2(self._add(self.up3(e3), e2))
        d1 = self.dec1(self._add(self.up2(d2), e1))
        d0 = self.dec0(self._add(self.up1(d1), s0))

        # Re-align spconv's output rows to *our* voxel order (the order of `uniq`,
        # which `inverse` indexes into). Submanifold preserves the voxel set, so the
        # two index sets are a permutation of each other; match by coordinate key.
        of = d0.features                                   # (M, c0), spconv order
        oi = d0.indices.long()                             # columns [batch, t, y, x]
        out_key = ((oi[:, 0] * T + oi[:, 1]) * H + oi[:, 2]) * W + oi[:, 3]
        in_key = ((vox_batch * T + vox_coords[:, 0]) * H + vox_coords[:, 1]) * W + vox_coords[:, 2]
        if of.shape[0] != in_key.shape[0]:
            raise RuntimeError(
                f"submanifold voxel-count invariant violated: model returned "
                f"{of.shape[0]} voxels for {in_key.shape[0]} input voxels"
            )
        vox_out = torch.empty_like(of)
        vox_out[torch.argsort(in_key)] = of[torch.argsort(out_key)]

        # Per-event head: gather each event's voxel context feature, concat the
        # event's own raw features, MLP -> per-event logit.
        ev_ctx = vox_out[inverse]                          # (N, c0)
        logits = self.head(torch.cat([ev_ctx, feats], dim=1))   # (N, num_classes)

        if self.num_classes == 1:
            return logits.squeeze(-1)                      # (N,)
        return logits                                      # (N, num_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(module: nn.Module) -> int:
    """Module-level convenience helper (mirrors ``model/event_unet.py``)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
