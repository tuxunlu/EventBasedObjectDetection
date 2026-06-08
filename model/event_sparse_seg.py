"""Event-native per-event segmenter: a submanifold sparse-conv U-Net.

Takes a sparse set of active event sites ``(x, y)`` with per-site features and
emits **one logit per site** — a foreground(hand/arm)/background label per event.
Submanifold convolutions keep the active-site set fixed, so the output sites equal
the input event sites with no dense decoder; strided ``SparseConv`` down/up stages
give the receptive field needed to segment a whole hand from sparse edges. The
network only computes where events fired, so cost scales with event count, not
pixel count.

Shape contract
--------------
``forward(batch: SparseEventBatch) -> logits``:
  * ``(N,)``           when ``num_classes == 1`` (default), or
  * ``(N, num_classes)`` otherwise,
where ``N == batch.coords.shape[0]`` and **row ``i`` corresponds to
``batch.coords[i]`` / ``batch.labels[i]``** (the forward re-aligns spconv's output
to the input order by coordinate key, so this holds regardless of any internal
reordering by the backend).

Parameter budget
----------------
With the default ``stage_channels=(16, 32, 48, 64)`` the model is ≈0.19 M params —
well under the ≤300 K target, leaving room to widen if mask quality is
capacity-limited. ``EventSparseSeg.count_parameters()`` reports the exact number.

Backend
-------
spconv v2 via ``model/sparse_backend.py`` (import-guarded — instantiating this
class is what triggers the spconv import, with a clear install hint on failure).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from model.sparse_backend import build_sparse_tensor, require_spconv


def _make_norm1d(c: int, kind: str) -> nn.Module:
    """Norm over the per-site feature vector ``(N, C)``.

    ``"ln"`` (LayerNorm over channels) is the default and the LOSO-safe choice: it
    normalizes each site independently, so the held-out subject's statistics are
    never mixed across the batch the way ``BatchNorm1d`` over sites would.
    """
    if kind == "bn":
        return nn.BatchNorm1d(c)
    if kind == "ln":
        return nn.LayerNorm(c)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm kind: {kind!r} (expected 'bn', 'ln', or 'none')")


class _SubMBlock(nn.Module):
    """SubMConv -> norm -> ReLU. Preserves the active-site set (per-site refinement)."""

    def __init__(self, c_in: int, c_out: int, indice_key: str, norm: str = "ln", k: int = 3):
        super().__init__()
        spconv = require_spconv()
        self.conv = spconv.SubMConv2d(c_in, c_out, kernel_size=k, bias=False,
                                      indice_key=indice_key)
        self.norm = _make_norm1d(c_out, norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return x.replace_feature(self.act(self.norm(x.features)))


class _DownBlock(nn.Module):
    """Strided SparseConv (halves spatial res) -> norm -> ReLU. Expands the active set."""

    def __init__(self, c_in: int, c_out: int, indice_key: str, norm: str = "ln", k: int = 3):
        super().__init__()
        spconv = require_spconv()
        self.conv = spconv.SparseConv2d(c_in, c_out, kernel_size=k, stride=2,
                                        padding=k // 2, bias=False, indice_key=indice_key)
        self.norm = _make_norm1d(c_out, norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return x.replace_feature(self.act(self.norm(x.features)))


class _UpBlock(nn.Module):
    """SparseInverseConv keyed to a prior ``_DownBlock`` -> norm -> ReLU. Restores its sites."""

    def __init__(self, c_in: int, c_out: int, indice_key: str, norm: str = "ln", k: int = 3):
        super().__init__()
        spconv = require_spconv()
        self.conv = spconv.SparseInverseConv2d(c_in, c_out, kernel_size=k, bias=False,
                                               indice_key=indice_key)
        self.norm = _make_norm1d(c_out, norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return x.replace_feature(self.act(self.norm(x.features)))


class EventSparseSeg(nn.Module):
    """Submanifold sparse-conv U-Net for per-event hand/arm segmentation.

    Parameters
    ----------
    in_features
        Number of per-site input feature channels (matches the dataset's feature
        builder; default 3 = signed polarity, normalized time-in-window, log-count).
    stage_channels
        4-tuple ``(c0, c1, c2, c3)`` of channel widths (stem + three encoder
        levels). Default ``(16, 32, 48, 64)`` ≈ 0.19 M params.
    num_classes
        Output channels per site. ``1`` for a binary foreground logit.
    norm
        ``"ln"`` (default, LOSO-safe), ``"bn"``, or ``"none"``.
    """

    def __init__(
        self,
        in_features: int = 3,
        stage_channels: Sequence[int] = (16, 32, 48, 64),
        num_classes: int = 1,
        norm: str = "ln",
    ):
        super().__init__()
        spconv = require_spconv()  # fail fast with a clear install hint
        if len(stage_channels) != 4:
            raise ValueError("stage_channels must be a 4-tuple (c0, c1, c2, c3)")
        c0, c1, c2, c3 = (int(c) for c in stage_channels)
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)

        # Stem keeps full resolution (submanifold) -> sites == input event sites.
        self.stem = _SubMBlock(in_features, c0, indice_key="subm0", norm=norm)

        # Encoder: each level downsamples (sp{n}) then refines (subm{n}).
        self.down1 = _DownBlock(c0, c1, indice_key="sp1", norm=norm)
        self.enc1 = _SubMBlock(c1, c1, indice_key="subm1", norm=norm)
        self.down2 = _DownBlock(c1, c2, indice_key="sp2", norm=norm)
        self.enc2 = _SubMBlock(c2, c2, indice_key="subm2", norm=norm)
        self.down3 = _DownBlock(c2, c3, indice_key="sp3", norm=norm)
        self.enc3 = _SubMBlock(c3, c3, indice_key="subm3", norm=norm)

        # Decoder: inverse conv keyed to the matching down stage restores its
        # exact sites; skip-add then refine.
        self.up3 = _UpBlock(c3, c2, indice_key="sp3", norm=norm)
        self.dec2 = _SubMBlock(c2, c2, indice_key="subm2d", norm=norm)
        self.up2 = _UpBlock(c2, c1, indice_key="sp2", norm=norm)
        self.dec1 = _SubMBlock(c1, c1, indice_key="subm1d", norm=norm)
        self.up1 = _UpBlock(c1, c0, indice_key="sp1", norm=norm)
        self.dec0 = _SubMBlock(c0, c0, indice_key="subm0d", norm=norm)

        # Per-site 1x1 classifier (submanifold) -> one logit per event site.
        self.head = spconv.SubMConv2d(c0, num_classes, kernel_size=1, bias=True,
                                      indice_key="head")

    @staticmethod
    def _add(a, b):
        """Add features of two sparse tensors sharing the same (inverse-restored) sites."""
        return a.replace_feature(a.features + b.features)

    def forward(self, batch) -> torch.Tensor:
        device = batch.feats.device
        if batch.coords.shape[0] == 0:
            # Whole batch empty (pathological): return an empty, grad-carrying tensor.
            return batch.feats.new_zeros((0,) if self.num_classes == 1 else (0, self.num_classes))

        x = build_sparse_tensor(
            batch.coords, batch.feats, batch.batch_idx,
            batch.height, batch.width, batch.batch_size,
        )

        s0 = self.stem(x)
        e1 = self.enc1(self.down1(s0))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))
        d2 = self.dec2(self._add(self.up3(e3), e2))
        d1 = self.dec1(self._add(self.up2(d2), e1))
        d0 = self.dec0(self._add(self.up1(d1), s0))
        out = self.head(d0)
        feats = out.features                       # (N, num_classes), spconv site order

        # Re-align spconv's output rows to the *input* coord order so logits[i]
        # pairs with batch.labels[i]. Both index sets are identical (submanifold
        # invariant) and unique per (batch, pixel), so a sort-based key match is exact.
        H, W = batch.height, batch.width
        in_key = ((batch.batch_idx.long() * H + batch.coords[:, 1].long()) * W
                  + batch.coords[:, 0].long())
        oi = out.indices.long()                    # columns: [batch, y, x]
        out_key = (oi[:, 0] * H + oi[:, 1]) * W + oi[:, 2]
        if feats.shape[0] != in_key.shape[0]:
            raise RuntimeError(
                f"submanifold site-count invariant violated: model returned "
                f"{feats.shape[0]} sites for {in_key.shape[0]} input events"
            )
        order_in = torch.argsort(in_key)
        order_out = torch.argsort(out_key)
        aligned = torch.empty_like(feats)
        aligned[order_in] = feats[order_out]

        if self.num_classes == 1:
            return aligned.squeeze(-1)             # (N,)
        return aligned                             # (N, num_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(module: nn.Module) -> int:
    """Module-level convenience helper (mirrors ``model/event_unet.py``)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
