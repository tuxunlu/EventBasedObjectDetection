"""Event-native per-event segmenter with a **dense global-context bottleneck**.

Why a new model (the diagnosis this architecture is built around)
----------------------------------------------------------------
The prior :class:`model.event_sparse_seg.EventSparseSeg` is a *purely submanifold*
3D sparse-conv U-Net: every conv (and every strided down-conv) only ever connects
voxels that are already active, and submanifold convs **never propagate features
across empty space**. That is exactly the property that keeps it cheap — cost ∝
#active voxels (≈ #events) — but it caps the effective receptive field at the span
of the locally-connected active set. Graham, Engelcke & van der Maaten (CVPR 2018,
arXiv:1711.10275) state the limitation directly: submanifold conv locks the active
set to the input so the field does not grow across empty space and *"two
neighboring connected components are treated completely independently,"* and
strided/pooling layers are *"essential … as they allow information to flow between
disconnected components."*

Three independent measurement passes on this dataset (project memory) converged on
the same conclusion about the model's dominant error:

  * The errors are a **"trajectory wake"** of false positives strung along where the
    hand/arm moved. It is **not** a within-window temporal-order/smear artifact (the
    hand moves only ~0.13× its radius per 30 ms window) and **not** mask
    under-segmentation. Adding temporal resolution (``time_bins`` 6→14, time-resolved
    density, fractional-time head) did **not** beat the simple ``time_bins=6`` GJS
    baseline (0.654 F1); the v2/antismear variants scored 0.640 / 0.626.
  * Locally, per event, busy dynamic background looks **~identical** to the hand —
    background event density is ~0.73× the in-hand density. So a per-event /
    submanifold classifier *cannot* separate "this dense moving edge belongs to the
    connected hand blob" from "this dense moving edge is background" **without a
    global spatial-shape signal**.

The fix this model implements is a cheap **global spatial context** injected into
the per-event features. Submanifold backbones cannot grow that context by stacking
more sparse convs (they don't cross empty space). The standard, cheapest way to get
a whole-frame receptive field is a **dense, low-resolution bottleneck**: at the
encoder's deepest (spatially ``/8``) level the active voxels are scatter-collapsed
over time into a tiny dense ``(B, C, H/8, W/8)`` feature map (≈ 60×80), a small
**dense** 2D context net (whose convs *do* propagate across empty pixels, plus a
global-average-pool branch for true image-level context) processes it, and the
context vector is **broadcast back onto every voxel** before decoding. Now the
decoder and the per-event head reason *with* global hand-blob shape, so they can
veto locally-identical background — directly attacking the wake.

This is the encoder–decoder global-context recipe that makes dense 2D segmentation
work — pyramid/global pooling (PSPNet, Zhao CVPR 2017), atrous context (DeepLab,
Chen 2017), squeeze-excite channel context (Hu CVPR 2018, arXiv:1709.01507) —
ported onto a sparse event backbone the way the strided-sparse→dense-bottleneck
detectors (SECOND, Yan 2018) and sparse point-voxel hybrids (SPVNAS, Tang ECCV
2020, arXiv:2007.16100) fuse a cheap dense/voxel branch with the sparse one. The
dense map is fixed-size and tiny (~2.5 M floats regardless of event count), so
inference stays event-rate-proportional and streaming-friendly — the sparse path
still dominates cost, and the synchronous net can later be wrapped AsyNet-style
(Messikommer ECCV 2020) for async per-event inference.

The train-only auxiliary occupancy head (``aux_shape_head``) follows SA-SSD (He et
al., CVPR 2020): a detachable structure-aware head, *"detached during inference,
introducing no additional computational cost,"* that forces the dense bottleneck to
localize the hand. (The complementary SA-SSD/VoteNet center-vote head and an
rloss-style neighbor-consistency loss are the documented next FP levers, left for a
follow-up so this model's global-context contribution stays attributable.)

Anti-overfitting (the secondary requirement)
--------------------------------------------
The LOSO setup has only ~3 training subjects, so the train→val gap opens within a
few epochs (train F1 ~0.85 vs val ~0.65). This model bakes in the *generalization*
levers prior analysis and the survey flagged as the genuine ones:

  * ``coord_mode='relative'`` (default) — geometry features are **centroid-relative
    offsets** computed per-sample from the event cloud (available at inference; no
    labels needed), not absolute ``(x, y)``. Absolute position is a subject
    work-*location* fingerprint that leaks across the LOSO split; the relative
    offset keeps the useful "where am I within the hand blob" signal without it.
  * ``drop_path`` — stochastic depth (Huang et al. ECCV 2016, arXiv:1603.09382) on
    the residual refine blocks and the global-context injection: closes the gap
    without removing any inference-time capacity.
  * ``dropout`` on the gathered context + inside the head, and the auxiliary
    occupancy head, which is itself a strong regularizer (it forces the bottleneck
    to localize the hand — a task that cannot be memorized per-subject).

Pair with SWA (already wired in ``main.py``; enable in the config) and the
event-native augmentation already configured (EventDrop / area-drop / flips /
jitter) for the weight-averaging + augmentation gains.

Shape contract (identical to :class:`EventSparseSeg`)
-----------------------------------------------------
``forward(batch) -> logits``: ``(N,)`` when ``num_classes == 1`` (default), else
``(N, num_classes)``, with row ``i`` aligned to ``batch.coords[i]`` /
``batch.labels[i]`` regardless of internal spconv reordering. The model stashes a
``(B, 1, G, G)`` occupancy logit map on ``self._aux_logits`` for the train-only aux
supervision in ``model_interface`` (reused unchanged), and exposes
``aux_shape_weight``.

Backend: spconv v2 via ``model/sparse_backend.py`` (import-guarded).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sparse_backend import build_sparse_tensor_3d, require_spconv


# --------------------------------------------------------------------------- #
# Small shared pieces                                                          #
# --------------------------------------------------------------------------- #
def _resolve_algo(algo):
    """Map a friendly algo name (or ``None``/``ConvAlgo``) to an spconv ``ConvAlgo``.

    ``"native"`` (gather + dense GEMM + scatter) sidesteps the implicit-GEMM tile
    heuristics that SIGFPE when the spconv wheel's build-time CUDA is older than the
    torch runtime CUDA (the reason the configs pin ``algo: native``).
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

    ``"ln"`` (LayerNorm over channels) is the LOSO-safe default: each voxel is
    normalized independently, so the held-out subject's statistics never mix across
    the batch the way ``BatchNorm1d`` over voxels would (a silent eval leak).
    """
    if kind == "bn":
        return nn.BatchNorm1d(c)
    if kind == "ln":
        return nn.LayerNorm(c)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm kind: {kind!r} (expected 'bn', 'ln', or 'none')")


class _DropPath(nn.Module):
    """Per-row stochastic depth (Huang et al. ECCV 2016) on a residual branch.

    Drops the whole residual contribution for a random subset of *voxels/events*
    with probability ``p`` at train time (scaled by ``1/(1-p)`` to preserve the
    expectation), identity at eval. Row-wise over ``(M, C)`` features — a regularizer
    that closes the train→val gap with no inference-time capacity change.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], 1).bernoulli_(keep).div_(keep)
        return x * mask


class _SubMBlock(nn.Module):
    """SubMConv3d -> norm -> ReLU. Preserves the active-voxel set (per-voxel refine).

    With ``residual`` and ``c_in == c_out`` the block is ``x + drop_path(f(x))`` so
    stochastic depth has a residual branch to drop.
    """

    def __init__(self, c_in: int, c_out: int, indice_key: str, norm: str = "ln",
                 k: int = 3, algo=None, residual: bool = False, drop_path: float = 0.0):
        super().__init__()
        spconv = require_spconv()
        self.conv = spconv.SubMConv3d(c_in, c_out, kernel_size=k, bias=False,
                                      indice_key=indice_key, algo=algo)
        self.norm = _make_norm1d(c_out, norm)
        self.act = nn.ReLU(inplace=True)
        self.residual = bool(residual) and (c_in == c_out)
        self.drop_path = _DropPath(drop_path) if self.residual else None

    def forward(self, x):
        y = self.conv(x)
        y = y.replace_feature(self.act(self.norm(y.features)))
        if self.residual:
            return x.replace_feature(x.features + self.drop_path(y.features))
        return y


class _DownBlock(nn.Module):
    """Strided SparseConv3d -> norm -> ReLU. Halves spatial res, keeps temporal res.

    ``stride=(1, 2, 2)`` / ``kernel=(1, 3, 3)`` over ``(t, y, x)`` — anisotropic on
    purpose: grow the spatial receptive field but *preserve* the time axis
    (collapsing ``t`` is the documented #1 event-cloud failure mode). These strided
    layers are also what let information flow between disconnected components, per the
    submanifold-conv remedy (Graham et al. 2018).
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


class _DenseContext(nn.Module):
    """Dense low-resolution global-context module — the core of this model.

    Operates on a small dense ``(B, c_in + 1, Hd, Wd)`` map (the time-collapsed
    bottleneck plus a 1-channel occupancy mask). Because it is **dense**, its 3×3
    convs propagate across empty pixels — exactly the cross-empty-space mixing the
    submanifold backbone cannot do — and a parallel **global-average-pool branch**
    (over the valid cells) adds true image-level context (the PSPNet/ASPP/SE idea).
    Output is a ``(B, c_ctx, Hd, Wd)`` context map, broadcast back to the voxels by
    the caller.

    The grid is tiny (≈ 60×80) and fixed-size, so this costs a constant ~few-MFLOP
    regardless of event count — inference stays event-rate-proportional. All norms
    are GroupNorm (no batch statistics) so the held-out subject never leaks in.
    """

    def __init__(self, c_in: int, c_ctx: int, norm_groups: int = 8):
        super().__init__()
        g = max(1, min(norm_groups, c_ctx))
        # +1 input channel = occupancy mask (which cells are real vs. zero-filled).
        self.conv1 = nn.Conv2d(c_in + 1, c_ctx, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g, c_ctx)
        # dilated conv widens the dense receptive field further, still cheaply.
        self.conv2 = nn.Conv2d(c_ctx, c_ctx, kernel_size=3, padding=2, dilation=2, bias=False)
        self.gn2 = nn.GroupNorm(g, c_ctx)
        # global-context branch: image-level descriptor broadcast back over the map.
        self.gpool_fc = nn.Conv2d(c_ctx, c_ctx, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(c_ctx * 2, c_ctx, kernel_size=3, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(g, c_ctx)
        self.act = nn.ReLU(inplace=True)

    def forward(self, dense: torch.Tensor, occ: torch.Tensor) -> torch.Tensor:
        x = torch.cat([dense, occ], dim=1)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.act(self.gn2(self.conv2(x)))
        # whole-frame context: average over valid cells only, MLP, broadcast back.
        denom = occ.flatten(2).sum(dim=2).clamp(min=1.0)               # (B, 1)
        gvec = (x * occ).flatten(2).sum(dim=2) / denom                 # (B, c_ctx)
        gvec = self.gpool_fc(gvec[..., None, None])                    # (B, c_ctx, 1, 1)
        gvec = gvec.expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, gvec], dim=1)
        x = self.act(self.gn3(self.conv3(x)))
        return x


class EventSparseSegGC(nn.Module):
    """3D submanifold sparse-conv U-Net **+ dense global-context bottleneck** + per-event head.

    Parameters
    ----------
    in_features
        Per-event input feature channels emitted by the dataset (2 = signed
        polarity, normalized time-in-window). Geometry/density channels are
        synthesized inside ``forward`` from the post-augmentation events.
    stage_channels
        4-tuple ``(c0, c1, c2, c3)`` (stem + three encoder levels).
    time_bins
        Number of temporal voxel bins ``T``. The winning baseline used 6; finer bins
        did not help, so 6 is the default.
    num_classes
        Output channels per event (1 = binary foreground logit).
    head_hidden
        Hidden width of the per-event MLP head.
    dropout
        Dropout on the gathered per-event context and inside the head MLP.
    drop_path
        Stochastic-depth rate on the residual refine blocks and the global-context
        injection (0 = off). The main inference-cost-free regularizer for the fast
        LOSO overfitting.
    norm
        ``"ln"`` (default, LOSO-safe), ``"bn"``, or ``"none"`` for the sparse path.
    algo
        Sparse-conv algorithm (see ``_resolve_algo``). ``"native"`` avoids the
        implicit-GEMM SIGFPE when the wheel's CUDA is older than the torch runtime.
    coord_mode
        Geometry features appended per event: ``"relative"`` (default) =
        centroid-relative offset (LOSO-robust — no absolute work-location leak),
        ``"absolute"`` = normalized ``(x, y)`` in ``[-1, 1]``, ``"both"`` = 4
        channels, ``"none"`` = no geometry channels.
    density_features
        Append 3 neighborhood density/timing channels (log local count, neighborhood
        event-time mean & std), time-resolved per ``(b, t_bin, y, x)``. Default off —
        kept for ablation; prior runs showed it did not move the needle, and the
        global-context branch is the intended lever.
    density_radius
        Half-width of the square neighborhood for ``density_features`` (kernel
        ``2*r+1``). Default 2 (5×5).
    context_channels
        Width ``c_ctx`` of the dense global-context map. Default 48.
    context_gather
        Also gather the dense context vector directly to each event at its ``/8``
        cell and feed it to the per-event head (a direct global skip in addition to
        the bottleneck injection that flows through the decoder). Default ``True``.
    aux_shape_head
        Emit a coarse ``(B, 1, G, G)`` occupancy logit map (from the dense context)
        on ``self._aux_logits`` for the train-only teacher-mask supervision in
        ``model_interface``. Strong regularizer + forces the context to localize the
        hand. Zero inference cost. Default ``True``.
    aux_grid
        Side ``G`` of the square aux occupancy map. Default 32.
    aux_shape_weight
        Weight on the aux occupancy loss (read by ``model_interface``). Default 0.2.
    """

    def __init__(
        self,
        in_features: int = 2,
        stage_channels: Sequence[int] = (24, 32, 48, 64),
        time_bins: int = 6,
        num_classes: int = 1,
        head_hidden: int = 64,
        dropout: float = 0.2,
        drop_path: float = 0.1,
        norm: str = "ln",
        algo=None,
        coord_mode: str = "relative",
        density_features: bool = False,
        density_radius: int = 2,
        context_channels: int = 48,
        context_gather: bool = True,
        aux_shape_head: bool = True,
        aux_grid: int = 32,
        aux_shape_weight: float = 0.2,
        motion_features: bool = False,
        motion_radius: int = 3,
        motion_dir: bool = False,
        motion_min_count: int = 6,
        null_loss_weight: float = 0.0,
        null_margin: float = 1.0,
        presence_gate: bool = False,
        presence_gate_weight: float = 0.3,
        presence_gate_scale: float = 1.0,
        presence_min_fg: int = 0,
        context_gather_bilinear: bool = False,
        context_veto: bool = False,
    ):
        super().__init__()
        require_spconv()  # fail fast with a clear install hint
        if len(stage_channels) != 4:
            raise ValueError("stage_channels must be a 4-tuple (c0, c1, c2, c3)")
        c0, c1, c2, c3 = (int(c) for c in stage_channels)
        self.in_features = int(in_features)
        self.time_bins = int(time_bins)
        self.num_classes = int(num_classes)
        self.coord_mode = str(coord_mode).strip().lower()
        if self.coord_mode not in ("relative", "absolute", "both", "none"):
            raise ValueError(
                f"coord_mode must be relative|absolute|both|none, got {coord_mode!r}")
        self.density_features = bool(density_features)
        self.density_radius = max(1, int(density_radius))
        self.context_channels = int(context_channels)
        self.context_gather = bool(context_gather)
        self.aux_shape_head = bool(aux_shape_head)
        self.aux_grid = int(aux_grid)
        self.aux_shape_weight = float(aux_shape_weight)
        # Motion descriptor (normal-flow + correlation) channels, and the optional
        # background-prototype "null state" loss. Both default OFF so every existing
        # config/subclass is byte-identical; the gc_motion config opts in.
        self.motion_features = bool(motion_features)
        self.motion_radius = max(1, int(motion_radius))
        self.motion_dir = bool(motion_dir)
        self.motion_min_count = max(3, int(motion_min_count))
        self.null_loss_weight = float(null_loss_weight)
        self.null_margin = float(null_margin)
        # Per-window presence gate (the null state) + per-event shape veto. All default
        # OFF so every existing config/checkpoint is byte-identical.
        self.presence_gate = bool(presence_gate)
        self.presence_gate_weight = float(presence_gate_weight)
        self.presence_gate_scale = float(presence_gate_scale)
        self.presence_min_fg = int(presence_min_fg)
        self.context_gather_bilinear = bool(context_gather_bilinear)
        self.context_veto = bool(context_veto)
        self._aux_logits = None        # set per forward when aux active
        self._event_embedding = None   # set per forward when the null loss is on
        self._presence_logit = None    # set per forward when the presence gate is on

        # Extra per-event channels synthesized in forward (post-augmentation).
        n_geom = {"relative": 2, "absolute": 2, "both": 4, "none": 0}[self.coord_mode]
        # Motion: |grad t| (inverse normal-flow speed), planarity, log local count,
        # (+ 2 unit normal-flow direction channels when motion_dir).
        n_motion = (3 + (2 if self.motion_dir else 0)) if self.motion_features else 0
        self.n_motion = n_motion
        self.n_extra = n_geom + (3 if self.density_features else 0) + n_motion
        feat_dim = self.in_features + self.n_extra
        self._feat_dim = feat_dim
        algo = _resolve_algo(algo)

        # Per-voxel input = mean of the voxel's per-event features + a log-count channel.
        vox_in = feat_dim + 1

        # Stem keeps full (t, y, x) resolution (submanifold) -> sites == voxels.
        self.stem = _SubMBlock(vox_in, c0, indice_key="subm0", norm=norm, algo=algo)

        # Encoder: each level spatially downsamples (sp{n}) then refines (subm{n}).
        # The refine blocks are residual + drop_path (regularization).
        self.down1 = _DownBlock(c0, c1, indice_key="sp1", norm=norm, algo=algo)
        self.enc1 = _SubMBlock(c1, c1, indice_key="subm1", norm=norm, algo=algo,
                               residual=True, drop_path=drop_path)
        self.down2 = _DownBlock(c1, c2, indice_key="sp2", norm=norm, algo=algo)
        self.enc2 = _SubMBlock(c2, c2, indice_key="subm2", norm=norm, algo=algo,
                               residual=True, drop_path=drop_path)
        self.down3 = _DownBlock(c2, c3, indice_key="sp3", norm=norm, algo=algo)
        self.enc3 = _SubMBlock(c3, c3, indice_key="subm3", norm=norm, algo=algo,
                               residual=True, drop_path=drop_path)

        # Dense global-context bottleneck (the new core). Reads the time-collapsed /8
        # bottleneck, returns a (B, c_ctx, Hd, Wd) context map. A projection maps it
        # back to c3 for the residual add onto the sparse bottleneck voxels.
        self.context = _DenseContext(c3, self.context_channels)
        self.ctx_to_c3 = nn.Linear(self.context_channels, c3)
        self.ctx_drop_path = _DropPath(drop_path)

        # Decoder: inverse conv keyed to the matching down stage restores its sites.
        self.up3 = _UpBlock(c3, c2, indice_key="sp3", norm=norm, algo=algo)
        self.dec2 = _SubMBlock(c2, c2, indice_key="subm2d", norm=norm, algo=algo,
                               residual=True, drop_path=drop_path)
        self.up2 = _UpBlock(c2, c1, indice_key="sp2", norm=norm, algo=algo)
        self.dec1 = _SubMBlock(c1, c1, indice_key="subm1d", norm=norm, algo=algo,
                               residual=True, drop_path=drop_path)
        self.up1 = _UpBlock(c1, c0, indice_key="sp1", norm=norm, algo=algo)
        self.dec0 = _SubMBlock(c0, c0, indice_key="subm0d", norm=norm, algo=algo,
                               residual=True, drop_path=drop_path)

        # Per-EVENT head: decoder voxel-context ⊕ event's own features
        # (⊕ optionally the directly-gathered global context) -> one logit per event.
        gathered = self.context_channels if self.context_gather else 0
        self.feat_drop = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Linear(c0 + feat_dim + gathered, head_hidden),
            nn.LayerNorm(head_hidden) if norm == "ln" else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(head_hidden, num_classes),
        )

        # Auxiliary coarse occupancy head off the dense context (train-only supervision).
        if self.aux_shape_head:
            self.aux_shape = nn.Conv2d(self.context_channels, 1, kernel_size=1)
        else:
            self.aux_shape = None

        # Per-window PRESENCE head (the null state): a single image-level "is a moving
        # hand present in this window?" logit, read from the masked global-average of
        # the dense context. Its log-sigmoid multiplicatively (in logit space) gates
        # EVERY per-event logit, so a window with no hand suppresses all events at once
        # — the direct fix for false positives on no-motion windows. Supervised in
        # model_interface by whether the window has any foreground event. Active at
        # train AND inference (the gate uses the PREDICTED presence, never a label).
        if self.presence_gate:
            self.presence_head = nn.Sequential(
                nn.Linear(self.context_channels, self.context_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.context_channels, 1),
            )
        else:
            self.presence_head = None

        # Per-EVENT shape VETO: a log-sigmoid suppression term derived from the event's
        # own gathered global context, so a location with no hand-shape evidence is
        # vetoed even within a hand-present window. Suppress-only (<=0), never boosts.
        if self.context_veto:
            self.veto_head = nn.Linear(self.context_channels if self.context_gather else c0, 1)
        else:
            self.veto_head = None

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

        Fc = feats.shape[1]
        ones = torch.ones(inverse.shape[0], device=feats.device, dtype=feats.dtype)
        cnt = torch.zeros(M, device=feats.device, dtype=feats.dtype).scatter_add_(0, inverse, ones)
        idx_f = inverse.unsqueeze(1).expand(-1, Fc)
        summ = torch.zeros(M, Fc, device=feats.device, dtype=feats.dtype).scatter_add_(0, idx_f, feats)
        vox_feats = summ / cnt.clamp(min=1.0).unsqueeze(1)
        vox_feats = torch.cat([vox_feats, torch.log1p(cnt).unsqueeze(1)], dim=1)

        k = uniq
        vx = (k % W); k = torch.div(k, W, rounding_mode="floor")
        vy = (k % H); k = torch.div(k, H, rounding_mode="floor")
        vt = (k % T); k = torch.div(k, T, rounding_mode="floor")
        vb = k
        vox_coords = torch.stack([vt, vy, vx], dim=1).to(torch.int64)
        return vox_coords, vb.to(torch.int64), vox_feats, inverse

    def _geom_feats(self, x, y, batch_idx, B, H, W):
        """Geometry channels per ``coord_mode`` (absolute / centroid-relative)."""
        if self.coord_mode == "none":
            return None
        xf = x.float(); yf = y.float()
        parts = []
        if self.coord_mode in ("absolute", "both"):
            xn = xf / max(W - 1, 1) * 2.0 - 1.0
            yn = yf / max(H - 1, 1) * 2.0 - 1.0
            parts.append(torch.stack([xn, yn], dim=1))
        if self.coord_mode in ("relative", "both"):
            # Per-sample centroid (mean over that sample's events), computed from the
            # event cloud itself -> available at inference (no labels). Normalize the
            # offset by a FIXED half-diagonal scale so it is ~[-1, 1] and NOT
            # spread-normalized (per-sample spread is a motion-rate fingerprint).
            b = batch_idx.long()
            cnt = torch.zeros(B, device=xf.device, dtype=xf.dtype).scatter_add_(
                0, b, torch.ones_like(xf))
            cx = torch.zeros(B, device=xf.device, dtype=xf.dtype).scatter_add_(0, b, xf)
            cy = torch.zeros(B, device=xf.device, dtype=xf.dtype).scatter_add_(0, b, yf)
            denom = cnt.clamp(min=1.0)
            cx = cx / denom; cy = cy / denom
            scale = 0.5 * float((H ** 2 + W ** 2) ** 0.5)
            dx = (xf - cx[b]) / scale
            dy = (yf - cy[b]) / scale
            parts.append(torch.stack([dx, dy], dim=1))
        return torch.cat(parts, dim=1)

    def _density_feats(self, x, y, times, t_bin, batch_idx, B, T, H, W):
        """3 neighborhood density/timing channels (log count, time mean, time std).

        Time-resolved per ``(b, t_bin, y, x)`` so a trajectory pixel's static-source
        burst at another instant does not inflate the density the hand passage reads.
        """
        if not self.density_features:
            return None
        x = x.long(); y = y.long(); b = batch_idx.long(); tb = t_bin.long()
        dev, dt = times.device, times.dtype
        r = self.density_radius
        k = 2 * r + 1
        lin = ((b * T + tb) * H + y) * W + x
        n = B * T * H * W
        view_shape = (B * T, 1, H, W)
        ones = torch.ones(times.shape[0], device=dev, dtype=dt)
        cnt = torch.zeros(n, device=dev, dtype=dt).scatter_add_(0, lin, ones)
        sumt = torch.zeros(n, device=dev, dtype=dt).scatter_add_(0, lin, times.to(dt))
        sumt2 = torch.zeros(n, device=dev, dtype=dt).scatter_add_(0, lin, (times * times).to(dt))

        def box_sum(g):
            g = g.view(*view_shape)
            g = F.avg_pool2d(g, kernel_size=k, stride=1, padding=r) * float(k * k)
            return g.view(-1)

        bcnt = box_sum(cnt); bsumt = box_sum(sumt); bsumt2 = box_sum(sumt2)
        denom = bcnt.clamp(min=1.0)
        mean_t = bsumt / denom
        std_t = ((bsumt2 / denom) - mean_t * mean_t).clamp(min=0.0).sqrt()
        return torch.stack([torch.log1p(bcnt)[lin], mean_t[lin], std_t[lin]], dim=1)

    def _motion_feats(self, x, y, times, batch_idx, B, H, W):
        """Per-event MOTION descriptor — the cue the global-shape prior lacks.

        Separates events from a coherently MOVING edge (the hand) from static-scene
        / sensor-noise events. From the local Surface of Active Events
        ``t_sae(x, y)`` (most-recent normalized event time per pixel) we fit a plane
        ``t ≈ a·Δx + b·Δy + c`` in a ``(2r+1)²`` neighborhood by occupancy-weighted
        least squares — the event-flow primitive of Benosman et al. (event visual
        flow, IEEE TNNLS 2014) — computed for the WHOLE frame at once via fixed-kernel
        2-D convolutions, then gathered per event. Channels:

          * ``|∇t|``      — gradient magnitude of the time surface = inverse
                            normal-flow speed; ≈0 on a flat/static/empty surface,
                            large for a fast moving edge.
          * ``planarity`` — ``1/(1+residual)``: ≈1 when the local surface is a clean
                            moving-edge ramp, →0 for an incoherent (noise/texture)
                            neighborhood. The "is this real motion" confidence.
          * ``log1p(n)``  — local spatiotemporal event count (BAF correlation gate;
                            an isolated noise event scores ≈0).
          * optional: the unit normal-flow direction ``(a, b)/|∇t|`` (``motion_dir``).

        The grid is fixed-size, so cost is constant in the event count. Computed under
        ``no_grad`` — it is a deterministic hand-crafted INPUT feature, not a learned
        path, so there is nothing to backprop into (coords/times are not parameters).
        """
        if not self.motion_features:
            return None
        with torch.no_grad():
            dev = times.device
            ft = torch.float32                         # condition the plane solve in fp32
            b = batch_idx.long(); xi = x.long(); yi = y.long()
            r = self.motion_radius
            k = 2 * r + 1
            n = B * H * W
            lin = (b * H + yi) * W + xi
            t = times.to(ft)
            # Surface of Active Events: most-recent time per pixel + integer count.
            sae = torch.zeros(n, device=dev, dtype=ft).scatter_reduce_(
                0, lin, t, reduce="amax", include_self=True)
            cnt = torch.zeros(n, device=dev, dtype=ft).index_add_(
                0, lin, torch.ones_like(t))
            sae = sae.view(B, 1, H, W)
            m = (cnt.view(B, 1, H, W) > 0).to(ft)      # occupancy weight in {0,1}
            mt = sae * m                               # m·t  (sae is 0 where empty)
            mt2 = mt * sae                             # m·t²
            # Fixed neighborhood-offset kernels (cross-correlation; offset = u-r, v-r).
            off = (torch.arange(k, device=dev, dtype=ft) - r)
            dxk = off.view(1, 1, 1, k).expand(1, 1, k, k).contiguous()   # Δx (column)
            dyk = off.view(1, 1, k, 1).expand(1, 1, k, k).contiguous()   # Δy (row)
            onek = torch.ones(1, 1, k, k, device=dev, dtype=ft)
            dx2k = dxk * dxk; dy2k = dyk * dyk; dxyk = dxk * dyk

            def cv(src, ker):
                return F.conv2d(src, ker, padding=r).view(-1)[lin]       # gather per event

            # Occupancy-weighted neighborhood sums, gathered at each event's pixel.
            S1 = cv(m, onek); Sx = cv(m, dxk); Sy = cv(m, dyk)
            Sxx = cv(m, dx2k); Syy = cv(m, dy2k); Sxy = cv(m, dxyk)
            St = cv(mt, onek); Stx = cv(mt, dxk); Sty = cv(mt, dyk); Stt = cv(mt2, onek)
            cnt_e = cv(m * cnt.view(B, 1, H, W), onek)                   # Σ neighborhood count

            # Solve the 3×3 weighted normal equations per event (ridge for stability).
            lam = 1e-3
            N = t.shape[0]
            M3 = torch.zeros(N, 3, 3, device=dev, dtype=ft)
            M3[:, 0, 0] = Sxx + lam; M3[:, 0, 1] = Sxy; M3[:, 0, 2] = Sx
            M3[:, 1, 0] = Sxy; M3[:, 1, 1] = Syy + lam; M3[:, 1, 2] = Sy
            M3[:, 2, 0] = Sx;  M3[:, 2, 1] = Sy;  M3[:, 2, 2] = S1 + lam
            rhs = torch.stack([Stx, Sty, St], dim=1).unsqueeze(-1)
            abc = torch.linalg.solve(M3, rhs).squeeze(-1)               # (N, 3)
            a, bb, cc = abc[:, 0], abc[:, 1], abc[:, 2]
            grad_mag = torch.sqrt(a * a + bb * bb + 1e-12)
            resid = (Stt - a * Stx - bb * Sty - cc * St).clamp_min(0.0)
            # Planarity = R² of the plane fit = fraction of local time-variance the
            # plane explains. A coherent moving-edge ramp -> R²≈1; a temporally
            # incoherent flicker -> the plane explains ~nothing -> R²≈0. Scale-free,
            # so it separates motion from look-alike dense background far better than
            # 1/(1+resid). A near-constant surface (total_var≈0) is degenerately
            # planar -> 1 (grad_mag≈0 there lets the head tell "flat" from "ramp").
            mean_t = St / S1.clamp_min(1.0)
            total_var = (Stt - St * mean_t).clamp_min(0.0)        # Σ m (t-mean_t)²
            r2 = 1.0 - resid / total_var.clamp_min(1e-6)
            planarity = torch.where(total_var > 1e-6, r2.clamp(0.0, 1.0),
                                    torch.ones_like(r2))
            # Gate: an under-populated neighborhood cannot constrain a 3-DOF plane, so
            # the fit is degenerate (residual ≈0 -> false-high planarity, noise-driven
            # gradient). Zero the motion evidence there. This doubles as a BAF
            # correlation gate: an isolated event (sparse neighborhood) reads as NO
            # motion — exactly the static-scene noise we want to reject.
            valid = (S1 >= float(self.motion_min_count)).to(ft)
            grad_mag = grad_mag * valid
            planarity = planarity * valid
            chans = [grad_mag, planarity, torch.log1p(cnt_e)]
            if self.motion_dir:
                inv = 1.0 / grad_mag.clamp_min(1e-6)
                chans += [a * inv * valid, bb * inv * valid]
            out = torch.stack(chans, dim=1)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_dense(self, sp, c3, B):
        """Time-collapse the bottleneck sparse tensor into a dense ``(B, c3, Hd, Wd)``
        map (scatter-mean over ``t`` per ``(b, y, x)`` column) plus a ``(B, 1, Hd, Wd)``
        occupancy mask. Returns ``(dense, occ, Hd, Wd, col)`` where ``col`` is the
        per-voxel flat column index (row-aligned to ``sp.features``) used to scatter
        in and gather the processed context back out.
        """
        feats = sp.features                                  # (M, c3)
        idx = sp.indices.long()                              # [b, t, y, x]
        bb, yy, xx = idx[:, 0], idx[:, 2], idx[:, 3]
        _, Hd, Wd = (int(s) for s in sp.spatial_shape)
        n = B * Hd * Wd
        col = (bb * Hd + yy) * Wd + xx                       # (M,) per (b,y,x) column
        cnt = feats.new_zeros(n, 1).index_add_(0, col, feats.new_ones(feats.shape[0], 1))
        summ = feats.new_zeros(n, c3).index_add_(0, col, feats)
        mean = summ / cnt.clamp(min=1.0)
        dense = mean.view(B, Hd, Wd, c3).permute(0, 3, 1, 2).contiguous()
        occ = (cnt > 0).to(feats.dtype).view(B, Hd, Wd, 1).permute(0, 3, 1, 2).contiguous()
        return dense, occ, Hd, Wd, col

    @staticmethod
    def _bilinear_gather(ctx_flat, x, y, batch_idx, Hd, Wd, H, W):
        """Bilinearly sample the dense ``(B*Hd*Wd, C)`` context at each event's
        continuous full-res ``(x, y)``.

        The nearest-``/8``-cell gather makes every ``8×8``-px block share one context
        vector — the source of the rectangular artifacts and of a veto that cannot act
        below cell resolution. Bilinear interpolation over the 4 surrounding cells
        gives a smoothly-varying, per-event-precise context (PointPainting/PVCNN-style
        continuous voxel→point feature propagation). Empty neighbor cells still carry
        context the dense convs propagated in, so the interpolation stays valid.
        """
        b = batch_idx.long()
        # Map full-res pixel centre -> dense-grid continuous coordinate.
        fy = (y.float() + 0.5) * Hd / H - 0.5
        fx = (x.float() + 0.5) * Wd / W - 0.5
        y0 = torch.floor(fy); x0 = torch.floor(fx)
        wy = (fy - y0); wx = (fx - x0)
        y0 = y0.long(); x0 = x0.long(); y1 = y0 + 1; x1 = x0 + 1
        y0c = y0.clamp(0, Hd - 1); y1c = y1.clamp(0, Hd - 1)
        x0c = x0.clamp(0, Wd - 1); x1c = x1.clamp(0, Wd - 1)

        def at(yi, xi):
            return ctx_flat[(b * Hd + yi) * Wd + xi]
        c00 = at(y0c, x0c); c01 = at(y0c, x1c)
        c10 = at(y1c, x0c); c11 = at(y1c, x1c)
        wy = wy.unsqueeze(1); wx = wx.unsqueeze(1)
        top = c00 * (1 - wx) + c01 * wx
        bot = c10 * (1 - wx) + c11 * wx
        return top * (1 - wy) + bot * wy

    def forward(self, batch) -> torch.Tensor:
        self._aux_logits = None
        self._event_embedding = None
        self._presence_logit = None
        feats = batch.feats
        N = feats.shape[0]
        if N == 0:
            return feats.new_zeros((0,) if self.num_classes == 1 else (0, self.num_classes))

        T, H, W = self.time_bins, int(batch.height), int(batch.width)
        B = batch.batch_size
        x = batch.coords[:, 0]
        y = batch.coords[:, 1]
        times = batch.times
        t_bin = (times * T).floor().clamp_(0, T - 1)

        # Enrich raw (polarity, time) with geometry (relative by default) + optional
        # density, then use the enriched set everywhere (voxel mean + per-event head).
        parts = [feats]
        gfeat = self._geom_feats(x, y, batch.batch_idx, B, H, W)
        if gfeat is not None:
            parts.append(gfeat.to(feats.dtype))
        dfeat = self._density_feats(x, y, times, t_bin, batch.batch_idx, B, T, H, W)
        if dfeat is not None:
            parts.append(dfeat.to(feats.dtype))
        mfeat = self._motion_feats(x, y, times, batch.batch_idx, B, H, W)
        if mfeat is not None:
            parts.append(mfeat.to(feats.dtype))
        feats = torch.cat(parts, dim=1) if len(parts) > 1 else feats

        vox_coords, vox_batch, vox_feats, inverse = self._voxelize(
            x, y, t_bin, batch.batch_idx, feats, T, H, W,
        )
        sp = build_sparse_tensor_3d(vox_coords, vox_feats, vox_batch, T, H, W, B)

        s0 = self.stem(sp)
        e1 = self.enc1(self.down1(s0))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))

        # ---- dense global-context bottleneck -------------------------------------
        c3 = e3.features.shape[1]
        dense, occ, Hd, Wd, col = self._build_dense(e3, c3, B)
        ctx_map = self.context(dense, occ)                   # (B, c_ctx, Hd, Wd)
        # Broadcast back onto every bottleneck voxel (gather its (b,y,x) column),
        # project to c3, residual-add (with drop_path) so the decoder reasons with
        # global shape. `col` is row-aligned to e3.features.
        ctx_flat = ctx_map.permute(0, 2, 3, 1).reshape(B * Hd * Wd, self.context_channels)
        vox_ctx = ctx_flat[col]                              # (M3, c_ctx)
        e3 = e3.replace_feature(
            e3.features + self.ctx_drop_path(self.ctx_to_c3(vox_ctx)))

        # Per-window presence logit from the masked global-average of the context map
        # (true image-level descriptor). Stashed for the auxiliary BCE in
        # model_interface and used below to gate every event's logit.
        if self.presence_head is not None:
            denom = occ.flatten(2).sum(dim=2).clamp(min=1.0)             # (B, 1)
            gdesc = (ctx_map * occ).flatten(2).sum(dim=2) / denom        # (B, c_ctx)
            self._presence_logit = self.presence_head(gdesc).squeeze(-1)  # (B,)

        # Aux coarse occupancy map (train-only supervision in model_interface).
        if self.aux_shape is not None and self.training:
            G = self.aux_grid
            pooled = F.adaptive_avg_pool2d(ctx_map, (G, G))
            self._aux_logits = self.aux_shape(pooled)        # (B, 1, G, G)

        d2 = self.dec2(self._add(self.up3(e3), e2))
        d1 = self.dec1(self._add(self.up2(d2), e1))
        d0 = self.dec0(self._add(self.up1(d1), s0))

        # Re-align spconv's output rows to OUR voxel order (the order of `uniq`, which
        # `inverse` indexes). Submanifold preserves the voxel set, so the two index
        # sets are a permutation; match by coordinate key.
        of = d0.features                                     # (M, c0), spconv order
        oi = d0.indices.long()
        out_key = ((oi[:, 0] * T + oi[:, 1]) * H + oi[:, 2]) * W + oi[:, 3]
        in_key = ((vox_batch * T + vox_coords[:, 0]) * H + vox_coords[:, 1]) * W + vox_coords[:, 2]
        if of.shape[0] != in_key.shape[0]:
            raise RuntimeError(
                f"submanifold voxel-count invariant violated: model returned "
                f"{of.shape[0]} voxels for {in_key.shape[0]} input voxels")
        vox_out = torch.empty_like(of)
        vox_out[torch.argsort(in_key)] = of[torch.argsort(out_key)]

        # Per-event head: decoder context ⊕ event features (⊕ direct global context).
        ev_ctx = self.feat_drop(vox_out[inverse])            # (N, c0)
        head_parts = [ev_ctx, feats]
        gathered_ctx = None
        if self.context_gather:
            # Gather the dense context at each event directly (a global skip straight to
            # the head, routed by the event's own coords -> preserves the logits[i] <->
            # labels[i] contract). Bilinear (default off) removes the 8×8-block sharing
            # that nearest-cell gather imposes. Empty cells still carry context the dense
            # convs propagated from neighbors — the whole point.
            if self.context_gather_bilinear:
                gathered_ctx = self._bilinear_gather(
                    ctx_flat, x, y, batch.batch_idx, Hd, Wd, H, W)
            else:
                egy = (y.long() * Hd // H).clamp(0, Hd - 1)
                egx = (x.long() * Wd // W).clamp(0, Wd - 1)
                ecol = (batch.batch_idx.long() * Hd + egy) * Wd + egx
                gathered_ctx = ctx_flat[ecol]
            head_parts.append(gathered_ctx)                  # (N, c_ctx)
        # Run the head in two pieces so the penultimate EMBEDDING is exposed for the
        # background-prototype null loss (``self.head`` itself is unchanged, so the
        # gate/ml subclasses that call it as a whole still work).
        head_in = torch.cat(head_parts, dim=1)
        emb = self.head[:-1](head_in)
        logits = self.head[-1](emb)
        if self.training and self.null_loss_weight > 0.0:
            self._event_embedding = emb

        # Per-event shape VETO: suppress-only (log-sigmoid <= 0) gate from the event's
        # gathered context (falls back to decoder context if context_gather is off).
        if self.veto_head is not None:
            vsrc = gathered_ctx if gathered_ctx is not None else ev_ctx
            logits = logits + F.logsigmoid(self.veto_head(vsrc))

        # Per-window PRESENCE gate (null state): add the window's log-sigmoid presence
        # to every event logit. presence→1 ⇒ +0 (no change); presence→0 ⇒ large
        # negative ⇒ all events suppressed. Applied at train AND eval (uses the
        # predicted presence, no label) so the operating point matches.
        if self._presence_logit is not None:
            gate = self.presence_gate_scale * F.logsigmoid(self._presence_logit)
            logits = logits + gate[batch.batch_idx.long()].view(logits.shape)

        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(module: nn.Module) -> int:
    """Module-level convenience helper (mirrors ``model/event_unet.py``)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
