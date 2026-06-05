"""Phase-B ANN baseline: a residual depthwise-separable U-Net that distills
SAM 2's hand/arm mask from an event voxel grid.

Supervision contract
---------------------
This model is trained on the **cached SAM 2 teacher mask only** — there is no
online RGB teacher and no feature distillation on this path. ``ModelInterface.
_segmentation_step`` calls ``logits = self(voxel)`` and supervises with
``DistillationLoss`` = ``BCE + Dice`` against the cached mask. The whole point
of the architecture below is to transfer SAM 2's *globally-reasoned* mask into
an event-only student that, frame to frame, only sees sparse motion.

Input:  ``(B, C_in, H, W)`` where ``C_in == voxel_bins`` (default 5).
Output: ``(B, num_classes, H, W)`` full-resolution segmentation logits
        (``num_classes == 1`` for binary hand+arm masks), spatially aligned
        with the cached mask so BCE/Dice apply at native resolution.

Why this shape of network (vs. a plain tiny U-Net)
--------------------------------------------------
SAM 2's masks come from a model with a global receptive field — it labels a
*stationary* forearm as "arm" because it reasons about the whole scene. An
event student only sees pixels that moved in the last window, so the single
hardest part of the distillation is recovering that global, shape-level prior
from a sparse, locally-informative input. Three design choices target that:

1. **Global-context bottleneck** (``_ContextModule``): an ASPP-lite block —
   parallel *dilated* depthwise convs (rates 1/2/4) plus an image-level
   global-pooling branch — fused by a 1×1. At the bottleneck stride this gives
   a near-global receptive field for only a few hundred K params, which is the
   single biggest lever for matching a global teacher from local evidence.
2. **Residual DW-separable blocks** (``_ResidualDW``): residual connections so
   the deeper/global features add to, rather than overwrite, the high-resolution
   motion evidence — and gradients reach the stem cleanly.
3. **Squeeze-excitation gating** (``_SE``): cheap per-channel reweighting.
   Voxel time-bins carry uneven information when motion is bursty; letting the
   network rescale channels helps it lean on the bins that actually fired.

The decoder mirrors the encoder with bilinear upsampling + skip connections and
upsamples all the way back to full resolution for a dense mask.

Parameter budget
-----------------
With the config default ``encoder_channels=[48, 96, 128, 160]`` the model is
~0.65 M params (run this file as ``__main__`` for the exact count) — inside the
plan's 0.4–0.8 M target band. ``[64, 128, 192, 256]`` widens it to ~1.4 M;
``use_context=False`` / ``use_se=False`` strip it back toward the local
baseline for ablations.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(c: int, kind: str, gn_groups: int) -> nn.Module:
    """Factory for the normalization layer used throughout the encoder/decoder.

    ``"gn"`` is preferred for LOSO training: it has no running statistics,
    so the held-out subject's voxel distribution doesn't get rescaled by
    training-subject means. Pick group count that divides the channel width.
    """
    if kind == "bn":
        return nn.BatchNorm2d(c)
    if kind == "gn":
        g = min(gn_groups, c)
        while g > 1 and c % g != 0:
            g -= 1
        return nn.GroupNorm(num_groups=g, num_channels=c)
    raise ValueError(f"unknown norm kind: {kind!r} (expected 'bn' or 'gn')")


class _DWSepConv(nn.Module):
    """Depthwise (3×3, optionally dilated) + norm + ReLU + pointwise 1×1 + norm.

    The trailing activation is optional so this block can be the last op before
    a residual add (post-activation is then applied by the caller).
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1, dilation: int = 1,
                 norm: str = "gn", gn_groups: int = 8, final_act: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride,
                            padding=dilation, dilation=dilation,
                            groups=c_in, bias=False)
        self.n1 = _make_norm(c_in, norm, gn_groups)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.n2 = _make_norm(c_out, norm, gn_groups)
        self.act = nn.ReLU(inplace=True)
        self.final_act = final_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.n1(self.dw(x)))
        x = self.n2(self.pw(x))
        return self.act(x) if self.final_act else x


class _SE(nn.Module):
    """Squeeze-excitation: global-pooled channel descriptor → per-channel gate.

    Cheap (two 1×1 convs on a 1×1 map). Helps the student lean on the voxel
    bins / feature channels that actually carry motion in a given window.
    """

    def __init__(self, c: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, c // reduction)
        self.fc1 = nn.Conv2d(c, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, c, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.act(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class _ResidualDW(nn.Module):
    """Two DW-separable convs + SE, wrapped in a residual connection.

    Handles a change of width and/or a stride-2 downsample via a 1×1 projection
    on the identity branch, so the block doubles as both the encoder's
    downsampler and its refinement stage.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1,
                 norm: str = "gn", gn_groups: int = 8, use_se: bool = True):
        super().__init__()
        self.conv1 = _DWSepConv(c_in, c_out, stride=stride,
                                norm=norm, gn_groups=gn_groups)
        # Last conv emits pre-activation features so the residual add happens
        # before the final ReLU (standard post-activation residual block).
        self.conv2 = _DWSepConv(c_out, c_out, stride=1,
                                norm=norm, gn_groups=gn_groups, final_act=False)
        self.se = _SE(c_out) if use_se else nn.Identity()

        if stride != 1 or c_in != c_out:
            self.proj: nn.Module = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                _make_norm(c_out, norm, gn_groups),
            )
        else:
            self.proj = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.se(y)
        return self.act(y + identity)


class _ContextModule(nn.Module):
    """ASPP-lite global-context block applied at the bottleneck.

    Parallel depthwise *dilated* 3×3 branches (one per dilation rate) plus an
    image-level global-average-pool branch, each reduced to ``c // 2`` channels
    by a 1×1, concatenated, then fused back to ``c`` by a 1×1. The dilations
    enlarge the effective receptive field without extra downsampling, and the
    pooled branch injects a truly global descriptor — together they recover the
    scene-level reasoning that lets SAM 2 label a momentarily-still limb.
    """

    def __init__(self, c: int, dilations: Sequence[int] = (1, 2, 4),
                 norm: str = "gn", gn_groups: int = 8):
        super().__init__()
        branch_c = max(8, c // 2)

        self.dilated = nn.ModuleList()
        for d in dilations:
            self.dilated.append(nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=d, dilation=d,
                          groups=c, bias=False),
                _make_norm(c, norm, gn_groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, branch_c, kernel_size=1, bias=False),
                _make_norm(branch_c, norm, gn_groups),
                nn.ReLU(inplace=True),
            ))

        # Image-level branch: pool to 1×1, project, then broadcast back.
        self.global_branch = nn.Sequential(
            nn.Conv2d(c, branch_c, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        fused_in = branch_c * (len(dilations) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_in, c, kernel_size=1, bias=False),
            _make_norm(c, norm, gn_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.dilated]
        g = self.global_branch(F.adaptive_avg_pool2d(x, 1))
        g = g.expand(-1, -1, x.shape[-2], x.shape[-1])
        feats.append(g)
        return self.fuse(torch.cat(feats, dim=1))


class _EncoderStage(nn.Module):
    """Residual DW-sep block (optionally stride-2) then a residual refinement."""

    def __init__(self, c_in: int, c_out: int, downsample: bool,
                 norm: str = "gn", gn_groups: int = 8, use_se: bool = True):
        super().__init__()
        first_stride = 2 if downsample else 1
        self.body = nn.Sequential(
            _ResidualDW(c_in, c_out, stride=first_stride,
                        norm=norm, gn_groups=gn_groups, use_se=use_se),
            _ResidualDW(c_out, c_out, stride=1,
                        norm=norm, gn_groups=gn_groups, use_se=use_se),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _DecoderStage(nn.Module):
    """Bilinear upsample + skip concat + two residual DW-sep refinement blocks."""

    def __init__(self, c_in: int, c_skip: int, c_out: int,
                 norm: str = "gn", gn_groups: int = 8, use_se: bool = True):
        super().__init__()
        self.body = nn.Sequential(
            _ResidualDW(c_in + c_skip, c_out, stride=1,
                        norm=norm, gn_groups=gn_groups, use_se=use_se),
            _ResidualDW(c_out, c_out, stride=1,
                        norm=norm, gn_groups=gn_groups, use_se=use_se),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.body(torch.cat([x, skip], dim=1))


class EventUnet(nn.Module):
    """Residual DW-separable U-Net that distills SAM 2 masks from event voxels.

    Parameters
    ----------
    in_channels
        Number of voxel time bins fed as input channels (default 5, matching
        ``HandEventDataset.voxel_bins``).
    encoder_channels
        Per-stage channel widths of the encoder, from the highest resolution
        (post-stem) to the bottleneck. Length controls depth; the config
        default is ``[64, 128, 192, 256]``.
    num_classes
        Output channels of the segmentation head. ``1`` for binary hand+arm.
    norm
        ``"gn"`` (default) for GroupNorm or ``"bn"`` for BatchNorm. GroupNorm
        is preferred under the subject-disjoint LOSO split because it has no
        running statistics — the held-out subject's voxel distribution is
        normalized using the current batch only, not training-subject means.
    gn_groups
        Target number of GroupNorm groups (clamped down to the largest divisor
        of ``c`` that does not exceed this value). Ignored when ``norm == "bn"``.
    use_se
        Enable squeeze-excitation channel gating inside every residual block.
    use_context
        Insert the ASPP-lite global-context module at the bottleneck. This is
        the main driver of long-range mask transfer; disable it for a lighter,
        purely-local ablation.
    context_dilations
        Dilation rates of the context module's parallel dilated branches.
        Ignored when ``use_context=False``.
    """

    def __init__(
        self,
        in_channels: int = 5,
        encoder_channels: Sequence[int] = (64, 128, 192, 256),
        num_classes: int = 1,
        norm: str = "gn",
        gn_groups: int = 8,
        use_se: bool = True,
        use_context: bool = True,
        context_dilations: Sequence[int] = (1, 2, 4),
    ):
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("encoder_channels must list at least 2 stages")
        ch: List[int] = list(encoder_channels)
        stem_ch = ch[0]

        # Stem: full-resolution 3×3 expansion before any downsampling.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_ch, kernel_size=3, padding=1, bias=False),
            _make_norm(stem_ch, norm, gn_groups),
            nn.ReLU(inplace=True),
        )

        # Encoder: stage 0 refines at full resolution; stages 1..N each halve
        # resolution (stride-2) and widen ch[i-1] -> ch[i]. We keep enc[0] as a
        # full-res refinement so the encoder skip set is symmetric with the
        # decoder.
        self.encoder = nn.ModuleList()
        self.encoder.append(_EncoderStage(stem_ch, ch[0], downsample=False,
                                          norm=norm, gn_groups=gn_groups,
                                          use_se=use_se))
        for i in range(1, len(ch)):
            self.encoder.append(_EncoderStage(ch[i - 1], ch[i], downsample=True,
                                              norm=norm, gn_groups=gn_groups,
                                              use_se=use_se))

        # Global-context bottleneck: operates on the deepest (ch[-1]) feature
        # map, where the spatial grid is smallest so dilated/global ops are
        # cheap. Identity passthrough keeps the forward path uniform when off.
        if use_context:
            self.context: nn.Module = _ContextModule(
                ch[-1], dilations=context_dilations,
                norm=norm, gn_groups=gn_groups,
            )
        else:
            self.context = nn.Identity()

        # Decoder: walk back up, consuming skips in reverse. Each stage takes
        # ch[i] channels in (bottleneck for the first stage, previous decoder's
        # output for the rest), concatenates ch[i-1] skip channels, and emits
        # ch[i-1] channels.
        self.decoder = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.decoder.append(
                _DecoderStage(c_in=ch[i], c_skip=ch[i - 1], c_out=ch[i - 1],
                              norm=norm, gn_groups=gn_groups, use_se=use_se)
            )

        self.head = nn.Conv2d(ch[0], num_classes, kernel_size=1)

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        skips: List[torch.Tensor] = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        # Inject global context into the bottleneck feature only.
        skips[-1] = self.context(skips[-1])
        return skips

    def _decode(self, skips: List[torch.Tensor]) -> torch.Tensor:
        x = skips[-1]
        for dec, skip in zip(self.decoder, reversed(skips[:-1])):
            x = dec(x, skip)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns full-resolution mask logits ``(B, num_classes, H, W)``."""
        return self._decode(self._encode(x))


def count_parameters(module: nn.Module) -> int:
    """Convenience helper for the smoke test."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test: shape contract + parameter budget at the config default width.
    model = EventUnet(in_channels=5, encoder_channels=(64, 128, 192, 256),
                      num_classes=1, norm="gn", gn_groups=8)
    model.eval()
    dummy = torch.randn(2, 5, 480, 640)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 1, 480, 640), out.shape
    print(f"output: {tuple(out.shape)}  params: {count_parameters(model):,}")
