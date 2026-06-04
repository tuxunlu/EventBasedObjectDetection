"""Real-time event-domain segmenter distilled from SAM via multi-level
feature alignment.

Designed as a *pre-filter* in front of heavier downstream perception:
the head outputs a coarse attention map (default stride 4, configurable)
that gates which event tokens reach the downstream model. Latency, not
mask sharpness, is the priority.

Architecture
------------
Encoder: 4-stage depthwise-separable CNN with GroupNorm. Channels widen
geometrically (`stage_channels = (stem, s1, s2, s3, s4)`); each post-stem
stage halves spatial resolution. Three feature *taps* are exposed for
distillation, at the output of stages 2, 3, and 4 (strides 4, 8, 16
respectively).

Head: lightweight FPN fusion — top-down lateral 1×1 convs add upsampled
deeper features into shallower features, ending with one DW-separable
refinement and a 1×1 mask classifier. Output is at `output_stride` (default 4).

Distillation heads (training-only): for each tap, a 1×1 conv projects the
student feature map to the SAM teacher's channel dimension (768 for ViT-B
intermediate layers, 256 for the post-neck embedding). These projections
are stored in `self.distill_projections` and can be discarded for inference
by calling `model.strip_distillation()`.

Loss composition (computed in `loss/feature_distillation.py`):

    L =   λ_mask · BCE(mask_logits, teacher_mask)
        + λ_mask · Dice(mask_logits, teacher_mask)
        + Σ_level λ_align[level] · ( 1 - cosine( norm(W·s_level), norm(t_level) ) )

where `W·s_level` is the projected student feature at that level (spatially
bilinearly resized to the teacher token grid), `t_level` is the SAM feature
at the paired layer, and cosine is averaged over tokens.

Parameter budget
----------------
With the default `stage_channels=(16, 24, 32, 48, 64)`:
  * inference model: ~25K params
  * +distillation projection heads (training-only): ~78K extra

Throughput target: <1ms/frame on a desktop GPU at 480×640, single-digit ms
on Jetson Orin Nano.

Output contract
---------------
`forward(x)` returns mask logits ``(B, num_classes, H_out, W_out)``.
`forward_with_features(x)` returns ``(mask_logits, feature_dict)`` where
``feature_dict`` has keys ``"low"``, ``"mid"``, ``"high"`` mapping to projected
student features ready for distillation (shape ``(B, C_proj_level, H_s, W_s)``
at the student's native stride — the loss handles resizing to the teacher grid).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Norm factory and depthwise-separable conv (shared shape with event_unet's
# building blocks but kept local so this file is self-contained and the two
# models can evolve independently).
# ---------------------------------------------------------------------------

def _make_norm(c: int, kind: str, gn_groups: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(c)
    if kind == "gn":
        g = min(gn_groups, c)
        while g > 1 and c % g != 0:
            g -= 1
        return nn.GroupNorm(num_groups=g, num_channels=c)
    raise ValueError(f"unknown norm kind: {kind!r} (expected 'bn' or 'gn')")


class _DWSep(nn.Module):
    """Depthwise 3×3 + norm + ReLU + pointwise 1×1 + norm + ReLU."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1,
                 norm: str = "gn", gn_groups: int = 4):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride,
                            padding=1, groups=c_in, bias=False)
        self.n1 = _make_norm(c_in, norm, gn_groups)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.n2 = _make_norm(c_out, norm, gn_groups)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.n1(self.dw(x)))
        x = self.act(self.n2(self.pw(x)))
        return x


class _EncStage(nn.Module):
    """One encoder stage: stride-2 DW-sep downsampler followed by a refinement DW-sep."""

    def __init__(self, c_in: int, c_out: int, norm: str, gn_groups: int):
        super().__init__()
        self.body = nn.Sequential(
            _DWSep(c_in, c_out, stride=2, norm=norm, gn_groups=gn_groups),
            _DWSep(c_out, c_out, stride=1, norm=norm, gn_groups=gn_groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Tiny FPN-style head: top-down fusion ending at `target_stride`.
# Cheaper than a U-Net decoder because there's no skip-concat-then-conv chain.
# ---------------------------------------------------------------------------

class _TinyFpnHead(nn.Module):
    """Top-down lateral fusion: c4(stride16) + c3(stride8) + c2(stride4) → mask at c2's stride.

    Each tap is reduced to ``mid_ch`` via a 1×1 lateral conv; deeper taps are
    upsampled bilinearly to the next-shallower tap's spatial size and summed.
    A single DW-sep refinement on the fused map sharpens, then a 1×1 classifier
    produces ``num_classes`` channels.
    """

    def __init__(self, c2: int, c3: int, c4: int, mid_ch: int, num_classes: int,
                 norm: str, gn_groups: int):
        super().__init__()
        self.lat2 = nn.Conv2d(c2, mid_ch, kernel_size=1, bias=False)
        self.lat3 = nn.Conv2d(c3, mid_ch, kernel_size=1, bias=False)
        self.lat4 = nn.Conv2d(c4, mid_ch, kernel_size=1, bias=False)
        self.refine = _DWSep(mid_ch, mid_ch, norm=norm, gn_groups=gn_groups)
        self.classifier = nn.Conv2d(mid_ch, num_classes, kernel_size=1)

    def forward(self, s2: torch.Tensor, s3: torch.Tensor, s4: torch.Tensor) -> torch.Tensor:
        p4 = self.lat4(s4)
        p3 = self.lat3(s3) + F.interpolate(p4, size=s3.shape[-2:],
                                            mode="bilinear", align_corners=False)
        p2 = self.lat2(s2) + F.interpolate(p3, size=s2.shape[-2:],
                                            mode="bilinear", align_corners=False)
        return self.classifier(self.refine(p2))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class EventTinySeg(nn.Module):
    """Real-time event-domain pre-filter, distilled from SAM features.

    Parameters
    ----------
    in_channels
        Number of voxel time bins (== ``HandEventDataset.voxel_bins``).
    stage_channels
        5-tuple ``(stem, s1, s2, s3, s4)`` of channel widths. Default is the
        smallest configuration that comfortably fits the ~25K inference budget
        while leaving headroom for the SAM projection heads at training time.
    num_classes
        Output channels of the mask head. ``1`` for binary actor mask.
    output_stride
        Spatial stride of the predicted mask. ``4`` (default) gives 120×160 for
        a 480×640 input — coarse enough to be cheap, fine enough to crop ROIs
        for downstream models. ``2`` doubles cost; ``8`` halves it (and stops
        before the stride-4 lateral fusion).
    head_mid_channels
        Internal channel width of the FPN lateral path. Smaller = faster.
    norm, gn_groups
        See ``EventUnet`` — GroupNorm preferred under LOSO to avoid running-stats
        drift across subjects.
    align_dims
        Optional dict ``{"low": dim, "mid": dim, "high": dim}`` specifying the
        teacher feature dimensions to project student taps into for distillation.
        Default ``{"low": 256, "mid": 256, "high": 256}`` matches SAM 2's FPN
        neck, which unifies all three multi-scale Hiera outputs to ``d_model``
        (256 in stock configs). For original SAM ViT-B alignment instead, use
        ``{"low": 768, "mid": 768, "high": 256}``. Set to ``None`` to disable
        the distillation heads entirely (inference-only build).
    """

    def __init__(
        self,
        in_channels: int = 5,
        stage_channels: Sequence[int] = (16, 24, 32, 48, 64),
        num_classes: int = 1,
        output_stride: int = 4,
        head_mid_channels: int = 32,
        norm: str = "gn",
        gn_groups: int = 4,
        align_dims: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        if len(stage_channels) != 5:
            raise ValueError("stage_channels must be (stem, s1, s2, s3, s4)")
        if output_stride not in (2, 4, 8):
            raise ValueError(f"output_stride must be one of (2, 4, 8), got {output_stride}")
        c0, c1, c2, c3, c4 = stage_channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.output_stride = output_stride

        # ---- encoder ----
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c0, kernel_size=3, padding=1, bias=False),
            _make_norm(c0, norm, gn_groups),
            nn.ReLU(inplace=True),
        )
        self.enc1 = _EncStage(c0, c1, norm=norm, gn_groups=gn_groups)  # stride 2
        self.enc2 = _EncStage(c1, c2, norm=norm, gn_groups=gn_groups)  # stride 4   ← tap "low"
        self.enc3 = _EncStage(c2, c3, norm=norm, gn_groups=gn_groups)  # stride 8   ← tap "mid"
        self.enc4 = _EncStage(c3, c4, norm=norm, gn_groups=gn_groups)  # stride 16  ← tap "high"

        # ---- FPN-style head ----
        # Always fuse all three taps; if output_stride > 4, we additionally
        # downsample the final mask logits with adaptive_avg_pool at forward time.
        self.head = _TinyFpnHead(
            c2=c2, c3=c3, c4=c4,
            mid_ch=head_mid_channels,
            num_classes=num_classes,
            norm=norm, gn_groups=gn_groups,
        )

        # ---- distillation projections (training-only) ----
        if align_dims is None:
            # SAM 2 FPN: all three FPN levels are 256-d (d_model).
            align_dims = {"low": 256, "mid": 256, "high": 256}
        self.align_dims = dict(align_dims)
        # 1×1 conv per tap, no norm/activation — pure linear remap to teacher dim.
        # Stored in a ModuleDict so .strip_distillation() can drop them cleanly.
        self.distill_projections = nn.ModuleDict({
            "low":  nn.Conv2d(c2, align_dims["low"],  kernel_size=1),
            "mid":  nn.Conv2d(c3, align_dims["mid"],  kernel_size=1),
            "high": nn.Conv2d(c4, align_dims["high"], kernel_size=1),
        })

    # ----------------------------------------------------------------- forward

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        return s2, s3, s4

    def _mask(self, s2: torch.Tensor, s3: torch.Tensor, s4: torch.Tensor) -> torch.Tensor:
        mask = self.head(s2, s3, s4)  # at stride 4
        if self.output_stride == 2:
            mask = F.interpolate(mask, scale_factor=2.0, mode="bilinear", align_corners=False)
        elif self.output_stride == 8:
            mask = F.avg_pool2d(mask, kernel_size=2, stride=2)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path: returns mask logits only (no distillation features)."""
        s2, s3, s4 = self._encode(x)
        return self._mask(s2, s3, s4)

    def forward_with_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Training path: returns ``(mask_logits, feature_dict)``.

        ``feature_dict`` keys correspond to ``align_dims``. Each value is the
        projected student feature at its native stride; the distillation loss
        is responsible for spatially resizing to the teacher grid.
        """
        s2, s3, s4 = self._encode(x)
        mask = self._mask(s2, s3, s4)
        feats = {
            "low":  self.distill_projections["low"](s2),
            "mid":  self.distill_projections["mid"](s3),
            "high": self.distill_projections["high"](s4),
        }
        return mask, feats

    # ------------------------------------------------------------------ tools

    def strip_distillation(self) -> "EventTinySeg":
        """Delete the training-only projection heads. Returns ``self`` for chaining.

        Call before exporting the inference model so the parameter count and
        compute graph reflect what actually runs at deploy time.
        """
        if hasattr(self, "distill_projections"):
            del self.distill_projections
            self.distill_projections = None  # type: ignore[assignment]
        return self

    def count_parameters(self, inference_only: bool = False) -> int:
        """Sum of trainable parameter counts.

        ``inference_only=True`` excludes the distillation projection heads so
        you can see the deploy-time budget.
        """
        excluded: List[int] = []
        if inference_only and self.distill_projections is not None:
            excluded = [id(p) for p in self.distill_projections.parameters()]
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad and id(p) not in excluded)
