"""Phase-B ANN baseline: depthwise-separable U-Net over an event voxel grid.

Input:  ``(B, C_in, H, W)`` where ``C_in == voxel_bins`` (default 5).
Output: ``(B, num_classes, H, W)`` segmentation logits (``num_classes == 1``
        for binary hand+arm masks).

The encoder follows the channel progression from the plan
(``[32, 64, 96, 128]`` by default). Each stage is two depthwise-separable
3×3 conv blocks; spatial resolution is halved by a stride-2 DW conv at the
start of every stage after the stem. The decoder mirrors the encoder with
bilinear upsampling and skip connections.

With the default ``encoder_channels=[32, 64, 96, 128]`` the model lands well
under the 1 M parameter budget — bump ``encoder_channels`` (e.g. ``[64, 96, 128,
160]``) to land near the 0.4–0.8 M target if mask quality is capacity-limited.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DWSepConv(nn.Module):
    """Depthwise 3×3 + BN + ReLU + pointwise 1×1 + BN + ReLU."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride,
                            padding=1, groups=c_in, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class _EncoderStage(nn.Module):
    """Optional stride-2 downsample, then a second DW-sep refinement."""

    def __init__(self, c_in: int, c_out: int, downsample: bool):
        super().__init__()
        first_stride = 2 if downsample else 1
        self.body = nn.Sequential(
            _DWSepConv(c_in, c_out, stride=first_stride),
            _DWSepConv(c_out, c_out, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _DecoderStage(nn.Module):
    """Bilinear upsample + skip concat + two DW-sep refinement convs."""

    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.body = nn.Sequential(
            _DWSepConv(c_in + c_skip, c_out, stride=1),
            _DWSepConv(c_out, c_out, stride=1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.body(torch.cat([x, skip], dim=1))


class EventUnet(nn.Module):
    """Tiny event-voxel U-Net.

    Parameters
    ----------
    in_channels
        Number of voxel time bins fed as input channels (default 5, matching
        ``HandEventDataset.voxel_bins``).
    encoder_channels
        Per-stage channel widths of the encoder, from the highest resolution
        (post-stem) to the bottleneck. Length controls depth; default
        ``[32, 64, 96, 128]`` follows the plan.
    num_classes
        Output channels of the segmentation head. ``1`` for binary hand+arm.
    """

    def __init__(
        self,
        in_channels: int = 5,
        encoder_channels: Sequence[int] = (32, 64, 96, 128),
        num_classes: int = 1,
    ):
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("encoder_channels must list at least 2 stages")
        ch: List[int] = list(encoder_channels)
        stem_ch = ch[0]

        # Stem: full-resolution 3x3 expansion before any downsampling.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )

        # Encoder: stage i goes ch[i-1] -> ch[i] with stride-2 downsample.
        # Stage 0 (the stem itself) operates at full resolution; stages 1..N are
        # downsamplers. We keep `enc[0]` as a single full-res refinement so the
        # encoder skip set is symmetric with the decoder.
        self.encoder = nn.ModuleList()
        self.encoder.append(_EncoderStage(stem_ch, ch[0], downsample=False))
        for i in range(1, len(ch)):
            self.encoder.append(_EncoderStage(ch[i - 1], ch[i], downsample=True))

        # Decoder: walk back up, consuming skips in reverse. Each stage takes
        # ch[i] channels in (bottleneck for the first stage, prev decoder's
        # output for the rest), concatenates ch[i-1] skip channels, and emits
        # ch[i-1] channels.
        self.decoder = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.decoder.append(
                _DecoderStage(c_in=ch[i], c_skip=ch[i - 1], c_out=ch[i - 1])
            )

        self.head = nn.Conv2d(ch[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips: List[torch.Tensor] = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        # skips: [enc0_out, enc1_out, ..., bottleneck]
        x = skips[-1]
        for dec, skip in zip(self.decoder, reversed(skips[:-1])):
            x = dec(x, skip)
        return self.head(x)


def count_parameters(module: nn.Module) -> int:
    """Convenience helper for the smoke test."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
