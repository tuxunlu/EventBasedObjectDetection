"""Frozen SAM 2 image-encoder wrapper for online feature distillation.

Reuses the same checkpoint and Hydra config that
``data/sam2_pseudo_labels.py`` uses for offline mask generation, but exposes
only the image encoder — the prompt encoder, mask decoder, and memory
attention are dropped since the student is supervised by the cached masks
(produced by the full GroundingDINO + SAM 2 pipeline) plus the encoder's
multi-scale features.

What the encoder returns
------------------------
SAM 2's image encoder is Hiera + an FPN neck. For a 1024×1024 input, the
neck produces three FPN levels at strides ``(4, 8, 16)``, all unified to
``d_model`` channels (256 by default — confirmed at ``__init__`` time):

    backbone_fpn[0]:  (B, 256, 256, 256)   stride 4   "low"   ← shallow texture
    backbone_fpn[1]:  (B, 256, 128, 128)   stride 8   "mid"   ← mid-level grouping
    backbone_fpn[2]:  (B, 256,  64,  64)   stride 16  "high"  ← top semantic, what
                                                              the mask decoder sees

The mapping to the student's three taps is geometric, not just nominal:
each student tap also sits at strides 4/8/16 of its native input resolution.
``FeatureDistillationLoss`` does the bilinear resize so the two grids align
before cosine alignment.

Memory
------
Hiera-L image encoder is ~200 M params. With a frozen teacher and
``torch.no_grad()`` forward, peak activation memory at 1024×1024 batch 4 is
typically 4–6 GB on a modern GPU — well within budget for distillation
when sharing the GPU with a 25 K-param student.

If memory is tight, lower ``input_size`` to 512 — FPN levels then come out
at strides (4, 8, 16) of 512, i.e. (128, 64, 32). The student loss still
works (the alignment is geometric, not absolute-resolution).

Use
---

    teacher = Sam2ImageEncoderTeacher(
        checkpoint_path="sam2_hiera_large.pt",
        config_name="sam2_hiera_l.yaml",
    )
    feats = teacher(rgb_batch_BCHW_0_to_1)
    # feats == {"low":  (B, 256, H/4, W/4),
    #           "mid":  (B, 256, H/8, W/8),
    #           "high": (B, 256, H/16, W/16)}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sam2ImageEncoderTeacher(nn.Module):
    """Frozen SAM 2 image encoder; outputs the FPN's 3 multi-scale features."""

    DEFAULT_INPUT_SIZE = 1024  # SAM 2's pre-trained input geometry.

    # SAM 2's image encoder expects ImageNet-normalized RGB tensors.
    PIXEL_MEAN = (0.485, 0.456, 0.406)
    PIXEL_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        checkpoint_path: str,
        config_name: str,
        input_size: int = 1024,
        expected_feature_keys=("low", "mid", "high"),
    ):
        super().__init__()
        from sam2.build_sam import build_sam2

        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"SAM 2 checkpoint not found: {ckpt}")

        # Build the full SAM 2 model on CPU first, then strip everything that's
        # not the image encoder. Avoids holding the prompt encoder / mask
        # decoder / memory module on GPU when we only need the trunk + neck.
        full = build_sam2(config_name, str(ckpt), device="cpu")
        self.image_encoder = full.image_encoder
        del full

        # Freeze. No autograd state, no parameter updates.
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        self.input_size = int(input_size)
        self.expected_feature_keys = tuple(expected_feature_keys)

        # Normalization buffers, registered so .to(device) carries them along.
        self.register_buffer(
            "pixel_mean", torch.tensor(self.PIXEL_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std", torch.tensor(self.PIXEL_STD).view(1, 3, 1, 1),
        )

        # Lock to eval mode and surface the FPN channel dim so callers can
        # configure EventTinySeg.align_dims to match.
        self.eval()
        with torch.no_grad():
            probe = torch.zeros(1, 3, self.input_size, self.input_size)
            out = self.image_encoder(probe)
            fpn = self._extract_fpn_list(out)
        self.feature_channels = [int(f.shape[1]) for f in fpn]
        if len(fpn) < len(self.expected_feature_keys):
            raise RuntimeError(
                f"SAM 2 FPN returned {len(fpn)} levels; need at least "
                f"{len(self.expected_feature_keys)} for "
                f"{self.expected_feature_keys}"
            )

    # ------------------------------------------------------------------ public

    def train(self, mode: bool = True):
        """Keep the frozen teacher in eval mode regardless of caller intent.

        This stops a downstream ``.train()`` call (e.g. from Lightning's
        epoch hook) from re-enabling BN/Dropout inside the teacher.
        """
        return super().train(False)

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """RGB ``(B, 3, H, W)`` in ``[0, 1]`` → dict of FPN feature maps.

        The input is resized to ``self.input_size`` and ImageNet-normalized
        before being passed to SAM 2's image encoder.
        """
        if rgb.shape[1] != 3:
            raise ValueError(f"expected 3-channel RGB, got shape {tuple(rgb.shape)}")
        if rgb.shape[-2:] != (self.input_size, self.input_size):
            rgb = F.interpolate(
                rgb, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )
        rgb = (rgb - self.pixel_mean) / self.pixel_std

        out = self.image_encoder(rgb)
        fpn = self._extract_fpn_list(out)
        # FPN ordering in SAM 2's image_encoder is finest-first
        # ((B, C, H/4, W/4), (B, C, H/8, W/8), (B, C, H/16, W/16)).
        # Confirm at runtime so a future SAM 2 release that flips ordering
        # produces a clear error rather than silently corrupted alignment.
        strides = [self.input_size // f.shape[-1] for f in fpn]
        if not (strides == sorted(strides)):
            raise RuntimeError(
                f"SAM 2 FPN level ordering is not ascending by stride "
                f"(got strides {strides}); the wrapper assumes "
                f"[stride4, stride8, stride16]. Check your SAM 2 version."
            )
        return dict(zip(self.expected_feature_keys, fpn))

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _extract_fpn_list(image_encoder_out):
        """Be robust to small variations in SAM 2's image-encoder return type."""
        if isinstance(image_encoder_out, dict) and "backbone_fpn" in image_encoder_out:
            return list(image_encoder_out["backbone_fpn"])
        if isinstance(image_encoder_out, (list, tuple)):
            return list(image_encoder_out)
        raise RuntimeError(
            f"unrecognised SAM 2 image_encoder output type: "
            f"{type(image_encoder_out).__name__}"
        )
