"""Temporal tracking student: EventUnet + a recurrent memory that carries the
object across frames, so the mask stops jittering between independent windows.

Why this exists
---------------
``EventUnet`` segments every frame independently. On event input that produces
a *flickering* mask: a forearm that stops moving emits no events for a window,
so the per-frame student loses it and the mask blinks on/off frame to frame.
SAM 2 avoids this by propagating a memory of the object through the video. This
model is the lightweight event-only analog of that idea:

1. **Recurrent bottleneck memory** (``_ConvGRUCell``): a ConvGRU hidden state at
   the U-Net bottleneck is carried from frame ``t-1`` to ``t``. It remembers
   *where the object was and what it looked like* even across a window with no
   events, which is the direct cause of the flicker. Depthwise-separable gates
   keep it cheap (~0.15 M params at the default width).
2. **Previous-mask feedback** (``use_prev_mask``): the previous frame's predicted
   probability map is concatenated to the input voxel as an extra channel — the
   same "condition on the last mask" trick SAM 2 uses. It biases the current
   prediction toward temporal continuity (and gives the network an explicit
   prior for momentarily-still regions). Detached, so it informs but does not
   backprop through time via the mask channel.

Everything else — residual DW-separable blocks, SE gating, the ASPP-lite
global-context bottleneck, the bilinear-skip decoder — is reused verbatim from
``EventUnet`` so the spatial backbone (and its ~0.65 M param budget) is unchanged.

Batch / shape contract
----------------------
``forward`` accepts either:

* a **clip** ``(B, T, C, H, W)`` → returns ``(B, T, num_classes, H, W)``; the
  memory + prev-mask are carried internally across the ``T`` frames (truncated
  BPTT through the GRU hidden state), starting from a zero state. This is what
  ``ModelInterface._tracking_step`` calls during training/validation.
* a single **frame** ``(B, C, H, W)`` → returns ``(B, num_classes, H, W)``,
  evaluated from a fresh (zero) state. Lets per-frame call sites (e.g. an
  ablation, or a sanity check) use the model unchanged.

For true online tracking over an arbitrarily long stream use ``step``:

    state = None
    for frame in stream:                      # frame: (B, C, H, W)
        logits, state = model.step(frame, state)

``C == in_channels == voxel_bins``. The mask feedback / hidden state are kept at
full and bottleneck resolution respectively, so there is no resolution mismatch.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from model.event_unet import (
    _ContextModule,
    _DecoderStage,
    _EncoderStage,
    _make_norm,
    count_parameters,
)


def _sep_conv(c_in: int, c_out: int) -> nn.Sequential:
    """Depthwise 3×3 + pointwise 1×1 separable conv producing raw gate logits.

    No norm/activation: the caller applies the gate non-linearity (sigmoid/tanh).
    Cheap by construction — this is what keeps the recurrent memory inside the
    model's parameter budget.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in, bias=False),
        nn.Conv2d(c_in, c_out, kernel_size=1, bias=True),
    )


class _ConvGRUCell(nn.Module):
    """Convolutional GRU with depthwise-separable gates.

    Standard GRU recurrence applied per spatial location:

        z = σ(W_z · [x, h])           update gate
        r = σ(W_r · [x, h])           reset gate
        n = tanh(W_n · [x, r ⊙ h])    candidate
        h' = (1 − z) ⊙ n + z ⊙ h

    Operates on the bottleneck feature map (smallest spatial grid), so the 3×3
    gate convs are cheap. ``z``/``r`` share one conv (output split in two).
    """

    def __init__(self, channels: int):
        super().__init__()
        c = channels
        self.conv_zr = _sep_conv(2 * c, 2 * c)
        self.conv_n = _sep_conv(2 * c, c)

    def forward(self, x: torch.Tensor,
                h: Optional[torch.Tensor]) -> torch.Tensor:
        if h is None:
            h = torch.zeros_like(x)
        zr = self.conv_zr(torch.cat([x, h], dim=1))
        z, r = torch.chunk(zr, 2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        n = torch.tanh(self.conv_n(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * n + z * h


# State carried between frames: (gru_hidden, prev_mask_prob). Either field may be
# None at the start of a clip/stream.
TrackState = Dict[str, Optional[torch.Tensor]]


class EventTrackUnet(nn.Module):
    """Recurrent EventUnet that tracks the mask across frames (anti-flicker).

    Parameters mirror :class:`model.event_unet.EventUnet`, plus:

    use_prev_mask
        Concatenate the previous frame's predicted probability map to the input
        voxel as an extra channel (the SAM2-style "condition on the last mask"
        signal). When ``True`` the stem sees ``in_channels + 1`` channels.
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
        use_prev_mask: bool = True,
    ):
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("encoder_channels must list at least 2 stages")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.use_prev_mask = bool(use_prev_mask)

        ch: List[int] = list(encoder_channels)
        stem_ch = ch[0]
        stem_in = self.in_channels + (1 if self.use_prev_mask else 0)

        # Stem: full-resolution 3×3 expansion before any downsampling. The +1
        # input channel (when use_prev_mask) carries the previous mask prob.
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, stem_ch, kernel_size=3, padding=1, bias=False),
            _make_norm(stem_ch, norm, gn_groups),
            nn.ReLU(inplace=True),
        )

        # Encoder: stage 0 refines at full res; stages 1..N each stride-2 down.
        self.encoder = nn.ModuleList()
        self.encoder.append(_EncoderStage(stem_ch, ch[0], downsample=False,
                                          norm=norm, gn_groups=gn_groups,
                                          use_se=use_se))
        for i in range(1, len(ch)):
            self.encoder.append(_EncoderStage(ch[i - 1], ch[i], downsample=True,
                                              norm=norm, gn_groups=gn_groups,
                                              use_se=use_se))

        # Recurrent memory at the bottleneck — applied to the deepest feature map
        # before the global-context block, so context reasons over the
        # memory-fused feature each frame.
        self.memory = _ConvGRUCell(ch[-1])

        if use_context:
            self.context: nn.Module = _ContextModule(
                ch[-1], dilations=context_dilations,
                norm=norm, gn_groups=gn_groups,
            )
        else:
            self.context = nn.Identity()

        self.decoder = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.decoder.append(
                _DecoderStage(c_in=ch[i], c_skip=ch[i - 1], c_out=ch[i - 1],
                              norm=norm, gn_groups=gn_groups, use_se=use_se)
            )

        self.head = nn.Conv2d(ch[0], num_classes, kernel_size=1)

    # ------------------------------------------------------------------ pieces

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encoder skips with NO context injected (context happens post-memory)."""
        x = self.stem(x)
        skips: List[torch.Tensor] = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        return skips

    def _decode(self, skips: List[torch.Tensor]) -> torch.Tensor:
        x = skips[-1]
        for dec, skip in zip(self.decoder, reversed(skips[:-1])):
            x = dec(x, skip)
        return self.head(x)

    # ------------------------------------------------------------------ stepping

    def step(self, frame: torch.Tensor,
             state: Optional[TrackState]) -> Tuple[torch.Tensor, TrackState]:
        """One tracking step.

        frame : ``(B, C, H, W)`` voxel for the current frame.
        state : previous ``{"h": hidden, "prev_mask": prob}`` or ``None`` to
                start a fresh track (zero memory, zero prev-mask).

        Returns ``(logits (B, num_classes, H, W), new_state)``. The returned
        ``prev_mask`` is the detached current-class probability map so the mask
        feedback informs the next frame without backpropagating through time via
        that channel; the GRU hidden state stays attached so gradients flow
        across the clip (truncated BPTT over the clip length).
        """
        b, _, h, w = frame.shape
        prev_h = state["h"] if state is not None else None
        prev_mask = state["prev_mask"] if state is not None else None

        if self.use_prev_mask:
            if prev_mask is None:
                prev_mask = frame.new_zeros(b, 1, h, w)
            inp = torch.cat([frame, prev_mask], dim=1)
        else:
            inp = frame

        skips = self._encode(inp)
        skips[-1] = self.memory(skips[-1], prev_h)   # bottleneck memory update
        new_h = skips[-1]
        skips[-1] = self.context(skips[-1])
        logits = self._decode(skips)

        # Feedback = detached probability of the (first) foreground class.
        new_prev = torch.sigmoid(logits[:, :1]).detach()
        return logits, {"h": new_h, "prev_mask": new_prev}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clip ``(B,T,C,H,W)`` → ``(B,T,num_classes,H,W)``; or a single frame
        ``(B,C,H,W)`` → ``(B,num_classes,H,W)`` from a fresh state."""
        if x.dim() == 4:
            logits, _ = self.step(x, None)
            return logits
        if x.dim() != 5:
            raise ValueError(
                f"expected (B,T,C,H,W) clip or (B,C,H,W) frame, got {tuple(x.shape)}"
            )
        B, T = x.shape[:2]
        state: Optional[TrackState] = None
        outs: List[torch.Tensor] = []
        for t in range(T):
            logits, state = self.step(x[:, t], state)
            outs.append(logits)
        return torch.stack(outs, dim=1)


if __name__ == "__main__":
    # Smoke test: clip + single-frame + streaming shape contracts and param count.
    model = EventTrackUnet(in_channels=5, encoder_channels=(48, 96, 128, 160),
                           num_classes=1, norm="gn", gn_groups=8)
    model.eval()

    clip = torch.randn(2, 4, 5, 120, 160)
    with torch.no_grad():
        out = model(clip)
    assert out.shape == (2, 4, 1, 120, 160), out.shape

    frame = torch.randn(2, 5, 120, 160)
    with torch.no_grad():
        out1 = model(frame)
    assert out1.shape == (2, 1, 120, 160), out1.shape

    # Streaming equals batched-clip frame 0 from a fresh state.
    with torch.no_grad():
        l0, st = model.step(clip[:, 0], None)
    assert torch.allclose(l0, out[:, 0], atol=1e-5)

    print(f"clip {tuple(out.shape)} | frame {tuple(out1.shape)} | "
          f"params: {count_parameters(model):,}")
