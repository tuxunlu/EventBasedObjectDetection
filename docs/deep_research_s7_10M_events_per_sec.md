# EventStreamSegS7 → 10 M events/sec: speed analysis + accuracy-safe per-event improvements

**Date:** 2026-07-13 · **Run analyzed:** `lightning_logs/20260712-21-29-12-hand_event_seg_stream_s7_ee3d_w_fast`
(tron46, 4× RTX A5000 24 GB) · **Method:** read-only grounding of the live run + a 103-agent
deep-research sweep (21 sources, 97 claims, 25 adversarially verified → 22 confirmed / 3 refuted).

---

## 0. Bottom line (read this first)

**The 10 M events/sec bar is almost certainly NOT reachable at *full per-event resolution* on a
single A5000/L40S.** No published per-event GPU model reaches it: the strongest purpose-built
asynchronous per-event model, **EVA (RWKV-6), measures only ~0.611 M ev/s on an RTX 3090** — 16×
short — and only approaches ~9.78 M ev/s by applying **8× event downsampling that merges events and
drops per-event resolution** (arXiv 2505.11165, Tables 4 & 11). The 10 M bar is only *demonstrably*
met on **FPGA** (an event graph-conv network at **13.3 M ev/s, 4.47 ms deterministic latency**,
arXiv 2406.07318) — for classification, not per-event segmentation.

**Where we actually are:** this model already runs *faster* than EVA — **~1.76 M ev/s measured**
(L40S, batch 1, 30 k events = 17 ms forward) — because our "per-event" path is a **windowed parallel
scan**, not a one-event-at-a-time recurrence. So we are already in the batched-window regime where
10 M is *most* plausible, and the gap is **~6× (L40S) / ~9× (A5000)**, not 16×.

**Realistic verdict:** stacking the accuracy-safe GPU levers below (SSD tensor-core kernel + fp16 +
CUDA graphs, optionally real-diagonal-with-RoPE) plausibly reaches **~3–8 M ev/s** in a streaming
(batch-1) regime. **Clearing 10 M reliably needs one of:** (a) **event batching** — process several
windows/streams at once (throughput mode, adds windowing latency); (b) **event-pooling with a learned
per-event scatter head** (mild per-event-fidelity risk); or (c) **FPGA/neuromorphic** hardware.
This is a *throughput-vs-latency* choice, and the target should be split accordingly.

---

## 1. Grounded current state (measured on this run)

| Quantity | Value | Source |
|---|---|---|
| Training rate | **3.46 it/s** steady | live log, Epoch 2/49 |
| Training throughput / GPU | **~235 k ev/s** | 3.46 × 4 (batch) × 17 k ev/frame |
| Training throughput / 4-GPU system | **~0.94 M ev/s** | ×4 GPUs (fwd+bwd, ckpt ON, dual OFF) |
| **Inference (measured anchor)** | **~1.76 M ev/s** | 30 k / 17 ms forward, batch 1, L40S — `docs/deep_research_s7_faster_capable.md` |
| Inference (A5000 fast-config est.) | ~1.0–1.2 M ev/s | L40S anchor × ~0.59 hardware ratio |
| **Gap to 10 M ev/s (inference)** | **~6× (L40S) / ~9× (A5000)** | — |

**The bottleneck is NOT matmul.** Profiler op table (`tools/profile_s7_step.py`, 30 k/B4/L40S):
**~36 % complex-multiply, ~27 % scan `where`/`cat`/`copy`** (Hillis-Steele doubling gathers),
**matmul only 3.5 %**. Root cause = the pure-PyTorch segmented Hillis-Steele scan issues *many small
sequential kernel launches* × layers, plus on-the-fly `argsort`s and gradient-checkpoint recompute.
Recorded ablation ladder (fwd+bwd ms/step): **full 770 → drop dual-scan 273 (2.8×, already applied) →
real-diagonal 277 (2.8×, accuracy-UNVALIDATED) → both 107 (7.2×)**; `use_checkpoint:False` frees
169 ms (~1.3×, but OOMs the 24 GB A5000, so it's kept ON here).

Already-available but unbuilt levers: `mamba_chunk_scan_combined` (SSD kernel) **installed, not
wired**; `event_pool` is a `NotImplementedError` stub; `torch.compile` **fails** on the complex path
(`aten::_conj` + data-dependent scan length).

---

## 2. Angle 1 — Fastest GPU kernels for a per-event selective SSM

**Finding (confirmed 3-0, 6 merged claims).** The single biggest accuracy-safe lever is to **retire
the pure-PyTorch Hillis-Steele scan and route the recurrence through Mamba-2's SSD kernel**
(`mamba_chunk_scan_combined`, already installed). SSD computes steps 1/2/4 as **parallel tensor-core
matmuls** with only *one* short inter-chunk scan step, and is "significantly faster than Mamba-1's
`selective_scan_cuda` for the same state dimension."
Sources: pytorch.org/blog/accelerating-mamba2-with-kernel-fusion, tridao.me Mamba-2 parts 3–4,
state-spaces/mamba `ssd_combined.py`.

> **Read the numbers carefully.** *Kernel fusion* of the five SSD sub-kernels is only **1.50–2.51× on
> the SSD portion, ~8–13 % end-to-end** (up to ~20 % at 1 K context), measured on **A100/H100**. That
> 8–13 % is *fused SSD vs unfused SSD* — **not** our situation. **We are replacing a pure-PyTorch
> log-depth scan** (project-measured **2.55× slower** than a kernel baseline; §1 shows it dominates
> ~63 % of step time) with the tensor-core matmul path. The report's own evidence: *"the real win is
> retiring the pure-PyTorch scan for the tensor-core matmul path rather than the fusion increment."*
> Expect this to recover most of the 2.55× and more — a **~2–2.5× inference win**, the largest single
> GPU lever we have.

**Varlen packing works (confirmed 3-0).** `mamba_chunk_scan_combined` accepts `cu_seqlens` with
`return_varlen_states=True` (**requires `batch == 1`**), returning per-sequence final states — you pack
one long concatenated event stream with per-sample offsets. This is a natural fit for the existing
`batch_idx` segment resets. Source: verified verbatim against installed `mamba_ssm 2.2.4`.

**The catch — SSD is real-diagonal only (confirmed 3-0, and this is the accuracy-risky part).** The
SSD/FLA kernels use `tl.float32` throughout, no complex/conjugate path; Tri Dao: *"everything in
Mamba-2 is taken over the reals."* Dropping our complex diagonal is genuinely risky: a **NeurIPS-2024
lower bound (arXiv 2410.14067, Thm 2)** shows real-diagonal SSMs need *exponentially* large dimension
to approximate certain complex-representable LTI maps, and **real-only / no-RoPE variants score no
better than random on state-tracking** (Mamba-2 = **0.9 % on parity vs 100 %** for the complex
version). *(Caveat: the theorem covers non-selective diagonal SSMs, and the stronger "complex strictly
more expressive per dimension" claim was **REFUTED 1-2** — so the risk is real but of **contested
magnitude**.)*

**The low-cost hedge — Mamba-3's "RoPE trick" (confirmed 3-0).** A **data-dependent rotary embedding
applied to B, C** implements complex/rotational state transitions **"with minimal computational
overhead compared to real-valued SSMs"** and *restores* parity/modular-arithmetic state-tracking
(arXiv 2603.15569, ICLR 2026). This lets you keep the real-valued tensor-core kernel path **and** the
oscillatory (Koopman-frequency) expressivity our complex SSM currently provides.

**Refuted lever — do NOT pursue Blelloch (refuted 0-3).** Switching Hillis-Steele → Blelloch /
work-efficient scan is **not** a speed fix: same **O(log T) span**, and the literature is explicit
that the memory-I/O of materializing the 2-D hidden state every step (not the addition count) is what
kills real-world scan speed (GLA, arXiv 2312.06635). The fix is the matmul kernel, not a better scan
schedule.

**FLA is an alternative kernel family (confirmed 3-0), but not installed.** `flash-linear-attention`
ships Triton kernels for Mamba/Mamba2/**Mamba3** + 20+ variants (GLA, RWKV6/7, DeltaNet…), supports
varlen, and its chunkwise kernel beats FlashAttention-2 even at ~1 K length. Still real-diagonal, so
the same complex→RoPE hedge applies. Installing it opens the Mamba-3 kernel path directly.

---

## 3. Angle 2 — Event-pooling / hierarchical downsampling

**Finding (confirmed 3-0, 4 merged claims): event-pooling gives the biggest L→L/p speedups but breaks
strict per-event output.** **Event-SSM** (NeurIPS 2024, arXiv 2404.18508) *"integrates M inputs into
the state-space but only forwards a subsampled sequence of length M/p"* per pooling block — evaluated
**only on classification**, no per-event decoder. **EventMamba** (arXiv 2405.06116) is **point-based**:
FPS+KNN-normalize each window to a fixed 1024/2048 points, hierarchically downsample 512→256→128
centroids, emit **one label per window**. Neither yields one label per raw event.

**Implication for us:** event-pooling is viable **only** if paired with a **learned scatter/upsample
head** that re-expands pooled features back to every original event (our head + loss are row-aligned
to `batch.labels`). That scatter is unbuilt and its accuracy cost is **unmeasured** — this is the
`event_pool` stub's real design challenge (open question, §7). Treat as **risky to per-event fidelity**
and lower priority than the SSD kernel.

---

## 4. Angle 3 — Is 10 M ev/s even achievable?

**Finding (confirmed 3-0).** Bounded from both sides:
- **GPU per-event ceiling today:** EVA (RWKV-6) **0.611 M ev/s** event-by-event on RTX 3090; ~9.78 M
  *only* with 8× event-merging downsampling (arXiv 2505.11165).
- **10 M is physically reachable — on FPGA:** event GCN **13.3 M ev/s @ 75 ns/event, 4.47 ms latency**
  (arXiv 2406.07318); an embedded GPU (Jetson Orin NX) needed 71–107 ms/forward on the same task.

So 10 M at full per-event resolution on a single A5000/L40S is **unproven** and, per the evidence,
likely requires **event batching** (throughput mode) or non-GPU hardware. Systems levers that *do*
stack on GPU (with the caveat that fusion numbers are A100/H100, not our cards): **fp16/bf16 states +
chunk size 128** (required for the SSD speedup; fp32 states give less), **CUDA graphs** (directly
attacks our "many small kernel launches" bottleneck — the single most launch-bound part of the
profile), **torch.compile** (needs the complex path removed first), reduced state dim, and INT8.

---

## 5. Angle 4 — Low-cost per-event *accuracy* improvements

**Finding (confirmed 3-0, medium confidence): SSM decode is memory-bound (~2.5 ops/byte), so extra
state-update compute is nearly free.** **Mamba-3's MIMO (matrix) state update** *"increases decoding
FLOPs by up to 4× at fixed state size while maintaining similar wall-clock decode latency, and
simultaneously improving perplexity and downstream performance"* (arXiv 2603.15569). Combined with the
RoPE complex transitions (§2), this is the main **low-compute, per-event-preserving accuracy lever**
the sweep surfaced. **Caveat (why medium, not high):** measured on **language modeling**, not
event-segmentation — transfer to the hand/body mask task is unverified; latency parity was at batch
128 on H100, and *prefill* (our regime) carries moderate overhead.

**Honest gap.** The question asked about **boundary/topology losses (clDice)** and better tokenization
(normal-flow, time-surfaces, learned embeddings). **No surviving claim measured clDice or any low-cost
accuracy loss for *per-event* segmentation** — that angle is under-covered in the literature. This
repo, however, already has substantial validated capability work to lean on (clDice/boundary/
skeleton-match losses, time-surface representation, normal-flow motion features) — the research
doesn't refute those, it just can't cite external per-event numbers. Prioritize the **architectural**
levers (RoPE + MIMO) that came back with evidence, and treat clDice-on-events as an internal
experiment, not a literature-backed given.

---

## 6. Ranked ladder (biggest safe win first)

| # | Change | Expected speed | Accuracy risk | Status |
|---|---|---|---|---|
| **1** | **Wire the installed SSD kernel** (`mamba_chunk_scan_combined`, varlen `cu_seqlens`, batch=1) to replace the pure-PyTorch Hillis-Steele scan | **~2–2.5×** (retires the 63%-of-step scan/complex-mult) | Low **IF complex preserved via RoPE** | installed, unbuilt |
| **2** | **fp16/bf16 states + chunk 128 + CUDA graphs** (+ `torch.compile` once complex path is gone) | ~1.3–2× (attacks kernel-launch overhead) | Low | not built |
| **3** | **RoPE complex transitions + MIMO state update** (Mamba-3) — keeps oscillatory expressivity on the real kernel, adds accuracy at ~free decode cost | speed-neutral | Low–mild (LLM-measured only) | not built |
| **4** | **Plain real-diagonal reparam** — *only* if RoPE is infeasible | enables kernel path | **Risky** (parity→random without RoPE; contested magnitude) | this is the existing `use_complex:False` R1 gate |
| **5** | **Event-pooling + learned per-event scatter head** (`event_pool` stub) | further L→L/p | **Risky to per-event fidelity** | stub `NotImplementedError` |
| **6** | **Event batching** (multiple windows/streams) to cross 10 M in *throughput* mode | ~linear until compute-bound | none (but adds latency) | config change |

**Sequencing.** 1→2 are the accuracy-safe core and should land first (gate #1 on: does real-diagonal
+ RoPE hold `val_event_f1` / `f1_best`? — matched ~8-epoch run). #3 rides along with #1. Only reach for
#4 if RoPE can't be wired. #5/#6 are how you actually *clear* 10 M if 1–3 land you at ~3–8 M — pick #6
(throughput) if latency budget allows, #5 (per-event pooling) if it doesn't.

**Framing to lock down before building:** decide whether 10 M ev/s is a **throughput** target (event
batching legal → reachable) or a **streaming-latency** target (batch=1, one window at a time → likely
needs FPGA). The current bench measures batch-1 latency mode; the two answers differ by the batch
factor.

---

## 7. Open questions (what the research could not settle)

1. **Actual SSD throughput on A5000/L40S** at our event lengths (30 k–150 k/window) — all fusion
   numbers are A100/H100; does the tensor-core path alone clear 10 M or fall short? *(Measure once
   wired.)*
2. **How much per-event F1/IoU is lost by complex→real**, and does RoPE fully recover it on this
   non-language task?
3. **Can `event_pool` + a learned scatter head recover per-event labels** while keeping the L→L/p
   speedup, and at what accuracy cost?
4. **Do clDice / normal-flow / time-surface tokenization help per-event edge quality at ~0 inference
   cost?** — no external measured evidence; internal experiment needed.
5. **What full systems stack** (bf16 + compile + CUDA graphs + INT8 + event batching) closes the gap,
   and does the required batching violate the streaming-latency requirement?

---

## Sources (verified, most load-bearing)

- PyTorch blog — *Accelerating Mamba2 with Kernel Fusion* (1.50–2.51× SSD, 8–13 % e2e, fp16/chunk-128, >99.9995 % @1 % tol) — primary
- Tri Dao — *Mamba-2 parts 3–4* (SSD = tensor-core matmuls + one scan; faster than `selective_scan_cuda`) — author blog
- state-spaces/mamba `ssd_combined.py` (varlen `cu_seqlens`, batch=1; real-only `tl.float32`) — primary
- arXiv 2410.14067 (NeurIPS 2024) — real-diagonal exponential lower bound, parity 0.9 % vs 100 % — primary
- arXiv 2603.15569 (Mamba-3, ICLR 2026) — RoPE trick + MIMO state update, memory-bound decode — primary
- arXiv 2312.06635 (GLA, ICML 2024) — parallel-scan memory-I/O bottleneck; chunkwise > FlashAttn-2 @1 K — primary
- arXiv 2511.10363 — Hillis-Steele vs Blelloch same O(log T) depth (refutes Blelloch as a speed lever) — primary
- arXiv 2404.18508 (Event-SSM) / 2405.06116 (EventMamba) — event-pooling, no per-event output — primary
- arXiv 2505.11165 (EVA) — 0.611 M ev/s per-event, ~9.78 M only w/ 8× downsampling — primary
- arXiv 2406.07318 — FPGA event GCN 13.3 M ev/s @ 4.47 ms — primary
- github.com/fla-org/flash-linear-attention — Mamba/2/3 + 20+ variants, varlen (not installed) — primary

**Refuted (did not survive 2/3 adversarial verification):** "complex strictly more expressive per
dimension" (1-2); "Blelloch much faster than Hillis-Steele in practice" as a speed lever (0-3); "larger
SSD state dims are accuracy-positive" (1-2).
