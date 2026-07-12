# Making EventStreamSegS7 Faster **and** More Capable — Deep-Research Report

**Question.** Novel model components and loss functions to make a per-event selective
SSM segmentation network (EventStreamSegS7) both *faster* (training throughput **and**
inference latency) and *more capable* (higher segmentation accuracy), for event-camera
per-event hand/arm segmentation. Custom CUDA kernels permitted.

**Provenance.** `/deep-research` workflow (run `wf_74e06754-6ea`, 2026-07-12): 5 search
angles → 22 sources fetched → 99 claims extracted → **25 adversarially verified (3-vote,
2/3-refute-to-kill) → 23 confirmed, 2 refuted** → 7 synthesized findings. 104 agents,
0 errors. Scoping (both-speed+capability / custom-kernels-OK / both-training+inference)
per user selection.

**Integrity note.** Two tiers of evidence below: **(A) Verified core** = the 7 findings
that passed 3-vote adversarial verification (high confidence). **(B) Supplementary
signals** = source-extracted claims (fetch phase) that were *not* put through the verify
gate — relevant and single-source, flagged as such. **(C) Refuted** = do NOT rely on.

---

## 0. Grounding: what we already measured about S7 (this session)

Not from the web — from our own training logs and benchmark, so the recommendations
below attach to real numbers:

| Fact | Value | Source |
|---|---|---|
| S7 train speed | **0.731 it/s, 131 min/epoch** (2.55× slower than grid baseline's 1.87 it/s) | tfevents |
| Root cause | scan = many small **sequential kernel launches** × layers × dual passes + on-the-fly argsorts + grad-checkpoint recompute | bench + code |
| S7 val F1 | 0.7738 @ep8 (baseline 0.7859 @ep7) — **near-parity, not converged** | tfevents |
| S7 wins on | val_loss 0.567<0.634, **f1_best 0.9145>0.9093** (better separability) | tfevents |
| **Dual z-order (Morton) scan is INERT** | `train_scan_loss` ≈0 from epoch 1 | tfevents |
| Inference (batch 1) | S7 **faster** than baseline (~17 ms / 30k events) | bench |
| Event budget | max_events 30k (dense IoU artifact, not a defect) | logs + previews |

The three levers the research must serve: **(a)** kill the sequential-scan launch
overhead; **(b)** cut scan length; **(c)** fix the inert spatial scan and sharpen thin
fingers — without erasing the speed win.

---

## A. VERIFIED CORE (3-vote confirmed)

### A1 — PRIMARY SPEED FIX: chunkwise-parallel scan kernel (replace Hillis-Steele) `[high]`
The single biggest lever for **both** training and inference. Mamba-2's **state-space
duality (SSD)** reformulates the sequential associative (Hillis-Steele-style) scan as
**chunked matmuls on tensor cores** — exactly removing "many small sequential kernel
launches." The **flash-linear-attention (FLA)** library ships production Triton kernels
for this family (Mamba-2/SSD, Gated DeltaNet, GLA, RetNet, DeltaNet, RWKV7) with **native
variable-length `cu_seqlens` packing** — which maps directly onto our ragged per-sample
event batches under DDP. Measured: FLA `chunk_gdn` forward **1.265 ms vs FlashAttention-2
3.753 ms (2.97×)** at T=8192 (reverses at short T — honest). **TFLA** (NeurIPS 2025)
explains the "why": small chunks → low arithmetic intensity + materializing many small
states; two-level (chunk + intra-chunk) tiling enables arbitrarily large chunks.
- Cite: FLA repo; TFLA arXiv 2503.14376; Mamba-2 SSD (tridao.me/blog/2024/mamba2-part3).
- **⚠ Caveat (critical):** S7's **complex-diagonal + STREAM per-event Δt** scan is *not*
  a stock FLA/TFLA kernel. Capturing the speedup requires **reformulating to a
  real-diagonal Mamba-2 SSD structure** (likely dropping/approximating the complex
  readout). No source demonstrates a chunkwise complex-diagonal per-event-Δt scan — this
  is the main integration risk (see Ladder Tier 1).

### A2 — SSMs are a proven event-camera throughput win + free frequency-generalization `[high]`
Swapping a recurrent/transformer temporal backbone for an SSM **trains 33% faster at
matched accuracy** (S5-ViT-B 47.2 vs RVT-B 47.4 mAP, 1Mpx). SSMs' **learnable timescale
(Δ)** let a model trained at one event rate **run at higher inference frequency without
retraining** (only −3.76 mAP vs −20+ for RNN/Transformer). S7 already has per-event Δt, so
this robustness is *inherent* — exploit it for deployment.
- Cite: Zubić et al. "State Space Models for Event Cameras", CVPR 2024, arXiv 2402.15584.
- ⚠ Caveat: demonstrated on **dense-grid / T=10 time-bin** representations, not a raw
  per-event stream — validates the SSM *choice*, not per-event scan throughput.

### A3 — SECOND SPEED AXIS: sequence-length reduction (event-pooling / merging) `[high]`
Orthogonal to A1 and event-native. **Event-SSM's event-pooling** compresses M→M/p *after
each state-space layer* (all M integrated into state, only M/p forwarded) — directly the
`event_pool` knob already stubbed in our config. **ToMe** (training-free token merging)
gives **~2× inference** (0.2–0.3% acc drop) *and* **~2× training** (MAE video) by merging
similar tokens. **PAST-SSM/PEAS** learns a Gumbel-Softmax selector for informative frames.
- Cite: Event-SSM arXiv 2404.18508; ToMe arXiv 2210.09461; PAST-SSM arXiv 2409.16953.
- ⚠ Caveat: Event-SSM pooling shown on **classification**; ToMe on ViT attention. For our
  **dense per-event** task a pooled output must still **recover a label for every dropped
  event** (unpool/scatter) — the key open question (§D).

### A4 — CAPABILITY: topology/boundary losses sharpen thin fingers without trading overlap `[high]`
**soft-clDice** (skeleton∩mask, differentiable, drop-in) yields more accurate connectivity
+ higher graph similarity + **better volumetric scores** on 5 datasets — the exact property
for thin hand/arm/finger structure. **Boundary loss** (distance metric on *contours*, not
regions) avoids the order-of-magnitude class-imbalance summation of region losses; added on
top of a regional loss gives **significant gains + more stable training** (up to +8% Dice,
+10% Hausdorff). We already validated clDice as a thin-finger supplement (memory).
- Cite: clDice arXiv 2003.07311 (CVPR 2021); Boundary loss arXiv 1812.07032 (MIDL 2019).
- ⚠ Caveat: clDice works **best combined with soft-Dice** (pure clDice slows convergence);
  boundary loss needs a regional→boundary **α-schedule**.

### A5 — SPATIAL LOCALITY: Hilbert > Morton for the (inert) second scan `[high]`
The Hilbert space-filling curve preserves 2D locality **strictly better** than Morton/
z-order (bounded vs unbounded worst-case dilation; ~2.0 vs 2.625 clusters/query) — adjacent
points in the 1D sequence stay geometrically closer. Since the SSM scan is order-agnostic,
switching is just a **precomputed index permutation**.
- Cite: EventMamba arXiv 2503.19721; PTv3 arXiv 2312.10035; Moon et al. TKDE 2001.
- ⚠ Caveat: single-swap accuracy gain is **small (~1%)**. Our dual-Morton signal was
  measured **inert** → the win is a *better single curve*, not more Morton.

### A6 — SPATIAL LOCALITY: alternating/shuffled orders beat a fixed dual pass `[high]`
PTv3 uses **four serialization patterns** (Z, Trans-Z, Hilbert, Trans-Hilbert) with random
Shuffle Order so each layer's receptive field isn't one pattern; EventMamba concatenates
Hilbert+Trans-Hilbert; PointMamba sorts+concatenates along x/y/z (**+0.69/+1.38%**);
VMamba's SS2D traverses four routes. Direct precedent to replace S7's fixed inert Morton.
- Cite: PTv3 2312.10035; EventMamba 2503.19721; PointMamba 2402.10739; VMamba 2401.10166.
- ⚠ Caveat: gains **small (~1%)**, on point-cloud **classification**, and multi-order
  **adds sequence length** (scan cost) — directly *against* the speed goal. VMamba's route
  sub-claim was the only split vote (2-1).

### A7 — Serialization can replace expensive KNN (validates the design choice) `[high]`
PTv3: space-filling-curve **serialized neighbor mapping** replaces precise KNN → **3× faster,
10× less memory** than PTv2, receptive field 16→1024. Validates that argsort/serialization
(already how S7 builds its scan) substitutes for costly neighbor ops.
- Cite: PTv3 arXiv 2312.10035 (CVPR 2024, 1st Waymo'24 semantic-seg).
- ⚠ Caveat: PTv3's speedup is inside **attention KNN**, not the SSM-scan launches that are
  *our* bottleneck — validates the design, doesn't itself fix launch overhead.

---

## B. SUPPLEMENTARY SIGNALS (source-extracted, NOT 3-vote verified — treat as leads)

- **E-3DPSM (CVPR 2026):** event-native **band-limited S5**, **train-parallel /
  infer-recurrent** bidirectional design — parallel conv form bidirectionally in training,
  causal forward-only recurrence at inference. **Real-time 80 Hz (A6000)**, **+19% MPJPE,
  2.7× smoother** vs SSM-free EventEgo3D++. → The cleanest architectural template for
  "train fast in parallel, deploy as a streaming recurrence." (openaccess CVPR2026; arXiv 2606.17966)
- **Hierarchical multi-directional (4-dir) Mamba + segmentation-specific heads:** boundary
  supervision + **class-uncertainty gating** + hierarchical multi-level refinement →
  **+2.2 mIoU ADE20K** over a direct-port Mamba, countering Mamba "response dilution" in
  dense prediction. (arXiv 2411.12603)
- **Multi-scale SSM placement matters:** full model 84.45 MPJPE vs single-deep-SSM 90.18 vs
  no-SSM 118.53 — hierarchical/multi-stage SSMs >> one deep SSM.
- **Guided point contrastive loss** as a *pure auxiliary* in the fully-supervised setting:
  ScanNet 72.9→74.0, S3DIS 66.4→68.8, SemKITTI 65.0→65.8 mIoU — bolt-on, no extra data. (arXiv 2110.08188)
- **A-ToMe:** merging *adjacent* similar tokens → 57% fewer tokens, **+70% GPU inference**,
  no notable accuracy loss (ASR transducer).
- **EventMamba inference 13.6× faster** than a Transformer event baseline (136 vs 1850 ms
  @720p); **Event-SSM scales to 1.5M events/sample**, trains in 2–10 h on one A100 — feasibility
  evidence for event-by-event SSM at scale.
- **Bidirectional B-Mamba on event data +8.31%** (HARDVS 98.41 vs 90.10) — bidirectional
  selective scan helps on event streams. **fla supports hybrid SSM+attention** layers.

---

## C. REFUTED — do NOT use these as levers

- ❌ **"Tversky > Dice by up to 6.36% DSC"** — **0-3 refuted**. Do not swap in Tversky
  expecting a class-imbalance win. (arXiv 2312.05391)
- ❌ **"Focal/Combo best for small/sparse objects"** — **1-2 refuted**. Do not rely on
  focal as the sparse-per-event loss lever. (Our own repo memory already measured focal
  collapsing on easy interior → mask holes — consistent with this refutation.)

The verified loss lever is **clDice + boundary-on-top-of-region (A4)**, not Tversky/focal.

---

## D. CONCRETE IMPLEMENTATION LADDER FOR S7 (impact × effort, ranked)

Ordered by expected value. Each maps a finding to a specific change in
`model/event_stream_seg_s7.py` / config / loss. Marked **[drop-in]**, **[medium]**, or
**[research]**.

### Tier 1 — The two big speed wins (do these to close the 2.55× gap)

**T1a. Reformulate the selective scan to Mamba-2 SSD chunkwise form + FLA/causal-conv1d
kernels.** `[research — highest impact]`
*Why:* directly removes the launch-overhead bottleneck (A1); helps training **and**
batched inference. *What:* replace the Hillis-Steele `_segmented_scan` with a chunkwise SSD
scan; use FLA's `cu_seqlens` varlen path for ragged event batches (no padding waste under
DDP). *Cost:* must move **complex-diagonal → real-diagonal SSD** (SSD is real). *Risk to
manage:* our `f1_best` edge may partly come from complex oscillatory modes — **ablate
real-diagonal vs complex first** on a short run before committing. *Expected:* approach or
beat baseline throughput; this is the only lever that structurally fixes the scan.

**T1b. Turn on event-pooling (`event_pool`) between SSM blocks.** `[medium — high impact,
event-native]`
*Why:* A3, orthogonal to T1a; the knob is already stubbed. *What:* after each
`_SelectiveSSMBlock`, integrate all M events into state but forward only M/p (Event-SSM
style); at the head, **unpool/scatter** the pooled logits back to every event (bilinear or
nearest in the serialized order) so we keep a per-event label. *Cost:* need the unpool path
(open question D-2). *Expected:* ~p× on the scan length — the cleanest capacity-preserving
speedup, and it also raises the effective event budget (helps the dense-IoU artifact).

### Tier 2 — Fix the inert scan (net speed + small accuracy)

**T2. Replace the inert bidirectional Morton scan with a single Hilbert scan.** `[drop-in]`
*Why:* A5+A6 — Morton is the weaker curve and our second pass is *inert* (train_scan_loss
≈0), so it's paying ~⅓ of scan cost for nothing. *What:* swap `_morton2d`→Hilbert index
(precomputed permutation; scan code unchanged), and **drop `dual_scan`** (or keep a single
Hilbert pass). *Expected:* recovers most of the ~⅓ dual-scan cost **and** a possible ~1%
accuracy bump from better locality. If we later want more, cycle shuffled orders
(Z/Trans-Z/Hilbert/Trans-Hilbert) — but only if it beats the added sequence-length cost.
*This is the cheapest concrete win and resolves the inert-scan finding directly.*

### Tier 3 — Capability (layer on top, cheap)

**T3a. Add soft-clDice + contour boundary loss (α-scheduled) on top of GJS+Lovász.**
`[drop-in]`
*Why:* A4 — sharpen thin fingers/arm, improve F1 without trading region overlap; repo
already has clDice validated. *What:* extend `EventDistillationLoss` with soft-clDice
(kept ON TOP of the region loss, not replacing) + boundary loss with a regional→boundary
α-warmup. **Do not** add Tversky/focal (§C). *Expected:* thin-structure F1 gain; addresses
the "better model, worse operating point" gap we measured.

**T3b. (Optional) Guided point-contrastive auxiliary + class-uncertainty gating.**
`[medium/research]`
*Why:* B — pure-auxiliary contrastive gave +1–3 mIoU fully-supervised; uncertainty gating
countered Mamba response-dilution (+2.2 mIoU). *What:* an auxiliary per-event contrastive
head (train-only, zero inference cost) and/or an uncertainty gate on boundary events.
*Expected:* incremental; try after T3a.

### Tier 4 — Architecture rethink (if pursuing a v2)

**T4. Train-parallel / infer-recurrent bidirectional band-limited S5 (E-3DPSM template).**
`[research]`
*Why:* B — the CVPR-2026 event-native design that is simultaneously **real-time (80 Hz)**
and **more accurate**; matches S7's needs (bidirectional in training, causal streaming at
inference). Pairs naturally with T1a (SSD chunkwise = the parallel training form) and gives
a principled streaming-inference story. Consider multi-scale SSM placement (B: hierarchical
≫ single deep SSM).

**Suggested sequence:** T2 (drop-in, immediate) → T3a (drop-in capability) → T1b
(event-pool, big speed) → T1a (SSD kernel, biggest but riskiest) → T4 (v2). Run the
real-vs-complex ablation before T1a.

---

## E. Caveats & open questions (from the verify phase)

**Applicability gap (most important):** nearly every cited number is from a *different*
setting — FLA/TFLA on LLM long-token workloads (and Blackwell GB200, so relative speedups
differ on our L40S/A5000); the 33% SSM win, PEAS, Event-SSM pooling on dense-grid /
classification; ToMe/clDice/boundary on ViTs and medical imagery; scan-order gains on
point-cloud classification. **Per-event dense-segmentation transfer is a reasonable but not
directly measured inference.**

Open questions the research could not close:
1. **Can complex-diagonal + STREAM per-event Δt be expressed chunkwise (SSD)** to reuse FLA
   kernels, or must the complex readout be dropped/approximated? (No source shows it.)
2. **Does event-pooling/merging preserve per-event DENSE accuracy** (boundary sharpness)
   when every event needs a label, not a pooled class? How to recover dropped events' labels?
3. **Does Hilbert/multi-order beat ~1% for per-event segmentation**, given our dual-Morton
   was inert — and does multi-order's added length erase the throughput goal?
4. **Inference-latency under strict online one-event streaming:** chunkwise parallelism is a
   *batch/training* lever and may not map to real-time per-event latency — the "faster
   inference" goal may need a separate streaming/state-carry eval (the E-3DPSM
   train-parallel/infer-recurrent split is the likely answer).

---

## F. Sources (22 fetched; primary unless noted)

**Speed/kernels:** FLA github.com/fla-org/flash-linear-attention · TFLA arXiv 2503.14376 ·
Mamba-2 SSD tridao.me/blog/2024/mamba2-part3-algorithm (blog).
**Sequence reduction:** Event-SSM 2404.18508 · PAST-SSM 2409.16953 · ToMe 2210.09461 ·
(A-ToMe 2306.16009).
**Losses:** clDice 2003.07311 · Boundary loss 1812.07032 · Guided point contrastive
2110.08188 · Lovász-Softmax (CVPR 2018) · loss survey 2312.05391 (source of the 2 refuted
claims).
**Scan order:** EventMamba 2503.19721 · PTv3 2312.10035 · PointMamba 2402.10739 · VMamba
2401.10166.
**Architecture / event-native SSM:** SSMs-for-Event-Cameras 2402.15584 · hierarchical
multi-dir Mamba seg 2411.12603 · E-3DPSM CVPR 2026 (2606.17966).

*Stats: 5 angles · 22 sources · 99 claims · 25 verified · 23 confirmed / 2 refuted · 7
synthesized · 104 agents.*
