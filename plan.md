# Research Plan: Efficient Event-Based Hand/Arm Segmentation via Video-Model Distillation

## Context

The internship aims to build a low-power, low-latency perception front-end that
detects/segments animated objects (initially human hand+arm during tool-use
miming) from an event-camera stream. The output mask will be consumed by
downstream action-recognition models, so the front-end must run at very high
FPS, fit in ≤1M parameters, and produce temporally stable masks even when
parts of the hand are momentarily stationary (and therefore invisible to the
event sensor).

The starting asset is a synced **RGB + event-camera dataset (beam-splitter,
pixel-aligned, timestamps synchronized)** of arm/hand miming tool use. The
research strategy is **cross-modal knowledge distillation**: a heavy video
segmentation teacher (SAM 2) generates per-frame hand/arm masks on the RGB
stream; a tiny event-only student is trained to reproduce these masks from the
event stream alone. The current repository is a clean PyTorch Lightning
classification template (`main.py`, `model_interface.py`, `data_interface.py`,
`configs/`, `model/simple_net.py`, `data/cifar10.py`) and contains **no
event-camera code** — all event-domain modules will be new additions
following the existing snake_case/CamelCase convention.

---

## Literature Snapshot (orients the design choices)

- **Event representations**: voxel grids (Zhu et al. 2019), time-surfaces /
  HATS (Sironi et al.), ERGO-12 (Gehrig 2024), event histograms, async point
  clouds. Voxel grids and time-surfaces dominate detection/segmentation work
  because they are dense, GPU-friendly, and differentiable.
- **Event detection/segmentation**: RVT (Gehrig & Scaramuzza, CVPR 2023)
  recurrent vision transformer for object detection; ESS (Sun et al., ECCV
  2022) for event semantic segmentation via UDA from frames; EvDistill (Wang
  et al., CVPR 2021) for cross-modal distillation.
- **Hand-specific event work**: EventHands (Rudnev, ICCV 2021), Ev2Hands
  (Millerdurai, 2024), EvHandPose (2024). All target pose, not dense
  segmentation — there is a real gap for **event-based hand/arm segmentation**.
- **Video segmentation teachers**: SAM 2 (Meta, 2024) gives temporally
  consistent video masks with prompt propagation; Cutie / XMem / DEVA are
  alternatives. GroundingDINO + SAM 2 enables text-prompted ("hand", "arm")
  automatic seeding.
- **State-space sequence models**: S4, S5, Mamba, and visual variants
  (Vim, VMamba, EVSSM 2024). Linear-time recurrence is a strong fit for
  high-FPS event streams that need persistence of static regions.
- **Synthetic events**: v2e (Hu et al.), ESIM (Rebecq et al.), DVS-Voltmeter.
  MANO + tool meshes → rendered video → v2e → paired events+GT masks.

---

## Research Questions & Novel Contributions

1. **RQ1 — Cross-modal mask distillation under sparsity.** Can a ≤1M-param
   event-only student match SAM 2's dense mask on hand/arm using only events,
   when stationary regions emit no events?
2. **RQ2 — Persistence via state-space backbones.** Does a Mamba/S5-style
   linear-recurrent backbone outperform pure feed-forward CNNs and ConvLSTM
   at maintaining masks across motion pauses, at equal parameter budget?
3. **RQ3 — Motion-confidence-aware distillation.** Does weighting the
   distillation loss by local event density (high weight where events are
   informative, lower elsewhere) yield better generalisation than uniform
   pixel BCE?
4. **RQ4 — Predictive front-end.** Can the same backbone forecast a
   short-horizon (50–200 ms) future mask, useful as an early-warning signal
   for downstream action recognition?
5. **RQ5 (optional fork) — Sim-to-real with v2e.** Does synthetic
   MANO+v2e pretraining reduce real-data requirements and improve
   robustness?

Primary novelty target: **(2) + (3) combined** — a tiny event-native SSM
segmenter trained with motion-confidence-weighted distillation — is a clean
contribution with no direct prior art.

---

## System Design

### 1. Data & Alignment

- Beam-splitter is pixel-aligned and time-synced → SAM 2 masks on RGB
  transfer **directly** as per-pixel targets for events at the same
  timestamps. No homography step needed; include a one-time sanity check
  (overlay SAM mask on event accumulation image, eyeball alignment on a few
  clips).
- Add `data/hand_event_dataset.py` (a `torch.utils.data.Dataset`) that yields
  windows of `(event_tensor[T, B, H, W], teacher_mask[T_keyframes, H, W],
  timestamps)`. Hook it into `data_interface.py` via the existing class-name
  convention.

### 2. Teacher Pipeline (offline, run once per recording)

- For each recording: run **GroundingDINO** with prompt "hand. arm." on
  frame 0 → boxes → seed **SAM 2** video predictor → propagate masks across
  the full clip.
- Cache masks as compressed PNGs / `.npz` per frame.
- Manual spot-check on a held-out subset (5–10 clips) to validate teacher
  quality; correct failure cases with click prompts.

### 3. Event Representation (input to student)

- Primary: **voxel grid** with B time bins over a sliding window Δt
  (start with B=5, Δt=33 ms aligning to 30 FPS RGB; ablate Δt down to
  5 ms for high-FPS regime).
- Secondary representation to ablate: **two-channel time-surface** (per
  polarity) — cheaper, lower memory.
- Implement as `data/event_representations.py` with vectorised PyTorch ops
  (no Python loops over events).

### 4. Student Architectures (≤1M params)

Two backbones, shared lightweight U-Net-style decoder, both new files under
`model/`:

- **`model/event_unet.py`** — ANN baseline. 4-stage encoder (32→64→96→128
  channels), depthwise-separable convs, skip connections, decoder upsamples
  to full resolution and outputs a 1-channel mask logit. Target ~0.4–0.8 M
  params.
- **`model/event_ssm.py`** — SSM backbone. Same encoder stem, but stages 3–4
  replaced with 1D-Mamba (or S5) blocks operating along the time axis on
  per-pixel patch tokens (à la Vim/VMamba). Maintains a recurrent hidden
  state across windows so static regions persist. Target ~0.6–1.0 M params.

Both register through `model_interface.py`'s existing class-name lookup.

### 5. Distillation Losses (`loss/distillation.py`, new)

- **L_mask**: per-pixel BCE + Dice between student logits and SAM 2 mask.
- **L_motion**: motion-confidence reweighting — multiply per-pixel BCE by
  `(1 + α · event_density_map)`. Stationary pixels keep gradient signal at
  baseline weight, but moving pixels (where the student actually has
  information) dominate.
- **L_temporal**: optional consistency loss penalising mask flicker between
  consecutive windows when no events occurred in that region.
- **L_feat** (optional): align an intermediate student feature map with a
  frozen RGB encoder feature (e.g. DINOv2 patch features) via 1×1 projection
  + cosine similarity. Cheap to add; well-studied to help cross-modal
  transfer.
- Total: `L = L_mask + λ_m · L_motion + λ_t · L_temporal + λ_f · L_feat`.

### 6. Predictive Head (RQ4, Phase D)

- Add a second 1-channel output head that predicts the mask `k` windows
  ahead; train with the same losses but using a delayed teacher mask. Reuses
  the SSM hidden state — minimal extra parameters.

### 7. Synthetic-Data Fork (sketched, decision deferred)

- Pipeline: MANO hand mesh + simple tool primitives (Blender or PyTorch3D) →
  rendered RGB video with GT masks → v2e → synthetic events with **exact**
  paired masks (no teacher noise).
- Use cases: (a) pretraining the student before real distillation;
  (b) controlled evaluation of persistence and camera-motion robustness
  (render with simulated ego-motion).
- **Decision point**: revisit after Phase B baseline numbers. Go/no-go
  criterion — if real-data student plateaus below 0.8 IoU on held-out
  recordings, invest in synthetic; otherwise defer to a future sub-project.

---

## Evaluation Protocol

- **Splits**: subject-disjoint train/val/test split across recordings to
  test generalisation across people and tools.
- **Metrics**:
  - Mask quality: mIoU, boundary F-score, temporal mask stability
    (per-pixel flicker rate).
  - Efficiency: parameters, MACs/window, measured FPS on a target device
    (Jetson Orin Nano as a stand-in; ONNX export path).
  - Latency: end-to-end from event arrival to mask output.
- **Stress tests**: motion-pause clips (hand briefly still), high-speed
  clips, fast camera-motion clips (record a small dedicated set if not
  in the existing dataset).
- **Baselines**: (i) frame-only U-Net on event-frame accumulations,
  (ii) ESS-style adapted segmentation, (iii) teacher upper bound on RGB.

---

## Phased Timeline

- **Phase A (weeks 1–2)** — Dataset loader, alignment sanity check, SAM 2
  teacher pipeline producing cached masks; visualisation tooling.
- **Phase B (weeks 3–4)** — Voxel-grid representation + `event_unet.py`
  ANN baseline trained with `L_mask` only. Establish first numbers.
- **Phase C (weeks 5–6)** — Add `L_motion` and `L_temporal`; introduce
  `event_ssm.py` and run head-to-head ANN vs SSM ablation. **Decide on
  synthetic-data fork here.**
- **Phase D (weeks 7–8)** — Predictive head; high-FPS evaluation;
  efficiency benchmarking and ONNX export.
- **Phase E (weeks 9–10)** — (Conditional) synthetic pretraining; dynamic
  camera evaluation; integration smoke-test with a downstream action head.
- **Phase F (weeks 11–12)** — Ablations, write-up, repository cleanup.

---

## Critical Files to Add (in this repo)

Following the existing template's snake_case-file / CamelCase-class convention
auto-discovered by `data_interface.py` and `model_interface.py`:

- `data/hand_event_dataset.py` — `HandEventDataset` yielding event windows
  + teacher masks.
- `data/event_representations.py` — voxel-grid and time-surface builders.
- `data/sam2_pseudo_labels.py` — offline teacher inference script
  (GroundingDINO + SAM 2 → cached masks).
- `model/event_unet.py` — `EventUnet` ANN baseline.
- `model/event_ssm.py` — `EventSsm` state-space backbone.
- `loss/distillation.py` — `DistillationLoss` combining BCE/Dice, motion
  weighting, temporal consistency, optional feature alignment.
- `utils/metrics/segmentation.py` — mIoU, boundary F, mask stability.
- `configs/sections/` additions: event window/representation config, teacher
  paths, loss weights; `configs/config.yaml` switched to the new
  dataset/model.
- New requirements: `segment-anything-2`, `groundingdino`, `mamba-ssm`
  (or `s5-pytorch`), `h5py`, `tonic` (for event-format utilities), `v2e`
  (only if synthetic fork is activated).

The existing `main.py` Lightning entry point, callbacks, and YAML config
machinery can be reused unchanged — the project plugs into the template
rather than replacing it.

---

## Verification

- **Alignment**: render an overlay of SAM mask + accumulated events on
  10 random windows per recording; visually confirm pixel alignment.
- **Teacher quality**: report mask mIoU vs hand-corrected masks on a 30-clip
  audit set; require ≥0.9 mIoU before using as supervision.
- **Student quality**: held-out subject mIoU and boundary F vs teacher;
  efficiency measured on Jetson with `torch.cuda.Event` timing and ONNX
  Runtime; mask stability measured as per-pixel flicker rate over still
  frames.
- **End-to-end smoke test**: stream a recording through the trained student
  and visualise the mask overlaid on accumulated events at the target FPS.
- **Reproducibility**: every Phase ends with a tagged commit, a config file
  checked in under `configs/`, and a short results table in
  `results/phaseX.md`.
