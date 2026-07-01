---
title: Learning the Dynamics of a Continuous (x,y,t) Event Manifold — Literature Landscape & Action Plan
date: 2026-06-29
project: EventBasedObjectDetection — per-event hand segmentation
generated_by: multi-agent literature workflow (9 angles, 101 papers, adversarial citation verification + completeness critic)
---

# Learning the Dynamics of a Continuous (x,y,t) Event Manifold — Landscape and Action Plan

## 1. The landscape

The field of "learning point-cloud / manifold dynamics" splits into six coherent families. Each learns *how a surface evolves through time*, but they differ in whether the evolution is modeled in **latent coordinates**, as a **spectral operator**, as a **predictive residual**, as an **implicit field**, as a **layer decomposition**, or as a **structure-constrained flow**.

**(A) Latent dynamics on a constrained manifold.** Compress high-dimensional point streams into low-dimensional manifold coordinates and evolve them with a Neural ODE/SDE that respects the geometry. Neural Manifold ODEs (2006.10254) and Riemannian CNFs (2006.10605) give the continuous substrate; CaSPR (2008.02792) canonicalizes irregular spatiotemporal point clouds then learns latent dynamics via Neural ODE + normalizing flow; hierarchical latent-SDE with inducing points (2507.21531) makes the latent cost O(1) in data dimension. The signal is the *shape of the latent trajectory*, not the input magnitude.

**(B) Operator / Koopman / DMD (spectral).** Lift the nonlinear dynamics to a latent space where evolution is *linear* (Ax_t = x_{t+1}), then read off modes. Deep Koopman autoencoders (1712.09707) discover eigenfunctions end-to-end; temporally-consistent KAEs (2403.12335) stabilize them from sparse/noisy data; and — most relevant — real-time DMD background/foreground separation (1404.7592, 2405.05057) shows that **zero-frequency modes = predictably stationary content, non-zero modes = motion**, a parameter-free static/dynamic gate.

**(C) Predictive / world-model (JEPA family).** Predict the *next latent state*, not the next pixels/points; the prediction residual is the discriminative signal. V-JEPA 2 (2506.09985) is the nearest precedent (predict masked future latents → static content is maximally predictable). DINO-as-world-model (2507.19468) decouples a frozen spatial encoder from a small learned temporal predictor; Point-JEPA (2404.16432) confirms JEPA transfers to raw point clouds; F³ (2509.25146) is the event-native instance — a hash-encoded net predicting future events at 120–440 Hz, where unpredictable = motion boundary/noise.

**(D) Neural-field surface evolution.** Represent the surface as an implicit field f(x,y,t)=0 and evolve it with a level-set PDE ∂f/∂t = v|∇f|. Neural Implicit Surface Evolution (2201.09636) and Occupancy Flow (ICCV 2019, *conference-only, no arXiv preprint*) learn a continuous velocity field over spacetime; Neural Eulerian Scene Flow (2410.02031) argues the Eulerian (per-point instantaneous velocity) view beats Lagrangian trajectories for dense streams; D-NPC (2406.10078) uses **dual static/dynamic grids** — an explicit "always-off vs. moving" decomposition that maps directly onto the static-FP problem.

**(E) Motion-as-submanifold / common-fate decomposition.** Treat the scene as several motion-coherent layers and assign each event to one. On events this is the contrast-maximization lineage: the unifying CMax framework (1804.01306), per-event motion-compensation segmentation (1904.01293), spatio-temporal graph-cuts EMSGC (2012.08730), unsupervised Un-EVIMO (2312.00114), and the **direct competitor**, Iterative Variational Contrast-Max (2504.18447), which alternates a dominant-motion (background) hypothesis against independent-motion residuals (foreground) for a >30% detection gain. Slot attention / SAVi++ (2006.15055, 2206.07764) and GPCA subspace clustering (1202.4002) are the generic vision analogues.

**(F) Continuous-time SSM/CDE + structure-preserving.** Stream the manifold state with a continuous-time recurrence. Neural CDEs (2005.08926) and Latent ODEs (1907.03907) handle irregular sampling per-observation; Mamba (2312.00752) uses **selective state-space parameters that become functions of the input token** (not "content-aware routing") so each event can gate the state update; Liquid Time-constant nets (2006.04439) adapt their time constant to slow down on quiet windows. Orthogonally, structure-preserving nets (Tangent-Bundle Convolutions 2303.11323, Equivariant Manifold Neural ODEs + differential invariants 2401.14131, port-Hamiltonian passivity 2603.10078) bake in that *static input cannot create motion energy* — a hard architectural prior against hallucinated FPs.

---

## 2. What transfers to the EVENT (x,y,t) manifold

For each family: the concrete transfer, and how it meets the four measured constraints — **dense async rate (~1.35M ev/s)**, **LOSO diversity ceiling (3 subjects)**, **static = predictable** (the FP failure), and **structure > magnitude**.

- **(A) Latent dynamics.** Transfer: CaSPR-style canonicalization to a hand-centric frame + Neural ODE rollout; per-event *prediction error in canonical space* is the gate (hand sheet = low error, noise/static = divergent). Meets structure>magnitude (loss is on trajectory shape) and static=predictable (null window → near-zero canonical motion). Risk to the constraints: Neural ODE rollout is not obviously <1ms/event, and canonicalization needs a reliable hand-center estimate that LOSO scarcity may not give.

- **(B) Koopman/DMD.** Transfer: a micro-window DMD or deep-Koopman gate where **zero-mode energy ⇒ suppress (static), non-zero modal energy aligned to a learned hand-eigenspace ⇒ keep**. This is the cleanest principled answer to static=predictable and is magnitude-free (frequency structure, not displacement). Constraint tension: DMD needs ≥2 snapshots so it is intrinsically micro-window, not per-event; and under LOSO the hand vs. background eigenspaces may not separate — that is the open measurement.

- **(C) Predictive/JEPA.** Transfer: frozen event encoder (time-surface / contrast-max features) + small temporal predictor; **per-event surprise = ‖h_actual − h_pred‖** gates the logit. Directly operationalizes static=predictable (static → low surprise → veto) and structure>magnitude (latent MSE compresses orientation/crease, not velocity). Async/low-power friendly via DINO-WM's frozen-encoder/light-predictor split. LOSO caveat: SSL pretraining gains shown on dense RGB do **not** automatically transfer to sparse events — must be validated.

- **(D) Neural fields.** Transfer: D-NPC's dual static/dynamic grids → one field for "predictably silent" regions, one for hand motion; Eulerian velocity field predicts whether a region *should* fire. Strong on static=predictable. But implicit fields assume offline batch fitting and conflate "no motion" with "no data" (event dropout on truly static surfaces) — the exact ambiguity that breaks on sparse async input.

- **(E) Motion-as-submanifold.** Transfer: keep IVC-Max (2504.18447) as both baseline and front-end — its background-motion hypothesis *is* a learned null-state for static windows — then learn a light per-event soft-assignment head on the residuals. Meets structure>magnitude (it is a coherence/contrast criterion) and is O(n log n), async-friendly. Constraint tension: sub-pixel hand motion at ms windows produces a barely-perceptible manifold kink, and GPCA/hard-slot binding under-samples at 3 subjects — use *soft* clustering.

- **(F) Continuous-time SSM/structure-preserving.** Transfer: replace the coarse global-context bottleneck (`EventSparseSegGC`) with a Mamba/LTC state that carries (orientation, velocity, presence) and gates per event; port-Hamiltonian/passivity loss forbids energy on static windows. Best fit to dense async + ms latency (constant per-event cost) and to static=predictable (decay on quiet input). Caveat: state drift over 1.35M ev/s is unquantified in the literature; continuous-time models do **not** by themselves fix cross-subject transfer.

---

## 3. Ranked concrete approaches (highest-EV first)

### 1. Predictive-surprise null-gate on a frozen encoder (JEPA-for-events, drop-in)
Keep the existing sparse-conv / event-graph encoder **frozen** as the spatial feature extractor; add a small causal temporal predictor (1-D Transformer with causal masking, or a Mamba block) that predicts the next micro-window's latent from the past. Gate each event's hand logit by predictive surprise: `logit *= σ(surprise/scale)` — high surprise (unpredictable motion) keeps the event, low surprise (static, predictable) vetoes it. Train the predictor self-supervised on *all* unlabeled event windows (no LOSO label needed), then attach the gate. Grounded in V-JEPA 2 (2506.09985), DINO-WM (2507.19468), F³ (2509.25146), I-JEPA design (2301.08243). **Why it fits:** directly attacks static-FP (static=maximally predictable), is magnitude-free (latent MSE), and exploits unlabeled data so it sidesteps the 3-subject label bottleneck for the *predictor*. **Risk:** static noise may be *unpredictable* (shot noise) and thus look like motion — surprise could fire on noise, not just hand; needs the inverse check (motion-aligned, spatially-coherent surprise). **Implementability:** high — bolts onto the existing pipeline as a gating head; reuses the per-event-JEPA scaffolding already in `model/event_jepa_seg.py`.

### 2. DMD/Koopman zero-mode static gate (parameter-light, principled null-state)
Run a streaming micro-window DMD (single SVD + linear solve) or a deep temporally-consistent Koopman autoencoder (2403.12335) on event time-surfaces; classify each event by whether its local temporal spectrum is dominated by the **zero/near-zero mode** (stationary ⇒ veto) vs. non-zero modal energy aligned to a learned hand-eigenspace (keep). Grounded in 1404.7592 / 2405.05057 (proven real-time static/dynamic separation) and 1712.09707. **Why it fits:** it is the single most direct, interpretable solution to "static = predictable," costs almost nothing (laptop-real-time), and the consistency regularizer is designed for the sparse/noisy small-data regime that LOSO imposes. **Risk:** DMD is intrinsically micro-window (needs ≥2 snapshots), not per-event; and the open question is whether hand and background eigenspaces are separable under LOSO. **Implementability:** medium — classical DMD is a quick standalone gate to prototype *before* any training; deep Koopman is a larger lift.

### 3. IVC-Max front-end + learned soft-assignment residual head (beat the competitor by extending it)
Use Iterative Variational Contrast-Max (2504.18447) as the analytic decomposition: its dominant-motion hypothesis is a *free, learned-per-window null model* for the background/static sheet; the foreground residual is the candidate hand. Then learn a shallow per-event head that converts residuals into *soft* hand-vs-background assignments (not hard slots), using local flow orientation + crease coherence as features. Grounded in 1804.01306, 1904.01293, 2504.18447, with soft-clustering motivated by the sub-pixel-kink gap. **Why it fits:** directly improves on the project's measured competitor, is O(n log n) and async-friendly, and the background-hypothesis residual is exactly the missing null-state. **Risk:** contrast-max assumes all events are motion-driven — truly static windows produce low-contrast *and* jitter artifacts that fake creases; soft-assignment may not separate them. **Implementability:** medium — IVC-Max is non-neural and standalone; the head plugs into the existing per-event classifier.

### 4. Continuous-time selective-SSM bottleneck replacing the GC context (streaming-native)
Replace `EventSparseSegGC`'s coarse /8 dense global bottleneck with a Mamba/LTC/CDE state (2312.00752, 2006.04439, 2005.08926) carrying a compact manifold state (edge orientation, velocity, presence flag) updated per micro-window; selective parameters let high-structure events propagate and static events decay. Add a port-Hamiltonian passivity term (2603.10078) so static input cannot manufacture motion energy. **Why it fits:** constant per-event cost (the ms/low-power production goal), content-selective gating gives a learnable null-state, and passivity is a hard architectural prior against the hallucinated-blob failure. **Risk:** state drift over 1.35M ev/s is unquantified; selectivity alone won't fix cross-subject transfer. **Implementability:** medium-high — it is a bottleneck swap in the existing voxel/GC model, but training stability on dense streams is unproven here.

### 5. Tangent-bundle / flow-coherence structure features (cheap, addresses magnitude-non-discriminative head-on)
Compute per-event **normal-flow orientation** (plane-fit, the classic event-flow operator) and feed *orientation coherence + crease (flow-discontinuity) features* — not flow magnitude — to the existing head, optionally via tangent-bundle convolution (2303.11323). The completeness critic correctly flags that flow *is* the manifold tangent and should be a first-class feature, while RAFT-style learned flow (2003.12039) is the dense-flow benchmark to adapt. **Why it fits:** turns the measured "magnitude non-discriminative, structure discriminative" finding into an explicit feature; cheap and local. **Risk:** prior project memory already measured a plane-fit motion feature as *non-discriminative per-event* (FP-noise vs TP-hand distributions identical) — so this likely needs *coherence over a neighborhood*, not a per-event scalar, or it repeats that null result. **Implementability:** high, but lowest EV given the prior negative result — only worth it as orientation-*coherence* (structure), not per-event flow.

### Threads the digest under-weighted (fold in, lower individual EV)
- **Few-shot / meta-learning for the LOSO ceiling** (MAML 1703.03400, Prototypical Nets 1703.05175): the ceiling is a *data* problem; a meta-learned initialization or per-subject prototype that fine-tunes from a few events of the test subject targets the *actual* bottleneck more directly than any dynamics model.
- **Temporal/continuous-time GNNs** (EvolveGCN 2005.06820, T-GCN 1811.05320) for tracking persistent hand-edge graph clusters vs. fragmentary noise — natural fit for the existing event-graph pipeline.
- **Causal masking** (causal attention / dilated causal convs, WaveNet 1609.03499): static-window FPs are causally implausible (motion with no prior hand state); a causal inductive bias is a cheap correctness prior.
- **Disentangled latent state** (β-TCVAE 1802.05957): architect presence ⟂ position ⟂ velocity so magnitude cannot leak into the presence gate — directly encodes the measured finding.

---

## 4. Must-cite references

### Core methods to build on
- **V-JEPA 2 (2506.09985)** — latent predictive world model; static = maximally predictable → surprise gate. *Verified: 1M+ hrs video.*
- **F³: Fast Feature Field (2509.25146)** — event-native predictive representation at 120–440 Hz; proves streaming per-event predictability is feasible.
- **DINO as a Video World Model (2507.19468)** — frozen encoder + light temporal predictor; the low-power decoupling pattern.
- **Real-time DMD background/foreground (1404.7592, 2405.05057)** — zero-mode = static, non-zero = motion; parameter-free static-FP gate.
- **Temporally-Consistent Koopman Autoencoders (2403.12335)** — robust eigenfunctions from sparse/noisy data (the LOSO regime).
- **CaSPR (2008.02792)** — canonicalization + Neural ODE for irregular point clouds; canonical-space prediction error.
- **Mamba (2312.00752)** — selective SSM (input-dependent state params), constant per-event streaming cost.
- **Neural CDEs (2005.08926)** — per-observation latent-trajectory updates for irregular streams.
- **D-NPC (2406.10078)** — dual static/dynamic grids = explicit null-region decomposition.
- **Tangent-Bundle Convolutional Learning (2303.11323)** — convolutions on the velocity (tangent) field; crease = sharp tangent change.
- **Masked Spatio-Temporal Structure Prediction / MaST-Pre (2308.09245)** — learns motion via *cardinality/structure change*, not magnitude; direct support for the structure>magnitude finding.
- **EvHandPose (2303.02862)** — in-domain (event hand): confirms motion alone fails for static hands; edge structure is the signal.

### Baselines / competitors to beat
- **Iterative Variational Contrast-Max (2504.18447)** — the stated closest competitor; background-hypothesis + residual decomposition, >30% detection gain. *Verified.*
- **Event-based Motion Segmentation by Motion Compensation (1904.01293)** + **EMSGC graph-cuts (2012.08730)** — per-event motion-segmentation baselines.
- **Un-EVIMO (2312.00114)** — unsupervised event motion segmentation; the no-label bar to clear.
- **SparseVoxelDet / "No Dense Tensors Needed" (2603.21638)** — engineering baseline for fully-sparse, low-memory per-event inference.
- **Slot Attention / SAVi++ (2006.15055, 2206.07764)** — generic common-fate decomposition baselines.

---

## 5. Honest caveats and the first experiment

**Where the manifold-dynamics framing will NOT help.** The dominant measured constraint is **data diversity, not architecture**: many backbones cluster at 0.47–0.51 F1 LOSO and jump to 0.81–0.92 when all subjects are in-distribution. *None* of families A–F change the number of subjects. Koopman, JEPA, neural-fields, and SSMs all learn dynamics, but if hand-vs-background eigenspaces / latent manifolds are not *separable across unseen hands*, every one of them inherits the same ceiling — possibly worse, since latent-SDE/JEPA/Koopman priors are trained on full batches and overfit fast at 3 subjects (the project already measured ~2× overfit from epoch 0 on the GC model). Second, the prior measured result that a **per-event plane-fit motion feature is non-discriminative** is a direct warning shot for any approach (especially #5, and the local-flow parts of #3) that hopes a per-event geometric scalar separates dense noise from hand — the literature's "structure" signals (orientation, crease, predictability) must be computed over *coherent neighborhoods/time*, or they collapse to the same null. Third, the entire literature operates in micro-windows or frames; **strictly per-event streaming at 1.35M ev/s with ms latency is unpublished** for dense segmentation — the async/low-power production goal is a genuine gap, not a transfer.

**The single most important experiment to run first.** Before building any new dynamics model, run the cheapest static-FP discriminator and measure it *in-distribution first, then LOSO*: take the **classical DMD zero-mode gate (1404.7592/2405.05057)** — or equivalently the **IVC-Max background-hypothesis residual** — as a fixed, non-trained front-end, and measure whether it reduces static-window false positives without killing hand recall. This is a one-to-two-day, training-free test that answers the load-bearing question for *all* of families B–E: **are the static (predictable/zero-mode) and hand (non-zero/residual) sub-manifolds actually separable on this data, and does separability survive LOSO?** If yes, invest in approach #1 or #2. If the gap collapses under LOSO, the result is decisive evidence that the bottleneck is subject diversity, and the highest-EV move shifts to **getting a 4th subject** (and meta-learning, 1703.03400/1703.05175) rather than any manifold-dynamics architecture.
