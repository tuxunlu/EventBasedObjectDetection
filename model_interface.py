import importlib
import inspect
from dataclasses import asdict

import torch
import lightning.pytorch as pl

from loss.loss_funcs import cross_entropy_loss
from loss.distillation import DistillationLoss
from loss.feature_distillation import FeatureDistillationLoss
from loss.event_distillation import EventDistillationLoss
from torchmetrics.functional.classification import multiclass_accuracy
from utils.metrics.segmentation import (
    binary_iou, binary_dice, boundary_f_score,
)
from utils.metrics.event_seg import (
    event_f1, event_accuracy, event_precision, event_recall,
    event_pred_to_dense_iou, event_pred_to_dense_iou_clean,
    sweep_counts, f1_from_counts,
)
from configs.sections import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    DataConfig,
)


class ModelInterface(pl.LightningModule):
    SUPPORTED_TASKS = ("classification", "segmentation", "tracking", "event_segmentation")

    def __init__(
        self,
        model_cfg: ModelConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: SchedulerConfig,
        training_cfg: TrainingConfig,
        data_cfg: DataConfig,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.training_cfg = training_cfg
        self.data_cfg = data_cfg
        self.task = getattr(training_cfg, "task", "classification")
        if self.task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"TRAINING.task={self.task!r} not in {self.SUPPORTED_TASKS}"
            )
        self.num_classes = self.data_cfg.dataset.dataset_init_args["num_classes"]

        self.save_hyperparameters(
            {
                "model": asdict(self.model_cfg),
                "optimizer": asdict(self.optimizer_cfg),
                "scheduler": asdict(self.scheduler_cfg),
                "training": asdict(self.training_cfg),
                "data": asdict(self.data_cfg),
            }
        )

        self.model = self.__load_model()
        # The frozen teacher is held in a plain Python list — NOT assigned as a
        # bare ``nn.Module`` attribute. Reason: ``nn.Module.__setattr__`` would
        # otherwise register it as a child, which would:
        #   1. serialize all ~200M teacher params into every Lightning checkpoint
        #      (with save_top_k + save_last that's ~1.6 GB of redundant SAM 2
        #      weights per run);
        #   2. require ``sam2`` importable with matching key layout on every
        #      ``load_from_checkpoint`` call, including for inference-only use;
        #   3. show up under ``self.parameters()`` if a future contributor
        #      accidentally swaps ``self.model.parameters()`` for ``self.parameters()``
        #      in ``configure_optimizers`` (the frozen teacher would then
        #      receive an optimizer entry, harmless but wasteful).
        # The list wrapping sidesteps all three. ``self.teacher`` is exposed
        # as a property; device movement happens lazily in
        # ``_feature_distillation_step`` because Lightning's ``setup()`` runs
        # before the strategy has finished placing the model on its device.
        self._teacher_holder = []
        # Random-but-stable held-out sequence ids for validation preview videos,
        # chosen lazily on the first validation epoch (see _save_validation_previews).
        self._preview_seq_ids = None
        # Static-region temporal-consistency weight (tracking task only): an
        # anti-flicker penalty on prediction changes across consecutive frames
        # in pixels with no events. 0 disables. See _tracking_step.
        self.temporal_consistency_weight = float(
            getattr(training_cfg, "temporal_consistency_weight", 0.0) or 0.0
        )
        self.feature_distill_loss = None
        self.event_loss = None
        if self.task in ("segmentation", "tracking"):
            # The tracking task is Phase-B (cached-mask BCE+Dice) only; the
            # online feature-distillation teacher is a segmentation-only path.
            teacher_cfg = (getattr(training_cfg, "teacher_config", None)
                           if self.task == "segmentation" else None)
            if teacher_cfg:
                self._teacher_holder.append(self.__build_teacher(teacher_cfg))
                self.feature_distill_loss = FeatureDistillationLoss(
                    mask_weight=float(teacher_cfg.get("mask_weight", 1.0)),
                    align_weights=dict(teacher_cfg.get("align_weights", {
                        "low": 0.5, "mid": 1.0, "high": 1.0,
                    })),
                )
                # The cached-mask DistillationLoss is unused on this path.
                self.seg_loss = None
            else:
                # Phase B path: BCE + Dice only on cached SAM 2 masks. The
                # optional pos_weight upweights the (small) foreground class to
                # counter the heavy background bias amplified by all-empty frames.
                pos_weight = getattr(training_cfg, "seg_pos_weight", None)
                self.seg_loss = DistillationLoss(
                    bce_weight=1.0, dice_weight=1.0,
                    pos_weight=float(pos_weight) if pos_weight is not None else None,
                )
            self.loss_function = None  # not used in segmentation path
        elif self.task == "event_segmentation":
            # Per-event sparse path: the model emits one logit per event, supervised
            # directly against the per-event labels (label = mask[y,x]) that the
            # dataloader provides. EventDistillationLoss is the noise/imbalance-aware
            # per-event loss (SCE + per-sample Lovász-hinge + optional Focal-Tversky).
            event_cfg = dict(getattr(training_cfg, "event_loss_config", None) or {})
            self.event_loss = EventDistillationLoss(**event_cfg)
            self.seg_loss = None
            self.loss_function = None
            # Tier-3 decision rebalancing. The fixed-0.5 cut over-predicts foreground
            # (pred 11.6% vs GT 9.9%); we sweep the threshold on val each epoch and
            # report the best-F1 operating point + a spatial-coherence-cleaned IoU.
            eval_cfg = dict(getattr(training_cfg, "event_eval_config", None) or {})
            lo = float(eval_cfg.get("thr_min", 0.05))
            hi = float(eval_cfg.get("thr_max", 0.95))
            steps = int(eval_cfg.get("thr_steps", 19))
            self._cal_thresholds = torch.linspace(lo, hi, max(2, steps))
            self._cal_threshold = 0.5                  # calibrated operating point
            self._cal_tp = self._cal_fp = self._cal_fn = None
            # Post-hoc spatial cleanup applied to the rasterized prediction for the
            # *_iou_clean diagnostic (and reused at inference). off by default.
            self._clean_open_ksize = int(eval_cfg.get("clean_open_ksize", 3))
            self._clean_keep_largest = bool(eval_cfg.get("clean_keep_largest", False))
            self._clean_min_area = int(eval_cfg.get("clean_min_area", 0))
            self._clean_enabled = bool(eval_cfg.get("clean_enabled", True))
        else:
            self.loss_function = self.__configure_loss()

    @property
    def teacher(self):
        return self._teacher_holder[0] if self._teacher_holder else None

    def _ensure_teacher_on_device(self, ref: torch.Tensor):
        """Lazily move the teacher onto the same device as ``ref``.

        Called from the feature-distillation step instead of relying on
        ``setup()`` — Lightning's lifecycle places the model on its strategy
        device *after* ``setup()`` finishes, so ``self.device`` may still be
        ``cpu`` at the point ``setup()`` runs. Doing the move here is
        bulletproof and the conditional is a single tensor-device compare per
        step (zero-cost after the first call).
        """
        teacher = self.teacher
        if teacher is None:
            return None
        try:
            t_dev = next(teacher.parameters()).device
        except StopIteration:
            return teacher
        if t_dev != ref.device:
            self._teacher_holder[0] = teacher.to(ref.device)
            teacher = self._teacher_holder[0]
        return teacher

    def forward(self, x):
        return self.model(x)

    # For all these hook functions like on_XXX_<epoch|batch>_<end|start>(),
    # check document: https://lightning.ai/docs/pytorch/LTS/common/lightning_module.html
    # Epoch level training logging
    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_start(self):
        # Reset the per-threshold calibration accumulators for the event task.
        if self.task == "event_segmentation":
            self._cal_tp = self._cal_fp = self._cal_fn = None

    def _calibrate_event_threshold(self):
        """Pick the best-F1 decision threshold from the val-epoch count sweep.

        Aggregates the per-threshold ``(tp, fp, fn)`` accumulated across the epoch
        (and across DDP ranks), logs the best F1 and its threshold, and stores the
        operating point on ``self._cal_threshold`` for the preview renderer / export.
        """
        if self._cal_tp is None:
            return
        tp, fp, fn = self._cal_tp, self._cal_fp, self._cal_fn
        # Sum the counts across ranks so the calibration uses the whole val set.
        # all_gather adds a leading world dim; only reduce when actually distributed
        # (in single-process it would otherwise collapse the threshold axis).
        ws = int(getattr(self.trainer, "world_size", 1) or 1)
        if ws > 1:
            try:
                tp = self.all_gather(tp).sum(dim=0)
                fp = self.all_gather(fp).sum(dim=0)
                fn = self.all_gather(fn).sum(dim=0)
            except Exception:  # noqa: BLE001 - no strategy / not initialized
                pass
        f1s = f1_from_counts(tp, fp, fn)
        best = int(torch.argmax(f1s).item())
        self._cal_threshold = float(self._cal_thresholds[best].item())
        bs = 1
        self.log("val_event_f1_best", f1s[best], on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=False, batch_size=bs)
        self.log("val_event_thr", torch.tensor(self._cal_threshold),
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=False, batch_size=bs)

    def on_validation_epoch_end(self):
        """Render one held-out preview video per rank (segmentation/tracking).

        Every DDP rank renders exactly one sequence, so the visualization work
        is symmetric across the process group: no rank sits idle and races ahead
        to the next collective (metric all-reduce / ModelCheckpoint) while
        another is still encoding. That symmetry — not a barrier — is what keeps
        the ranks in lockstep, which is why no explicit barrier is needed here.
        The earlier rank-0-only version deadlocked precisely because the work
        was asymmetric: ranks 1..N-1 reached the next collective and blocked
        waiting for rank 0, which was still encoding videos.

        For the tracking task the model is run *statefully* across the whole
        sequence (memory + prev-mask carried frame to frame), so the preview
        shows the actual temporal behavior — i.e. whether the jitter is gone. The
        event_segmentation task renders the same Events | Teacher (GT) |
        Prediction triptych, but per-frame over the sparse event-site stream (the
        per-event prediction is rasterized to a dense mask for the panel).

        Skipped during the sanity-check pass. Visualization must never take
        training down, so any failure is caught and logged; a rank that fails
        just renders nothing and still meets the others at the next collective.
        """
        if self.task not in ("segmentation", "tracking", "event_segmentation"):
            return
        # Threshold calibration runs first and unconditionally for the event task
        # (it is a collective via all_gather, so every rank must reach it together,
        # and it must not depend on the preview count).
        if self.task == "event_segmentation" \
                and not getattr(self.trainer, "sanity_checking", False):
            self._calibrate_event_threshold()
        n = int(getattr(self.training_cfg, "val_preview_count", 0) or 0)
        if n <= 0:
            return
        if getattr(self.trainer, "sanity_checking", False):
            return
        try:
            if self.task == "event_segmentation":
                self._save_event_validation_previews(n)
            else:
                self._save_validation_previews(n)
        except Exception as exc:  # noqa: BLE001 - never crash training on viz
            print(f"[viz] validation preview skipped ({type(exc).__name__}: {exc})")

    def _select_preview_sequence(self, val_set, n: int, max_frames: int = 600):
        """Pick this rank's preview sequence + strided frame positions, or None.

        Shared by the dense (segmentation/tracking) and sparse (event) preview
        renderers so they select sequences identically: a deterministic shuffle
        of all val sequences — seeded the same on every rank so the ordering
        matches without any cross-rank communication — then this rank renders the
        sequence at its global-rank slot within the first ``n`` eligible. The
        in-sequence frame positions (``val_set.index`` is in ``(seq, frame)``
        order) are strided down to ``max_frames`` to keep a preview cheap.
        Returns ``(s_idx, positions)`` or ``None`` when this rank has nothing to
        render or the val set exposes no sequence index.
        """
        from math import ceil

        if val_set is None or not hasattr(val_set, "sequences") \
                or not hasattr(val_set, "index"):
            print("[viz] validation set has no sequence index; skipping previews")
            return None

        if self._preview_seq_ids is None:
            import random as _random
            rng = _random.Random(int(getattr(self.training_cfg, "seed", 0)))
            ids = list(range(len(val_set.sequences)))
            rng.shuffle(ids)
            self._preview_seq_ids = ids

        # One video per rank: ranks beyond the available sequences (or the cap)
        # render nothing and just proceed to the next collective.
        rank = int(getattr(self.trainer, "global_rank", 0) or 0)
        eligible = self._preview_seq_ids[:n]
        if rank >= len(eligible):
            return None
        s_idx = eligible[rank]

        positions = [i for i, (si, _f) in enumerate(val_set.index) if si == s_idx]
        if not positions:
            return None
        if len(positions) > max_frames:
            stride = ceil(len(positions) / max_frames)
            positions = positions[::stride]
        return s_idx, positions

    @torch.no_grad()
    def _save_validation_previews(self, n: int):
        """Render this rank's single held-out preview sequence → a 3-panel MP4.

        Each rank renders the sequence at its global-rank slot in a
        deterministically shuffled list (so the same clip per rank re-renders
        every epoch, letting you watch it sharpen). ``val_preview_count`` (``n``)
        caps how many sequences are eligible across the group; with one video
        per rank, at most ``min(n, world_size, num_val_sequences)`` are written.
        Panels are Events | Teacher (GT) Mask | Student Prediction, matching the
        offline teacher preview in ``data/sam2_pseudo_labels.py``. Long clips are
        strided down to a frame cap so a preview stays cheap.
        """
        from pathlib import Path

        from utils.metrics.segmentation import binary_iou
        from utils.viz.val_preview import (
            events_panel_from_voxel, infer_fps, write_triptych_video,
        )

        dm = getattr(self.trainer, "datamodule", None)
        val_set = getattr(dm, "validation_set", None) if dm is not None else None
        picked = self._select_preview_sequence(val_set, n)
        if picked is None:
            return
        s_idx, positions = picked

        out_dir = Path(self.trainer.log_dir or ".") / "val_previews"
        out_dir.mkdir(parents=True, exist_ok=True)
        rank = int(getattr(self.trainer, "global_rank", 0) or 0)

        # Single-frame accessor: the clip dataset returns clips from __getitem__,
        # so it exposes frame_sample() for one-frame access; the plain seg dataset
        # returns single frames from __getitem__ directly.
        frame_getter = getattr(val_set, "frame_sample", None) or val_set.__getitem__
        # Tracking models expose step(frame, state) for stateful streaming; run
        # the whole sequence through one carried state so the preview reflects
        # the temporal memory (per-frame call for the stateless seg model).
        streaming = hasattr(self.model, "step")

        was_training = self.model.training
        self.model.eval()
        try:
            event_panels, gt_masks, pred_masks, ious = [], [], [], []
            state = None
            for pos in positions:
                sample = frame_getter(pos)
                voxel, mask = sample[0], sample[1]  # (C,H,W), (H,W)
                inp = voxel.unsqueeze(0).to(self.device)
                if streaming:
                    logits, state = self.model.step(inp, state)
                else:
                    logits = self.model(inp)
                prob = torch.sigmoid(logits)[0, 0]
                pred = (prob > 0.5).float()

                event_panels.append(events_panel_from_voxel(voxel))
                gt_np = (mask.detach().cpu().numpy() * 255).astype("uint8")
                gt_masks.append(gt_np)
                pred_masks.append((pred.cpu().numpy() * 255).astype("uint8"))
                # Per-frame IoU only meaningful where the GT has foreground.
                if float(mask.sum()) > 0:
                    ious.append(float(binary_iou(
                        logits.detach(), mask.unsqueeze(0).to(self.device))))
                else:
                    ious.append(None)

            seq_name = val_set.sequences[s_idx].name
            fps = self._infer_preview_fps(val_set, s_idx, infer_fps)
            out_path = out_dir / f"epoch{self.current_epoch:03d}_{seq_name}.mp4"
            write_triptych_video(
                event_panels, gt_masks, pred_masks, out_path,
                fps=fps, title=seq_name, per_frame_iou=ious,
            )
            print(f"[viz] rank{rank} wrote {out_path} ({len(positions)} frames, "
                  f"{fps:.1f} fps)")
        finally:
            if was_training:
                self.model.train()

    @staticmethod
    def _infer_preview_fps(val_set, s_idx, infer_fps):
        """Best-effort fps from the sequence's FLIR timestamps; 30 on failure."""
        try:
            handle = val_set._get_handle(val_set.sequences[s_idx])
            return infer_fps(handle.get("flir_t"))
        except Exception:  # noqa: BLE001
            return 30.0

    @torch.no_grad()
    def _save_event_validation_previews(self, n: int):
        """Render this rank's held-out preview sequence for the sparse event path.

        Per-event 3-panel video (the in-training analog of
        ``tools/viz_per_event_labels_video.py``), painted directly on the event
        sites — no rasterization:

          1. **Prediction** — events green where the model predicts foreground.
          2. **GT label**   — events green where ``label == 1`` (``mask[y,x]``).
          3. **Context**    — events (gray) + the GT FLIR mask outline (yellow).

        Each frame's title carries the per-event F1, so you can watch prediction vs
        label converge over a held-out sequence frame by frame.
        """
        from pathlib import Path

        import cv2

        from data.sparse_event_collate import collate_sparse_events
        from utils.metrics.event_seg import event_f1
        from utils.viz.val_preview import event_pred_triptych, infer_fps

        dm = getattr(self.trainer, "datamodule", None)
        val_set = getattr(dm, "validation_set", None) if dm is not None else None
        picked = self._select_preview_sequence(val_set, n)
        if picked is None:
            return
        s_idx, positions = picked

        out_dir = Path(self.trainer.log_dir or ".") / "val_previews"
        out_dir.mkdir(parents=True, exist_ok=True)
        rank = int(getattr(self.trainer, "global_rank", 0) or 0)
        seq_name = val_set.sequences[s_idx].name
        fps = self._infer_preview_fps(val_set, s_idx, infer_fps)
        out_path = out_dir / f"epoch{self.current_epoch:03d}_{seq_name}.mp4"

        was_training = self.model.training
        self.model.eval()
        writer = None
        try:
            for k, pos in enumerate(positions):
                coords, feats, times, labels, dense_mask, meta = val_set[pos]
                if coords.shape[0] == 0:
                    continue
                batch = collate_sparse_events(
                    [(coords, feats, times, labels, dense_mask, meta)]
                ).to(self.device)
                logits = self(batch)                         # (N,) per-event logits
                h, w = batch.height, batch.width
                thr = float(getattr(self, "_cal_threshold", 0.5))
                pred = (torch.sigmoid(logits.detach()) > thr).float()
                f1 = float(event_f1(logits.detach(), batch.labels, threshold=thr)) \
                    if float(dense_mask.sum()) > 0 else None

                tag = f"{seq_name}  f{meta.get('frame_index', k):04d}  ({k + 1}/{len(positions)})"
                frame = event_pred_triptych(coords, feats, labels, pred, dense_mask,
                                            h, w, tag=tag, f1=f1)
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, fps,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            if writer is not None:
                writer.release()
                print(f"[viz] rank{rank} wrote {out_path} ({len(positions)} frames, "
                      f"{fps:.1f} fps)")
        finally:
            if writer is not None:
                writer.release()
            if was_training:
                self.model.train()

    # Caution: self.model.train() is invoked
    # For logging, check document: https://lightning.ai/docs/pytorch/stable/extensions/logging.html#automatic-logging
    # Important clarification for new users:
    # 1. If on_step=True, a _step suffix will be concatenated to metric name. Same for on_epoch, but epoch-level metrics will be automatically averaged using batch_size as weight.
    # 2. If enable_graph=True, .detach() will not be invoked on the value of metric. Could introduce potential error.
    # 3. If sync_dist=True, logger will average metrics across devices. This introduces additional communication overhead, and not suggested for large metric tensors.
    # We can also define customized metrics aggregator for incremental step-level aggregation(to be merged into epoch-level metrics).
    def training_step(self, batch, batch_idx):
        if self.task == "tracking":
            return self._tracking_step(batch, "train")
        if self.task == "event_segmentation":
            return self._event_segmentation_step(batch, "train")
        if self.task == "segmentation":
            if self.teacher is not None:
                return self._feature_distillation_step(batch, "train")
            return self._segmentation_step(batch, "train")
        return self._classification_step(batch, "train")

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def validation_step(self, batch, batch_idx):
        if self.task == "tracking":
            return self._tracking_step(batch, "val")
        if self.task == "event_segmentation":
            return self._event_segmentation_step(batch, "val")
        if self.task == "segmentation":
            if self.teacher is not None:
                return self._feature_distillation_step(batch, "val")
            return self._segmentation_step(batch, "val")
        return self._classification_step(batch, "val")

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def test_step(self, batch, batch_idx):
        if self.task == "tracking":
            return self._tracking_step(batch, "test")
        if self.task == "event_segmentation":
            return self._event_segmentation_step(batch, "test")
        if self.task == "segmentation":
            if self.teacher is not None:
                return self._feature_distillation_step(batch, "test")
            return self._segmentation_step(batch, "test")
        return self._classification_step(batch, "test")

    def _classification_step(self, batch, stage):
        x, labels = batch
        logits = self(x)
        loss = self.loss_function(logits, labels, stage)

        top1 = multiclass_accuracy(logits, labels, num_classes=self.num_classes,
                                   average='micro', top_k=1)
        top5 = multiclass_accuracy(logits, labels, num_classes=self.num_classes,
                                   average='micro', top_k=5)
        self.log(f'{stage}_top1_acc', top1, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=x.shape[0])
        self.log(f'{stage}_top5_acc', top5, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=x.shape[0])

        return {'loss': loss, 'pred': logits, 'ground_truth': labels}

    def _segmentation_step(self, batch, stage):
        # HandEventDataset yields (voxel(B,H,W), mask(H,W), meta_dict). Default
        # collate gives voxel(N,B,H,W), mask(N,H,W), meta_dict_of_lists.
        if len(batch) == 3:
            voxel, mask, _meta = batch
        else:
            voxel, mask = batch  # tolerate a 2-tuple form
        logits = self(voxel)  # (N, 1, H, W)

        terms = self.seg_loss(logits, mask)
        loss = terms["total"]

        bs = voxel.shape[0]
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=bs)
        self.log(f'{stage}_bce_loss', terms["bce"], on_step=True, on_epoch=True,
                 prog_bar=False, sync_dist=True, batch_size=bs)
        self.log(f'{stage}_dice_loss', terms["dice"], on_step=True, on_epoch=True,
                 prog_bar=False, sync_dist=True, batch_size=bs)

        # Cheap eval metric every step; boundary-F only during val/test to keep
        # training step fast.
        iou = binary_iou(logits.detach(), mask)
        self.log(f'{stage}_iou', iou, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=bs)
        if stage != "train":
            dice = binary_dice(logits.detach(), mask)
            bf = boundary_f_score(logits.detach(), mask, d_tolerance=2)
            self.log(f'{stage}_dice', dice, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True, batch_size=bs)
            self.log(f'{stage}_boundary_f', bf, on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=True, batch_size=bs)

        return {'loss': loss, 'pred': logits, 'ground_truth': mask}

    def _tracking_step(self, batch, stage):
        """Temporal tracking step over a clip.

        Batch contract (from ``HandEventClipDataset`` + default collate):
        ``(voxel(N,T,C,H,W), mask(N,T,H,W), meta)``. The model carries a
        recurrent memory + previous-mask feedback across the ``T`` frames and
        returns ``logits(N,T,1,H,W)``. Supervision is per-frame BCE+Dice (the
        clip flattened to ``N*T`` independent frames), plus an optional
        static-region temporal-consistency penalty that directly attacks the
        frame-to-frame mask jitter this model exists to fix.
        """
        if len(batch) == 3:
            voxel, mask, _meta = batch
        else:
            voxel, mask = batch  # tolerate a 2-tuple form
        # voxel (N,T,C,H,W), mask (N,T,H,W)
        logits = self(voxel)            # (N, T, 1, H, W)
        n, t = logits.shape[0], logits.shape[1]
        h, w = logits.shape[-2:]

        # Flatten the clip's frames into the batch dim so the per-pixel
        # BCE+Dice (and the IoU/Dice/boundary metrics) apply per-frame.
        logits_flat = logits.reshape(n * t, logits.shape[2], h, w)
        mask_flat = mask.reshape(n * t, h, w)

        terms = self.seg_loss(logits_flat, mask_flat)
        loss = terms["total"]

        # Static-region anti-flicker: penalize change in predicted probability
        # across consecutive frames at pixels that saw no events in the later
        # frame (which therefore *should* stay put). Targets the exact failure
        # mode — a still limb's mask blinking — without over-smoothing moving
        # boundaries, where events are present and the term is inactive.
        if self.temporal_consistency_weight > 0.0 and t >= 2:
            prob = torch.sigmoid(logits[:, :, 0])      # (N, T, H, W)
            dens = voxel.abs().sum(dim=2)              # (N, T, H, W) event activity
            static = (dens[:, 1:] <= 0).float()
            diff = (prob[:, 1:] - prob[:, :-1]).abs()
            tc = (diff * static).sum() / static.sum().clamp(min=1.0)
            loss = loss + self.temporal_consistency_weight * tc
            self.log(f'{stage}_tc_loss', tc, on_step=True, on_epoch=True,
                     prog_bar=False, sync_dist=True, batch_size=n)

        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=n)
        self.log(f'{stage}_bce_loss', terms["bce"], on_step=True, on_epoch=True,
                 prog_bar=False, sync_dist=True, batch_size=n)
        self.log(f'{stage}_dice_loss', terms["dice"], on_step=True, on_epoch=True,
                 prog_bar=False, sync_dist=True, batch_size=n)

        iou = binary_iou(logits_flat.detach(), mask_flat)
        self.log(f'{stage}_iou', iou, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=n)
        if stage != "train":
            dice = binary_dice(logits_flat.detach(), mask_flat)
            bf = boundary_f_score(logits_flat.detach(), mask_flat, d_tolerance=2)
            # Flicker: per-pixel fraction of binary on/off flips between
            # consecutive frames, averaged over clips. The headline temporal
            # metric — lower is steadier. (0 for degenerate T==1 clips.)
            pred = (torch.sigmoid(logits[:, :, 0]) > 0.5).float()  # (N,T,H,W)
            if t >= 2:
                flicker = (pred[:, 1:] != pred[:, :-1]).float().mean()
            else:
                flicker = torch.zeros((), device=pred.device)
            self.log(f'{stage}_dice', dice, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True, batch_size=n)
            self.log(f'{stage}_boundary_f', bf, on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=True, batch_size=n)
            self.log(f'{stage}_flicker', flicker, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True, batch_size=n)

        return {'loss': loss, 'pred': logits, 'ground_truth': mask}

    def _feature_distillation_step(self, batch, stage):
        """Online feature distillation: SAM 2 image encoder on RGB → student features.

        Batch contract: ``(voxel(N,B,H,W), mask(N,H,W), rgb(N,3,H,W), meta)``
        produced by ``HandEventDataset(provide_rgb=True)`` + default collate.

        The student must expose ``forward_with_features(voxel) -> (logits, feats_dict)``
        with keys aligned to the teacher's. ``EventTinySeg`` matches this contract.
        """
        if len(batch) != 4:
            raise ValueError(
                f"feature-distillation expects 4-tuple (voxel, mask, rgb, meta); "
                f"got {len(batch)}-tuple. Did you set DATA.dataset.dataset_init_args."
                f"provide_rgb: True ?"
            )
        voxel, mask, rgb, _meta = batch

        if not hasattr(self.model, "forward_with_features"):
            raise TypeError(
                f"model {type(self.model).__name__} has no forward_with_features(); "
                "use EventTinySeg or another distillation-aware model when "
                "teacher_config is set."
            )
        logits, student_feats = self.model.forward_with_features(voxel)

        # Teacher: frozen, no grad, no autocast (let SAM 2 use its own dtype).
        # Lazy device move on first call — see _ensure_teacher_on_device for why.
        teacher = self._ensure_teacher_on_device(rgb)
        with torch.no_grad():
            teacher_feats = teacher(rgb)

        terms = self.feature_distill_loss(
            mask_logits=logits,
            student_feats=student_feats,
            teacher_mask=mask,
            teacher_feats=teacher_feats,
        )
        loss = terms["total"]

        bs = voxel.shape[0]
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=bs)
        self.log(f'{stage}_bce_loss', terms["bce"], on_step=True, on_epoch=True,
                 prog_bar=False, sync_dist=True, batch_size=bs)
        self.log(f'{stage}_dice_loss', terms["dice"], on_step=True, on_epoch=True,
                 prog_bar=False, sync_dist=True, batch_size=bs)
        for k in ("align_low", "align_mid", "align_high", "align_total"):
            if k in terms:
                self.log(f'{stage}_{k}', terms[k], on_step=True, on_epoch=True,
                         prog_bar=False, sync_dist=True, batch_size=bs)

        # IoU here is at the student's output stride (e.g. stride-4 for
        # EventTinySeg); the mask target is downsampled inside the loss but
        # logits are still student-resolution, so we downsample the mask
        # ourselves for the metric.
        import torch.nn.functional as F
        with torch.no_grad():
            tgt = F.interpolate(mask.unsqueeze(1).float(),
                                size=logits.shape[-2:],
                                mode="nearest").squeeze(1)
            iou = binary_iou(logits.detach(), tgt)
            self.log(f'{stage}_iou', iou, on_step=True, on_epoch=True,
                     prog_bar=True, sync_dist=True, batch_size=bs)
            if stage != "train":
                dice = binary_dice(logits.detach(), tgt)
                bf = boundary_f_score(logits.detach(), tgt, d_tolerance=2)
                self.log(f'{stage}_dice', dice, on_step=False, on_epoch=True,
                         prog_bar=True, sync_dist=True, batch_size=bs)
                self.log(f'{stage}_boundary_f', bf, on_step=False, on_epoch=True,
                         prog_bar=False, sync_dist=True, batch_size=bs)

        return {'loss': loss, 'pred': logits, 'ground_truth': mask}

    def _event_segmentation_step(self, batch, stage):
        """Per-event sparse path: ``SparseEventBatch`` -> one logit per event -> loss.

        ``batch`` is a ``SparseEventBatch`` (from ``data/sparse_event_collate.py``).
        The model returns one logit per **event**, row-aligned to ``batch.labels``
        (the per-event ``mask[y,x]`` target the dataloader provides). Empty windows
        (no events) are skipped by returning ``None``.
        """
        logits = self(batch)                 # (N,) per-event logits
        labels = batch.labels
        if labels.numel() == 0:
            return None                      # nothing to supervise this batch
        bs = batch.batch_size

        terms = self.event_loss(logits, labels, batch_idx=batch.batch_idx)
        loss = terms["total"]
        # Auxiliary dense-shape head loss (TRAIN ONLY). The model stashes a coarse
        # (B,1,G,G) occupancy logit map on ``self.model._aux_logits``; supervise it
        # against the avg-pooled teacher mask (BCE + soft-Dice). This teaches the
        # backbone the global hand-blob extent so the per-event head can reject
        # locally-identical background. Zero effect on val/test loss and (with
        # aux_gather off) zero inference cost — the map is dropped at export.
        aux_logits = getattr(self.model, "_aux_logits", None)
        if aux_logits is not None and stage == "train":
            import torch.nn.functional as F
            # IMPORTANT: build the GxG occupancy target from the AUGMENTED per-event
            # labels (which travel with the events under hflip/affine), NOT from
            # batch.dense_mask — the dataset leaves dense_mask UN-augmented (dataset
            # L252) while coords are warped, so dense_mask is spatially misaligned on
            # ~half the train batches. Per cell: foreground fraction over supervised
            # (label>=0) events; empty cells are masked out of the loss.
            G = aux_logits.shape[-1]
            H, W = int(batch.height), int(batch.width)
            xb = batch.coords[:, 0].long(); yb = batch.coords[:, 1].long()
            bb = batch.batch_idx.long()
            keep = labels >= 0
            gy = (yb * G // H).clamp(0, G - 1); gx = (xb * G // W).clamp(0, G - 1)
            cell = (bb * G + gy) * G + gx
            nbins = bs * G * G
            ck = cell[keep]
            fgc = aux_logits.new_zeros(nbins).index_add_(0, ck, (labels[keep] > 0.5).to(aux_logits.dtype))
            totc = aux_logits.new_zeros(nbins).index_add_(0, ck, aux_logits.new_ones(ck.shape[0]))
            tgt = (fgc / totc.clamp(min=1.0)).view(bs, 1, G, G)    # per-cell foreground fraction
            cmask = (totc > 0).view(bs, 1, G, G).to(aux_logits.dtype)  # supervise only non-empty cells
            denom = cmask.sum().clamp(min=1.0)
            bce = (F.binary_cross_entropy_with_logits(aux_logits, tgt, reduction="none") * cmask).sum() / denom
            p = torch.sigmoid(aux_logits) * cmask; t = tgt * cmask
            dice = 1.0 - (2.0 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0)
            w = float(getattr(self.model, "aux_shape_weight", 0.3))
            aux = w * (0.5 * bce + 0.5 * dice)
            loss = loss + aux
            self.log(f'{stage}_aux_shape_loss', aux, on_step=True, on_epoch=True,
                     prog_bar=False, sync_dist=True, batch_size=bs)
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=bs)
        # Log every active sub-loss (bce/rce/gce/gjs/nrdice/asl/lovasz/...) so the
        # ablation across configs is legible.
        for name, value in terms.items():
            if name == "total":
                continue
            self.log(f'{stage}_{name}_loss', value, on_step=True, on_epoch=True,
                     prog_bar=False, sync_dist=True, batch_size=bs)

        det = logits.detach()
        f1 = event_f1(det, labels)
        acc = event_accuracy(det, labels)
        self.log(f'{stage}_event_f1', f1, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=bs)
        self.log(f'{stage}_event_acc', acc, on_step=True, on_epoch=True,
                 prog_bar=False, sync_dist=True, batch_size=bs)
        # Precision / recall separately so the false-positive flood (the symptom:
        # other-motion events leaking into the hand class) is directly visible, not
        # hidden inside F1. The trimap-ignored (label<0) events are dropped by the
        # metric's count helper.
        self.log(f'{stage}_event_precision', event_precision(det, labels),
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bs)
        self.log(f'{stage}_event_recall', event_recall(det, labels),
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bs)

        # Accumulate the per-threshold counts over the val epoch for post-hoc
        # threshold calibration (computed in on_validation_epoch_end).
        if stage == "val":
            tp, fp, fn = sweep_counts(det, labels, self._cal_thresholds)
            if self._cal_tp is None:
                self._cal_tp = tp.clone(); self._cal_fp = fp.clone(); self._cal_fn = fn.clone()
            else:
                self._cal_tp += tp; self._cal_fp += fp; self._cal_fn += fn

        # Rasterized dense IoU: scatter per-event predictions onto a dense grid and
        # reuse binary_iou vs the teacher mask. val/test only, to keep train lean.
        if stage != "train":
            iou = event_pred_to_dense_iou(
                batch.coords, det, batch.batch_idx,
                batch.dense_mask, bs, batch.height, batch.width,
            )
            self.log(f'{stage}_iou', iou, on_step=True, on_epoch=True,
                     prog_bar=True, sync_dist=True, batch_size=bs)
            # Spatial-coherence-cleaned IoU: removes isolated FP speckle from the
            # rasterized prediction. The gap vs val_iou quantifies how much of the
            # error is incoherent noise the cleanup can remove for free at inference.
            if self._clean_enabled:
                iou_clean = event_pred_to_dense_iou_clean(
                    batch.coords, det, batch.batch_idx,
                    batch.dense_mask, bs, batch.height, batch.width,
                    threshold=0.5, open_ksize=self._clean_open_ksize,
                    keep_largest=self._clean_keep_largest, min_area=self._clean_min_area,
                )
                self.log(f'{stage}_iou_clean', iou_clean, on_step=False, on_epoch=True,
                         prog_bar=False, sync_dist=True, batch_size=bs)
        return {'loss': loss, 'pred': logits, 'ground_truth': labels}

    def __build_teacher(self, teacher_cfg):
        """Construct the configured online teacher. Currently SAM 2 only."""
        kind = teacher_cfg.get("type", "sam2_image_encoder")
        if kind != "sam2_image_encoder":
            raise ValueError(f"unsupported teacher type: {kind!r}")
        from model.sam2_teacher import Sam2ImageEncoderTeacher
        return Sam2ImageEncoderTeacher(
            checkpoint_path=teacher_cfg["checkpoint"],
            config_name=teacher_cfg["config"],
            input_size=int(teacher_cfg.get("input_size", 1024)),
        )

    def configure_optimizers(self):
        # https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html
        try:
            optimizer_class = getattr(torch.optim, self.optimizer_cfg.name)
        except AttributeError as exc:
            raise ValueError(f"Invalid optimizer: OPTIMIZER.{self.optimizer_cfg.name}") from exc

        optimizer_arguments = dict(self.optimizer_cfg.arguments or {})
        optimizer_instance = optimizer_class(params=self.model.parameters(), **optimizer_arguments)

        learning_rate_scheduler_cfg = self.scheduler_cfg.learning_rate
        if not learning_rate_scheduler_cfg.enabled:
            return [optimizer_instance]

        try:
            scheduler_class = getattr(torch.optim.lr_scheduler, learning_rate_scheduler_cfg.name)
        except AttributeError as exc:
            raise ValueError(
                f"Invalid learning rate scheduler: SCHEDULER.learning_rate.{learning_rate_scheduler_cfg.name}."
            ) from exc

        scheduler_arguments = dict(learning_rate_scheduler_cfg.arguments or {})
        scheduler_instance = scheduler_class(optimizer=optimizer_instance, **scheduler_arguments)

        return [optimizer_instance], [scheduler_instance]

    def __configure_loss(self):
        def loss_func(preds, labels, stage):
            CE_loss = 1.0 * cross_entropy_loss(pred=preds, gt=labels)
            self.log(f'{stage}_CE_loss', CE_loss, on_step=True, on_epoch=True, prog_bar=True)

            final_loss = CE_loss
            self.log(f'{stage}_loss', final_loss, on_step=True, on_epoch=True, prog_bar=True)

            return final_loss

        return loss_func
    
    @staticmethod
    def filter_init_args(cls, config_dict):
        """
        Build the init kwargs for ``cls`` from ``config_dict``.

        A parameter is pulled from the config when present. Parameters that have a
        default are optional (their default is used when the config omits them) so
        that adding a new defaulted ``__init__`` arg to a model/dataset does NOT
        require every pre-existing config to list it. Only parameters WITHOUT a
        default are required, and ``self`` / ``*args`` / ``**kwargs`` are skipped.
        """
        init_args = dict()
        missing_required = []
        for name, param in inspect.signature(cls.__init__).parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                continue
            if name in config_dict:
                init_args[name] = config_dict[name]
            elif param.default is inspect.Parameter.empty:
                missing_required.append(name)
            # else: defaulted and not provided -> let cls use its own default.

        if missing_required:
            raise ValueError(
                f"In {cls.__name__} initialization, found missing required config "
                f"keys: {missing_required}"
            )

        return init_args

    def __load_model(self):
        file_name = self.model_cfg.file_name
        class_name = self.model_cfg.class_name
        if class_name is None:
            raise ValueError("MODEL.class_name must be specified in the configuration.")
        if file_name is None:
            raise ValueError("MODEL.file_name must be specified in the configuration.")
        try:
            model_class = getattr(importlib.import_module('model.' + file_name, package=__package__), class_name)
        except Exception:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {file_name}.{class_name}!')

        model_init_kwargs = self.model_cfg.model_init_args
        # Only validate highest level keyword arguments. This is a tradeoff between flexibility and rigour.
        # If you want to enable recursive validation for every keyword including nested ones, define them as template
        # in config schema instead of using raw dictionary.
        # We assume that dataset_kwargs is a superset of data_class's init arg set.
        filtered_model_init_kwargs = self.filter_init_args(cls=model_class, config_dict=model_init_kwargs)
        model = model_class(**filtered_model_init_kwargs)
        if self.training_cfg.use_compile:
            model = torch.compile(model)
        return model
