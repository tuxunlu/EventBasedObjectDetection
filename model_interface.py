import importlib
import inspect
from dataclasses import asdict

import torch
import lightning.pytorch as pl

from loss.loss_funcs import cross_entropy_loss
from loss.distillation import DistillationLoss
from loss.feature_distillation import FeatureDistillationLoss
from torchmetrics.functional.classification import multiclass_accuracy
from utils.metrics.segmentation import binary_iou, binary_dice, boundary_f_score
from configs.sections import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    DataConfig,
)


class ModelInterface(pl.LightningModule):
    SUPPORTED_TASKS = ("classification", "segmentation")

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
        self.feature_distill_loss = None
        if self.task == "segmentation":
            teacher_cfg = getattr(training_cfg, "teacher_config", None)
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
                # Phase B path: BCE + Dice only on cached SAM 2 masks.
                self.seg_loss = DistillationLoss(bce_weight=1.0, dice_weight=1.0)
            self.loss_function = None  # not used in segmentation path
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

    # Caution: self.model.train() is invoked
    # For logging, check document: https://lightning.ai/docs/pytorch/stable/extensions/logging.html#automatic-logging
    # Important clarification for new users:
    # 1. If on_step=True, a _step suffix will be concatenated to metric name. Same for on_epoch, but epoch-level metrics will be automatically averaged using batch_size as weight.
    # 2. If enable_graph=True, .detach() will not be invoked on the value of metric. Could introduce potential error.
    # 3. If sync_dist=True, logger will average metrics across devices. This introduces additional communication overhead, and not suggested for large metric tensors.
    # We can also define customized metrics aggregator for incremental step-level aggregation(to be merged into epoch-level metrics).
    def training_step(self, batch, batch_idx):
        if self.task == "segmentation":
            if self.teacher is not None:
                return self._feature_distillation_step(batch, "train")
            return self._segmentation_step(batch, "train")
        return self._classification_step(batch, "train")

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def validation_step(self, batch, batch_idx):
        if self.task == "segmentation":
            if self.teacher is not None:
                return self._feature_distillation_step(batch, "val")
            return self._segmentation_step(batch, "val")
        return self._classification_step(batch, "val")

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def test_step(self, batch, batch_idx):
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
        Checks if config_dict has all required arguments for cls.__init__
        """
        init_args = dict()
        for name in inspect.signature(cls.__init__).parameters.keys():
            # Skip 'self', '*args', '**kwargs' and parameters with defaults
            if name not in ('self'):
                init_args[name] = config_dict[name]
        provided_keys = set(config_dict.keys())
        missing_keys = init_args.keys() - provided_keys
        
        if missing_keys:
            raise ValueError(f"In dataset initialization, found missing config keys for {cls.__name__}: {missing_keys}")
        
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
