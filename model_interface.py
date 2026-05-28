import importlib
import inspect
from dataclasses import asdict

import torch
import lightning.pytorch as pl

from loss.loss_funcs import cross_entropy_loss
from loss.distillation import DistillationLoss
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
        if self.task == "segmentation":
            # Phase B: BCE + Dice only; Phase C will turn on the motion /
            # temporal / feature terms via the same module.
            self.seg_loss = DistillationLoss(bce_weight=1.0, dice_weight=1.0)
            self.loss_function = None  # not used in segmentation path
        else:
            self.loss_function = self.__configure_loss()

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
            return self._segmentation_step(batch, "train")
        return self._classification_step(batch, "train")

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def validation_step(self, batch, batch_idx):
        if self.task == "segmentation":
            return self._segmentation_step(batch, "val")
        return self._classification_step(batch, "val")

    # Caution: self.model.eval() is invoked and this function executes within a <with torch.no_grad()> context
    def test_step(self, batch, batch_idx):
        if self.task == "segmentation":
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
