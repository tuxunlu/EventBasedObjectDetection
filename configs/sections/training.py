from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from configs.config_tracker import TrackedConfigMixin


@dataclass
class TrainingConfig(TrackedConfigMixin):
    deterministic: bool = False
    use_compile: bool = False
    inference_mode: bool = False
    seed: int = 42
    max_epochs: int = 1
    # Task selector consumed by ModelInterface. "classification" preserves the
    # original Cifar template flow; "segmentation" switches to a 3-tuple
    # (voxel, mask, meta) batch contract with BCE+Dice supervision.
    task: str = "classification"
    # Optional online teacher for the segmentation task. When set, the
    # ModelInterface loads the teacher in __init__, switches to the 4-tuple
    # (voxel, mask, rgb, meta) batch contract, and runs FeatureDistillationLoss
    # instead of the cached-mask-only DistillationLoss.
    #
    # Schema:
    #   type:        currently only "sam2_image_encoder"
    #   checkpoint:  path to SAM 2 .pt file
    #   config:      SAM 2 Hydra config name (e.g. "sam2_hiera_l.yaml")
    #   input_size:  side length to resize RGB to before the teacher (default 1024)
    #   align_weights: dict of per-level cosine-alignment weights
    #                  e.g. {"low": 0.5, "mid": 1.0, "high": 1.0}
    #   mask_weight: scalar weight on the BCE+Dice term (default 1.0)
    teacher_config: Optional[Dict[str, Any]] = None
