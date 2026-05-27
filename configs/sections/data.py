from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from configs.config_tracker import TrackedConfigMixin


@dataclass
class DatasetConfig(TrackedConfigMixin):
    file_name: str = "cifar10"
    class_name: str = "Cifar10"
    dataset_init_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataloaderConfig(TrackedConfigMixin):
    batch_size: int = 32
    test_batch_size: Optional[int] = None
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = False
    multiprocessing_context: str = "fork"
    drop_last: bool = False
    shuffle_train: bool = True
    shuffle_val: bool = False
    shuffle_test: bool = False


@dataclass
class DataConfig(TrackedConfigMixin):
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
