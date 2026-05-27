from dataclasses import dataclass, field
from typing import Any, Dict

from configs.config_tracker import TrackedConfigMixin

@dataclass
class ModelConfig(TrackedConfigMixin):
    file_name: str = "simple_net"
    class_name: str = "SimpleNet"
    model_init_args: Dict[str, Any] = field(default_factory=dict)
