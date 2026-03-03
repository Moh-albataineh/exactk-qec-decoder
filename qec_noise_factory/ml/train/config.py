"""
Training Configuration — Day 16

Immutable training config with full provenance support.
Serializable to YAML for artifact storage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class TrainConfig:
    """Immutable training configuration."""
    # Reproducibility
    seed: int = 42
    device: str = "cpu"

    # Data
    shards_dir: str = ""
    pack_names: tuple = ("baseline_symmetric",)

    # Splits
    split_policy: str = "within_model"  # within_model | cross_model | ood_p_range
    split_ratio: float = 0.8
    split_seed: int = 42

    # Filters
    p_min: float = 0.0
    p_max: float = 1.0

    # Training
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 10
    grad_clip: float = 0.0       # 0 = disabled
    class_weights: bool = False

    # Model
    model_type: str = "mlp"      # trivial | mlp
    hidden_dim: int = 64
    num_hidden_layers: int = 2

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuple to list for YAML
        d["pack_names"] = list(d["pack_names"])
        return d

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        # Convert list back to tuple
        if "pack_names" in d and isinstance(d["pack_names"], list):
            d["pack_names"] = tuple(d["pack_names"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "TrainConfig":
        return cls.from_dict(yaml.safe_load(yaml_str))

    def config_hash(self) -> str:
        """Deterministic hash of the config for provenance."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
