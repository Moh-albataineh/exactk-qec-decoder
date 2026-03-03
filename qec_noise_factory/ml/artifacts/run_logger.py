"""
Run Logger — Day 16

Manages ML experiment artifacts with full provenance tracking.
Each run produces:
  ml_artifacts/runs/<run_id>/
    ├── train_manifest.yaml
    ├── metrics.json
    ├── checkpoint.pt
    └── checksums.sha256
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import torch
import torch.nn as nn

from qec_noise_factory.ml.data.schema import ShardMeta
from qec_noise_factory.ml.train.config import TrainConfig


def dataset_hash(
    shard_metas: List[ShardMeta],
    split_policy: str,
    p_min: float = 0.0,
    p_max: float = 1.0,
    split_seed: int = 42,
) -> str:
    """
    Deterministic hash of the dataset configuration.

    Built from:
      - sorted shard sample_keys
      - split policy + seed
      - p-range filters
    """
    hasher = hashlib.sha256()

    # Shard identity: sorted sample_keys for determinism
    keys = sorted(m.sample_key for m in shard_metas)
    for k in keys:
        hasher.update(k.encode("utf-8"))

    # Split params
    hasher.update(f"split:{split_policy}:{split_seed}".encode())

    # Filters
    hasher.update(f"p_range:{p_min:.8f}:{p_max:.8f}".encode())

    return hasher.hexdigest()


def _get_git_commit() -> str:
    """Get current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class RunLogger:
    """
    Manages artifacts for a single training run.

    Usage:
        logger = RunLogger(run_id="mlp_v0_001", base_dir="ml_artifacts")
        logger.save_manifest(config, train_metas, val_metas, test_metas, ds_hash)
        logger.save_metrics({"train": {...}, "val": {...}, "test": {...}})
        logger.save_checkpoint(model)
        logger.write_checksums()
    """

    def __init__(
        self,
        run_id: str,
        base_dir: str = "ml_artifacts",
    ):
        self.run_id = run_id
        self.run_dir = Path(base_dir) / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_manifest(
        self,
        config: TrainConfig,
        train_metas: List[ShardMeta],
        val_metas: List[ShardMeta],
        test_metas: Optional[List[ShardMeta]],
        ds_hash: str,
    ) -> Path:
        """Save train_manifest.yaml with full provenance."""
        manifest = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_version": _get_git_commit(),
            "dataset_hash": ds_hash,
            "config": config.to_dict(),
            "split": {
                "policy": config.split_policy,
                "ratio": config.split_ratio,
                "seed": config.split_seed,
            },
            "filters": {
                "p_min": config.p_min,
                "p_max": config.p_max,
            },
            "data_summary": {
                "train_blocks": len(train_metas),
                "train_samples": sum(m.record_count for m in train_metas),
                "val_blocks": len(val_metas),
                "val_samples": sum(m.record_count for m in val_metas),
                "test_blocks": len(test_metas) if test_metas else 0,
                "test_samples": sum(m.record_count for m in test_metas) if test_metas else 0,
                "train_physics_hashes": sorted(set(
                    m.physics_hash for m in train_metas if m.physics_hash
                )),
                "val_physics_hashes": sorted(set(
                    m.physics_hash for m in val_metas if m.physics_hash
                )),
            },
            "seeds": {
                "train_seed": config.seed,
                "split_seed": config.split_seed,
            },
        }

        path = self.run_dir / "train_manifest.yaml"
        path.write_text(
            yaml.dump(manifest, default_flow_style=False, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def save_metrics(self, metrics: Dict[str, Any]) -> Path:
        """Save metrics.json."""
        path = self.run_dir / "metrics.json"
        path.write_text(
            json.dumps(metrics, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        extra: Optional[Dict] = None,
    ) -> Path:
        """Save model checkpoint."""
        path = self.run_dir / "checkpoint.pt"
        state = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)
        return path

    def write_checksums(self) -> Path:
        """Write SHA256 checksums for all files in the run directory."""
        path = self.run_dir / "checksums.sha256"
        lines = []
        for f in sorted(self.run_dir.iterdir()):
            if f.is_file() and f.name != "checksums.sha256":
                sha = _file_sha256(f)
                lines.append(f"{sha}  {f.name}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
