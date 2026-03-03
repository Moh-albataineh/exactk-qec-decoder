"""
Model Evaluator — Day 18

Unified evaluation report for any decoder (MLP, GNN, etc.).
Includes graph_hash in report if model is GNN.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from qec_noise_factory.ml.train.trainer import (
    make_dataloader,
    eval_one_epoch,
    _is_graph_model,
)
from qec_noise_factory.ml.train.config import TrainConfig


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    config: TrainConfig,
    dataset_hash: str = "",
    graph_hash: str = "",
    edge_index: Optional[torch.Tensor] = None,
    num_nodes: int = 0,
    feature_dim: int = 1,
    extra_meta: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model and produce a standardized report.

    Args:
        model: trained decoder model
        X_test: (N, D) test features
        Y_test: (N, O) test labels
        config: training config
        dataset_hash: dataset provenance hash
        graph_hash: graph hash (GNN only, "" for MLP)
        edge_index: edge index tensor (GNN only)
        num_nodes: graph nodes (GNN only)
        feature_dim: feature dim per node (GNN only)
        extra_meta: additional metadata to include

    Returns:
        eval_report dict with metrics + provenance
    """
    loader = make_dataloader(
        X_test, Y_test,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Get model name
    model_name = getattr(model, "model_name", model.__class__.__name__)
    is_gnn = _is_graph_model(model)

    # Evaluate
    metrics = eval_one_epoch(
        model, loader,
        device=config.device,
        edge_index=edge_index if is_gnn else None,
        num_nodes=num_nodes if is_gnn else 0,
        feature_dim=feature_dim if is_gnn else 1,
    )

    # Build report
    report = {
        "model_name": model_name,
        "model_type": "gnn" if is_gnn else "mlp",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_hash": dataset_hash,
        "split_policy": config.split_policy,
        "config_hash": config.config_hash(),
        "metrics": metrics,
        "data": {
            "num_test_samples": len(X_test),
            "input_dim": X_test.shape[1],
            "output_dim": Y_test.shape[1] if Y_test.ndim > 1 else 1,
        },
    }

    # Add graph info for GNN
    if is_gnn:
        report["graph_hash"] = graph_hash
        report["graph"] = {
            "num_nodes": num_nodes,
            "num_edges": edge_index.shape[1] if edge_index is not None else 0,
            "feature_dim": feature_dim,
        }

    if extra_meta:
        report["extra"] = extra_meta

    return report


def save_eval_report(
    report: Dict[str, Any],
    out_path: str | Path,
) -> Path:
    """Save evaluation report as JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return out_path
