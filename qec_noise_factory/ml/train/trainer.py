"""
Training Loop — Day 16 / Updated Day 18

Epoch-level training and evaluation with BCE loss.
Supports both flat models (MLP) and graph models (GNN).

GNN path:
    - Model has `needs_graph = True`
    - Trainer reshapes X from (B, D) → (B, N, F) using graph_spec
    - Passes edge_index to model
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from qec_noise_factory.ml.metrics.classification import compute_metrics


def make_dataloader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader from numpy arrays."""
    X_t = torch.from_numpy(X.astype(np.float32))
    Y_t = torch.from_numpy(Y.astype(np.float32))
    ds = TensorDataset(X_t, Y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _is_graph_model(model: nn.Module) -> bool:
    """Check if model requires graph input."""
    return getattr(model, "needs_graph", False)


def _prepare_gnn_input(
    X_batch: torch.Tensor,
    num_nodes: int,
    feature_dim: int,
    edge_index: torch.Tensor,
    device: str,
    edge_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Reshape flat X (B, D) → (B, N, F) for GNN input.

    Args:
        X_batch: (B, D) where D = N * F (or D = N if F=1)
        num_nodes: N
        feature_dim: F
        edge_index: (2, E) precomputed edge index
        device: target device
        edge_weight: (E,) optional DEM edge weights (Day 28, V2 only)

    Returns:
        (node_features, edge_index, edge_weight_on_device) ready for GNN forward
    """
    B = X_batch.shape[0]
    D = X_batch.shape[1]

    if D == num_nodes:
        # D == N, assume F=1
        node_features = X_batch.reshape(B, num_nodes, 1)
    elif D == num_nodes * feature_dim:
        node_features = X_batch.reshape(B, num_nodes, feature_dim)
    else:
        raise ValueError(
            f"Cannot reshape X (B={B}, D={D}) to (B, N={num_nodes}, F={feature_dim})"
        )

    ew = edge_weight.to(device) if edge_weight is not None else None
    return node_features.to(device), edge_index.to(device), ew


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    grad_clip: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    num_nodes: int = 0,
    feature_dim: int = 1,
    edge_weight: Optional[torch.Tensor] = None,
) -> float:
    """
    Train one epoch. Returns average loss.

    Args:
        model: PyTorch model (outputs logits)
        loader: DataLoader of (X, Y) batches
        optimizer: optimizer instance
        device: "cpu" or "cuda"
        grad_clip: max gradient norm (0 = disabled)
        class_weights: optional per-class weight tensor for BCEWithLogitsLoss
        edge_index: (2, E) edge index for GNN models (None for MLP)
        num_nodes: number of graph nodes (GNN only)
        feature_dim: feature dimension per node (GNN only)
        edge_weight: (E,) DEM edge weights for V2 models (Day 28)
    """
    model.train()
    is_gnn = _is_graph_model(model)
    _uses_ew = getattr(model, "uses_edge_weight", False)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=class_weights.to(device) if class_weights is not None else None,
    )

    total_loss = 0.0
    n_batches = 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()

        if is_gnn:
            node_feats, ei, ew = _prepare_gnn_input(
                X_batch, num_nodes, feature_dim, edge_index, device,
                edge_weight=edge_weight if _uses_ew else None,
            )
            if _uses_ew:
                logits = model(node_feats, ei, edge_weight=ew)
            else:
                logits = model(node_feats, ei)
        else:
            logits = model(X_batch)

        # Shape check
        if logits.shape != Y_batch.shape:
            raise RuntimeError(
                f"Logits shape {logits.shape} != Y shape {Y_batch.shape}"
            )

        loss = criterion(logits, Y_batch)

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"NaN/Inf loss detected at batch {n_batches}")

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    edge_index: Optional[torch.Tensor] = None,
    num_nodes: int = 0,
    feature_dim: int = 1,
    edge_weight: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Evaluate one epoch. Returns metrics dict.

    Supports both MLP (flat) and GNN (graph) models.
    """
    model.eval()
    is_gnn = _is_graph_model(model)
    _uses_ew = getattr(model, "uses_edge_weight", False)

    all_y_true = []
    all_y_pred = []
    total_loss = 0.0
    n_batches = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            if is_gnn:
                node_feats, ei, ew = _prepare_gnn_input(
                    X_batch, num_nodes, feature_dim, edge_index, device,
                    edge_weight=edge_weight if _uses_ew else None,
                )
                if _uses_ew:
                    logits = model(node_feats, ei, edge_weight=ew)
                else:
                    logits = model(node_feats, ei)
            else:
                logits = model(X_batch)

            loss = criterion(logits, Y_batch)
            total_loss += loss.item()
            n_batches += 1

            # Predictions: sigmoid > 0.5
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(bool)
            all_y_true.append(Y_batch.cpu().numpy().astype(bool))
            all_y_pred.append(preds)

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)

    metrics = compute_metrics(y_true, y_pred)
    metrics["eval_loss"] = total_loss / max(1, n_batches)
    metrics["num_samples"] = len(y_true)

    return metrics
