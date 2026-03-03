"""
Feature Mapper — Day 17

Maps raw detector data X to node-level features for GNN input.

Contract:
    Input:   X_batch  shape (B, num_detectors)  bool/float
    Output:  features shape (B, N, F)           float32

    where N = graph.num_nodes, F = feature_dim per node
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from qec_noise_factory.ml.graph.graph_types import GraphSpec


def X_to_node_features(
    X_batch: np.ndarray,
    graph: GraphSpec,
    feature_dim: int = 1,
) -> np.ndarray:
    """
    Convert detector readout X into per-node feature vectors.

    For demo_repetition:
        Each detector maps to one node, feature = detector bit value.
        Output shape: (B, N, 1) where N = num_detectors

    For surface_code (future):
        Could include positional encoding, syndrome context, etc.

    Args:
        X_batch: (B, num_detectors) bool or float array
        graph: GraphSpec defining the node structure
        feature_dim: features per node (1 for raw detector bits)

    Returns:
        features: (B, N, F) float32 array
    """
    B = X_batch.shape[0]
    N = graph.num_nodes
    num_det = X_batch.shape[1]

    if num_det != N:
        raise ValueError(
            f"Detector count mismatch: X has {num_det} detectors "
            f"but graph has {N} nodes"
        )

    # Base feature: detector bit as float32
    # Conversion: bool True → 1.0, False → 0.0 (deterministic, no other values)
    features = X_batch.astype(np.float32)

    # Reshape to (B, N, F)
    if feature_dim == 1:
        features = features.reshape(B, N, 1)
    else:
        # Pad with zeros for future multi-feature support
        padded = np.zeros((B, N, feature_dim), dtype=np.float32)
        padded[:, :, 0] = features
        features = padded

    return features


def add_positional_features(
    features: np.ndarray,
    graph: GraphSpec,
) -> np.ndarray:
    """
    Append node position as an additional feature.

    Input:  (B, N, F)
    Output: (B, N, F+1)
    """
    B, N, F = features.shape

    # Normalize positions to [0, 1]
    positions = np.array(graph.node_attrs.get("position", list(range(N))),
                         dtype=np.float32)
    if positions.max() > 0:
        positions = positions / positions.max()

    # Broadcast to (B, N, 1)
    pos_feature = np.broadcast_to(positions.reshape(1, N, 1), (B, N, 1))

    return np.concatenate([features, pos_feature], axis=2)


# ---------------------------------------------------------------------------
# Day 23: Feature Enrichment v1
# ---------------------------------------------------------------------------

V1_FEATURE_NAMES = [
    "detector_bit",   # 0/1 float
    "deg_norm",       # normalized node degree
    "pos_idx_norm",   # i / (N-1)
    "p_value",        # physical error rate (broadcast)
    "basis_01",       # Z=0, X=1 (broadcast)
    "distance_norm",  # d / 7.0 (broadcast, max d=7)
]
V1_FEATURE_DIM = len(V1_FEATURE_NAMES)  # 6
V1_FEATURE_VERSION = "v1"


def _compute_node_degrees(graph: GraphSpec) -> np.ndarray:
    """Compute degree per node from GraphSpec edges (undirected)."""
    degrees = np.zeros(graph.num_nodes, dtype=np.float32)
    for u, v in graph.edges:
        degrees[u] += 1
        degrees[v] += 1
    return degrees


def X_to_node_features_v1(
    X_batch: np.ndarray,
    graph: GraphSpec,
    p_value: float = 0.0,
    basis: str = "",
    distance: int = 0,
) -> np.ndarray:
    """
    Convert detector readout X to enriched node features (Day 23).

    Features per node (F=6):
        0: detector_bit   — 0.0 or 1.0
        1: deg_norm        — degree / max_degree (from graph edges)
        2: pos_idx_norm    — i / (N-1) linear position
        3: p_value         — physical error rate (broadcast)
        4: basis_01        — Z→0, X→1 (broadcast)
        5: distance_norm   — d / 7.0 (broadcast)

    Returns:
        features: (B, N, 6) float32
    """
    B = X_batch.shape[0]
    N = graph.num_nodes
    num_det = X_batch.shape[1]

    if num_det != N:
        raise ValueError(
            f"Detector count mismatch: X has {num_det} detectors "
            f"but graph has {N} nodes"
        )

    features = np.zeros((B, N, V1_FEATURE_DIM), dtype=np.float32)

    # F0: detector bit
    features[:, :, 0] = X_batch.astype(np.float32)

    # F1: normalized node degree
    degrees = _compute_node_degrees(graph)
    max_deg = max(degrees.max(), 1.0)
    features[:, :, 1] = degrees / max_deg  # broadcast (N,) → (B, N)

    # F2: positional index normalized
    if N > 1:
        pos_idx = np.arange(N, dtype=np.float32) / (N - 1)
    else:
        pos_idx = np.zeros(N, dtype=np.float32)
    features[:, :, 2] = pos_idx  # broadcast

    # F3: p_value (broadcast)
    features[:, :, 3] = float(p_value)

    # F4: basis encoding (Z=0, X=1)
    basis_val = 1.0 if basis.upper() == "X" else 0.0
    features[:, :, 4] = basis_val

    # F5: distance normalized (d / 7.0)
    features[:, :, 5] = float(distance) / 7.0

    return features


def get_v1_feature_meta() -> dict:
    """Return metadata dict for v1 features."""
    return {
        "feature_version": V1_FEATURE_VERSION,
        "feature_dim": V1_FEATURE_DIM,
        "feature_names": list(V1_FEATURE_NAMES),
    }


# ---------------------------------------------------------------------------
# Day 24: Feature Gating — featureset variants
# ---------------------------------------------------------------------------

# Indices into V1_FEATURE_NAMES:
# 0=detector_bit, 1=deg_norm, 2=pos_idx_norm, 3=p_value, 4=basis_01, 5=distance_norm

FEATURESET_REGISTRY = {
    "v1_full": {
        "indices": [0, 1, 2, 3, 4, 5],
        "names": ["detector_bit", "deg_norm", "pos_idx_norm", "p_value", "basis_01", "distance_norm"],
        "dim": 6,
    },
    "v1_nop": {
        "indices": [0, 1, 2, 4, 5],
        "names": ["detector_bit", "deg_norm", "pos_idx_norm", "basis_01", "distance_norm"],
        "dim": 5,
    },
    "v1_nop_nodist": {
        "indices": [0, 1, 2, 4],
        "names": ["detector_bit", "deg_norm", "pos_idx_norm", "basis_01"],
        "dim": 4,
    },
}


def X_to_node_features_gated(
    X_batch: np.ndarray,
    graph: GraphSpec,
    featureset: str = "v1_full",
    p_value: float = 0.0,
    basis: str = "",
    distance: int = 0,
) -> np.ndarray:
    """
    Day 24: Build node features with featureset gating.

    First computes full v1 features (F=6), then selects columns
    based on featureset name.

    Args:
        featureset: "v1_full" (F=6), "v1_nop" (F=5), "v1_nop_nodist" (F=4)

    Returns:
        features: (B, N, F_gated) float32
    """
    if featureset not in FEATURESET_REGISTRY:
        raise ValueError(f"Unknown featureset: {featureset!r}. "
                         f"Available: {list(FEATURESET_REGISTRY.keys())}")

    full = X_to_node_features_v1(X_batch, graph, p_value=p_value,
                                  basis=basis, distance=distance)
    indices = FEATURESET_REGISTRY[featureset]["indices"]
    return full[:, :, indices].copy()


def get_featureset_meta(featureset: str = "v1_full") -> dict:
    """Return metadata dict for a given featureset."""
    if featureset not in FEATURESET_REGISTRY:
        raise ValueError(f"Unknown featureset: {featureset!r}")
    entry = FEATURESET_REGISTRY[featureset]
    return {
        "featureset_name": featureset,
        "feature_version": V1_FEATURE_VERSION,
        "feature_dim": entry["dim"],
        "feature_names": list(entry["names"]),
    }


# ---------------------------------------------------------------------------
# Day 28: Boundary node padding for DEM graph V2
# ---------------------------------------------------------------------------

def pad_boundary_node(
    features: np.ndarray,
    add_is_boundary_feature: bool = True,
) -> np.ndarray:
    """
    Pad node features with a boundary node row (Day 28).

    DEM graphs have N+1 nodes (N detectors + 1 boundary).
    This adds a zeros row for the boundary node and optionally
    appends an 'is_boundary' feature column.

    Args:
        features: (B, N, F) node features for detector nodes
        add_is_boundary_feature: if True, append F+1 column (0 for detectors, 1 for boundary)

    Returns:
        (B, N+1, F+1) if add_is_boundary_feature else (B, N+1, F)
    """
    B, N, F = features.shape

    if add_is_boundary_feature:
        # Add is_boundary column: 0 for detectors, 1 for boundary
        boundary_col = np.zeros((B, N, 1), dtype=np.float32)
        features_aug = np.concatenate([features, boundary_col], axis=2)  # (B, N, F+1)

        # Boundary node row: zeros except is_boundary=1
        boundary_row = np.zeros((B, 1, F + 1), dtype=np.float32)
        boundary_row[:, :, -1] = 1.0  # is_boundary = 1

        return np.concatenate([features_aug, boundary_row], axis=1)  # (B, N+1, F+1)
    else:
        # Just add zeros row
        boundary_row = np.zeros((B, 1, F), dtype=np.float32)
        return np.concatenate([features, boundary_row], axis=1)  # (B, N+1, F)
