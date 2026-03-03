"""
Day 42 — Scrambler Clumpiness Diagnostic

Measures structural properties of detector activation patterns before and
after scrambling: connected components, largest component, mean component size.
Uses detector adjacency from the DEM bipartite graph.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List

import numpy as np
import torch


def _find_components(active_indices: np.ndarray, adj_pairs: np.ndarray) -> List[set]:
    """Find connected components among active detectors using union-find.

    Args:
        active_indices: indices of active (syndrome=1) detectors.
        adj_pairs: (E, 2) array of adjacent detector pairs.

    Returns:
        list of sets, each containing detector indices in one component.
    """
    if len(active_indices) == 0:
        return []

    active_set = set(active_indices.tolist())
    parent = {i: i for i in active_set}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Connect adjacent active detectors
    if len(adj_pairs) > 0:
        for e in range(adj_pairs.shape[0]):
            a, b = int(adj_pairs[e, 0]), int(adj_pairs[e, 1])
            if a in active_set and b in active_set:
                union(a, b)

    # Group by root
    groups = {}
    for i in active_set:
        root = find(i)
        groups.setdefault(root, set()).add(i)

    return list(groups.values())


def compute_clumpiness_metrics(
    syndrome: np.ndarray,
    adj_pairs: np.ndarray,
) -> Dict[str, float]:
    """Compute clumpiness metrics for one syndrome pattern.

    Args:
        syndrome: (N_det,) binary syndrome (0/1).
        adj_pairs: (E, 2) array of adjacent detector pairs.

    Returns:
        dict with n_components, largest_component, mean_component_size, n_active.
    """
    active = np.where(syndrome.ravel() > 0.5)[0]
    n_active = len(active)

    if n_active == 0:
        return {
            "n_components": 0,
            "largest_component": 0,
            "mean_component_size": 0.0,
            "n_active": 0,
        }

    components = _find_components(active, adj_pairs)
    sizes = [len(c) for c in components]

    return {
        "n_components": len(components),
        "largest_component": max(sizes) if sizes else 0,
        "mean_component_size": float(np.mean(sizes)) if sizes else 0.0,
        "n_active": n_active,
    }


def build_detector_adjacency(
    edge_index_d2e: torch.Tensor,
    n_det: int,
) -> np.ndarray:
    """Build detector adjacency from D→E edges.

    Two detectors are adjacent if they share at least one error node.
    Returns (E_det, 2) array of adjacent detector pairs.
    """
    d_src = edge_index_d2e[0].numpy()
    e_dst = edge_index_d2e[1].numpy()

    # Group detectors by error node
    err_to_dets = {}
    for d, e in zip(d_src, e_dst):
        if d < n_det:  # exclude boundary
            err_to_dets.setdefault(int(e), []).append(int(d))

    # Pairs of detectors sharing an error node
    pairs = set()
    for dets in err_to_dets.values():
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                a, b = min(dets[i], dets[j]), max(dets[i], dets[j])
                pairs.add((a, b))

    if not pairs:
        return np.zeros((0, 2), dtype=int)
    return np.array(sorted(pairs), dtype=int)


def compute_scrambler_clumpiness_effect(
    det_features: torch.Tensor,
    adj_pairs: np.ndarray,
    seed: int = 42,
    syndrome_channel: int = 0,
) -> Dict[str, Any]:
    """Compare clumpiness metrics before and after scrambling.

    Args:
        det_features: (B, N_det, F) detector features.
        adj_pairs: (E_det, 2) adjacent detector pairs.
        seed: scrambler seed.

    Returns:
        dict with clean/scrambled means and deltas.
    """
    from qec_noise_factory.ml.bench.density_scrambler import scramble_detector_syndromes

    det_scrambled = scramble_detector_syndromes(det_features, seed=seed)

    B = det_features.shape[0]
    n_det = det_features.shape[1] - 1  # exclude boundary

    clean_metrics = []
    scrambled_metrics = []

    for b in range(B):
        syn_clean = det_features[b, :n_det, syndrome_channel].numpy()
        syn_scrambled = det_scrambled[b, :n_det, syndrome_channel].numpy()

        clean_metrics.append(compute_clumpiness_metrics(syn_clean, adj_pairs))
        scrambled_metrics.append(compute_clumpiness_metrics(syn_scrambled, adj_pairs))

    def _mean_field(metrics, field):
        vals = [m[field] for m in metrics]
        return float(np.mean(vals)) if vals else 0.0

    def _median_field(metrics, field):
        vals = [m[field] for m in metrics]
        return float(np.median(vals)) if vals else 0.0

    fields = ["n_components", "largest_component", "mean_component_size"]
    result = {}

    for f in fields:
        c_mean = _mean_field(clean_metrics, f)
        s_mean = _mean_field(scrambled_metrics, f)
        c_med = _median_field(clean_metrics, f)
        s_med = _median_field(scrambled_metrics, f)
        result[f"clean_mean_{f}"] = c_mean
        result[f"scrambled_mean_{f}"] = s_mean
        result[f"delta_mean_{f}"] = c_mean - s_mean
        result[f"clean_median_{f}"] = c_med
        result[f"scrambled_median_{f}"] = s_med
        result[f"delta_median_{f}"] = c_med - s_med

    result["n_samples"] = B
    result["mean_n_active_clean"] = _mean_field(clean_metrics, "n_active")
    result["mean_n_active_scrambled"] = _mean_field(scrambled_metrics, "n_active")

    # Summary: is scrambler effective?
    delta_comp = result["delta_mean_n_components"]
    delta_largest = result["delta_mean_largest_component"]
    result["scrambler_destroys_clumps"] = abs(delta_largest) > 0.5 or abs(delta_comp) > 0.5

    return result
