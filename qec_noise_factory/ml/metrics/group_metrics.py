"""
Group-Level Metrics — Day 20.5

Per-group accuracy/BER/TPR/TNR breakdown by metadata axes:
  - physics model (noise_model)
  - basis (X/Z)
  - distance (3/5/7)
  - p-bucket

Prevents illusory improvements from class imbalance.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qec_noise_factory.ml.data.schema import ShardMeta
from qec_noise_factory.ml.metrics.classification import compute_metrics


# ---------------------------------------------------------------------------
# Circuit info extraction
# ---------------------------------------------------------------------------

def _extract_group_keys(m: ShardMeta) -> Dict[str, str]:
    """Extract grouping keys from a ShardMeta block."""
    try:
        obj = json.loads(m.params_canonical)
        c = obj.get("circuit", {})
    except (json.JSONDecodeError, TypeError):
        c = {}

    model = c.get("noise_model", "") or m.physics_model_name or m.pack_name
    basis = c.get("basis", "").upper() or "?"
    distance = c.get("distance", 0)
    p = m.p

    # p-bucket
    if p < 0.005:
        p_bucket = "p<0.005"
    elif p < 0.01:
        p_bucket = "0.005≤p<0.01"
    elif p < 0.02:
        p_bucket = "0.01≤p<0.02"
    else:
        p_bucket = "p≥0.02"

    return {
        "by_model": model,
        "by_basis": basis,
        "by_distance": f"d={distance}" if distance else "d=?",
        "by_p_bucket": p_bucket,
    }


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: List[ShardMeta],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute per-group metrics breakdown.

    Args:
        y_true: (N, O) ground truth labels
        y_pred: (N, O) predicted labels
        meta: list of ShardMeta blocks matching the data order

    Returns:
        Nested dict: {axis_name: {group_value: {metric: value}}}
        Example: {"by_model": {"sd6_like": {"accuracy": 0.85, "ber": 0.15, ...}}}
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    # Build per-sample group assignments
    axes = ["by_model", "by_basis", "by_distance", "by_p_bucket"]
    # Map sample_index -> group_keys
    sample_groups: List[Dict[str, str]] = []

    offset = 0
    for m in meta:
        n = m.record_count
        keys = _extract_group_keys(m)
        for _ in range(n):
            sample_groups.append(keys)
        offset += n

    n_samples = y_true.shape[0]
    if len(sample_groups) != n_samples:
        # Fallback: if meta doesn't cover all samples, pad with unknowns
        while len(sample_groups) < n_samples:
            sample_groups.append({a: "?" for a in axes})

    # Compute metrics per axis per group
    result: Dict[str, Dict[str, Dict[str, float]]] = {}

    for axis in axes:
        # Collect unique group values
        groups: Dict[str, List[int]] = {}
        for i, sg in enumerate(sample_groups):
            gv = sg[axis]
            if gv not in groups:
                groups[gv] = []
            groups[gv].append(i)

        axis_result: Dict[str, Dict[str, float]] = {}
        for gv, indices in sorted(groups.items()):
            idx = np.array(indices)
            gt = y_true[idx]
            gp = y_pred[idx]
            m = compute_metrics(gt, gp)
            axis_result[gv] = {
                "accuracy": m["macro_accuracy"],
                "ber": m["macro_ber"],
                "balanced_accuracy": m["macro_balanced_accuracy"],
                "tpr": m.get("obs_0_tpr", 0.0),
                "tnr": m.get("obs_0_tnr", 0.0),
                "n_samples": len(indices),
            }

        result[axis] = axis_result

    return result


def format_group_table(group_metrics: Dict) -> str:
    """Format group metrics as a readable table string."""
    lines = []
    for axis, groups in group_metrics.items():
        lines.append(f"\n  {axis}:")
        lines.append(f"  {'Group':<20} {'Acc':>7} {'BER':>7} {'BalAcc':>7} {'TPR':>7} {'TNR':>7} {'N':>7}")
        lines.append(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for gv, m in groups.items():
            lines.append(
                f"  {gv:<20} {m['accuracy']:>7.2%} {m['ber']:>7.2%} "
                f"{m['balanced_accuracy']:>7.2%} {m['tpr']:>7.2%} {m['tnr']:>7.2%} "
                f"{m['n_samples']:>7,}"
            )
    return "\n".join(lines)
