"""
Density-Only Baseline — Day 36

Provides a density-only predictor (logistic regression on syndrome count K)
to separate density-based performance from true topology learning.

Key functions:
  - compute_syndrome_count(X) -> K array
  - density_only_auroc(K, y_true) -> {auroc_density, pr_auc_density}
  - compute_topology_gain(auroc_clean, auroc_density) -> float
"""
from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np


def compute_syndrome_count(X: np.ndarray) -> np.ndarray:
    """Compute per-sample syndrome count K = sum of detector bits.

    Args:
        X: (N, D) binary detector matrix (0/1 or bool).

    Returns:
        K: (N,) integer array of syndrome counts.
    """
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        return np.array([int(X_arr.sum())])
    return X_arr.sum(axis=1).astype(int)


def density_only_auroc(
    K: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, Any]:
    """Compute AUROC/PR-AUC using only syndrome count as predictor.

    Uses logistic regression on K -> y for a proper probability model.
    Falls back to raw K as score if logistic regression fails.

    Args:
        K: (N,) syndrome counts.
        y_true: (N,) boolean labels.

    Returns:
        dict with auroc_density, pr_auc_density (None if single-class).
    """
    from qec_noise_factory.ml.metrics.ranking import compute_auroc

    y = np.asarray(y_true, dtype=bool).ravel()
    K_arr = np.asarray(K, dtype=float).ravel()

    # Single-class guard
    if y.all() or (~y).all():
        return {
            "auroc_density": None,
            "pr_auc_density": None,
            "method": "single_class",
        }

    # Logistic regression on K
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=200, solver="lbfgs")
        lr.fit(K_arr.reshape(-1, 1), y.astype(int))
        probs = lr.predict_proba(K_arr.reshape(-1, 1))[:, 1]
        method = "logistic_regression"
    except Exception:
        # Fallback: use raw K as score (higher K -> more likely error)
        probs = K_arr.astype(float)
        # Normalize to [0, 1]
        k_range = probs.max() - probs.min()
        if k_range > 0:
            probs = (probs - probs.min()) / k_range
        else:
            probs = np.full_like(probs, 0.5)
        method = "raw_score_fallback"

    auroc = compute_auroc(y, probs)

    # PR-AUC (use internal metrics, no sklearn needed)
    pr_auc = None
    try:
        from qec_noise_factory.ml.metrics.ranking import compute_ranking_diagnostics
        diag = compute_ranking_diagnostics(y, probs)
        pr_auc = diag.get("pr_auc")
    except Exception:
        pass

    return {
        "auroc_density": auroc,
        "pr_auc_density": pr_auc,
        "method": method,
    }


def compute_topology_gain(
    auroc_clean: Optional[float],
    auroc_density: Optional[float],
) -> Optional[float]:
    """TopologyGain = AUROC_model_clean - AUROC_density.

    Measures how much the model learns beyond simple syndrome counting.

    Returns:
        Float gain, or None if either input is None.
    """
    if auroc_clean is None or auroc_density is None:
        return None
    return auroc_clean - auroc_density


def extract_p_distribution(meta_list) -> Dict[str, Any]:
    """Extract p-value distribution from shard metadata.

    Args:
        meta_list: list of ShardMeta objects.

    Returns:
        dict with p_values (sorted unique), p_min, p_max, p_count, p_first.
    """
    from qec_noise_factory.ml.data.schema import extract_p

    p_values = []
    for m in meta_list:
        try:
            p = extract_p(m.params_canonical)
            p_values.append(p)
        except Exception:
            continue

    if not p_values:
        return {
            "p_values": [],
            "p_min": None,
            "p_max": None,
            "p_count": 0,
            "p_first": None,
        }

    p_unique = sorted(set(p_values))
    return {
        "p_values": p_unique,
        "p_min": min(p_unique),
        "p_max": max(p_unique),
        "p_count": len(p_values),
        "p_first": p_values[0],
    }


def check_p_regime(
    meta_list,
    target_p: float,
    tolerance: float = 0.5,
) -> Dict[str, Any]:
    """Check if shard p-values match target_p within tolerance.

    Tolerance is relative: |p - target_p| / target_p <= tolerance.

    Returns:
        dict with match (bool), p_dist, closest_p, relative_error.
    """
    p_dist = extract_p_distribution(meta_list)

    if not p_dist["p_values"]:
        return {
            "match": False,
            "p_dist": p_dist,
            "closest_p": None,
            "relative_error": None,
            "reason": "no_p_values_found",
        }

    # Find closest p
    closest = min(p_dist["p_values"], key=lambda p: abs(p - target_p))
    rel_err = abs(closest - target_p) / max(target_p, 1e-10)

    return {
        "match": rel_err <= tolerance,
        "p_dist": p_dist,
        "closest_p": closest,
        "relative_error": rel_err,
        "reason": "ok" if rel_err <= tolerance else f"closest_p={closest:.4f}, target={target_p:.4f}, rel_err={rel_err:.2f}",
    }
