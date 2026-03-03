"""
Ranking Metrics — Day 34

Unthresholded ranking quality metrics for QEC decoder evaluation.
No sklearn dependency — pure numpy implementation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def compute_auroc(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> Optional[float]:
    """Area Under the ROC Curve via trapezoidal rule.

    Args:
        y_true: (N,) or (N,1) binary ground truth
        probs: (N,) or (N,1) predicted probabilities

    Returns:
        AUROC float, or None if only one class is present.
    """
    y_true = np.asarray(y_true, dtype=bool).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()

    if len(y_true) < 2:
        return None
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None  # single class — AUROC undefined

    # Sort by descending probability
    order = np.argsort(-probs)
    y_sorted = y_true[order]

    # Compute TPR and FPR at each threshold
    tps = np.cumsum(y_sorted).astype(np.float64)
    fps = np.cumsum(~y_sorted).astype(np.float64)

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend origin (0, 0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # Trapezoidal integration (np.trapezoid in NumPy 2.0+, np.trapz before)
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auroc = float(_trapz(tpr, fpr))
    return auroc


def compute_pr_auc(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> Optional[float]:
    """Average Precision (area under Precision-Recall curve).

    Args:
        y_true: (N,) or (N,1) binary ground truth
        probs: (N,) or (N,1) predicted probabilities

    Returns:
        PR-AUC float, or None if no positive samples.
    """
    y_true = np.asarray(y_true, dtype=bool).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()

    if len(y_true) < 2:
        return None
    n_pos = y_true.sum()
    if n_pos == 0:
        return None  # no positives — PR-AUC undefined

    # Sort by descending probability
    order = np.argsort(-probs)
    y_sorted = y_true[order]

    tps = np.cumsum(y_sorted).astype(np.float64)
    # Precision at each threshold: TP / (TP + FP) = TP / rank
    ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    precision = tps / ranks

    # Average precision: mean of precision values at recall changes (positive samples)
    ap = float(np.sum(precision[y_sorted]) / n_pos)
    return ap


def compute_decile_table(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, Any]]:
    """Split predictions into deciles by probability, compute stats per bin.

    Args:
        y_true: (N,) or (N,1) binary ground truth
        probs: (N,) or (N,1) predicted probabilities
        n_bins: number of bins (default 10 = deciles)

    Returns:
        List of dicts, one per bin (from highest to lowest prob):
        [{"bin": 0, "prob_lo": 0.9, "prob_hi": 1.0, "n": 100,
          "n_pos": 80, "pos_frac": 0.80}, ...]
    """
    y_true = np.asarray(y_true, dtype=bool).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()

    # Sort descending
    order = np.argsort(-probs)
    y_sorted = y_true[order]
    p_sorted = probs[order]

    n = len(y_sorted)
    if n == 0:
        return []

    bin_size = max(1, n // n_bins)
    table = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        if start >= n:
            break
        bin_y = y_sorted[start:end]
        bin_p = p_sorted[start:end]
        n_in = len(bin_y)
        n_pos = int(bin_y.sum())
        table.append({
            "bin": i,
            "prob_lo": float(bin_p[-1]) if n_in > 0 else 0.0,
            "prob_hi": float(bin_p[0]) if n_in > 0 else 0.0,
            "n": n_in,
            "n_pos": n_pos,
            "pos_frac": float(n_pos / max(1, n_in)),
        })

    return table


def compute_ranking_diagnostics(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, Any]:
    """Compute all ranking diagnostics in one call.

    Returns dict with auroc, pr_auc, decile_table.
    Safe for single-class batches (returns None for auroc/pr_auc).
    """
    return {
        "auroc": compute_auroc(y_true, probs),
        "pr_auc": compute_pr_auc(y_true, probs),
        "decile_table": compute_decile_table(y_true, probs),
    }


def compute_canonical_auroc(
    y_true: np.ndarray,
    probs: np.ndarray,
    flip_threshold: float = 0.02,
) -> Dict[str, Any]:
    """Compute AUROC with orientation detection.

    If AUROC(y, 1-prob) > AUROC(y, prob) + flip_threshold, the model
    has flipped orientation (anti-predicting).

    Returns:
        dict with auroc_prob, auroc_inv, auroc_canonical, orientation_flipped.
    """
    y_true = np.asarray(y_true, dtype=bool).ravel()
    probs = np.asarray(probs, dtype=np.float64).ravel()

    auroc_prob = compute_auroc(y_true, probs)
    auroc_inv = compute_auroc(y_true, 1.0 - probs)

    if auroc_prob is None or auroc_inv is None:
        return {
            "auroc_prob": auroc_prob,
            "auroc_inv": auroc_inv,
            "auroc_canonical": auroc_prob,
            "orientation_flipped": False,
        }

    flipped = auroc_inv > auroc_prob + flip_threshold
    canonical = max(auroc_prob, auroc_inv)

    return {
        "auroc_prob": float(auroc_prob),
        "auroc_inv": float(auroc_inv),
        "auroc_canonical": float(canonical),
        "orientation_flipped": bool(flipped),
    }
