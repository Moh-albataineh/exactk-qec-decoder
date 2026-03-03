"""
Threshold Calibration — Day 21

Grid-search over decision thresholds to maximize balanced accuracy.
Used to fix majority-class collapse when y-rate != 50%.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from qec_noise_factory.ml.metrics.classification import compute_metrics


def calibrate_threshold(
    logits: np.ndarray,
    y_true: np.ndarray,
    metric: str = "bal_acc",
    grid_start: float = 0.05,
    grid_end: float = 0.95,
    grid_step: float = 0.05,
    fpr_lambda: float = 0.25,
) -> Dict[str, Any]:
    """
    Search for the threshold that maximizes the given metric.

    Args:
        logits: (N, O) raw model logits (pre-sigmoid)
        y_true: (N, O) ground truth labels
        metric: metric to maximize — "bal_acc", "f1", or "bal_acc_minus_fpr"
        grid_start: start of threshold grid
        grid_end: end of threshold grid
        grid_step: step size
        fpr_lambda: penalty weight for FPR in "bal_acc_minus_fpr" metric

    Returns:
        Dict with:
          - best_threshold: float
          - best_score: float
          - calibration_metric: str
          - fpr_lambda: float (only for bal_acc_minus_fpr)
          - scores: dict mapping threshold -> score
    """
    y_true = np.asarray(y_true, dtype=bool)
    probs = _sigmoid(np.asarray(logits, dtype=np.float64))

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)

    thresholds = np.arange(grid_start, grid_end + grid_step / 2, grid_step)
    best_t = 0.5
    best_score = -1.0
    scores = {}

    for t in thresholds:
        t_val = round(float(t), 4)
        y_pred = (probs > t_val).astype(bool)
        m = compute_metrics(y_true, y_pred)

        if metric == "bal_acc":
            score = m["macro_balanced_accuracy"]
        elif metric == "f1":
            tpr = m.get("obs_0_tpr", 0.0)
            ppv = _precision(y_true, y_pred)
            score = 2 * tpr * ppv / max(tpr + ppv, 1e-10)
        elif metric == "bal_acc_minus_fpr":
            bal_acc = m["macro_balanced_accuracy"]
            fpr = m.get("macro_fpr", m.get("obs_0_fpr", 0.0))
            score = bal_acc - fpr_lambda * fpr
        else:
            score = m["macro_balanced_accuracy"]

        scores[t_val] = round(score, 6)
        if score > best_score:
            best_score = score
            best_t = t_val

    result = {
        "best_threshold": best_t,
        "best_score": round(best_score, 6),
        "calibration_metric": metric,
        "scores": scores,
    }
    if metric == "bal_acc_minus_fpr":
        result["fpr_lambda"] = fpr_lambda
    return result


def compute_auto_pos_weight(
    y_train: np.ndarray,
    clamp_min: float = 1.0,
    clamp_max: float = 20.0,
) -> float:
    """
    Compute pos_weight = neg_count / pos_count, clamped to [clamp_min, clamp_max].

    Args:
        y_train: (N, O)  training labels
        clamp_min: minimum pos_weight
        clamp_max: maximum pos_weight
    """
    y = np.asarray(y_train, dtype=bool)
    pos = y.sum()
    neg = y.size - pos
    if pos == 0:
        return clamp_max
    ratio = float(neg / pos)
    return float(np.clip(ratio, clamp_min, clamp_max))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro precision for F1 calculation."""
    tp = ((y_true == True) & (y_pred == True)).sum()
    fp = ((y_true == False) & (y_pred == True)).sum()
    return float(tp / max(tp + fp, 1))
