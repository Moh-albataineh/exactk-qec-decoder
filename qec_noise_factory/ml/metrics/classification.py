"""
Classification Metrics — Day 16

Per-observable and macro-average metrics for QEC decoding evaluation.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute classification metrics for multi-label QEC decoding.

    Args:
        y_true: (N, num_observables) bool ground truth
        y_pred: (N, num_observables) bool predictions

    Returns:
        Dict with per-observable and macro-average metrics:
          - accuracy: fraction of correct predictions
          - ber: bit error rate (fraction incorrect)
          - balanced_accuracy: mean of TPR and TNR
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    n_samples, n_obs = y_true.shape
    metrics: Dict[str, Any] = {}

    accs = []
    bers = []
    bal_accs = []
    tprs = []
    precisions = []
    fprs = []
    f1s = []

    for obs_idx in range(n_obs):
        yt = y_true[:, obs_idx]
        yp = y_pred[:, obs_idx]

        correct = (yt == yp).sum()
        acc = float(correct / max(1, n_samples))
        ber = 1.0 - acc

        # Balanced accuracy: (TPR + TNR) / 2
        tp = ((yt == True) & (yp == True)).sum()
        tn = ((yt == False) & (yp == False)).sum()
        fn = ((yt == True) & (yp == False)).sum()
        fp = ((yt == False) & (yp == True)).sum()

        tpr = float(tp / max(1, tp + fn))  # sensitivity / recall
        tnr = float(tn / max(1, tn + fp))  # specificity
        bal_acc = (tpr + tnr) / 2.0

        # Day 22: precision, FPR, F1
        precision = float(tp / max(1, tp + fp))
        fpr = float(fp / max(1, fp + tn))  # false positive rate
        f1 = float(2 * tp / max(1, 2 * tp + fp + fn))

        metrics[f"obs_{obs_idx}_accuracy"] = acc
        metrics[f"obs_{obs_idx}_ber"] = ber
        metrics[f"obs_{obs_idx}_balanced_accuracy"] = bal_acc
        metrics[f"obs_{obs_idx}_tpr"] = tpr
        metrics[f"obs_{obs_idx}_tnr"] = tnr
        metrics[f"obs_{obs_idx}_precision"] = precision
        metrics[f"obs_{obs_idx}_fpr"] = fpr
        metrics[f"obs_{obs_idx}_f1"] = f1

        accs.append(acc)
        bers.append(ber)
        bal_accs.append(bal_acc)
        tprs.append(tpr)
        precisions.append(precision)
        fprs.append(fpr)
        f1s.append(f1)

    metrics["macro_accuracy"] = float(np.mean(accs))
    metrics["macro_ber"] = float(np.mean(bers))
    metrics["macro_balanced_accuracy"] = float(np.mean(bal_accs))
    metrics["macro_tpr"] = float(np.mean(tprs))
    metrics["macro_precision"] = float(np.mean(precisions))
    metrics["macro_fpr"] = float(np.mean(fprs))
    metrics["macro_f1"] = float(np.mean(f1s))
    metrics["num_observables"] = n_obs
    metrics["num_samples"] = n_samples

    return metrics
