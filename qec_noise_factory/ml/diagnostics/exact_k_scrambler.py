"""
Day 46 — Exact-K Scrambler Metric

Physics-correct scrambler evaluation: compare AUROC on Z_norm(clean) vs
Z_norm(scrambled) within each exact-K slice. Avoids bin-width artifacts.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_exact_k_scrambler_metric(
    y_test: np.ndarray,
    z_norm_clean: np.ndarray,
    z_norm_scr: np.ndarray,
    K_test: np.ndarray,
    min_k_support: int = 20,
    min_pos_neg: int = 3,
) -> Dict[str, object]:
    """Exact-K scrambler drop metric.

    For each K value with sufficient support:
      AUROC_clean_k = AUROC(y, Z_norm_clean | K=k)
      AUROC_scr_k   = AUROC(y, Z_norm_scr   | K=k)
      drop_k        = AUROC_clean_k - AUROC_scr_k

    Args:
        y_test: (N,) binary labels.
        z_norm_clean: (N,) Z_norm from clean graphs.
        z_norm_scr: (N,) Z_norm from scrambled graphs.
        K_test: (N,) syndrome counts.
        min_k_support: minimum samples per K slice.
        min_pos_neg: minimum of each class per slice.

    Returns:
        dict with mean_slice_clean, mean_slice_scr, mean_drop,
        fraction_pass (drop >= 0.10), slice details.
    """
    from qec_noise_factory.ml.metrics.ranking import compute_auroc

    y = np.asarray(y_test).ravel().astype(bool)
    K = np.asarray(K_test).ravel()
    z_c = np.asarray(z_norm_clean).ravel()
    z_s = np.asarray(z_norm_scr).ravel()

    unique_k = np.unique(K)
    slices = []

    for k_val in unique_k:
        mask = K == k_val
        n = mask.sum()
        if n < min_k_support:
            continue
        y_k = y[mask]
        n_pos = y_k.sum()
        n_neg = n - n_pos
        if n_pos < min_pos_neg or n_neg < min_pos_neg:
            continue

        auc_c = compute_auroc(y_k, z_c[mask])
        if auc_c is None:
            continue
        auc_s = compute_auroc(y_k, z_s[mask])
        if auc_s is None:
            auc_s = 0.5

        drop = auc_c - auc_s
        slices.append({
            "K": int(k_val),
            "n": int(n),
            "auroc_clean": float(auc_c),
            "auroc_scr": float(auc_s),
            "drop": float(drop),
        })

    if len(slices) == 0:
        return {
            "mean_slice_clean_exactK": None,
            "mean_slice_scr_exactK": None,
            "mean_drop_exactK": None,
            "fraction_pass": None,
            "n_slices": 0,
            "slices": [],
        }

    mean_clean = np.mean([s["auroc_clean"] for s in slices])
    mean_scr = np.mean([s["auroc_scr"] for s in slices])
    mean_drop = np.mean([s["drop"] for s in slices])
    n_pass = sum(1 for s in slices if s["drop"] >= 0.10)
    frac_pass = n_pass / len(slices)

    return {
        "mean_slice_clean_exactK": float(mean_clean),
        "mean_slice_scr_exactK": float(mean_scr),
        "mean_drop_exactK": float(mean_drop),
        "fraction_pass": float(frac_pass),
        "n_slices": len(slices),
        "slices": slices,
    }
