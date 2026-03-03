"""
Day 42 — Residual vs K Statistics

Per-K (or per-bin) statistics: mean, std, mean|residual|, std|residual|.
Detects heteroscedastic residual variance that may indicate K-dependent leakage.
"""
from __future__ import annotations

from typing import Dict, Any, List

import numpy as np


def compute_residual_vs_k_stats(
    residual_logits: np.ndarray,
    K: np.ndarray,
    min_bucket_size: int = 25,
    use_bins: bool = False,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute per-K statistics for residual logits.

    Args:
        residual_logits: (N,) residual logit values.
        K: (N,) syndrome counts.
        min_bucket_size: minimum samples per exact-K bucket.
        use_bins: if True, use quantile bins instead of exact K.
        n_bins: number of quantile bins (only if use_bins=True).

    Returns:
        dict with buckets, summary slope, overall stats.
    """
    r = np.asarray(residual_logits, dtype=float).ravel()
    k = np.asarray(K, dtype=float).ravel()
    assert len(r) == len(k), f"Length mismatch: {len(r)} vs {len(k)}"

    if use_bins:
        return _compute_binned(r, k, min_bucket_size, n_bins)
    return _compute_exact_k(r, k, min_bucket_size)


def _compute_exact_k(r, k, min_bucket_size):
    """Exact-K buckets."""
    unique_k = np.unique(k.astype(int))
    buckets = []

    for kv in unique_k:
        mask = k.astype(int) == kv
        n = int(mask.sum())
        if n < min_bucket_size:
            continue
        r_slice = r[mask]
        abs_slice = np.abs(r_slice)
        buckets.append({
            "K": int(kv),
            "n": n,
            "mean_residual": float(r_slice.mean()),
            "std_residual": float(r_slice.std()),
            "mean_abs_residual": float(abs_slice.mean()),
            "std_abs_residual": float(abs_slice.std()),
        })

    return _build_result(buckets, r, k)


def _compute_binned(r, k, min_bucket_size, n_bins):
    """Quantile-binned buckets."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(k, percentiles))
    if len(edges) < 2:
        return _build_result([], r, k)

    buckets = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            mask = (k >= lo) & (k < hi)
        else:
            mask = (k >= lo) & (k <= hi)
        n = int(mask.sum())
        if n < min_bucket_size:
            continue
        r_slice = r[mask]
        abs_slice = np.abs(r_slice)
        buckets.append({
            "K_lo": float(lo),
            "K_hi": float(hi),
            "K_center": float((lo + hi) / 2),
            "n": n,
            "mean_residual": float(r_slice.mean()),
            "std_residual": float(r_slice.std()),
            "mean_abs_residual": float(abs_slice.mean()),
            "std_abs_residual": float(abs_slice.std()),
        })

    return _build_result(buckets, r, k)


def _build_result(buckets, r, k):
    """Build result dict with summary statistics."""
    result = {
        "buckets": buckets,
        "n_buckets": len(buckets),
        "overall_mean_residual": float(r.mean()),
        "overall_std_residual": float(r.std()),
        "n_total": len(r),
    }

    if len(buckets) >= 3:
        # Slope: correlation of std(residual) with K (or bin center)
        stds = np.array([b["std_residual"] for b in buckets])
        centers = np.array([b.get("K", b.get("K_center", 0)) for b in buckets])
        if stds.std() > 1e-12 and centers.std() > 1e-12:
            corr = float(np.corrcoef(centers, stds)[0, 1])
        else:
            corr = 0.0
        result["std_vs_K_correlation"] = corr
        result["heteroscedastic"] = abs(corr) > 0.5
    else:
        result["std_vs_K_correlation"] = None
        result["heteroscedastic"] = None

    return result
