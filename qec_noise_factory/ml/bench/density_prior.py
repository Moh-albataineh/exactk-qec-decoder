"""
Density Prior + Truth Gates — Day 37

Provides:
  - K->prior_logit mapping with Laplace smoothing (frozen prior)
  - Iso-density AUROC (within-K-bucket ranking)
  - Residual-K Pearson correlation

The prior captures the density-only signal so the model residual
must learn topology to improve predictions.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple

import numpy as np


def build_k_prior_table(
    K_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """Build K -> log-odds prior from training data with Laplace smoothing.

    For each observed syndrome count K:
        p(y=1|K) = (n_pos + alpha) / (n_total + 2*alpha)
        logit = log(p / (1 - p))

    Unseen K values fall back to global base rate logit.

    Args:
        K_train: (N,) syndrome counts.
        y_train: (N,) boolean labels.
        alpha: Laplace smoothing parameter (default 1.0).

    Returns:
        dict with:
            mapping: {K_value: logit}
            global_logit: fallback for unseen K
            alpha: smoothing used
            k_range: (min_K, max_K)
    """
    K = np.asarray(K_train, dtype=int).ravel()
    y = np.asarray(y_train, dtype=bool).ravel()

    # Global base rate (smoothed)
    n_total = len(y)
    n_pos_global = int(y.sum())
    p_global = (n_pos_global + alpha) / (n_total + 2 * alpha)
    p_global = np.clip(p_global, 1e-4, 1 - 1e-4)
    global_logit = float(np.log(p_global / (1 - p_global)))

    # Per-K mapping
    mapping = {}
    unique_K = np.unique(K)
    for k_val in unique_K:
        mask = K == k_val
        n_k = int(mask.sum())
        n_pos_k = int(y[mask].sum())
        p_k = (n_pos_k + alpha) / (n_k + 2 * alpha)
        p_k = np.clip(p_k, 1e-4, 1 - 1e-4)
        logit_k = float(np.log(p_k / (1 - p_k)))
        mapping[int(k_val)] = logit_k

    return {
        "mapping": mapping,
        "global_logit": global_logit,
        "alpha": alpha,
        "k_range": (int(unique_K.min()), int(unique_K.max())),
        "n_unique_k": len(unique_K),
    }


def lookup_prior_logits(
    K_batch: np.ndarray,
    prior_table: Dict[str, Any],
) -> np.ndarray:
    """Lookup prior logits for a batch of syndrome counts.

    Args:
        K_batch: (B,) syndrome counts.
        prior_table: output of build_k_prior_table.

    Returns:
        (B,) array of prior logits.
    """
    mapping = prior_table["mapping"]
    fallback = prior_table["global_logit"]
    K = np.asarray(K_batch, dtype=int).ravel()
    logits = np.array([mapping.get(int(k), fallback) for k in K], dtype=np.float32)
    return logits


def compute_iso_density_auroc(
    y_true: np.ndarray,
    probs: np.ndarray,
    K: np.ndarray,
    n_min: int = 30,
    n_bins: int = 10,
    canonicalize: bool = False,
) -> Dict[str, Any]:
    """Compute AUROC within K-quantile bins (iso-density evaluation).

    Day 37.2: Uses K-quantile (decile) bins instead of exact-K buckets
    to ensure sufficient samples per bin, especially at high p.
    Day 37.3: canonicalize=True uses max(auroc, 1-auroc) per bin.

    Only bins with n >= n_min AND both classes present qualify.

    Args:
        y_true: (N,) boolean labels.
        probs: (N,) predicted probabilities.
        K: (N,) syndrome counts.
        n_min: minimum samples per bin.
        n_bins: number of quantile bins (default 10 = deciles).

    Returns:
        dict with bucket_list, macro_auroc, n_qualified.
    """
    from qec_noise_factory.ml.metrics.ranking import compute_auroc

    y = np.asarray(y_true, dtype=bool).ravel()
    p = np.asarray(probs, dtype=float).ravel()
    k = np.asarray(K, dtype=float).ravel()

    # Compute quantile bin edges
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(k, percentiles)
    # Make edges unique to avoid empty bins
    edges = np.unique(edges)
    if len(edges) < 2:
        return {"bucket_list": [], "macro_auroc": None, "n_qualified": 0}

    buckets = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            mask = (k >= lo) & (k < hi)
        else:
            mask = (k >= lo) & (k <= hi)  # include right edge for last bin

        n = int(mask.sum())
        if n < n_min:
            continue
        y_bucket = y[mask]
        if y_bucket.all() or (~y_bucket).all():
            continue  # single class — can't compute AUROC
        p_bucket = p[mask]
        auroc = compute_auroc(y_bucket, p_bucket)
        if auroc is not None:
            # Day 37.3: canonicalize per-bin AUROC
            auroc_used = max(auroc, 1.0 - auroc) if canonicalize else auroc
            buckets.append({
                "K_lo": float(lo),
                "K_hi": float(hi),
                "n": n,
                "n_pos": int(y_bucket.sum()),
                "pos_rate": float(y_bucket.mean()),
                "auroc": float(auroc_used),
            })

    if not buckets:
        return {
            "bucket_list": [],
            "macro_auroc": None,
            "n_qualified": 0,
            "canonicalized": canonicalize,
        }

    macro = float(np.mean([b["auroc"] for b in buckets]))
    return {
        "bucket_list": buckets,
        "macro_auroc": macro,
        "n_qualified": len(buckets),
        "canonicalized": canonicalize,
    }


def compute_residual_k_correlation(
    residual_logits: np.ndarray,
    K: np.ndarray,
) -> float:
    """Pearson correlation between residual logits and syndrome count K.

    Low correlation means the residual has learned something beyond density.

    Returns:
        Pearson r (float). Returns 0.0 if constant inputs.
    """
    r = np.asarray(residual_logits, dtype=float).ravel()
    k = np.asarray(K, dtype=float).ravel()

    if r.std() < 1e-10 or k.std() < 1e-10:
        return 0.0

    return float(np.corrcoef(r, k)[0, 1])
