"""
Day 42 — Exact-K Slice AUROC

Per exact-K value AUROC on residual logits. Stronger topology proof than binned.
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np


def compute_exact_k_slice_auroc(
    y_true: np.ndarray,
    probs: np.ndarray,
    K: np.ndarray,
    top_k_slices: int = 5,
    min_slice_size: int = 25,
    canonicalize: bool = True,
) -> Dict[str, Any]:
    """Compute AUROC within each exact-K slice.

    Args:
        y_true: (N,) boolean labels.
        probs: (N,) predicted probabilities.
        K: (N,) syndrome counts.
        top_k_slices: number of most-frequent K values to evaluate.
        min_slice_size: minimum samples per slice.
        canonicalize: use max(auroc, 1-auroc) per slice.

    Returns:
        dict with slice_aurocs, n_per_slice, summary stats.
    """
    from qec_noise_factory.ml.metrics.ranking import compute_auroc

    y = np.asarray(y_true, dtype=bool).ravel()
    p = np.asarray(probs, dtype=float).ravel()
    k = np.asarray(K, dtype=int).ravel()

    # Find top-K most frequent values
    unique_k, counts = np.unique(k, return_counts=True)
    freq_order = np.argsort(-counts)
    selected = []
    for idx in freq_order:
        if counts[idx] >= min_slice_size:
            selected.append(unique_k[idx])
        if len(selected) >= top_k_slices:
            break

    slices = []
    for kv in sorted(selected):
        mask = k == kv
        n = int(mask.sum())
        y_s = y[mask]
        p_s = p[mask]

        # Skip single-class slices
        if y_s.all() or (~y_s).all():
            slices.append({
                "K": int(kv), "n": n, "n_pos": int(y_s.sum()),
                "auroc": None, "skipped": True, "reason": "single_class",
            })
            continue

        auroc = compute_auroc(y_s, p_s)
        if auroc is not None and canonicalize:
            auroc = max(auroc, 1.0 - auroc)

        slices.append({
            "K": int(kv), "n": n, "n_pos": int(y_s.sum()),
            "auroc": float(auroc) if auroc is not None else None,
            "skipped": False,
        })

    # Summary
    valid_aurocs = [s["auroc"] for s in slices if s["auroc"] is not None and not s.get("skipped")]
    n_above_055 = sum(1 for a in valid_aurocs if a >= 0.55)

    return {
        "slices": slices,
        "n_slices": len(slices),
        "n_valid": len(valid_aurocs),
        "mean_slice_auroc": float(np.mean(valid_aurocs)) if valid_aurocs else None,
        "n_above_0.55": n_above_055,
        "topology_evidence": n_above_055 >= 3,
    }
