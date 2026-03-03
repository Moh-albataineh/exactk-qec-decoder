"""
Day 42 — Nonlinear K Leakage Diagnostic

Measures whether the residual logit leaks K information through nonlinear
channels (K², K³, sqrt(K), log(1+K)) or via a small MLP regressor.
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returns 0.0 if degenerate."""
    a = a.ravel().astype(float)
    b = b.ravel().astype(float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _r_squared_linear(X: np.ndarray, y: np.ndarray) -> float:
    """R² from simple OLS (no gradient, eval-only)."""
    X = X.reshape(-1, 1).astype(float)
    y = y.ravel().astype(float)
    n = len(y)
    if n < 5 or y.std() < 1e-12:
        return 0.0
    X_aug = np.hstack([X, np.ones((n, 1))])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_hat = X_aug @ beta
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12)))
    except Exception:
        return 0.0


def _r_squared_mlp(residual: np.ndarray, K: np.ndarray, seed: int = 42) -> float:
    """R² from a tiny 2-layer MLP (eval-only, no gradient to main model).

    Uses numpy-only implementation for portability.
    Architecture: input(1) -> hidden(8) -> ReLU -> hidden(4) -> output(1)
    Fitted via scipy.optimize.minimize (L-BFGS-B).
    """
    X = residual.ravel().astype(float)
    y = K.ravel().astype(float)
    n = len(y)
    if n < 20 or y.std() < 1e-12:
        return 0.0

    # Normalize
    X_mean, X_std = X.mean(), max(X.std(), 1e-8)
    y_mean, y_std = y.mean(), max(y.std(), 1e-8)
    X_n = (X - X_mean) / X_std
    y_n = (y - y_mean) / y_std

    rng = np.random.RandomState(seed)
    h1, h2 = 8, 4

    # Pack/unpack parameters
    def _sizes():
        return [(1, h1), (h1,), (h1, h2), (h2,), (h2, 1), (1,)]

    n_params = sum(np.prod(s) for s in _sizes())

    def _unpack(theta):
        parts = []
        idx = 0
        for s in _sizes():
            sz = int(np.prod(s))
            parts.append(theta[idx:idx+sz].reshape(s))
            idx += sz
        return parts  # W1, b1, W2, b2, W3, b3

    def _forward(theta, X_in):
        W1, b1, W2, b2, W3, b3 = _unpack(theta)
        h = np.maximum(0, X_in.reshape(-1, 1) @ W1 + b1)
        h = np.maximum(0, h @ W2 + b2)
        return (h @ W3 + b3).ravel()

    def _loss(theta):
        pred = _forward(theta, X_n)
        return 0.5 * np.mean((pred - y_n) ** 2)

    theta0 = rng.randn(n_params) * 0.1

    try:
        from scipy.optimize import minimize
        res = minimize(_loss, theta0, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-8})
        pred = _forward(res.x, X_n)
        ss_res = ((y_n - pred) ** 2).sum()
        ss_tot = ((y_n - y_n.mean()) ** 2).sum()
        return float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12)))
    except ImportError:
        return -1.0  # scipy not available
    except Exception:
        return 0.0


def measure_nonlinear_k_leakage(
    residual_logits: np.ndarray,
    K: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    """Measure nonlinear K leakage in residual logits.

    Returns dict with correlations, R² linear, R² MLP.
    """
    r = np.asarray(residual_logits, dtype=float).ravel()
    k = np.asarray(K, dtype=float).ravel()

    result = {
        "corr_K": _pearson_corr(r, k),
        "corr_K2": _pearson_corr(r, k ** 2),
        "corr_K3": _pearson_corr(r, k ** 3),
        "corr_sqrtK": _pearson_corr(r, np.sqrt(np.maximum(k, 0))),
        "corr_logK": _pearson_corr(r, np.log1p(np.maximum(k, 0))),
        "R2_linear": _r_squared_linear(r, k),
        "R2_mlp": _r_squared_mlp(r, k, seed=seed),
        "n_samples": len(r),
    }

    # Evidence of nonlinear leakage if MLP R² >> linear R²
    r2_gap = result["R2_mlp"] - result["R2_linear"]
    result["R2_gap_mlp_vs_linear"] = r2_gap
    result["nonlinear_leakage_detected"] = r2_gap > 0.05

    return result
