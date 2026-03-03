"""
Day 55 — G1 Probe: Linear Ridge CV (blocking) + MLP telemetry (non-blocking)

Replaces the unstable MLP-only R² gate with a low-variance linear ridge
regression probe using K-fold cross-validation on a large, isolated ProbeSet.

Gate rule:
    G1_score = mean_cv_R2 + std_cv_R2
    PASS if G1_score <= THRESH_G1 (default 0.01)

The MLP probe remains as non-blocking telemetry only.
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch


# ── Default threshold ───────────────────────────────────────────────────
THRESH_G1 = 0.01


# ── Feature builder ─────────────────────────────────────────────────────
def build_probe_features(model: torch.nn.Module,
                         det_feats: torch.Tensor,
                         err_feats: torch.Tensor,
                         ei_d2e: torch.Tensor,
                         ei_e2d: torch.Tensor,
                         error_weights: torch.Tensor = None,
                         observable_mask: torch.Tensor = None,
                         probe_eval_batch_size: int = 128,
                         ) -> np.ndarray:
    """Extract residual logit features for probing, fully detached.

    Returns numpy array of shape (N, 1) — the residual logit norm.
    Gradients are fully detached; this cannot affect the model.

    Mini-batches the forward pass to avoid OOM / CPU stall on large
    ProbeSet (e.g. N=4096 at d=7).
    """
    model.eval()
    device = next(model.parameters()).device
    N = det_feats.shape[0]
    chunks = []
    with torch.no_grad():
        for start in range(0, N, probe_eval_batch_size):
            end = min(start + probe_eval_batch_size, N)
            batch_det = det_feats[start:end].to(device)
            split = model.forward_split(
                batch_det,
                err_feats.to(device),
                ei_d2e.to(device) if not ei_d2e.device == device else ei_d2e,
                ei_e2d.to(device) if not ei_e2d.device == device else ei_e2d,
                error_weights=error_weights.to(device) if error_weights is not None else None,
                observable_mask=observable_mask.to(device) if observable_mask is not None else None)
            z = split['logit_residual_norm']
            chunks.append(z.detach().cpu())
    X = torch.cat(chunks, dim=0).numpy().reshape(-1, 1).astype(np.float64)
    return X


# ── Linear Ridge CV probe (blocking gate) ───────────────────────────────
def _ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge regression: (X'X + αI)^{-1} X'y."""
    n, d = X_train.shape
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train
    I = np.eye(d)
    try:
        w = np.linalg.solve(XtX + alpha * I, Xty)
    except np.linalg.LinAlgError:
        return np.full(X_test.shape[0], y_train.mean())
    return X_test @ w


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score, clamped to [0, 1]."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - ss_res / ss_tot))


def run_linear_probe_cv(X_np: np.ndarray,
                        y_k_np: np.ndarray,
                        cv: int = 5,
                        ridge_alphas: tuple = (0.01, 0.1, 1.0, 10.0, 100.0),
                        seed: int = 42,
                        ) -> Dict[str, Any]:
    """Linear Ridge probe with K-fold cross-validation (pure numpy).

    Args:
        X_np: features (N, D), must be numpy float64, detached.
        y_k_np: syndrome count (N,), must be numpy.
        cv: number of CV folds.
        ridge_alphas: regularization values to try.
        seed: for reproducible fold splits.

    Returns:
        dict with r2_mean, r2_std, r2_score (=mean+std), cv_folds,
        n_samples, alpha_selected.
    """
    X = np.asarray(X_np, dtype=np.float64).reshape(-1, 1) if X_np.ndim == 1 else np.asarray(X_np, dtype=np.float64)
    y = np.asarray(y_k_np, dtype=np.float64).ravel()

    n = len(y)
    if n < cv * 2 or y.std() < 1e-12:
        return {
            "r2_mean": 0.0, "r2_std": 0.0, "r2_score": 0.0,
            "cv_folds": cv, "n_samples": n, "alpha_selected": None,
            "fold_r2s": [],
        }

    # Deterministic K-fold splits
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_sizes = np.full(cv, n // cv, dtype=int)
    fold_sizes[:n % cv] += 1
    folds = []
    start = 0
    for fs in fold_sizes:
        folds.append(indices[start:start + fs])
        start += fs

    # Try each alpha, pick best mean CV R²
    best_alpha = ridge_alphas[0]
    best_mean_r2 = -1.0
    best_fold_r2s = []

    for alpha in ridge_alphas:
        fold_r2s = []
        for i in range(cv):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(cv) if j != i])

            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Standardize (fit on train, transform both)
            x_mean = X_tr.mean(axis=0)
            x_std = np.maximum(X_tr.std(axis=0), 1e-8)
            X_tr_s = (X_tr - x_mean) / x_std
            X_te_s = (X_te - x_mean) / x_std

            # Add intercept
            X_tr_aug = np.hstack([X_tr_s, np.ones((X_tr_s.shape[0], 1))])
            X_te_aug = np.hstack([X_te_s, np.ones((X_te_s.shape[0], 1))])

            y_pred = _ridge_fit_predict(X_tr_aug, y_tr, X_te_aug, alpha)
            fold_r2s.append(_r2_score(y_te, y_pred))

        mean_r2 = np.mean(fold_r2s)
        if mean_r2 > best_mean_r2:
            best_mean_r2 = mean_r2
            best_alpha = alpha
            best_fold_r2s = fold_r2s

    r2_mean = float(np.mean(best_fold_r2s))
    r2_std = float(np.std(best_fold_r2s))

    return {
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "r2_score": r2_mean + r2_std,
        "cv_folds": cv,
        "n_samples": n,
        "alpha_selected": float(best_alpha),
        "fold_r2s": [float(x) for x in best_fold_r2s],
    }


# ── MLP telemetry probe (non-blocking) ──────────────────────────────────
def run_mlp_telemetry(X_np: np.ndarray,
                      y_k_np: np.ndarray,
                      seed: int = 42) -> Dict[str, float]:
    """MLP R² as non-blocking telemetry. Same features as linear probe.

    Uses the existing MLP probe from nonlinear_k_leakage.py.
    """
    from qec_noise_factory.ml.diagnostics.nonlinear_k_leakage import (
        _r_squared_mlp, _r_squared_linear)

    residual = X_np.ravel()
    K = y_k_np.ravel()

    r2_linear_simple = _r_squared_linear(residual, K)
    r2_mlp = _r_squared_mlp(residual, K, seed=seed)

    return {
        "diag_r2_mlp": r2_mlp,
        "diag_r2_linear_simple": r2_linear_simple,
        "nonlinear_leakage_suspected": (r2_mlp - r2_linear_simple) > 0.05,
    }


# ── ProbeSet generator ──────────────────────────────────────────────────
def generate_probe_set(distance: int, p: float, basis: str,
                       n_probe: int, seed: int,
                       corr_strength: float = 0.5,
                       cached_bg=None) -> dict:
    """Generate an isolated ProbeSet for G1 evaluation.

    Uses a deterministic seed (probe_seed = seed + 99991) that is
    independent from train/test seeds.

    Args:
        cached_bg: Optional pre-built BipartiteGraphSpec to avoid
                   redundant graph rebuilds at large d.

    Returns dict with det_feats, err_feats, edge indices, X_raw, Y_raw,
    K, and metadata.
    """
    from qec_noise_factory.ml.bench.regime_lock import RegimeLock, generate_locked_data
    from qec_noise_factory.ml.graph.dem_bipartite import (
        build_bipartite_graph, bipartite_graph_to_tensors)
    from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count

    probe_seed = seed + 99991

    lock = RegimeLock(distance=distance, target_p=p, basis=basis,
                      require_generated=True, n_samples=n_probe,
                      corr_strength=corr_strength, seed=probe_seed)
    X, Y = generate_locked_data(lock)
    n_det = X.shape[1]

    if cached_bg is not None:
        bg = cached_bg
    else:
        bg = build_bipartite_graph(distance=distance, rounds=distance, p=p,
                                   basis=basis,
                                   noise_model="correlated_crosstalk_like")
    ei_d2e, ei_e2d, err_w, obs_mask = bipartite_graph_to_tensors(bg)
    N_det_graph = bg.num_detectors

    B = X.shape[0]
    feats = np.zeros((B, N_det_graph, 2), dtype=np.float32)
    feats[:, :n_det, 0] = X.astype(np.float32)
    feats[:, -1, 1] = 1.0
    det_feats = torch.from_numpy(feats)

    K = compute_syndrome_count(X)

    return {
        "det_feats": det_feats,
        "err_feats": torch.from_numpy(bg.error_weights.reshape(-1, 1)).float(),
        "ei_d2e": ei_d2e, "ei_e2d": ei_e2d,
        "err_w": err_w, "obs_mask": obs_mask,
        "X_raw": X, "Y_raw": Y, "K": K,
        "probe_seed": probe_seed, "n_probe": B,
        "n_det": n_det, "N_det_graph": N_det_graph,
    }


# ── Full G1 evaluation ──────────────────────────────────────────────────
def evaluate_g1(model: torch.nn.Module,
                probe_set: dict,
                cv: int = 5,
                probe_seed: int = None,
                thresh: float = THRESH_G1,
                ) -> Dict[str, Any]:
    """Full G1 evaluation: linear probe (blocking) + MLP telemetry.

    Args:
        model: trained FactorGraphDecoderV1
        probe_set: from generate_probe_set()
        cv: number of CV folds
        probe_seed: seed for fold splits (defaults to probe_set's seed)
        thresh: G1 pass threshold

    Returns:
        dict with linear probe results, MLP telemetry, G1 pass/fail.
    """
    if probe_seed is None:
        probe_seed = probe_set["probe_seed"]

    # Extract features (fully detached)
    X_np = build_probe_features(
        model, probe_set["det_feats"], probe_set["err_feats"],
        probe_set["ei_d2e"], probe_set["ei_e2d"],
        error_weights=probe_set["err_w"],
        observable_mask=probe_set["obs_mask"])

    K_np = probe_set["K"].astype(np.float64)

    # Linear probe (blocking)
    linear = run_linear_probe_cv(X_np, K_np, cv=cv, seed=probe_seed)

    # MLP telemetry (non-blocking)
    mlp = run_mlp_telemetry(X_np, K_np, seed=probe_seed)

    g1_pass = linear["r2_score"] <= thresh

    return {
        "G1_pass": g1_pass,
        "G1_r2_linear_mean": linear["r2_mean"],
        "G1_r2_linear_std": linear["r2_std"],
        "G1_r2_linear_score": linear["r2_score"],
        "G1_thresh": thresh,
        "probe_n": linear["n_samples"],
        "probe_seed": probe_seed,
        "probe_cv_folds": linear["cv_folds"],
        "ridge_alpha": linear["alpha_selected"],
        "fold_r2s": linear["fold_r2s"],
        **mlp,
    }


# ── Day 62: Aligned G1 Evaluation (source of truth) ────────────────────

def evaluate_g1_aligned(
    model: torch.nn.Module,
    probe_set: dict,
    ortho_state: dict,
    cv: int = 5,
    probe_seed: int = None,
    thresh: float = THRESH_G1,
) -> Dict[str, Any]:
    """Day 62: Single-pass aligned G1 evaluation.

    Computes G1_raw_probe and G1_post_probe from a SINGLE forward_split()
    call with ALL ortho paths disabled.  G1_post is computed algebraically
    by simulating the forward correction on the extracted Z_raw tensor.

    This eliminates the Day 61 bug where two separate forward_split() calls
    with different ortho state caused G1_post != G1_raw under zero correction.

    Args:
        model: trained FactorGraphDecoderV1
        probe_set: from generate_probe_set()
        ortho_state: dict with:
            'beta_E': float — applied beta (0 for Control/ShieldOnly)
            'muK_probe': float — K mean from probe
            'w': float — ramp schedule weight
            'eff_corr_expected': bool — True if forward correction is applied
        cv: number of CV folds
        probe_seed: seed for fold splits
        thresh: G1 pass threshold

    Returns:
        dict with G1_raw_probe, G1_post_probe, alignment invariant fields.
    """
    if probe_seed is None:
        probe_seed = probe_set["probe_seed"]

    # ── Step 1: Extract Z_raw with ALL ortho OFF ────────────────────
    saved_attrs = {}
    for attr in ['_k_ortho_probeset_synced', '_k_ortho_epoch_rolling',
                 '_k_ortho_frozen', '_k_ortho_module', '_k_ortho_window']:
        saved_attrs[attr] = getattr(model, attr, None)
        setattr(model, attr, None)
    old_use = getattr(model, 'fg_use_k_ortho', False)
    model.fg_use_k_ortho = False

    X_raw = build_probe_features(
        model, probe_set["det_feats"], probe_set["err_feats"],
        probe_set["ei_d2e"], probe_set["ei_e2d"],
        error_weights=probe_set.get("err_w"),
        observable_mask=probe_set.get("obs_mask"))

    # Log tensor metadata (once per call)
    probe_tensor_name = "logit_residual_norm"
    probe_tensor_shape = list(X_raw.shape)

    # Restore model state
    model.fg_use_k_ortho = old_use
    for attr, val in saved_attrs.items():
        setattr(model, attr, val)

    K_np = np.asarray(probe_set["K"], dtype=np.float64).ravel()

    # ── Step 2: G1_raw_probe ────────────────────────────────────────
    linear_raw = run_linear_probe_cv(X_raw, K_np, cv=cv, seed=probe_seed)
    G1_raw_probe = linear_raw["r2_score"]

    # ── Step 3: Simulate G1_post if forward correction is applied ──
    beta_E = ortho_state.get("beta_E", 0.0)
    muK = ortho_state.get("muK_probe", 0.0)
    w = ortho_state.get("w", 0.0)
    eff_corr_expected = ortho_state.get("eff_corr_expected", False)

    if eff_corr_expected and abs(beta_E) > 1e-8 and w > 0:
        # Simulate: Z_post = Z_raw - w * beta_E * (K - muK)
        Kc = (K_np - muK).reshape(-1, 1)
        correction = w * beta_E * Kc
        X_post = X_raw - correction

        # eff_corr_ratio
        corr_norm = float(np.linalg.norm(correction))
        z_norm_val = float(np.linalg.norm(X_raw))
        eff_corr_ratio = corr_norm / (z_norm_val + 1e-6)

        linear_post = run_linear_probe_cv(X_post, K_np, cv=cv, seed=probe_seed)
        G1_post_probe = linear_post["r2_score"]
    else:
        # No forward correction — G1_post = G1_raw by definition
        G1_post_probe = G1_raw_probe
        eff_corr_ratio = 0.0

    # ── Step 4: Invariant check ─────────────────────────────────────
    alignment_abs_diff = abs(G1_post_probe - G1_raw_probe)
    if eff_corr_ratio < 1e-8:
        # No forward correction → G1_post MUST equal G1_raw
        alignment_invariant_pass = alignment_abs_diff <= 1e-6
    else:
        # Forward correction applied — diff is expected
        alignment_invariant_pass = True

    G1_delta_probe = G1_post_probe - G1_raw_probe

    return {
        # Core G1 metrics
        "G1_raw_probe": float(G1_raw_probe),
        "G1_post_probe": float(G1_post_probe),
        "G1_delta_probe": float(G1_delta_probe),
        "G1_pass_raw": G1_raw_probe <= thresh,
        "G1_pass_post": G1_post_probe <= thresh,
        # Alignment telemetry
        "probe_tensor_name": probe_tensor_name,
        "probe_tensor_shape": probe_tensor_shape,
        "alignment_invariant_pass": alignment_invariant_pass,
        "alignment_abs_diff": float(alignment_abs_diff),
        "eff_corr_ratio": float(eff_corr_ratio),
        # Probe metadata
        "probe_n": linear_raw["n_samples"],
        "probe_seed": probe_seed,
        "probe_cv_folds": linear_raw["cv_folds"],
        "ridge_alpha_raw": linear_raw["alpha_selected"],
        "G1_r2_raw_mean": linear_raw["r2_mean"],
        "G1_r2_raw_std": linear_raw["r2_std"],
    }
