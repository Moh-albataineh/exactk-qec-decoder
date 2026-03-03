"""
Density Scrambler — Day 34

Diagnostic tool to detect density shortcut / OR-gate washout.
Preserves the count of active syndrome bits per sample but randomizes positions.
If model performance is unchanged after scrambling, it relies on density, not structure.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def scramble_detector_syndromes(
    det_features: torch.Tensor,
    seed: int = 42,
    syndrome_channel: int = 0,
) -> torch.Tensor:
    """Scramble detector syndrome bits while preserving active count per sample.

    Args:
        det_features: (B, N_det, F) detector node features tensor.
            Channel `syndrome_channel` contains the syndrome bits (0/1).
        seed: random seed for reproducibility.
        syndrome_channel: which feature channel holds the syndrome bits (default 0).

    Returns:
        New tensor (deep copy) with scrambled syndrome positions.
        The boundary node (last node) is always excluded from scrambling.
    """
    # Deep copy — never mutate original
    out = det_features.clone()
    B, N_det, F = out.shape

    rng = np.random.RandomState(seed)

    # Exclude last node (boundary marker) from scrambling
    n_scramble = N_det - 1  # scramble indices [0, n_scramble)

    for b in range(B):
        syndrome = out[b, :n_scramble, syndrome_channel].numpy().copy()
        k_active = int(syndrome.sum())

        if k_active == 0 or k_active == n_scramble:
            continue  # nothing to scramble (all-zero or all-one)

        # Zero out and randomly place k_active ones
        new_positions = rng.choice(n_scramble, size=k_active, replace=False)
        new_syndrome = np.zeros(n_scramble, dtype=syndrome.dtype)
        new_syndrome[new_positions] = 1.0

        out[b, :n_scramble, syndrome_channel] = torch.from_numpy(new_syndrome)

    # Day 37.2: K-preservation invariant assert
    K_orig = det_features[:, :n_scramble, syndrome_channel].sum(dim=1)
    K_scram = out[:, :n_scramble, syndrome_channel].sum(dim=1)
    k_diff = (K_orig - K_scram).abs()
    if k_diff.max().item() > 0.5:
        from qec_noise_factory.ml.bench.reason_codes import ERR_SCRAMBLER_K_MISMATCH
        mismatch_idx = torch.where(k_diff > 0.5)[0].tolist()[:5]
        raise RuntimeError(
            f"{ERR_SCRAMBLER_K_MISMATCH}: K changed for samples {mismatch_idx}. "
            f"max|K_diff|={k_diff.max().item():.0f}"
        )

    return out


def compute_scrambler_delta(
    model: torch.nn.Module,
    det_features: torch.Tensor,
    err_features: torch.Tensor,
    edge_index_d2e: torch.Tensor,
    edge_index_e2d: torch.Tensor,
    y_true: np.ndarray,
    seed: int = 42,
    error_weights: Optional[torch.Tensor] = None,
    observable_mask: Optional[torch.Tensor] = None,
) -> dict:
    """Run model on clean + scrambled inputs, compute AUROC delta.

    Returns:
        dict with auroc_clean, auroc_scrambled, delta_auroc, warn.
    """
    from qec_noise_factory.ml.metrics.ranking import compute_auroc

    model.eval()
    with torch.no_grad():
        # Clean inference
        logits_clean = model(
            det_features, err_features,
            edge_index_d2e, edge_index_e2d,
            error_weights=error_weights,
            observable_mask=observable_mask,
        )
        probs_clean = torch.sigmoid(logits_clean).numpy().ravel()

        # Scrambled inference
        det_scrambled = scramble_detector_syndromes(det_features, seed=seed)
        logits_scrambled = model(
            det_scrambled, err_features,
            edge_index_d2e, edge_index_e2d,
            error_weights=error_weights,
            observable_mask=observable_mask,
        )
        probs_scrambled = torch.sigmoid(logits_scrambled).numpy().ravel()

    y_flat = np.asarray(y_true, dtype=bool).ravel()
    auroc_clean = compute_auroc(y_flat, probs_clean)
    auroc_scrambled = compute_auroc(y_flat, probs_scrambled)

    if auroc_clean is not None and auroc_scrambled is not None:
        delta = auroc_clean - auroc_scrambled
    else:
        delta = None

    warn = False
    if delta is not None and delta < 0.15:
        warn = True  # model may rely on density, not structure

    return {
        "auroc_clean": auroc_clean,
        "auroc_scrambled": auroc_scrambled,
        "delta_auroc": delta,
        "warn_density_leakage": warn,
    }


def compute_iso_scrambler_drop(
    model: torch.nn.Module,
    det_features: torch.Tensor,
    err_features: torch.Tensor,
    edge_index_d2e: torch.Tensor,
    edge_index_e2d: torch.Tensor,
    y_true: np.ndarray,
    seed: int = 42,
    error_weights: Optional[torch.Tensor] = None,
    observable_mask: Optional[torch.Tensor] = None,
    n_min: int = 30,
    n_bins: int = 10,
) -> dict:
    """Day 40: Iso-density scrambler drop on residual-only logits.

    Computes iso_density_auroc on clean residual vs scrambled residual.
    Uses forward_split() if available, else falls back to forward().

    Returns:
        dict with iso_auroc_clean, iso_auroc_scrambled, iso_drop, K array.
    """
    from qec_noise_factory.ml.bench.density_prior import compute_iso_density_auroc

    model.eval()
    use_split = hasattr(model, 'forward_split')

    def _get_residual_probs_and_K(det_feats):
        with torch.no_grad():
            if use_split:
                out = model.forward_split(
                    det_feats, err_features,
                    edge_index_d2e, edge_index_e2d,
                    error_weights=error_weights,
                    observable_mask=observable_mask,
                )
                probs = torch.sigmoid(out['logit_residual']).numpy().ravel()
                K = out['K'].numpy().ravel() if out['K'] is not None else None
            else:
                logits = model(
                    det_feats, err_features,
                    edge_index_d2e, edge_index_e2d,
                    error_weights=error_weights,
                    observable_mask=observable_mask,
                )
                probs = torch.sigmoid(logits).numpy().ravel()
                K = None
        return probs, K

    # Clean residual
    probs_clean, K = _get_residual_probs_and_K(det_features)

    # Scrambled residual
    det_scrambled = scramble_detector_syndromes(det_features, seed=seed)
    probs_scrambled, _ = _get_residual_probs_and_K(det_scrambled)

    y_flat = np.asarray(y_true, dtype=bool).ravel()

    # Compute K from syndrome if not returned by model
    if K is None:
        n_det = det_features.shape[1] - 1  # exclude boundary
        K = det_features[:, :n_det, 0].sum(dim=1).numpy().ravel()

    iso_clean = compute_iso_density_auroc(
        y_flat, probs_clean, K, n_min=n_min, n_bins=n_bins, canonicalize=True)
    iso_scrambled = compute_iso_density_auroc(
        y_flat, probs_scrambled, K, n_min=n_min, n_bins=n_bins, canonicalize=True)

    clean_auroc = iso_clean.get("macro_auroc")
    scram_auroc = iso_scrambled.get("macro_auroc")
    drop = None
    if clean_auroc is not None and scram_auroc is not None:
        drop = clean_auroc - scram_auroc

    return {
        "iso_auroc_clean_residual": clean_auroc,
        "iso_auroc_scrambled_residual": scram_auroc,
        "iso_scrambler_drop_residual": drop,
        "iso_clean_detail": iso_clean,
        "iso_scrambled_detail": iso_scrambled,
    }

