"""
Day 46 — Targeted Leakage Penalty (non-adversarial)

Kills nonlinear K leakage by matching per-K-bin moments of Z_norm to
standard-normal targets. Also penalizes |Z|-to-bin-index envelope correlation.

L_leak = Σ_bins [w1*(μ_b)² + wabs*(mabs_b - mabs_target)² + w2*(m2_b - 1)² + w3*(m3_b)² + w4*(m4_b - 3)²]
L_env  = corr(|Z|, normalized_bin_idx)²
"""
from __future__ import annotations

import math

import torch


# Standard normal targets
_MABS_TARGET = math.sqrt(2.0 / math.pi)  # ≈ 0.7979


def compute_leakage_penalty(
    Z_norm: torch.Tensor,
    K: torch.Tensor,
    bin_edges: torch.Tensor,
    num_bins: int,
    w1: float = 1.0,
    wabs: float = 1.0,
    w2: float = 0.5,
    w3: float = 0.2,
    w4: float = 0.2,
) -> torch.Tensor:
    """Moment-matching leakage penalty.

    Args:
        Z_norm: (B, 1) or (B,) standardized residual logits.
        K: (B,) syndrome counts.
        bin_edges: Tensor of bin edges from KCS.
        num_bins: Number of bins.
        w1..w4, wabs: moment penalty weights.

    Returns:
        Scalar penalty (0.0 if insufficient data).
    """
    z = Z_norm.squeeze(-1) if Z_norm.dim() > 1 else Z_norm

    # Bin assignment
    if bin_edges is not None and len(bin_edges) > 2:
        bin_idx = torch.bucketize(K.float(), bin_edges[1:-1]).clamp(0, num_bins - 1)
    else:
        bin_idx = torch.zeros_like(K).long()

    penalties = []
    for b in range(num_bins):
        mask = bin_idx == b
        n = mask.sum().item()
        if n < 4:  # need enough for moments
            continue
        z_b = z[mask]
        m1 = z_b.mean()
        mabs = z_b.abs().mean()
        m2 = (z_b ** 2).mean()
        m3 = (z_b ** 3).mean()
        m4 = (z_b ** 4).mean()

        pen = (w1 * m1 ** 2
               + wabs * (mabs - _MABS_TARGET) ** 2
               + w2 * (m2 - 1.0) ** 2
               + w3 * m3 ** 2
               + w4 * (m4 - 3.0) ** 2)
        penalties.append(pen)

    if len(penalties) == 0:
        return torch.tensor(0.0, device=Z_norm.device)
    return torch.stack(penalties).mean()


def compute_envelope_penalty(
    Z_norm: torch.Tensor,
    K: torch.Tensor,
    bin_edges: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    """Envelope correlation penalty: corr(|Z|, K)².

    Day 47 fix: correlate directly with K (not bin index) for robustness.
    Kills the |Z|-magnitude-vs-K correlation that MLP probes exploit.
    """
    z = Z_norm.squeeze(-1) if Z_norm.dim() > 1 else Z_norm
    abs_z = z.abs()
    k_float = K.float()

    if abs_z.shape[0] < 4:
        return torch.tensor(0.0, device=Z_norm.device)

    # Pearson correlation between |Z| and K
    abs_z_c = abs_z - abs_z.mean()
    k_c = k_float - k_float.mean()
    cov = (abs_z_c * k_c).mean()
    std_z = abs_z_c.std().clamp(min=1e-6)
    std_k = k_c.std().clamp(min=1e-6)
    corr = cov / (std_z * std_k)

    return corr ** 2


def compute_full_leakage_penalty(
    Z_norm: torch.Tensor,
    K: torch.Tensor,
    bin_edges: torch.Tensor,
    num_bins: int,
    **kwargs,
) -> torch.Tensor:
    """Combined moment-matching + envelope penalty."""
    l_leak = compute_leakage_penalty(Z_norm, K, bin_edges, num_bins, **kwargs)
    l_env = compute_envelope_penalty(Z_norm, K, bin_edges, num_bins)
    return l_leak + l_env
