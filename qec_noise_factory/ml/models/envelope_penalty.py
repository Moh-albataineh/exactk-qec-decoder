"""
Day 53/54 — Envelope Independence Penalty

Differentiable penalty that prevents the residual from encoding K.
No σ normalization, no GRL — just direct correlation penalties.

Day 53:  L_env = Corr(|z|, K)² + Corr(z², K)²
Day 54:  L_env = Corr(z, K)²   + Corr(|z|, K)²    (remove z² outlier term)

Gradients flow to the model, making this a training-time regularizer.
"""
from __future__ import annotations

import torch


def corrcoef_1d(x: torch.Tensor, y: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """Stable Pearson correlation between 1-D tensors.

    Returns scalar tensor in [-1, 1].  If either input has zero
    variance, returns 0 (not NaN).
    """
    assert x.dim() == 1 and y.dim() == 1 and x.shape == y.shape
    n = x.shape[0]
    if n < 2:
        return torch.zeros(1, device=x.device, dtype=x.dtype).squeeze()

    x_c = x - x.mean()
    y_c = y - y.mean()
    cov = (x_c * y_c).sum() / (n - 1)
    std_x = x_c.pow(2).sum().div(n - 1).clamp(min=eps * eps).sqrt()
    std_y = y_c.pow(2).sum().div(n - 1).clamp(min=eps * eps).sqrt()
    return cov / (std_x * std_y + eps)


def envelope_penalty(z_used_real: torch.Tensor,
                     K_real: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """Differentiable envelope-independence penalty (Day 54 refined).

    Args:
        z_used_real: residual output (B,) or (B,1).  Gradients MUST flow.
        K_real: true K values (B,) as float. NOT bin index.
        eps: numerical stability constant.

    Returns:
        L_env = Corr(z, K)² + Corr(|z|, K)²   (scalar ≥ 0)

    Day 54 change: replaced Corr(z², K)² with Corr(z, K)² to avoid
    quadratic outlier gradients that crush topology.
    """
    z = z_used_real.view(-1)
    K = K_real.float().view(-1)
    assert z.shape == K.shape

    if z.shape[0] < 4:
        return torch.zeros(1, device=z.device, dtype=z.dtype).squeeze()

    env_mean = corrcoef_1d(z, K, eps=eps)       # first-moment leakage
    env_abs = corrcoef_1d(z.abs(), K, eps=eps)   # envelope leakage
    return env_mean.pow(2) + env_abs.pow(2)

