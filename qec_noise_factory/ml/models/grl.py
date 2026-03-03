"""
Day 44+45 — Gradient Reversal Layer (GRL) + K Adversary

Day 45 changes:
  - Shrunk adversary to 1 hidden layer (was 2)
  - λ_adv warmup schedule: linear ramp over warmup epochs
  - Dropout option for stability
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Reverses gradient sign during backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_adv: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to tensor."""
    return GradientReversalFunction.apply(x, lambda_adv)


def compute_warmup_lambda(epoch: int, warmup_epochs: int,
                           lambda_target: float) -> float:
    """Day 45: Linear warmup for λ_adv.
    
    epoch 0 → λ_target * 1/warmup
    epoch warmup-1 → λ_target
    epoch >= warmup → λ_target
    """
    if warmup_epochs <= 0:
        return lambda_target
    if epoch >= warmup_epochs:
        return lambda_target
    return lambda_target * (epoch + 1) / warmup_epochs


class KAdversary(nn.Module):
    """Day 45: Shrunk 1-hidden-layer MLP adversary predicting K from Z_norm.

    Uses MSE on normalized K for stable training.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        # Day 45: single hidden layer (was 2), with optional dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._k_mean = 0.0
        self._k_std = 1.0

    def set_k_stats(self, K_train):
        """Set normalization stats from training K."""
        import numpy as np
        K = np.asarray(K_train, dtype=float)
        self._k_mean = float(K.mean())
        self._k_std = float(max(K.std(), 1e-6))

    def normalize_k(self, K: torch.Tensor) -> torch.Tensor:
        """Normalize K for MSE target."""
        return (K.float() - self._k_mean) / self._k_std

    def forward(self, Z_norm: torch.Tensor, lambda_adv: float = 1.0) -> torch.Tensor:
        """Forward: apply GRL then predict K."""
        Z_rev = gradient_reversal(Z_norm.detach() if not self.training else Z_norm,
                                   lambda_adv)
        return self.net(Z_rev)

    def compute_loss(self, Z_norm: torch.Tensor, K: torch.Tensor,
                     lambda_adv: float = 1.0) -> torch.Tensor:
        """Compute adversary MSE loss with GRL."""
        K_target = self.normalize_k(K).unsqueeze(-1)  # (B, 1)
        K_pred = self.forward(Z_norm, lambda_adv=lambda_adv)
        return nn.functional.mse_loss(K_pred, K_target)
