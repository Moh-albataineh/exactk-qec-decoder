"""
Day 44+45 — K-Conditional Standardizer (KCS)

Per-K-bin affine standardization of residual logits:
  Z_norm = (Z_raw - μ_bin) / (σ_bin + eps)

Day 45 upgrade: EMA running stats per bin (Welford-style).
  Training: update running mean/var each batch, use running stats for standardization.
  Eval: use frozen running stats.
Preserves within-bin ordering (monotone affine → slice AUROC unchanged).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class KConditionalStandardizer(nn.Module):
    """Per-K-bin standardization with EMA running stats."""

    def __init__(self, num_bins: int = 12, eps: float = 1e-4, momentum: float = 0.05,
                 stopgrad: bool = False):
        super().__init__()
        self.num_bins = num_bins
        self.eps = eps  # Day 45: raised floor to 1e-4
        self.momentum = momentum
        self.stopgrad = stopgrad  # Day 47: detach mu/sigma in forward
        self._bin_edges: Optional[torch.Tensor] = None

        # EMA running stats (Day 45)
        self.register_buffer('running_mean', torch.zeros(num_bins))
        self.register_buffer('running_var', torch.ones(num_bins))
        self.register_buffer('running_count', torch.zeros(num_bins))

        # Frozen flag
        self._stats_frozen = False

    def fit_bins(self, K_train: np.ndarray) -> None:
        """Compute quantile bin edges from training K values."""
        percentiles = np.linspace(0, 100, self.num_bins + 1)
        edges = np.unique(np.percentile(K_train, percentiles)).astype(np.float32)
        self._bin_edges = torch.from_numpy(edges)

    def _k_to_bin(self, K: torch.Tensor) -> torch.Tensor:
        """Map K values to bin indices."""
        if self._bin_edges is None:
            return torch.zeros_like(K).long()
        edges = self._bin_edges.to(K.device)
        bin_idx = torch.bucketize(K.float(), edges[1:-1])
        return bin_idx.clamp(0, self.num_bins - 1)

    def _update_running_stats(self, Z_raw: torch.Tensor, bin_idx: torch.Tensor):
        """Update EMA running mean/var from current batch (training only)."""
        z_flat = Z_raw.detach()
        if z_flat.dim() > 1:
            z_flat = z_flat.squeeze(-1)

        for b in range(self.num_bins):
            mask = bin_idx == b
            n = mask.sum().item()
            if n < 2:
                continue
            z_bin = z_flat[mask]
            batch_mean = z_bin.mean()
            batch_var = z_bin.var()

            if self.running_count[b] < 1:
                # First time: initialize directly
                self.running_mean[b] = batch_mean
                self.running_var[b] = batch_var
            else:
                # EMA update
                self.running_mean[b] = (1 - self.momentum) * self.running_mean[b] + self.momentum * batch_mean
                self.running_var[b] = (1 - self.momentum) * self.running_var[b] + self.momentum * batch_var

            self.running_count[b] += n

    def forward(self, Z_raw: torch.Tensor, K: torch.Tensor,
                center_only: bool = False) -> torch.Tensor:
        """Standardize Z_raw per K-bin.

        Args:
            Z_raw: residual logits.
            K: syndrome counts.
            center_only: Day 47 Arm C — subtract mean only, no divide.

        Training: update running stats, use running stats for standardization.
        Eval: use frozen running stats.
        """
        bin_idx = self._k_to_bin(K)

        if self.training and not self._stats_frozen:
            self._update_running_stats(Z_raw, bin_idx)

        # Always use running stats
        mu = self.running_mean[bin_idx]      # (B,)
        sigma = torch.sqrt(self.running_var[bin_idx].clamp(min=self.eps * self.eps) + self.eps)

        # Day 47: stop-gradient — detach mu and sigma to prevent normalizer gaming
        if self.stopgrad:
            mu = mu.detach()
            sigma = sigma.detach()

        if Z_raw.dim() > 1:
            mu = mu.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)

        if center_only:
            return Z_raw - mu
        return (Z_raw - mu) / sigma

    def freeze_stats(self, Z_raw_all: torch.Tensor = None, K_all: torch.Tensor = None) -> None:
        """Freeze running stats. Optionally recompute from full training set."""
        if Z_raw_all is not None and K_all is not None:
            bin_idx = self._k_to_bin(K_all)
            z_flat = Z_raw_all.detach()
            if z_flat.dim() > 1:
                z_flat = z_flat.squeeze(-1)
            for b in range(self.num_bins):
                mask = bin_idx == b
                if mask.sum() < 2:
                    continue
                z_bin = z_flat[mask]
                self.running_mean[b] = z_bin.mean()
                self.running_var[b] = z_bin.var()
        self._stats_frozen = True

    def get_stats_dict(self):
        """Return current stats for artifacts."""
        return {
            "running_mean": self.running_mean.cpu().numpy().tolist(),
            "running_var": self.running_var.cpu().numpy().tolist(),
            "running_count": self.running_count.cpu().numpy().tolist(),
            "stats_frozen": self._stats_frozen,
            "bin_edges": self._bin_edges.numpy().tolist() if self._bin_edges is not None else [],
            "eps": self.eps,
            "momentum": self.momentum,
        }
