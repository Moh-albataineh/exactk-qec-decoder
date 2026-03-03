"""
Day 43 — K-binned Alpha Table

Learnable α(K) indexed by quantile bins. Used for recombination:
  logit_final = logit_prior + α(bin(K)) * logit_residual

Alpha values are sigmoid(alpha_raw) ∈ (0, 1).
TV penalty encourages smoothness: L_tv = sum_i |alpha_raw[i] - alpha_raw[i-1]|
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class AlphaKBinTable(nn.Module):
    """Learnable α(K) table with quantile binning and TV penalty."""

    def __init__(
        self,
        num_bins: int = 12,
        alpha_init: float = 0.0,
        lambda_tv: float = 0.05,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.lambda_tv = lambda_tv
        self.alpha_raw = nn.Parameter(torch.full((num_bins,), alpha_init))
        self._bin_edges: Optional[torch.Tensor] = None  # set by fit_bins()

    def fit_bins(self, K_train: np.ndarray) -> None:
        """Compute quantile bin edges from training K values."""
        percentiles = np.linspace(0, 100, self.num_bins + 1)
        edges = np.percentile(K_train, percentiles)
        edges = np.unique(edges).astype(np.float32)
        self._bin_edges = torch.from_numpy(edges)

    def _k_to_bin(self, K: torch.Tensor) -> torch.Tensor:
        """Map K values to bin indices."""
        if self._bin_edges is None:
            return torch.zeros_like(K).long()
        edges = self._bin_edges.to(K.device)
        # bucketize: find bin index for each K
        bin_idx = torch.bucketize(K.float(), edges[1:-1])  # between interior edges
        return bin_idx.clamp(0, self.alpha_raw.shape[0] - 1)

    def forward(self, K: torch.Tensor) -> torch.Tensor:
        """Return α(K) values for a batch. Shape: (B,)."""
        bin_idx = self._k_to_bin(K)
        alpha = torch.sigmoid(self.alpha_raw[bin_idx])
        return alpha

    def compute_tv_penalty(self) -> torch.Tensor:
        """Total Variation penalty on alpha_raw."""
        if self.alpha_raw.shape[0] < 2:
            return torch.tensor(0.0)
        diffs = self.alpha_raw[1:] - self.alpha_raw[:-1]
        return self.lambda_tv * diffs.abs().sum()

    def get_alpha_profile(self) -> Dict:
        """Return current alpha values and bin info for artifacts."""
        alphas = torch.sigmoid(self.alpha_raw).detach().cpu().numpy().tolist()
        edges = self._bin_edges.numpy().tolist() if self._bin_edges is not None else []
        return {
            "num_bins": self.num_bins,
            "alpha_values": alphas,
            "alpha_raw": self.alpha_raw.detach().cpu().numpy().tolist(),
            "bin_edges": edges,
        }
