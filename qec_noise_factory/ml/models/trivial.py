"""
Trivial Baseline Model — Day 16

Always predicts 0 (no logical flips).
Used as a sanity baseline: any real decoder MUST beat this.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TrivialModel(nn.Module):
    """
    Always predicts logits = -10 (→ sigmoid ≈ 0 → predicted flip = 0).
    No trainable parameters.
    """

    def __init__(self, output_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim
        # Register a non-trainable buffer for the constant logit
        self.register_buffer(
            "neg_logit", torch.tensor(-10.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.neg_logit.expand(batch_size, self.output_dim)
