"""
MLP Decoder — Day 16

Simple feedforward decoder: detections → observable flip predictions.
2 hidden layers with ReLU, outputs logits for BCEWithLogitsLoss.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """
    Multi-layer perceptron for QEC decoding.

    Architecture:
        input (num_detectors) → hidden1 → ReLU → hidden2 → ReLU → output (num_observables)

    Args:
        input_dim: number of detectors (X columns)
        output_dim: number of observables (Y columns)
        hidden_dim: hidden layer width
        num_hidden_layers: number of hidden layers (default 2)
        dropout: dropout rate (0 = disabled)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) float tensor of detector values

        Returns:
            logits: (batch_size, output_dim) — pass to sigmoid for probabilities
        """
        return self.net(x)
