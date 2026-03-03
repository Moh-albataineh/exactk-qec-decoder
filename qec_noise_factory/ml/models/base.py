"""
Decoder Base Interface — Day 18

Common interface for MLP, GNN, and future decoder models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from qec_noise_factory.ml.graph.graph_types import GraphSpec


class DecoderBase(nn.Module, ABC):
    """
    Base class for all QEC decoder models.

    Subclasses must implement forward() and set:
      - needs_graph: bool  (True if model requires graph_spec)
      - model_name: str
    """
    needs_graph: bool = False
    model_name: str = "base"

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, input_dim) for MLP or (B, N, F) for GNN
            edge_index: (2, E) edge index tensor (GNN only)

        Returns:
            logits: (B, output_dim)
        """
        ...
