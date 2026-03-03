"""
GNN Decoder v0 — Day 18

Simple message-passing GNN for QEC decoding.
No dependency on torch_geometric — manual message passing.

Architecture:
    node_features (B, N, F) → MessagePassingLayers → graph_readout → logits (B, O)

Message passing:
    h_v^{l+1} = ReLU( W_self · h_v^l + W_msg · AGG({h_u^l : u ∈ N(v)}) )

Readout:
    mean pool over nodes → MLP head → logits
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from qec_noise_factory.ml.graph.graph_types import GraphSpec


class MessagePassingLayer(nn.Module):
    """
    Single message-passing layer.

    Update rule:
        h_v' = ReLU( W_self · h_v + W_msg · mean({h_u : u ∈ N(v)}) + bias )
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_msg = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(
        self,
        h: torch.Tensor,         # (B, N, D)
        edge_index: torch.Tensor,  # (2, E) long
    ) -> torch.Tensor:
        B, N, D = h.shape

        # Self transform
        h_self = self.W_self(h)  # (B, N, out_dim)

        # Message aggregation (mean over neighbors)
        src, dst = edge_index[0], edge_index[1]  # (E,)

        # Gather source node features
        h_src = h[:, src, :]  # (B, E, D)

        # Transform messages
        msg = self.W_msg(h_src)  # (B, E, out_dim)

        # Aggregate: scatter mean to destination nodes
        out_dim = h_self.shape[-1]
        h_agg = torch.zeros(B, N, out_dim, device=h.device, dtype=h.dtype)
        count = torch.zeros(B, N, 1, device=h.device, dtype=h.dtype)

        # Expand dst indices for batched scatter
        dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, out_dim)
        h_agg.scatter_add_(1, dst_expanded, msg)

        dst_count = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        count.scatter_add_(1, dst_count, torch.ones_like(dst_count, dtype=h.dtype))

        # Avoid division by zero
        count = count.clamp(min=1.0)
        h_agg = h_agg / count

        # Combine
        h_out = F.relu(h_self + h_agg + self.bias)
        return h_out


class GNNDecoderV0(nn.Module):
    """
    Graph Neural Network decoder for QEC.

    Architecture:
        input (B, N, F) → 2 MP layers → mean pool → MLP head → logits (B, O)

    Args:
        input_dim: feature dimension per node (F)
        output_dim: number of observables
        hidden_dim: hidden dimension for MP layers
        num_mp_layers: number of message passing layers
    """
    needs_graph = True
    model_name = "gnn_v0"

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 32,
        num_mp_layers: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for _ in range(num_mp_layers):
            self.mp_layers.append(MessagePassingLayer(hidden_dim, hidden_dim))

        # Readout head: graph embedding → output logits
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, N, F) node features
            edge_index: (2, E) long tensor of edge indices

        Returns:
            logits: (B, output_dim)
        """
        if edge_index is None:
            raise ValueError("GNNDecoderV0 requires edge_index")

        # Input projection: (B, N, F) → (B, N, hidden_dim)
        h = F.relu(self.input_proj(x))

        # Message passing
        for mp in self.mp_layers:
            h = mp(h, edge_index)

        # Readout: mean pool over nodes → (B, hidden_dim)
        graph_emb = h.mean(dim=1)

        # Head: (B, hidden_dim) → (B, output_dim)
        logits = self.head(graph_emb)

        return logits


def graph_spec_to_edge_index(graph: GraphSpec) -> torch.Tensor:
    """
    Convert GraphSpec edges to PyTorch edge_index tensor.

    Makes edges bidirectional (undirected graph).

    Returns:
        edge_index: (2, 2*E) long tensor
    """
    if not graph.edges:
        # No edges: return empty tensor
        return torch.zeros(2, 0, dtype=torch.long)

    # Make bidirectional
    src, dst = [], []
    for u, v in graph.edges:
        src.extend([u, v])
        dst.extend([v, u])

    return torch.tensor([src, dst], dtype=torch.long)


# ---------------------------------------------------------------------------
# Day 23: GNN Decoder v1 — residual + LayerNorm + dropout + readout options
# ---------------------------------------------------------------------------

class MessagePassingLayerV1(nn.Module):
    """
    Message-passing layer with residual connection, LayerNorm, and dropout.

    h_v' = LayerNorm(h_v + dropout(ReLU(W_self·h + W_msg·agg + b)))
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.W_self = nn.Linear(dim, dim, bias=False)
        self.W_msg = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        h_self = self.W_self(h)

        src, dst = edge_index[0], edge_index[1]
        h_src = h[:, src, :]
        msg = self.W_msg(h_src)

        h_agg = torch.zeros(B, N, D, device=h.device, dtype=h.dtype)
        count = torch.zeros(B, N, 1, device=h.device, dtype=h.dtype)
        dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        h_agg.scatter_add_(1, dst_expanded, msg)
        dst_count = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        count.scatter_add_(1, dst_count, torch.ones_like(dst_count, dtype=h.dtype))
        count = count.clamp(min=1.0)
        h_agg = h_agg / count

        # Residual + LayerNorm + dropout
        h_out = self.norm(h + self.dropout(F.relu(h_self + h_agg + self.bias)))
        return h_out


class AttentionReadout(nn.Module):
    """Learned attention pooling: weighted sum over nodes."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, D) → (B, D)"""
        scores = self.attn(h)           # (B, N, 1)
        weights = torch.softmax(scores, dim=1)  # (B, N, 1)
        return (h * weights).sum(dim=1)  # (B, D)


class GNNDecoderV1(nn.Module):
    """
    GNN Decoder v1 — Day 23.

    Upgrades over v0:
    - Residual connections + LayerNorm in MP layers
    - Dropout (0.1) for regularization
    - Configurable readout: "mean" | "mean_max" | "attn"
    """
    needs_graph = True
    model_name = "gnn_v1"

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 32,
        num_mp_layers: int = 2,
        readout: str = "mean_max",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.readout_type = readout

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.mp_layers = nn.ModuleList()
        for _ in range(num_mp_layers):
            self.mp_layers.append(MessagePassingLayerV1(hidden_dim, dropout))

        # Readout
        if readout == "mean_max":
            head_in = hidden_dim * 2
        elif readout == "attn":
            self.attn_pool = AttentionReadout(hidden_dim)
            head_in = hidden_dim
        else:  # "mean"
            head_in = hidden_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("GNNDecoderV1 requires edge_index")

        h = F.relu(self.input_proj(x))

        for mp in self.mp_layers:
            h = mp(h, edge_index)

        # Readout
        if self.readout_type == "mean_max":
            graph_emb = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        elif self.readout_type == "attn":
            graph_emb = self.attn_pool(h)
        else:
            graph_emb = h.mean(dim=1)

        return self.head(graph_emb)


# ---------------------------------------------------------------------------
# Day 28: GNN Decoder v2 — DEM edge-weighted message passing
# ---------------------------------------------------------------------------

class WeightedMessagePassingLayer(nn.Module):
    """
    Message-passing layer with DEM edge weights.

    Edge weight transform: w_norm = sigmoid(-edge_weight)
        - High matching weight (low error p) → w_norm near 0 → weak msg
        - Low matching weight (high error p) → w_norm near 1 → strong msg
    Wait — reversed: high DEM weight = high confidence = LOW error.
    We want high-confidence edges to pass STRONGER messages.
    So: w_norm = sigmoid(edge_weight) ... but edge_weight = ln((1-p)/p)
    is large when p is small. sigmoid(large) → 1. That's correct.

    Actually let's think again:
        edge_weight = ln((1-p)/p)
        p=0.01 → W≈4.6 → sigmoid(4.6)≈0.99 → strong msg ✓
        p=0.4  → W≈0.4 → sigmoid(0.4)≈0.60 → moderate msg ✓
        p=0.5  → W=0.0 → sigmoid(0)=0.5 → half msg ✓

    So: w_norm = sigmoid(edge_weight) is correct.

    Update:  h' = LayerNorm(h + dropout(ReLU(W_self·h + w_agg + bias)))
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.W_self = nn.Linear(dim, dim, bias=False)
        self.W_msg = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,            # (B, N, D)
        edge_index: torch.Tensor,    # (2, E) long
        edge_weight: torch.Tensor | None = None,  # (E,) float32 — matching weights
    ) -> torch.Tensor:
        B, N, D = h.shape
        h_self = self.W_self(h)

        src, dst = edge_index[0], edge_index[1]
        h_src = h[:, src, :]            # (B, E, D)
        msg = self.W_msg(h_src)          # (B, E, D)

        if edge_weight is not None:
            # Transform: sigmoid(edge_weight) → (0, 1)
            w_norm = torch.sigmoid(edge_weight)  # (E,)
            # Scale messages: (B, E, D) * (1, E, 1)
            msg = msg * w_norm.unsqueeze(0).unsqueeze(-1)

        # Weighted aggregation: sum weighted messages, divide by weight sum
        h_agg = torch.zeros(B, N, D, device=h.device, dtype=h.dtype)
        dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        h_agg.scatter_add_(1, dst_expanded, msg)

        if edge_weight is not None:
            # Normalize by sum of weights per destination node
            w_sum = torch.zeros(B, N, 1, device=h.device, dtype=h.dtype)
            w_vals = torch.sigmoid(edge_weight)  # (E,)
            w_expanded = w_vals.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
            dst_count = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
            w_sum.scatter_add_(1, dst_count, w_expanded)
            w_sum = w_sum.clamp(min=1e-8)
            h_agg = h_agg / w_sum
        else:
            # Unweighted fallback: mean aggregation (same as V1)
            count = torch.zeros(B, N, 1, device=h.device, dtype=h.dtype)
            dst_count = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
            count.scatter_add_(1, dst_count, torch.ones_like(dst_count, dtype=h.dtype))
            count = count.clamp(min=1.0)
            h_agg = h_agg / count

        h_out = self.norm(h + self.dropout(F.relu(h_self + h_agg + self.bias)))
        return h_out


class GNNDecoderV2(nn.Module):
    """
    GNN Decoder v2 — Day 28.

    Upgrades over v1:
    - DEM edge-weighted message passing (sigmoid transform)
    - Graceful fallback: works without edge_weight (degrades to V1-style mean)
    - Boundary node support (handled at feature level, not model level)

    Edge weight transform: sigmoid(W) where W = ln((1-p)/p) from DEM.
    """
    needs_graph = True
    uses_edge_weight = True

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 32,
        num_mp_layers: int = 2,
        readout: str = "mean_max",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.readout_type = readout

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.mp_layers = nn.ModuleList()
        for _ in range(num_mp_layers):
            self.mp_layers.append(WeightedMessagePassingLayer(hidden_dim, dropout))

        # Readout (same options as V1)
        if readout == "mean_max":
            head_in = hidden_dim * 2
        elif readout == "attn":
            self.attn_pool = AttentionReadout(hidden_dim)
            head_in = hidden_dim
        else:  # "mean"
            head_in = hidden_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("GNNDecoderV2 requires edge_index")

        h = F.relu(self.input_proj(x))

        for mp in self.mp_layers:
            h = mp(h, edge_index, edge_weight)

        # Readout
        if self.readout_type == "mean_max":
            graph_emb = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        elif self.readout_type == "attn":
            graph_emb = self.attn_pool(h)
        else:
            graph_emb = h.mean(dim=1)

        return self.head(graph_emb)
