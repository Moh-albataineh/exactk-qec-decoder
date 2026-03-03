"""
Graph Types — Day 17

Deterministic graph representation for QEC detector connectivity.
Designed for GNN-readiness (Day 18) while being circuit-family agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GraphBuildKey:
    """
    Unique key defining how to build a graph.
    Same key → same graph (determinism contract).
    """
    circuit_family: str       # "demo_repetition" | "surface_code" | ...
    num_detectors: int        # total detector count
    num_observables: int      # total observable count
    distance: int = 0         # code distance (0 = N/A for demo)
    rounds: int = 0           # number of QEC rounds
    basis: str = ""           # "X" | "Z" | "" (N/A)
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphSpec:
    """
    Deterministic graph structure.

    Nodes represent detectors. Edges represent connectivity
    (spatial or temporal adjacency in the detector graph).
    """
    build_key: GraphBuildKey
    num_nodes: int
    edges: List[Tuple[int, int]]       # sorted list of (src, dst) pairs, normalized (min,max)
    node_attrs: Dict[str, List[Any]]   # node-level attributes (e.g., type, position)
    edge_attrs: Dict[str, List[Any]]   # edge-level attributes (e.g., weight)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate_edges(self) -> None:
        """
        Validate edge integrity:
          - No self-loops
          - No duplicates
          - All indices in [0, num_nodes)
          - Normalized: each edge is (min, max)
        """
        seen = set()
        for u, v in self.edges:
            if u == v:
                raise ValueError(f"Self-loop detected: ({u}, {v})")
            if u < 0 or v < 0 or u >= self.num_nodes or v >= self.num_nodes:
                raise ValueError(
                    f"Edge ({u}, {v}) out of range [0, {self.num_nodes})"
                )
            normalized = (min(u, v), max(u, v))
            if normalized != (u, v):
                raise ValueError(
                    f"Edge ({u}, {v}) not normalized — expected ({normalized[0]}, {normalized[1]})"
                )
            if normalized in seen:
                raise ValueError(f"Duplicate edge: ({u}, {v})")
            seen.add(normalized)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "build_key": self.build_key.to_dict(),
            "num_nodes": self.num_nodes,
            "edges": [list(e) for e in self.edges],
            "node_attrs": self.node_attrs,
            "edge_attrs": self.edge_attrs,
            "metadata": self.metadata,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphSpec":
        bk = GraphBuildKey(**d["build_key"])
        graph = cls(
            build_key=bk,
            num_nodes=d["num_nodes"],
            edges=[tuple(e) for e in d["edges"]],
            node_attrs=d.get("node_attrs", {}),
            edge_attrs=d.get("edge_attrs", {}),
            metadata=d.get("metadata", {}),
        )
        graph.validate_edges()
        return graph


@dataclass(frozen=True)
class GraphArtifacts:
    """Reference to a saved graph."""
    graph_hash: str
    path: str
    build_key: GraphBuildKey
