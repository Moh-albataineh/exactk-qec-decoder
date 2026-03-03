"""
Graph Builder — Day 17

Deterministic graph construction from a GraphBuildKey.
Same inputs → same nodes/edges (ordering guaranteed).

Currently supports:
  - demo_repetition: linear chain of detectors
  - (extensible) surface_code: 2D lattice (future)
"""
from __future__ import annotations

from typing import List, Tuple

from qec_noise_factory.ml.graph.graph_types import GraphBuildKey, GraphSpec


def build_graph(key: GraphBuildKey) -> GraphSpec:
    """
    Build a deterministic graph from a GraphBuildKey.

    Dispatches to the appropriate builder based on circuit_family.
    """
    family = key.circuit_family.lower()

    if family in ("demo_repetition", "repetition"):
        graph = _build_demo_repetition(key)
    elif family in ("surface_code", "surface"):
        graph = _build_surface_code_stub(key)
    else:
        # Fallback: fully connected graph
        graph = _build_generic(key)

    graph.validate_edges()
    return graph


def _build_demo_repetition(key: GraphBuildKey) -> GraphSpec:
    """
    Demo repetition code graph.

    Topology: linear chain of detectors.
      D0 — D1 — D2 — ... — D(n-1)

    For num_detectors=2: nodes=[0,1], edges=[(0,1)]
    """
    n = key.num_detectors
    edges: List[Tuple[int, int]] = []

    # Linear chain: each detector connected to next
    for i in range(n - 1):
        edges.append((i, i + 1))

    # Node attributes: position along chain, type
    node_types = ["detector"] * n
    node_positions = list(range(n))

    return GraphSpec(
        build_key=key,
        num_nodes=n,
        edges=sorted(edges),  # deterministic ordering
        node_attrs={
            "type": node_types,
            "position": node_positions,
        },
        edge_attrs={
            "weight": [1.0] * len(edges),
        },
        metadata={
            "topology": "linear_chain",
            "circuit_family": key.circuit_family,
        },
    )


def _build_surface_code_stub(key: GraphBuildKey) -> GraphSpec:
    """
    Surface code graph stub — placeholder for Day 18+.

    Creates a 2D grid of detectors based on distance and rounds.
    For now, builds a simple grid matching num_detectors.
    """
    n = key.num_detectors
    d = key.distance if key.distance > 0 else 3
    r = key.rounds if key.rounds > 0 else d

    edges: List[Tuple[int, int]] = []

    # Simple linear fallback for now
    for i in range(n - 1):
        edges.append((i, i + 1))

    return GraphSpec(
        build_key=key,
        num_nodes=n,
        edges=sorted(edges),
        node_attrs={
            "type": ["detector"] * n,
            "position": list(range(n)),
        },
        edge_attrs={
            "weight": [1.0] * len(edges),
        },
        metadata={
            "topology": "surface_code_stub",
            "circuit_family": key.circuit_family,
            "distance": d,
            "rounds": r,
        },
    )


def _build_generic(key: GraphBuildKey) -> GraphSpec:
    """Fallback: fully connected graph for unknown circuit families."""
    n = key.num_detectors
    edges: List[Tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))

    return GraphSpec(
        build_key=key,
        num_nodes=n,
        edges=sorted(edges),
        node_attrs={
            "type": ["detector"] * n,
            "position": list(range(n)),
        },
        edge_attrs={
            "weight": [1.0] * len(edges),
        },
        metadata={
            "topology": "fully_connected",
            "circuit_family": key.circuit_family,
        },
    )


def build_key_from_meta(meta) -> GraphBuildKey:
    """
    Extract a GraphBuildKey from shard metadata.

    Parses circuit_family and params from params_canonical.
    """
    import json

    try:
        params = json.loads(meta.params_canonical)
        circuit = params.get("circuit", {})
    except (json.JSONDecodeError, AttributeError):
        circuit = {}

    return GraphBuildKey(
        circuit_family=circuit.get("type", "unknown"),
        num_detectors=meta.num_detectors,
        num_observables=meta.num_observables,
        distance=circuit.get("distance", 0),
        rounds=circuit.get("rounds", 0),
        basis=circuit.get("basis", ""),
    )
