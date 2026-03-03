"""
Graph Hashing — Day 17

Deterministic SHA256 hash of a GraphSpec.
Invariant: same build_key → same graph → same hash.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from qec_noise_factory.ml.graph.graph_types import GraphSpec


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON with sorted keys and no whitespace variance."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_graph_hash(graph: GraphSpec) -> str:
    """
    Compute a deterministic SHA256 hash of a GraphSpec.

    Hash depends on:
      - build_key (all fields)
      - num_nodes
      - edges (sorted, normalized)
      - node_attrs (sorted keys)
      - edge_attrs (sorted keys)

    Does NOT depend on metadata (which may vary).
    """
    # Normalize edges: sort each edge tuple, then sort list
    normalized_edges = sorted(
        [tuple(sorted(e)) for e in graph.edges]
    )

    hashable = {
        "build_key": graph.build_key.to_dict(),
        "num_nodes": graph.num_nodes,
        "edges": [list(e) for e in normalized_edges],
        "node_attrs": {k: v for k, v in sorted(graph.node_attrs.items())},
        "edge_attrs": {k: v for k, v in sorted(graph.edge_attrs.items())},
    }

    canonical = _canonical_json(hashable)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
