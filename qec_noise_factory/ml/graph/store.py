"""
Graph Store — Day 17

Save/load GraphSpec as JSON files with hash-based naming.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from qec_noise_factory.ml.graph.graph_types import GraphSpec, GraphArtifacts
from qec_noise_factory.ml.graph.hash import compute_graph_hash


def save_graph(
    graph: GraphSpec,
    out_dir: str | Path,
) -> GraphArtifacts:
    """
    Save a GraphSpec to JSON.

    Creates:
      <out_dir>/<graph_hash>.json       — full graph
      <out_dir>/<graph_hash>.meta.json  — build_key + hash + schema_version

    Returns GraphArtifacts reference.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_hash = compute_graph_hash(graph)

    # Save full graph
    graph_path = out_dir / f"{graph_hash}.json"
    graph_path.write_text(
        json.dumps(graph.to_dict(), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    # Save metadata
    meta_path = out_dir / f"{graph_hash}.meta.json"
    meta = {
        "graph_hash": graph_hash,
        "build_key": graph.build_key.to_dict(),
        "schema_version": graph.build_key.schema_version,
        "num_nodes": graph.num_nodes,
        "num_edges": len(graph.edges),
        "feature_schema": {
            "dtype": "float32",
            "meaning": "detector_bit",
            "conversion": "bool_to_float (True->1.0, False->0.0)",
            "shape": "(B, N, F)",
        },
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return GraphArtifacts(
        graph_hash=graph_hash,
        path=str(graph_path),
        build_key=graph.build_key,
    )


def load_graph(path: str | Path) -> GraphSpec:
    """Load a GraphSpec from a JSON file."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return GraphSpec.from_dict(data)
