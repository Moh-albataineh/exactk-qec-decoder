"""
DEM Bipartite Factor Graph — Day 32

Builds a bipartite factor graph (Detectors ↔ Error-Mechanisms) from a Stim
Detector Error Model (DEM).  Unlike the clique-expanded DEM graph (Day 27),
this representation preserves k>2 hyperedge structure exactly.

Node sets:
    V_D  — detector nodes (num_detectors + 1 boundary)
    V_E  — error-mechanism nodes (merged by support set)

Edges:
    D→E and E→D  — each detector in an error's support set is connected

Merge rule:
    If multiple DEM mechanisms share the same (sorted_detectors, obs_mask):
        p_merge = 1 − Π(1 − p_i)

Hashing:
    dem_topology_hash = sha256(canonical_json(nodes, error_keys, edges, masks))
    Deterministic regardless of Stim parsing order.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qec_noise_factory.ml.graph.dem_graph import (
    DemBuildKey,
    _P_CLAMP_HI,
    _P_CLAMP_LO,
    matching_weight,
    merge_probabilities,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class BipartiteGraphSpec:
    """Bipartite factor-graph representation of a DEM.

    Two disjoint node sets:
      - Detector nodes V_D  (indices 0 .. num_detectors-1, plus boundary)
      - Error nodes    V_E  (indices 0 .. num_errors-1)

    Edges connect each error node to every detector in its support set.
    """
    num_detectors: int          # |V_D| including boundary node
    num_errors: int             # |V_E| (after merge)
    has_boundary_node: bool
    edge_index_d2e: np.ndarray  # (2, E) int64 — (detector_idx, error_idx)
    edge_index_e2d: np.ndarray  # (2, E) int64 — (error_idx, detector_idx)
    error_weights: np.ndarray   # (num_errors,) float32 — ln((1-p)/p)
    error_probs: np.ndarray     # (num_errors,) float32 — merged probability
    observable_mask: np.ndarray # (num_errors,) bool — True if error affects observable
    error_keys: List[str]       # canonical key per error node
    dem_topology_hash: str      # SHA-256 of canonical graph structure
    build_key: DemBuildKey
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_detectors": self.num_detectors,
            "num_errors": self.num_errors,
            "has_boundary_node": self.has_boundary_node,
            "edge_index_d2e": self.edge_index_d2e.tolist(),
            "edge_index_e2d": self.edge_index_e2d.tolist(),
            "error_weights": self.error_weights.tolist(),
            "error_probs": self.error_probs.tolist(),
            "observable_mask": self.observable_mask.tolist(),
            "error_keys": self.error_keys,
            "dem_topology_hash": self.dem_topology_hash,
            "build_key": self.build_key.to_dict(),
            "stats": self.stats,
        }


# ---------------------------------------------------------------------------
# DEM Parsing → Bipartite Structure (no clique expansion)
# ---------------------------------------------------------------------------

def _parse_dem_bipartite(
    dem,
    num_detectors: int,
) -> Tuple[Dict[str, Dict], Dict[str, int]]:
    """Parse DEM into error-mechanism groups WITHOUT clique expansion.

    Each error mechanism is keyed by (sorted_detector_indices, observable_mask).
    Multiple DEM terms sharing the same key are merged via p_merge.

    Returns:
        mechanisms: dict mapping canonical_key → {
            "det_indices": sorted list of detector indices,
            "obs_mask": bool (True if any observable target),
            "probs": list of probabilities to merge,
        }
        info: dict with raw counts
    """
    boundary_node = num_detectors  # virtual boundary node index

    mechanisms: Dict[str, Dict] = {}
    info = {
        "num_terms": 0,
        "num_boundary_terms": 0,
        "num_single_det": 0,
        "num_pair_det": 0,
        "num_hyperedge_gt2": 0,
        "num_observables_only": 0,
        "k_distribution": {},  # k → count
        "term_probs": [],
    }

    for instruction in dem.flattened():
        if instruction.type != "error":
            continue

        info["num_terms"] += 1
        prob = instruction.args_copy()[0]
        info["term_probs"].append(float(prob))

        # Extract detector indices and observable info
        det_ids = []
        has_observable = False
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                det_ids.append(target.val)
            elif target.is_logical_observable_id():
                has_observable = True

        k = len(det_ids)
        info["k_distribution"][k] = info["k_distribution"].get(k, 0) + 1

        if k == 0:
            info["num_observables_only"] += 1
            if not has_observable:
                continue
            # Observable-only term: create an error node with boundary connection
            det_ids_sorted = [boundary_node]
        elif k == 1:
            info["num_single_det"] += 1
            info["num_boundary_terms"] += 1
            det_ids_sorted = sorted([det_ids[0], boundary_node])
        elif k == 2:
            info["num_pair_det"] += 1
            det_ids_sorted = sorted(det_ids)
        else:
            info["num_hyperedge_gt2"] += 1
            det_ids_sorted = sorted(det_ids)

        # Canonical key: (sorted_detectors, obs_mask)
        obs_key = "1" if has_observable else "0"
        canonical_key = f"{det_ids_sorted}|{obs_key}"

        if canonical_key not in mechanisms:
            mechanisms[canonical_key] = {
                "det_indices": det_ids_sorted,
                "obs_mask": has_observable,
                "probs": [],
            }
        mechanisms[canonical_key]["probs"].append(float(prob))

    return mechanisms, info


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_bipartite_graph(
    distance: int,
    rounds: int,
    p: float,
    basis: str,
    noise_model: str = "baseline_symmetric",
    physics_hash: str = "",
) -> BipartiteGraphSpec:
    """Build a bipartite factor graph from circuit parameters.

    Steps:
        1. Rebuild Stim circuit
        2. Extract DEM (decompose_errors=True)
        3. Parse DEM → error mechanisms (no clique expansion)
        4. Merge duplicate mechanisms (same support set)
        5. Compute matching weights
        6. Build edge indices
        7. Canonical ordering + hash

    Returns:
        BipartiteGraphSpec with deterministic hash.
    """
    from qec_noise_factory.ml.stim.rebuild import rebuild_stim_circuit

    circuit = rebuild_stim_circuit(
        distance=distance, rounds=rounds, p=p, basis=basis,
        noise_model=noise_model,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    num_detectors = circuit.num_detectors
    boundary_node = num_detectors  # index of boundary node
    total_det_nodes = num_detectors + 1  # +1 for boundary

    # Parse DEM
    mechanisms, parse_info = _parse_dem_bipartite(dem, num_detectors)

    # Sort mechanisms by canonical key for determinism
    sorted_keys = sorted(mechanisms.keys())

    # Build merged error nodes
    error_keys: List[str] = []
    error_det_indices: List[List[int]] = []
    error_probs_list: List[float] = []
    error_obs_mask: List[bool] = []

    for key in sorted_keys:
        mech = mechanisms[key]
        p_merged = merge_probabilities(mech["probs"])
        error_keys.append(key)
        error_det_indices.append(mech["det_indices"])
        error_probs_list.append(p_merged)
        error_obs_mask.append(mech["obs_mask"])

    num_errors = len(error_keys)

    # Build edge indices
    d2e_src, d2e_dst = [], []  # detector → error
    for e_idx, det_list in enumerate(error_det_indices):
        for d_idx in det_list:
            d2e_src.append(d_idx)
            d2e_dst.append(e_idx)

    edge_index_d2e = np.array([d2e_src, d2e_dst], dtype=np.int64) if d2e_src else np.empty((2, 0), dtype=np.int64)
    edge_index_e2d = np.array([d2e_dst, d2e_src], dtype=np.int64) if d2e_src else np.empty((2, 0), dtype=np.int64)

    # Compute weights and probs
    error_probs = np.array(error_probs_list, dtype=np.float32)
    error_weights = np.array(
        [matching_weight(p) for p in error_probs_list],
        dtype=np.float32,
    )
    obs_mask = np.array(error_obs_mask, dtype=bool)

    # Build key
    build_key = DemBuildKey(
        circuit_family="surface_code",
        distance=distance,
        rounds=rounds,
        basis=basis,
        noise_model=noise_model,
        p=p,
        physics_hash=physics_hash,
    )

    # Compute topology hash (deterministic)
    hash_payload = json.dumps({
        "num_detectors": total_det_nodes,
        "num_errors": num_errors,
        "error_keys": error_keys,  # already sorted
        "observable_mask": obs_mask.tolist(),
    }, sort_keys=True, separators=(",", ":"))
    dem_topology_hash = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()[:16]

    # Stats
    k_counts = {}
    for det_list in error_det_indices:
        k = len(det_list)
        k_counts[k] = k_counts.get(k, 0) + 1

    total_prob_mass = float(error_probs.sum()) if num_errors > 0 else 0.0
    k_gt2_mask = np.array([len(d) > 2 for d in error_det_indices], dtype=bool)
    k_gt2_mass = float(error_probs[k_gt2_mask].sum()) if k_gt2_mask.any() else 0.0
    k_gt2_mass_ratio = k_gt2_mass / total_prob_mass if total_prob_mass > 0 else 0.0

    # Connected components (union-find on detector nodes)
    parent = list(range(total_det_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for det_list in error_det_indices:
        for i in range(1, len(det_list)):
            union(det_list[0], det_list[i])

    components = len(set(find(i) for i in range(total_det_nodes)))

    stats = {
        "raw_terms": parse_info["num_terms"],
        "merged_errors": num_errors,
        "total_edges": edge_index_d2e.shape[1],
        "connected_components": components,
        "k_distribution": k_counts,
        "k_gt2_count": int(k_gt2_mask.sum()),
        "k_gt2_mass_ratio": round(k_gt2_mass_ratio, 6),
        "total_prob_mass": round(total_prob_mass, 6),
        "observable_error_count": int(obs_mask.sum()),
        "parse_info": parse_info,
    }

    return BipartiteGraphSpec(
        num_detectors=total_det_nodes,
        num_errors=num_errors,
        has_boundary_node=True,
        edge_index_d2e=edge_index_d2e,
        edge_index_e2d=edge_index_e2d,
        error_weights=error_weights,
        error_probs=error_probs,
        observable_mask=obs_mask,
        error_keys=error_keys,
        dem_topology_hash=dem_topology_hash,
        build_key=build_key,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Tensor Conversion
# ---------------------------------------------------------------------------

def bipartite_graph_to_tensors(spec: BipartiteGraphSpec):
    """Convert BipartiteGraphSpec arrays to PyTorch tensors.

    Returns:
        edge_index_d2e: (2, E) long tensor
        edge_index_e2d: (2, E) long tensor
        error_weights:  (num_errors,) float32 tensor
        observable_mask: (num_errors,) bool tensor
    """
    import torch
    return (
        torch.from_numpy(spec.edge_index_d2e).long(),
        torch.from_numpy(spec.edge_index_e2d).long(),
        torch.from_numpy(spec.error_weights).float(),
        torch.from_numpy(spec.observable_mask).bool(),
    )


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_bipartite_graph(spec: BipartiteGraphSpec, output_dir: Path) -> Path:
    """Save bipartite graph to disk as .npz + meta.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fname = f"bipartite_{spec.dem_topology_hash}"

    # Save arrays
    npz_path = output_dir / f"{fname}.npz"
    np.savez_compressed(
        npz_path,
        edge_index_d2e=spec.edge_index_d2e,
        edge_index_e2d=spec.edge_index_e2d,
        error_weights=spec.error_weights,
        error_probs=spec.error_probs,
        observable_mask=spec.observable_mask,
    )

    # Save metadata
    meta = {
        "num_detectors": spec.num_detectors,
        "num_errors": spec.num_errors,
        "has_boundary_node": spec.has_boundary_node,
        "error_keys": spec.error_keys,
        "dem_topology_hash": spec.dem_topology_hash,
        "build_key": spec.build_key.to_dict(),
        "stats": spec.stats,
    }
    meta_path = output_dir / f"{fname}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    return npz_path


def load_bipartite_graph(npz_path: Path) -> BipartiteGraphSpec:
    """Load bipartite graph from .npz + meta.json."""
    npz_path = Path(npz_path)
    meta_path = npz_path.with_name(npz_path.stem + "_meta.json")

    data = np.load(npz_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    bk = DemBuildKey(**meta["build_key"])

    return BipartiteGraphSpec(
        num_detectors=meta["num_detectors"],
        num_errors=meta["num_errors"],
        has_boundary_node=meta["has_boundary_node"],
        edge_index_d2e=data["edge_index_d2e"],
        edge_index_e2d=data["edge_index_e2d"],
        error_weights=data["error_weights"],
        error_probs=data["error_probs"],
        observable_mask=data["observable_mask"].astype(bool),
        error_keys=meta["error_keys"],
        dem_topology_hash=meta["dem_topology_hash"],
        build_key=bk,
        stats=meta.get("stats", {}),
    )


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def bipartite_graph_stats(spec: BipartiteGraphSpec) -> Dict[str, Any]:
    """Compute comprehensive stats for a bipartite graph.

    Returns dict with:
        num_detectors, num_errors, total_edges, k_distribution,
        k_gt2_count, k_gt2_mass_ratio, connected_components,
        observable_error_count, weight_stats
    """
    stats = dict(spec.stats)

    # Weight statistics
    if spec.num_errors > 0:
        w = spec.error_weights
        stats["weight_mean"] = round(float(w.mean()), 4)
        stats["weight_std"] = round(float(w.std()), 4)
        stats["weight_min"] = round(float(w.min()), 4)
        stats["weight_max"] = round(float(w.max()), 4)
        stats["weight_all_positive"] = bool(np.all(w > 0))
    else:
        stats["weight_mean"] = 0.0
        stats["weight_std"] = 0.0
        stats["weight_min"] = 0.0
        stats["weight_max"] = 0.0
        stats["weight_all_positive"] = True

    return stats


# ---------------------------------------------------------------------------
# Cache / convenience
# ---------------------------------------------------------------------------

_BIPARTITE_CACHE: Dict[str, BipartiteGraphSpec] = {}


def get_or_build_bipartite_graph(
    distance: int,
    rounds: int,
    p: float,
    basis: str,
    noise_model: str = "baseline_symmetric",
    **kwargs,
) -> BipartiteGraphSpec:
    """Build or return cached bipartite graph for given params."""
    cache_key = f"{distance}_{rounds}_{p}_{basis}_{noise_model}"
    if cache_key not in _BIPARTITE_CACHE:
        _BIPARTITE_CACHE[cache_key] = build_bipartite_graph(
            distance=distance, rounds=rounds, p=p,
            basis=basis, noise_model=noise_model,
        )
    return _BIPARTITE_CACHE[cache_key]
