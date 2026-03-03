"""
DEM Graph Builder — Day 27

Builds physics-informed graphs from Stim Detector Error Models (DEM).
Extracts edge weights from DEM error mechanisms for GNN v2 training.

Key features:
- Edge weights: W = ln((1-p)/p), matching-compatible
- Edge merging: p_total = 1 - Π(1-p_i) for duplicate (u,v) pairs
- Boundary handling: 1-detector terms → edges to virtual boundary node
- Hyperedge approximation: k>2 detector terms → clique expansion
- Deterministic hashing: canonical ordering + SHA-256
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qec_noise_factory.ml.stim.rebuild import (
    rebuild_stim_circuit,
    params_from_canonical,
)


# ---------------------------------------------------------------------------
# Weight formula
# ---------------------------------------------------------------------------

_P_CLAMP_LO = 1e-12
_P_CLAMP_HI = 1.0 - 1e-12


def matching_weight(p: float) -> float:
    """
    Compute matching-compatible weight: W = ln((1-p)/p).

    Clamps p to [1e-12, 1-1e-12] to prevent inf.
    """
    p_c = max(_P_CLAMP_LO, min(_P_CLAMP_HI, p))
    return math.log((1.0 - p_c) / p_c)


def merge_probabilities(probs: List[float]) -> float:
    """
    Merge independent error probabilities: p_total = 1 - Π(1 - p_i).
    """
    product = 1.0
    for p in probs:
        product *= (1.0 - p)
    return 1.0 - product


# ---------------------------------------------------------------------------
# DEM Graph Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DemBuildKey:
    """
    Hashable key for a DEM graph build — includes physics info.
    """
    circuit_family: str
    distance: int
    rounds: int
    basis: str
    noise_model: str
    p: float
    physics_hash: str = ""
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "circuit_family": self.circuit_family,
            "distance": self.distance,
            "rounds": self.rounds,
            "basis": self.basis,
            "noise_model": self.noise_model,
            "p": self.p,
            "physics_hash": self.physics_hash,
            "schema_version": self.schema_version,
        }


@dataclass
class DemGraphSpec:
    """
    Physics-informed graph from DEM extraction.

    Does NOT replace or modify Day 17 GraphSpec — this is additive.
    """
    # Core graph data
    node_count: int                          # num_detectors + 1 (boundary)
    num_detectors: int                       # original detector count
    has_boundary_node: bool                  # True if boundary node added
    edge_index: np.ndarray                   # (2, E) int64 — canonical u<v sorted
    edge_weight: np.ndarray                  # (E,) float32 — matching weights
    edge_prob: np.ndarray                    # (E,) float32 — merged probabilities

    # Build provenance
    build_key: DemBuildKey
    dem_graph_hash: str = ""

    # DEM statistics
    dem_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": self.node_count,
            "num_detectors": self.num_detectors,
            "has_boundary_node": self.has_boundary_node,
            "edge_count": self.edge_index.shape[1],
            "dem_graph_hash": self.dem_graph_hash,
            "build_key": self.build_key.to_dict(),
            "dem_info": self.dem_info,
            "weight_formula": "ln((1-p)/p)",
            "merge_formula": "p_total = 1 - prod(1-p_i)",
            "hyperedge_strategy": "clique_expansion",
        }


# ---------------------------------------------------------------------------
# DEM Extraction
# ---------------------------------------------------------------------------

def _extract_dem_edges(
    dem,
    num_detectors: int,
) -> Tuple[Dict[Tuple[int, int], List[float]], Dict[str, int]]:
    """
    Parse DEM instructions into edge→prob mapping.

    Returns:
        edges: dict mapping (u, v) -> list of probabilities (before merge)
        info: dict with counts (num_terms, num_boundary, num_hyperedges_gt2, num_observables_only)
    """
    boundary_node = num_detectors  # virtual boundary node index

    edges: Dict[Tuple[int, int], List[float]] = {}
    info = {
        "num_terms": 0,
        "num_boundary_edges": 0,
        "num_hyperedges_gt2": 0,
        "num_observables_only": 0,
        "hyperedge_sizes": [],  # Day 30: track k for each DEM term
        "term_probs": [],       # Day 31.5: probability per DEM term
        "term_k": [],           # Day 31.5: detector count per DEM term
    }

    for instruction in dem.flattened():
        if instruction.type != "error":
            continue

        info["num_terms"] += 1
        prob = instruction.args_copy()[0]
        info["term_probs"].append(float(prob))

        # Extract detector indices from targets
        det_ids = []
        has_observable = False
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                det_ids.append(target.val)
            elif target.is_logical_observable_id():
                has_observable = True

        n_det = len(det_ids)
        info["hyperedge_sizes"].append(n_det)
        info["term_k"].append(n_det)

        if n_det == 0:
            # Observable-only term — no graph edge
            info["num_observables_only"] += 1
            continue

        elif n_det == 1:
            # Boundary edge: detector → boundary node
            u = det_ids[0]
            e = (min(u, boundary_node), max(u, boundary_node))
            if e not in edges:
                edges[e] = []
            edges[e].append(prob)
            info["num_boundary_edges"] += 1

        elif n_det == 2:
            # Standard 2-detector edge
            u, v = det_ids[0], det_ids[1]
            e = (min(u, v), max(u, v))
            if e not in edges:
                edges[e] = []
            edges[e].append(prob)

        else:
            # k>2: clique expansion (pairs within the set)
            info["num_hyperedges_gt2"] += 1
            for i in range(n_det):
                for j in range(i + 1, n_det):
                    u, v = det_ids[i], det_ids[j]
                    e = (min(u, v), max(u, v))
                    if e not in edges:
                        edges[e] = []
                    edges[e].append(prob)

    return edges, info


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_dem_graph(
    distance: int,
    rounds: int,
    p: float,
    basis: str,
    noise_model: str = "baseline_symmetric",
    physics_hash: str = "",
) -> DemGraphSpec:
    """
    Build a DEM-based graph from circuit parameters.

    Steps:
    1. Rebuild Stim circuit
    2. Extract DEM (decompose_errors=True)
    3. Parse DEM → edges with probabilities
    4. Merge duplicate edges
    5. Compute matching weights
    6. Canonical ordering + hash
    """
    circuit = rebuild_stim_circuit(distance, rounds, p, basis, noise_model)
    dem = circuit.detector_error_model(decompose_errors=True)
    num_detectors = circuit.num_detectors

    # Extract raw edges
    raw_edges, dem_info = _extract_dem_edges(dem, num_detectors)

    if not raw_edges:
        # Empty graph (shouldn't happen with valid circuits)
        boundary_node = num_detectors
        return DemGraphSpec(
            node_count=num_detectors + 1,
            num_detectors=num_detectors,
            has_boundary_node=True,
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_weight=np.zeros(0, dtype=np.float32),
            edge_prob=np.zeros(0, dtype=np.float32),
            build_key=DemBuildKey(
                "surface_code_rotated_memory", distance, rounds,
                basis.upper(), noise_model, p, physics_hash,
            ),
            dem_info=dem_info,
        )

    # Merge duplicate edges + compute weights
    sorted_edges = sorted(raw_edges.keys())
    edge_u = []
    edge_v = []
    weights = []
    probs = []
    num_merged = 0

    for (u, v) in sorted_edges:
        p_list = raw_edges[(u, v)]
        if len(p_list) > 1:
            num_merged += 1
        p_total = merge_probabilities(p_list)
        w = matching_weight(p_total)
        edge_u.append(u)
        edge_v.append(v)
        weights.append(w)
        probs.append(p_total)

    edge_index = np.array([edge_u, edge_v], dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32)
    edge_prob = np.array(probs, dtype=np.float32)

    dem_info["num_edges_merged"] = num_merged
    dem_info["num_edges_final"] = len(sorted_edges)

    # Boundary node present if any boundary edge exists
    has_boundary = any(num_detectors in e for e in sorted_edges)

    build_key = DemBuildKey(
        circuit_family="surface_code_rotated_memory",
        distance=distance,
        rounds=rounds,
        basis=basis.upper(),
        noise_model=noise_model,
        p=p,
        physics_hash=physics_hash,
    )

    spec = DemGraphSpec(
        node_count=num_detectors + (1 if has_boundary else 0),
        num_detectors=num_detectors,
        has_boundary_node=has_boundary,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_prob=edge_prob,
        build_key=build_key,
        dem_info=dem_info,
    )

    # Compute deterministic hash
    spec.dem_graph_hash = compute_dem_graph_hash(spec)

    return spec


def build_dem_graph_from_meta(params_canonical: str) -> DemGraphSpec:
    """Build DEM graph from shard metadata params_canonical."""
    params = params_from_canonical(params_canonical)
    if params["distance"] == 0:
        raise ValueError(f"Cannot extract circuit params: {params_canonical}")

    # Extract physics hash if available
    try:
        obj = json.loads(params_canonical)
        c = obj.get("circuit", {})
        physics_hash = c.get("physics_hash", "")
    except (json.JSONDecodeError, TypeError):
        physics_hash = ""

    return build_dem_graph(
        **params,
        physics_hash=physics_hash,
    )


# ---------------------------------------------------------------------------
# Deterministic Hashing
# ---------------------------------------------------------------------------

def compute_dem_graph_hash(spec: DemGraphSpec) -> str:
    """
    Compute deterministic SHA-256 hash of a DEM graph.

    Hash = sha256( canonical_json(build_key) + edge_index.tobytes() + edge_weight.tobytes() )

    Edge index and weights are already canonically sorted (u<v, lexicographic).
    Weights use float32 for stable byte representation.
    """
    h = hashlib.sha256()

    # Key part: canonical JSON of build key
    key_json = json.dumps(spec.build_key.to_dict(), sort_keys=True, separators=(",", ":"))
    h.update(key_json.encode("utf-8"))

    # Data part: edge_index + edge_weight as raw bytes
    h.update(spec.edge_index.tobytes())
    h.update(spec.edge_weight.tobytes())

    return h.hexdigest()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def save_dem_graph(spec: DemGraphSpec, output_dir: Path) -> Path:
    """
    Save DEM graph to disk as .npz + meta.json.

    Returns path to the saved .npz file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"dem_d{spec.build_key.distance}_{spec.build_key.basis}_{spec.build_key.noise_model}_p{spec.build_key.p}"

    # Save graph data
    npz_path = output_dir / f"{prefix}.npz"
    np.savez(
        npz_path,
        edge_index=spec.edge_index,
        edge_weight=spec.edge_weight,
        edge_prob=spec.edge_prob,
    )

    # Save metadata (use same prefix, not .with_suffix which breaks on dots in p-value)
    meta_path = output_dir / f"{prefix}.meta.json"
    meta = spec.to_dict()
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return npz_path


def load_dem_graph(npz_path: Path) -> DemGraphSpec:
    """Load DEM graph from .npz + meta.json."""
    npz_path = Path(npz_path)
    # Can't use .with_suffix() — filenames like p0.01.npz have dots in p-value
    meta_path = npz_path.parent / (str(npz_path.name).replace(".npz", ".meta.json"))

    # Use context manager to ensure file handle is closed (Windows compat)
    with np.load(npz_path) as data:
        edge_index = data["edge_index"].copy()
        edge_weight = data["edge_weight"].copy()
        edge_prob = data["edge_prob"].copy()

    with open(meta_path, "r") as f:
        meta = json.load(f)

    bk = meta["build_key"]
    build_key = DemBuildKey(
        circuit_family=bk["circuit_family"],
        distance=bk["distance"],
        rounds=bk["rounds"],
        basis=bk["basis"],
        noise_model=bk["noise_model"],
        p=bk["p"],
        physics_hash=bk.get("physics_hash", ""),
        schema_version=bk.get("schema_version", 1),
    )

    return DemGraphSpec(
        node_count=meta["node_count"],
        num_detectors=meta["num_detectors"],
        has_boundary_node=meta["has_boundary_node"],
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_prob=edge_prob,
        build_key=build_key,
        dem_graph_hash=meta["dem_graph_hash"],
        dem_info=meta.get("dem_info", {}),
    )


# ---------------------------------------------------------------------------
# Day 29.1: DEM Graph Stats — sanity-check helper
# ---------------------------------------------------------------------------


def _largest_component_size(n: int, edge_index: np.ndarray, E: int) -> int:
    """Return the size (node count) of the largest connected component."""
    if n == 0:
        return 0
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(E):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u < n and v < n:
            union(u, v)

    from collections import Counter
    roots = [find(i) for i in range(n)]
    counts = Counter(roots)
    return max(counts.values()) if counts else 0


def dem_graph_stats(spec: "DemGraphSpec") -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a DEM graph.

    Used by strict smoke tests to verify the DEM graph is non-trivial
    and scientifically plausible.

    Returns dict with:
        dem_terms_raw, merged_edges, boundary_edges, hyperedges_k_gt_2,
        nodes, edges_undirected, edges_bidir,
        weight_min, weight_max, weight_mean,
        connected_components,
        # Day 30 extensions:
        hyperedge_size_histogram, hyperedge_ratio, clique_expansion_factor,
        largest_component_size, weight_positive
    """
    E = spec.edge_index.shape[1] if spec.edge_index.ndim == 2 else 0

    # Weight stats
    if E > 0:
        w = spec.edge_weight
        w_min = float(np.min(w))
        w_max = float(np.max(w))
        w_mean = float(np.mean(w))
        weight_positive = bool(w_min > 0)
    else:
        w_min = w_max = w_mean = float("nan")
        weight_positive = False

    # Connected components via union-find
    n_components = _count_components(spec.node_count, spec.edge_index, E)

    # Largest component size
    largest_comp = _largest_component_size(spec.node_count, spec.edge_index, E)

    info = spec.dem_info or {}
    dem_terms = info.get("num_terms", 0)
    hyper_k_gt2 = info.get("num_hyperedges_gt2", 0)

    # Day 30: hyperedge size histogram
    hyperedge_sizes = info.get("hyperedge_sizes", [])
    size_hist: Dict[int, int] = {}
    for k in hyperedge_sizes:
        size_hist[k] = size_hist.get(k, 0) + 1

    # Hyperedge ratio: fraction of DEM terms that are k>2
    hyperedge_ratio = hyper_k_gt2 / max(1, dem_terms)

    # Clique expansion factor: how many graph edges per DEM term
    clique_expansion_factor = (2 * E) / max(1, dem_terms)

    return {
        "dem_terms_raw": dem_terms,
        "merged_edges": info.get("num_edges_merged", 0),
        "boundary_edges": info.get("num_boundary_edges", 0),
        "hyperedges_k_gt_2": hyper_k_gt2,
        "observables_only": info.get("num_observables_only", 0),
        "nodes": spec.node_count,
        "num_detectors": spec.num_detectors,
        "has_boundary_node": spec.has_boundary_node,
        "edges_undirected": E,
        "edges_bidir": 2 * E,
        "weight_min": w_min,
        "weight_max": w_max,
        "weight_mean": w_mean,
        "weight_positive": weight_positive,
        "connected_components": n_components,
        "largest_component_size": largest_comp,
        "hyperedge_size_histogram": size_hist,
        "hyperedge_ratio": round(hyperedge_ratio, 4),
        "clique_expansion_factor": round(clique_expansion_factor, 4),
    }


def run_dem_diagnostics(
    spec: "DemGraphSpec",
    noise_model: Optional[str] = None,
    debug_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run hard-fail diagnostics on a DEM graph.

    Checks:
      - connected_components == 1 (single connected graph)
      - edges_undirected >= nodes - 1 (at least spanning tree)
      - weight_min > 0 and finite (no NaN/Inf/zero)
      - dem_terms_raw > 0 (non-trivial DEM)
      - hyperedges_k_gt_2 > 0 when noise_model is correlated

    Returns dict with {pass: bool, checks: [...], debug_path: str | None}.
    On failure, writes dem_graph_debug.json if debug_dir given.
    """
    stats = dem_graph_stats(spec)
    result: Dict[str, Any] = {"pass": True, "checks": [], "debug_path": None}

    def check(name: str, ok: bool, msg: str = ""):
        result["checks"].append({"name": name, "ok": ok, "msg": msg})
        if not ok:
            result["pass"] = False

    n = stats["nodes"]
    E = stats["edges_undirected"]
    w_min = stats["weight_min"]

    check("components_eq_1",
          stats["connected_components"] == 1,
          f"components={stats['connected_components']}")

    check("edges_ge_nodes_minus_1",
          E >= n - 1,
          f"edges={E}, nodes={n}")

    check("weight_min_positive",
          isinstance(w_min, (int, float)) and math.isfinite(w_min) and w_min > 0,
          f"weight_min={w_min}")

    check("dem_terms_nonzero",
          stats["dem_terms_raw"] > 0,
          f"dem_terms_raw={stats['dem_terms_raw']}")

    # Correlated model must produce hyperedges
    if noise_model and "correlated" in noise_model:
        check("hyperedges_present",
              stats["hyperedges_k_gt_2"] > 0,
              f"hyperedges_k_gt_2={stats['hyperedges_k_gt_2']} "
              f"(expected >0 for {noise_model})")

    # Write debug artifact on failure
    if not result["pass"] and debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / "dem_graph_debug.json"
        with open(debug_path, "w") as f:
            import json as _json
            _json.dump({
                "diagnostics": result,
                "stats": {k: v for k, v in stats.items()
                          if not isinstance(v, (np.ndarray,))},
                "build_key": spec.build_key.to_dict(),
            }, f, indent=2, default=str)
        result["debug_path"] = str(debug_path)

    return result


def _count_components(n: int, edge_index: np.ndarray, E: int) -> int:
    """Count connected components using union-find on edge_index (2, E)."""
    if n == 0:
        return 0
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(E):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u < n and v < n:
            union(u, v)

    return len({find(i) for i in range(n)})


# ---------------------------------------------------------------------------
# Day 28: Torch conversion + cache
# ---------------------------------------------------------------------------

import torch as _torch

_DEM_GRAPH_CACHE: Dict[str, "DemGraphSpec"] = {}


def dem_graph_to_edge_index(
    spec: "DemGraphSpec",
) -> tuple:
    """
    Convert DEM graph to bidirectional PyTorch tensors.

    Returns:
        edge_index: (2, 2E) long tensor (bidirectional)
        edge_weight: (2E,) float32 tensor (same weight both directions)
    """
    ei = spec.edge_index  # (2, E) with u < v
    # Make bidirectional: add reverse edges
    fwd = ei  # (2, E)
    rev = np.stack([ei[1], ei[0]], axis=0)  # (2, E)
    bi_ei = np.concatenate([fwd, rev], axis=1)  # (2, 2E)

    # Duplicate weights for reverse edges
    bi_w = np.concatenate([spec.edge_weight, spec.edge_weight])  # (2E,)

    return (
        _torch.from_numpy(bi_ei.astype(np.int64)),
        _torch.from_numpy(bi_w.astype(np.float32)),
    )


def get_or_build_dem_graph(
    distance: int,
    rounds: int,
    p: float,
    basis: str,
    noise_model: str = "baseline_symmetric",
    physics_hash: str = "",
) -> "DemGraphSpec":
    """Build or retrieve cached DEM graph."""
    cache_key = f"{distance}_{rounds}_{p}_{basis}_{noise_model}"
    if cache_key in _DEM_GRAPH_CACHE:
        return _DEM_GRAPH_CACHE[cache_key]

    spec = build_dem_graph(distance, rounds, p, basis, noise_model, physics_hash)
    _DEM_GRAPH_CACHE[cache_key] = spec
    return spec


# ---------------------------------------------------------------------------
# Day 31.5: Correlation-mass statistics from DEM
# ---------------------------------------------------------------------------

def dem_corr_stats(
    distance: int,
    rounds: int,
    p: float,
    basis: str = "X",
    noise_model: str = "correlated_crosstalk_like",
    corr_strength: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute DEM correlation statistics for a given circuit configuration.

    Returns dict with:
        dem_terms_total, hyperedges_k_gt_2_count,
        total_prob_mass, k_gt_2_prob_mass, k_gt_2_mass_ratio,
        connected_components, topk_mass_ratio
    """
    from qec_noise_factory.factory.circuits_qc import (
        build_surface_code_memory_circuit_level,
    )

    overrides = {}
    if noise_model == "correlated_crosstalk_like":
        overrides["corr_strength"] = corr_strength

    circuit, _ = build_surface_code_memory_circuit_level(
        distance=distance,
        rounds=rounds,
        p_base=p,
        basis=basis,
        noise_model=noise_model,
        noise_params_overrides=overrides if overrides else None,
    )

    dem = circuit.detector_error_model(decompose_errors=True)
    num_det = circuit.num_detectors
    _, info = _extract_dem_edges(dem, num_det)

    term_probs = np.array(info.get("term_probs", []), dtype=np.float64)
    term_k = np.array(info.get("term_k", []), dtype=np.int32)

    total_prob_mass = float(np.sum(term_probs)) if len(term_probs) > 0 else 0.0
    k_gt_2_mask = term_k > 2
    k_gt_2_prob_mass = float(np.sum(term_probs[k_gt_2_mask])) if np.any(k_gt_2_mask) else 0.0
    k_gt_2_mass_ratio = k_gt_2_prob_mass / total_prob_mass if total_prob_mass > 0 else 0.0
    hyperedges_k_gt_2_count = int(np.sum(k_gt_2_mask))

    # Top-20 mass ratio
    topk = 20
    if len(term_probs) >= topk:
        sorted_probs = np.sort(term_probs)[::-1]
        topk_mass = float(np.sum(sorted_probs[:topk]))
        topk_mass_ratio = topk_mass / total_prob_mass if total_prob_mass > 0 else 0.0
    else:
        topk_mass_ratio = 1.0  # all terms are in top-k

    # Build graph for connected components
    spec = build_dem_graph(
        distance=distance, rounds=rounds, p=p,
        basis=basis, noise_model=noise_model,
    )
    E = spec.edge_index.shape[1] if spec.edge_index.ndim == 2 else 0
    n_components = _count_components(spec.node_count, spec.edge_index, E)

    return {
        "distance": distance,
        "rounds": rounds,
        "p": p,
        "basis": basis,
        "noise_model": noise_model,
        "dem_terms_total": info["num_terms"],
        "hyperedges_k_gt_2_count": hyperedges_k_gt_2_count,
        "total_prob_mass": round(total_prob_mass, 6),
        "k_gt_2_prob_mass": round(k_gt_2_prob_mass, 6),
        "k_gt_2_mass_ratio": round(k_gt_2_mass_ratio, 6),
        "connected_components": n_components,
        "topk_mass_ratio": round(topk_mass_ratio, 6),
        "num_detectors": num_det,
        "graph_nodes": spec.node_count,
        "graph_edges": E,
    }

