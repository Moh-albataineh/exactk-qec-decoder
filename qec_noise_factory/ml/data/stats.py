"""
ML Dataset Statistics — Day 15

Computes and saves summary statistics for ML training data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from qec_noise_factory.ml.data.schema import ShardMeta, load_meta_file
from qec_noise_factory.ml.data.reader import ShardDataset, read_shards_dir


@dataclass
class DatasetStats:
    """Summary statistics for an ML dataset."""
    num_shards: int
    num_blocks: int
    total_samples: int
    num_detectors: int
    num_observables: int
    y_rate: float                     # fraction of Y=1
    p_min: float
    p_max: float
    p_mean: float
    p_distribution: Dict[str, int]   # p-bucket → count
    pack_names: List[str]
    det_shape: List[int]             # [N, num_detectors]
    obs_shape: List[int]             # [N, num_observables]

    # Day 15 review additions
    detector_distribution: Dict[int, int]    # num_detectors → block count
    observable_distribution: Dict[int, int]  # num_observables → block count
    circuit_params: Dict[str, Any]           # rounds/distance/type distribution
    integrity_checks: List[str]              # cross-check results

    # Day 20 additions
    detectors_min: int = 0
    detectors_median: int = 0
    detectors_max: int = 0
    distances_distribution: Dict[int, int] = None     # distance → block count
    rounds_distribution: Dict[int, int] = None        # rounds → block count
    bases_distribution: Dict[str, int] = None          # basis → block count
    y_rate_by_group: Dict[str, float] = None           # "model|basis|d" → y_rate

    def __post_init__(self):
        if self.distances_distribution is None:
            self.distances_distribution = {}
        if self.rounds_distribution is None:
            self.rounds_distribution = {}
        if self.bases_distribution is None:
            self.bases_distribution = {}
        if self.y_rate_by_group is None:
            self.y_rate_by_group = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _p_bucket_label(p: float) -> str:
    """Human-readable p-bucket label."""
    if p < 0.01:
        return "p<0.01"
    elif p < 0.05:
        return "0.01≤p<0.05"
    elif p < 0.1:
        return "0.05≤p<0.1"
    elif p < 0.3:
        return "0.1≤p<0.3"
    else:
        return "p≥0.3"


def _extract_circuit_params(params_canonical: str) -> Dict[str, Any]:
    """Extract circuit params for distribution tracking."""
    try:
        obj = json.loads(params_canonical)
        c = obj.get("circuit", {})
        return {
            "type": c.get("type", "unknown"),
            "rounds": c.get("rounds", 0),
            "distance": c.get("distance", 0),
            "basis": c.get("basis", "").upper(),
            "noise_model": c.get("noise_model", ""),
        }
    except (json.JSONDecodeError, TypeError):
        return {"type": "unknown", "rounds": 0, "distance": 0, "basis": "", "noise_model": ""}


def compute_stats(
    shards_dir: str | Path,
) -> DatasetStats:
    """
    Compute dataset statistics from a shards directory.
    Reads all shard pairs and computes summary.
    Includes integrity cross-checks (metadata vs actual shapes).
    """
    shards_dir = Path(shards_dir)
    datasets = read_shards_dir(shards_dir)

    if not datasets:
        raise ValueError(f"No shards found in {shards_dir}")

    total_samples = 0
    total_y_sum = 0
    all_p = []
    p_dist: Dict[str, int] = {}
    pack_names = set()
    num_det = 0
    num_obs = 0

    # Diagnostics
    det_dist: Dict[int, int] = {}    # num_detectors → block count
    obs_dist: Dict[int, int] = {}    # num_observables → block count
    circuit_types: Dict[str, int] = {}
    rounds_set: set = set()
    distance_set: set = set()

    # Day 20: per-block detector counts & distributions
    all_det_counts: List[int] = []
    dist_dist: Dict[int, int] = {}    # distance → block count
    rounds_dist: Dict[int, int] = {}  # rounds → block count
    bases_dist: Dict[str, int] = {}   # basis → block count
    group_y_sum: Dict[str, int] = {}  # group key → sum(Y)
    group_n: Dict[str, int] = {}      # group key → total samples
    integrity: List[str] = []

    for ds in datasets:
        total_samples += ds.X.shape[0]
        total_y_sum += int(ds.Y.sum())
        actual_det = ds.X.shape[1]
        actual_obs = ds.Y.shape[1]
        num_det = actual_det
        num_obs = actual_obs

        for m in ds.meta:
            all_p.append(m.p)
            pack_names.add(m.pack_name)
            bucket = _p_bucket_label(m.p)
            p_dist[bucket] = p_dist.get(bucket, 0) + m.record_count

            # Detector/observable distribution
            det_dist[m.num_detectors] = det_dist.get(m.num_detectors, 0) + 1
            obs_dist[m.num_observables] = obs_dist.get(m.num_observables, 0) + 1

            # Circuit params
            cp = _extract_circuit_params(m.params_canonical)
            ct = cp["type"]
            circuit_types[ct] = circuit_types.get(ct, 0) + 1
            if cp["rounds"]:
                rounds_set.add(cp["rounds"])
                rounds_dist[cp["rounds"]] = rounds_dist.get(cp["rounds"], 0) + 1
            if cp["distance"]:
                distance_set.add(cp["distance"])
                dist_dist[cp["distance"]] = dist_dist.get(cp["distance"], 0) + 1
            if cp["basis"]:
                bases_dist[cp["basis"]] = bases_dist.get(cp["basis"], 0) + 1

            # Day 20: per-block detector count
            all_det_counts.append(m.num_detectors)

            # Day 20: y_rate per (model, basis, distance) group
            gk = f"{cp.get('noise_model', m.physics_model_name or m.pack_name)}|{cp['basis']}|d={cp['distance']}"
            block_offset_local = sum(
                prev_m.record_count for prev_m in ds.meta[:ds.meta.index(m)]
            )
            block_y = int(ds.Y[block_offset_local:block_offset_local + m.record_count].sum())
            group_y_sum[gk] = group_y_sum.get(gk, 0) + block_y
            group_n[gk] = group_n.get(gk, 0) + m.record_count * actual_obs

            # Integrity cross-check: metadata num_detectors vs X.shape[1]
            if m.num_detectors != actual_det:
                integrity.append(
                    f"MISMATCH: meta says {m.num_detectors} det, "
                    f"X has {actual_det} (shard {ds.shard_path})"
                )
            if m.num_observables != actual_obs:
                integrity.append(
                    f"MISMATCH: meta says {m.num_observables} obs, "
                    f"Y has {actual_obs} (shard {ds.shard_path})"
                )

    if not integrity:
        integrity.append("ALL OK: metadata matches actual array shapes")

    y_rate = total_y_sum / max(1, total_samples * num_obs)
    p_arr = np.array(all_p) if all_p else np.array([0.0])

    # Day 20: detector stats
    det_arr = np.array(all_det_counts) if all_det_counts else np.array([0])
    y_rate_groups = {
        gk: group_y_sum[gk] / max(1, group_n[gk])
        for gk in sorted(group_y_sum.keys())
    }

    return DatasetStats(
        num_shards=len(datasets),
        num_blocks=sum(len(d.meta) for d in datasets),
        total_samples=total_samples,
        num_detectors=num_det,
        num_observables=num_obs,
        y_rate=y_rate,
        p_min=float(p_arr.min()),
        p_max=float(p_arr.max()),
        p_mean=float(p_arr.mean()),
        p_distribution=dict(sorted(p_dist.items())),
        pack_names=sorted(pack_names),
        det_shape=[total_samples, num_det],
        obs_shape=[total_samples, num_obs],
        detector_distribution={k: v for k, v in sorted(det_dist.items())},
        observable_distribution={k: v for k, v in sorted(obs_dist.items())},
        circuit_params={
            "types": circuit_types,
            "rounds": sorted(rounds_set) if rounds_set else [],
            "distances": sorted(distance_set) if distance_set else [],
        },
        integrity_checks=integrity,
        detectors_min=int(det_arr.min()),
        detectors_median=int(np.median(det_arr)),
        detectors_max=int(det_arr.max()),
        distances_distribution={k: v for k, v in sorted(dist_dist.items())},
        rounds_distribution={k: v for k, v in sorted(rounds_dist.items())},
        bases_distribution=dict(sorted(bases_dist.items())),
        y_rate_by_group=y_rate_groups,
    )


def save_stats(stats: DatasetStats, out_path: str | Path) -> Path:
    """Save dataset statistics to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(stats.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    return out_path


def print_stats(stats: DatasetStats) -> str:
    """Format stats as a readable summary string."""
    lines = [
        f"=== ML Dataset Statistics ===",
        f"Shards:       {stats.num_shards}",
        f"Blocks:       {stats.num_blocks}",
        f"Total Samples: {stats.total_samples:,}",
        f"Detectors:    {stats.num_detectors}",
        f"Observables:  {stats.num_observables}",
        f"Y Rate:       {stats.y_rate:.4%}",
        f"p Range:      [{stats.p_min:.6f}, {stats.p_max:.6f}]",
        f"p Mean:       {stats.p_mean:.6f}",
        f"X Shape:      {stats.det_shape}",
        f"Y Shape:      {stats.obs_shape}",
        f"Detector Distribution:",
    ]
    for nd, count in stats.detector_distribution.items():
        lines.append(f"  num_det={nd}: {count} blocks")
    lines.append("Observable Distribution:")
    for no, count in stats.observable_distribution.items():
        lines.append(f"  num_obs={no}: {count} blocks")
    lines.append(f"Circuit Types: {stats.circuit_params.get('types', {})}")
    lines.append(f"Rounds: {stats.circuit_params.get('rounds', [])}")
    lines.append(f"Integrity: {stats.integrity_checks[0]}")

    # Day 20 additions
    lines.append(f"Detectors:    min={stats.detectors_min}, median={stats.detectors_median}, max={stats.detectors_max}")
    if stats.distances_distribution:
        lines.append("Distances Distribution:")
        for d, count in stats.distances_distribution.items():
            lines.append(f"  d={d}: {count} blocks")
    if stats.rounds_distribution:
        lines.append("Rounds Distribution:")
        for r, count in stats.rounds_distribution.items():
            lines.append(f"  rounds={r}: {count} blocks")
    if stats.bases_distribution:
        lines.append("Bases Distribution:")
        for b, count in stats.bases_distribution.items():
            lines.append(f"  {b}: {count} blocks")
    if stats.y_rate_by_group:
        lines.append("Y Rate by Group:")
        for gk, yr in stats.y_rate_by_group.items():
            lines.append(f"  {gk}: {yr:.4%}")

    lines.append("P Distribution:")
    for bucket, count in stats.p_distribution.items():
        lines.append(f"  {bucket}: {count:,}")
    return "\n".join(lines)
