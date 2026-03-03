"""
ML Shard Reader — Day 15

Reads bitpacked .bin shards into numpy arrays for ML training.

Record layout (per shot):
    [det_bytes_per_shot bytes] [obs_bytes_per_shot bytes]

Output:
    X: np.ndarray[bool] shape (N, num_detectors)
    Y: np.ndarray[bool] shape (N, num_observables)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from qec_noise_factory.ml.data.schema import ShardMeta, load_meta_file


@dataclass
class ShardDataset:
    """One shard's worth of ML-ready data + metadata."""
    X: np.ndarray         # (N, num_detectors) bool
    Y: np.ndarray         # (N, num_observables) bool
    meta: List[ShardMeta] # block metadata entries
    shard_path: str


def _unpack_bits(packed: bytes, n_bits: int) -> np.ndarray:
    """
    Unpack bitpacked bytes into a bool array of length n_bits.
    Stim uses LSB-first packing within each byte.
    """
    arr = np.frombuffer(packed, dtype=np.uint8)
    # Unpack all 8 bits per byte (LSB first)
    bits = np.unpackbits(arr, bitorder="little")
    return bits[:n_bits].astype(bool)


def read_shard_block(
    bin_path: Path,
    meta: ShardMeta,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a single block from a shard .bin file.

    Returns:
        X: (record_count, num_detectors) bool array
        Y: (record_count, num_observables) bool array
    """
    rec_size = meta.record_size
    n = meta.record_count
    offset = meta.record_start * rec_size

    with open(bin_path, "rb") as f:
        f.seek(offset)
        raw = f.read(n * rec_size)

    if len(raw) != n * rec_size:
        raise ValueError(
            f"Expected {n * rec_size} bytes, got {len(raw)} "
            f"from {bin_path} offset {offset}"
        )

    # Parse record by record
    det_list = []
    obs_list = []

    for i in range(n):
        rec_start = i * rec_size
        det_bytes = raw[rec_start : rec_start + meta.det_bytes_per_shot]
        obs_bytes = raw[rec_start + meta.det_bytes_per_shot : rec_start + rec_size]

        det_bits = _unpack_bits(det_bytes, meta.num_detectors)
        obs_bits = _unpack_bits(obs_bytes, meta.num_observables)

        det_list.append(det_bits)
        obs_list.append(obs_bits)

    X = np.stack(det_list, axis=0)  # (N, num_detectors)
    Y = np.stack(obs_list, axis=0)  # (N, num_observables)

    return X, Y


def read_shard(
    bin_path: str | Path,
    meta_path: str | Path,
) -> ShardDataset:
    """
    Read all blocks from a shard pair (.bin + .meta.jsonl).

    Returns a ShardDataset with concatenated X, Y arrays.
    """
    bin_path = Path(bin_path)
    meta_path = Path(meta_path)

    metas = load_meta_file(str(meta_path))
    if not metas:
        raise ValueError(f"No blocks in {meta_path}")

    x_parts = []
    y_parts = []

    for m in metas:
        x, y = read_shard_block(bin_path, m)
        x_parts.append(x)
        y_parts.append(y)

    X = np.concatenate(x_parts, axis=0)
    Y = np.concatenate(y_parts, axis=0)

    return ShardDataset(X=X, Y=Y, meta=metas, shard_path=str(bin_path))


def read_shards_dir(
    shards_dir: str | Path,
) -> List[ShardDataset]:
    """
    Read all shard pairs from a directory.

    Discovers shard_XXXX.bin + shard_XXXX.meta.jsonl pairs.
    """
    shards_dir = Path(shards_dir)
    bin_files = sorted(shards_dir.glob("shard_*.bin"))

    datasets = []
    for bp in bin_files:
        mp = bp.with_suffix("").with_suffix(".meta.jsonl")
        if not mp.exists():
            continue
        ds = read_shard(bp, mp)
        datasets.append(ds)

    return datasets


def merge_datasets(datasets: List[ShardDataset]) -> ShardDataset:
    """Merge multiple ShardDatasets into one."""
    if not datasets:
        raise ValueError("No datasets to merge")

    X = np.concatenate([d.X for d in datasets], axis=0)
    Y = np.concatenate([d.Y for d in datasets], axis=0)
    all_meta = []
    for d in datasets:
        all_meta.extend(d.meta)

    return ShardDataset(
        X=X, Y=Y, meta=all_meta,
        shard_path=f"merged({len(datasets)} shards)",
    )


def filter_by_p_range(
    dataset: ShardDataset,
    p_lo: float = 0.0,
    p_hi: float = 1.0,
) -> ShardDataset:
    """
    Filter a ShardDataset to only include blocks within [p_lo, p_hi].

    Useful for ML training:
      - Exclude saturated regime (p > 0.3) where y ≈ coin-flip
      - Focus on informative near-threshold data
      - Use high-p only for OOD stress testing

    Args:
        dataset: input ShardDataset (potentially merged)
        p_lo: minimum p (inclusive)
        p_hi: maximum p (inclusive)

    Returns:
        Filtered ShardDataset with only matching blocks.
    """
    # We need to re-read block by block since X/Y are concatenated.
    # Use record_count from metadata to slice.
    kept_meta = []
    kept_x = []
    kept_y = []

    offset = 0
    for m in dataset.meta:
        n = m.record_count
        if p_lo <= m.p <= p_hi:
            kept_meta.append(m)
            kept_x.append(dataset.X[offset:offset + n])
            kept_y.append(dataset.Y[offset:offset + n])
        offset += n

    if not kept_meta:
        # Return empty dataset with correct shape
        n_det = dataset.X.shape[1] if dataset.X.shape[0] > 0 else 0
        n_obs = dataset.Y.shape[1] if dataset.Y.shape[0] > 0 else 0
        return ShardDataset(
            X=np.empty((0, n_det), dtype=bool),
            Y=np.empty((0, n_obs), dtype=bool),
            meta=[],
            shard_path=f"filtered({dataset.shard_path})",
        )

    return ShardDataset(
        X=np.concatenate(kept_x, axis=0),
        Y=np.concatenate(kept_y, axis=0),
        meta=kept_meta,
        shard_path=f"filtered({dataset.shard_path})",
    )


# ---------------------------------------------------------------------------
# Flexible filter (Day 20)
# ---------------------------------------------------------------------------

def _extract_circuit_info(params_canonical: str) -> dict:
    """Extract circuit info from params_canonical for filtering."""
    import json
    try:
        obj = json.loads(params_canonical)
        c = obj.get("circuit", {})
        return {
            "type": c.get("type", "unknown"),
            "distance": c.get("distance", 0),
            "basis": c.get("basis", "").upper(),
            "noise_model": c.get("noise_model", ""),
        }
    except (json.JSONDecodeError, TypeError):
        return {"type": "unknown", "distance": 0, "basis": "", "noise_model": ""}


def filter_dataset(
    dataset: ShardDataset,
    *,
    min_detectors: int = 0,
    allowed_circuit_families: Optional[List[str]] = None,
    allowed_distances: Optional[List[int]] = None,
    allowed_bases: Optional[List[str]] = None,
) -> ShardDataset:
    """
    Filter a ShardDataset by metadata criteria.

    Args:
        dataset: input ShardDataset
        min_detectors: exclude blocks with fewer detectors
        allowed_circuit_families: if set, only keep blocks with matching circuit type
        allowed_distances: if set, only keep blocks with matching distance
        allowed_bases: if set, only keep blocks with matching basis (case-insensitive)

    Returns:
        Filtered ShardDataset.
    """
    # Normalize bases to uppercase
    norm_bases = None
    if allowed_bases is not None:
        norm_bases = [b.upper() for b in allowed_bases]

    kept_meta = []
    kept_x = []
    kept_y = []

    offset = 0
    for m in dataset.meta:
        n = m.record_count

        # Check min_detectors
        if m.num_detectors < min_detectors:
            offset += n
            continue

        # Parse circuit info for remaining filters
        info = _extract_circuit_info(m.params_canonical)

        # Check circuit family
        if allowed_circuit_families is not None:
            if info["type"] not in allowed_circuit_families:
                offset += n
                continue

        # Check distance
        if allowed_distances is not None:
            if info["distance"] not in allowed_distances:
                offset += n
                continue

        # Check basis
        if norm_bases is not None:
            if info["basis"] not in norm_bases:
                offset += n
                continue

        kept_meta.append(m)
        kept_x.append(dataset.X[offset:offset + n])
        kept_y.append(dataset.Y[offset:offset + n])
        offset += n

    if not kept_meta:
        n_det = dataset.X.shape[1] if dataset.X.shape[0] > 0 else 0
        n_obs = dataset.Y.shape[1] if dataset.Y.shape[0] > 0 else 0
        return ShardDataset(
            X=np.empty((0, n_det), dtype=bool),
            Y=np.empty((0, n_obs), dtype=bool),
            meta=[],
            shard_path=f"filtered({dataset.shard_path})",
        )

    return ShardDataset(
        X=np.concatenate(kept_x, axis=0),
        Y=np.concatenate(kept_y, axis=0),
        meta=kept_meta,
        shard_path=f"filtered({dataset.shard_path})",
    )
