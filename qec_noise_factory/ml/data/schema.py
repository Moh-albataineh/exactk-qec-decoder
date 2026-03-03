"""
ML Data Schema — Day 15

Typed representation of shard metadata (.meta.jsonl) for the ML pipeline.
Provides parsing, validation, and parameter extraction utilities.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ShardMeta:
    """Typed view of one block entry from a .meta.jsonl file."""
    schema_version: int
    pack_name: str
    sample_key: str
    attempt_id: str
    seed: int
    shots: int
    num_detectors: int
    num_observables: int
    det_bytes_per_shot: int
    obs_bytes_per_shot: int
    record_start: int
    record_count: int
    params_canonical: str

    # Derived (populated by parse)
    p: float = 0.0
    physics_model_name: str = ""
    physics_hash: str = ""

    @property
    def record_size(self) -> int:
        """Bytes per record in the .bin file."""
        return self.det_bytes_per_shot + self.obs_bytes_per_shot


# Required fields in every meta entry
_REQUIRED = {
    "schema_version", "pack_name", "sample_key", "attempt_id",
    "seed", "shots", "num_detectors", "num_observables",
    "det_bytes_per_shot", "obs_bytes_per_shot",
    "record_start", "record_count", "params_canonical",
}


def extract_p(params_canonical: str) -> float:
    """Extract physical error rate `p` from params_canonical JSON string."""
    try:
        obj = json.loads(params_canonical)
        circuit = obj.get("circuit", {})
        return float(circuit.get("p", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def extract_physics(params_canonical: str) -> tuple:
    """Extract (physics_model_name, physics_hash) from params_canonical."""
    try:
        obj = json.loads(params_canonical)
        circuit = obj.get("circuit", {})
        return (
            circuit.get("noise_model", ""),
            circuit.get("physics_hash", ""),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return ("", "")


def parse_meta_line(line: str) -> ShardMeta:
    """Parse a single JSON line from a .meta.jsonl file into ShardMeta."""
    raw = json.loads(line.strip())

    missing = _REQUIRED - set(raw.keys())
    if missing:
        raise ValueError(f"Meta entry missing required fields: {missing}")

    p = extract_p(raw["params_canonical"])
    model_name, phash = extract_physics(raw["params_canonical"])

    return ShardMeta(
        schema_version=int(raw["schema_version"]),
        pack_name=str(raw["pack_name"]),
        sample_key=str(raw["sample_key"]),
        attempt_id=str(raw["attempt_id"]),
        seed=int(raw["seed"]),
        shots=int(raw["shots"]),
        num_detectors=int(raw["num_detectors"]),
        num_observables=int(raw["num_observables"]),
        det_bytes_per_shot=int(raw["det_bytes_per_shot"]),
        obs_bytes_per_shot=int(raw["obs_bytes_per_shot"]),
        record_start=int(raw["record_start"]),
        record_count=int(raw["record_count"]),
        params_canonical=str(raw["params_canonical"]),
        p=p,
        physics_model_name=model_name,
        physics_hash=phash,
    )


def load_meta_file(meta_path: str) -> List[ShardMeta]:
    """Load all block entries from a .meta.jsonl file."""
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metas.append(parse_meta_line(line))
    return metas
