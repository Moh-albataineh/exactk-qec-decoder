from __future__ import annotations
import hashlib
import json
from typing import Any, Dict

def canonical_json(obj: Dict[str, Any]) -> str:
    # separators removes whitespace for stable bytes; sort_keys for stable ordering
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def make_sample_key(
    pack_name: str,
    schema_version: int,
    params_canonical_json: str,
    seed: int,
    code_version: str,
) -> str:
    # IMPORTANT: sample_key must be deterministic and include everything that changes the output.
    material = canonical_json({
        "pack": pack_name,
        "schema": int(schema_version),
        "params": params_canonical_json,  # already canonical string
        "seed": int(seed),
        "code_version": code_version,
    })
    return sha256_hex(material)
