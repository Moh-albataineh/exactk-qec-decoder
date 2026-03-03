"""
Shared Stim Circuit Rebuild Utility — Day 27

Single source of truth for rebuilding Stim circuits from shard metadata.
Used by mwpm_decoder (Day 26) and DEM graph builder (Day 27).
"""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import stim


def rebuild_stim_circuit(
    distance: int,
    rounds: int,
    p: float,
    basis: str,
    noise_model: str = "baseline_symmetric",
) -> stim.Circuit:
    """
    Rebuild a Stim circuit from shard metadata parameters.

    Matches the circuit builders in circuits_qc.py to ensure
    the DEM is consistent with the training data.
    """
    if noise_model in ("baseline_symmetric", ""):
        return stim.Circuit.generated(
            f"surface_code:rotated_memory_{basis.lower()}",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p,
        )
    elif noise_model in ("sd6_like", "si1000_like", "correlated_crosstalk_like"):
        from qec_noise_factory.factory.circuits_qc import (
            build_surface_code_memory_circuit_level,
        )
        overrides = {}
        if noise_model == "correlated_crosstalk_like":
            overrides["corr_strength"] = 0.5  # default; can be overridden by caller
        circuit, _ = build_surface_code_memory_circuit_level(
            distance=distance, rounds=rounds, p_base=p,
            basis=basis, noise_model=noise_model,
            noise_params_overrides=overrides if overrides else None,
        )
        return circuit
    else:
        return stim.Circuit.generated(
            f"surface_code:rotated_memory_{basis.lower()}",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p,
        )


def rebuild_from_meta(params_canonical: str) -> stim.Circuit:
    """Rebuild a Stim circuit from shard metadata params_canonical JSON."""
    params = params_from_canonical(params_canonical)
    if params["distance"] == 0:
        raise ValueError(f"Cannot extract valid circuit params from: {params_canonical}")
    return rebuild_stim_circuit(**params)


def params_from_canonical(params_canonical: str) -> Dict[str, Any]:
    """Extract circuit build parameters from params_canonical JSON."""
    try:
        obj = json.loads(params_canonical)
        c = obj.get("circuit", {})
        return {
            "distance": c.get("distance", 0),
            "rounds": c.get("rounds", 0),
            "basis": c.get("basis", "X").upper(),
            "noise_model": c.get("noise_model", "baseline_symmetric"),
            "p": float(c.get("p", 0.0)),
        }
    except (json.JSONDecodeError, TypeError):
        return {"distance": 0, "rounds": 0, "basis": "X", "noise_model": "", "p": 0.0}


def circuit_cache_key(params_canonical: str) -> Tuple:
    """Extract hashable cache key from params_canonical."""
    try:
        obj = json.loads(params_canonical)
        c = obj.get("circuit", {})
        return (
            c.get("type", "unknown"),
            c.get("distance", 0),
            c.get("rounds", 0),
            c.get("basis", "").upper(),
            c.get("noise_model", ""),
            float(c.get("p", 0.0)),
        )
    except (json.JSONDecodeError, TypeError):
        return ("unknown", 0, 0, "", "", 0.0)
