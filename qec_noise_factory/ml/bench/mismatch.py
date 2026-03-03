"""
Mismatch Utilities — Day 29

Provides strategies for testing MWPM decoder under mismatched physics:
- Model mismatch: build MWPM DEM from wrong noise_model
- P-scale mismatch: multiply physical p before DEM weight computation

All functions are deterministic and return mismatch_info dicts for reporting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import numpy as np

from qec_noise_factory.ml.bench.mwpm_decoder import MWPMDecoder


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------

_EPS = 1e-12


def compute_matching_weight(p: float) -> float:
    """Compute MWPM matching weight: W = ln((1-p)/p). p must be in (0,1)."""
    p = float(np.clip(p, _EPS, 1.0 - _EPS))
    return math.log((1.0 - p) / p)


def compute_scaled_weight(p: float, p_scale: float) -> float:
    """
    Recompute matching weight with scaled p.

    p' = clamp(p * p_scale, eps, 1-eps)
    W' = ln((1-p')/p')
    """
    p_prime = float(np.clip(p * p_scale, _EPS, 1.0 - _EPS))
    return compute_matching_weight(p_prime)


def scale_p(p: float, p_scale: float) -> float:
    """Scale p and clamp to valid range (eps, 1-eps)."""
    return float(np.clip(p * p_scale, _EPS, 1.0 - _EPS))


# ---------------------------------------------------------------------------
# Mismatch info dataclass
# ---------------------------------------------------------------------------

@dataclass
class MismatchInfo:
    """Report-friendly record of what mismatch was applied."""
    strategy: str                # "oracle", "model_mismatch", "p_mismatch"
    true_noise_model: str = ""
    mismatch_noise_model: str = ""
    true_p: float = 0.0
    p_scale: float = 1.0
    effective_p: float = 0.0
    true_weight: float = 0.0
    mismatch_weight: float = 0.0
    build_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Mismatch builders
# ---------------------------------------------------------------------------

def build_oracle_mwpm(
    decoder: MWPMDecoder,
    *,
    distance: int,
    rounds: int,
    p: float,
    basis: str,
    noise_model: str = "baseline_symmetric",
) -> MismatchInfo:
    """
    Build MWPM with the correct (oracle) parameters.

    Returns MismatchInfo with strategy="oracle".
    """
    build_time = decoder.build(
        distance=distance, rounds=rounds, p=p,
        basis=basis, noise_model=noise_model,
    )
    return MismatchInfo(
        strategy="oracle",
        true_noise_model=noise_model,
        mismatch_noise_model=noise_model,
        true_p=p,
        p_scale=1.0,
        effective_p=p,
        true_weight=compute_matching_weight(p),
        mismatch_weight=compute_matching_weight(p),
        build_time_s=build_time,
    )


def build_model_mismatched_mwpm(
    decoder: MWPMDecoder,
    *,
    distance: int,
    rounds: int,
    p: float,
    basis: str,
    true_noise_model: str,
    mismatch_noise_model: str,
) -> MismatchInfo:
    """
    Build MWPM using a different noise model than the data was generated with.

    Example: data from si1000_like, MWPM built from baseline_symmetric.
    """
    build_time = decoder.build(
        distance=distance, rounds=rounds, p=p,
        basis=basis, noise_model=mismatch_noise_model,
    )
    return MismatchInfo(
        strategy="model_mismatch",
        true_noise_model=true_noise_model,
        mismatch_noise_model=mismatch_noise_model,
        true_p=p,
        p_scale=1.0,
        effective_p=p,
        true_weight=compute_matching_weight(p),
        mismatch_weight=compute_matching_weight(p),
        build_time_s=build_time,
    )


def build_p_scaled_mwpm(
    decoder: MWPMDecoder,
    *,
    distance: int,
    rounds: int,
    p: float,
    basis: str,
    noise_model: str = "baseline_symmetric",
    p_scale: float = 2.0,
) -> MismatchInfo:
    """
    Build MWPM with scaled p (mismatched error rate).

    The DEM is built with p' = clamp(p * p_scale, eps, 1-eps).
    This simulates having inaccurate knowledge of the physical error rate.
    """
    p_prime = scale_p(p, p_scale)
    build_time = decoder.build(
        distance=distance, rounds=rounds, p=p_prime,
        basis=basis, noise_model=noise_model,
    )
    return MismatchInfo(
        strategy="p_mismatch",
        true_noise_model=noise_model,
        mismatch_noise_model=noise_model,
        true_p=p,
        p_scale=p_scale,
        effective_p=p_prime,
        true_weight=compute_matching_weight(p),
        mismatch_weight=compute_matching_weight(p_prime),
        build_time_s=build_time,
    )
