"""
Data & Physics Quality Gates — Day 10

Computes shard-level metrics and applies quality gates to reject
invalid or uninformative data before it enters packs.

Metrics measured per shard:
  - detector_density:    fraction of 1-bits in detector array
  - row_all_zero_rate:   fraction of samples with all-zero detectors
  - row_all_one_rate:    fraction of samples with all-one detectors
  - y_rate:              mean of observable labels (logical flip rate)
  - num_samples, num_detectors

Quality gate decisions:
  - PASS  → shard is valid for pack inclusion
  - REJECT → shard is invalid (with reason_code)
  - WARN   → shard is borderline (included but flagged)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qec_noise_factory.verify import reason_codes as RC


# ---------------------------------------------------------------------------
# Default thresholds (can be overridden per-call)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "density_min": 1e-6,
    "density_max": 0.20,
    "row_all_zero_max": 0.95,
    "row_all_one_max": 0.10,
    "y_rate_extreme_lo": 1e-4,
    "y_rate_extreme_hi": 1.0 - 1e-4,
    "y_rate_warn_min_samples": 10_000,
    "qc_inconclusive_min_shots": 50_000,
}

# ---------------------------------------------------------------------------
# Per-family threshold overrides (hook for Day 11+)
# Circuit families may have different natural density ranges.
# Registered families override DEFAULT_THRESHOLDS for specific keys.
# Can also be loaded from YAML pack definitions.
# ---------------------------------------------------------------------------

THRESHOLDS_BY_FAMILY: dict[str, dict] = {
    # "surface_memory": {"density_max": 0.25},
    # "repetition_code": {"density_max": 0.30},
}


def resolve_thresholds(
    family: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> dict:
    """Merge default → family → explicit overrides (last wins)."""
    t = dict(DEFAULT_THRESHOLDS)
    if family and family in THRESHOLDS_BY_FAMILY:
        t.update(THRESHOLDS_BY_FAMILY[family])
    if overrides:
        t.update(overrides)
    return t


# ---------------------------------------------------------------------------
# Metrics result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShardMetrics:
    """Computed metrics for a single shard."""
    num_samples: int
    num_detectors: int
    detector_density: float
    row_all_zero_rate: float
    row_all_one_rate: float
    y_rate: float
    has_nan: bool = False
    has_inf: bool = False
    has_physics_provenance: bool = True


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    """Result of a quality gate check."""
    status: str                     # "pass", "reject", "warn"
    reason_code: str                # RC.ACCEPTED, RC.DOMAIN_INVALID_VALUES, etc.
    warnings: list = field(default_factory=list)
    metrics: Optional[ShardMetrics] = None


# ---------------------------------------------------------------------------
# 1. compute_shard_metrics
# ---------------------------------------------------------------------------

def compute_shard_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    meta: Optional[dict] = None,
) -> ShardMetrics:
    """
    Compute quality metrics for a shard of detector/observable data.

    Args:
        X: Detector array, shape (num_samples, num_detectors), dtype bool/uint8
        Y: Observable array, shape (num_samples,) or (num_samples, num_obs)
        meta: Optional metadata dict to check for NaN/Inf/provenance
    """
    if meta is None:
        meta = {}

    num_samples = X.shape[0]
    num_detectors = X.shape[1] if X.ndim > 1 else 1

    # Detector density: fraction of 1-bits
    total_bits = num_samples * num_detectors
    if total_bits > 0:
        detector_density = float(np.sum(X)) / total_bits
    else:
        detector_density = 0.0

    # Row saturation: all-zero and all-one rates
    if X.ndim > 1 and num_detectors > 0:
        row_sums = np.sum(X, axis=1)
        row_all_zero_rate = float(np.mean(row_sums == 0))
        row_all_one_rate = float(np.mean(row_sums == num_detectors))
    else:
        row_all_zero_rate = 0.0
        row_all_one_rate = 0.0

    # Label balance
    y_flat = Y.flatten() if Y.ndim > 1 else Y
    y_rate = float(np.mean(y_flat)) if len(y_flat) > 0 else 0.0

    # Schema checks on metadata
    has_nan = _check_nan_inf(meta, check_nan=True)
    has_inf = _check_nan_inf(meta, check_nan=False)

    # Physics provenance check
    has_physics_provenance = (
        "physics_model_name" in meta or "physics_hash" in meta
    )

    return ShardMetrics(
        num_samples=num_samples,
        num_detectors=num_detectors,
        detector_density=detector_density,
        row_all_zero_rate=row_all_zero_rate,
        row_all_one_rate=row_all_one_rate,
        y_rate=y_rate,
        has_nan=has_nan,
        has_inf=has_inf,
        has_physics_provenance=has_physics_provenance,
    )


def _check_nan_inf(meta: dict, check_nan: bool = True) -> bool:
    """Check metadata values for NaN or Inf."""
    for v in meta.values():
        if isinstance(v, float):
            if check_nan and math.isnan(v):
                return True
            if not check_nan and math.isinf(v):
                return True
    return False


# ---------------------------------------------------------------------------
# 2. quality_gate
# ---------------------------------------------------------------------------

def quality_gate(
    metrics: ShardMetrics,
    thresholds: Optional[dict] = None,
) -> GateResult:
    """
    Apply quality gate to shard metrics.

    Returns GateResult with status, reason_code, and warnings.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    warnings = []

    # --- Hard rejects ---

    # NaN or Inf in metadata
    if metrics.has_nan or metrics.has_inf:
        return GateResult("reject", RC.DOMAIN_INVALID_VALUES, ["NaN/Inf in metadata"], metrics)

    # Detector density too low = no signal
    if metrics.detector_density < t["density_min"]:
        return GateResult("reject", RC.DOMAIN_INVALID_VALUES,
                          [f"density={metrics.detector_density:.2e} < {t['density_min']:.0e}"], metrics)

    # Detector density too high = saturated
    if metrics.detector_density > t["density_max"]:
        return GateResult("reject", RC.PATHOLOGICAL_DISTRIBUTION,
                          [f"density={metrics.detector_density:.4f} > {t['density_max']}"], metrics)

    # All-zero saturation
    if metrics.row_all_zero_rate > t["row_all_zero_max"]:
        return GateResult("reject", RC.DOMAIN_INVALID_VALUES,
                          [f"row_all_zero_rate={metrics.row_all_zero_rate:.4f} > {t['row_all_zero_max']}"], metrics)

    # All-one saturation
    if metrics.row_all_one_rate > t["row_all_one_max"]:
        return GateResult("reject", RC.PATHOLOGICAL_DISTRIBUTION,
                          [f"row_all_one_rate={metrics.row_all_one_rate:.4f} > {t['row_all_one_max']}"], metrics)

    # --- Warnings ---

    # Label imbalance (only warn if enough samples)
    if metrics.num_samples >= t["y_rate_warn_min_samples"]:
        if metrics.y_rate < t["y_rate_extreme_lo"]:
            warnings.append(f"y_rate={metrics.y_rate:.6f} very low (< {t['y_rate_extreme_lo']:.0e})")
        elif metrics.y_rate > t["y_rate_extreme_hi"]:
            warnings.append(f"y_rate={metrics.y_rate:.6f} very high (> {t['y_rate_extreme_hi']})")

    # Missing physics provenance
    if not metrics.has_physics_provenance:
        warnings.append("missing physics_model_name / physics_hash in metadata")

    # Return pass (possibly with warnings)
    status = "warn" if warnings else "pass"
    return GateResult(status, RC.ACCEPTED, warnings, metrics)


# ---------------------------------------------------------------------------
# 3. qc_quality_check  (for QC / Sinter results)
# ---------------------------------------------------------------------------

def qc_quality_check(
    errors: int,
    shots: int,
    thresholds: Optional[dict] = None,
) -> GateResult:
    """
    Check QC simulation results for conclusiveness.

    If errors == 0 with insufficient shots, the result is inconclusive
    (we can't tell if the error rate is truly zero or just very small).
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    if errors == 0 and shots < t["qc_inconclusive_min_shots"]:
        return GateResult(
            "warn", RC.QC_INCONCLUSIVE,
            [f"errors=0 with only {shots} shots (< {t['qc_inconclusive_min_shots']})"],
        )

    return GateResult("pass", RC.ACCEPTED)
