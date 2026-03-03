"""
p-Grid Selector — Day 31 + Day 31.5 upgrades

Deterministic selection of informative p values for correlated noise benchmarks.
Avoids trivial (near-zero error) and saturated (too many errors) regimes.

Day 31.5 additions:
  - MWPM triviality probe during pre-scan
  - Nearest-p binning (replaces ±10% window)
  - Data availability checking + on-demand generation
  - Candidate vs selected separation

Strategy:
  1. Define a wide candidate pool of p values (log-spaced).
  2. Pre-scan each candidate: build circuit, sample a small batch, measure
     detector_density (fraction of fired detectors) and y_rate (observable
     flip rate).
  3. Optionally run MWPM probe: quick decode to detect trivial regimes.
  4. Reject candidates outside target bands or with probe F1 > 0.995.
  5. Select ≥5 p values spanning low/mid/high informative bands.
  6. Return deterministic, serializable result.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import stim

from qec_noise_factory.factory.circuits_qc import (
    build_surface_code_memory_circuit_level,
)
from qec_noise_factory.utils.hashing import canonical_json


# ---------------------------------------------------------------------------
# Defaults (tunable, but must exist)
# ---------------------------------------------------------------------------

DEFAULT_CANDIDATE_P = [
    0.001, 0.002, 0.003, 0.005, 0.007,
    0.01, 0.015, 0.02, 0.03, 0.04,
    0.05, 0.07, 0.10, 0.12, 0.15, 0.20,
]

# Target bands for informativeness
DEFAULT_MIN_Y_RATE = 0.01       # reject if y_rate < this (trivial)
DEFAULT_MAX_Y_RATE = 0.45       # reject if y_rate > this (saturated)
DEFAULT_MIN_DENSITY = 0.005     # reject if density < this (trivial)
DEFAULT_MAX_DENSITY = 0.22      # reject if density > this (saturated)

# Bands for binning (low / mid / high)
BAND_LOW = (0.01, 0.08)    # y_rate range for "low" band
BAND_MID = (0.08, 0.22)    # y_rate range for "mid" band
BAND_HIGH = (0.22, 0.45)   # y_rate range for "high" band

# MWPM probe defaults
DEFAULT_MWPM_PROBE_SHOTS = 256
DEFAULT_MWPM_TRIVIAL_THRESHOLD = 0.995


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PGridThresholds:
    """Thresholds used for p-grid selection."""
    min_y_rate: float = DEFAULT_MIN_Y_RATE
    max_y_rate: float = DEFAULT_MAX_Y_RATE
    min_density: float = DEFAULT_MIN_DENSITY
    max_density: float = DEFAULT_MAX_DENSITY
    mwpm_trivial_f1: float = DEFAULT_MWPM_TRIVIAL_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PPointStats:
    """Pre-scan statistics for one candidate p value."""
    p: float
    detector_density: float     # mean fraction of fired detectors
    y_rate: float               # fraction of observable flips
    num_detectors: int
    num_observables: int
    shots_used: int
    accepted: bool
    reject_reason: str = ""     # empty if accepted
    mwpm_probe_f1: float = -1.0  # Day 31.5: -1 means not probed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PGridResult:
    """Result of p-grid selection."""
    p_grid: List[float]                    # selected informative p values
    per_p_stats: List[PPointStats]         # stats for ALL candidates
    thresholds: PGridThresholds
    rejected_p: List[float]               # rejected candidates
    distance: int
    basis: str
    noise_model: str
    corr_strength: float
    seed: int
    scan_shots: int
    elapsed_s: float = 0.0
    config_hash: str = ""                  # hash of selection config
    candidate_reject_rate: float = 0.0     # Day 31.5: fraction rejected

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "p_grid": self.p_grid,
            "per_p_stats": [s.to_dict() for s in self.per_p_stats],
            "thresholds": self.thresholds.to_dict(),
            "rejected_p": self.rejected_p,
            "distance": self.distance,
            "basis": self.basis,
            "noise_model": self.noise_model,
            "corr_strength": self.corr_strength,
            "seed": self.seed,
            "scan_shots": self.scan_shots,
            "elapsed_s": self.elapsed_s,
            "config_hash": self.config_hash,
            "n_selected": len(self.p_grid),
            "n_rejected": len(self.rejected_p),
            "n_candidates": len(self.per_p_stats),
            "candidate_reject_rate": self.candidate_reject_rate,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Pre-scan
# ---------------------------------------------------------------------------

def _prescan_p(
    p: float,
    distance: int,
    basis: str,
    noise_model: str,
    corr_strength: float,
    shots: int,
    seed: int,
    run_mwpm_probe: bool = False,
    mwpm_probe_shots: int = DEFAULT_MWPM_PROBE_SHOTS,
) -> PPointStats:
    """Sample a small batch at one p and compute informativeness stats.

    If run_mwpm_probe=True, also runs a fast MWPM decode to measure
    mwpm_probe_f1 (detects trivially solvable regimes).
    """
    overrides = {}
    if noise_model == "correlated_crosstalk_like":
        overrides["corr_strength"] = corr_strength

    circuit, _ = build_surface_code_memory_circuit_level(
        distance=distance,
        rounds=distance,
        p_base=p,
        basis=basis,
        noise_model=noise_model,
        noise_params_overrides=overrides if overrides else None,
    )

    num_det = circuit.num_detectors
    num_obs = circuit.num_observables

    sampler = circuit.compile_detector_sampler(seed=seed)
    det_obs = sampler.sample(shots=shots, separate_observables=True)
    det_array, obs_array = det_obs

    # Detector density: mean fraction of detectors that fire per shot
    detector_density = float(np.mean(det_array))

    # Y-rate: fraction of shots where at least one observable flips
    any_flip = np.any(obs_array, axis=1)
    y_rate = float(np.mean(any_flip))

    # MWPM probe
    probe_f1 = -1.0
    if run_mwpm_probe:
        probe_f1 = _run_mwpm_probe(circuit, mwpm_probe_shots, seed + 99_000)

    return PPointStats(
        p=p,
        detector_density=detector_density,
        y_rate=y_rate,
        num_detectors=num_det,
        num_observables=num_obs,
        shots_used=shots,
        accepted=True,
        mwpm_probe_f1=probe_f1,
    )


def _run_mwpm_probe(
    circuit: stim.Circuit,
    shots: int,
    seed: int,
) -> float:
    """Quick MWPM decode probe to measure F1 on a small sample.

    Returns F1 score (or 1.0 if perfect / too few errors to measure).
    """
    try:
        import pymatching

        sampler = circuit.compile_detector_sampler(seed=seed)
        det_obs = sampler.sample(shots=shots, separate_observables=True)
        det_array, obs_array = det_obs

        dem = circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(dem)

        predictions = np.zeros(obs_array.shape, dtype=np.uint8)
        for i in range(shots):
            predictions[i] = matching.decode(det_array[i])

        # Flatten to 1D for F1
        y_true = obs_array.flatten().astype(int)
        y_pred = predictions.flatten().astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        if tp + fp == 0 or tp + fn == 0:
            return 1.0  # no errors to classify → trivially perfect

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    except Exception:
        return -1.0  # probe failed, don't block


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _classify_band(y_rate: float) -> str:
    """Classify a y_rate into low/mid/high band."""
    if BAND_LOW[0] <= y_rate < BAND_LOW[1]:
        return "low"
    elif BAND_MID[0] <= y_rate < BAND_MID[1]:
        return "mid"
    elif BAND_HIGH[0] <= y_rate <= BAND_HIGH[1]:
        return "high"
    return "out_of_band"


def select_p_grid_correlated(
    distance: int = 5,
    basis: str = "X",
    noise_model: str = "correlated_crosstalk_like",
    corr_strength: float = 0.5,
    candidate_p: Optional[List[float]] = None,
    thresholds: Optional[PGridThresholds] = None,
    scan_shots: int = 512,
    seed: int = 31_000,
    min_points: int = 5,
    run_mwpm_probe: bool = False,
    mwpm_probe_shots: int = DEFAULT_MWPM_PROBE_SHOTS,
) -> PGridResult:
    """Select informative p values for correlated noise benchmarks.

    Strategy:
      1. Pre-scan all candidates
      2. Reject trivial / saturated / MWPM-trivial
      3. Pick ≥min_points spanning low/mid/high bands
      4. If not enough points, relax thresholds and retry

    Returns PGridResult with selected grid + full diagnostics.
    """
    t0 = time.time()

    if candidate_p is None:
        candidate_p = list(DEFAULT_CANDIDATE_P)
    if thresholds is None:
        thresholds = PGridThresholds()

    # Config hash for reproducibility
    config_obj = {
        "distance": distance, "basis": basis, "noise_model": noise_model,
        "corr_strength": corr_strength, "candidate_p": candidate_p,
        "thresholds": thresholds.to_dict(), "scan_shots": scan_shots,
        "seed": seed, "run_mwpm_probe": run_mwpm_probe,
    }
    config_hash = hashlib.sha256(
        canonical_json(config_obj).encode()
    ).hexdigest()[:16]

    # 1. Pre-scan all candidates
    all_stats: List[PPointStats] = []
    for i, p in enumerate(sorted(candidate_p)):
        stats = _prescan_p(
            p=p, distance=distance, basis=basis,
            noise_model=noise_model, corr_strength=corr_strength,
            shots=scan_shots, seed=seed + i,
            run_mwpm_probe=run_mwpm_probe,
            mwpm_probe_shots=mwpm_probe_shots,
        )
        all_stats.append(stats)

    # 2. Apply thresholds
    accepted: List[PPointStats] = []
    rejected_p: List[float] = []

    for s in all_stats:
        reasons = []
        if s.y_rate < thresholds.min_y_rate:
            reasons.append(f"y_rate={s.y_rate:.4f}<{thresholds.min_y_rate}")
        if s.y_rate > thresholds.max_y_rate:
            reasons.append(f"y_rate={s.y_rate:.4f}>{thresholds.max_y_rate}")
        if s.detector_density < thresholds.min_density:
            reasons.append(f"density={s.detector_density:.4f}<{thresholds.min_density}")
        if s.detector_density > thresholds.max_density:
            reasons.append(f"density={s.detector_density:.4f}>{thresholds.max_density}")
        # Day 31.5: MWPM probe rejection
        if run_mwpm_probe and s.mwpm_probe_f1 >= 0:
            if s.mwpm_probe_f1 > thresholds.mwpm_trivial_f1:
                reasons.append(
                    f"mwpm_probe_f1={s.mwpm_probe_f1:.4f}>"
                    f"{thresholds.mwpm_trivial_f1}"
                )

        if reasons:
            s.accepted = False
            s.reject_reason = "; ".join(reasons)
            rejected_p.append(s.p)
        else:
            accepted.append(s)

    # 3. Select spanning low/mid/high bands
    selected = _select_spanning(accepted, min_points)

    # 4. If not enough, relax thresholds by 50% and retry
    if len(selected) < min_points:
        relaxed = PGridThresholds(
            min_y_rate=thresholds.min_y_rate * 0.5,
            max_y_rate=min(thresholds.max_y_rate * 1.5, 0.50),
            min_density=thresholds.min_density * 0.5,
            max_density=min(thresholds.max_density * 1.5, 0.30),
            mwpm_trivial_f1=thresholds.mwpm_trivial_f1,
        )
        # Re-accept with relaxed thresholds
        accepted_relaxed = []
        for s in all_stats:
            ok = (relaxed.min_y_rate <= s.y_rate <= relaxed.max_y_rate and
                  relaxed.min_density <= s.detector_density <= relaxed.max_density)
            if ok:
                accepted_relaxed.append(s)

        selected = _select_spanning(accepted_relaxed, min_points)
        thresholds = relaxed  # record which thresholds were used

    p_grid = sorted([s.p for s in selected])

    # Day 31.5: candidate rejection rate
    n_candidates = len(all_stats)
    candidate_reject_rate = len(rejected_p) / n_candidates if n_candidates > 0 else 0.0

    result = PGridResult(
        p_grid=p_grid,
        per_p_stats=all_stats,
        thresholds=thresholds,
        rejected_p=rejected_p,
        distance=distance,
        basis=basis,
        noise_model=noise_model,
        corr_strength=corr_strength,
        seed=seed,
        scan_shots=scan_shots,
        elapsed_s=round(time.time() - t0, 2),
        config_hash=config_hash,
        candidate_reject_rate=round(candidate_reject_rate, 4),
    )

    return result


def _select_spanning(
    accepted: List[PPointStats],
    min_points: int,
) -> List[PPointStats]:
    """Select points spanning low/mid/high bands.

    Picks up to 2 from each band, then fills remaining from any band.
    """
    if not accepted:
        return []

    bands: Dict[str, List[PPointStats]] = {"low": [], "mid": [], "high": []}
    unclassified: List[PPointStats] = []

    for s in accepted:
        band = _classify_band(s.y_rate)
        if band in bands:
            bands[band].append(s)
        else:
            unclassified.append(s)

    selected: List[PPointStats] = []

    # Pick 1-2 from each band (prefer middle of band)
    for band_name in ["low", "mid", "high"]:
        candidates = bands[band_name]
        if not candidates:
            continue
        # Sort by y_rate, pick median and one neighbor
        candidates.sort(key=lambda s: s.y_rate)
        mid_idx = len(candidates) // 2
        selected.append(candidates[mid_idx])
        if len(candidates) > 1 and len(selected) < min_points:
            other_idx = 0 if mid_idx > 0 else min(1, len(candidates) - 1)
            if candidates[other_idx] not in selected:
                selected.append(candidates[other_idx])

    # Fill remaining from unclassified or additional band members
    remaining = [s for s in accepted if s not in selected]
    remaining.sort(key=lambda s: s.y_rate)
    for s in remaining:
        if len(selected) >= min_points:
            break
        selected.append(s)

    return selected


# ---------------------------------------------------------------------------
# Day 31.5: Nearest-p binning
# ---------------------------------------------------------------------------

def assign_nearest_p(
    sample_p_values: np.ndarray,
    grid_p: List[float],
    max_distance: float = 0.5,
) -> Dict[str, Any]:
    """Assign each sample to the nearest grid p value.

    Args:
        sample_p_values: 1D array of p values from dataset samples
        grid_p: sorted list of selected p grid points
        max_distance: maximum relative distance to accept (fraction of grid p)

    Returns dict with:
        bin_assignments: 1D int array (index into grid_p, -1 if unassigned)
        bin_counts: dict {p_str: count} for each grid point
        total_assigned: int
        total_unassigned: int
    """
    grid = np.array(sorted(grid_p), dtype=np.float64)
    assignments = np.full(len(sample_p_values), -1, dtype=np.int32)
    bin_counts: Dict[str, int] = {f"{p:.6f}": 0 for p in grid}

    for i, sp in enumerate(sample_p_values):
        # Find nearest grid point
        dists = np.abs(grid - sp)
        nearest_idx = int(np.argmin(dists))
        nearest_p = grid[nearest_idx]
        rel_dist = dists[nearest_idx] / nearest_p if nearest_p > 0 else float("inf")

        if rel_dist <= max_distance:
            assignments[i] = nearest_idx
            bin_counts[f"{nearest_p:.6f}"] += 1

    total_assigned = int(np.sum(assignments >= 0))
    return {
        "bin_assignments": assignments,
        "bin_counts": bin_counts,
        "total_assigned": total_assigned,
        "total_unassigned": len(sample_p_values) - total_assigned,
        "grid_p": list(grid),
    }


# ---------------------------------------------------------------------------
# Day 31.5: Data availability check + on-demand generation
# ---------------------------------------------------------------------------

def check_data_availability(
    grid_p: List[float],
    sample_p_values: np.ndarray,
    max_distance: float = 0.5,
    min_samples: int = 10,
) -> Dict[str, Any]:
    """Check which grid p values have sufficient data.

    Returns dict with:
        available: list of p with enough samples
        missing: list of p with < min_samples
        counts: dict of {p: count}
    """
    result = assign_nearest_p(sample_p_values, grid_p, max_distance)
    available = []
    missing = []
    counts: Dict[str, int] = {}
    for p_str, count in result["bin_counts"].items():
        p_val = float(p_str)
        counts[p_str] = count
        if count >= min_samples:
            available.append(p_val)
        else:
            missing.append(p_val)

    return {
        "available": available,
        "missing": missing,
        "counts": counts,
    }


def generate_mini_dataset(
    p: float,
    distance: int = 5,
    basis: str = "X",
    noise_model: str = "correlated_crosstalk_like",
    corr_strength: float = 0.5,
    n_samples: int = 2048,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a mini synthetic dataset on-demand for a missing p value.

    Returns dict with X (detector array), y (observable array),
    metadata, and provenance.
    """
    overrides = {}
    if noise_model == "correlated_crosstalk_like":
        overrides["corr_strength"] = corr_strength

    circuit, _ = build_surface_code_memory_circuit_level(
        distance=distance,
        rounds=distance,
        p_base=p,
        basis=basis,
        noise_model=noise_model,
        noise_params_overrides=overrides if overrides else None,
    )

    sampler = circuit.compile_detector_sampler(seed=seed)
    det_obs = sampler.sample(shots=n_samples, separate_observables=True)
    det_array, obs_array = det_obs

    return {
        "X": det_array.astype(np.float32),
        "y": obs_array.flatten().astype(np.int32),
        "p": p,
        "distance": distance,
        "basis": basis,
        "noise_model": noise_model,
        "corr_strength": corr_strength,
        "n_samples": n_samples,
        "seed": seed,
        "generated": True,
        "num_detectors": circuit.num_detectors,
    }

