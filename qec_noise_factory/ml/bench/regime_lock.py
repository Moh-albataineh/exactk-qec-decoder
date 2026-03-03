"""
Regime Lock — Day 37.1

Enforces that decision experiments use ONLY generated data at a fixed
target_p. No silent fallback to real shards allowed.

Usage:
    lock = RegimeLock(distance=5, target_p=0.04, require_generated=True)
    X, Y = generate_locked_data(lock)
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclasses.dataclass
class RegimeLock:
    """Regime parameters for decision experiments."""
    distance: int = 5
    target_p: float = 0.04
    basis: str = "X"
    require_generated: bool = True
    n_samples: int = 4096
    noise_model: str = "correlated_crosstalk_like"
    corr_strength: float = 0.5
    seed: int = 37101


class RegimeLockError(Exception):
    """Raised when regime lock constraints are violated."""
    def __init__(self, message: str, reason_code: str):
        super().__init__(message)
        self.reason_code = reason_code


def check_regime(
    lock: RegimeLock,
    data_source: str,
    p_used: Optional[float] = None,
) -> None:
    """Validate that data matches regime lock constraints.

    Raises RegimeLockError if any constraint is violated.
    """
    from qec_noise_factory.ml.bench.reason_codes import (
        ERR_REGIME_LOCK_REQUIRED,
        ERR_TARGET_P_MISSING,
    )

    if lock.require_generated and not data_source.startswith("generated"):
        raise RegimeLockError(
            f"Regime lock requires generated data, got: {data_source}",
            ERR_REGIME_LOCK_REQUIRED,
        )

    if p_used is not None and abs(p_used - lock.target_p) > 1e-6:
        raise RegimeLockError(
            f"p_used={p_used} != target_p={lock.target_p}",
            ERR_TARGET_P_MISSING,
        )


def generate_locked_data(
    lock: RegimeLock,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data at exact target_p using stim. FAILS if stim unavailable.

    Returns:
        (X, Y) — detector matrix and observable labels.

    Raises:
        RegimeLockError if stim is not installed.
    """
    from qec_noise_factory.ml.bench.reason_codes import ERR_STIM_REQUIRED_FOR_DECISION

    try:
        import stim  # noqa: F401
    except ImportError:
        raise RegimeLockError(
            "stim is required for decision experiments (regime lock). "
            "Install with: pip install stim",
            ERR_STIM_REQUIRED_FOR_DECISION,
        )

    from qec_noise_factory.ml.stim.rebuild import rebuild_stim_circuit

    circuit = rebuild_stim_circuit(
        distance=lock.distance,
        rounds=lock.distance,
        p=lock.target_p,
        basis=lock.basis,
        noise_model=lock.noise_model,
    )

    sampler = circuit.compile_detector_sampler(seed=lock.seed)
    det_data, obs_data = sampler.sample(
        shots=lock.n_samples,
        separate_observables=True,
        bit_packed=False,
    )

    return det_data.astype(np.uint8), obs_data.astype(np.uint8)
