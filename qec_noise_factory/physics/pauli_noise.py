"""
Core Physics Data Structures for Pauli Noise.

This module defines the mathematical representation of quantum noise channels.
It provides:
  - PauliChannel: a single-qubit Pauli error channel (px, py, pz)
  - PauliNoiseModel: a complete noise model combining data, measurement,
    reset, idle, and 2-qubit gate noise

Both classes are frozen (immutable) for safe hashing and provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# PauliChannel — single-qubit error channel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PauliChannel:
    """
    Represents a single-qubit Pauli error channel.
    Defined by probabilities (px, py, pz) where:
      - px = probability of X error
      - py = probability of Y error
      - pz = probability of Z error
      - pi = 1 - (px + py + pz) = probability of no error (Identity)

    Frozen (immutable) to allow hashing and safe storage.
    """
    px: float = 0.0
    py: float = 0.0
    pz: float = 0.0

    def __post_init__(self):
        """Validate probabilities on initialization."""
        if self.px < 0 or self.py < 0 or self.pz < 0:
            raise ValueError("Probabilities must be non-negative")
        total = self.px + self.py + self.pz
        if total > 1.0 + 1e-9:
            raise ValueError(f"Total error probability {total} exceeds 1.0")

    @property
    def pi(self) -> float:
        """Probability of Identity (no error)."""
        return max(0.0, 1.0 - (self.px + self.py + self.pz))

    def total_error(self) -> float:
        """Returns the sum of error probabilities (px + py + pz)."""
        return self.px + self.py + self.pz

    def validate(self):
        """Explicit validation method."""
        self.__post_init__()

    # --- Utilities ---

    def scale(self, factor: float) -> PauliChannel:
        """Returns a new channel with all probabilities scaled by *factor*.

        The result is clamped so that the total does not exceed 1.0.
        """
        if factor < 0:
            raise ValueError("Scale factor must be non-negative")
        px = self.px * factor
        py = self.py * factor
        pz = self.pz * factor
        total = px + py + pz
        if total > 1.0:
            # Scale down proportionally to fit within valid range
            r = 1.0 / total
            px *= r
            py *= r
            pz *= r
        return PauliChannel(px=px, py=py, pz=pz)

    def clamp(self) -> PauliChannel:
        """Returns a new channel with probabilities clamped to valid range.

        Each probability is clamped to [0, 1] individually, then if the sum
        exceeds 1.0 they are scaled down proportionally.
        """
        px = max(0.0, min(1.0, self.px))
        py = max(0.0, min(1.0, self.py))
        pz = max(0.0, min(1.0, self.pz))
        total = px + py + pz
        if total > 1.0:
            r = 1.0 / total
            px *= r
            py *= r
            pz *= r
        return PauliChannel(px=px, py=py, pz=pz)

    def canonical_tuple(self) -> tuple[float, float, float]:
        """Returns (px, py, pz) rounded to 12 significant digits.

        Used for stable hashing and comparison regardless of floating-point
        representation differences.
        """
        return (round(self.px, 12), round(self.py, 12), round(self.pz, 12))

    def to_dict(self) -> dict:
        """Serialize to dict for manifest/DB storage."""
        return {"px": self.px, "py": self.py, "pz": self.pz}

    @classmethod
    def from_dict(cls, d: dict) -> PauliChannel:
        """Reconstruct from dict."""
        return cls(px=float(d["px"]), py=float(d["py"]), pz=float(d["pz"]))

    def __repr__(self) -> str:
        return f"PauliChannel(X={self.px:.4f}, Y={self.py:.4f}, Z={self.pz:.4f})"


# ---------------------------------------------------------------------------
# PauliNoiseModel — full noise configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PauliNoiseModel:
    """
    Represents a complete noise model configuration for a simulation.

    Layers:
      - data_noise:   1-qubit Pauli channel on data qubits (gate errors)
      - meas_flip:    measurement bit-flip probability
      - reset_flip:   reset error probability
      - idle_noise:   1-qubit Pauli channel on idle qubits (optional)
      - prob_2q_gate: 2-qubit gate depolarization probability

    Metadata:
      - name:           preset name (e.g. 'si1000_like')
      - schema_version: version of the model schema for forward compatibility

    This class describes **what** the noise is (physics).
    The NoiseCompiler decides **how** to express it in Stim (engineering).
    """
    data_noise: PauliChannel
    meas_flip: float = 0.0
    reset_flip: float = 0.0
    idle_noise: Optional[PauliChannel] = None
    prob_2q_gate: float = 0.0
    name: str = ""
    schema_version: int = 1

    # --- Canonicalization & Hashing ---

    def canonical_dict(self) -> dict:
        """Returns a sorted dictionary of all model parameters.

        Used for stable JSON serialization and provenance tracking.
        All values are rounded to 12 digits for stability.
        """
        d = {
            "data_px": round(self.data_noise.px, 12),
            "data_py": round(self.data_noise.py, 12),
            "data_pz": round(self.data_noise.pz, 12),
            "meas_flip": round(self.meas_flip, 12),
            "reset_flip": round(self.reset_flip, 12),
            "prob_2q_gate": round(self.prob_2q_gate, 12),
        }
        if self.idle_noise is not None:
            d["idle_px"] = round(self.idle_noise.px, 12)
            d["idle_py"] = round(self.idle_noise.py, 12)
            d["idle_pz"] = round(self.idle_noise.pz, 12)
        return dict(sorted(d.items()))

    def canonical_hash(self) -> str:
        """Returns SHA256 hex digest of the canonical representation.

        Two models with identical physics will always produce the same hash,
        regardless of construction order or floating-point path.
        """
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        """Full serialization for manifest/DB storage."""
        d = {
            "name": self.name,
            "schema_version": self.schema_version,
            "data_noise": self.data_noise.to_dict(),
            "meas_flip": self.meas_flip,
            "reset_flip": self.reset_flip,
            "prob_2q_gate": self.prob_2q_gate,
        }
        if self.idle_noise is not None:
            d["idle_noise"] = self.idle_noise.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PauliNoiseModel:
        """Reconstruct from dict (manifest/DB)."""
        idle = None
        if "idle_noise" in d:
            idle = PauliChannel.from_dict(d["idle_noise"])
        return cls(
            data_noise=PauliChannel.from_dict(d["data_noise"]),
            meas_flip=float(d.get("meas_flip", 0.0)),
            reset_flip=float(d.get("reset_flip", 0.0)),
            idle_noise=idle,
            prob_2q_gate=float(d.get("prob_2q_gate", 0.0)),
            name=str(d.get("name", "")),
            schema_version=int(d.get("schema_version", 1)),
        )

    # --- Utilities ---

    def scale(self, factor: float) -> PauliNoiseModel:
        """Returns a new model with all error rates scaled by *factor*.

        Useful for parameter sweeps: model.scale(0.5) halves all noise.
        """
        if factor < 0:
            raise ValueError("Scale factor must be non-negative")
        return PauliNoiseModel(
            data_noise=self.data_noise.scale(factor),
            meas_flip=min(1.0, self.meas_flip * factor),
            reset_flip=min(1.0, self.reset_flip * factor),
            idle_noise=self.idle_noise.scale(factor) if self.idle_noise else None,
            prob_2q_gate=min(1.0, self.prob_2q_gate * factor),
        )

    def clamp(self) -> PauliNoiseModel:
        """Returns a new model with all probabilities clamped to [0, 1]."""
        return PauliNoiseModel(
            data_noise=self.data_noise.clamp(),
            meas_flip=max(0.0, min(1.0, self.meas_flip)),
            reset_flip=max(0.0, min(1.0, self.reset_flip)),
            idle_noise=self.idle_noise.clamp() if self.idle_noise else None,
            prob_2q_gate=max(0.0, min(1.0, self.prob_2q_gate)),
        )

    def compose(self, other: PauliNoiseModel) -> PauliNoiseModel:
        """Combines two noise models by adding their error rates (clamped).

        Useful for modeling compound noise sources. The result is clamped
        so all probabilities remain physically valid.
        """
        # Combine idle: use other's if self has none, otherwise add
        if self.idle_noise is None:
            combined_idle = other.idle_noise
        elif other.idle_noise is None:
            combined_idle = self.idle_noise
        else:
            combined_idle = PauliChannel(
                px=self.idle_noise.px + other.idle_noise.px,
                py=self.idle_noise.py + other.idle_noise.py,
                pz=self.idle_noise.pz + other.idle_noise.pz,
            ).clamp()

        return PauliNoiseModel(
            data_noise=PauliChannel(
                px=self.data_noise.px + other.data_noise.px,
                py=self.data_noise.py + other.data_noise.py,
                pz=self.data_noise.pz + other.data_noise.pz,
            ).clamp(),
            meas_flip=min(1.0, self.meas_flip + other.meas_flip),
            reset_flip=min(1.0, self.reset_flip + other.reset_flip),
            idle_noise=combined_idle,
            prob_2q_gate=min(1.0, self.prob_2q_gate + other.prob_2q_gate),
        ).clamp()

    # --- Factory Presets ---

    @classmethod
    def from_symmetric_depolarizing(cls, p: float, meas_flip: float = 0.0) -> PauliNoiseModel:
        """Creates a model where X, Y, Z errors are equally likely.

        This is the simplest noise model: px = py = pz = p/3.
        Used by baseline_symmetric packs.
        """
        if p == 0:
            return cls(data_noise=PauliChannel(), meas_flip=meas_flip, name="baseline_symmetric")

        p_per_pauli = p / 3.0
        channel = PauliChannel(px=p_per_pauli, py=p_per_pauli, pz=p_per_pauli)
        return cls(data_noise=channel, meas_flip=meas_flip, name="baseline_symmetric")

    @classmethod
    def from_biased_z(cls, p: float, bias: float = 1.0, meas_flip: float = 0.0) -> PauliNoiseModel:
        """Creates a model biased towards Z errors.

        Physics: Total error probability = p, but Pz = bias × Px.
        Math: Px = Py = p / (2 + bias), Pz = bias × Px.
        When bias=1 this reduces to symmetric depolarizing.
        """
        if p == 0:
            return cls(data_noise=PauliChannel(), meas_flip=meas_flip, name="biased_z")

        px = p / (2.0 + bias)
        py = px
        pz = bias * px

        channel = PauliChannel(px=px, py=py, pz=pz)
        return cls(data_noise=channel, meas_flip=meas_flip, name="biased_z")

    @classmethod
    def from_sd6(cls, p: float) -> PauliNoiseModel:
        """Creates an SD6-like noise model.

        SD6: gate depolarization + measurement/reset + idle per tick.
        p represents the base physical error rate.

        Ratios (SD6-spirit, pragmatic defaults):
          - p_1q = 0.1 × p      (1-qubit gate depolarization)
          - p_2q = 1.0 × p      (2-qubit gate depolarization)
          - p_meas = 1.0 × p    (measurement flip)
          - p_reset = 0.2 × p   (reset error)
          - p_idle = 0.02 × p   (idle depolarization)
        """
        p1 = 0.1 * p
        p1_pauli = p1 / 3.0
        data_ch = PauliChannel(px=p1_pauli, py=p1_pauli, pz=p1_pauli)

        p_idle = 0.02 * p
        p_idle_pauli = p_idle / 3.0
        idle_ch = PauliChannel(px=p_idle_pauli, py=p_idle_pauli, pz=p_idle_pauli)

        return cls(
            data_noise=data_ch,
            meas_flip=1.0 * p,
            reset_flip=0.2 * p,
            idle_noise=idle_ch,
            prob_2q_gate=1.0 * p,
            name="sd6_like",
        )

    @classmethod
    def from_si1000(cls, p: float) -> PauliNoiseModel:
        """Creates an SI1000-like (superconducting-inspired) noise model.

        p represents the 2-qubit gate error rate (dominant error source).

        Ratios:
          - p_1q = p / 10       (1-qubit gate error)
          - p_2q = p            (2-qubit gate error)
          - p_meas = 2 × p      (measurement flip)
          - p_reset = 5 × p     (reset error)
          - p_idle = p / 20     (idle depolarization)
        """
        p1 = p / 10.0
        p_per_pauli = p1 / 3.0
        data_ch = PauliChannel(px=p_per_pauli, py=p_per_pauli, pz=p_per_pauli)

        p_idle = p / 20.0
        p_idle_pauli = p_idle / 3.0
        idle_ch = PauliChannel(px=p_idle_pauli, py=p_idle_pauli, pz=p_idle_pauli)

        return cls(
            data_noise=data_ch,
            meas_flip=2.0 * p,
            reset_flip=5.0 * p,
            idle_noise=idle_ch,
            prob_2q_gate=p,
            name="si1000_like",
        )

    @classmethod
    def from_correlated_crosstalk(cls, p: float, corr_strength: float = 0.5) -> PauliNoiseModel:
        """Creates a correlated-crosstalk noise model (Day 30).

        Combines SD6-like independent noise at reduced rate with correlated
        two-qubit errors that produce hyperedges (k>2) in the DEM.

        MWPM must approximate these hyperedges via clique expansion, which
        is inherently lossy — this is the regime where GNN could outperform.

        Parameters:
            p: base physical error rate
            corr_strength: fraction of error budget allocated to correlated
                           errors (0-1). Default 0.5 = half independent,
                           half correlated.

        Independent noise (SD6-like, scaled by 1-corr_strength):
          - p_1q = 0.1 × p_ind, p_2q = 1.0 × p_ind
          - p_meas = 1.0 × p_ind, p_reset = 0.2 × p_ind
          - p_idle = 0.02 × p_ind
          where p_ind = (1 - corr_strength) × p

        Correlated noise:
          - p_corr = corr_strength × p (applied as CORRELATED_ERROR
            after CX/CZ gates by the circuit builder)
        """
        if not (0.0 <= corr_strength <= 1.0):
            raise ValueError(f"corr_strength must be in [0,1], got {corr_strength}")

        p_ind = (1.0 - corr_strength) * p

        p1 = 0.1 * p_ind
        p1_pauli = p1 / 3.0
        data_ch = PauliChannel(px=p1_pauli, py=p1_pauli, pz=p1_pauli)

        p_idle = 0.02 * p_ind
        p_idle_pauli = p_idle / 3.0
        idle_ch = PauliChannel(px=p_idle_pauli, py=p_idle_pauli, pz=p_idle_pauli)

        return cls(
            data_noise=data_ch,
            meas_flip=1.0 * p_ind,
            reset_flip=0.2 * p_ind,
            idle_noise=idle_ch,
            prob_2q_gate=1.0 * p_ind,
            name="correlated_crosstalk_like",
        )
