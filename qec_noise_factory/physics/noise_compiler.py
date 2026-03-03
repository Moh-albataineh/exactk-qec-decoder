"""
Noise Compiler — translates PauliNoiseModel into Stim representations.

This module is the bridge between the physics layer (pauli_noise.py) and the
circuit construction layer (circuits_qc.py / noise_models.py).

Three compilation paths:
  1. compile_to_phenomenological — for Stim's Circuit.generated() parameters
  2. compile_to_circuit_level    — for inject_circuit_level_noise()
  3. compile_to_pauli_channel_1  — for PAULI_CHANNEL_1 injection (biased noise)
"""

from __future__ import annotations

from qec_noise_factory.physics.pauli_noise import PauliNoiseModel
from qec_noise_factory.factory.noise_models import CircuitLevelParams


def compile_to_phenomenological(model: PauliNoiseModel) -> dict:
    """Compiles a PauliNoiseModel into Stim's phenomenological noise parameters.

    Returns a dict suitable for passing to stim.Circuit.generated():
        {
            "after_clifford_depolarization": ...,
            "before_round_data_depolarization": ...,
            "after_reset_flip_probability": ...,
            "before_measure_flip_probability": ...,
        }

    This path is used for:
      - baseline_symmetric packs (uniform depolarizing)
      - Any pack that uses Stim's built-in noise injection

    The data_noise channel is converted to a total depolarization rate.
    Bias information is lost in this path (Stim's generated circuits
    use symmetric depolarizing only). For biased noise, use
    compile_to_pauli_channel_1() instead.
    """
    p_data = model.data_noise.total_error()
    return {
        "after_clifford_depolarization": p_data,
        "before_round_data_depolarization": p_data,
        "after_reset_flip_probability": model.reset_flip,
        "before_measure_flip_probability": model.meas_flip,
    }


def compile_to_circuit_level(model: PauliNoiseModel) -> CircuitLevelParams:
    """Compiles a PauliNoiseModel into CircuitLevelParams for circuit-level injection.

    Returns a CircuitLevelParams(p1, p2, p_idle, p_meas, p_reset) suitable
    for inject_circuit_level_noise().

    This path is used for:
      - sd6_like packs
      - si1000_like packs
      - Any pack requiring gate-by-gate noise injection

    Mapping:
      - p1     = data_noise.total_error()     (1-qubit gate depolarization)
      - p2     = prob_2q_gate                  (2-qubit gate depolarization)
      - p_idle = idle_noise.total_error()      (idle depolarization per TICK)
      - p_meas = meas_flip                     (measurement bit-flip)
      - p_reset = reset_flip                   (reset error)
    """
    p_idle = model.idle_noise.total_error() if model.idle_noise else 0.0

    return CircuitLevelParams(
        p1=model.data_noise.total_error(),
        p2=model.prob_2q_gate,
        p_idle=p_idle,
        p_meas=model.meas_flip,
        p_reset=model.reset_flip,
    )


def compile_to_pauli_channel_1(model: PauliNoiseModel) -> tuple[float, float, float]:
    """Compiles a PauliNoiseModel into (px, py, pz) for PAULI_CHANNEL_1.

    Returns the raw Pauli probabilities from the data_noise channel.

    This path is used for:
      - biased_z packs (where we need asymmetric X/Y/Z noise)
      - Any scenario where Stim's PAULI_CHANNEL_1 instruction is used

    Unlike compiple_to_phenomenological(), this preserves the full bias
    structure of the noise model.
    """
    ch = model.data_noise
    return (ch.px, ch.py, ch.pz)


# ---------------------------------------------------------------------------
# Convenience: Model name → PauliNoiseModel
# ---------------------------------------------------------------------------

def model_from_preset(name: str, p: float, **kwargs) -> PauliNoiseModel:
    """Creates a PauliNoiseModel from a named preset.

    Supported presets:
      - "baseline_symmetric" → from_symmetric_depolarizing(p)
      - "biased_z"           → from_biased_z(p, bias=kwargs["bias"])
      - "sd6_like"           → from_sd6(p)
      - "si1000_like"        → from_si1000(p)

    This replaces scattered if/elif chains with a single entry point.
    """
    name = name.strip().lower()

    if name == "baseline_symmetric":
        return PauliNoiseModel.from_symmetric_depolarizing(p, meas_flip=kwargs.get("meas_flip", 0.0))

    if name == "biased_z":
        bias = float(kwargs.get("bias", 1.0))
        return PauliNoiseModel.from_biased_z(p, bias=bias, meas_flip=kwargs.get("meas_flip", 0.0))

    if name == "sd6_like":
        return PauliNoiseModel.from_sd6(p)

    if name == "si1000_like":
        return PauliNoiseModel.from_si1000(p)

    if name == "correlated_crosstalk_like":
        corr_strength = float(kwargs.get("corr_strength", 0.5))
        return PauliNoiseModel.from_correlated_crosstalk(p, corr_strength=corr_strength)

    raise ValueError(f"Unknown noise model preset: {name}")
