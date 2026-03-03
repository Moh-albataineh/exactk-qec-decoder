"""
circuits_qc.py — Surface Code Circuit Builders

Provides functions to build Stim circuits for surface code memory experiments
with various noise models:

  1. Baseline symmetric (uniform depolarizing)
  2. Z-biased Pauli noise
  3. Circuit-level noise (SD6/SI1000/correlated)
  4. Unified builder (auto-dispatches by noise model type)
"""

from __future__ import annotations
import stim
from qec_noise_factory.factory.noise_models import CircuitLevelParams, make_model


# ---------------------------------------------------------------------------
# 1. Baseline Symmetric (Uniform Depolarizing)
# ---------------------------------------------------------------------------

def build_surface_code_memory(
    distance: int, rounds: int, p: float, basis: str
) -> stim.Circuit:
    """
    Build a surface code memory circuit with uniform depolarizing noise.

    Args:
        distance: Code distance (e.g., 3, 5, 7).
        rounds:   Number of QEC rounds.
        p:        Physical error rate (applied uniformly).
        basis:    Logical basis ("X" or "Z").

    Returns:
        A Stim circuit with phenomenological noise.
    """
    return stim.Circuit.generated(
        "surface_code:rotated_memory_" + basis.lower(),
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


# ---------------------------------------------------------------------------
# 2. Z-Biased Pauli Noise
# ---------------------------------------------------------------------------

def _biased_pauli_probs(p: float, bias: float) -> tuple[float, float, float]:
    """
    Compute per-axis Pauli probabilities from total error rate and Z-bias.

    Given total error rate p and bias = Pz/Px (with Px = Py):
        p = Px + Py + Pz = Px * (2 + bias)

    Args:
        p:    Total error probability.
        bias: Ratio Pz/Px (must be > 0).

    Returns:
        (px, py, pz) tuple.
    """
    if bias <= 0:
        raise ValueError("bias must be > 0")
    px = p / (2.0 + bias)
    py = px
    pz = bias * px
    return px, py, pz


def build_surface_code_memory_biased_z(
    distance: int, rounds: int, p: float, basis: str, bias: float
) -> stim.Circuit:
    """
    Build a surface code memory circuit with Z-biased Pauli noise.

    Generates a clean Stim circuit and injects PAULI_CHANNEL_1
    after every TICK with the computed (px, py, pz) probabilities.

    Args:
        distance: Code distance.
        rounds:   Number of QEC rounds.
        p:        Total error probability.
        basis:    Logical basis ("X" or "Z").
        bias:     Z-bias ratio (Pz/Px).

    Returns:
        A noisy Stim circuit with biased Pauli channels.
    """
    px, py, pz = _biased_pauli_probs(p, bias)

    clean_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_" + basis.lower(),
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.0,
        after_reset_flip_probability=0.0,
        before_measure_flip_probability=0.0,
        before_round_data_depolarization=0.0,
    )

    noisy_circuit = stim.Circuit()
    all_qubits = list(range(clean_circuit.num_qubits))
    for instr in clean_circuit:
        noisy_circuit.append(instr)
        if instr.name == "TICK":
            noisy_circuit.append("PAULI_CHANNEL_1", all_qubits, [px, py, pz])

    return noisy_circuit


# ---------------------------------------------------------------------------
# 3. Circuit-Level Noise (SD6 / SI1000 / Correlated Crosstalk)
# ---------------------------------------------------------------------------

def build_surface_code_memory_clean(
    distance: int, rounds: int, basis: str
) -> stim.Circuit:
    """
    Generate a perfectly clean Stim circuit skeleton (all noise rates = 0).

    Used as the base for circuit-level noise injection.
    """
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.0,
        before_round_data_depolarization=0.0,
        after_reset_flip_probability=0.0,
        before_measure_flip_probability=0.0,
    )


def build_surface_code_memory_circuit_level(
    distance: int,
    rounds: int,
    p_base: float,
    basis: str,
    noise_model: str,
    noise_params_overrides: dict | None = None,
) -> tuple[stim.Circuit, dict]:
    """
    Build a circuit with realistic gate-level noise (SD6/SI1000/correlated).

    Pipeline:
        1. Generate clean circuit skeleton
        2. Instantiate noise model and compute canonical parameters
        3. Apply any YAML overrides
        4. Inject noise into the circuit
        5. Return (noisy_circuit, metadata)

    Args:
        distance:               Code distance.
        rounds:                 Number of QEC rounds.
        p_base:                 Base physical error rate (scales all params).
        basis:                  Logical basis ("X" or "Z").
        noise_model:            One of "sd6_like", "si1000_like",
                                "correlated_crosstalk_like".
        noise_params_overrides: Optional dict to override canonical params.

    Returns:
        (noisy_circuit, meta) — circuit and provenance metadata dict.
    """
    # Clean skeleton
    clean = build_surface_code_memory_clean(distance, rounds, basis)

    # Noise model setup
    model_kwargs = {}
    if noise_model == "correlated_crosstalk_like":
        corr_strength = float(
            (noise_params_overrides or {}).get("corr_strength", 0.5)
        )
        model_kwargs["corr_strength"] = corr_strength
    model = make_model(noise_model, **model_kwargs)
    params = model.canonical(base_p=p_base)

    # Apply overrides
    if noise_params_overrides:
        for k, v in noise_params_overrides.items():
            if k in params:
                params[k] = float(v)

    # Build parameter object
    clp = CircuitLevelParams(
        p1=float(params["p1"]),
        p2=float(params["p2"]),
        p_idle=float(params["p_idle"]),
        p_meas=float(params["p_meas"]),
        p_reset=float(params["p_reset"]),
    )

    # Inject noise
    if noise_model == "correlated_crosstalk_like":
        p_corr = corr_strength * p_base
        noisy = model.apply_to_clean_circuit(clean, params=clp, p_corr=p_corr)
    else:
        noisy = model.apply_to_clean_circuit(clean, params=clp)

    # Provenance metadata
    meta = {
        "noise_model": model.name if hasattr(model, "name") else noise_model,
        "p_base": float(p_base),
        "p1": clp.p1,
        "p2": clp.p2,
        "p_idle": clp.p_idle,
        "p_meas": clp.p_meas,
        "p_reset": clp.p_reset,
    }
    if noise_model == "correlated_crosstalk_like":
        meta["corr_strength"] = corr_strength
        meta["p_corr"] = p_corr

    return noisy, meta


# ---------------------------------------------------------------------------
# 4. Unified Builder (Auto-Dispatch by Noise Model Type)
# ---------------------------------------------------------------------------

def build_surface_code_from_model(
    distance: int,
    rounds: int,
    basis: str,
    model: "PauliNoiseModel",
) -> tuple[stim.Circuit, dict]:
    """
    Unified circuit builder — compiles a PauliNoiseModel into a Stim circuit.

    Automatically selects the compilation path:
        - Circuit-level (SD6/SI1000): model has prob_2q_gate > 0
        - PAULI_CHANNEL_1: model has asymmetric X/Y/Z noise
        - Phenomenological: baseline symmetric

    Args:
        distance: Code distance.
        rounds:   Number of QEC rounds.
        basis:    Logical basis ("X" or "Z").
        model:    A PauliNoiseModel instance from the physics engine.

    Returns:
        (circuit, metadata) — noisy Stim circuit and provenance dict.
    """
    from qec_noise_factory.physics.noise_compiler import (
        compile_to_phenomenological,
        compile_to_circuit_level,
        compile_to_pauli_channel_1,
    )

    meta = model.canonical_dict()
    meta["distance"] = distance
    meta["rounds"] = rounds
    meta["basis"] = basis

    # Path 1: Circuit-level — has 2-qubit gate noise
    if model.prob_2q_gate > 0:
        clp = compile_to_circuit_level(model)
        clean = build_surface_code_memory_clean(distance, rounds, basis)
        circuit_model = make_model("sd6_like")
        noisy = circuit_model.apply_to_clean_circuit(clean, params=clp)
        meta["compilation_path"] = "circuit_level"
        return noisy, meta

    # Path 2: Biased noise — asymmetric X/Y/Z
    ch = model.data_noise
    if ch.px > 0 and abs(ch.pz - ch.px) > 1e-12:
        px, py, pz = compile_to_pauli_channel_1(model)
        clean = stim.Circuit.generated(
            f"surface_code:rotated_memory_{basis.lower()}",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=0.0,
            before_round_data_depolarization=0.0,
            after_reset_flip_probability=0.0,
            before_measure_flip_probability=0.0,
        )
        noisy = stim.Circuit()
        all_qubits = list(range(clean.num_qubits))
        for instr in clean:
            noisy.append(instr)
            if instr.name == "TICK":
                noisy.append("PAULI_CHANNEL_1", all_qubits, [px, py, pz])
        meta["compilation_path"] = "pauli_channel_1"
        return noisy, meta

    # Path 3: Phenomenological (baseline symmetric)
    params = compile_to_phenomenological(model)
    circuit = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}",
        distance=distance,
        rounds=rounds,
        **params,
    )
    meta["compilation_path"] = "phenomenological"
    return circuit, meta