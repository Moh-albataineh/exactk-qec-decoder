from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import stim


# ---------------------------
# Small utilities
# ---------------------------

_RE_REPEAT = re.compile(r"^\s*REPEAT\s+(\d+)\s*\{\s*$")
_RE_END = re.compile(r"^\s*\}\s*$")


def _fmt_p(x: float) -> str:
    # stable float formatting for stim
    return f"{x:.12g}"


def _is_int_token(tok: str) -> bool:
    # qubit targets are plain integers in generated surface code circuits
    tok = tok.strip()
    if tok.startswith("!"):
        tok = tok[1:]
    return tok.isdigit()


def _extract_qubits_from_line(line: str) -> List[int]:
    """
    Extract qubit indices from a stim instruction line.
    Ignores rec[...] and obs[...] tokens.
    """
    # remove params: GATE(p) -> GATE
    # we just want targets after gate name
    parts = line.strip().split()
    if not parts:
        return []
    gate = parts[0]
    if gate.startswith("#"):
        return []

    qubits: List[int] = []
    for tok in parts[1:]:
        # ignore record/observable references
        if tok.startswith("rec[") or tok.startswith("obs["):
            continue
        # sometimes tokens can be like "0," if copied badly; strip punctuation
        tok2 = tok.strip().strip(",")
        if _is_int_token(tok2):
            qubits.append(int(tok2.lstrip("!")))
    return qubits


def _discover_all_qubits(text: str) -> List[int]:
    mx = -1
    for line in text.splitlines():
        qs = _extract_qubits_from_line(line)
        if qs:
            mx = max(mx, max(qs))
    return list(range(mx + 1)) if mx >= 0 else []


# ---------------------------
# Noise Models
# ---------------------------

@dataclass(frozen=True)
class CircuitLevelParams:
    p1: float       # 1Q gate depolarization
    p2: float       # 2Q gate depolarization
    p_idle: float   # idling depolarization per TICK
    p_meas: float   # measurement bit-flip
    p_reset: float  # reset error


class CircuitNoiseModel:
    name: str

    def canonical(self, base_p: float) -> Dict[str, float]:
        raise NotImplementedError

    def apply_to_clean_circuit(self, clean: stim.Circuit, *, params: CircuitLevelParams) -> stim.Circuit:
        text = str(clean)
        noisy_text = inject_circuit_level_noise(text, params=params)
        return stim.Circuit(noisy_text)


class Sd6LikeNoiseModel(CircuitNoiseModel):
    """
    SD6-like: gate depolarization + measurement/reset + idle per tick.
    This is a pragmatic "SD6 spirit" model, not a claim of paper-exact mapping.
    """
    name = "sd6_like"

    def canonical(self, base_p: float) -> Dict[str, float]:
        # Ratios chosen to be reasonable defaults; tune later.
        # You can override via YAML.
        return {
            "p1": 0.1 * base_p,
            "p2": 1.0 * base_p,
            "p_idle": 0.02 * base_p,
            "p_meas": 1.0 * base_p,
            "p_reset": 0.2 * base_p,
        }


class Si1000LikeNoiseModel(CircuitNoiseModel):
    """
    SI1000-like: measurement/idling relatively heavier vs gates.
    Again: "like", not paper-exact, unless you later lock exact mapping.
    """
    name = "si1000_like"

    def canonical(self, base_p: float) -> Dict[str, float]:
        return {
            "p1": 0.05 * base_p,
            "p2": 0.5 * base_p,
            "p_idle": 0.2 * base_p,
            "p_meas": 2.0 * base_p,
            "p_reset": 0.2 * base_p,
        }


class CorrelatedCrosstalkModel(CircuitNoiseModel):
    """
    Day 30 — Correlated Crosstalk: SD6-like independent noise PLUS
    CORRELATED_ERROR after 2Q gates to produce hyperedges (k>2) in DEM.

    This is the regime where MWPM's clique-expansion approximation degrades.
    """
    name = "correlated_crosstalk_like"

    def __init__(self, corr_strength: float = 0.5):
        self.corr_strength = corr_strength

    def canonical(self, base_p: float) -> Dict[str, float]:
        p_ind = (1.0 - self.corr_strength) * base_p
        return {
            "p1": 0.1 * p_ind,
            "p2": 1.0 * p_ind,
            "p_idle": 0.02 * p_ind,
            "p_meas": 1.0 * p_ind,
            "p_reset": 0.2 * p_ind,
        }

    def apply_to_clean_circuit(
        self, clean: stim.Circuit, *, params: CircuitLevelParams,
        p_corr: float = 0.0,
    ) -> stim.Circuit:
        """Inject SD6-like noise + CORRELATED_ERROR after 2Q gates."""
        text = str(clean)
        # Step 1: standard circuit-level noise (independent)
        noisy_text = inject_circuit_level_noise(text, params=params)
        if p_corr <= 0:
            return stim.Circuit(noisy_text)
        # Step 2: add CORRELATED_ERROR after every CX/CZ pair
        corr_text = _inject_correlated_errors(noisy_text, p_corr)
        return stim.Circuit(corr_text)


def _inject_correlated_errors(text: str, p_corr: float) -> str:
    """Insert CORRELATED_ERROR(p) after DEPOLARIZE2 lines following CX/CZ.

    Each CX q0 q1 already has DEPOLARIZE2(p2) q0 q1 injected.
    We add CORRELATED_ERROR(p_corr) X q0 X q1 after the DEPOLARIZE2.
    This produces correlated XX errors that create hyperedges in the DEM.
    """
    lines = text.splitlines()
    out: List[str] = []
    p_str = _fmt_p(p_corr)

    for i, line in enumerate(lines):
        out.append(line)
        stripped = line.strip()
        # Look for DEPOLARIZE2 lines (they follow CX/CZ in injected circuits)
        if stripped.startswith("DEPOLARIZE2("):
            # Extract qubit targets
            qs = _extract_qubits_from_line(stripped)
            if len(qs) >= 2:
                # Add correlated XX error on the qubit pair
                # CORRELATED_ERROR format: probability then X/Y/Z targets
                for j in range(0, len(qs) - 1, 2):
                    q0, q1 = qs[j], qs[j + 1]
                    out.append(f"CORRELATED_ERROR({p_str}) X{q0} X{q1}")

    return "\n".join(out) + "\n"


# ---------------------------
# Injection Pass (text-level, supports REPEAT + TICK)
# ---------------------------

_ONE_Q_UNITARY = {
    "H", "S", "SQRT_X", "SQRT_Y", "X", "Y", "Z",
}
_TWO_Q = {"CX", "CZ", "CY", "SWAP"}

_RESET = {"R", "RX", "RY", "RZ"}
_MEASURE_Z = {"M", "MZ"}
_MEASURE_X = {"MX"}
_MEASURE_Y = {"MY"}

# Lines that should pass through without noise logic
_META_OPS = {
    "TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS",
}


def _inject_for_gate(line: str, params: CircuitLevelParams) -> List[str]:
    """
    Returns list of output lines (may include injected noise lines around the gate).
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return [line]

    head = stripped.split()[0]
    gate = head.split("(")[0]  # strip possible params

    # Always pass meta ops through
    if gate in _META_OPS:
        return [line]

    qs = _extract_qubits_from_line(line)

    # Measurement noise: insert *before* measurement
    if gate in _MEASURE_Z:
        if qs and params.p_meas > 0:
            return [f"X_ERROR({_fmt_p(params.p_meas)}) " + " ".join(map(str, qs)), line]
        return [line]

    if gate in _MEASURE_X:
        # Z before X-measure flips outcome
        if qs and params.p_meas > 0:
            return [f"Z_ERROR({_fmt_p(params.p_meas)}) " + " ".join(map(str, qs)), line]
        return [line]

    if gate in _MEASURE_Y:
        # For Y measurement, either X or Z flips; keep simple: X_ERROR
        if qs and params.p_meas > 0:
            return [f"X_ERROR({_fmt_p(params.p_meas)}) " + " ".join(map(str, qs)), line]
        return [line]

    # Reset noise: insert *after* reset (wrong state)
    if gate in _RESET:
        out = [line]
        if qs and params.p_reset > 0:
            out.append(f"X_ERROR({_fmt_p(params.p_reset)}) " + " ".join(map(str, qs)))
        return out

    # 2Q depolarization after gate
    if gate in _TWO_Q:
        out = [line]
        if qs and params.p2 > 0:
            out.append(f"DEPOLARIZE2({_fmt_p(params.p2)}) " + " ".join(map(str, qs)))
        return out

    # 1Q unitary depolarization after gate
    if gate in _ONE_Q_UNITARY:
        out = [line]
        if qs and params.p1 > 0:
            out.append(f"DEPOLARIZE1({_fmt_p(params.p1)}) " + " ".join(map(str, qs)))
        return out

    # Unknown gate: keep as-is (safe)
    return [line]


def inject_circuit_level_noise(text: str, params: CircuitLevelParams) -> str:
    """
    Injects circuit-level noise:
    - After 1Q unitary: DEPOLARIZE1(p1)
    - After 2Q gate: DEPOLARIZE2(p2)
    - Before measurement: basis-appropriate error (X/Z)
    - After reset: X_ERROR(p_reset)
    - At each TICK boundary: DEPOLARIZE1(p_idle) on idle qubits (those not touched since last TICK)
    Supports nested REPEAT blocks.
    """
    all_qubits = _discover_all_qubits(text)

    def process_block(lines: List[str]) -> List[str]:
        out: List[str] = []
        i = 0

        # For idling within this block
        active_since_tick: Set[int] = set()

        def flush_idle_before_tick():
            nonlocal active_since_tick
            if params.p_idle <= 0 or not all_qubits:
                active_since_tick = set()
                return
            idle = [q for q in all_qubits if q not in active_since_tick]
            if idle:
                out.append(f"DEPOLARIZE1({_fmt_p(params.p_idle)}) " + " ".join(map(str, idle)))
            active_since_tick = set()

        while i < len(lines):
            line = lines[i]
            m = _RE_REPEAT.match(line)
            if m:
                # parse nested block
                reps = int(m.group(1))
                # find matching }
                depth = 1
                j = i + 1
                body: List[str] = []
                while j < len(lines):
                    if _RE_REPEAT.match(lines[j]):
                        depth += 1
                    elif _RE_END.match(lines[j]):
                        depth -= 1
                        if depth == 0:
                            break
                    body.append(lines[j])
                    j += 1
                if depth != 0:
                    raise ValueError("Unbalanced REPEAT block braces in circuit text.")

                # Recursively process body
                injected_body = process_block(body)

                # Emit REPEAT block (preserve formatting)
                out.append(line.rstrip())
                out.extend([ln.rstrip() for ln in injected_body])
                out.append("}")

                # Repeat blocks count as activity? We'll be conservative:
                # treat as activity on all qubits (prevents idle injection incorrectly)
                # You can refine later by analyzing body.
                active_since_tick.update(all_qubits)

                i = j + 1
                continue

            # handle TICK: before tick, inject idle noise for the previous segment
            stripped = line.strip()
            if stripped.startswith("TICK"):
                flush_idle_before_tick()
                out.append("TICK")
                i += 1
                continue

            # inject around gate line
            produced = _inject_for_gate(line.rstrip(), params)

            # update active set based on original line targets (not the injected noise)
            qs = _extract_qubits_from_line(line)
            active_since_tick.update(qs)

            out.extend([ln.rstrip() for ln in produced])
            i += 1

        # At end of block, also flush idle (acts like implicit tick end)
        flush_idle_before_tick()
        return out

    lines = text.splitlines()
    processed = process_block(lines)
    return "\n".join(processed) + "\n"


def make_model(name: str, **kwargs) -> CircuitNoiseModel:
    name = name.strip().lower()
    if name == "sd6_like":
        return Sd6LikeNoiseModel()
    if name == "si1000_like":
        return Si1000LikeNoiseModel()
    if name == "correlated_crosstalk_like":
        corr_strength = float(kwargs.get("corr_strength", 0.5))
        return CorrelatedCrosstalkModel(corr_strength=corr_strength)
    raise ValueError(f"Unknown circuit-level noise model: {name}")
