from __future__ import annotations
import stim

def build_demo_repetition(rounds: int, p: float) -> stim.Circuit:
    """
    Minimal circuit with:
    - detectors (so detection_events are meaningful)
    - one observable (so logical flip rate is meaningful)
    """
    if rounds < 2:
        raise ValueError("demo_repetition requires rounds >= 2")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")

    c = stim.Circuit()

    # Loop over rounds
    for r in range(rounds):
        # 1. Reset and Error injection (simplistic)
        c.append("R", [0])
        if p > 0:
            c.append("X_ERROR", [0], p)
        
        # 2. Measurement
        c.append("M", [0])

        # 3. Detectors
        # Logic: Compare current measurement (rec[-1]) with previous (rec[-2])
        if r == 0:
            # First round: just detect the value itself (initializer)
            c.append("DETECTOR", [stim.target_rec(-1)], [0.0])
        else:
            # Subsequent rounds: Parity check (current vs previous)
            c.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)], [r, 0.0])

    # Define one observable (observable 0) as the last measurement
    # Syntax: append("OBSERVABLE_INCLUDE", targets, arg=index)
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0.0)
    
    return c