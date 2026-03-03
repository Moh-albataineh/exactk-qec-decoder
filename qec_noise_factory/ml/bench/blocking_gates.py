"""
Blocking Gates — Day 37.1 / 37.2

All gates must PASS on decision runs. Failed gates write dump bundles
and return FAIL reason codes.

Gates:
  1. TopologyGain >= +0.02
  2. scrambler_delta >= 0.10
  3. |residual_k_corr_clean| <= 0.10
  4. |residual_k_corr_scrambled| <= 0.10
  5. iso_density_auroc >= 0.55 (>=3 qualified bins)
  6. no_collapse (TPR > 0)
  7. orientation_flipped == False  (Day 37.2)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


def run_blocking_gates(
    metrics: Dict[str, Any],
    artifact_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run all blocking gates on experiment metrics.

    Args:
        metrics: dict with topology_gain, scrambler_delta, residual_k_corr_clean,
                 residual_k_corr_scrambled, iso_density (macro_auroc, n_qualified),
                 tpr.
        artifact_dir: directory to write dump bundles on failure.

    Returns:
        dict with gate_results (list), all_pass (bool), summary.
    """
    from qec_noise_factory.ml.bench import reason_codes as RC

    gates = []

    # Gate 1: TopologyGain
    tg = metrics.get("topology_gain")
    if tg is not None:
        passed = tg >= 0.02
        gates.append({
            "gate": "topology_gain",
            "value": tg,
            "threshold": 0.02,
            "status": "PASS" if passed else "FAIL",
            "reason_code": None if passed else RC.ERR_TOPOLOGYGAIN_FAIL,
        })
    else:
        gates.append({"gate": "topology_gain", "status": "SKIP", "value": None, "reason_code": None})

    # Gate 2: scrambler_delta
    sd = metrics.get("scrambler_delta")
    if sd is not None:
        passed = sd >= 0.10
        gates.append({
            "gate": "scrambler_delta",
            "value": sd,
            "threshold": 0.10,
            "status": "PASS" if passed else "FAIL",
            "reason_code": None if passed else "ERR_SCRAMBLER_SHORTCUT",
        })
    else:
        gates.append({"gate": "scrambler_delta", "status": "SKIP", "value": None, "reason_code": None})

    # Gate 3: residual_k_corr_clean
    rkc = metrics.get("residual_k_corr_clean", metrics.get("residual_k_corr"))
    if rkc is not None:
        passed = abs(rkc) <= 0.10
        gates.append({
            "gate": "residual_k_corr_clean",
            "value": rkc,
            "threshold": 0.10,
            "status": "PASS" if passed else "FAIL",
            "reason_code": None if passed else RC.ERR_RESIDUAL_K_CORR_HIGH,
        })
    else:
        gates.append({"gate": "residual_k_corr_clean", "status": "SKIP", "value": None, "reason_code": None})

    # Gate 4: residual_k_corr_scrambled
    rks = metrics.get("residual_k_corr_scrambled")
    if rks is not None:
        passed = abs(rks) <= 0.10
        gates.append({
            "gate": "residual_k_corr_scrambled",
            "value": rks,
            "threshold": 0.10,
            "status": "PASS" if passed else "FAIL",
            "reason_code": None if passed else RC.ERR_RESIDUAL_K_CORR_HIGH,
        })
    else:
        gates.append({"gate": "residual_k_corr_scrambled", "status": "SKIP", "value": None, "reason_code": None})

    # Gate 5: iso_density_auroc
    iso = metrics.get("iso_density", {})
    macro = iso.get("macro_auroc")
    n_q = iso.get("n_qualified", 0)
    if macro is not None and n_q >= 3:
        passed = macro >= 0.55
        gates.append({
            "gate": "iso_density_auroc",
            "value": macro,
            "n_qualified": n_q,
            "threshold": 0.55,
            "status": "PASS" if passed else "FAIL",
            "reason_code": None if passed else RC.ERR_ISODENSITY_AUROC_FAIL,
        })
    else:
        gates.append({
            "gate": "iso_density_auroc",
            "status": f"SKIP(buckets={n_q})",
            "value": macro,
            "reason_code": None,
        })

    # Gate 6: no_collapse
    tpr = metrics.get("tpr", 0)
    gates.append({
        "gate": "no_collapse",
        "value": tpr,
        "status": "PASS" if tpr > 0 else "FAIL",
        "reason_code": None if tpr > 0 else "ERR_COLLAPSE",
    })

    # Gate 7: orientation_flipped (Day 37.2)
    flipped = metrics.get("orientation_flipped", False)
    gates.append({
        "gate": "auroc_orientation",
        "value": flipped,
        "status": "PASS" if not flipped else "FAIL",
        "reason_code": None if not flipped else RC.ERR_AUROC_ORIENTATION_FLIPPED,
    })

    all_pass = all(g["status"] == "PASS" for g in gates if g["status"] not in ("SKIP",) and not g["status"].startswith("SKIP"))
    n_pass = sum(1 for g in gates if g["status"] == "PASS")
    n_fail = sum(1 for g in gates if g["status"] == "FAIL")

    result = {
        "gate_results": gates,
        "all_pass": all_pass,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_total": len(gates),
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
    }

    # Write dump bundle on failure
    if not all_pass and artifact_dir is not None:
        _write_fail_bundle(artifact_dir, result, metrics)

    return result


def _write_fail_bundle(
    artifact_dir: Path,
    gate_result: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    """Write failure dump bundle for postmortem analysis."""
    ts = gate_result["timestamp"]
    failed_gates = [g["gate"] for g in gate_result["gate_results"] if g["status"] == "FAIL"]
    reason = "_".join(failed_gates[:3])
    bundle_dir = artifact_dir / f"FAIL_{reason}_{ts}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / "gate_report.json").write_text(
        json.dumps(gate_result, indent=2, default=str), encoding="utf-8"
    )
    (bundle_dir / "metrics_dump.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
