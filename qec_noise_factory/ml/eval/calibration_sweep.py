"""
Calibration Sweep Runner + Pareto Selection — Day 25

Evaluates a grid of configs on the same generalization suite A/B/C
and selects Pareto-optimal configurations.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qec_noise_factory.ml.eval.generalization_suite import (
    ExperimentConfig, ExperimentReport, run_experiment,
)


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

SWEEP_GRID = {
    "featureset": ["v1_full", "v1_nop"],
    "readout": ["mean_max", "attn"],
    "calibrate_metric": ["bal_acc", "f1", "bal_acc_minus_fpr"],
    "bal_acc_minus_fpr_lambdas": [0.0, 0.05, 0.10, 0.15, 0.25, 0.35],
}


def generate_sweep_configs(
    grid: Optional[Dict] = None,
    epochs: int = 3,
    hidden_dim: int = 64,
) -> List[Dict[str, Any]]:
    """
    Generate unique config dicts for the sweep grid.

    Returns list of dicts with keys: featureset, readout, calibrate_metric, calibrate_lambda.
    """
    g = grid or SWEEP_GRID
    configs = []
    seen = set()

    for fs, ro, cm in product(
        g["featureset"], g["readout"], g["calibrate_metric"],
    ):
        if cm == "bal_acc_minus_fpr":
            lambdas = g.get("bal_acc_minus_fpr_lambdas", [0.25])
        else:
            lambdas = [0.0]

        for lam in lambdas:
            key = (fs, ro, cm, lam)
            if key in seen:
                continue
            seen.add(key)
            configs.append({
                "featureset": fs,
                "readout": ro,
                "calibrate_metric": cm,
                "calibrate_lambda": lam,
                "epochs": epochs,
                "hidden_dim": hidden_dim,
            })

    return configs


def _make_experiment_configs(
    sweep_cfg: Dict[str, Any],
    variant_id: str,
) -> List[ExperimentConfig]:
    """Create A/B/C experiment configs from a sweep config dict."""
    common = dict(
        epochs=sweep_cfg["epochs"],
        hidden_dim=sweep_cfg["hidden_dim"],
        loss_pos_weight=0,
        calibrate_threshold=True,
        pos_weight_max=8.0,
        gnn_version="v1",
        gnn_feature_version="v1",
        gnn_readout=sweep_cfg["readout"],
        featureset=sweep_cfg["featureset"],
        calibrate_metric=sweep_cfg["calibrate_metric"],
        calibrate_lambda=sweep_cfg["calibrate_lambda"],
    )
    return [
        ExperimentConfig(
            exp_id=f"{variant_id}-A", name=f"Cross-model ({variant_id})",
            split_policy="cross_model", train_ratio=0.5, **common,
        ),
        ExperimentConfig(
            exp_id=f"{variant_id}-B", name=f"Within-model ({variant_id})",
            split_policy="within_model", **common,
        ),
        ExperimentConfig(
            exp_id=f"{variant_id}-C", name=f"OOD p-range ({variant_id})",
            split_policy="ood_p_range", ood_test_p_lo=0.005, ood_test_p_hi=1.0,
            **common,
        ),
    ]


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    merged_dataset,
    arts_dir: Path,
    grid: Optional[Dict] = None,
    epochs: int = 3,
    hidden_dim: int = 64,
    quick: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run calibration sweep over grid of configs.

    Args:
        quick: If True, run only 4 configs (baseline + top candidates)
    """
    sweep_cfgs = generate_sweep_configs(grid, epochs, hidden_dim)

    if quick:
        # Quick mode: baseline + 3 promising configs
        quick_cfgs = [
            c for c in sweep_cfgs
            if (c["featureset"] == "v1_full" and c["readout"] == "mean_max"
                and c["calibrate_metric"] == "bal_acc")
        ][:1]
        # Add best OOD candidate
        quick_cfgs += [
            c for c in sweep_cfgs
            if (c["featureset"] == "v1_nop" and c["readout"] == "attn"
                and c["calibrate_metric"] == "bal_acc_minus_fpr"
                and c["calibrate_lambda"] == 0.05)
        ][:1]
        # Add nop + mean_max + f1
        quick_cfgs += [
            c for c in sweep_cfgs
            if (c["featureset"] == "v1_nop" and c["readout"] == "mean_max"
                and c["calibrate_metric"] == "f1")
        ][:1]
        sweep_cfgs = quick_cfgs

    all_results = []
    arts_dir.mkdir(parents=True, exist_ok=True)

    for i, scfg in enumerate(sweep_cfgs):
        variant_id = f"S{i:02d}"
        exp_configs = _make_experiment_configs(scfg, variant_id)
        variant_results = dict(scfg)
        variant_results["variant_id"] = variant_id
        variant_results["experiments"] = {}

        for ecfg in exp_configs:
            t0 = time.time()
            report = run_experiment(ecfg, merged_dataset, arts_dir / variant_id / ecfg.exp_id)
            elapsed = time.time() - t0
            rd = report.to_dict()
            rd["_runtime"] = elapsed
            variant_results["experiments"][ecfg.exp_id] = rd

        all_results.append(variant_results)

    return all_results


# ---------------------------------------------------------------------------
# Pareto selection
# ---------------------------------------------------------------------------

def _gm(report_dict: Dict, key: str) -> float:
    """Get GNN metric from report dict."""
    return (report_dict.get("gnn_metrics") or {}).get(key, 0.0)


def _is_collapse(report_dict: Dict) -> bool:
    """Check if experiment reports a collapse."""
    ppr = report_dict.get("gnn_pred_positive_rate", 0.0)
    return ppr < 0.005 or ppr > 0.95


def select_pareto(
    sweep_results: List[Dict[str, Any]],
    baseline_variant_id: str = "S00",
) -> Dict[str, Any]:
    """
    Pareto selection from sweep results.

    Returns dict with:
      best_cross: config that minimizes FPR on A, tiebreak precision then F1
      best_ood: non-collapse config that maximizes BalAcc on C, tiebreak F1
      best_overall: Pareto optimal over (Cross FPR low, OOD BalAcc high, non-collapse)
      baseline: baseline metrics for delta computation
    """
    # Extract per-variant summaries
    summaries = []
    baseline_summary = None

    for v in sweep_results:
        vid = v["variant_id"]
        exps = v.get("experiments", {})

        # Find A, B, C experiments
        exp_a = next((e for k, e in exps.items() if k.endswith("-A")), None)
        exp_b = next((e for k, e in exps.items() if k.endswith("-B")), None)
        exp_c = next((e for k, e in exps.items() if k.endswith("-C")), None)

        if not all([exp_a, exp_b, exp_c]):
            continue

        summary = {
            "variant_id": vid,
            "featureset": v["featureset"],
            "readout": v["readout"],
            "calibrate_metric": v["calibrate_metric"],
            "calibrate_lambda": v["calibrate_lambda"],
            "cross_fpr": _gm(exp_a, "macro_fpr"),
            "cross_prec": _gm(exp_a, "macro_precision"),
            "cross_f1": _gm(exp_a, "macro_f1"),
            "within_f1": _gm(exp_b, "macro_f1"),
            "ood_bal_acc": _gm(exp_c, "macro_balanced_accuracy"),
            "ood_f1": _gm(exp_c, "macro_f1"),
            "ood_fpr": _gm(exp_c, "macro_fpr"),
            "ood_collapse": _is_collapse(exp_c),
            "cross_collapse": _is_collapse(exp_a),
            "ood_fallback": exp_c.get("gnn_fallback_applied", False),
            "ood_pred_pos_rate": exp_c.get("gnn_pred_positive_rate", 0.0),
        }
        summaries.append(summary)
        if vid == baseline_variant_id:
            baseline_summary = summary

    if not summaries:
        return {"error": "No valid results"}

    if baseline_summary is None:
        baseline_summary = summaries[0]

    # Best Cross: minimize FPR, tiebreak maximize precision, then F1
    best_cross = min(
        [s for s in summaries if not s["cross_collapse"]],
        key=lambda s: (s["cross_fpr"], -s["cross_prec"], -s["cross_f1"]),
        default=None,
    )

    # Best OOD: non-collapse, maximize BalAcc, tiebreak F1, penalize high FPR
    non_collapse_ood = [s for s in summaries if not s["ood_collapse"]]
    best_ood = max(
        non_collapse_ood,
        key=lambda s: (s["ood_bal_acc"], s["ood_f1"], -s["ood_fpr"]),
        default=None,
    ) if non_collapse_ood else None

    # Best Overall: Pareto over (cross FPR low, OOD BalAcc high, non-collapse)
    # Score = -cross_fpr + ood_bal_acc (both in [0,1])
    non_collapse_both = [s for s in summaries if not s["ood_collapse"] and not s["cross_collapse"]]
    best_overall = max(
        non_collapse_both,
        key=lambda s: (-s["cross_fpr"] + s["ood_bal_acc"] + 0.5 * s["within_f1"]),
        default=None,
    ) if non_collapse_both else None

    def _with_deltas(selected, baseline):
        if not selected or not baseline:
            return selected
        selected["delta_cross_fpr"] = selected["cross_fpr"] - baseline["cross_fpr"]
        selected["delta_ood_bal_acc"] = selected["ood_bal_acc"] - baseline["ood_bal_acc"]
        selected["delta_within_f1"] = selected["within_f1"] - baseline["within_f1"]
        return selected

    return {
        "best_cross": _with_deltas(best_cross, baseline_summary),
        "best_ood": _with_deltas(best_ood, baseline_summary),
        "best_overall": _with_deltas(best_overall, baseline_summary),
        "baseline": baseline_summary,
        "total_variants": len(summaries),
        "collapse_count": sum(1 for s in summaries if s["ood_collapse"]),
        "non_collapse_count": sum(1 for s in summaries if not s["ood_collapse"]),
    }
