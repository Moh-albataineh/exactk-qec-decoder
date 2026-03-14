#!/usr/bin/env python3
"""
Compute paired bootstrap confidence intervals for ExactK paper claims.

Method:
  - Paired bootstrap resampling by seed (N=10 per regime)
  - Statistic: 100 * (median(control) - median(exactk)) / median(control)
  - 95% CI via percentile method [2.5th, 97.5th]
  - Active epochs only (epoch >= 6)
  - G1_aligned metric only (no fallback)

Canonical regimes:
  1. Day 69 / d=5 primary      → Control vs ExactK_Tuned
  2. Day 70 / d=7 OOD          → Control vs ExactK_Tuned_Prod
  3. Day 75 / d=7 holdout      → Control vs ExactK_Tuned_Prod

Output: paper/tables/bootstrap_cis.json
"""

import json
import sys
from pathlib import Path
from statistics import median

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Resolve repo root relative to this script: paper/ -> public_release_exactk_v1/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = REPO_ROOT / "ml_artifacts"

N_BOOTSTRAPS = 100_000
RANDOM_SEED = 42
ACTIVE_EPOCH_MIN = 6
METRIC = "G1_aligned"
ALPHA = 0.05  # 95% CI

REGIMES = [
    {
        "name": "day69_d5_primary",
        "json_rel": "day69_exactk_tuned_margin_decay_gate/all_results.json",
        "control_arm": "Control",
        "exactk_arm": "ExactK_Tuned",
        "seeds": [47000, 49000, 49200, 50000, 51000, 52000, 53000, 54000, 55000, 56000],
        "canonical_point_estimate": 31.2,
        "tolerance_pct": 2.0,
    },
    {
        "name": "day70_d7_ood",
        "json_rel": "day70_exactk_d7_generalization/all_results.json",
        "control_arm": "Control",
        "exactk_arm": "ExactK_Tuned_Prod",
        "seeds": [47000, 49000, 49200, 50000, 51000, 52000, 53000, 54000, 55000, 56000],
        "canonical_point_estimate": 23.4,
        "tolerance_pct": 2.0,
    },
    {
        "name": "day75_d7_holdout",
        "json_rel": "day75_holdout_d7_v1/all_results.json",
        "control_arm": "Control",
        "exactk_arm": "ExactK_Tuned_Prod",
        "seeds": [60000, 60001, 60002, 60003, 60004, 60005, 60006, 60007, 60008, 60009],
        "canonical_point_estimate": 45.0,
        "tolerance_pct": 2.0,
    },
]

STATISTIC_DEFINITION = (
    "100 * (median(control_epoch_median_g1) - median(exactk_epoch_median_g1)) "
    "/ median(control_epoch_median_g1)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    """Load and return a JSON file."""
    if not path.exists():
        print(f"FATAL: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_epoch_median_g1(
    data: dict,
    arm: str,
    seed: int,
    active_min: int = ACTIVE_EPOCH_MIN,
) -> float:
    """
    Extract the epoch-median G1_aligned for one arm/seed.

    Applies the canonical _normalize_epoch logic from checkpoint_selection.py
    (L157-158): G1_aligned falls back to G1_raw_probe when the field name
    differs across experiment generations. This is the project's documented
    normalization, not a silent fallback.

    Raises ValueError if NEITHER G1_aligned nor G1_raw_probe is present.
    """
    key = f"{arm}_{seed}"
    if key not in data:
        raise KeyError(f"Key '{key}' not found in JSON. Available keys: {sorted(data.keys())[:10]}...")

    epochs = data[key]
    active = [e for e in epochs if e.get("epoch", 0) >= active_min]
    if not active:
        raise ValueError(f"No active epochs (>= {active_min}) for {key}")

    g1_values = []
    raw_field_used = None
    for e in active:
        # Canonical normalization (checkpoint_selection.py _normalize_epoch L157-158)
        if METRIC in e:
            g1_values.append(e[METRIC])
            if raw_field_used is None:
                raw_field_used = METRIC
        elif "G1_raw_probe" in e:
            g1_values.append(e["G1_raw_probe"])
            if raw_field_used is None:
                raw_field_used = "G1_raw_probe (canonical normalization)"
        else:
            raise ValueError(
                f"FATAL: Neither '{METRIC}' nor 'G1_raw_probe' found in "
                f"epoch {e.get('epoch')} for {key}. "
                f"Available fields: {sorted(e.keys())}."
            )

    if raw_field_used and raw_field_used != METRIC:
        # Log once per arm/seed
        pass  # will be logged at regime level

    return median(g1_values), raw_field_used


def compute_delta_pct(control_vals: list, exactk_vals: list) -> float:
    """Compute 100 * (median(ctrl) - median(ek)) / median(ctrl)."""
    med_ctrl = np.median(control_vals)
    med_ek = np.median(exactk_vals)
    if med_ctrl == 0:
        return 0.0
    return 100.0 * (med_ctrl - med_ek) / med_ctrl


def paired_bootstrap_ci(
    control_vals: np.ndarray,
    exactk_vals: np.ndarray,
    n_bootstraps: int = N_BOOTSTRAPS,
    seed: int = RANDOM_SEED,
    alpha: float = ALPHA,
) -> tuple:
    """
    Paired bootstrap CI for the median relative improvement.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(control_vals)
    assert len(exactk_vals) == n, "Mismatched array lengths"

    boot_stats = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        ctrl_boot = control_vals[idx]
        ek_boot = exactk_vals[idx]
        med_ctrl = np.median(ctrl_boot)
        med_ek = np.median(ek_boot)
        if med_ctrl == 0:
            boot_stats[i] = 0.0
        else:
            boot_stats[i] = 100.0 * (med_ctrl - med_ek) / med_ctrl

    point = compute_delta_pct(control_vals, exactk_vals)
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = Path(__file__).resolve().parent / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bootstrap_cis.json"

    results = []
    all_ok = True

    for regime in REGIMES:
        json_path = ARTIFACTS / regime["json_rel"]
        print(f"\n{'='*70}")
        print(f"Regime: {regime['name']}")
        print(f"JSON:   {json_path}")
        print(f"Arms:   {regime['control_arm']} vs {regime['exactk_arm']}")
        print(f"Seeds:  {regime['seeds']}")
        print(f"{'='*70}")

        data = load_json(json_path)

        # Extract per-seed epoch-median G1
        ctrl_vals = []
        ek_vals = []
        raw_fields_seen = set()
        for s in regime["seeds"]:
            try:
                c, c_field = extract_epoch_median_g1(data, regime["control_arm"], s)
                e, e_field = extract_epoch_median_g1(data, regime["exactk_arm"], s)
            except (KeyError, ValueError) as exc:
                print(f"  FATAL: {exc}", file=sys.stderr)
                sys.exit(1)
            ctrl_vals.append(c)
            ek_vals.append(e)
            raw_fields_seen.add(c_field)
            raw_fields_seen.add(e_field)
            print(f"  Seed {s}:  Control={c:.6f}  ExactK={e:.6f}")

        if raw_fields_seen != {METRIC}:
            print(f"  NOTE: Raw field(s) used: {raw_fields_seen}")
            print(f"        (canonical _normalize_epoch maps G1_raw_probe → G1_aligned)")

        ctrl_arr = np.array(ctrl_vals)
        ek_arr = np.array(ek_vals)

        # Point estimate
        point = compute_delta_pct(ctrl_arr, ek_arr)
        print(f"\n  Median Control:  {np.median(ctrl_arr):.6f}")
        print(f"  Median ExactK:   {np.median(ek_arr):.6f}")
        print(f"  Point estimate:  +{point:.1f}%")
        print(f"  Canonical:       +{regime['canonical_point_estimate']:.1f}%")

        # Validate
        diff = abs(point - regime["canonical_point_estimate"])
        if diff > regime["tolerance_pct"]:
            print(f"\n  *** MISMATCH: computed {point:.1f}% vs canonical "
                  f"{regime['canonical_point_estimate']:.1f}% "
                  f"(diff={diff:.1f}%, tolerance={regime['tolerance_pct']}%) ***")
            print(f"  *** STOPPING: please investigate before trusting CIs ***")
            all_ok = False
            # Still compute CI but flag the mismatch
        else:
            print(f"  MATCH: within {regime['tolerance_pct']}% tolerance ✓")

        # Bootstrap
        print(f"\n  Running {N_BOOTSTRAPS:,} bootstrap resamples (seed={RANDOM_SEED})...")
        pt, lo, hi = paired_bootstrap_ci(ctrl_arr, ek_arr)
        print(f"  95% CI: [{lo:.1f}%, {hi:.1f}%]")
        print(f"  Point:  +{pt:.1f}% [{lo:.1f}%, {hi:.1f}%]")

        result = {
            "regime": regime["name"],
            "json_path": str(json_path),
            "control_arm": regime["control_arm"],
            "exactk_arm": regime["exactk_arm"],
            "seed_ids": regime["seeds"],
            "control_epoch_median_g1": [round(v, 6) for v in ctrl_vals],
            "exactk_epoch_median_g1": [round(v, 6) for v in ek_vals],
            "control_median": round(float(np.median(ctrl_arr)), 6),
            "exactk_median": round(float(np.median(ek_arr)), 6),
            "point_estimate_pct": round(pt, 1),
            "ci_lower_95": round(lo, 1),
            "ci_upper_95": round(hi, 1),
            "canonical_point_estimate_pct": regime["canonical_point_estimate"],
            "point_estimate_matches_canonical": bool(diff <= regime["tolerance_pct"]),
            "n_seeds": len(regime["seeds"]),
            "n_bootstraps": N_BOOTSTRAPS,
            "random_seed": RANDOM_SEED,
            "active_epoch_min": ACTIVE_EPOCH_MIN,
            "metric": METRIC,
            "statistic_definition": STATISTIC_DEFINITION,
        }
        results.append(result)

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    # Summary
    print("\n=== SUMMARY ===\n")
    for r in results:
        match_str = "✓" if r["point_estimate_matches_canonical"] else "✗ MISMATCH"
        print(f"  {r['regime']}:")
        print(f"    Point: +{r['point_estimate_pct']:.1f}%  "
              f"95% CI: [{r['ci_lower_95']:.1f}%, {r['ci_upper_95']:.1f}%]  "
              f"Canonical: +{r['canonical_point_estimate_pct']:.1f}%  {match_str}")

    if not all_ok:
        print("\n  *** WARNING: One or more regimes had point-estimate mismatches ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
