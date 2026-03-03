"""
Day 75.3 — Regenerate selection receipts from all_results.json (zero-compute).
Overwrites receipts in Day 75 artifact dir with properly computed tg_roll values.
"""
import json, os, sys
from pathlib import Path

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)


def _normalize_fields(rec):
    r = dict(rec)
    if "topo_slice_clean" in r and "slice_clean" not in r:
        r["slice_clean"] = r["topo_slice_clean"]
    if "topo_mean_drop" in r and "mean_drop" not in r:
        r["mean_drop"] = r["topo_mean_drop"]
    r.setdefault("slice_clean", None)
    r.setdefault("mean_drop", None)
    r.setdefault("topo_TG", None)
    return r


def regenerate_receipts(artifact_dir: str):
    from qec_noise_factory.ml.ops.checkpoint_selection import (
        select_epoch_for_seed, write_selection_receipt)

    root = Path(artifact_dir)
    ar_path = root / "all_results.json"
    with open(ar_path) as f:
        all_results = json.load(f)

    arms = ["Control", "ExactK_Tuned_Prod"]
    seeds = sorted(set(
        int(k.split("_")[-1]) for k in all_results
        if any(k.startswith(a + "_") for a in arms)))

    print(f"Regenerating receipts for {len(arms)} arms x {len(seeds)} seeds")

    for arm in arms:
        for seed in seeds:
            key = f"{arm}_{seed}"
            recs = all_results.get(key, [])
            if not recs:
                print(f"  WARN: no records for {key}")
                continue

            normed = [_normalize_fields(r) for r in recs]
            sel = select_epoch_for_seed(
                normed,
                tau_clean=0.025, tau_clean_hi=0.035,
                slice_floor=0.0, tg_floor=-0.015,
                active_epoch_min=6, roll_window=3,
                drop_slice_floor=True)

            # Build canonical receipt with ALL required fields
            receipt = {
                "selector_version": "v6_drop_slice_floor",
                "seed": seed,
                "arm": arm,
                "chosen_epoch": sel["epoch"],
                "selection_mode": sel["selection_mode"],
                "g1_aligned": sel["G1_aligned"],
                "g1_roll": sel["g1_roll"],
                "g1_spike_delta": sel["g1_spike_delta"],
                "tg_roll": sel["tg_roll"],
                "slice_clean": sel.get("slice_clean"),
                "n_surviving": sel["n_surviving"],
                "n_clean": sel["n_clean"],
                "thresholds": {
                    "tau_clean": 0.025,
                    "tau_clean_hi": 0.035,
                    "tg_roll_floor": -0.015,
                },
                # Canonical field names for KPI extraction
                "tg_roll_selected": sel["tg_roll"],
                "g1roll_selected": sel["g1_roll"],
                "g1_inst_selected": sel["G1_aligned"],
                "spike_delta": sel["g1_spike_delta"],
                "selector_pool": sel["selection_mode"],
            }

            out_path = root / f"selection_receipt_{arm}_{seed}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(receipt, f, indent=2, ensure_ascii=False)

            print(f"  {arm}_{seed}: ep={sel['epoch']} "
                  f"tg_roll={sel['tg_roll']:.4f} "
                  f"g1roll={sel['g1_roll']:.4f} "
                  f"mode={sel['selection_mode']}")

    print(f"\nDone. Receipts written to {root}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default="ml_artifacts/day75_holdout_d7_v1")
    args = parser.parse_args()
    regenerate_receipts(args.artifact_dir)
