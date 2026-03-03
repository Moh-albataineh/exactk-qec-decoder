#!/usr/bin/env python3
"""
Day 74 -- Post-training Selector v6 from JSONL + Checkpoints.

Reads JSONL logs + checkpoint files, applies Selector v6, copies
chosen checkpoint → best_model_{seed}.pt, writes selection receipts.
"""
from __future__ import annotations
import argparse, json, os, sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Post-training selector v6 from JSONL + checkpoints")
    parser.add_argument("--artifact_dir", required=True,
                        help="Dir with metrics_*.jsonl and ckpts/")
    parser.add_argument("--out_dir", default=None,
                        help="Output dir (default: same as artifact_dir)")
    parser.add_argument("--tau_clean", type=float, default=0.025)
    parser.add_argument("--tau_clean_hi", type=float, default=0.035)
    parser.add_argument("--tg_floor", type=float, default=-0.015)
    parser.add_argument("--roll_window", type=int, default=3)
    parser.add_argument("--active_min", type=int, default=6)
    parser.add_argument("--arm", type=str, default=None,
                        help="Arm to select for (default: all non-Control)")
    parser.add_argument("--no_cleanup", action="store_true",
                        help="Do not delete unselected checkpoints")
    args = parser.parse_args(argv)

    from pathlib import Path
    from qec_noise_factory.ml.ops.checkpoint_selection import (
        load_day_artifacts_auto, select_epoch_for_seed, rolling_median,
        copy_best_checkpoint, cleanup_unselected_checkpoints,
        write_selection_receipt, _parse_key,
    )

    artifact_dir = Path(args.artifact_dir)
    out_dir = Path(args.out_dir) if args.out_dir else artifact_dir
    ckpt_dir = artifact_dir / "ckpts"

    # Load
    merged = load_day_artifacts_auto(artifact_dir)
    print(f"Loaded {len(merged)} arm/seed combos from {artifact_dir}")

    for key in sorted(merged.keys()):
        arm, seed = _parse_key(key)
        if args.arm and arm != args.arm:
            continue

        result = select_epoch_for_seed(
            merged[key],
            tau_clean=args.tau_clean, tau_clean_hi=args.tau_clean_hi,
            slice_floor=0.0, tg_floor=args.tg_floor,
            active_epoch_min=args.active_min, roll_window=args.roll_window,
            drop_slice_floor=True,
        )

        print(f"  [{arm}] seed={seed}: ep={result['epoch']} "
              f"G1={result['G1_aligned']:.4f} g1roll={result['g1_roll']:.4f} "
              f"mode={result['selection_mode']}")

        # Write receipt
        receipt = {
            "selector_version": "v6_drop_slice_floor",
            "seed": seed,
            "arm": arm,
            "chosen_epoch": result["epoch"],
            "selection_mode": result["selection_mode"],
            "g1_aligned": result["G1_aligned"],
            "g1_roll": result["g1_roll"],
            "g1_spike_delta": result["g1_spike_delta"],
            "tg_roll": result["tg_roll"],
            "slice_clean": result["slice_clean"],
            "thresholds": {
                "tau_clean": args.tau_clean,
                "tau_clean_hi": args.tau_clean_hi,
                "tg_roll_floor": args.tg_floor,
            },
            "n_surviving": result["n_surviving"],
            "n_clean": result["n_clean"],
        }
        write_selection_receipt(receipt, out_dir, seed)

        # Copy checkpoint if available
        ckpt_path = ckpt_dir / f"ckpt_{seed}_ep{result['epoch']}.pt"
        if ckpt_path.exists():
            copy_best_checkpoint(ckpt_dir, out_dir, seed, result["epoch"])
            print(f"    → best_model_{seed}.pt")
            if not args.no_cleanup:
                deleted = cleanup_unselected_checkpoints(
                    ckpt_dir, seed, result["epoch"])
                if deleted:
                    print(f"    Cleaned {len(deleted)} unselected ckpts")
        else:
            print(f"    (no checkpoint at {ckpt_path})")

    print(f"\nDone! Receipts in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
