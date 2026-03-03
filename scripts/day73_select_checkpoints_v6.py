#!/usr/bin/env python3
"""
Day 73 -- Selector v6 CLI (smoothed topology floors, retroactive).
"""
from __future__ import annotations
import argparse, os, sys, time

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)


def compute_per_seed_floors(seeds, distance, p, basis, corr_strength,
                            n_probe, n_shuffles=200) -> dict:
    """Regenerate probe (Y, K) per seed and compute empirical null p05."""
    from qec_noise_factory.ml.ops.checkpoint_selection import compute_slice_clean_null_p05

    floors = {}
    for seed in seeds:
        print(f"  [Floor] Seed {seed}...", end=" ", flush=True)
        t0 = time.time()
        from qec_noise_factory.ml.bench.regime_lock import RegimeLock, generate_locked_data
        from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count

        lock = RegimeLock(distance=distance, target_p=p, basis=basis,
                          require_generated=True, n_samples=n_probe,
                          corr_strength=corr_strength, seed=seed)
        X_probe, Y_probe = generate_locked_data(lock)
        K_probe = compute_syndrome_count(X_probe)

        p05 = compute_slice_clean_null_p05(Y_probe, K_probe,
                                           n_shuffles=n_shuffles, seed=0)
        floors[str(seed)] = p05
        print(f"p05={p05:.4f} ({time.time()-t0:.1f}s)", flush=True)
    return floors


def main(argv=None):
    parser = argparse.ArgumentParser(description="Selector v6 CLI (Day 73)")
    parser.add_argument("--artifact_dir", type=str,
                        default="ml_artifacts/day70_exactk_d7_generalization")
    parser.add_argument("--out_dir", type=str,
                        default="ml_artifacts/day73_ckpt_selection_v6")
    parser.add_argument("--tau_clean", type=float, default=0.025)
    parser.add_argument("--tau_clean_hi", type=float, default=0.035)
    parser.add_argument("--tg_floor", type=float, default=-0.015)
    parser.add_argument("--roll_window", type=int, default=3)
    parser.add_argument("--active_min", type=int, default=6)
    parser.add_argument("--n_shuffles", type=int, default=200)
    parser.add_argument("--distance", type=int, default=7)
    parser.add_argument("--p", type=float, default=0.04)
    parser.add_argument("--basis", type=str, default="X")
    parser.add_argument("--corr_strength", type=float, default=0.5)
    parser.add_argument("--n_probe", type=int, default=4096)
    parser.add_argument("--skip_floors", action="store_true",
                        help="Skip per-seed floor computation, use default 0.500")
    parser.add_argument("--drop_slice_floor", action="store_true",
                        help="Fallback: remove SliceClean from survival filter")
    args = parser.parse_args(argv)

    from qec_noise_factory.ml.ops.checkpoint_selection import (
        run_selection, compute_slice_clean_baseline
    )

    baseline = compute_slice_clean_baseline()
    print(f"[Sanity] SliceClean random baseline = {baseline:.4f} (expect ~0.50)")

    seeds = [47000, 49000, 49200, 50000, 51000, 52000, 53000, 54000, 55000, 56000]

    slice_floors = None
    if not args.skip_floors:
        print(f"\n[v6] Computing per-seed empirical floors (n={args.n_shuffles})...")
        slice_floors = compute_per_seed_floors(
            seeds, args.distance, args.p, args.basis,
            args.corr_strength, args.n_probe, args.n_shuffles)
        print(f"  Floors: { {k: round(v, 4) for k, v in slice_floors.items()} }")

    print(f"\n[Selector v6]")
    print(f"  artifact_dir:      {args.artifact_dir}")
    print(f"  out_dir:           {args.out_dir}")
    print(f"  tau_clean:         {args.tau_clean}")
    print(f"  tau_clean_hi:      {args.tau_clean_hi}")
    print(f"  tg_floor:          {args.tg_floor}")
    print(f"  roll_window:       {args.roll_window}")
    print(f"  drop_slice_floor:  {args.drop_slice_floor}")
    if slice_floors:
        print(f"  slice_floors:      per-seed empirical p05")
    else:
        print(f"  slice_floor:       0.500 (default)")
    print()

    report = run_selection(
        artifact_dir=args.artifact_dir, out_dir=args.out_dir,
        tau_clean=args.tau_clean, tau_clean_hi=args.tau_clean_hi,
        slice_floors=slice_floors, default_slice_floor=0.500,
        tg_floor=args.tg_floor,
        active_epoch_min=args.active_min, roll_window=args.roll_window,
        drop_slice_floor=args.drop_slice_floor,
    )

    print("=" * 100)
    print("AGGREGATE SUMMARY")
    print("=" * 100)
    print(f"{'Arm':<28} {'Med G1(old)':>12} {'Med G1(sel)':>12} "
          f"{'D vs Ctrl':>10} {'Clean':>6} {'Leaky':>6} {'Fall':>6} "
          f"{'TF%':>5} {'spike':>8}")
    print("-" * 100)
    for a in report["aggregate"]:
        print(f"{a['arm']:<28} {a['median_G1_old']:>12.4f} "
              f"{a['median_G1_selected']:>12.4f} "
              f"{a['delta_vs_control_pct']:>+9.1f}% "
              f"{a['count_clean']:>6} {a['count_leaky']:>6} "
              f"{a['count_fallback']:>6} "
              f"{a['topo_fail_pct']:>4.0f}% "
              f"{a['median_g1_spike_delta']:>8.4f}")

    print(f"\nDual-cap violations: {report['dual_cap_violations']}")

    # Per-seed for main arm
    main_arms = [a for a in report["aggregate"]
                 if "Prod" in a["arm"] or "DNH" in a["arm"]]
    arm_name = main_arms[0]["arm"] if main_arms else report["aggregate"][-1]["arm"]

    print(f"\n{'='*100}\nPER-SEED ({arm_name})\n{'='*100}")
    print(f"{'Seed':>6} {'Ep':>4} {'G1':>8} {'g1roll':>8} {'spike':>8} "
          f"{'SC':>8} {'SC_r':>8} {'TG':>8} {'TG_r':>8} "
          f"{'floor':>7} {'nSurv':>6} {'Mode':<20}")
    print("-" * 100)
    for s in sorted(report["per_seed_selections"], key=lambda x: x["seed"]):
        if s["arm"] == arm_name:
            tg = f"{s['topo_TG']:.4f}" if s["topo_TG"] is not None else "N/A"
            print(f"{s['seed']:>6} {s['epoch']:>4} {s['G1_aligned']:>8.4f} "
                  f"{s['g1_roll']:>8.4f} {s['g1_spike_delta']:>+8.4f} "
                  f"{s['slice_clean']:>8.4f} {s['slice_clean_roll']:>8.4f} "
                  f"{tg:>8} {s['tg_roll']:>8.4f} "
                  f"{s['slice_floor_used']:>7.4f} "
                  f"{s['n_surviving']:>6} {s['selection_mode']:<20}")

    print(f"\nDone! -> {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
