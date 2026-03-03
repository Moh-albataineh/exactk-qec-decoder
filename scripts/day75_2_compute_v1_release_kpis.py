"""
Day 75.2 — V1.0 Release Closure KPIs (Selector-Consistent)

Post-hoc analytics only. Computes deployment-grade KPIs that are
aligned with Selector v6's actual objective (max tg_roll once clean).

KPIs:
  Science Δ: epoch-median G1 (epochs >= 6)
  Safe Yield: % seeds in CLEAN pool
  KPI-A: Clean-Basin Utility (tg_roll comparison)
  KPI-B: Leaky cohort epoch-median improvement
  KPI-C: Do-No-Harm (clean seeds, tau_clean_hi)
  Safety: TOPO_FAIL, alignment, collapses, NaN, spike, dual-cap
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, statistics, sys
from pathlib import Path
from typing import Any, Dict, List

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# ── Thresholds (frozen v6) ──────────────────────────────────────────

TAU_CLEAN = 0.025
TAU_CLEAN_HI = 0.035
TG_FLOOR = -0.015
ACTIVE_EPOCH_MIN = 6
ROLL_WINDOW = 3
LEAKY_THRESHOLD = 0.025
SPIKE_LIMIT = 0.015


# ── Helpers ─────────────────────────────────────────────────────────

def _normalize_fields(rec: Dict) -> Dict:
    r = dict(rec)
    if "topo_slice_clean" in r and "slice_clean" not in r:
        r["slice_clean"] = r["topo_slice_clean"]
    if "topo_mean_drop" in r and "mean_drop" not in r:
        r["mean_drop"] = r["topo_mean_drop"]
    r.setdefault("slice_clean", None)
    r.setdefault("mean_drop", None)
    r.setdefault("topo_TG", None)
    return r


def load_all_results(path: str) -> Dict[str, List[Dict]]:
    with open(path) as f:
        data = json.load(f)
    return {k: [_normalize_fields(r) for r in v] for k, v in data.items()}


def compute_epoch_median(recs: List[Dict], epoch_min: int = 6) -> float:
    vals = [r["G1_aligned"] for r in recs
            if r.get("epoch", 0) >= epoch_min and r.get("G1_aligned") is not None]
    return statistics.median(vals) if vals else float("nan")


def _select_for_arm_seed(recs: List[Dict]) -> Dict:
    from qec_noise_factory.ml.ops.checkpoint_selection import select_epoch_for_seed
    return select_epoch_for_seed(
        recs, tau_clean=TAU_CLEAN, tau_clean_hi=TAU_CLEAN_HI,
        slice_floor=0.0, tg_floor=TG_FLOOR,
        active_epoch_min=ACTIVE_EPOCH_MIN, roll_window=ROLL_WINDOW,
        drop_slice_floor=True)


def discover_receipts(root: Path) -> Dict[str, Dict]:
    """Load actual receipts from disk (keyed by arm_seed)."""
    receipts = {}
    for f in sorted(root.glob("*receipt*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            arm = r.get("arm", "")
            seed = r.get("seed")
            if arm and seed is not None:
                receipts[f"{arm}_{seed}"] = r
        except Exception:
            pass
    return receipts


def extract_required_float(receipt: Dict, key: str, context: str = "") -> float:
    """Extract a required float from a receipt. Raises KeyError if missing/None."""
    val = receipt.get(key)
    if val is None:
        raise KeyError(
            f"Required field '{key}' missing or None in receipt"
            + (f" ({context})" if context else ""))
    return float(val)


# ── Main KPI computation ────────────────────────────────────────────

def compute_v1_release_kpis(artifact_dir: str, out_dir: str) -> Dict:
    root = Path(artifact_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_res = load_all_results(str(root / "all_results.json"))
    disk_receipts = discover_receipts(root)

    arms = ["Control", "ExactK_Tuned_Prod"]
    seeds = sorted(set(
        int(k.split("_")[-1]) for k in all_res
        if any(k.startswith(a + "_") for a in arms)))

    # ── Per-seed collection ──────────────────────────────────────
    rows = []
    for arm in arms:
        for seed in seeds:
            key = f"{arm}_{seed}"
            recs = all_res.get(key, [])
            g1_ep_med = compute_epoch_median(recs)

            # Selection (from disk receipt if available, else compute)
            sel = {}
            disk_key = key
            if disk_key in disk_receipts:
                r = disk_receipts[disk_key]
                ctx = f"{arm}_{seed}"
                sel = {
                    "epoch": r.get("chosen_epoch"),
                    "g1_inst": extract_required_float(r, "g1_aligned", ctx),
                    "g1roll": extract_required_float(r, "g1_roll", ctx),
                    "tg_roll": extract_required_float(r, "tg_roll", ctx),
                    "spike": extract_required_float(r, "g1_spike_delta", ctx),
                    "mode": r.get("selection_mode"),
                    "n_clean": r.get("n_clean"),
                    "n_surviving": r.get("n_surviving"),
                    "slice_clean": r.get("slice_clean"),
                }
            elif recs:
                try:
                    r = _select_for_arm_seed(recs)
                    sel = {
                        "epoch": r["epoch"],
                        "g1_inst": r["G1_aligned"],
                        "g1roll": r["g1_roll"],
                        "tg_roll": r["tg_roll"],
                        "spike": r["g1_spike_delta"],
                        "mode": r["selection_mode"],
                        "n_clean": r["n_clean"],
                        "n_surviving": r["n_surviving"],
                        "slice_clean": r.get("slice_clean"),
                    }
                except Exception as e:
                    sel = {"error": str(e)}

            is_clean = sel.get("mode", "").startswith("CLEAN")
            is_leaky = sel.get("mode", "") == "LEAKY_MIN_G1ROLL"
            is_topo_fail = "TOPO_FAIL" in sel.get("mode", "")

            deployable = False
            if sel and "error" not in sel:
                g1r = sel.get("g1roll", 999)
                g1i = sel.get("g1_inst", 999)
                tgr = sel.get("tg_roll", -999)
                if (g1r is not None and g1i is not None and tgr is not None
                        and g1r <= TAU_CLEAN and g1i <= TAU_CLEAN_HI
                        and tgr >= TG_FLOOR):
                    deployable = True

            rows.append({
                "arm": arm, "seed": seed,
                "g1_epoch_median": g1_ep_med,
                "sel_epoch": sel.get("epoch"),
                "g1_inst": sel.get("g1_inst"),
                "g1roll": sel.get("g1roll"),
                "tg_roll": sel.get("tg_roll"),
                "spike": sel.get("spike"),
                "mode": sel.get("mode"),
                "is_clean": is_clean,
                "is_leaky": is_leaky,
                "is_topo_fail": is_topo_fail,
                "deployable_clean": deployable,
            })

    # ── Science Δ ────────────────────────────────────────────────
    sci = {}
    for arm in arms:
        meds = [r["g1_epoch_median"] for r in rows if r["arm"] == arm]
        sci[arm] = {"median": statistics.median(meds), "per_seed": meds}
    ctrl_med = sci["Control"]["median"]
    prod_med = sci["ExactK_Tuned_Prod"]["median"]
    sci_delta_pct = (ctrl_med - prod_med) / ctrl_med * 100
    sci_delta_abs = ctrl_med - prod_med

    # ── Safe Yield ───────────────────────────────────────────────
    safe_yield = {}
    for arm in arms:
        ar = [r for r in rows if r["arm"] == arm]
        n_clean = sum(1 for r in ar if r["is_clean"])
        safe_yield[arm] = {
            "clean_count": n_clean, "total": len(ar),
            "pct": n_clean / max(len(ar), 1) * 100,
        }

    # ── Safety ───────────────────────────────────────────────────
    prod_rows = [r for r in rows if r["arm"] == "ExactK_Tuned_Prod"]
    topo_fail_count = sum(1 for r in prod_rows if r["is_topo_fail"])
    topo_fail_pct = topo_fail_count / max(len(prod_rows), 1) * 100

    # Spike + dual-cap
    integrity = {}
    for arm in arms:
        ar = [r for r in rows if r["arm"] == arm]
        clean_r = [r for r in ar if r["is_clean"]]
        spikes = [abs(r["spike"]) for r in clean_r if r.get("spike") is not None]
        spike_viols = sum(1 for s in spikes if s >= SPIKE_LIMIT)
        # dual-cap: g1roll <= tau_clean AND g1_inst <= tau_clean_hi
        dual_viols = sum(1 for r in clean_r
                         if r.get("g1roll") is not None
                         and (r["g1roll"] > TAU_CLEAN or r["g1_inst"] > TAU_CLEAN_HI))
        modes = {}
        for r in ar:
            m = r.get("mode", "UNKNOWN")
            modes[m] = modes.get(m, 0) + 1
        integrity[arm] = {
            "max_spike_clean": max(spikes) if spikes else 0.0,
            "spike_violations": spike_viols,
            "dual_cap_violations": dual_viols,
            "modes": modes,
        }

    # ── Cohort partition ─────────────────────────────────────────
    ctrl_ep_meds = {r["seed"]: r["g1_epoch_median"]
                    for r in rows if r["arm"] == "Control"}
    leaky_seeds = sorted(s for s, m in ctrl_ep_meds.items() if m >= LEAKY_THRESHOLD)
    clean_seeds = sorted(s for s, m in ctrl_ep_meds.items() if m < LEAKY_THRESHOLD)

    # ── KPI-A: Clean-Basin Utility (tg_roll) ─────────────────────
    kpi_a = {}
    for arm in arms:
        tg_vals = [r["tg_roll"] for r in rows
                   if r["arm"] == arm and r["is_clean"] and r.get("tg_roll") is not None]
        kpi_a[arm] = {
            "median_tg_roll": statistics.median(tg_vals) if tg_vals else None,
            "count": len(tg_vals),
            "values": tg_vals,
        }
    ctrl_tg = kpi_a["Control"]["median_tg_roll"]
    prod_tg = kpi_a["ExactK_Tuned_Prod"]["median_tg_roll"]
    if ctrl_tg is not None and prod_tg is not None and ctrl_tg > 0:
        kpi_a_delta_pct = (prod_tg - ctrl_tg) / ctrl_tg * 100
    else:
        kpi_a_delta_pct = None
    kpi_a["delta_pct"] = kpi_a_delta_pct

    # ── KPI-B: Leaky cohort epoch-median efficacy ────────────────
    leaky_improvements = []
    leaky_details = []
    for seed in leaky_seeds:
        cr = next((r for r in rows if r["arm"] == "Control" and r["seed"] == seed), None)
        pr = next((r for r in rows if r["arm"] == "ExactK_Tuned_Prod" and r["seed"] == seed), None)
        if cr and pr:
            abs_imp = cr["g1_epoch_median"] - pr["g1_epoch_median"]
            rel_imp = abs_imp / cr["g1_epoch_median"] * 100 if cr["g1_epoch_median"] > 0 else 0
            leaky_improvements.append(abs_imp)
            leaky_details.append({
                "seed": seed,
                "ctrl_ep_med": round(cr["g1_epoch_median"], 6),
                "prod_ep_med": round(pr["g1_epoch_median"], 6),
                "abs_improvement": round(abs_imp, 6),
                "rel_improvement_pct": round(rel_imp, 1),
                "prod_wins": pr["g1_epoch_median"] < cr["g1_epoch_median"],
            })

    kpi_b = {
        "leaky_seeds": leaky_seeds,
        "n_leaky": len(leaky_seeds),
        "median_abs_improvement": round(statistics.median(leaky_improvements), 6) if leaky_improvements else None,
        "median_rel_improvement_pct": round(statistics.median(
            [d["rel_improvement_pct"] for d in leaky_details]), 1) if leaky_details else None,
        "pct_prod_wins": round(sum(1 for d in leaky_details if d["prod_wins"])
                               / max(len(leaky_details), 1) * 100, 0),
        "details": leaky_details,
    }

    # ── KPI-C: Do-No-Harm ────────────────────────────────────────
    dnh_details = []
    dnh_violations = []
    for seed in clean_seeds:
        pr = next((r for r in rows if r["arm"] == "ExactK_Tuned_Prod" and r["seed"] == seed), None)
        if pr:
            g1_sel = pr.get("g1_inst")
            ok = g1_sel is not None and g1_sel <= TAU_CLEAN_HI
            dnh_details.append({
                "seed": seed,
                "prod_g1_selected": round(g1_sel, 6) if g1_sel is not None else None,
                "threshold": TAU_CLEAN_HI,
                "pass": ok,
            })
            if not ok:
                dnh_violations.append(seed)

    kpi_c = {
        "clean_seeds": clean_seeds,
        "n_clean": len(clean_seeds),
        "violations": dnh_violations,
        "n_violations": len(dnh_violations),
        "details": dnh_details,
    }

    # ── Assemble report ──────────────────────────────────────────
    report = {
        "version": "V1.0_release",
        "day": "75.2",
        "source": str(artifact_dir),
        "thresholds": {
            "tau_clean": TAU_CLEAN, "tau_clean_hi": TAU_CLEAN_HI,
            "tg_floor": TG_FLOOR, "leaky_threshold": LEAKY_THRESHOLD,
            "spike_limit": SPIKE_LIMIT, "active_epoch_min": ACTIVE_EPOCH_MIN,
        },
        "science": {
            "ctrl_med": round(ctrl_med, 6),
            "prod_med": round(prod_med, 6),
            "delta_pct": round(sci_delta_pct, 1),
            "delta_abs": round(sci_delta_abs, 6),
        },
        "safe_yield": safe_yield,
        "topo_fail": {
            "count": topo_fail_count,
            "pct": round(topo_fail_pct, 1),
        },
        "kpi_a_clean_basin_utility": {
            "ctrl_median_tg_roll": round(ctrl_tg, 6) if ctrl_tg is not None else None,
            "prod_median_tg_roll": round(prod_tg, 6) if prod_tg is not None else None,
            "delta_pct": round(kpi_a_delta_pct, 1) if kpi_a_delta_pct is not None else None,
            "ctrl_count": kpi_a["Control"]["count"],
            "prod_count": kpi_a["ExactK_Tuned_Prod"]["count"],
        },
        "kpi_b_leaky_cohort_efficacy": kpi_b,
        "kpi_c_do_no_harm": kpi_c,
        "integrity": integrity,
        "seeds": seeds,
        "arms": arms,
    }

    # ── Write outputs ────────────────────────────────────────────
    (out / "v1_release_kpis.json").write_text(
        json.dumps(report, indent=2, default=_convert), encoding="utf-8")

    # CSV
    csv_fields = ["arm", "seed", "cohort", "g1_epoch_median",
                  "sel_epoch", "g1_inst", "g1roll", "tg_roll", "spike",
                  "mode", "is_clean", "deployable_clean"]
    with open(out / "v1_release_kpis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in rows:
            cohort = "leaky" if r["seed"] in leaky_seeds else "clean"
            w.writerow({k: (round(r[k], 6) if isinstance(r.get(k), float) else r.get(k))
                        for k in csv_fields if k != "cohort"} | {"cohort": cohort})

    # Markdown
    md = _build_md(report, rows, leaky_seeds, clean_seeds)
    (out / "v1_release_kpis.md").write_text(md, encoding="utf-8")

    # Checksums
    checksums = {}
    for f in sorted(out.iterdir()):
        if f.is_file() and f.name != "checksums.sha256":
            checksums[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    (out / "checksums.sha256").write_text(
        "\n".join(f"{v}  {k}" for k, v in sorted(checksums.items())) + "\n",
        encoding="utf-8")

    return report


def _build_md(rpt, rows, leaky_seeds, clean_seeds):
    s = rpt["science"]
    sy = rpt["safe_yield"]
    ka = rpt["kpi_a_clean_basin_utility"]
    kb = rpt["kpi_b_leaky_cohort_efficacy"]
    kc = rpt["kpi_c_do_no_harm"]
    ig = rpt["integrity"]
    tf = rpt["topo_fail"]

    P = lambda v: "PASS" if v else "FAIL"

    lines = [
        "# V1.0 Release KPIs -- Day 75.2",
        "",
        "## Headline",
        "",
        "| KPI | Value | Target | Status |",
        "|-----|-------|--------|--------|",
        f"| Science Delta (epoch-median) | **{s['delta_pct']:+.1f}%** (abs {s['delta_abs']:.4f}) | >= 20% | {P(s['delta_pct'] >= 20)} |",
        f"| Safe Yield (Prod) | **{sy['ExactK_Tuned_Prod']['pct']:.0f}%** ({sy['ExactK_Tuned_Prod']['clean_count']}/{sy['ExactK_Tuned_Prod']['total']}) | >= 80% | {P(sy['ExactK_Tuned_Prod']['pct'] >= 80)} |",
        f"| TOPO_FAIL | **{tf['count']}/{len(rpt['seeds'])}** ({tf['pct']:.0f}%) | <= 10% | {P(tf['pct'] <= 10)} |",
    ]

    # KPI-A
    if ka.get("prod_median_tg_roll") is not None and ka.get("ctrl_median_tg_roll") is not None:
        delta_str = f" ({ka['delta_pct']:+.1f}%)" if ka.get('delta_pct') is not None else ""
        lines.append(
            f"| KPI-A: Clean tg_roll (Prod vs Ctrl) | Prod={ka['prod_median_tg_roll']:.4f} vs Ctrl={ka['ctrl_median_tg_roll']:.4f}{delta_str} | informational | -- |")
    # KPI-B
    if kb["median_abs_improvement"] is not None:
        lines.append(
            f"| KPI-B: Leaky cohort ep-median improvement | **{kb['median_abs_improvement']:.4f}** ({kb['median_rel_improvement_pct']:+.1f}%) | > 0 | {P(kb['median_abs_improvement'] > 0)} |")
        lines.append(
            f"| KPI-B: Prod wins on leaky seeds | {kb['pct_prod_wins']:.0f}% ({kb['n_leaky']} seeds) | > 50% | {P(kb['pct_prod_wins'] > 50)} |")
    # KPI-C
    lines.append(
        f"| KPI-C: Do-No-Harm violations | **{kc['n_violations']}** (threshold {TAU_CLEAN_HI}) | 0 | {P(kc['n_violations'] == 0)} |")
    # Integrity
    prod_ig = ig["ExactK_Tuned_Prod"]
    lines.extend([
        f"| Spike violations (Prod CLEAN) | **{prod_ig['spike_violations']}** (max={prod_ig['max_spike_clean']:.4f}) | 0 | {P(prod_ig['spike_violations'] == 0)} |",
        f"| Dual-cap violations (Prod) | **{prod_ig['dual_cap_violations']}** | 0 | {P(prod_ig['dual_cap_violations'] == 0)} |",
    ])

    lines.extend([
        "",
        "## Metric Deprecations",
        "",
        "> **Selected-G1 Delta% is deprecated permanently.** Selector v6 uses asymmetric",
        "> objectives: LEAKY pool minimizes g1roll (reduce leakage) while CLEAN pool",
        "> maximizes tg_roll (optimize topology). Comparing selected G1 across pools",
        "> conflates two optimization targets and produces meaningless near-zero ratios.",
        "",
        "## Science Metric (E1)",
        "",
        f"- Control epoch-median G1: {s['ctrl_med']:.6f}",
        f"- Prod epoch-median G1: {s['prod_med']:.6f}",
        f"- Delta = {s['delta_pct']:+.1f}% (abs {s['delta_abs']:.6f})",
        "",
        "## KPI-A: Clean-Basin Utility",
        "",
        "For seeds selected as CLEAN, compare median tg_roll (topology quality once safe).",
        "",
        f"- Control: median tg_roll = {ka['ctrl_median_tg_roll']:.4f} ({ka['ctrl_count']} seeds)",
        f"- Prod: median tg_roll = {ka['prod_median_tg_roll']:.4f} ({ka['prod_count']} seeds)",
    ])
    if ka["delta_pct"] is not None:
        lines.append(f"- Delta = {ka['delta_pct']:+.1f}%")

    lines.extend([
        "",
        "## KPI-B: Leaky Cohort Epoch-Median Efficacy",
        "",
        f"Leaky cohort: seeds where Control epoch-median >= {LEAKY_THRESHOLD}: {leaky_seeds}",
        "",
        "| Seed | Ctrl ep-med | Prod ep-med | Abs Improv | Rel% | Prod wins |",
        "|------|-------------|-------------|------------|------|-----------|",
    ])
    for d in kb["details"]:
        lines.append(
            f"| {d['seed']} | {d['ctrl_ep_med']:.4f} | {d['prod_ep_med']:.4f} "
            f"| {d['abs_improvement']:+.4f} | {d['rel_improvement_pct']:+.1f}% "
            f"| {'YES' if d['prod_wins'] else 'NO'} |")
    if kb["median_abs_improvement"] is not None:
        lines.extend([
            "",
            f"Median abs improvement: **{kb['median_abs_improvement']:.4f}**",
            f"Median rel improvement: **{kb['median_rel_improvement_pct']:+.1f}%**",
            f"Prod wins: **{kb['pct_prod_wins']:.0f}%**",
        ])

    lines.extend([
        "",
        "## KPI-C: Do-No-Harm",
        "",
        f"Clean seeds (Control epoch-median < {LEAKY_THRESHOLD}): {clean_seeds}",
        "",
        "| Seed | Prod Sel G1 | <= {:.3f}? |".format(TAU_CLEAN_HI),
        "|------|-------------|----------|",
    ])
    for d in kc["details"]:
        g = f"{d['prod_g1_selected']:.4f}" if d["prod_g1_selected"] is not None else "N/A"
        lines.append(f"| {d['seed']} | {g} | {'PASS' if d['pass'] else 'FAIL'} |")
    lines.append(f"\nViolations: **{kc['n_violations']}**")

    lines.extend([
        "",
        "## Integrity",
        "",
        "| Arm | Modes | Max Spike (CLEAN) | Spike Viols | Dual-cap Viols |",
        "|-----|-------|-------------------|-------------|----------------|",
    ])
    for arm in rpt["arms"]:
        a = ig[arm]
        modes_str = ", ".join(f"{k}={v}" for k, v in sorted(a["modes"].items()))
        lines.append(
            f"| {arm} | {modes_str} | {a['max_spike_clean']:.4f} "
            f"| {a['spike_violations']} | {a['dual_cap_violations']} |")

    return "\n".join(lines) + "\n"


def _convert(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Day 75.2 V1.0 release KPIs")
    parser.add_argument("--artifact_dir", default="ml_artifacts/day75_holdout_d7_v1")
    parser.add_argument("--out_dir", default="ml_artifacts/day75_2_v1_release_closure")
    args = parser.parse_args()

    rpt = compute_v1_release_kpis(args.artifact_dir, args.out_dir)

    s = rpt["science"]
    sy = rpt["safe_yield"]
    ka = rpt["kpi_a_clean_basin_utility"]
    kb = rpt["kpi_b_leaky_cohort_efficacy"]
    kc = rpt["kpi_c_do_no_harm"]
    ig = rpt["integrity"]
    tf = rpt["topo_fail"]

    print(f"\n{'='*70}")
    print("  V1.0 RELEASE KPIs -- Day 75.2")
    print(f"{'='*70}")
    print(f"\n  Science Delta:         {s['delta_pct']:+.1f}% (abs={s['delta_abs']:.6f})")
    print(f"  Safe Yield (Prod):     {sy['ExactK_Tuned_Prod']['pct']:.0f}% "
          f"({sy['ExactK_Tuned_Prod']['clean_count']}/{sy['ExactK_Tuned_Prod']['total']})")
    print(f"  Safe Yield (Ctrl):     {sy['Control']['pct']:.0f}% "
          f"({sy['Control']['clean_count']}/{sy['Control']['total']})")
    print(f"  TOPO_FAIL:             {tf['count']}/{len(rpt['seeds'])} ({tf['pct']:.0f}%)")
    if ka["prod_median_tg_roll"] is not None and ka["ctrl_median_tg_roll"] is not None:
        print(f"  KPI-A tg_roll (CLEAN): Prod={ka['prod_median_tg_roll']:.4f} "
              f"Ctrl={ka['ctrl_median_tg_roll']:.4f}"
              + (f" ({ka['delta_pct']:+.1f}%)" if ka['delta_pct'] is not None else ""))
    if kb["median_abs_improvement"] is not None:
        print(f"  KPI-B Leaky ep-med:    abs={kb['median_abs_improvement']:.4f} "
              f"rel={kb['median_rel_improvement_pct']:+.1f}% "
              f"(wins {kb['pct_prod_wins']:.0f}%)")
    print(f"  KPI-C Do-No-Harm:     {kc['n_violations']} violations")
    print(f"  Spike (Prod CLEAN):   max={ig['ExactK_Tuned_Prod']['max_spike_clean']:.4f} "
          f"violations={ig['ExactK_Tuned_Prod']['spike_violations']}")
    print(f"  Dual-cap (Prod):      {ig['ExactK_Tuned_Prod']['dual_cap_violations']}")

    print(f"\n  Output: {args.out_dir}/")
    for f in sorted(Path(args.out_dir).iterdir()):
        if f.is_file():
            print(f"    {f.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
