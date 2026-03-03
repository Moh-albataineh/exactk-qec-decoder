"""
Day 73 -- Checkpoint Selector v6 (Smoothed Topology Floors for d=7).

Key changes from v5:
  - Rolling medians for SliceClean and TG (not just G1)
  - Survival uses smoothed values: slice_clean_roll >= floor AND tg_roll >= tg_floor
  - CLEAN/TOPO_FAIL sort by tg_roll (smoothed), with precise tie-breakers
  - LEAKY: argmin(g1roll), tie-break by higher tg_roll

Also keeps DNHGate class (unchanged) and compute_slice_clean_null_p05().
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# DNH Hysteresis Gate (unchanged)
# ---------------------------------------------------------------------------

class DNHGate:
    """Do-No-Harm hysteresis gate based on aligned G1 probe."""

    def __init__(self, tau_off=0.020, tau_on=0.025, warmup_epochs=5,
                 lambda_base=0.10, decay_rate=0.85, decay_start_epoch=8):
        self.tau_off = tau_off
        self.tau_on = tau_on
        self.warmup_epochs = warmup_epochs
        self.lambda_base = lambda_base
        self.decay_rate = decay_rate
        self.decay_start_epoch = decay_start_epoch
        self._state = 0

    @property
    def state(self) -> int:
        return self._state

    def update(self, g1_aligned: float, epoch: int) -> int:
        if epoch < self.warmup_epochs:
            self._state = 0
            return self._state
        if g1_aligned >= self.tau_on:
            self._state = 1
        elif g1_aligned <= self.tau_off:
            self._state = 0
        return self._state

    def compute_iso_weight(self, epoch: int) -> float:
        if self._state == 0 or epoch <= self.warmup_epochs:
            return 0.0
        decay = self.decay_rate ** max(0, epoch - self.decay_start_epoch)
        return self.lambda_base * decay


# ---------------------------------------------------------------------------
# Empirical SliceClean null floor (unchanged from v5)
# ---------------------------------------------------------------------------

def compute_slice_clean_null_p05(
    probe_Y: np.ndarray,
    probe_K: np.ndarray,
    n_shuffles: int = 200,
    seed: int = 0,
    top_k_slices: int = 5,
    min_slice_size: int = 25,
) -> float:
    """Compute p05 of SliceClean distribution under random scores.

    Returns slice_floor = min(0.500, p05).
    """
    from qec_noise_factory.ml.metrics.ranking import compute_auroc

    rng = np.random.RandomState(seed)
    y = np.asarray(probe_Y, dtype=bool).ravel()
    k = np.asarray(probe_K, dtype=int).ravel()

    unique_k, counts = np.unique(k, return_counts=True)
    freq_order = np.argsort(-counts)
    selected_k = []
    for idx in freq_order:
        if counts[idx] >= min_slice_size:
            selected_k.append(unique_k[idx])
        if len(selected_k) >= top_k_slices:
            break

    slice_masks = []
    for kv in sorted(selected_k):
        mask = k == kv
        y_s = y[mask]
        if y_s.all() or (~y_s).all():
            continue
        slice_masks.append((kv, mask))

    if not slice_masks:
        return 0.500

    null_scores = []
    for _ in range(n_shuffles):
        scores = rng.randn(len(y))
        aurocs = []
        for kv, mask in slice_masks:
            y_s = y[mask]
            s_s = scores[mask]
            auroc = compute_auroc(y_s, s_s)
            if auroc is not None:
                aurocs.append(max(auroc, 1.0 - auroc))
        if aurocs:
            null_scores.append(float(np.mean(aurocs)))

    if not null_scores:
        return 0.500

    p05 = float(np.percentile(null_scores, 5))
    return min(0.500, p05)


def compute_slice_clean_baseline(n_samples=10000, n_trials=50) -> float:
    """Quick random baseline sanity check (~0.50)."""
    import random
    rng = random.Random(42)
    results = []
    for _ in range(n_trials):
        labels = [rng.randint(0, 1) for _ in range(n_samples)]
        scores = [rng.random() for _ in range(n_samples)]
        n_bins = 20
        bin_size = n_samples // n_bins
        overall_pos_rate = sum(labels) / len(labels)
        clean_bins = 0
        paired = sorted(zip(scores, labels), key=lambda x: x[0])
        for b in range(n_bins):
            start = b * bin_size
            end = start + bin_size
            bin_labels = [p[1] for p in paired[start:end]]
            bin_pos_rate = sum(bin_labels) / len(bin_labels) if bin_labels else 0
            if bin_pos_rate >= overall_pos_rate:
                clean_bins += 1
        results.append(clean_bins / n_bins)
    return sum(results) / len(results)


# ---------------------------------------------------------------------------
# Data loading (multi-format)
# ---------------------------------------------------------------------------

def _normalize_epoch(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize epoch data to canonical fields regardless of source format."""
    return {
        "epoch": raw["epoch"],
        "G1_aligned": raw.get("G1_aligned",
                              raw.get("G1_raw_probe", 0.0)),
        "slice_clean": raw.get("slice_clean",
                               raw.get("topo_slice_clean", 0.0)),
        "mean_drop": raw.get("mean_drop",
                             raw.get("topo_mean_drop", 0.0)),
        "topo_TG": raw.get("topo_TG", None),
        "loss": raw.get("loss", None),
    }


def load_day_artifacts_auto(artifact_dir: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """Auto-detect artifact format and load.

    Supports:
      - Day 70+ format: best_epoch_candidates.json + all_results.json
      - Day 69 format: all_results.json only (dict of key -> list of epoch dicts)
      - JSONL format: metrics_{seed}.jsonl files
    """
    artifact_dir = Path(artifact_dir)
    bec_path = artifact_dir / "best_epoch_candidates.json"
    ar_path = artifact_dir / "all_results.json"

    if bec_path.exists() and ar_path.exists():
        return _load_day70_format(bec_path, ar_path)
    elif ar_path.exists():
        return _load_day69_format(ar_path)
    else:
        # Try JSONL
        jsonl_files = list(artifact_dir.glob("metrics_*.jsonl"))
        if jsonl_files:
            return _load_jsonl_format(jsonl_files)
        raise FileNotFoundError(
            f"No recognized artifact format in {artifact_dir}. "
            f"Expected best_epoch_candidates.json + all_results.json, "
            f"or all_results.json alone, or metrics_*.jsonl files."
        )


# Keep old name as alias
def load_day_artifacts(artifact_dir: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    return load_day_artifacts_auto(artifact_dir)


def _load_day70_format(bec_path: Path, ar_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Day 70+ format: best_epoch_candidates + all_results."""
    with open(bec_path, "r", encoding="utf-8") as f:
        bec = json.load(f)
    with open(ar_path, "r", encoding="utf-8") as f:
        ar = json.load(f)

    merged: Dict[str, List[Dict[str, Any]]] = {}
    for key, entry in sorted(bec.items()):
        candidates = entry.get("candidates", [])
        ar_epochs = {}
        if key in ar:
            for ep_data in ar[key]:
                ar_epochs[ep_data["epoch"]] = ep_data

        epoch_list = []
        for cand in candidates:
            ep = cand["epoch"]
            ar_ep = ar_epochs.get(ep, {})
            combined = {**ar_ep, **cand}
            epoch_list.append(_normalize_epoch(combined))
        epoch_list.sort(key=lambda x: x["epoch"])
        merged[key] = epoch_list
    return merged


def _load_day69_format(ar_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Day 69 format: all_results.json only (dict of key -> epoch list)."""
    with open(ar_path, "r", encoding="utf-8") as f:
        ar = json.load(f)

    merged: Dict[str, List[Dict[str, Any]]] = {}
    for key, epochs in sorted(ar.items()):
        epoch_list = [_normalize_epoch(ep) for ep in epochs]
        epoch_list.sort(key=lambda x: x["epoch"])
        merged[key] = epoch_list
    return merged


def _load_jsonl_format(jsonl_files: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    """JSONL format: one file per seed with per-epoch JSON lines."""
    merged: Dict[str, List[Dict[str, Any]]] = {}
    for fp in jsonl_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                arm = rec.get("arm", "Unknown")
                seed = rec.get("seed", 0)
                key = f"{arm}_{seed}"
                if key not in merged:
                    merged[key] = []
                merged[key].append(_normalize_epoch(rec))
    for key in merged:
        merged[key].sort(key=lambda x: x["epoch"])
    return merged


# ---------------------------------------------------------------------------
# JSONL Write-Ahead Logger (Day 74 v1.0 MLOps)
# ---------------------------------------------------------------------------

class EpochLogger:
    """Write-ahead JSONL logger that flushes every epoch."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log_epoch(self, record: Dict[str, Any]) -> None:
        """Write one JSON line and flush. All values must be primitives."""
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Checkpoint utilities (Day 74 v1.0 MLOps)
# ---------------------------------------------------------------------------

def save_checkpoint(model_state_dict: dict, ckpt_dir: str | Path,
                    seed: int, epoch: int) -> Path:
    """Save model state_dict to ckpt_{seed}_ep{epoch}.pt."""
    import torch
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_{seed}_ep{epoch}.pt"
    torch.save(model_state_dict, path)
    return path


def write_selection_receipt(
    receipt: Dict[str, Any],
    out_dir: str | Path,
    seed: int,
) -> Path:
    """Write selection_receipt_{seed}.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"selection_receipt_{seed}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    return path


def copy_best_checkpoint(
    ckpt_dir: str | Path,
    out_dir: str | Path,
    seed: int,
    chosen_epoch: int,
) -> Path:
    """Copy chosen checkpoint to best_model_{seed}.pt."""
    import shutil
    ckpt_dir = Path(ckpt_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = ckpt_dir / f"ckpt_{seed}_ep{chosen_epoch}.pt"
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src}")
    dst = out_dir / f"best_model_{seed}.pt"
    shutil.copy2(src, dst)
    return dst


def cleanup_unselected_checkpoints(
    ckpt_dir: str | Path,
    seed: int,
    chosen_epoch: int,
) -> List[Path]:
    """Delete unselected checkpoint files. Returns deleted paths."""
    ckpt_dir = Path(ckpt_dir)
    deleted = []
    for p in sorted(ckpt_dir.glob(f"ckpt_{seed}_ep*.pt")):
        ep_str = p.stem.split("_ep")[-1]
        try:
            ep = int(ep_str)
        except ValueError:
            continue
        if ep != chosen_epoch:
            p.unlink()
            deleted.append(p)
    return deleted


def _parse_key(key: str) -> Tuple[str, int]:

    parts = key.rsplit("_", 1)
    return parts[0], int(parts[1])


# ---------------------------------------------------------------------------
# Rolling median (generic, works for any key)
# ---------------------------------------------------------------------------

def rolling_median(
    metrics_by_epoch: List[Dict[str, Any]],
    key: str = "G1_aligned",
    window: int = 3,
    active_epoch_min: int = 6,
) -> Dict[int, float]:
    """Trailing rolling median over active epochs for the given key."""
    active = sorted(
        [m for m in metrics_by_epoch if m["epoch"] >= active_epoch_min],
        key=lambda m: m["epoch"],
    )
    result: Dict[int, float] = {}
    for i, m in enumerate(active):
        val = m.get(key)
        if val is None:
            val = 0.0
        start = max(0, i - window + 1)
        window_vals = []
        for j in range(start, i + 1):
            v = active[j].get(key)
            window_vals.append(v if v is not None else 0.0)
        result[m["epoch"]] = statistics.median(window_vals)
    return result


# ---------------------------------------------------------------------------
# Selector v6
# ---------------------------------------------------------------------------

def select_epoch_for_seed(
    metrics_by_epoch: List[Dict[str, Any]],
    tau_clean: float = 0.025,
    tau_clean_hi: float = 0.035,
    slice_floor: float = 0.500,
    tg_floor: float = -0.015,
    active_epoch_min: int = 6,
    roll_window: int = 3,
    drop_slice_floor: bool = False,
) -> Dict[str, Any]:
    """Select best epoch for one seed/arm (v6: smoothed topology floors).

    Policy:
      1. active = epochs >= active_epoch_min
      2. Compute rolling medians: g1roll, slice_clean_roll, tg_roll
      3. surviving = active where slice_clean_roll >= floor AND tg_roll >= tg_floor
         (if drop_slice_floor: skip slice_clean_roll check)
      4. clean_pool = surviving where g1roll <= tau_clean AND g1_aligned <= tau_clean_hi
      5. Selection:
         - surviving empty  → TOPO_FAIL_MAX_TG: argmax(tg_roll) over all active,
                              tie-break: lower g1roll
         - clean non-empty  → CLEAN_MAX_TG: argmax(tg_roll),
                              tie-break: higher slice_clean_roll, then lower g1roll
         - else             → LEAKY_MIN_G1ROLL: argmin(g1roll),
                              tie-break: higher tg_roll
    """
    active = [m for m in metrics_by_epoch if m["epoch"] >= active_epoch_min]
    if not active:
        raise ValueError(f"No epochs >= {active_epoch_min} found")

    # Compute all rolling medians
    g1_roll_map = rolling_median(
        metrics_by_epoch, key="G1_aligned",
        window=roll_window, active_epoch_min=active_epoch_min)
    sc_roll_map = rolling_median(
        metrics_by_epoch, key="slice_clean",
        window=roll_window, active_epoch_min=active_epoch_min)
    tg_roll_map = rolling_median(
        metrics_by_epoch, key="topo_TG",
        window=roll_window, active_epoch_min=active_epoch_min)

    n_active = len(active)

    # Survival filter (smoothed)
    surviving = []
    for m in active:
        sc_r = sc_roll_map.get(m["epoch"], 0.0)
        tg_r = tg_roll_map.get(m["epoch"], 0.0)
        if tg_r >= tg_floor and (drop_slice_floor or sc_r >= slice_floor):
            surviving.append(m)

    n_surviving = len(surviving)

    if not surviving:
        # TOPO_FAIL: argmax(tg_roll) over ALL active, tie-break: lower g1roll
        chosen = max(active, key=lambda m: (
            tg_roll_map.get(m["epoch"], -999.0),
            -g1_roll_map.get(m["epoch"], 999.0),
        ))
        return _build_result(
            chosen, g1_roll_map, sc_roll_map, tg_roll_map,
            mode="TOPO_FAIL_MAX_TG",
            n_active=n_active, n_surviving=0, n_clean=0, n_leaky=0,
            slice_floor_used=slice_floor,
        )

    # Dual-cap clean pool
    clean_pool = [
        m for m in surviving
        if (g1_roll_map.get(m["epoch"], 999.0) <= tau_clean
            and m["G1_aligned"] <= tau_clean_hi)
    ]
    leaky_pool = [m for m in surviving if m not in clean_pool]
    n_clean = len(clean_pool)
    n_leaky = len(leaky_pool)

    if clean_pool:
        # CLEAN: argmax(tg_roll), tie-break: higher slice_clean_roll, lower g1roll
        chosen = max(clean_pool, key=lambda m: (
            tg_roll_map.get(m["epoch"], -999.0),
            sc_roll_map.get(m["epoch"], 0.0),
            -g1_roll_map.get(m["epoch"], 999.0),
        ))
        return _build_result(
            chosen, g1_roll_map, sc_roll_map, tg_roll_map,
            mode="CLEAN_MAX_TG",
            n_active=n_active, n_surviving=n_surviving,
            n_clean=n_clean, n_leaky=n_leaky,
            slice_floor_used=slice_floor,
        )
    else:
        # LEAKY: argmin(g1roll), tie-break: higher tg_roll
        chosen = min(surviving, key=lambda m: (
            g1_roll_map.get(m["epoch"], 999.0),
            -tg_roll_map.get(m["epoch"], -999.0),
        ))
        return _build_result(
            chosen, g1_roll_map, sc_roll_map, tg_roll_map,
            mode="LEAKY_MIN_G1ROLL",
            n_active=n_active, n_surviving=n_surviving,
            n_clean=n_clean, n_leaky=n_leaky,
            slice_floor_used=slice_floor,
        )


def _build_result(
    chosen: Dict[str, Any],
    g1_roll_map: Dict[int, float],
    sc_roll_map: Dict[int, float],
    tg_roll_map: Dict[int, float],
    mode: str,
    n_active: int,
    n_surviving: int,
    n_clean: int,
    n_leaky: int,
    slice_floor_used: float,
) -> Dict[str, Any]:
    ep = chosen["epoch"]
    g1r = g1_roll_map.get(ep, chosen["G1_aligned"])
    scr = sc_roll_map.get(ep, chosen["slice_clean"])
    tgr = tg_roll_map.get(ep, chosen.get("topo_TG") or 0.0)
    return {
        "epoch": ep,
        "G1_aligned": chosen["G1_aligned"],
        "g1_roll": round(g1r, 6),
        "g1_spike_delta": round(chosen["G1_aligned"] - g1r, 6),
        "slice_clean": chosen["slice_clean"],
        "slice_clean_roll": round(scr, 6),
        "topo_TG": chosen.get("topo_TG"),
        "tg_roll": round(tgr, 6),
        "mean_drop": chosen["mean_drop"],
        "loss": chosen.get("loss"),
        "selection_mode": mode,
        "n_active": n_active,
        "n_surviving": n_surviving,
        "n_clean": n_clean,
        "n_leaky": n_leaky,
        "slice_floor_used": round(slice_floor_used, 6),
    }


# ---------------------------------------------------------------------------
# Full selection run
# ---------------------------------------------------------------------------

def run_selection(
    artifact_dir: str | Path,
    out_dir: str | Path,
    tau_clean: float = 0.025,
    tau_clean_hi: float = 0.035,
    slice_floors: Optional[Dict[str, float]] = None,
    default_slice_floor: float = 0.500,
    tg_floor: float = -0.015,
    active_epoch_min: int = 6,
    roll_window: int = 3,
    drop_slice_floor: bool = False,
) -> Dict[str, Any]:
    """Run v6 checkpoint selection on artifacts, write outputs."""
    artifact_dir = Path(artifact_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_day_artifacts(artifact_dir)

    selections: List[Dict[str, Any]] = []
    for key in sorted(merged.keys()):
        arm, seed = _parse_key(key)
        floor = default_slice_floor
        if slice_floors:
            floor = slice_floors.get(key, slice_floors.get(str(seed), default_slice_floor))

        result = select_epoch_for_seed(
            merged[key],
            tau_clean=tau_clean, tau_clean_hi=tau_clean_hi,
            slice_floor=floor, tg_floor=tg_floor,
            active_epoch_min=active_epoch_min, roll_window=roll_window,
            drop_slice_floor=drop_slice_floor,
        )
        result["arm"] = arm
        result["seed"] = seed
        selections.append(result)

    # Aggregates
    arms = sorted(set(s["arm"] for s in selections))
    seeds = sorted(set(s["seed"] for s in selections))

    old_medians: Dict[str, float] = {}
    new_medians: Dict[str, float] = {}

    for arm in arms:
        old_g1s, new_g1s = [], []
        for seed in seeds:
            key = f"{arm}_{seed}"
            if key not in merged:
                continue
            active_eps = [m for m in merged[key] if m["epoch"] >= active_epoch_min]
            if active_eps:
                old_g1s.append(statistics.median([m["G1_aligned"] for m in active_eps]))
            sel = [s for s in selections if s["arm"] == arm and s["seed"] == seed]
            if sel:
                new_g1s.append(sel[0]["G1_aligned"])
        old_medians[arm] = statistics.median(old_g1s) if old_g1s else 0.0
        new_medians[arm] = statistics.median(new_g1s) if new_g1s else 0.0

    aggregate: List[Dict[str, Any]] = []
    for arm in arms:
        arm_sels = [s for s in selections if s["arm"] == arm]
        g1s = [s["G1_aligned"] for s in arm_sels]
        ctrl_median = new_medians.get("Control", 0.0)
        arm_median = statistics.median(g1s) if g1s else 0.0
        delta_pct = (1 - arm_median / ctrl_median) * 100 if ctrl_median > 0 else 0.0

        spike_deltas = [abs(s["g1_spike_delta"]) for s in arm_sels]
        n_tf = sum(1 for s in arm_sels if "TOPO_FAIL" in s["selection_mode"])
        tf_pct = n_tf / len(arm_sels) * 100 if arm_sels else 0

        aggregate.append({
            "arm": arm,
            "median_G1_selected": round(arm_median, 6),
            "median_G1_old": round(old_medians.get(arm, 0.0), 6),
            "delta_vs_control_pct": round(delta_pct, 1),
            "count_clean": sum(1 for s in arm_sels if s["selection_mode"] == "CLEAN_MAX_TG"),
            "count_leaky": sum(1 for s in arm_sels if s["selection_mode"] == "LEAKY_MIN_G1ROLL"),
            "count_fallback": n_tf,
            "topo_fail_pct": round(tf_pct, 1),
            "median_g1_spike_delta": round(statistics.median(spike_deltas), 6) if spike_deltas else 0.0,
        })

    dual_cap_violations = [
        s for s in selections
        if s["selection_mode"] == "CLEAN_MAX_TG" and s["G1_aligned"] > tau_clean_hi
    ]

    report = {
        "day": "73",
        "title": "Selector v6 (Smoothed Topology Floors)",
        "version": "v6",
        "policy": {
            "tau_clean": tau_clean, "tau_clean_hi": tau_clean_hi,
            "default_slice_floor": default_slice_floor,
            "tg_floor": tg_floor,
            "active_epoch_min": active_epoch_min, "roll_window": roll_window,
            "drop_slice_floor": drop_slice_floor,
        },
        "artifact_dir_used": str(artifact_dir),
        "per_seed_selections": selections,
        "aggregate": aggregate,
        "old_medians": {k: round(v, 6) for k, v in old_medians.items()},
        "new_medians": {k: round(v, 6) for k, v in new_medians.items()},
        "dual_cap_violations": len(dual_cap_violations),
    }

    json_path = out_dir / "selection_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    csv_path = out_dir / "selection_table.csv"
    _write_csv(selections, csv_path)

    md_path = out_dir / "selection_report.md"
    _write_markdown(report, md_path)

    cksum_path = out_dir / "checksums.sha256"
    _write_checksums([json_path, csv_path, md_path], cksum_path)

    return report


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_csv(selections: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "seed", "arm", "epoch", "G1_aligned", "g1_roll", "g1_spike_delta",
        "slice_clean", "slice_clean_roll", "topo_TG", "tg_roll",
        "mean_drop", "loss", "selection_mode",
        "n_active", "n_surviving", "n_clean", "n_leaky", "slice_floor_used",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for s in sorted(selections, key=lambda x: (x["seed"], x["arm"])):
            row = {k: s.get(k, "") for k in fieldnames}
            for k in ["G1_aligned", "g1_roll", "g1_spike_delta",
                       "slice_clean", "slice_clean_roll", "topo_TG", "tg_roll",
                       "mean_drop", "loss", "slice_floor_used"]:
                if row[k] is not None and row[k] != "":
                    try:
                        row[k] = f"{float(row[k]):.6f}"
                    except (ValueError, TypeError):
                        pass
            writer.writerow(row)


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    p = report["policy"]
    lines = [
        "# Selector v6 Report",
        "",
        f"**Policy**: slice_floor(default)={p['default_slice_floor']}, "
        f"tg_floor={p['tg_floor']}, "
        f"tau_clean={p['tau_clean']}, tau_clean_hi={p['tau_clean_hi']}, "
        f"roll_window={p['roll_window']}, "
        f"drop_slice_floor={p['drop_slice_floor']}",
        "",
        f"**Dual-cap violations**: {report['dual_cap_violations']}",
        "",
        "## Aggregate Summary",
        "",
        "| Arm | Med G1(old) | Med G1(sel) | Δ vs Ctrl "
        "| Clean | Leaky | Fall | TF% | spike |",
        "|-----|-------------|------------|----------"
        "|-------|-------|------|-----|-------|",
    ]
    for agg in report["aggregate"]:
        lines.append(
            f"| {agg['arm']} | {agg['median_G1_old']:.4f} "
            f"| {agg['median_G1_selected']:.4f} "
            f"| {agg['delta_vs_control_pct']:+.1f}% | {agg['count_clean']} "
            f"| {agg['count_leaky']} | {agg['count_fallback']} "
            f"| {agg['topo_fail_pct']:.0f}% "
            f"| {agg['median_g1_spike_delta']:.4f} |"
        )

    lines += [
        "",
        "## Per-Seed Selections",
        "",
        "| Seed | Arm | Ep | G1 | g1roll | spike | SC | SC_roll | TG | TG_roll "
        "| floor | nSurv | Mode |",
        "|------|-----|----|----|--------|-------|----|---------|----|--------"
        "|-------|-------|------|",
    ]
    for s in sorted(report["per_seed_selections"],
                    key=lambda x: (x["seed"], x["arm"])):
        tg = f"{s['topo_TG']:.4f}" if s["topo_TG"] is not None else "N/A"
        lines.append(
            f"| {s['seed']} | {s['arm']} | {s['epoch']} "
            f"| {s['G1_aligned']:.4f} | {s['g1_roll']:.4f} "
            f"| {s['g1_spike_delta']:+.4f} "
            f"| {s['slice_clean']:.4f} | {s['slice_clean_roll']:.4f} "
            f"| {tg} | {s['tg_roll']:.4f} "
            f"| {s['slice_floor_used']:.4f} | {s['n_surviving']} "
            f"| {s['selection_mode']} |"
        )

    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_checksums(files: List[Path], cksum_path: Path) -> None:
    lines = []
    for fp in sorted(files):
        lines.append(f"{_sha256_file(fp)}  {fp.name}")
    with open(cksum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
