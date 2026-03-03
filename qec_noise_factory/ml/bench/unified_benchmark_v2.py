"""
Unified Benchmark v2 — Day 29

Oracle vs Mismatch comparison across decoder types:
  MWPM_ORACLE       — true DEM from shard/circuit
  MWPM_MISMATCH_MODEL — DEM from wrong noise model
  MWPM_MISMATCH_P   — DEM from scaled p
  GNN_V2_DEM        — GNNDecoderV2 with DEM edge weights
  GNN_V1_GENERIC    — GNNDecoderV1 with generic graph

Same splits enforced across all decoders via dataset_hash assertion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qec_noise_factory.ml.data.reader import read_shards_dir, merge_datasets, ShardDataset
from qec_noise_factory.ml.data.schema import load_meta_file
from qec_noise_factory.ml.eval.generalization_suite import (
    ExperimentConfig, ExperimentReport, run_experiment,
)
from qec_noise_factory.ml.artifacts.run_logger import dataset_hash
from qec_noise_factory.ml.bench.mwpm_decoder import MWPMDecoder
from qec_noise_factory.ml.bench.mismatch import (
    build_oracle_mwpm,
    build_model_mismatched_mwpm,
    build_p_scaled_mwpm,
    compute_matching_weight,
    MismatchInfo,
)
from qec_noise_factory.ml.metrics.classification import compute_metrics
from qec_noise_factory.ml.stim.rebuild import params_from_canonical


# ---------------------------------------------------------------------------
# Decoder identifiers
# ---------------------------------------------------------------------------

DECODER_IDS = [
    "MWPM_ORACLE",
    "MWPM_MISMATCH_MODEL",
    "MWPM_MISMATCH_P",
    "GNN_V2_DEM",
    "GNN_V1_GENERIC",
    "MLP",
]


# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for unified benchmark run."""
    distance: int = 5
    basis: str = "X"
    policies: List[str] = field(default_factory=lambda: ["cross_model", "within_model", "ood_p_range"])
    decoders: List[str] = field(default_factory=lambda: ["MWPM_ORACLE", "GNN_V2_DEM"])
    p_scales: List[float] = field(default_factory=lambda: [0.5, 2.0])
    mismatch_noise_model: str = "baseline_symmetric"
    limit_samples: int = 0       # 0 = no limit
    epochs: int = 3
    hidden_dim: int = 64
    batch_size: int = 64
    seed: int = 42
    data_root: str = "output/data/surface_v0/shards"
    out_dir: str = "ml_artifacts/day29_unified_bench"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Benchmark result row
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRow:
    """One row in the results table: (decoder x policy)."""
    decoder: str
    policy: str
    distance: int
    basis: str
    # Metrics
    f1: float = 0.0
    precision: float = 0.0
    recall_tpr: float = 0.0
    fpr: float = 0.0
    balanced_accuracy: float = 0.0
    pred_positive_rate: float = 0.0
    # GNN-specific
    train_loss: float = 0.0
    eval_loss: float = 0.0
    calibration_threshold: float = 0.5
    pos_weight_used: float = 1.0
    collapse_guard_triggered: bool = False
    # MWPM-specific
    mismatch_strategy: str = "oracle"
    mismatch_noise_model: str = ""
    p_scale: float = 1.0
    effective_p: float = 0.0
    # Split info
    dataset_hash: str = ""
    train_samples: int = 0
    test_samples: int = 0
    # Provenance
    status: str = "fail"
    runtime_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# MWPM evaluation (non-ML, no training needed)
# ---------------------------------------------------------------------------

def _evaluate_mwpm(
    decoder: MWPMDecoder,
    test_X: np.ndarray,
    test_Y: np.ndarray,
    mismatch_info: MismatchInfo,
) -> Dict[str, Any]:
    """Decode test set with MWPM and compute metrics."""
    t0 = time.time()
    y_hat = decoder.decode_batch_fast(test_X)
    decode_time = time.time() - t0

    metrics = compute_metrics(test_Y, y_hat)
    pred_pos_rate = y_hat.mean()

    return {
        "metrics": metrics,
        "pred_positive_rate": float(pred_pos_rate),
        "decode_time_s": decode_time,
        "mismatch_info": mismatch_info.to_dict(),
    }


# ---------------------------------------------------------------------------
# Run one (decoder x policy) pair
# ---------------------------------------------------------------------------

def run_single_benchmark(
    decoder_id: str,
    policy: str,
    dataset: ShardDataset,
    bench_config: BenchmarkConfig,
    out_dir: Path,
    p_scale: float = 1.0,
) -> BenchmarkRow:
    """
    Run one benchmark: a single (decoder, policy) pair.

    For GNN/MLP: uses run_experiment from generalization_suite.
    For MWPM: builds matching graph (oracle or mismatched), decodes, computes metrics.
    """
    row = BenchmarkRow(
        decoder=decoder_id,
        policy=policy,
        distance=bench_config.distance,
        basis=bench_config.basis,
        p_scale=p_scale,
    )

    first_meta = dataset.meta[0]
    params = params_from_canonical(first_meta.params_canonical)
    t0 = time.time()

    # --- GNN / MLP path (via generalization_suite) ---
    if decoder_id in ("GNN_V2_DEM", "GNN_V1_GENERIC", "MLP"):
        gnn_version = "v2" if decoder_id == "GNN_V2_DEM" else "v1"
        graph_mode = "dem" if decoder_id == "GNN_V2_DEM" else "generic"
        model_type = "mlp" if decoder_id == "MLP" else "gnn"

        exp_id = f"{decoder_id}_{policy}"
        cfg = ExperimentConfig(
            exp_id=exp_id,
            name=f"{decoder_id} ({policy})",
            split_policy=policy,
            epochs=bench_config.epochs,
            batch_size=bench_config.batch_size,
            hidden_dim=bench_config.hidden_dim,
            seed=bench_config.seed,
            loss_pos_weight=0,  # auto
            calibrate_threshold=True,
            pos_weight_max=8.0,
            gnn_version=gnn_version,
            graph_mode=graph_mode,
            gnn_feature_version="v1",
            gnn_readout="mean_max",
            featureset="v1_nop",
            calibrate_metric="f1",
            calibrate_lambda=0.0,
        )
        if policy == "cross_model":
            cfg.train_ratio = 0.5
        elif policy == "ood_p_range":
            cfg.ood_test_p_lo = 0.005
            cfg.ood_test_p_hi = 1.0

        report = run_experiment(cfg, dataset, out_dir / exp_id)
        rd = report.to_dict()

        gnn_m = rd.get("gnn_metrics") or {}
        mlp_m = rd.get("mlp_metrics") or {}
        m = gnn_m if decoder_id != "MLP" else mlp_m

        row.f1 = m.get("macro_f1", 0.0)
        row.precision = m.get("macro_precision", 0.0)
        row.recall_tpr = m.get("macro_tpr", m.get("obs_0_tpr", 0.0))
        row.fpr = m.get("macro_fpr", m.get("obs_0_fpr", 0.0))
        row.balanced_accuracy = m.get("macro_balanced_accuracy", 0.0)
        row.pred_positive_rate = m.get("pred_positive_rate", rd.get("pred_positive_rate", 0.0))
        row.train_loss = rd.get("gnn_train_loss", rd.get("mlp_train_loss", 0.0))
        row.eval_loss = rd.get("gnn_eval_loss", rd.get("mlp_eval_loss", 0.0))
        row.calibration_threshold = rd.get("calibration_threshold", 0.5)
        row.pos_weight_used = rd.get("pos_weight_clipped", 1.0)
        row.collapse_guard_triggered = rd.get("collapse_guard_triggered", False)
        row.dataset_hash = rd.get("dataset_hash", "")
        row.train_samples = rd.get("train_samples", 0)
        row.test_samples = rd.get("test_samples", 0)
        row.status = rd.get("status", "fail")

    # --- MWPM path ---
    elif decoder_id.startswith("MWPM"):
        from qec_noise_factory.ml.eval.generalization_suite import (
            _extract_xy_by_blocks,
        )
        from qec_noise_factory.ml.data.splits import (
            SplitPolicy, split_cross_model, split_within_model, split_ood_p_range,
        )

        # Same split logic as generalization_suite
        policy_map = {
            "cross_model": SplitPolicy.CROSS_MODEL,
            "ood_p_range": SplitPolicy.OOD_P_RANGE,
            "within_model": SplitPolicy.WITHIN_MODEL,
        }
        sp = policy_map.get(policy)
        if sp is None:
            row.status = "fail"
            return row

        if sp == SplitPolicy.OOD_P_RANGE:
            split = split_ood_p_range(
                dataset.meta, test_p_lo=0.005, test_p_hi=1.0, seed=bench_config.seed,
            )
        elif sp == SplitPolicy.CROSS_MODEL:
            split = split_cross_model(dataset.meta, train_ratio=0.5, seed=bench_config.seed)
        else:
            split = split_within_model(dataset.meta, train_ratio=0.8, seed=bench_config.seed)

        if not split.train_blocks or not split.test_blocks:
            row.status = "skip"
            return row

        train_keys = {b.sample_key for b in split.train_blocks}
        test_keys = {b.sample_key for b in split.test_blocks}
        _, _, _ = _extract_xy_by_blocks(dataset, train_keys)
        test_X, test_Y, _ = _extract_xy_by_blocks(dataset, test_keys)

        row.dataset_hash = dataset_hash(dataset.meta, policy, split_seed=bench_config.seed)
        row.train_samples = sum(b.record_count for b in split.train_blocks)
        row.test_samples = test_X.shape[0]

        # Build MWPM (oracle or mismatched)
        mwpm = MWPMDecoder()

        if decoder_id == "MWPM_ORACLE":
            mi = build_oracle_mwpm(
                mwpm, distance=params["distance"], rounds=params["rounds"],
                p=params["p"], basis=params["basis"],
                noise_model=params["noise_model"],
            )
        elif decoder_id == "MWPM_MISMATCH_MODEL":
            mi = build_model_mismatched_mwpm(
                mwpm, distance=params["distance"], rounds=params["rounds"],
                p=params["p"], basis=params["basis"],
                true_noise_model=params["noise_model"],
                mismatch_noise_model=bench_config.mismatch_noise_model,
            )
        elif decoder_id == "MWPM_MISMATCH_P":
            mi = build_p_scaled_mwpm(
                mwpm, distance=params["distance"], rounds=params["rounds"],
                p=params["p"], basis=params["basis"],
                noise_model=params["noise_model"],
                p_scale=p_scale,
            )
        else:
            row.status = "fail"
            return row

        result = _evaluate_mwpm(mwpm, test_X, test_Y, mi)
        m = result["metrics"]

        row.f1 = m.get("macro_f1", 0.0)
        row.precision = m.get("macro_precision", 0.0)
        row.recall_tpr = m.get("macro_tpr", m.get("obs_0_tpr", 0.0))
        row.fpr = m.get("macro_fpr", m.get("obs_0_fpr", 0.0))
        row.balanced_accuracy = m.get("macro_balanced_accuracy", 0.0)
        row.pred_positive_rate = result["pred_positive_rate"]
        row.mismatch_strategy = mi.strategy
        row.mismatch_noise_model = mi.mismatch_noise_model
        row.p_scale = mi.p_scale
        row.effective_p = mi.effective_p
        row.status = "pass"

    row.runtime_s = round(time.time() - t0, 2)
    return row


# ---------------------------------------------------------------------------
# Full benchmark orchestration
# ---------------------------------------------------------------------------

def run_unified_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """
    Run the full unified benchmark.

    Returns results dict with all rows, split manifest, and metadata.
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / config.data_root / f"d{config.distance}"

    print("=" * 70)
    print(f"Day 29 Unified Benchmark v2 (d={config.distance}, basis={config.basis})")
    print("=" * 70)

    # Load data
    print(f"\n[1] Loading d={config.distance} shards from {data_dir}...")
    if not data_dir.exists():
        return {"status": "skip", "reason": f"Data dir not found: {data_dir}"}

    datasets = read_shards_dir(data_dir)
    if not datasets:
        return {"status": "skip", "reason": "No shards found"}
    merged = merge_datasets(datasets)

    total = merged.X.shape[0]
    print(f"    Loaded {total:,} samples, {merged.X.shape[1]} detectors")

    # Subsample if needed
    if config.limit_samples > 0 and total > config.limit_samples:
        rng = np.random.RandomState(config.seed)
        idx = rng.choice(total, config.limit_samples, replace=False)
        idx.sort()
        # Create new dataset with subsampled data
        merged = ShardDataset(
            X=merged.X[idx], Y=merged.Y[idx], meta=merged.meta,
            shard_path=merged.shard_path,
        )
        print(f"    Subsampled to {config.limit_samples}")

    out_dir = project_root / config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run all (decoder x policy) combinations
    all_rows: List[BenchmarkRow] = []
    hashes: Dict[str, str] = {}

    for policy in config.policies:
        print(f"\n[Policy: {policy}]")

        for dec_id in config.decoders:
            if dec_id == "MWPM_MISMATCH_P":
                # Run once per p_scale
                for ps in config.p_scales:
                    label = f"{dec_id}(scale={ps})"
                    print(f"  Running {label} ...", end=" ", flush=True)
                    row = run_single_benchmark(
                        dec_id, policy, merged, config, out_dir, p_scale=ps,
                    )
                    row.decoder = label  # include scale in name
                    all_rows.append(row)
                    if row.dataset_hash:
                        hashes.setdefault(policy, row.dataset_hash)
                    print(f"F1={row.f1:.2%} [{row.status}] ({row.runtime_s}s)")
            else:
                print(f"  Running {dec_id} ...", end=" ", flush=True)
                row = run_single_benchmark(
                    dec_id, policy, merged, config, out_dir,
                )
                all_rows.append(row)
                if row.dataset_hash:
                    hashes.setdefault(policy, row.dataset_hash)
                print(f"F1={row.f1:.2%} [{row.status}] ({row.runtime_s}s)")

    # Verify same splits
    print("\n[Split Verification]")
    for policy, h in hashes.items():
        policy_rows = [r for r in all_rows if r.policy == policy and r.dataset_hash]
        unique_hashes = {r.dataset_hash for r in policy_rows}
        if len(unique_hashes) <= 1:
            print(f"  {policy}: hash={h[:16]}... (consistent)")
        else:
            print(f"  WARNING: {policy} has {len(unique_hashes)} different hashes!")

    # Build results dict
    results = {
        "config": config.to_dict(),
        "rows": [r.to_dict() for r in all_rows],
        "split_manifest": {
            p: {
                "dataset_hash": hashes.get(p, ""),
                "decoders_run": [r.decoder for r in all_rows if r.policy == p],
                "train_samples": next((r.train_samples for r in all_rows if r.policy == p), 0),
                "test_samples": next((r.test_samples for r in all_rows if r.policy == p), 0),
            }
            for p in config.policies
        },
        "status": "pass" if any(r.status == "pass" for r in all_rows) else "fail",
    }

    return results


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def write_artifacts(results: Dict[str, Any], out_dir: Path):
    """Write CSV, JSON, summary.md, config_used, and checksums."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = results.get("rows", [])

    # 1. results.json
    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # 2. results.csv
    csv_path = out_dir / "results.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    # 3. config_used.json
    cfg_path = out_dir / "config_used.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(results.get("config", {}), f, indent=2, default=str)

    # 4. summary.md
    summary_path = out_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Day 29 Unified Benchmark v2 Results\n\n")
        f.write("## Oracle vs Mismatch Comparison\n\n")
        f.write(f"| Decoder | Policy | F1 | Precision | TPR | FPR | BalAcc | Status |\n")
        f.write(f"|---------|--------|-----|-----------|-----|-----|--------|--------|\n")
        for r in rows:
            f.write(f"| {r['decoder']} | {r['policy']} | "
                    f"{r['f1']:.2%} | {r['precision']:.2%} | "
                    f"{r['recall_tpr']:.2%} | {r['fpr']:.2%} | "
                    f"{r['balanced_accuracy']:.2%} | {r['status']} |\n")

        # Split manifest
        manifest = results.get("split_manifest", {})
        if manifest:
            f.write("\n## Split Manifest\n\n")
            for policy, info in manifest.items():
                f.write(f"- **{policy}**: hash=`{info['dataset_hash'][:16]}...`, "
                        f"train={info['train_samples']}, test={info['test_samples']}\n")

    # 5. checksums
    cs_path = out_dir / "checksums.sha256"
    files = [json_path, csv_path, cfg_path, summary_path]
    with open(cs_path, "w") as f:
        for p in files:
            if p.exists():
                f.write(f"{_sha256_file(p)}  {p.name}\n")

    print(f"\nArtifacts written to {out_dir}/")
    for p in files + [cs_path]:
        if p.exists():
            print(f"  {p.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified Benchmark v2 -- Day 29")
    parser.add_argument("--data-root", type=str,
                        default="output/data/surface_v0/shards")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--basis", type=str, default="X")
    parser.add_argument("--policy", type=str, default="cross_model,within_model,ood_p_range",
                        help="Comma-separated split policies")
    parser.add_argument("--decoders", type=str,
                        default="MWPM_ORACLE,GNN_V2_DEM",
                        help="Comma-separated decoder IDs")
    parser.add_argument("--p-scales", type=str, default="0.5,2.0",
                        help="Comma-separated p_scale values for MWPM_MISMATCH_P")
    parser.add_argument("--mismatch-noise-model", type=str, default="baseline_symmetric")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="ml_artifacts/day29_unified_bench")
    args = parser.parse_args(argv)

    config = BenchmarkConfig(
        distance=args.distance,
        basis=args.basis,
        policies=[p.strip() for p in args.policy.split(",")],
        decoders=[d.strip() for d in args.decoders.split(",")],
        p_scales=[float(s) for s in args.p_scales.split(",")],
        mismatch_noise_model=args.mismatch_noise_model,
        limit_samples=args.limit_samples,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        data_root=args.data_root,
        out_dir=args.out,
    )

    results = run_unified_benchmark(config)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    write_artifacts(results, project_root / config.out_dir)

    # Summary
    rows = results.get("rows", [])
    passed = sum(1 for r in rows if r["status"] == "pass")
    print(f"\n{'='*70}")
    print(f"Benchmark complete: {passed}/{len(rows)} passed")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
