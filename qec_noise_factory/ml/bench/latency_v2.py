"""
Latency Benchmark v2 — Day 29

Multi-decoder latency comparison with fair measurement:
- MWPM (per-sample via PyMatching)
- GNN v2 (batched with edge_weight)
- MLP (batched, optional)

Measures warmup + timed inference, reports per-sample and throughput.
Supports CLI via __main__.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def measure_decoder_latency(
    decode_fn,
    X: np.ndarray,
    *,
    warmup: int = 256,
    n_measure: Optional[int] = None,
    batch_size: int = 1,
    decoder_name: str = "unknown",
) -> Dict[str, Any]:
    """
    Measure per-sample inference latency for one decoder.

    Warmup is NOT counted in timing statistics.
    Returns dict with mean_ms, median_ms, p95_ms, throughput_sps, etc.
    """
    N = X.shape[0]
    warmup_n = min(warmup, N // 2)

    # Warmup phase
    if warmup_n > 0:
        decode_fn(X[:warmup_n])

    # Measure phase
    if n_measure is None:
        n_measure = N - warmup_n
    m_start = warmup_n
    m_end = min(m_start + n_measure, N)
    actual_n = m_end - m_start

    if actual_n <= 0:
        return _empty_result(decoder_name)

    latencies = []
    idx = m_start
    while idx < m_end:
        end_idx = min(idx + batch_size, m_end)
        batch = X[idx:end_idx]
        bs = len(batch)
        t0 = time.perf_counter()
        decode_fn(batch)
        t1 = time.perf_counter()
        per_sample = (t1 - t0) / bs
        latencies.extend([per_sample] * bs)
        idx = end_idx

    lat_ms = np.array(latencies) * 1000.0
    total_s = lat_ms.sum() / 1000.0

    return {
        "decoder": decoder_name,
        "mean_ms": float(np.mean(lat_ms)),
        "median_ms": float(np.median(lat_ms)),
        "p95_ms": float(np.percentile(lat_ms, 95)),
        "min_ms": float(np.min(lat_ms)),
        "max_ms": float(np.max(lat_ms)),
        "std_ms": float(np.std(lat_ms)),
        "throughput_sps": actual_n / total_s if total_s > 0 else 0.0,
        "total_samples": actual_n,
        "total_time_s": round(total_s, 6),
        "batch_size_used": batch_size,
        "warmup_samples": warmup_n,
    }


def _empty_result(name: str) -> Dict[str, Any]:
    return {
        "decoder": name,
        "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0,
        "min_ms": 0.0, "max_ms": 0.0, "std_ms": 0.0,
        "throughput_sps": 0.0, "total_samples": 0,
        "total_time_s": 0.0, "batch_size_used": 0, "warmup_samples": 0,
    }


# ---------------------------------------------------------------------------
# GNN v2 latency wrapper
# ---------------------------------------------------------------------------

def make_gnn_v2_decode_fn(model, edge_index, num_nodes, feature_dim,
                          edge_weight=None):
    """Create a decode function for GNN v2 that can be passed to measure_decoder_latency."""
    import torch

    model.eval()
    device = next(model.parameters()).device

    def _decode(batch_X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(batch_X, dtype=torch.float32).to(device)
            B = x_t.shape[0]
            x_t = x_t[:, :num_nodes * feature_dim].reshape(B, num_nodes, feature_dim)
            kwargs = {}
            if edge_weight is not None:
                kwargs["edge_weight"] = edge_weight.to(device)
            logits = model(x_t, edge_index.to(device), **kwargs)
            return (logits > 0).cpu().numpy()

    return _decode


# ---------------------------------------------------------------------------
# MLP latency wrapper
# ---------------------------------------------------------------------------

def make_mlp_decode_fn(model):
    """Create a decode function for MLP model."""
    import torch

    model.eval()
    device = next(model.parameters()).device

    def _decode(batch_X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(batch_X, dtype=torch.float32).to(device)
            logits = model(x_t)
            return (logits > 0).cpu().numpy()

    return _decode


# ---------------------------------------------------------------------------
# Multi-decoder runner
# ---------------------------------------------------------------------------

def run_latency_comparison(
    decoders: Dict[str, Any],
    X: np.ndarray,
    *,
    warmup: int = 256,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run latency benchmark for multiple decoders on the same data.

    Args:
        decoders: dict of {name: {"decode_fn": callable, "batch_size": int}}
        X: input data (N, D)
        warmup: warmup samples
        num_samples: samples to measure (default: all after warmup)
        seed: for reproducibility

    Returns:
        dict with per-decoder results and metadata
    """
    from qec_noise_factory.ml.bench.latency import _get_device_info

    rng = np.random.RandomState(seed)
    if num_samples and num_samples < X.shape[0]:
        idx = rng.choice(X.shape[0], num_samples, replace=False)
        idx.sort()
        X_measure = X[idx]
    else:
        X_measure = X

    results = {}
    for name, cfg in decoders.items():
        decode_fn = cfg["decode_fn"]
        bs = cfg.get("batch_size", 1)
        r = measure_decoder_latency(
            decode_fn, X_measure,
            warmup=warmup, batch_size=bs,
            decoder_name=name,
        )
        results[name] = r

    return {
        "decoders": results,
        "num_samples_total": X_measure.shape[0],
        "warmup": warmup,
        "seed": seed,
        "device_info": _get_device_info(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Latency Benchmark v2 — Day 29")
    parser.add_argument("--data-root", type=str, default="output/data/surface_v0/shards",
                        help="Root directory for shard data")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--basis", type=str, default="X")
    parser.add_argument("--decoders", type=str, default="mwpm,gnn_v2",
                        help="Comma-separated decoder list: mwpm,gnn_v2,mlp")
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--out", type=str, default="ml_artifacts/day29_unified_bench")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from qec_noise_factory.ml.data.reader import read_shards_dir, merge_datasets

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / args.data_root / f"d{args.distance}"

    print(f"Loading d={args.distance} shards from {data_dir}...")
    datasets = read_shards_dir(data_dir)
    merged = merge_datasets(datasets)
    X, Y = merged.X, merged.Y
    print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} detectors")

    decoder_list = [d.strip() for d in args.decoders.split(",")]
    decoder_configs = {}

    first_meta = merged.meta[0]

    if "mwpm" in decoder_list:
        from qec_noise_factory.ml.bench.mwpm_decoder import MWPMDecoder
        mwpm = MWPMDecoder()
        mwpm.build_from_meta(first_meta.params_canonical)
        decoder_configs["mwpm"] = {
            "decode_fn": mwpm.decode_batch_fast,
            "batch_size": X.shape[0],  # MWPM processes full batch
        }

    # GNN v2 and MLP require trained models — skip in CLI for now,
    # they're used programmatically from unified_benchmark_v2.py
    if "gnn_v2" in decoder_list and "gnn_v2" not in decoder_configs:
        print("  [NOTE] GNN v2 latency requires a trained model; use unified_benchmark_v2.py")

    if "mlp" in decoder_list and "mlp" not in decoder_configs:
        print("  [NOTE] MLP latency requires a trained model; use unified_benchmark_v2.py")

    if not decoder_configs:
        print("No decoders configured, exiting.")
        return

    results = run_latency_comparison(
        decoder_configs, X,
        warmup=args.warmup, num_samples=args.num_samples, seed=args.seed,
    )

    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "latency_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nLatency results: {out_path}")

    cs_path = out_dir / "latency_checksums.sha256"
    with open(cs_path, "w") as f:
        f.write(f"{_sha256_file(out_path)}  latency_v2.json\n")
    print(f"Checksums: {cs_path}")

    # Print summary
    print(f"\n{'Decoder':<20} {'Mean(ms)':<12} {'Median(ms)':<12} {'P95(ms)':<12} {'Throughput':<12}")
    print("-" * 68)
    for name, r in results["decoders"].items():
        print(f"{name:<20} {r['mean_ms']:<12.3f} {r['median_ms']:<12.3f} "
              f"{r['p95_ms']:<12.3f} {r['throughput_sps']:<12.0f}")


if __name__ == "__main__":
    main()
