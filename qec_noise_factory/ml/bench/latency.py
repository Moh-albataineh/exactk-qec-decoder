"""
Latency Harness — Day 26

Measures inference latency with proper warmup, timing, and reporting.
Reports mean/median/p95 ms/sample and throughput (samples/sec).
"""
from __future__ import annotations

import platform
import time
from typing import Any, Callable, Dict, Optional

import numpy as np


def measure_latency(
    decode_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    warmup: int = 200,
    n_measure: Optional[int] = None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Measure inference latency for a decoder function.

    Args:
        decode_fn: function that takes X and returns predictions
        X: (N, D) input array
        warmup: number of warmup samples (not timed)
        n_measure: number of samples to measure (default: all after warmup)
        batch_size: batch size for decode_fn calls

    Returns:
        dict with: mean_ms, median_ms, p95_ms, throughput_sps,
                   total_samples, total_time_s, device_info
    """
    N = X.shape[0]

    # Warmup
    warmup_n = min(warmup, N)
    if warmup_n > 0:
        decode_fn(X[:warmup_n])

    # Measure
    if n_measure is None:
        n_measure = N - warmup_n
    measure_start = warmup_n
    measure_end = min(measure_start + n_measure, N)
    actual_n = measure_end - measure_start

    if actual_n <= 0:
        return {
            "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0,
            "throughput_sps": 0.0, "total_samples": 0, "total_time_s": 0.0,
        }

    # Per-batch timing
    latencies = []
    idx = measure_start
    while idx < measure_end:
        end_idx = min(idx + batch_size, measure_end)
        batch = X[idx:end_idx]
        t0 = time.perf_counter()
        decode_fn(batch)
        t1 = time.perf_counter()
        batch_time = t1 - t0
        per_sample = batch_time / len(batch)
        for _ in range(len(batch)):
            latencies.append(per_sample)
        idx = end_idx

    latencies = np.array(latencies) * 1000  # convert to ms

    total_time = latencies.sum() / 1000  # back to seconds

    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies)),
        "throughput_sps": actual_n / total_time if total_time > 0 else 0.0,
        "total_samples": actual_n,
        "total_time_s": total_time,
        "batch_size": batch_size,
        "device_info": _get_device_info(),
    }


def measure_gnn_latency(
    model,
    X: np.ndarray,
    edge_index,
    num_nodes: int,
    feature_dim: int,
    warmup: int = 100,
    n_measure: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Measure GNN inference latency (forward pass only, no dataloader overhead).

    Args:
        model: GNN model (in eval mode)
        X: (N, num_detectors, feature_dim) or (N, num_detectors) array
        edge_index: graph edge index tensor
        num_nodes: number of graph nodes
        feature_dim: feature dimension
        warmup: warmup samples
        n_measure: samples to time
    """
    import torch

    model.eval()
    device = next(model.parameters()).device

    def _decode_batch(batch_X):
        with torch.no_grad():
            x_t = torch.tensor(batch_X, dtype=torch.float32).to(device)
            if x_t.ndim == 2:
                # (batch, det) -> (batch, nodes, feat_dim)
                B = x_t.shape[0]
                x_t = x_t[:, :num_nodes * feature_dim].reshape(B, num_nodes, feature_dim)
            logits = model(x_t, edge_index)
            return (logits > 0).cpu().numpy()

    return measure_latency(_decode_batch, X, warmup=warmup, n_measure=n_measure, batch_size=32)


def _get_device_info() -> Dict[str, str]:
    """Get CPU/device info."""
    info = {
        "cpu": platform.processor() or "unknown",
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        else:
            info["gpu"] = "N/A (CPU only)"
    except ImportError:
        pass
    return info
