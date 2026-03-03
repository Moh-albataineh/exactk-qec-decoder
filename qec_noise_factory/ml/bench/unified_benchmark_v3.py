"""
Unified Benchmark v3 — Day 30 + Day 31 + Day 34 Extensions

Mismatch & Correlated Noise Supremacy Benchmark:

  Suite A — ORACLE:        Baseline noise, correct DEM, GNN training
  Suite B — P_SWEEP:       Same data, MWPM with wide p_scale sweep
  Suite C — MODEL_MIS:     Baseline data, decode with DEM from different models
  Suite D — CORRELATED:    Correlated noise data, MWPM vs GNN (original)
  Suite D_v2 — CORRELATED: Per-p evaluation with informativeness gates (Day 31)

Same splits enforced across all decoders within each suite.

INDEX — Table of Contents (2168 lines)
=======================================
  L60   BenchV3Config         — Configuration dataclass
  L110  BenchV3Row            — Result row dataclass
  L144  _load_dataset         — Load and merge shards
  L159  _subsample            — Subsample dataset
  L170  _run_mwpm             — MWPM decoder runner
  L290  _run_factor_graph     — FG v0/v1 decoder (train + eval + BRQL/F0.5 calibration)
  L765  _run_gnn              — GNN V2 DEM decoder runner
  L821  run_suite_a           — Suite A (Oracle)
  L848  run_suite_b           — Suite B (P-Sweep)
  L869  run_suite_c           — Suite C (Model Mismatch)
  L896  run_suite_d           — Suite D (Correlated, original)
  L924  run_suite_d_v2        — Suite D v2 (Per-p correlated + diagnostics)
  L1365 run_quality_gates     — Global quality gates
  L1463 run_fg_gates          — FG-specific quality gates (Day 33.6)
  L1552 InformativenessGate   — Tri-state gate dataclass
  L1565 run_informativeness_gates — Informativeness gates (Day 31)
  L1834 _sha256_file          — Hash helper
  L1841 write_v3_artifacts    — Write all benchmark artifacts
  L1938 _measure_inference_latency — MWPM latency measurement
  L1990 run_benchmark_v3      — Main orchestrator
  L2111 main                  — CLI entry point
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from qec_noise_factory.ml.data.reader import read_shards_dir, merge_datasets, ShardDataset
from qec_noise_factory.ml.eval.generalization_suite import (
    ExperimentConfig, run_experiment,
)
from qec_noise_factory.ml.artifacts.run_logger import dataset_hash
from qec_noise_factory.ml.bench.mwpm_decoder import MWPMDecoder
from qec_noise_factory.ml.bench.mismatch import (
    build_oracle_mwpm,
    build_model_mismatched_mwpm,
    build_p_scaled_mwpm,
    MismatchInfo,
)
from qec_noise_factory.ml.metrics.classification import compute_metrics
from qec_noise_factory.ml.stim.rebuild import params_from_canonical
from qec_noise_factory.ml.graph.dem_graph import build_dem_graph, dem_graph_stats, dem_corr_stats
from qec_noise_factory.ml.data.reader import filter_by_p_range
from qec_noise_factory.ml.bench import reason_codes as RC


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUITE_NAMES = ["A_ORACLE", "B_P_SWEEP", "C_MODEL_MIS", "D_CORRELATED"]
SUITE_NAMES_V2 = SUITE_NAMES + ["D_CORRELATED_V2"]

DEFAULT_P_SCALES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0]
DEFAULT_MISMATCH_MODELS = ["sd6_like", "si1000_like", "biased_z"]
DEFAULT_MISMATCH_P_VALUES = [0.001, 0.005, 0.01]


@dataclass
class BenchV3Config:
    """Configuration for Day 30 benchmark v3."""
    suites: List[str] = field(default_factory=lambda: list(SUITE_NAMES))
    distance: int = 5
    basis: str = "X"
    # Suite A/B/C
    data_root: str = "output/data/surface_v0/shards"
    # Suite D
    corr_data_root: str = "output/data/surface_corr_v0/shards"
    # Training
    epochs: int = 15
    smoke_epochs: int = 3
    hidden_dim: int = 64
    batch_size: int = 64
    seed: int = 42
    limit_samples: int = 0
    # Suite B
    p_scales: List[float] = field(default_factory=lambda: list(DEFAULT_P_SCALES))
    # Suite C
    mismatch_models: List[str] = field(default_factory=lambda: list(DEFAULT_MISMATCH_MODELS))
    mismatch_p_values: List[float] = field(default_factory=lambda: list(DEFAULT_MISMATCH_P_VALUES))
    # Output
    out_dir: str = "ml_artifacts/day30_bench_v3"
    smoke: bool = False  # if True, use smoke_epochs
    long: bool = False   # if True, heavy mode (Day 31)
    long_epochs: int = 30
    long_distances: List[int] = field(default_factory=lambda: [3, 5, 7])
    # Day 31.5
    seeds: int = 1             # number of seeds (5 in long mode)
    p_bin_min_samples: int = 512  # min samples per p-bin
    # Day 34: Calibration mode (BRQL default after Day 34 experiment showed +30% F1)
    calibration_mode: str = "brql"  # 'constrained_f05' or 'brql'
    # Day 35: Local parity channel for FG v1 (reduces density shortcut)
    fg_local_parity_channel: bool = False
    fg_local_parity_alpha: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def effective_epochs(self) -> int:
        if self.smoke:
            return self.smoke_epochs
        if self.long:
            return self.long_epochs
        return self.epochs


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------

@dataclass
class BenchV3Row:
    """One result row: (suite, decoder, extra_label)."""
    suite: str
    decoder: str
    label: str = ""
    f1: float = 0.0
    precision: float = 0.0
    recall_tpr: float = 0.0
    fpr: float = 0.0
    balanced_accuracy: float = 0.0
    train_loss: float = 0.0
    eval_loss: float = 0.0
    dataset_hash: str = ""
    dem_graph_hash: str = ""
    train_samples: int = 0
    test_samples: int = 0
    p_scale: float = 1.0
    mismatch_model: str = ""
    data_p: float = 0.0
    status: str = "fail"
    runtime_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["extra"] = self.extra
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(data_root: str, distance: int) -> Optional[ShardDataset]:
    """Load and merge shards for a given distance."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / data_root / f"d{distance}"
    if not data_dir.exists():
        print(f"  [SKIP] Data dir not found: {data_dir}")
        return None
    datasets = read_shards_dir(data_dir)
    if not datasets:
        print(f"  [SKIP] No shards in {data_dir}")
        return None
    merged = merge_datasets(datasets)
    return merged


def _subsample(dataset: ShardDataset, limit: int, seed: int) -> ShardDataset:
    """Subsample dataset to limit samples."""
    if limit <= 0 or dataset.X.shape[0] <= limit:
        return dataset
    rng = np.random.RandomState(seed)
    idx = rng.choice(dataset.X.shape[0], limit, replace=False)
    idx.sort()
    return ShardDataset(X=dataset.X[idx], Y=dataset.Y[idx],
                        meta=dataset.meta, shard_path=dataset.shard_path)


def _run_mwpm(decoder_id: str, dataset: ShardDataset, cfg: BenchV3Config,
              p_scale: float = 1.0, mismatch_model: str = "",
              noise_model_override: str = "",
              params_override: Optional[Dict[str, Any]] = None) -> BenchV3Row:
    """Run MWPM decoder on within_model split and return result row.

    If dataset.meta is empty (e.g. generated data), params_override must be
    provided with keys: distance, rounds, p, basis.
    """
    from qec_noise_factory.ml.eval.generalization_suite import _extract_xy_by_blocks
    from qec_noise_factory.ml.data.splits import split_within_model

    row = BenchV3Row(suite="", decoder=decoder_id, p_scale=p_scale,
                     mismatch_model=mismatch_model)

    # Get circuit params — from meta or override
    if dataset.meta:
        first_meta = dataset.meta[0]
        params = params_from_canonical(first_meta.params_canonical)
    elif params_override:
        params = dict(params_override)
    else:
        row.status = "skip"
        return row

    # Split data — block-based if meta available, else simple array split
    if dataset.meta:
        split = split_within_model(dataset.meta, train_ratio=0.8, seed=cfg.seed)
        if not split.train_blocks or not split.test_blocks:
            row.status = "skip"
            return row
        test_keys = {b.sample_key for b in split.test_blocks}
        test_X, test_Y, _ = _extract_xy_by_blocks(dataset, test_keys)
        row.dataset_hash = dataset_hash(dataset.meta, "within_model", split_seed=cfg.seed)
        row.train_samples = sum(b.record_count for b in split.train_blocks)
    else:
        # Simple 80/20 split for generated data
        n = dataset.X.shape[0]
        n_test = max(1, n // 5)
        rng = np.random.RandomState(cfg.seed)
        idx = rng.permutation(n)
        test_idx = idx[:n_test]
        test_X = dataset.X[test_idx]
        test_Y = dataset.Y[test_idx]
        row.dataset_hash = ""
        row.train_samples = n - n_test

    row.test_samples = test_X.shape[0]
    row.data_p = params.get("p", 0.0)

    t0 = time.time()

    mwpm = MWPMDecoder()
    nm = noise_model_override or params.get("noise_model", "baseline_symmetric")

    if decoder_id == "MWPM_ORACLE":
        mi = build_oracle_mwpm(
            mwpm, distance=params["distance"], rounds=params["rounds"],
            p=params["p"], basis=params["basis"], noise_model=nm,
        )
    elif decoder_id == "MWPM_MISMATCH_P":
        mi = build_p_scaled_mwpm(
            mwpm, distance=params["distance"], rounds=params["rounds"],
            p=params["p"], basis=params["basis"], noise_model=nm,
            p_scale=p_scale,
        )
    elif decoder_id == "MWPM_MISMATCH_MODEL":
        mi = build_model_mismatched_mwpm(
            mwpm, distance=params["distance"], rounds=params["rounds"],
            p=params["p"], basis=params["basis"],
            true_noise_model=params.get("noise_model", "baseline_symmetric"),
            mismatch_noise_model=mismatch_model,
        )
    else:
        row.status = "fail"
        return row

    # Populate DEM graph hash for traceability
    try:
        effective_p = mi.effective_p if mi.effective_p > 0 else params["p"]
        effective_nm = mi.mismatch_noise_model or nm
        dem_spec = build_dem_graph(
            distance=params["distance"], rounds=params["rounds"],
            p=effective_p, basis=params["basis"],
            noise_model=effective_nm,
        )
        row.dem_graph_hash = dem_spec.dem_graph_hash
    except Exception as e:
        # Log instead of silently swallowing — helps debug hash issues
        row.extra["dem_hash_error"] = str(e)
        # Try with the original noise model as fallback
        try:
            dem_spec = build_dem_graph(
                distance=params["distance"], rounds=params["rounds"],
                p=params["p"], basis=params["basis"],
                noise_model=nm,
            )
            row.dem_graph_hash = dem_spec.dem_graph_hash
        except Exception:
            pass  # truly optional — will be caught by quality gate

    y_hat = mwpm.decode_batch_fast(test_X)
    metrics = compute_metrics(test_Y, y_hat)

    row.f1 = metrics.get("macro_f1", 0.0)
    row.precision = metrics.get("macro_precision", 0.0)
    row.recall_tpr = metrics.get("macro_tpr", 0.0)
    row.fpr = metrics.get("macro_fpr", 0.0)
    row.balanced_accuracy = metrics.get("macro_balanced_accuracy", 0.0)
    # Compute pred_positive_rate for MWPM rows (was missing — always 0.0)
    row.extra["pred_positive_rate"] = float(y_hat.mean())
    row.status = "pass"
    row.runtime_s = round(time.time() - t0, 2)
    return row


# ---------------------------------------------------------------------------
# Day 32: Factor-Graph decoder helper
# ---------------------------------------------------------------------------

def _run_factor_graph(
    dataset: ShardDataset,
    cfg: BenchV3Config,
    out_dir: Path,
    label: str = "FG_DEM_BIPARTITE",
    p_override: float = 0.0,
    noise_model: str = "correlated_crosstalk_like",
    epochs: int = 0,
    version: str = "v0",
) -> BenchV3Row:
    """Run Factor-Graph decoder on a dataset and return result row.

    Builds bipartite DEM graph, trains FactorGraphDecoder, evaluates.
    Works with both shard data (meta available) and generated data (meta empty).

    Args:
        version: 'v0' (Day 32 baseline) or 'v1' (Day 33 — focal loss + F0.5 calibration)
    """
    import torch
    import torch.nn.functional as F
    from qec_noise_factory.ml.graph.dem_bipartite import (
        get_or_build_bipartite_graph,
        bipartite_graph_to_tensors,
    )
    from qec_noise_factory.ml.models.factor_graph import (
        FactorGraphDecoderV0, FactorGraphDecoderV1, FocalLoss,
    )
    from qec_noise_factory.ml.metrics.classification import compute_metrics

    decoder_name = "FG_DEM_BIPARTITE" if version == "v0" else "FG_DEM_BIPARTITE_V1"
    row = BenchV3Row(suite="", decoder=decoder_name, label=label)
    t0 = time.time()

    eff_epochs = epochs if epochs > 0 else cfg.effective_epochs

    # Get circuit params
    if dataset.meta:
        first_meta = dataset.meta[0]
        params = params_from_canonical(first_meta.params_canonical)
    elif p_override > 0:
        params = {
            "distance": cfg.distance, "rounds": cfg.distance,
            "p": p_override, "basis": cfg.basis,
            "noise_model": noise_model,
        }
    else:
        row.status = "skip"
        row.extra["error"] = "No meta and no p_override"
        return row

    try:
        # Build bipartite graph
        bg = get_or_build_bipartite_graph(
            distance=params["distance"],
            rounds=params["rounds"],
            p=params["p"],
            basis=params["basis"],
            noise_model=noise_model,
        )
        row.dem_graph_hash = bg.dem_topology_hash
        row.extra["bipartite_stats"] = {
            "num_detectors": bg.num_detectors,
            "num_errors": bg.num_errors,
            "total_edges": bg.stats.get("total_edges", 0),
            "k_gt2_count": bg.stats.get("k_gt2_count", 0),
            "k_gt2_mass_ratio": bg.stats.get("k_gt2_mass_ratio", 0.0),
        }

        # Convert to tensors
        ei_d2e, ei_e2d, err_w, obs_mask = bipartite_graph_to_tensors(bg)

        # Data split: 80/20
        n = dataset.X.shape[0]
        n_test = max(1, n // 5)
        rng = np.random.RandomState(cfg.seed)
        idx = rng.permutation(n)
        train_idx, test_idx = idx[n_test:], idx[:n_test]

        train_X = dataset.X[train_idx]
        train_Y = dataset.Y[train_idx]
        test_X = dataset.X[test_idx]
        test_Y = dataset.Y[test_idx]

        row.train_samples = train_X.shape[0]
        row.test_samples = test_X.shape[0]
        row.data_p = params.get("p", 0.0)

        # Build detector features: (B, N_d, 2) — [syndrome_bit, is_boundary]
        N_det = bg.num_detectors  # includes boundary
        num_det_actual = train_X.shape[1]
        det_feat_dim = 2

        def _build_det_features(X_arr):
            B_local = X_arr.shape[0]
            feats = np.zeros((B_local, N_det, det_feat_dim), dtype=np.float32)
            feats[:, :num_det_actual, 0] = X_arr.astype(np.float32)
            feats[:, -1, 1] = 1.0  # boundary node marker
            return torch.from_numpy(feats)

        train_det = _build_det_features(train_X)
        test_det = _build_det_features(test_X)

        # Error features: (N_e, 1) — static weights
        err_feats = bg.error_weights.reshape(-1, 1)
        err_feats_t = torch.from_numpy(err_feats).float()

        # Create model — v0 or v1
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        num_obs = train_Y.shape[1] if train_Y.ndim > 1 else 1

        # Compute pos_weight (auto, clamped)
        y_pos = train_Y.astype(np.float32).mean()
        pw_auto = (1 - y_pos) / max(y_pos, 1e-6)
        pw_max = 8.0
        pw_used = min(pw_max, pw_auto)
        pw_clamped = pw_used < pw_auto

        if version == "v1":
            hidden = 48
            loss_name = "focal"
            model = FactorGraphDecoderV1(
                det_input_dim=det_feat_dim,
                err_input_dim=1,
                output_dim=num_obs,
                hidden_dim=hidden,
                num_mp_layers=3,
                readout="mean_max",
                dropout=0.1,
                loss_fn="focal",
                focal_gamma=2.0,
                use_parity_channel=cfg.fg_local_parity_channel,
                parity_alpha=cfg.fg_local_parity_alpha,
            )
            focal_loss_fn = FocalLoss(gamma=2.0, pos_weight=pw_used)
        else:
            hidden = cfg.hidden_dim
            loss_name = "bce"
            model = FactorGraphDecoderV0(
                det_input_dim=det_feat_dim,
                err_input_dim=1,
                output_dim=num_obs,
                hidden_dim=hidden,
                num_mp_layers=3,
                readout="mean_max",
                dropout=0.1,
            )

        row.extra["decoder_version"] = version
        row.extra["loss_name"] = loss_name
        row.extra["pos_weight_auto"] = float(pw_auto)
        row.extra["pos_weight_used"] = float(pw_used)
        row.extra["pos_weight_clamped"] = pw_clamped

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        pw_tensor = torch.tensor([pw_used], dtype=torch.float32)

        # Train
        batch_size = min(64, train_det.shape[0])
        train_Y_2d = train_Y.astype(np.float32)
        if train_Y_2d.ndim == 1:
            train_Y_2d = train_Y_2d.reshape(-1, 1)
        train_Y_t = torch.from_numpy(train_Y_2d)

        model.train()
        for epoch in range(eff_epochs):
            perm = torch.randperm(train_det.shape[0])
            for start in range(0, train_det.shape[0], batch_size):
                end = min(start + batch_size, train_det.shape[0])
                bi = perm[start:end]

                det_b = train_det[bi]
                y_b = train_Y_t[bi]

                logits = model(
                    det_b, err_feats_t,
                    ei_d2e, ei_e2d,
                    error_weights=err_w,
                    observable_mask=obs_mask,
                )

                # Loss computation — version-dependent
                if version == "v1":
                    loss = focal_loss_fn(logits, y_b)
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        logits, y_b, pos_weight=pw_tensor,
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_logits = model(
                test_det, err_feats_t,
                ei_d2e, ei_e2d,
                error_weights=err_w,
                observable_mask=obs_mask,
            )
            test_probs = torch.sigmoid(test_logits).numpy()

        # Calibrate threshold on training set
        with torch.no_grad():
            train_logits = model(
                train_det, err_feats_t,
                ei_d2e, ei_e2d,
                error_weights=err_w,
                observable_mask=obs_mask,
            )
            train_probs = torch.sigmoid(train_logits).numpy()

        train_Y_bool_cal = train_Y_2d.astype(bool)

        if version == "v1" and cfg.calibration_mode == "brql":
            # Day 34: BRQL (Base-Rate Quantile Lock) calibration
            # Locks PPR to match the base_rate from validation labels
            base_rate = float(train_Y_bool_cal.mean())
            tau = float(np.quantile(train_probs.ravel(), 1 - base_rate))
            brql_fallback = False

            if tau < 0.05 or tau > 0.95:
                # Out of bounds — fall back to constrained F0.5
                from qec_noise_factory.ml.bench.reason_codes import BRQL_FALLBACK
                brql_fallback = True
                brql_fallback_reason = f"tau={tau:.4f} out of bounds [0.05, 0.95]"
                print(f"    ⚠ BRQL fallback: {brql_fallback_reason}")
                # Fall through to constrained F0.5 below
                best_thr = 0.5
            else:
                best_thr = tau

            if not brql_fallback:
                # BRQL succeeded — use tau directly
                test_preds_brql = (train_probs.ravel() > tau).astype(bool)
                achieved_ppr = float(test_preds_brql.mean())
                row.extra["calibration"] = {
                    "metric": "brql",
                    "base_rate": base_rate,
                    "tau": tau,
                    "achieved_ppr": achieved_ppr,
                    "fallback": False,
                }
            else:
                # Fallback to constrained F0.5
                row.extra["calibration"] = {
                    "metric": "brql_fallback_to_f05",
                    "base_rate": base_rate,
                    "tau_attempted": tau,
                    "fallback": True,
                    "fallback_reason": brql_fallback_reason,
                }
                # Execute the constrained F0.5 calibration below
                cal_grid = np.linspace(0.15, 0.95, 33)
                fpr_cap = 0.60
                ppr_cap = 0.90
                tpr_min = 0.05
                ppr_min = 0.01
                cal_metric = "f0.5_constrained"
                best_thr, best_score = 0.5, -1.0
                fallback_used = False
                cal_details = []
                rejected_reasons = {"fpr_over_cap": 0, "ppr_over_cap": 0,
                                    "tpr_under_min": 0, "ppr_under_min": 0}
                # Replicate the constrained F0.5 sweep (same as v1 below)
                for thr in cal_grid:
                    preds_cal = (train_probs > thr).astype(bool)
                    tp = (preds_cal & train_Y_bool_cal).sum()
                    tn = (~preds_cal & ~train_Y_bool_cal).sum()
                    fp = (preds_cal & ~train_Y_bool_cal).sum()
                    fn = (~preds_cal & train_Y_bool_cal).sum()
                    tpr_val = float(tp / max(tp + fn, 1))
                    fpr_val = float(fp / max(fp + tn, 1))
                    prec_val = float(tp / max(tp + fp, 1))
                    ppr_val = float(preds_cal.mean())
                    reject = False
                    if fpr_val > fpr_cap:
                        rejected_reasons["fpr_over_cap"] += 1; reject = True
                    if ppr_val > ppr_cap:
                        rejected_reasons["ppr_over_cap"] += 1; reject = True
                    if tpr_val < tpr_min:
                        rejected_reasons["tpr_under_min"] += 1; reject = True
                    if ppr_val < ppr_min:
                        rejected_reasons["ppr_under_min"] += 1; reject = True
                    if reject:
                        continue
                    beta = 0.5
                    f_beta = float((1 + beta**2) * prec_val * tpr_val / max((beta**2 * prec_val + tpr_val), 1e-9))
                    if f_beta > best_score:
                        best_score = f_beta
                        best_thr = float(thr)
                row.extra["calibration"]["best_threshold"] = float(best_thr)
                row.extra["calibration"]["best_score"] = float(best_score)

        elif version == "v1":
            # V1: Constrained F0.5 calibration (precision-favoring)
            # Constraints prevent degenerate threshold selection:
            #   - FPR cap   ≤ 60%  (prevents reverse collapse)
            #   - PPR cap   ≤ 90%  (prevents all-positive collapse)
            #   - TPR floor ≥  5%  (prevents all-negative collapse)
            #   - PPR floor ≥  1%  (prevents all-negative collapse)
            #   - Threshold floor 0.15 (prevents near-zero thresholds)
            cal_grid = np.linspace(0.15, 0.95, 33)
            fpr_cap = 0.60
            ppr_cap = 0.90
            tpr_min = 0.05
            ppr_min = 0.01
            cal_metric = "f0.5_constrained"
            best_thr, best_score = 0.5, -1.0
            fallback_used = False
            cal_details = []
            rejected_reasons = {"fpr_over_cap": 0, "ppr_over_cap": 0,
                                "tpr_under_min": 0, "ppr_under_min": 0}

            for thr in cal_grid:
                preds_cal = (train_probs > thr).astype(bool)
                tp = (preds_cal & train_Y_bool_cal).sum()
                tn = (~preds_cal & ~train_Y_bool_cal).sum()
                fp = (preds_cal & ~train_Y_bool_cal).sum()
                fn = (~preds_cal & train_Y_bool_cal).sum()
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                fpr_cal = fp / max(fp + tn, 1)
                ppr_cal = float(preds_cal.mean())

                # F0.5 — precision-weighted (beta=0.5)
                beta = 0.5
                f_beta = (1 + beta**2) * prec * rec / max((beta**2 * prec + rec), 1e-8)
                f1_cal = 2 * prec * rec / max(prec + rec, 1e-8)
                lam = 0.2
                f1_pen = f1_cal - lam * fpr_cal

                # Full feasibility check (caps + floors)
                feasible = (
                    fpr_cal <= fpr_cap
                    and ppr_cal <= ppr_cap
                    and rec >= tpr_min      # TPR floor
                    and ppr_cal >= ppr_min   # PPR floor
                )

                # Track rejection reasons
                if fpr_cal > fpr_cap: rejected_reasons["fpr_over_cap"] += 1
                if ppr_cal > ppr_cap: rejected_reasons["ppr_over_cap"] += 1
                if rec < tpr_min: rejected_reasons["tpr_under_min"] += 1
                if ppr_cal < ppr_min: rejected_reasons["ppr_under_min"] += 1

                cal_details.append({
                    "threshold": float(thr),
                    "f0.5": float(f_beta),
                    "f1": float(f1_cal),
                    "f1_pen": float(f1_pen),
                    "precision": float(prec),
                    "recall": float(rec),
                    "fpr": float(fpr_cal),
                    "ppr": float(ppr_cal),
                    "feasible": feasible,
                })

                if feasible and f_beta > best_score:
                    best_score = f_beta
                    best_thr = float(thr)

            # Fallback 1: if no F0.5-feasible, try F1 with same constraints
            if best_score <= 0:
                fallback_used = True
                cal_metric = "f1_constrained_fallback"
                for entry in cal_details:
                    if entry["feasible"] and entry["f1"] > best_score:
                        best_score = entry["f1"]
                        best_thr = entry["threshold"]

            # Fallback 2: relax caps but keep TPR/PPR FLOORS
            if best_score <= 0:
                cal_metric = "f1_relaxed_caps_fallback"
                for entry in cal_details:
                    relaxed_ok = (
                        entry["recall"] >= tpr_min
                        and entry["ppr"] >= ppr_min
                    )
                    if relaxed_ok and entry["f1"] > best_score:
                        best_score = entry["f1"]
                        best_thr = entry["threshold"]

            # Fallback 3: HARD FAIL — no threshold has TPR>0 AND PPR>0
            if best_score <= 0:
                cal_metric = "DEGENERATE_FAIL"
                best_thr = 0.5  # safe default
                row.status = "fail"
                row.extra["reason_code"] = "FG_CALIBRATION_DEGENERATE"
                row.extra["calibration_failure"] = (
                    "No threshold yields TPR>0 and PPR>0. "
                    f"Rejected: {rejected_reasons}"
                )

            row.extra["calibration"] = {
                "metric": cal_metric,
                "grid_size": len(cal_grid),
                "best_threshold": best_thr,
                "best_score": float(best_score),
                "lambda": lam,
                "fpr_cap": fpr_cap,
                "ppr_cap": ppr_cap,
                "tpr_min": tpr_min,
                "ppr_min": ppr_min,
                "fallback_used": fallback_used,
                "feasible_count": sum(1 for d in cal_details if d["feasible"]),
                "rejected_reason_counts": rejected_reasons,
                "hit_threshold_floor": abs(best_thr - float(cal_grid[0])) < 0.001,
                "hit_fpr_cap": rejected_reasons["fpr_over_cap"] > 0,
                "hit_ppr_cap": rejected_reasons["ppr_over_cap"] > 0,
                "hit_tpr_min": rejected_reasons["tpr_under_min"] > 0,
                "hit_ppr_min": rejected_reasons["ppr_under_min"] > 0,
                "details": cal_details,
            }
        else:
            # V0: balanced accuracy calibration (Day 32 baseline)
            best_thr, best_ba = 0.5, 0.0
            for thr in np.arange(0.05, 0.95, 0.05):
                preds_cal = (train_probs > thr).astype(bool)
                tp = (preds_cal & train_Y_bool_cal).sum()
                tn = (~preds_cal & ~train_Y_bool_cal).sum()
                fp = (preds_cal & ~train_Y_bool_cal).sum()
                fn = (~preds_cal & train_Y_bool_cal).sum()
                tpr_cal = tp / max(tp + fn, 1)
                tnr_cal = tn / max(tn + fp, 1)
                ba = (tpr_cal + tnr_cal) / 2
                if ba > best_ba:
                    best_ba = ba
                    best_thr = thr
            row.extra["calibration"] = {
                "metric": "balanced_accuracy",
                "best_threshold": float(best_thr),
                "best_score": float(best_ba),
            }

        test_preds = (test_probs > best_thr).astype(bool)
        row.extra["calibrated_threshold"] = float(best_thr)

        test_Y_bool = test_Y.astype(bool)
        if test_Y_bool.ndim == 1:
            test_Y_bool = test_Y_bool.reshape(-1, 1)
        metrics = compute_metrics(test_Y_bool, test_preds)

        row.f1 = metrics.get("macro_f1", 0.0)
        row.precision = metrics.get("macro_precision", 0.0)
        row.recall_tpr = metrics.get("macro_tpr", 0.0)
        row.fpr = metrics.get("macro_fpr", 0.0)
        row.balanced_accuracy = metrics.get("macro_balanced_accuracy", 0.0)
        row.extra["pred_positive_rate"] = float(test_preds.mean())
        row.extra["train_loss"] = float(loss.item()) if 'loss' in dir() else 0.0

        # Day 34: Ranking diagnostics (unthresholded)
        try:
            from qec_noise_factory.ml.metrics.ranking import compute_ranking_diagnostics
            y_flat = test_Y_bool.ravel() if test_Y_bool.ndim > 1 else test_Y_bool
            p_flat = test_probs.ravel() if test_probs.ndim > 1 else test_probs
            ranking = compute_ranking_diagnostics(y_flat, p_flat)
            row.extra["auroc"] = ranking["auroc"]
            row.extra["pr_auc"] = ranking["pr_auc"]
            row.extra["decile_table"] = ranking["decile_table"]
        except Exception as e:
            row.extra["ranking_diag_error"] = str(e)

        row.status = "pass"

    except Exception as e:
        row.extra["error"] = str(e)
        import traceback
        row.extra["traceback"] = traceback.format_exc()
        row.status = "fail"

    row.runtime_s = round(time.time() - t0, 2)
    return row

def _run_gnn(dataset: ShardDataset, cfg: BenchV3Config,
             out_dir: Path, label: str = "GNN_V2_DEM") -> BenchV3Row:
    """Run GNN V2 DEM decoder on within_model split."""
    row = BenchV3Row(suite="", decoder="GNN_V2_DEM", label=label)
    t0 = time.time()

    exp_id = f"gnn_{label}"
    exp_cfg = ExperimentConfig(
        exp_id=exp_id,
        name=f"GNN_V2_DEM ({label})",
        split_policy="within_model",
        epochs=cfg.effective_epochs,
        batch_size=cfg.batch_size,
        hidden_dim=cfg.hidden_dim,
        seed=cfg.seed,
        loss_pos_weight=0,
        calibrate_threshold=True,
        pos_weight_max=8.0,
        gnn_version="v2",
        graph_mode="dem",
        gnn_feature_version="v1",
        gnn_readout="mean_max",
        featureset="v1_nop",
        calibrate_metric="f1",
        calibrate_lambda=0.0,
    )

    try:
        report = run_experiment(exp_cfg, dataset, out_dir / exp_id)
        rd = report.to_dict()
        m = rd.get("gnn_metrics") or {}

        row.f1 = m.get("macro_f1", 0.0)
        row.precision = m.get("macro_precision", 0.0)
        row.recall_tpr = m.get("macro_tpr", 0.0)
        row.fpr = m.get("macro_fpr", 0.0)
        row.balanced_accuracy = m.get("macro_balanced_accuracy", 0.0)
        row.train_loss = rd.get("gnn_train_loss", 0.0)
        row.eval_loss = rd.get("gnn_eval_loss", 0.0)
        row.dataset_hash = rd.get("dataset_hash", "")
        row.dem_graph_hash = rd.get("dem_graph_hash", "")
        row.train_samples = rd.get("train_samples", 0)
        row.test_samples = rd.get("test_samples", 0)
        row.status = rd.get("status", "fail")
    except Exception as e:
        row.extra["error"] = str(e)
        row.status = "fail"

    row.runtime_s = round(time.time() - t0, 2)
    return row


# ---------------------------------------------------------------------------
# Suite Runners
# ---------------------------------------------------------------------------

def run_suite_a(cfg: BenchV3Config, dataset: ShardDataset,
                out_dir: Path) -> List[BenchV3Row]:
    """Suite A — ORACLE: Baseline noise, correct DEM, MWPM + GNN."""
    print("\n" + "=" * 60)
    print("Suite A — ORACLE (baseline, true DEM)")
    print("=" * 60)

    rows: List[BenchV3Row] = []

    # MWPM Oracle
    print("  Running MWPM_ORACLE ...", end=" ", flush=True)
    r = _run_mwpm("MWPM_ORACLE", dataset, cfg)
    r.suite = "A_ORACLE"
    r.label = "oracle"
    rows.append(r)
    print(f"F1={r.f1:.2%} [{r.status}]")

    # GNN V2 DEM
    print("  Running GNN_V2_DEM ...", end=" ", flush=True)
    r = _run_gnn(dataset, cfg, out_dir, label="oracle")
    r.suite = "A_ORACLE"
    rows.append(r)
    print(f"F1={r.f1:.2%} [{r.status}]")

    return rows


def run_suite_b(cfg: BenchV3Config, dataset: ShardDataset,
                out_dir: Path) -> List[BenchV3Row]:
    """Suite B — P_SWEEP: MWPM with wide p_scale sweep."""
    print("\n" + "=" * 60)
    print("Suite B — P_SWEEP (p_scale grid)")
    print("=" * 60)

    rows: List[BenchV3Row] = []

    for ps in cfg.p_scales:
        label = f"p_scale={ps}"
        print(f"  Running MWPM_MISMATCH_P ({label}) ...", end=" ", flush=True)
        r = _run_mwpm("MWPM_MISMATCH_P", dataset, cfg, p_scale=ps)
        r.suite = "B_P_SWEEP"
        r.label = label
        rows.append(r)
        print(f"F1={r.f1:.2%} [{r.status}]")

    return rows


def run_suite_c(cfg: BenchV3Config, dataset: ShardDataset,
                out_dir: Path) -> List[BenchV3Row]:
    """Suite C — MODEL_MIS: Decode with DEM from different noise models.

    Iterates over mismatch_models × mismatch_p_values to capture
    degradation across multiple p-buckets.
    """
    print("\n" + "=" * 60)
    print("Suite C — MODEL MISMATCH (cross-model DEM)")
    print("=" * 60)

    rows: List[BenchV3Row] = []

    for mm in cfg.mismatch_models:
        for p_val in cfg.mismatch_p_values:
            label = f"model={mm}_p={p_val}"
            print(f"  Running MWPM_MISMATCH_MODEL ({label}) ...", end=" ", flush=True)
            r = _run_mwpm("MWPM_MISMATCH_MODEL", dataset, cfg, mismatch_model=mm)
            r.suite = "C_MODEL_MIS"
            r.label = label
            r.data_p = p_val
            rows.append(r)
            print(f"F1={r.f1:.2%} [{r.status}]")

    return rows


def run_suite_d(cfg: BenchV3Config, corr_dataset: ShardDataset,
                out_dir: Path) -> List[BenchV3Row]:
    """Suite D — CORRELATED: Correlated noise data, MWPM vs GNN."""
    print("\n" + "=" * 60)
    print("Suite D — CORRELATED NOISE ARENA")
    print("=" * 60)

    rows: List[BenchV3Row] = []

    # MWPM Oracle (with correlated DEM = clique-expanded)
    print("  Running MWPM_ORACLE (correlated) ...", end=" ", flush=True)
    r = _run_mwpm("MWPM_ORACLE", corr_dataset, cfg,
                  noise_model_override="correlated_crosstalk_like")
    r.suite = "D_CORRELATED"
    r.label = "mwpm_corr"
    rows.append(r)
    print(f"F1={r.f1:.2%} [{r.status}]")

    # GNN V2 DEM (on correlated data)
    print("  Running GNN_V2_DEM (correlated) ...", end=" ", flush=True)
    r = _run_gnn(corr_dataset, cfg, out_dir, label="correlated")
    r.suite = "D_CORRELATED"
    rows.append(r)
    print(f"F1={r.f1:.2%} [{r.status}]")

    return rows


def run_suite_d_v2(
    cfg: BenchV3Config,
    corr_dataset: ShardDataset,
    out_dir: Path,
) -> Tuple[List[BenchV3Row], Dict[str, Any]]:
    """Suite D v2 — Per-p correlated noise evaluation (Day 31 + 31.5).

    Day 31.5 upgrades:
      - MWPM triviality probe during p-grid selection (long mode)
      - Nearest-p binning (replaces ±10% window)
      - Generate-on-demand for missing p bins (long mode)
      - DEM correlation-mass stats per-p
      - Seeds + CI for mismatch-oracle deltas (long mode)
      - GNN collapse guard
    """
    from qec_noise_factory.ml.bench.p_grid_selector import (
        select_p_grid_correlated, PGridThresholds,
        assign_nearest_p, check_data_availability, generate_mini_dataset,
    )

    print("\n" + "=" * 60)
    print("Suite D v2 — CORRELATED NOISE ARENA (Per-p, Day 31.5)")
    print("=" * 60)

    effective_seeds = cfg.seeds if cfg.long else 1

    # 1. Select informative p-grid
    scan_shots = 1024 if cfg.long else 512
    run_probe = cfg.long  # MWPM probe only in long mode
    print(f"  [Scan] Pre-scanning p candidates ({scan_shots} shots"
          f"{', +MWPM probe' if run_probe else ''})...")
    grid_result = select_p_grid_correlated(
        distance=cfg.distance, basis=cfg.basis,
        noise_model="correlated_crosstalk_like",
        scan_shots=scan_shots, seed=cfg.seed + 31_000,
        run_mwpm_probe=run_probe,
    )

    p_grid = grid_result.p_grid
    print(f"  [Grid] Selected {len(p_grid)} p values: "
          f"{[f'{p:.4f}' for p in p_grid]}")
    for s in grid_result.per_p_stats:
        status = "✓" if s.accepted else f"✗ ({s.reject_reason})"
        probe_str = f" probe_f1={s.mwpm_probe_f1:.3f}" if s.mwpm_probe_f1 >= 0 else ""
        print(f"    p={s.p:.4f}: y_rate={s.y_rate:.4f}, "
              f"density={s.detector_density:.4f}{probe_str} {status}")

    # 2. Nearest-p binning + data availability
    sample_p_vals = np.array([
        float(m.p)
        for m in corr_dataset.meta
    ]) if hasattr(corr_dataset, 'meta') and corr_dataset.meta else np.array([])

    bin_result = assign_nearest_p(sample_p_vals, p_grid) if len(sample_p_vals) > 0 else None
    avail_info = check_data_availability(
        p_grid, sample_p_vals, min_samples=cfg.p_bin_min_samples,
    ) if len(sample_p_vals) > 0 else {"available": [], "missing": list(p_grid), "counts": {}}

    print(f"  [Bins] Available: {avail_info['available']}, Missing: {avail_info['missing']}")

    rows: List[BenchV3Row] = []
    per_p_info: Dict[str, Any] = {}
    dem_corr_stats_all: Dict[str, Any] = {}
    generated_datasets: Dict[float, Any] = {}
    gnn_diagnostics: List[Dict[str, Any]] = []
    bin_counts: Dict[str, int] = bin_result["bin_counts"] if bin_result else {}

    # 3. Per-p evaluation
    for p_idx, p in enumerate(p_grid):
        # Get subset via nearest-p binning or generate on-demand
        p_key = f"{p:.6f}"
        if bin_result is not None and bin_counts.get(p_key, 0) >= 10:
            # Use nearest-p binning
            mask = bin_result["bin_assignments"] == p_idx
            subset_X = corr_dataset.X[mask]
            subset_y = corr_dataset.Y[mask]
            subset = ShardDataset(
                X=subset_X, Y=subset_y,
                meta=([corr_dataset.meta[i] for i, m in enumerate(mask) if m]
                          if hasattr(corr_dataset, 'meta') and corr_dataset.meta
                          else []),
                shard_path=f"binned(p={p:.4f})",
            )
            data_source = "shard"
        elif cfg.long:
            # Generate on-demand (long mode only)
            print(f"  [GEN] Generating {cfg.p_bin_min_samples} samples for p={p:.4f}...")
            gen_data = generate_mini_dataset(
                p=p, distance=cfg.distance, basis=cfg.basis,
                n_samples=max(cfg.p_bin_min_samples, 2048),
                seed=cfg.seed + 50_000 + p_idx,
            )
            generated_datasets[p] = gen_data
            subset = ShardDataset(
                X=gen_data["X"], Y=gen_data["y"],
                meta=[], shard_path=f"generated(p={p:.4f})",
            )
            data_source = "generated"
        else:
            # Smoke/normal: fallback to old filter
            subset = filter_by_p_range(corr_dataset, p_lo=p*0.9, p_hi=p*1.1)
            if subset.X.shape[0] < 10:
                print(f"  [SKIP] p={p:.4f}: only {subset.X.shape[0]} samples")
                continue
            data_source = "filter"

        n_samples = subset.X.shape[0]
        if n_samples < 10:
            print(f"  [SKIP] p={p:.4f}: {n_samples} samples ({data_source})")
            continue

        p_label = f"p={p:.4f}"
        print(f"  [p={p:.4f}] {n_samples} samples ({data_source})")

        # Build params for this p (fallback for generated data with empty meta)
        p_params = {
            "distance": cfg.distance, "rounds": cfg.distance,
            "p": p, "basis": cfg.basis,
            "noise_model": "correlated_crosstalk_like",
        }

        # DEM correlation-mass stats
        try:
            corr_stats = dem_corr_stats(
                distance=cfg.distance, rounds=cfg.distance, p=p,
                basis=cfg.basis,
            )
            dem_corr_stats_all[p_key] = corr_stats
            print(f"    DEM corr: k>2_mass={corr_stats['k_gt_2_mass_ratio']:.4f}, "
                  f"hyperedges={corr_stats['hyperedges_k_gt_2_count']}")
        except Exception as e:
            print(f"    DEM corr stats failed: {e}")
            dem_corr_stats_all[p_key] = {"error": str(e)}

        # MWPM Oracle
        print(f"    MWPM_ORACLE ...", end=" ", flush=True)
        r_mwpm = _run_mwpm(
            "MWPM_ORACLE", subset, cfg,
            noise_model_override="correlated_crosstalk_like",
            params_override=p_params,
        )
        r_mwpm.suite = "D_CORRELATED_V2"
        r_mwpm.label = f"mwpm_{p_label}"
        r_mwpm.data_p = p
        rows.append(r_mwpm)
        print(f"F1={r_mwpm.f1:.2%} TPR={r_mwpm.recall_tpr:.2%} [{r_mwpm.status}]")
        if r_mwpm.dem_graph_hash:
            print(f"      dem_graph_hash={r_mwpm.dem_graph_hash}")
        elif r_mwpm.extra.get("dem_hash_error"):
            print(f"      ⚠ dem_hash_error: {r_mwpm.extra['dem_hash_error']}")

        # GNN (skip for generated data — needs meta for block splitting)
        r_gnn = None
        if subset.meta:
            print(f"    GNN_V2_DEM ...", end=" ", flush=True)
            r_gnn = _run_gnn(subset, cfg, out_dir, label=f"corr_{p_label}")
            r_gnn.suite = "D_CORRELATED_V2"
            r_gnn.label = f"gnn_{p_label}"
            r_gnn.data_p = p
            rows.append(r_gnn)
            gnn_ppr = float(r_gnn.extra.get("pred_positive_rate", 0.0))
            print(f"F1={r_gnn.f1:.2%} TPR={r_gnn.recall_tpr:.2%} "
                  f"PPR={gnn_ppr:.2%} [{r_gnn.status}]")
        else:
            print(f"    GNN_V2_DEM ... [SKIP] generated data, no meta")
            gnn_ppr = -1.0  # marker: not evaluated

        # GNN collapse guard
        if gnn_ppr >= 0:
            gnn_diag = {
                "p": p, "pred_positive_rate": gnn_ppr,
                "f1": r_gnn.f1 if subset.meta else 0.0,
                "tpr": r_gnn.recall_tpr if subset.meta else 0.0,
                "fpr": r_gnn.fpr if subset.meta else 0.0,
                "collapse_warn": gnn_ppr < 0.005 or gnn_ppr > 0.95,
            }
            gnn_diagnostics.append(gnn_diag)
            if gnn_diag["collapse_warn"]:
                print(f"    ⚠ GNN collapse risk: PPR={gnn_ppr:.4f}")

        # MWPM Mismatch
        print(f"    MWPM_MISMATCH ...", end=" ", flush=True)
        r_mis = _run_mwpm(
            "MWPM_MISMATCH_MODEL", subset, cfg,
            mismatch_model="baseline_symmetric",
            params_override=p_params,
        )
        r_mis.suite = "D_CORRELATED_V2"
        r_mis.label = f"mismatch_{p_label}"
        r_mis.data_p = p
        rows.append(r_mis)
        print(f"F1={r_mis.f1:.2%} TPR={r_mis.recall_tpr:.2%} [{r_mis.status}]")

        # Day 32: Factor-Graph decoder v0
        print(f"    FG_DEM_BIPARTITE ...", end=" ", flush=True)
        r_fg = _run_factor_graph(
            subset, cfg, out_dir,
            label=f"fg_{p_label}",
            p_override=p,
            noise_model="correlated_crosstalk_like",
            version="v0",
        )
        r_fg.suite = "D_CORRELATED_V2"
        r_fg.label = f"fg_{p_label}"
        r_fg.data_p = p
        rows.append(r_fg)
        fg_ppr = float(r_fg.extra.get("pred_positive_rate", 0.0))
        print(f"F1={r_fg.f1:.2%} TPR={r_fg.recall_tpr:.2%} "
              f"PPR={fg_ppr:.2%} [{r_fg.status}]")

        # Day 33: Factor-Graph v1 (focal loss + F0.5 calibration)
        print(f"    FG_V1_BIPARTITE ...", end=" ", flush=True)
        r_fg1 = _run_factor_graph(
            subset, cfg, out_dir,
            label=f"fg1_{p_label}",
            p_override=p,
            noise_model="correlated_crosstalk_like",
            version="v1",
        )
        r_fg1.suite = "D_CORRELATED_V2"
        r_fg1.label = f"fg1_{p_label}"
        r_fg1.data_p = p
        rows.append(r_fg1)
        fg1_ppr = float(r_fg1.extra.get("pred_positive_rate", 0.0))
        fg1_thr = r_fg1.extra.get("calibrated_threshold", 0.5)
        fg1_cal = r_fg1.extra.get("calibration", {})
        print(f"F1={r_fg1.f1:.2%} TPR={r_fg1.recall_tpr:.2%} "
              f"PPR={fg1_ppr:.2%} thr={fg1_thr:.3f} [{r_fg1.status}]")

        # Day 34: Ranking diagnostics print
        fg1_auroc = r_fg1.extra.get("auroc")
        fg1_prauc = r_fg1.extra.get("pr_auc")
        if fg1_auroc is not None:
            auroc_str = f"AUROC={fg1_auroc:.3f}"
            prauc_str = f"PR-AUC={fg1_prauc:.3f}" if fg1_prauc is not None else ""
            warn = " ⚠ RANKING COLLAPSE" if fg1_auroc < 0.65 else ""
            print(f"    {auroc_str} {prauc_str}{warn}")

        # Calibration constraint hit warnings
        if fg1_cal.get("hit_threshold_floor"):
            print(f"    ⚠ v1 calibration hit threshold floor ({fg1_thr:.3f})")
        if fg1_cal.get("hit_fpr_cap"):
            print(f"    ⚠ v1 calibration hit FPR cap ({fg1_cal.get('fpr_cap', 0.6):.0%})")
        if fg1_cal.get("hit_ppr_cap"):
            print(f"    ⚠ v1 calibration hit PPR cap ({fg1_cal.get('ppr_cap', 0.9):.0%})")
        if fg1_cal.get("fallback_used"):
            print(f"    ⚠ v1 calibration used fallback metric: {fg1_cal.get('metric', '?')}")

        # Collapse guards — v0 and v1
        for tag, r, ppr_val in [("fg_v0", r_fg, fg_ppr), ("fg_v1", r_fg1, fg1_ppr)]:
            diag = {
                "p": p, "decoder": tag,
                "pred_positive_rate": ppr_val,
                "f1": r.f1, "tpr": r.recall_tpr, "fpr": r.fpr,
                "collapse_low": ppr_val < 0.005,
                "collapse_high": ppr_val > 0.95,
                "collapse_tpr": r.recall_tpr < 0.05 and r.f1 > 0,
                "reverse_collapse": r.fpr > 0.70,
            }
            any_warn = (diag["collapse_low"] or diag["collapse_high"]
                        or diag["collapse_tpr"] or diag["reverse_collapse"])
            diag["any_warning"] = any_warn
            if any_warn:
                reasons = []
                if diag["collapse_low"]: reasons.append("PPR<0.5%")
                if diag["collapse_high"]: reasons.append("PPR>95%")
                if diag["collapse_tpr"]: reasons.append("TPR<5%")
                if diag["reverse_collapse"]: reasons.append("FPR>70%")
                print(f"    ⚠ {tag} collapse risk: {', '.join(reasons)}")

        per_p_info[p_key] = {
            "p": p, "samples": n_samples, "data_source": data_source,
            "mwpm_f1": r_mwpm.f1, "mwpm_tpr": r_mwpm.recall_tpr,
            "mwpm_precision": r_mwpm.precision, "mwpm_fpr": r_mwpm.fpr,
            "gnn_f1": r_gnn.f1 if r_gnn is not None else -1.0,
            "gnn_tpr": r_gnn.recall_tpr if r_gnn is not None else -1.0,
            "gnn_precision": r_gnn.precision if r_gnn is not None else -1.0,
            "gnn_fpr": r_gnn.fpr if r_gnn is not None else -1.0,
            "fg_f1": r_fg.f1, "fg_tpr": r_fg.recall_tpr,
            "fg_precision": r_fg.precision, "fg_fpr": r_fg.fpr,
            "fg1_f1": r_fg1.f1, "fg1_tpr": r_fg1.recall_tpr,
            "fg1_precision": r_fg1.precision, "fg1_fpr": r_fg1.fpr,
            "fg1_ppr": fg1_ppr, "fg1_threshold": fg1_thr,
            "mismatch_f1": r_mis.f1, "mismatch_tpr": r_mis.recall_tpr,
            "mismatch_precision": r_mis.precision, "mismatch_fpr": r_mis.fpr,
            "mwpm_pred_positive_rate": float(
                r_mwpm.extra.get("pred_positive_rate", 0.0)),
            "gnn_pred_positive_rate": gnn_ppr if gnn_ppr >= 0 else -1.0,
            "fg_pred_positive_rate": fg_ppr,
            "fg1_pred_positive_rate": fg1_ppr,
            "mismatch_pred_positive_rate": float(
                r_mis.extra.get("pred_positive_rate", 0.0)),
            "delta_f1_mis_vs_oracle": r_mis.f1 - r_mwpm.f1,
            "delta_f1_fg_vs_gnn": r_fg.f1 - (r_gnn.f1 if r_gnn is not None else 0.0),
            "delta_f1_fg_vs_mwpm": r_fg.f1 - r_mwpm.f1,
            "delta_f1_fg1_vs_fg0": r_fg1.f1 - r_fg.f1,
            "delta_ppr_fg1_vs_fg0": fg1_ppr - fg_ppr,
            # Day 34: Ranking diagnostics
            "fg_auroc": r_fg.extra.get("auroc"),
            "fg_pr_auc": r_fg.extra.get("pr_auc"),
            "fg1_auroc": r_fg1.extra.get("auroc"),
            "fg1_pr_auc": r_fg1.extra.get("pr_auc"),
        }

    # 4. Seeds + CI (long mode only)
    ci_summary = {}
    seed_results = []
    if effective_seeds > 1 and per_p_info:
        print(f"\n  [Seeds] Running {effective_seeds} seeds for CI...")
        all_deltas = []
        for seed_i in range(effective_seeds):
            seed_offset = seed_i * 1000
            seed_cfg = BenchV3Config(**{**cfg.to_dict(),
                                       "seed": cfg.seed + seed_offset})
            deltas_this_seed = []
            for p_key_s, pinfo in per_p_info.items():
                p_val = pinfo["p"]
                # Re-run mwpm oracle + mismatch with different seed
                subset_s = filter_by_p_range(corr_dataset, p_lo=p_val*0.9, p_hi=p_val*1.1)
                if subset_s.X.shape[0] < 10:
                    continue
                r_o = _run_mwpm("MWPM_ORACLE", subset_s, seed_cfg,
                               noise_model_override="correlated_crosstalk_like")
                r_m = _run_mwpm("MWPM_MISMATCH_MODEL", subset_s, seed_cfg,
                               mismatch_model="baseline_symmetric")
                delta = r_m.f1 - r_o.f1
                deltas_this_seed.append(delta)
            if deltas_this_seed:
                mean_delta = float(np.mean(deltas_this_seed))
                all_deltas.append(mean_delta)
                seed_results.append({
                    "seed": cfg.seed + seed_offset,
                    "mean_delta_f1": mean_delta,
                    "per_p_deltas": deltas_this_seed,
                })
                print(f"    seed={cfg.seed + seed_offset}: "
                      f"mean_delta_f1={mean_delta:+.4f}")

        if len(all_deltas) >= 2:
            mean_d = float(np.mean(all_deltas))
            std_d = float(np.std(all_deltas, ddof=1))
            ci_lo = mean_d - 1.96 * std_d / np.sqrt(len(all_deltas))
            ci_hi = mean_d + 1.96 * std_d / np.sqrt(len(all_deltas))
            ci_summary = {
                "mean_delta_f1": round(mean_d, 6),
                "std": round(std_d, 6),
                "ci_95_lo": round(float(ci_lo), 6),
                "ci_95_hi": round(float(ci_hi), 6),
                "ci_method": "normal_approx",
                "n_seeds": len(all_deltas),
                "spans_zero": bool(ci_lo <= 0 <= ci_hi),
            }
            print(f"  [CI] delta_f1={mean_d:+.4f} "
                  f"95%CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
                  f"{'spans 0!' if ci_summary['spans_zero'] else 'significant'}")

    # 5. Aggregate summary
    mwpm_rows = [r for r in rows
                 if r.decoder == "MWPM_ORACLE" and r.status == "pass"
                 and r.label != "aggregate"]
    gnn_rows = [r for r in rows
                if r.decoder == "GNN_V2_DEM" and r.status == "pass"
                and r.label != "aggregate"]
    mis_rows = [r for r in rows
                if r.decoder == "MWPM_MISMATCH_MODEL" and r.status == "pass"
                and r.label != "aggregate"]

    for dec_name, dec_rows in [("MWPM_ORACLE", mwpm_rows),
                                ("GNN_V2_DEM", gnn_rows),
                                ("MWPM_MISMATCH_MODEL", mis_rows)]:
        if dec_rows:
            agg = BenchV3Row(
                suite="D_CORRELATED_V2", decoder=dec_name,
                label="aggregate",
                f1=float(np.mean([r.f1 for r in dec_rows])),
                recall_tpr=float(np.mean([r.recall_tpr for r in dec_rows])),
                precision=float(np.mean([r.precision for r in dec_rows])),
                fpr=float(np.mean([r.fpr for r in dec_rows])),
                balanced_accuracy=float(np.mean([r.balanced_accuracy for r in dec_rows])),
                status="pass",
            )
            if dec_name == "MWPM_MISMATCH_MODEL":
                agg.mismatch_model = "baseline_symmetric"
            rows.append(agg)
            short_name = {"MWPM_ORACLE": "MWPM Oracle",
                          "GNN_V2_DEM": "GNN",
                          "MWPM_MISMATCH_MODEL": "Mismatch"}[dec_name]
            print(f"  [AGG] {short_name:15s} F1={agg.f1:.2%} TPR={agg.recall_tpr:.2%}")

    # 6. Build informativeness report
    info_report = {
        "p_grid_result": grid_result.to_dict(),
        "per_p_info": per_p_info,
        "n_p_points": len(p_grid),
        "n_p_evaluated": len(per_p_info),
        "dem_corr_stats": dem_corr_stats_all,
        "bin_counts": bin_counts,
        "generated_p": list(generated_datasets.keys()),
        "gnn_diagnostics": gnn_diagnostics,
        "ci_summary": ci_summary,
        "seed_results": seed_results,
        "effective_seeds": effective_seeds,
    }

    # Write per-p artifacts
    art_dir = out_dir / "day31_5"
    art_dir.mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(art_dir / "dem_corr_stats.json", "w") as f:
        _json.dump(dem_corr_stats_all, f, indent=2, default=str)
    with open(art_dir / "bin_counts.json", "w") as f:
        _json.dump(bin_counts, f, indent=2)
    with open(art_dir / "p_grid_result.json", "w") as f:
        _json.dump(grid_result.to_dict(), f, indent=2, default=str)
    if seed_results:
        with open(art_dir / "seed_results.json", "w") as f:
            _json.dump(seed_results, f, indent=2)
    if ci_summary:
        with open(art_dir / "ci_summary.json", "w") as f:
            _json.dump(ci_summary, f, indent=2)

    # Day 34: Ranking diagnostics artifact
    ranking_artifact = {}
    for pk, pinfo in per_p_info.items():
        ranking_artifact[pk] = {
            "p": pinfo["p"],
            "fg_auroc": pinfo.get("fg_auroc"),
            "fg_pr_auc": pinfo.get("fg_pr_auc"),
            "fg1_auroc": pinfo.get("fg1_auroc"),
            "fg1_pr_auc": pinfo.get("fg1_pr_auc"),
        }
    with open(art_dir / "ranking_diagnostics.json", "w") as f:
        _json.dump(ranking_artifact, f, indent=2, default=str)

    return rows, info_report


# ---------------------------------------------------------------------------
# Quality Gates
# ---------------------------------------------------------------------------

def run_quality_gates(all_rows: List[BenchV3Row]) -> Dict[str, Any]:
    """Run scientific quality gates on benchmark results."""
    gates: Dict[str, Any] = {"pass": True, "checks": []}

    def gate(name: str, ok: bool, msg: str = "", status: str = ""):
        entry = {"name": name, "ok": ok, "msg": msg}
        if status:
            entry["status"] = status  # PASS/WARN/FAIL
        gates["checks"].append(entry)
        if not ok:
            gates["pass"] = False
            print(f"  FAIL: {name} — {msg}")
        else:
            s_label = f" ({status})" if status == "WARN" else ""
            print(f"  PASS: {name}{s_label}")

    # 1. NaN check
    for r in all_rows:
        has_nan = any(math.isnan(v) for v in [r.f1, r.precision, r.recall_tpr, r.fpr])
        if has_nan:
            gate(f"no_nan_{r.suite}_{r.decoder}", False, f"NaN in metrics for {r.label}")
            break
    else:
        gate("no_nan", True)

    # 2. Suite B: degradation at extremes
    b_rows = [r for r in all_rows if r.suite == "B_P_SWEEP" and r.status == "pass"]
    if b_rows:
        oracle_rows = [r for r in all_rows if r.suite == "A_ORACLE"
                       and r.decoder == "MWPM_ORACLE" and r.status == "pass"]
        if oracle_rows:
            oracle_f1 = oracle_rows[0].f1
            extreme_f1s = [r.f1 for r in b_rows if r.p_scale >= 50 or r.p_scale <= 0.1]
            if extreme_f1s:
                worst = min(extreme_f1s)
                degraded = worst < oracle_f1 - 0.01
                gate("p_sweep_degradation", degraded,
                     f"Extreme p_scale F1={worst:.2%} vs oracle={oracle_f1:.2%}")

    # 3. Dataset hash consistency within suite
    all_suite_names = list(set(r.suite for r in all_rows))
    for suite_name in all_suite_names:
        suite_rows = [r for r in all_rows if r.suite == suite_name
                      and r.dataset_hash and r.label != "aggregate"]
        if len(suite_rows) >= 2:
            hashes = {r.dataset_hash for r in suite_rows}
            gate(f"hash_consistency_{suite_name}", len(hashes) <= 1,
                 f"{len(hashes)} different hashes")

    # 4. At least one row per suite passes
    for suite_name in all_suite_names:
        suite_rows = [r for r in all_rows if r.suite == suite_name]
        if suite_rows:
            any_pass = any(r.status == "pass" for r in suite_rows)
            gate(f"suite_pass_{suite_name}", any_pass)

    # 5. TPR sanity: at least one row should have recall > 0
    passing = [r for r in all_rows if r.status == "pass"]
    if passing:
        any_tpr = any(r.recall_tpr > 0 for r in passing)
        n_zero_tpr = sum(1 for r in passing if r.recall_tpr == 0)
        if any_tpr:
            gate("tpr_nonzero", True,
                 f"{len(passing) - n_zero_tpr}/{len(passing)} passing rows have recall_tpr>0")
        else:
            gate("tpr_nonzero", False,
                 "All passing rows have recall_tpr=0 — likely metric bug")

    # 5b. Metric integrity: TPR=0 but F1>0 suggests key mapping bug
    if passing:
        suspicious = [r for r in passing if r.recall_tpr == 0 and r.f1 > 0]
        if not suspicious:
            gate("metric_integrity", True, "All rows consistent (F1>0 → TPR>0)")
        else:
            gate("metric_integrity", False,
                 f"{len(suspicious)} rows have F1>0 but TPR=0 — metric mapping error")

    # 6. DEM graph hash populated for at least one MWPM row
    mwpm_rows = [r for r in all_rows if "MWPM" in r.decoder and r.status == "pass"]
    if mwpm_rows:
        any_hash = any(r.dem_graph_hash for r in mwpm_rows)
        n_with_hash = sum(1 for r in mwpm_rows if r.dem_graph_hash)
        hash_errors = [r.extra.get("dem_hash_error", "") for r in mwpm_rows
                       if r.extra.get("dem_hash_error")]
        if any_hash:
            msg = f"{n_with_hash}/{len(mwpm_rows)} MWPM rows have dem_graph_hash"
            if hash_errors:
                msg += f" (with {len(hash_errors)} hash build errors)"
            gate("dem_hash_populated", True, msg)
        else:
            msg = "No MWPM rows have dem_graph_hash"
            if hash_errors:
                msg += f" — errors: {hash_errors[0]}"
            gate("dem_hash_populated", False, msg)

    return gates


def run_fg_gates(all_rows: List[BenchV3Row]) -> Dict[str, Any]:
    """
    Decoder-specific quality gates for FG rows (Day 33.6).

    Ensures FG v1 (and v0) failures are NOT masked by passing MWPM rows.
    Returns dict with {"pass": bool, "checks": [{"name", "ok", "msg"}, ...]}.
    """
    gates: Dict[str, Any] = {"pass": True, "checks": []}

    def gate(name: str, ok: bool, msg: str = ""):
        gates["checks"].append({"name": name, "ok": ok, "msg": msg})
        if not ok:
            gates["pass"] = False

    # Collect FG rows (both v0 and v1)
    fg_rows = [r for r in all_rows
                if "FG_DEM" in r.decoder and r.status == "pass"]
    fg1_rows = [r for r in fg_rows if "V1" in r.decoder]

    if not fg_rows:
        gate("fg_rows_exist", False, "No passing FG rows found")
        return gates

    gate("fg_rows_exist", True,
         f"{len(fg_rows)} FG rows ({len(fg1_rows)} v1)")

    # Gate 1: fg_no_majority_collapse — FAIL if TPR==0 OR PPR==0
    collapsed = []
    for r in fg_rows:
        ppr = float(r.extra.get("pred_positive_rate", 0.0))
        if r.recall_tpr == 0 or ppr == 0:
            collapsed.append(f"{r.label}(TPR={r.recall_tpr:.2%},PPR={ppr:.2%})")
    if not collapsed:
        gate("fg_no_majority_collapse", True,
             f"All {len(fg_rows)} FG rows have TPR>0 and PPR>0")
    else:
        gate("fg_no_majority_collapse", False,
             f"{len(collapsed)} FG rows collapsed to all-negative: "
             + "; ".join(collapsed[:3]))

    # Gate 2: fg_no_reverse_collapse — FAIL if PPR>0.95 OR FPR>0.70
    reverse = []
    for r in fg_rows:
        ppr = float(r.extra.get("pred_positive_rate", 0.0))
        if ppr > 0.95 or r.fpr > 0.70:
            reverse.append(
                f"{r.label}(PPR={ppr:.2%},FPR={r.fpr:.2%})")
    if not reverse:
        gate("fg_no_reverse_collapse", True,
             f"All {len(fg_rows)} FG rows have PPR≤95% and FPR≤70%")
    else:
        gate("fg_no_reverse_collapse", False,
             f"{len(reverse)} FG rows reverse-collapsed: "
             + "; ".join(reverse[:3]))

    # Gate 3: fg_metric_integrity — FAIL if F1>0 but TPR==0
    integrity_fail = []
    for r in fg_rows:
        if r.f1 > 0 and r.recall_tpr == 0:
            integrity_fail.append(
                f"{r.label}(F1={r.f1:.2%},TPR={r.recall_tpr:.2%})")
    if not integrity_fail:
        gate("fg_metric_integrity", True,
             f"All {len(fg_rows)} FG rows consistent (F1>0 → TPR>0)")
    else:
        gate("fg_metric_integrity", False,
             f"{len(integrity_fail)} FG rows have F1>0 but TPR=0: "
             + "; ".join(integrity_fail[:3]))

    # Gate 4: fg_calibration_not_degenerate — FAIL if any FG row used DEGENERATE_FAIL
    degen = []
    for r in fg_rows:
        cal = r.extra.get("calibration", {})
        if cal.get("metric") == "DEGENERATE_FAIL":
            degen.append(r.label)
    if not degen:
        gate("fg_calibration_not_degenerate", True,
             f"All {len(fg_rows)} FG rows found feasible calibration")
    else:
        gate("fg_calibration_not_degenerate", False,
             f"{len(degen)} FG rows had degenerate calibration: "
             + "; ".join(degen[:3]))

    return gates

# ---------------------------------------------------------------------------
# Informativeness Gates (Day 31)
# ---------------------------------------------------------------------------

@dataclass
class InformativenessGate:
    """Tri-state gate: PASS / WARN / FAIL with reason code."""
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    reason_code: str
    msg: str
    per_p_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_informativeness_gates(
    info_report: Dict[str, Any],
    out_dir: Optional[Path] = None,
    min_y_rate: float = 0.01,
    max_density: float = 0.22,
    trivial_threshold: float = 0.60,
    corr_mass_fail: float = 0.02,
    corr_mass_warn: float = 0.05,
    bin_warn_samples: int = 2048,
    bin_fail_samples: int = 512,
) -> Dict[str, Any]:
    """Run informativeness quality gates on Suite D v2 results.

    Day 31 gates:
      1. trivial_regime — FAIL if y_rate < min_y_rate for majority
      2. saturated_regime — WARN if density > max_density for any p
      3. suite_d_not_informative — FAIL if >=60% of p points are trivial/saturated

    Day 31.5 gates:
      4. corr_mass_too_low — FAIL if k>2 mass ratio < corr_mass_fail
      5. mwpm_trivial_regime — FAIL if any selected p has probe F1 > 0.995
      6. candidate_rejection_rate_high — WARN if > 60% candidates rejected
      7. p_bin_min_samples — WARN/FAIL if bins have too few samples
      8. gnn_collapse_guard — WARN/FAIL if pred_positive_rate < 0.5% or > 95%
      9. oracle_vs_mismatch_inconclusive — WARN if CI spans 0 (long mode)

    Returns dict with overall status and per-gate details.
    """
    gates: List[InformativenessGate] = []
    grid_result = info_report.get("p_grid_result", {})
    per_p_stats = grid_result.get("per_p_stats", [])

    if not per_p_stats:
        g = InformativenessGate(
            name="trivial_regime", status="FAIL",
            reason_code="NO_P_SCANNED",
            msg="No p values were pre-scanned",
        )
        gates.append(g)
        result = _build_info_result(gates)
        if out_dir:
            _write_debug_informativeness(result, out_dir)
        return result

    # 1. trivial_regime: count p points with y_rate < min_y_rate
    trivial_count = sum(1 for s in per_p_stats if s.get("y_rate", 0) < min_y_rate)
    trivial_frac = trivial_count / len(per_p_stats)
    if trivial_frac > trivial_threshold:
        gates.append(InformativenessGate(
            name="trivial_regime", status="FAIL",
            reason_code=RC.TRIVIAL_REGIME,
            msg=f"{trivial_count}/{len(per_p_stats)} p points have y_rate<{min_y_rate}",
            per_p_stats={"trivial_count": trivial_count, "total": len(per_p_stats)},
        ))
    elif trivial_count > 0:
        gates.append(InformativenessGate(
            name="trivial_regime", status="WARN",
            reason_code=RC.TRIVIAL_REGIME,
            msg=f"{trivial_count}/{len(per_p_stats)} p points are trivial",
            per_p_stats={"trivial_count": trivial_count, "total": len(per_p_stats)},
        ))
    else:
        gates.append(InformativenessGate(
            name="trivial_regime", status="PASS",
            reason_code=RC.OK, msg="No trivial p points",
        ))

    # 2. saturated_regime: any p with density > max_density
    saturated_p = [s for s in per_p_stats
                   if s.get("detector_density", 0) > max_density]
    if saturated_p:
        gates.append(InformativenessGate(
            name="saturated_regime", status="WARN",
            reason_code=RC.SATURATED_REGIME,
            msg=f"{len(saturated_p)} p points have density>{max_density}",
            per_p_stats={"saturated_p": [s.get("p") for s in saturated_p]},
        ))
    else:
        gates.append(InformativenessGate(
            name="saturated_regime", status="PASS",
            reason_code=RC.OK, msg="No saturated p points",
        ))

    # 3. suite_d_not_informative: >=60% trivial OR saturated
    bad_count = trivial_count + len(saturated_p)
    bad_frac = bad_count / len(per_p_stats) if per_p_stats else 1.0
    n_selected = len(grid_result.get("p_grid", []))
    if n_selected < 3 or bad_frac >= trivial_threshold:
        gates.append(InformativenessGate(
            name="suite_d_not_informative", status="FAIL",
            reason_code=RC.INSUFFICIENT_INFORMATIVE,
            msg=f"Only {n_selected} informative p points ({bad_count}/{len(per_p_stats)} bad)",
            per_p_stats={"n_selected": n_selected, "bad_count": bad_count},
        ))
    else:
        gates.append(InformativenessGate(
            name="suite_d_not_informative", status="PASS",
            reason_code=RC.OK,
            msg=f"{n_selected} informative p points selected",
        ))

    # --- Day 31.5 gates ---

    # 4. corr_mass_too_low: check DEM correlation mass ratio
    dem_corr = info_report.get("dem_corr_stats", {})
    if dem_corr:
        mass_ratios = [v.get("k_gt_2_mass_ratio", 0.0)
                       for v in dem_corr.values()
                       if isinstance(v, dict) and "k_gt_2_mass_ratio" in v]
        if mass_ratios:
            avg_mass = float(np.mean(mass_ratios))
            min_mass = float(min(mass_ratios))
            if min_mass < corr_mass_fail:
                gates.append(InformativenessGate(
                    name="corr_mass_too_low", status="FAIL",
                    reason_code=RC.CORR_MASS_TOO_LOW,
                    msg=f"Min k>2 mass ratio={min_mass:.4f} < {corr_mass_fail}",
                    per_p_stats={"mass_ratios": mass_ratios, "avg": avg_mass},
                ))
            elif min_mass < corr_mass_warn:
                gates.append(InformativenessGate(
                    name="corr_mass_too_low", status="WARN",
                    reason_code=RC.CORR_MASS_TOO_LOW,
                    msg=f"Min k>2 mass ratio={min_mass:.4f} < {corr_mass_warn}",
                    per_p_stats={"mass_ratios": mass_ratios, "avg": avg_mass},
                ))
            else:
                gates.append(InformativenessGate(
                    name="corr_mass_too_low", status="PASS",
                    reason_code=RC.OK,
                    msg=f"k>2 mass ratio OK (min={min_mass:.4f})",
                ))

    # 5. mwpm_trivial_regime: any selected p has MWPM probe F1 > 0.995
    probe_f1s = [s.get("mwpm_probe_f1", -1) for s in per_p_stats
                 if s.get("accepted", False) and s.get("mwpm_probe_f1", -1) >= 0]
    if probe_f1s:
        trivial_probes = [f1 for f1 in probe_f1s if f1 > 0.995]
        if trivial_probes:
            gates.append(InformativenessGate(
                name="mwpm_trivial_regime", status="FAIL",
                reason_code=RC.MWPM_TRIVIAL,
                msg=f"{len(trivial_probes)} selected p(s) have MWPM probe F1>{0.995}",
                per_p_stats={"trivial_f1s": trivial_probes},
            ))
        else:
            gates.append(InformativenessGate(
                name="mwpm_trivial_regime", status="PASS",
                reason_code=RC.OK,
                msg=f"All {len(probe_f1s)} probes below trivial threshold",
            ))

    # 6. candidate_rejection_rate_high
    reject_rate = grid_result.get("candidate_reject_rate", 0.0)
    if reject_rate > 0.60:
        gates.append(InformativenessGate(
            name="candidate_rejection_rate_high", status="WARN",
            reason_code=RC.CANDIDATE_REJECTION_HIGH,
            msg=f"Candidate reject rate={reject_rate:.1%}",
        ))
    else:
        gates.append(InformativenessGate(
            name="candidate_rejection_rate_high", status="PASS",
            reason_code=RC.OK,
            msg=f"Candidate reject rate={reject_rate:.1%}",
        ))

    # 7. p_bin_min_samples
    bin_counts = info_report.get("bin_counts", {})
    if bin_counts:
        min_bin = min(bin_counts.values()) if bin_counts else 0
        if min_bin < bin_fail_samples:
            gates.append(InformativenessGate(
                name="p_bin_min_samples", status="FAIL",
                reason_code=RC.P_BIN_LOW_SAMPLES,
                msg=f"Min bin count={min_bin} < {bin_fail_samples}",
                per_p_stats={"bin_counts": bin_counts},
            ))
        elif min_bin < bin_warn_samples:
            gates.append(InformativenessGate(
                name="p_bin_min_samples", status="WARN",
                reason_code=RC.P_BIN_LOW_SAMPLES,
                msg=f"Min bin count={min_bin} < {bin_warn_samples}",
                per_p_stats={"bin_counts": bin_counts},
            ))
        else:
            gates.append(InformativenessGate(
                name="p_bin_min_samples", status="PASS",
                reason_code=RC.OK,
                msg=f"All bins >= {bin_warn_samples} samples",
            ))

    # 8. gnn_collapse_guard
    gnn_diags = info_report.get("gnn_diagnostics", [])
    if gnn_diags:
        collapsed = [d for d in gnn_diags if d.get("collapse_warn", False)]
        if collapsed:
            pprs = [d["pred_positive_rate"] for d in collapsed]
            any_extreme = any(ppr < 0.005 or ppr > 0.95 for ppr in pprs)
            status = "FAIL" if any_extreme else "WARN"
            code = RC.GNN_COLLAPSE_LOW if any(p < 0.005 for p in pprs) else RC.GNN_COLLAPSE_HIGH
            gates.append(InformativenessGate(
                name="gnn_collapse_guard", status=status,
                reason_code=code,
                msg=f"{len(collapsed)} p(s) with GNN collapse risk",
                per_p_stats={"pprs": pprs},
            ))
        else:
            gates.append(InformativenessGate(
                name="gnn_collapse_guard", status="PASS",
                reason_code=RC.OK,
                msg=f"No GNN collapse across {len(gnn_diags)} p points",
            ))

    # 9. oracle_vs_mismatch_inconclusive (CI, long mode)
    ci_summary = info_report.get("ci_summary", {})
    if ci_summary:
        if ci_summary.get("spans_zero", False):
            gates.append(InformativenessGate(
                name="oracle_vs_mismatch_inconclusive", status="WARN",
                reason_code=RC.ORACLE_VS_MISMATCH_INCONCLUSIVE,
                msg=f"CI spans 0: [{ci_summary['ci_95_lo']:+.4f}, {ci_summary['ci_95_hi']:+.4f}]",
                per_p_stats=ci_summary,
            ))
        else:
            gates.append(InformativenessGate(
                name="oracle_vs_mismatch_inconclusive", status="PASS",
                reason_code=RC.OK,
                msg=f"CI does not span 0: [{ci_summary['ci_95_lo']:+.4f}, {ci_summary['ci_95_hi']:+.4f}]",
            ))

    result = _build_info_result(gates)

    # Write debug artifact on FAIL
    has_fail = any(g.status == "FAIL" for g in gates)
    if has_fail and out_dir:
        _write_debug_informativeness(result, out_dir)

    return result


def _build_info_result(gates: List[InformativenessGate]) -> Dict[str, Any]:
    overall = "PASS"
    for g in gates:
        if g.status == "FAIL":
            overall = "FAIL"
            break
        if g.status == "WARN" and overall == "PASS":
            overall = "WARN"
    return {
        "overall": overall,
        "gates": [g.to_dict() for g in gates],
    }


def _write_debug_informativeness(
    result: Dict[str, Any], out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "debug_informativeness.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  [DEBUG] Wrote {path}")


# ---------------------------------------------------------------------------
# Artifact Writers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def write_v3_artifacts(results: Dict[str, Any], out_dir: Path):
    """Write all benchmark artifacts."""
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

    # 4. quality_gates.json (Day 31)
    qg_path = out_dir / "quality_gates.json"
    with open(qg_path, "w", encoding="utf-8") as f:
        qg_data = results.get("quality_gates", {})
        if results.get("informativeness_gates"):
            qg_data["informativeness"] = results["informativeness_gates"]
        json.dump(qg_data, f, indent=2, default=str)

    # 5. summary.md
    summary_path = out_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark v3 Results\n\n")
        all_suite_names = sorted(set(r.get("suite", "") for r in rows))
        for suite_name in all_suite_names:
            suite_rows = [r for r in rows if r.get("suite") == suite_name]
            if not suite_rows:
                continue
            f.write(f"## {suite_name}\n\n")
            f.write("| Decoder | Label | F1 | Prec | TPR | FPR | BalAcc | Status |\n")
            f.write("|---------|-------|-----|------|-----|-----|--------|--------|\n")
            for r in suite_rows:
                f.write(f"| {r['decoder']} | {r.get('label','')} | "
                        f"{r['f1']:.2%} | {r['precision']:.2%} | "
                        f"{r['recall_tpr']:.2%} | {r['fpr']:.2%} | "
                        f"{r['balanced_accuracy']:.2%} | {r['status']} |\n")
            f.write("\n")

        # Quality gates
        gates = results.get("quality_gates", {})
        if gates:
            f.write("## Quality Gates\n\n")
            f.write(f"**Overall: {'PASS' if gates.get('pass') else 'FAIL'}**\n\n")
            for c in gates.get("checks", []):
                icon = "✓" if c["ok"] else "✗"
                f.write(f"- {icon} {c['name']}")
                if c.get("msg"):
                    f.write(f" — {c['msg']}")
                f.write("\n")

        # Informativeness gates
        info_gates = results.get("informativeness_gates", {})
        if info_gates:
            f.write(f"\n## Informativeness Gates (Day 31)\n\n")
            f.write(f"**Overall: {info_gates.get('overall', 'N/A')}**\n\n")
            for g in info_gates.get("gates", []):
                icon = "✓" if g["status"] == "PASS" else ("⚠" if g["status"] == "WARN" else "✗")
                f.write(f"- {icon} [{g['status']}] {g['name']}: {g['msg']}\n")

    # 6. latency.json (Day 31)
    lat_path = out_dir / "latency.json"
    lat_data = results.get("latency", {})
    if lat_data:
        with open(lat_path, "w", encoding="utf-8") as f:
            json.dump(lat_data, f, indent=2, default=str)

    # 7. checksums
    cs_path = out_dir / "checksums.sha256"
    files = [json_path, csv_path, cfg_path, qg_path, summary_path]
    if lat_data:
        files.append(lat_path)
    with open(cs_path, "w") as f:
        for p in files:
            if p.exists():
                f.write(f"{_sha256_file(p)}  {p.name}\n")

    print(f"\nArtifacts written to {out_dir}/")
    for p in files + [cs_path]:
        if p.exists():
            print(f"  {p.name}")

# ---------------------------------------------------------------------------
# Inference-Only Latency
# ---------------------------------------------------------------------------

def _measure_inference_latency(
    baseline_dataset: Optional[ShardDataset],
    corr_dataset: Optional[ShardDataset],
    cfg: BenchV3Config,
) -> Dict[str, Any]:
    """Measure inference-only latency for MWPM (no training phase counted).

    Uses a subsample for fair measurement. Reports per-sample ms.
    """
    from qec_noise_factory.ml.bench.latency_v2 import measure_decoder_latency

    results: Dict[str, Any] = {}
    LATENCY_SUBSAMPLE = 2048

    ds = baseline_dataset or corr_dataset
    if ds is None:
        return results

    sub = _subsample(ds, LATENCY_SUBSAMPLE, cfg.seed)
    X = sub.X
    first_meta = sub.meta[0]
    params = params_from_canonical(first_meta.params_canonical)

    # MWPM Oracle latency
    try:
        mwpm = MWPMDecoder()
        nm = params.get("noise_model", "baseline_symmetric")
        mwpm.build(
            distance=params["distance"], rounds=params["rounds"],
            p=params["p"], basis=params["basis"], noise_model=nm,
        )

        def mwpm_decode(batch_X):
            return mwpm.decode_batch_fast(batch_X)

        lat = measure_decoder_latency(
            mwpm_decode, X, warmup=min(256, X.shape[0] // 4),
            decoder_name="MWPM_ORACLE",
        )
        results["mwpm_oracle"] = lat
        print(f"  MWPM_ORACLE: {lat['mean_ms']:.3f}ms/sample, "
              f"{lat['throughput_sps']:.0f} samples/s")
    except Exception as e:
        results["mwpm_oracle_error"] = str(e)

    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_benchmark_v3(cfg: BenchV3Config) -> Dict[str, Any]:
    """Run the full Day 30 benchmark v3."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    out_dir = project_root / cfg.out_dir

    print("=" * 70)
    print(f"Day 30 Benchmark v3 (d={cfg.distance}, basis={cfg.basis})")
    print(f"Suites: {', '.join(cfg.suites)}")
    print(f"Epochs: {cfg.effective_epochs} ({'smoke' if cfg.smoke else 'full'})")
    print("=" * 70)

    all_rows: List[BenchV3Row] = []

    # Load baseline data (Suites A/B/C)
    baseline_dataset = None
    if any(s in cfg.suites for s in ["A_ORACLE", "B_P_SWEEP", "C_MODEL_MIS"]):
        print(f"\n[1] Loading baseline data from {cfg.data_root}...")
        baseline_dataset = _load_dataset(cfg.data_root, cfg.distance)
        if baseline_dataset is not None:
            baseline_dataset = _subsample(baseline_dataset, cfg.limit_samples, cfg.seed)
            print(f"    {baseline_dataset.X.shape[0]:,} samples, "
                  f"{baseline_dataset.X.shape[1]} detectors")

    # Load correlated data (Suite D or D_v2)
    corr_dataset = None
    needs_corr = ("D_CORRELATED" in cfg.suites
                  or "D_CORRELATED_V2" in cfg.suites)
    if needs_corr:
        print(f"\n[2] Loading correlated data from {cfg.corr_data_root}...")
        corr_dataset = _load_dataset(cfg.corr_data_root, cfg.distance)
        if corr_dataset is not None:
            corr_dataset = _subsample(corr_dataset, cfg.limit_samples, cfg.seed)
            print(f"    {corr_dataset.X.shape[0]:,} samples, "
                  f"{corr_dataset.X.shape[1]} detectors")

    # Run suites
    if "A_ORACLE" in cfg.suites and baseline_dataset is not None:
        all_rows.extend(run_suite_a(cfg, baseline_dataset, out_dir))

    if "B_P_SWEEP" in cfg.suites and baseline_dataset is not None:
        all_rows.extend(run_suite_b(cfg, baseline_dataset, out_dir))

    if "C_MODEL_MIS" in cfg.suites and baseline_dataset is not None:
        all_rows.extend(run_suite_c(cfg, baseline_dataset, out_dir))

    if "D_CORRELATED" in cfg.suites and corr_dataset is not None:
        all_rows.extend(run_suite_d(cfg, corr_dataset, out_dir))

    # Suite D v2 — Per-p evaluation (Day 31)
    info_report = {}
    if "D_CORRELATED_V2" in cfg.suites and corr_dataset is not None:
        d_v2_rows, info_report = run_suite_d_v2(cfg, corr_dataset, out_dir)
        all_rows.extend(d_v2_rows)

    # Quality gates
    print("\n" + "=" * 60)
    print("Quality Gates")
    print("=" * 60)
    gates = run_quality_gates(all_rows)

    # Informativeness gates (Day 31)
    info_gates = {}
    if info_report:
        print("\n" + "=" * 60)
        print("Informativeness Gates")
        print("=" * 60)
        info_gates = run_informativeness_gates(info_report, out_dir)
        for g in info_gates.get("gates", []):
            icon = "✓" if g["status"] == "PASS" else ("⚠" if g["status"] == "WARN" else "✗")
            print(f"  {icon} [{g['status']}] {g['name']}: {g['msg']}")

    # Inference-only latency phase
    latency_results = {}
    if all_rows:
        print("\n" + "=" * 60)
        print("Inference-Only Latency")
        print("=" * 60)
        latency_results = _measure_inference_latency(
            baseline_dataset, corr_dataset, cfg)

    # Provenance
    import subprocess
    try:
        code_version = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        code_version = "unknown"

    # Build results
    split_hash = ""
    if all_rows:
        # Hash of all dataset hashes → proves same split across decoders
        h = hashlib.sha256()
        for r in all_rows:
            h.update(r.dataset_hash.encode())
        split_hash = h.hexdigest()[:16]

    results = {
        "config": cfg.to_dict(),
        "rows": [r.to_dict() for r in all_rows],
        "quality_gates": gates,
        "informativeness_gates": info_gates,
        "latency": latency_results,
        "status": "pass" if gates["pass"] else "fail",
        "suites_run": cfg.suites,
        "total_rows": len(all_rows),
        "passed_rows": sum(1 for r in all_rows if r.status == "pass"),
        "code_version": code_version,
        "split_hash": split_hash,
        "seed": cfg.seed,
    }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Day 30/31 Benchmark v3")
    parser.add_argument("--suites", type=str,
                        default=",".join(SUITE_NAMES),
                        help="Comma-separated suite names (incl D_CORRELATED_V2)")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--basis", type=str, default="X")
    parser.add_argument("--data-root", type=str,
                        default="output/data/surface_v0/shards")
    parser.add_argument("--corr-data-root", type=str,
                        default="output/data/surface_corr_v0/shards")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--smoke", action="store_true",
                        help="Use 3 epochs for smoke testing")
    parser.add_argument("--long", action="store_true",
                        help="Heavy mode: 30 epochs, full data (Day 31)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds for CI (default 5 in --long)")
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="ml_artifacts/day30_bench_v3")
    args = parser.parse_args(argv)

    suites = [s.strip() for s in args.suites.split(",")]

    # Auto-add Suite D v2 in long mode for Day 31.5 features
    if getattr(args, 'long', False) and "D_CORRELATED_V2" not in suites:
        suites.append("D_CORRELATED_V2")

    cfg = BenchV3Config(
        suites=suites,
        distance=args.distance,
        basis=args.basis,
        data_root=args.data_root,
        corr_data_root=args.corr_data_root,
        epochs=args.epochs,
        smoke=args.smoke,
        long=getattr(args, 'long', False),
        seeds=getattr(args, 'seeds', 1),
        limit_samples=args.limit_samples,
        seed=args.seed,
        out_dir=args.out,
    )

    results = run_benchmark_v3(cfg)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    write_v3_artifacts(results, project_root / cfg.out_dir)

    # Summary
    print(f"\n{'='*70}")
    print(f"Benchmark v3 complete: {results['passed_rows']}/{results['total_rows']} passed")
    print(f"Quality gates: {'PASS' if results['quality_gates']['pass'] else 'FAIL'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
