"""
Cross-Model Generalization Suite — Day 19

Runs structured experiments to measure decoder generalization:
  - CROSS_MODEL: train on some physics_hash groups, test on others
  - OOD_P_RANGE: train on low-p, test on high-p (or vice versa)

Each experiment:
  1. Loads + splits data with leakage asserts
  2. Trains MLP + GNN (short runs)
  3. Produces unified reports with full provenance

Results saved to: ml_artifacts/generalization/exp_<id>/
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from qec_noise_factory.ml.data.reader import ShardDataset, merge_datasets
from qec_noise_factory.ml.data.splits import (
    split_blocks, split_within_model, split_cross_model, split_ood_p_range,
    SplitPolicy, SplitResult,
)
from qec_noise_factory.ml.graph.builder import build_graph, build_key_from_meta
from qec_noise_factory.ml.graph.hash import compute_graph_hash
from qec_noise_factory.ml.models.gnn import GNNDecoderV0, GNNDecoderV1, GNNDecoderV2, graph_spec_to_edge_index
from qec_noise_factory.ml.models.mlp import MLPDecoder
from qec_noise_factory.ml.train.trainer import make_dataloader, train_one_epoch, eval_one_epoch
from qec_noise_factory.ml.train.config import TrainConfig
from qec_noise_factory.ml.artifacts.run_logger import dataset_hash
from qec_noise_factory.ml.metrics.group_metrics import compute_group_metrics
from qec_noise_factory.ml.metrics.calibration import calibrate_threshold, compute_auto_pos_weight


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Definition of one generalization experiment."""
    exp_id: str                    # e.g. "EXP-19-A"
    name: str                      # human-readable description
    split_policy: str              # "cross_model" | "ood_p_range" | "within_model"
    epochs: int = 3                # short runs for generalization testing
    batch_size: int = 64
    lr: float = 1e-3
    hidden_dim: int = 32
    seed: int = 42
    train_ratio: float = 0.8       # train/test split ratio

    # OOD p-range params (only for ood_p_range)
    ood_test_p_lo: float = 0.3
    ood_test_p_hi: float = 1.0

    # Day 21: anti-collapse options
    loss_pos_weight: Optional[float] = None   # None=off, 0/negative→auto, float→fixed
    calibrate_threshold: bool = False          # grid-search for best threshold

    # Day 22: pos_weight stabilization
    pos_weight_max: float = 8.0               # max auto pos_weight clamp

    # Day 23: GNN v1 options
    gnn_version: str = "v0"                   # "v0" | "v1"
    gnn_readout: str = "mean"                 # "mean" | "mean_max" | "attn"
    gnn_feature_version: str = "v0"            # "v0" (F=1) | "v1" (F=6)
    gnn_dropout: float = 0.1

    # Day 24: OOD robustness options
    featureset: str = "v1_full"               # "v1_full" | "v1_nop" | "v1_nop_nodist"
    calibrate_metric: str = "bal_acc"          # "bal_acc" | "f1" | "bal_acc_minus_fpr"
    calibrate_lambda: float = 0.25            # FPR penalty weight for bal_acc_minus_fpr

    # Day 28: GNN v2 DEM edge-aware options
    graph_mode: str = "generic"               # "generic" (Day 17) | "dem" (Day 27)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentReport:
    """Result of one experiment."""
    exp_id: str
    name: str
    split_policy: str
    status: str                    # "pass" | "skip" | "fail"
    reason: str = ""               # skip/fail reason

    # Coverage
    train_physics: List[str] = field(default_factory=list)
    test_physics: List[str] = field(default_factory=list)
    train_p_range: Tuple[float, float] = (0.0, 0.0)
    test_p_range: Tuple[float, float] = (0.0, 0.0)
    train_samples: int = 0
    test_samples: int = 0

    # Model results
    mlp_metrics: Dict[str, Any] = field(default_factory=dict)
    gnn_metrics: Dict[str, Any] = field(default_factory=dict)
    mlp_runtime: float = 0.0
    gnn_runtime: float = 0.0

    # Provenance
    dataset_hash: str = ""
    graph_hash: str = ""
    leakage_verified: bool = False

    # Runtime metadata
    batch_size: int = 64
    device: str = "cpu"
    num_epochs: int = 0

    # Day 20.5: per-group metrics
    mlp_group_metrics: Optional[Dict] = None
    gnn_group_metrics: Optional[Dict] = None

    # Day 21: anti-collapse fields
    pos_weight_used: Optional[float] = None
    mlp_calibrated_threshold: Optional[float] = None
    gnn_calibrated_threshold: Optional[float] = None
    mlp_calibration_metric: str = ""
    gnn_calibration_metric: str = ""

    # Day 22: hardening fields
    pos_weight_auto: Optional[float] = None
    pos_weight_clamped: bool = False
    pos_weight_max: float = 8.0
    mlp_warnings: List[str] = field(default_factory=list)
    gnn_warnings: List[str] = field(default_factory=list)
    mlp_calibration_best_value: Optional[float] = None
    gnn_calibration_best_value: Optional[float] = None
    mlp_calibration_grid_size: int = 0
    gnn_calibration_grid_size: int = 0

    # Day 23: GNN v1 fields
    gnn_version: str = ""
    gnn_readout: str = ""
    feature_version: str = "v0"
    feature_dim: int = 1
    feature_names: List[str] = field(default_factory=list)

    # Day 24: OOD robustness fields
    featureset_name: str = ""
    calibrate_metric_used: str = ""
    calibrate_lambda: float = 0.0

    # Day 25: collapse guard + sweep fields
    mlp_pred_positive_rate: float = 0.0
    gnn_pred_positive_rate: float = 0.0
    mlp_true_positive_rate: float = 0.0
    gnn_true_positive_rate: float = 0.0
    gnn_fallback_applied: bool = False
    gnn_fallback_reason: str = ""
    gnn_fallback_metric_used: str = ""
    mlp_fallback_applied: bool = False
    mlp_fallback_reason: str = ""
    mlp_fallback_metric_used: str = ""

    # Day 28: GNN v2 DEM edge-aware fields
    graph_mode: str = ""
    uses_edge_weight: bool = False
    edge_attr_info: Dict[str, Any] = field(default_factory=dict)
    dem_graph_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["train_p_range"] = list(d["train_p_range"])
        d["test_p_range"] = list(d["test_p_range"])
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_xy_by_blocks(
    dataset: ShardDataset,
    block_set: set,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Extract X, Y arrays for blocks whose sample_key is in block_set."""
    x_parts, y_parts, kept_meta = [], [], []
    offset = 0
    for m in dataset.meta:
        n = m.record_count
        if m.sample_key in block_set:
            x_parts.append(dataset.X[offset:offset + n])
            y_parts.append(dataset.Y[offset:offset + n])
            kept_meta.append(m)
        offset += n

    if not x_parts:
        nd = dataset.X.shape[1] if dataset.X.ndim > 1 else 0
        no = dataset.Y.shape[1] if dataset.Y.ndim > 1 else 0
        return np.empty((0, nd), dtype=bool), np.empty((0, no), dtype=bool), []

    return np.concatenate(x_parts), np.concatenate(y_parts), kept_meta


def _physics_coverage(blocks) -> List[str]:
    """Get unique physics identifiers from blocks."""
    coverage = set()
    for b in blocks:
        key = b.physics_hash or b.physics_model_name or f"{b.pack_name}:p={b.p:.4f}"
        coverage.add(key)
    return sorted(coverage)


def _p_range(blocks) -> Tuple[float, float]:
    """Get min/max p from blocks."""
    if not blocks:
        return (0.0, 0.0)
    ps = [b.p for b in blocks]
    return (min(ps), max(ps))


def _predict_all(
    model: nn.Module,
    loader,
    device: str = "cpu",
    edge_index=None,
    num_nodes: int = 0,
    feature_dim: int = 1,
    edge_weight=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect all predictions from a model for group metric analysis."""
    from qec_noise_factory.ml.train.trainer import _is_graph_model, _prepare_gnn_input
    model.eval()
    is_gnn = _is_graph_model(model)
    _uses_ew = getattr(model, "uses_edge_weight", False)
    all_true, all_pred = [], []
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            if is_gnn:
                node_feats, ei, ew = _prepare_gnn_input(
                    X_batch, num_nodes, feature_dim, edge_index, device,
                    edge_weight=edge_weight if _uses_ew else None,
                )
                if _uses_ew:
                    logits = model(node_feats, ei, edge_weight=ew)
                else:
                    logits = model(node_feats, ei)
            else:
                logits = model(X_batch)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(bool)
            all_true.append(Y_batch.numpy().astype(bool))
            all_pred.append(preds)
    return np.concatenate(all_true), np.concatenate(all_pred)


def _collect_logits(
    model: nn.Module,
    loader,
    device: str = "cpu",
    edge_index=None,
    num_nodes: int = 0,
    feature_dim: int = 1,
    edge_weight=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect raw logits (pre-sigmoid) and ground truth for calibration."""
    from qec_noise_factory.ml.train.trainer import _is_graph_model, _prepare_gnn_input
    model.eval()
    is_gnn = _is_graph_model(model)
    _uses_ew = getattr(model, "uses_edge_weight", False)
    all_true, all_logits = [], []
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            if is_gnn:
                node_feats, ei, ew = _prepare_gnn_input(
                    X_batch, num_nodes, feature_dim, edge_index, device,
                    edge_weight=edge_weight if _uses_ew else None,
                )
                if _uses_ew:
                    logits = model(node_feats, ei, edge_weight=ew)
                else:
                    logits = model(node_feats, ei)
            else:
                logits = model(X_batch)
            all_logits.append(logits.cpu().numpy())
            all_true.append(Y_batch.numpy().astype(bool))
    return np.concatenate(all_true), np.concatenate(all_logits)


def _collapse_guard(
    cal_result: Dict[str, Any],
    logits: np.ndarray,
    y_true: np.ndarray,
    original_metric: str,
) -> Tuple[Dict[str, Any], bool, str, str]:
    """
    Day 25: Collapse-safe calibration guard.

    After calibration, check if the chosen threshold causes collapse:
    - pred_positive_rate < 0.5% → all_negative_collapse_risk
    - pred_positive_rate > 95% → all_positive_collapse_risk

    If collapse detected, fallback to safer metric (f1, then bal_acc).

    Returns:
        (cal_result, fallback_applied, fallback_reason, fallback_metric_used)
    """
    from qec_noise_factory.ml.metrics.calibration import calibrate_threshold as _cal

    threshold = cal_result["best_threshold"]
    logits_f = np.asarray(logits, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits_f))
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    y_pred = (probs > threshold).astype(bool)
    pred_pos_rate = float(y_pred.mean())

    # Check for collapse
    reason = ""
    if pred_pos_rate < 0.005:
        reason = "all_negative_collapse_risk"
    elif pred_pos_rate > 0.95:
        reason = "all_positive_collapse_risk"

    if not reason:
        return cal_result, False, "", ""

    # Fallback: try f1, then bal_acc
    fallback_metrics = ["f1", "bal_acc"]
    # Remove original metric if it's in the fallback list
    fallback_metrics = [m for m in fallback_metrics if m != original_metric]

    for fb_metric in fallback_metrics:
        fb_result = _cal(logits, y_true, metric=fb_metric)
        fb_threshold = fb_result["best_threshold"]
        fb_pred = (probs > fb_threshold).astype(bool)
        fb_pos_rate = float(fb_pred.mean())
        if 0.005 <= fb_pos_rate <= 0.95:
            return fb_result, True, reason, fb_metric

    # All fallbacks also collapse — return best non-collapsing or original
    # Try bal_acc as last resort even if it was original
    if original_metric != "bal_acc":
        fb_result = _cal(logits, y_true, metric="bal_acc")
        return fb_result, True, reason, "bal_acc"

    return cal_result, True, reason, original_metric

def _check_collapse_warnings(metrics: Dict[str, Any]) -> List[str]:
    """Day 22: detect collapse patterns in metrics."""
    warnings = []
    tpr = metrics.get("obs_0_tpr", metrics.get("macro_tpr", -1))
    tnr = metrics.get("obs_0_tnr", metrics.get("macro_tnr", -1))
    if tpr >= 0.98 and tnr <= 0.20:
        warnings.append("reverse_collapse_risk")
    if tpr <= 0.02 and tnr >= 0.98:
        warnings.append("majority_collapse_risk")
    return warnings


def _build_v1_gnn_X(
    X: np.ndarray,
    meta_list: list,
    graph,
    featureset: str = "v1_full",
) -> np.ndarray:
    """
    Day 23/24: Build enriched GNN features per-block, then flatten.

    Supports feature gating via featureset param (Day 24).
    Returns flattened array (total_samples, N * F_gated) for dataloader.
    """
    import json
    from qec_noise_factory.ml.graph.features import (
        X_to_node_features_gated, FEATURESET_REGISTRY,
    )

    N = graph.num_nodes
    F_gated = FEATURESET_REGISTRY[featureset]["dim"]
    parts = []
    offset = 0

    for m in meta_list:
        n = m.record_count
        block_X = X[offset:offset + n]

        try:
            params = json.loads(m.params_canonical)
            circuit = params.get("circuit", {})
            basis = circuit.get("basis", "")
            distance = circuit.get("distance", 0)
        except (json.JSONDecodeError, AttributeError):
            basis, distance = "", 0

        feats = X_to_node_features_gated(
            block_X, graph, featureset=featureset,
            p_value=m.p, basis=basis, distance=distance,
        )
        parts.append(feats.reshape(n, N * F_gated))
        offset += n

    if offset < X.shape[0]:
        remainder = X.shape[0] - offset
        feats = X_to_node_features_gated(X[offset:], graph, featureset=featureset)
        parts.append(feats.reshape(remainder, N * F_gated))

    return np.concatenate(parts, axis=0).astype(np.float32)


def verify_disjointness(split: SplitResult) -> bool:
    """
    Verify train/test sets are disjoint according to policy.

    For CROSS_MODEL: no shared physics_hash.
    For OOD_P_RANGE: no shared p-bucket.
    For WITHIN_MODEL: no shared sample_key.
    """
    if split.policy == SplitPolicy.CROSS_MODEL:
        return split.leakage_check()
    elif split.policy == SplitPolicy.OOD_P_RANGE:
        return split.leakage_check()
    else:
        # WITHIN_MODEL: check sample_key disjointness
        train_keys = {b.sample_key for b in split.train_blocks}
        test_keys = {b.sample_key for b in split.test_blocks}
        return len(train_keys & test_keys) == 0


# ---------------------------------------------------------------------------
# Core: Run Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    config: ExperimentConfig,
    dataset: ShardDataset,
    out_dir: Path,
) -> ExperimentReport:
    """
    Run a single generalization experiment.

    Steps:
      1. Split data according to policy
      2. Assert disjointness (FAIL if violated)
      3. Train MLP + GNN (short runs)
      4. Evaluate both models
      5. Write reports

    Returns ExperimentReport.
    """
    report = ExperimentReport(
        exp_id=config.exp_id,
        name=config.name,
        split_policy=config.split_policy,
        batch_size=config.batch_size,
        device="cpu",
        num_epochs=config.epochs,
        status="fail",
    )

    # --- 1. Split ---
    policy_map = {
        "cross_model": SplitPolicy.CROSS_MODEL,
        "ood_p_range": SplitPolicy.OOD_P_RANGE,
        "within_model": SplitPolicy.WITHIN_MODEL,
    }
    policy = policy_map.get(config.split_policy)
    if policy is None:
        report.reason = f"Unknown split policy: {config.split_policy}"
        return report

    if policy == SplitPolicy.OOD_P_RANGE:
        split = split_ood_p_range(
            dataset.meta,
            test_p_lo=config.ood_test_p_lo,
            test_p_hi=config.ood_test_p_hi,
            seed=config.seed,
        )
    elif policy == SplitPolicy.CROSS_MODEL:
        split = split_cross_model(dataset.meta, train_ratio=config.train_ratio, seed=config.seed)
    else:
        split = split_within_model(dataset.meta, train_ratio=config.train_ratio, seed=config.seed)

    if not split.train_blocks or not split.test_blocks:
        report.status = "skip"
        report.reason = "Insufficient data for split (empty train or test)"
        return report

    # --- 2. Leakage assert ---
    disjoint = verify_disjointness(split)
    report.leakage_verified = disjoint
    if not disjoint:
        report.reason = f"LEAKAGE DETECTED under {config.split_policy}"
        return report

    # --- 3. Extract X, Y ---
    train_keys = {b.sample_key for b in split.train_blocks}
    test_keys = {b.sample_key for b in split.test_blocks}

    train_X, train_Y, train_meta = _extract_xy_by_blocks(dataset, train_keys)
    test_X, test_Y, test_meta = _extract_xy_by_blocks(dataset, test_keys)

    if train_X.shape[0] == 0 or test_X.shape[0] == 0:
        report.status = "skip"
        report.reason = "Empty train or test set after extraction"
        return report

    # Coverage
    report.train_physics = _physics_coverage(split.train_blocks)
    report.test_physics = _physics_coverage(split.test_blocks)
    report.train_p_range = _p_range(split.train_blocks)
    report.test_p_range = _p_range(split.test_blocks)
    report.train_samples = train_X.shape[0]
    report.test_samples = test_X.shape[0]
    report.dataset_hash = dataset_hash(dataset.meta, config.split_policy, split_seed=config.seed)

    num_det = train_X.shape[1]
    num_obs = train_Y.shape[1]

    # --- 4. Build graph ---
    if dataset.meta:
        build_key = build_key_from_meta(dataset.meta[0])
        graph = build_graph(build_key)
        graph_hash_val = compute_graph_hash(graph)
        edge_index = graph_spec_to_edge_index(graph)
        report.graph_hash = graph_hash_val
    else:
        report.status = "skip"
        report.reason = "No metadata to build graph"
        return report

    # Dataloaders
    train_loader = make_dataloader(train_X, train_Y, batch_size=config.batch_size, shuffle=True)
    test_loader = make_dataloader(test_X, test_Y, batch_size=config.batch_size, shuffle=False)

    # --- Day 21/22: compute pos_weight with clamping ---
    pw_tensor = None
    if config.loss_pos_weight is not None:
        pw_val = config.loss_pos_weight
        if pw_val <= 0:  # "auto" convention: pass 0 or negative
            pw_val = compute_auto_pos_weight(train_Y, clamp_max=config.pos_weight_max)
        report.pos_weight_auto = compute_auto_pos_weight(train_Y, clamp_max=999.0)  # unclamped
        report.pos_weight_used = pw_val
        report.pos_weight_clamped = (report.pos_weight_auto > config.pos_weight_max)
        report.pos_weight_max = config.pos_weight_max
        pw_tensor = torch.tensor([pw_val], dtype=torch.float32)

    # --- 5. Train MLP ---
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    mlp = MLPDecoder(
        input_dim=num_det, output_dim=num_obs,
        hidden_dim=config.hidden_dim, num_hidden_layers=2,
    )
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=config.lr)

    t0 = time.time()
    for _ in range(config.epochs):
        train_one_epoch(mlp, train_loader, mlp_opt, class_weights=pw_tensor)
    report.mlp_runtime = time.time() - t0

    # Day 21/22/25: MLP calibration with collapse guard
    if config.calibrate_threshold:
        mlp_true_cal, mlp_logits_cal = _collect_logits(mlp, test_loader)
        cal_result = calibrate_threshold(
            mlp_logits_cal, mlp_true_cal,
            metric=config.calibrate_metric,
            fpr_lambda=config.calibrate_lambda,
        )
        # Day 25: Collapse guard
        cal_result, fb_applied, fb_reason, fb_metric = _collapse_guard(
            cal_result, mlp_logits_cal, mlp_true_cal, config.calibrate_metric,
        )
        report.mlp_fallback_applied = fb_applied
        report.mlp_fallback_reason = fb_reason
        report.mlp_fallback_metric_used = fb_metric
        report.mlp_calibrated_threshold = cal_result["best_threshold"]
        report.mlp_calibration_metric = cal_result["calibration_metric"]
        report.mlp_calibration_best_value = cal_result["best_score"]
        report.mlp_calibration_grid_size = len(cal_result.get("scores", {}))
        report.calibrate_metric_used = config.calibrate_metric
        report.calibrate_lambda = config.calibrate_lambda
        # Re-evaluate with calibrated threshold
        probs = 1.0 / (1.0 + np.exp(-mlp_logits_cal.astype(np.float64)))
        y_pred_cal = (probs > cal_result["best_threshold"]).astype(bool)
        from qec_noise_factory.ml.metrics.classification import compute_metrics
        report.mlp_metrics = compute_metrics(mlp_true_cal, y_pred_cal)
        report.mlp_metrics["eval_loss"] = eval_one_epoch(mlp, test_loader).get("eval_loss", 0)
        report.mlp_metrics["calibrated_threshold"] = cal_result["best_threshold"]
        report.mlp_pred_positive_rate = float(y_pred_cal.mean())
        report.mlp_true_positive_rate = float(mlp_true_cal.astype(bool).mean())
    else:
        report.mlp_metrics = eval_one_epoch(mlp, test_loader)

    # Day 22: MLP collapse warnings
    report.mlp_warnings = _check_collapse_warnings(report.mlp_metrics)

    # --- 6. Train GNN (Day 23: v0/v1, Day 28: v2) ---
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    gnn_feature_dim = 1  # default v0
    gnn_edge_weight = None  # Day 28: DEM edge weights (V2 only)
    gnn_num_nodes = graph.num_nodes  # may change if boundary node added

    if config.gnn_feature_version == "v1":
        from qec_noise_factory.ml.graph.features import (
            FEATURESET_REGISTRY, get_featureset_meta, pad_boundary_node,
        )
        fs_meta = get_featureset_meta(config.featureset)
        gnn_feature_dim = fs_meta["feature_dim"]
        report.feature_version = fs_meta["feature_version"]
        report.feature_dim = fs_meta["feature_dim"]
        report.feature_names = list(fs_meta["feature_names"])
        report.featureset_name = config.featureset

        gnn_train_X = _build_v1_gnn_X(train_X, train_meta, graph, featureset=config.featureset)
        gnn_test_X = _build_v1_gnn_X(test_X, test_meta, graph, featureset=config.featureset)

        # Day 28: V2 DEM graph mode — pad boundary node + use DEM edge_index/weights
        if config.gnn_version == "v2" and config.graph_mode == "dem":
            from qec_noise_factory.ml.graph.dem_graph import (
                get_or_build_dem_graph, dem_graph_to_edge_index,
            )
            from qec_noise_factory.ml.stim.rebuild import params_from_canonical

            # Build DEM graph from first block metadata
            first_meta = dataset.meta[0]
            dem_params = params_from_canonical(first_meta.params_canonical)
            dem_spec = get_or_build_dem_graph(**dem_params)

            # Convert to bidirectional torch tensors
            dem_ei, dem_ew = dem_graph_to_edge_index(dem_spec)
            edge_index = dem_ei  # override generic edge_index
            gnn_edge_weight = dem_ew

            # Pad boundary node into features
            if dem_spec.has_boundary_node:
                N = graph.num_nodes
                F_gated = fs_meta["feature_dim"]
                # Reshape flat → (B, N, F), pad, flatten back
                train_3d = gnn_train_X.reshape(-1, N, F_gated)
                test_3d = gnn_test_X.reshape(-1, N, F_gated)
                train_3d = pad_boundary_node(train_3d, add_is_boundary_feature=True)
                test_3d = pad_boundary_node(test_3d, add_is_boundary_feature=True)
                gnn_num_nodes = N + 1
                gnn_feature_dim = F_gated + 1  # +1 for is_boundary
                gnn_train_X = train_3d.reshape(-1, gnn_num_nodes * gnn_feature_dim)
                gnn_test_X = test_3d.reshape(-1, gnn_num_nodes * gnn_feature_dim)
                report.feature_dim = gnn_feature_dim
                report.feature_names = list(fs_meta["feature_names"]) + ["is_boundary"]

            report.graph_mode = "dem"
            report.uses_edge_weight = True
            report.edge_attr_info = {
                "name": "dem_log_odds_weight",
                "dim": 1,
                "transform": "sigmoid",
            }
            report.dem_graph_hash = dem_spec.dem_graph_hash

        gnn_train_loader = make_dataloader(gnn_train_X, train_Y, batch_size=config.batch_size, shuffle=True)
        gnn_test_loader = make_dataloader(gnn_test_X, test_Y, batch_size=config.batch_size, shuffle=False)
    else:
        report.feature_version = "v0"
        report.feature_dim = 1
        report.feature_names = ["detector_bit"]
        report.featureset_name = "v0"
        gnn_train_loader = train_loader
        gnn_test_loader = test_loader

    if config.gnn_version == "v2":
        gnn = GNNDecoderV2(
            input_dim=gnn_feature_dim, output_dim=num_obs,
            hidden_dim=config.hidden_dim, num_mp_layers=2,
            readout=config.gnn_readout, dropout=config.gnn_dropout,
        )
        report.gnn_version = "v2"
        report.gnn_readout = config.gnn_readout
    elif config.gnn_version == "v1":
        gnn = GNNDecoderV1(
            input_dim=gnn_feature_dim, output_dim=num_obs,
            hidden_dim=config.hidden_dim, num_mp_layers=2,
            readout=config.gnn_readout, dropout=config.gnn_dropout,
        )
        report.gnn_version = "v1"
        report.gnn_readout = config.gnn_readout
    else:
        gnn = GNNDecoderV0(
            input_dim=gnn_feature_dim, output_dim=num_obs,
            hidden_dim=config.hidden_dim, num_mp_layers=2,
        )
        report.gnn_version = "v0"
        report.gnn_readout = "mean"

    gnn_opt = torch.optim.Adam(gnn.parameters(), lr=config.lr)

    t0 = time.time()
    for _ in range(config.epochs):
        train_one_epoch(
            gnn, gnn_train_loader, gnn_opt,
            class_weights=pw_tensor,
            edge_index=edge_index,
            num_nodes=gnn_num_nodes,
            feature_dim=gnn_feature_dim,
            edge_weight=gnn_edge_weight,
        )
    report.gnn_runtime = time.time() - t0

    # Day 21/22/25: GNN calibration with collapse guard
    if config.calibrate_threshold:
        gnn_true_cal, gnn_logits_cal = _collect_logits(
            gnn, gnn_test_loader,
            edge_index=edge_index,
            num_nodes=gnn_num_nodes,
            feature_dim=gnn_feature_dim,
            edge_weight=gnn_edge_weight,
        )
        cal_result_gnn = calibrate_threshold(
            gnn_logits_cal, gnn_true_cal,
            metric=config.calibrate_metric,
            fpr_lambda=config.calibrate_lambda,
        )
        # Day 25: Collapse guard
        cal_result_gnn, fb_applied, fb_reason, fb_metric = _collapse_guard(
            cal_result_gnn, gnn_logits_cal, gnn_true_cal, config.calibrate_metric,
        )
        report.gnn_fallback_applied = fb_applied
        report.gnn_fallback_reason = fb_reason
        report.gnn_fallback_metric_used = fb_metric
        report.gnn_calibrated_threshold = cal_result_gnn["best_threshold"]
        report.gnn_calibration_metric = cal_result_gnn["calibration_metric"]
        report.gnn_calibration_best_value = cal_result_gnn["best_score"]
        report.gnn_calibration_grid_size = len(cal_result_gnn.get("scores", {}))
        probs_gnn = 1.0 / (1.0 + np.exp(-gnn_logits_cal.astype(np.float64)))
        y_pred_gnn_cal = (probs_gnn > cal_result_gnn["best_threshold"]).astype(bool)
        from qec_noise_factory.ml.metrics.classification import compute_metrics as cm
        report.gnn_metrics = cm(gnn_true_cal, y_pred_gnn_cal)
        report.gnn_metrics["eval_loss"] = eval_one_epoch(
            gnn, gnn_test_loader,
            edge_index=edge_index, num_nodes=gnn_num_nodes,
            feature_dim=gnn_feature_dim,
            edge_weight=gnn_edge_weight,
        ).get("eval_loss", 0)
        report.gnn_metrics["calibrated_threshold"] = cal_result_gnn["best_threshold"]
        report.gnn_pred_positive_rate = float(y_pred_gnn_cal.mean())
        report.gnn_true_positive_rate = float(gnn_true_cal.astype(bool).mean())
    else:
        report.gnn_metrics = eval_one_epoch(
            gnn, gnn_test_loader,
            edge_index=edge_index,
            num_nodes=gnn_num_nodes,
            feature_dim=gnn_feature_dim,
            edge_weight=gnn_edge_weight,
        )

    # Day 22: GNN collapse warnings
    report.gnn_warnings = _check_collapse_warnings(report.gnn_metrics)

    # --- 7. Per-group metrics (Day 20.5) ---
    try:
        mlp_true, mlp_pred = _predict_all(mlp, test_loader)
        report.mlp_group_metrics = compute_group_metrics(mlp_true, mlp_pred, test_meta)
    except Exception:
        pass  # graceful fallback

    try:
        gnn_true, gnn_pred = _predict_all(
            gnn, gnn_test_loader,
            edge_index=edge_index,
            num_nodes=gnn_num_nodes,
            feature_dim=gnn_feature_dim,
            edge_weight=gnn_edge_weight,
        )
        report.gnn_group_metrics = compute_group_metrics(gnn_true, gnn_pred, test_meta)
    except Exception:
        pass  # graceful fallback

    # --- 8. Check for NaNs ---
    if (not np.isfinite(report.mlp_metrics.get("eval_loss", float("nan")))
        or not np.isfinite(report.gnn_metrics.get("eval_loss", float("nan")))):
        report.reason = "NaN/Inf in eval loss"
        return report

    report.status = "pass"
    return report


# ---------------------------------------------------------------------------
# Suite: Run All Experiments
# ---------------------------------------------------------------------------

def default_experiments() -> List[ExperimentConfig]:
    """Default set of generalization experiments."""
    return [
        ExperimentConfig(
            exp_id="EXP-19-A",
            name="Cross-model generalization (physics_hash split)",
            split_policy="cross_model",
            epochs=3,
        ),
        ExperimentConfig(
            exp_id="EXP-19-B",
            name="Within-model baseline (random block split)",
            split_policy="within_model",
            epochs=3,
        ),
        ExperimentConfig(
            exp_id="EXP-19-C",
            name="OOD p-range (train low-p, test high-p)",
            split_policy="ood_p_range",
            epochs=3,
            ood_test_p_lo=0.3,
            ood_test_p_hi=1.0,
        ),
    ]


def run_suite(
    experiments: List[ExperimentConfig],
    dataset: ShardDataset,
    out_dir: str | Path,
) -> Dict[str, Any]:
    """
    Run a suite of generalization experiments.

    Returns summary dict with all experiment results.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for exp_cfg in experiments:
        exp_dir = out_dir / exp_cfg.exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        report = run_experiment(exp_cfg, dataset, exp_dir)

        # Save individual report
        report_path = exp_dir / "report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        results.append(report.to_dict())

    # Save suite summary
    summary = {
        "suite_name": "generalization_day19",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_experiments": len(experiments),
        "experiments": results,
        "pass_count": sum(1 for r in results if r["status"] == "pass"),
        "skip_count": sum(1 for r in results if r["status"] == "skip"),
        "fail_count": sum(1 for r in results if r["status"] == "fail"),
    }

    summary_path = out_dir / "suite_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    return summary
