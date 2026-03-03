"""
Day 70 — ExactK d=7 OOD Generalization + EarlyCutoff

Validate ExactK_Tuned (Day 69 winner) on d=7 (harder lattice).
Add EarlyCutoff arm: same as Tuned but λ=0 after epoch 8 (targets 49200
late-epoch problem without gradient surgery).

3 arms × 10 seeds × 12 epochs.
d=7, p=0.04, B=256 (via gradient accumulation: micro=64, accum=4).
Model checkpoints saved for epochs 6-12.

HARDENED v4:
  - All metrics detached to Python primitives immediately
  - Incremental artifact saves after EACH seed (crash-safe)
  - Memory telemetry: CPU RSS + CUDA alloc per seed
  - Cleanup: del model/optimizer + gc + empty_cache per arm
  - Faulthandler: 600s watchdog (low-spam)
"""
from __future__ import annotations

import faulthandler
import gc
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from collections import defaultdict

faulthandler.enable()
# Low-frequency watchdog: every 10 minutes (not 2)
faulthandler.dump_traceback_later(600, repeat=True)
if hasattr(signal, 'SIGUSR1'):
    signal.signal(signal.SIGUSR1, lambda *_: faulthandler.dump_traceback())

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch

# ── Memory telemetry ─────────────────────────────────────────────────

def get_memory_stats():
    """Return CPU RSS (MB) and CUDA stats."""
    stats = {}
    try:
        import psutil
        proc = psutil.Process()
        stats['cpu_rss_mb'] = round(proc.memory_info().rss / (1024**2), 1)
    except ImportError:
        stats['cpu_rss_mb'] = -1

    if torch.cuda.is_available():
        stats['cuda_alloc_mb'] = round(torch.cuda.memory_allocated() / (1024**2), 1)
        stats['cuda_max_alloc_mb'] = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
        stats['cuda_reserved_mb'] = round(torch.cuda.memory_reserved() / (1024**2), 1)
    else:
        stats['cuda_alloc_mb'] = 0
        stats['cuda_max_alloc_mb'] = 0
        stats['cuda_reserved_mb'] = 0
    return stats


def log_memory(label):
    """Print memory telemetry with label."""
    s = get_memory_stats()
    print(f"  [MEM] {label}: CPU={s['cpu_rss_mb']}MB  "
          f"CUDA={s['cuda_alloc_mb']}MB (max={s['cuda_max_alloc_mb']}MB, "
          f"rsv={s['cuda_reserved_mb']}MB)", flush=True)
    return s


# ── Stage timer ──────────────────────────────────────────────────────

_stage_log = []

def stage(name):
    """Context manager that logs stage timing with flush."""
    class _Stage:
        def __enter__(self):
            self.t0 = time.perf_counter()
            print(f"  [STAGE] {name} ... ", end="", flush=True)
            return self
        def __exit__(self, *exc):
            elapsed = time.perf_counter() - self.t0
            print(f"{elapsed:.2f}s", flush=True)
            _stage_log.append({"stage": name, "elapsed_s": round(elapsed, 3)})
    return _Stage()


# ── Config ──────────────────────────────────────────────────────────────

ARMS = {
    "Control":             {"lambda_base": 0.0,  "delta_k": 0, "margin": 0.30, "early_cutoff": False},
    "ExactK_Tuned_Prod":   {"lambda_base": 0.10, "delta_k": 0, "margin": 0.30, "early_cutoff": False},
    "ExactK_EarlyCutoff":  {"lambda_base": 0.10, "delta_k": 0, "margin": 0.30, "early_cutoff": True},
}
SEEDS = [47000, 49000, 49200, 50000, 51000, 52000, 53000, 54000, 55000, 56000]
EPOCHS = 12
WARMUP_EPOCHS = 5
N_TRAIN = 4096
N_PROBE = 4096
DISTANCE = 7
P = 0.04
BASIS = "X"
CORR_STRENGTH = 0.5
GLOBAL_BATCH_SIZE = 256
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = GLOBAL_BATCH_SIZE // MICRO_BATCH_SIZE  # 4
MAX_PAIRS = 512
GRAD_SAMPLE_EVERY = 20
TOPO_COLLAPSE_DROP_DELTA = 0.10

LAMBDA_DECAY_RATE = 0.85
LAMBDA_DECAY_START_EPOCH = 8

ARTIFACT_DIR = Path("ml_artifacts/day70_exactk_d7_generalization")
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"


# ── Lambda schedules ──────────────────────────────────────────────────

def compute_lambda_tuned_prod(lambda_base, epoch):
    if epoch <= WARMUP_EPOCHS:
        return 0.0
    if epoch > LAMBDA_DECAY_START_EPOCH:
        return lambda_base * (LAMBDA_DECAY_RATE ** (epoch - LAMBDA_DECAY_START_EPOCH))
    return lambda_base


def compute_lambda_early_cutoff(lambda_base, epoch):
    if epoch <= WARMUP_EPOCHS:
        return 0.0
    if epoch > LAMBDA_DECAY_START_EPOCH:
        return 0.0
    return lambda_base


def get_lambda_for_arm(arm_cfg, epoch):
    if arm_cfg['lambda_base'] == 0.0:
        return 0.0
    if arm_cfg['early_cutoff']:
        return compute_lambda_early_cutoff(arm_cfg['lambda_base'], epoch)
    else:
        return compute_lambda_tuned_prod(arm_cfg['lambda_base'], epoch)


# ── Build bipartite graph ONCE ───────────────────────────────────────

_cached_graph = None

def get_bipartite_graph_cached():
    global _cached_graph
    if _cached_graph is not None:
        return _cached_graph

    from qec_noise_factory.ml.graph.dem_bipartite import build_bipartite_graph

    with stage("rebuild_stim_circuit (d=7)"):
        from qec_noise_factory.ml.stim.rebuild import rebuild_stim_circuit
        circuit = rebuild_stim_circuit(
            distance=DISTANCE, rounds=DISTANCE, p=P, basis=BASIS,
            noise_model="correlated_crosstalk_like")
        n_det = circuit.num_detectors
        n_qubits = circuit.num_qubits
        print(f"    circuit: {n_qubits} qubits, {n_det} detectors", flush=True)

    with stage("detector_error_model (decompose_errors=True)"):
        dem = circuit.detector_error_model(decompose_errors=True)
        n_dem_instructions = len(list(dem.flattened()))
        print(f"    DEM: {n_dem_instructions} instructions", flush=True)

    with stage("build_bipartite_graph (parse + merge)"):
        bg = build_bipartite_graph(
            distance=DISTANCE, rounds=DISTANCE, p=P,
            basis=BASIS, noise_model="correlated_crosstalk_like")
        print(f"    graph: {bg.num_detectors} det_nodes, {bg.num_errors} err_nodes, "
              f"{bg.edge_index_d2e.shape[1]} edges", flush=True)

    _cached_graph = bg
    return bg


# ── Data builder (uses cached graph) ─────────────────────────────────

def build_data(seed, cached_bg):
    from qec_noise_factory.ml.bench.regime_lock import RegimeLock, generate_locked_data
    from qec_noise_factory.ml.graph.dem_bipartite import bipartite_graph_to_tensors

    with stage(f"generate_locked_data (seed={seed}, n={N_TRAIN})"):
        lock = RegimeLock(distance=DISTANCE, target_p=P, basis=BASIS,
                          require_generated=True, n_samples=N_TRAIN,
                          corr_strength=CORR_STRENGTH, seed=seed)
        X, Y = generate_locked_data(lock)
        n_det = X.shape[1]
        print(f"    data: {X.shape[0]} samples, {n_det} detectors", flush=True)

    with stage(f"bipartite_to_tensors (seed={seed})"):
        ei_d2e, ei_e2d, err_w, obs_mask = bipartite_graph_to_tensors(cached_bg)
    N_det_graph = cached_bg.num_detectors

    def build_det_feats(X_np):
        B = X_np.shape[0]
        feats = np.zeros((B, N_det_graph, 2), dtype=np.float32)
        feats[:, :n_det, 0] = X_np.astype(np.float32)
        feats[:, -1, 1] = 1.0
        return torch.from_numpy(feats)

    rng = np.random.RandomState(seed)
    n_total = X.shape[0]
    n_test = max(1, n_total // 5)
    perm = rng.permutation(n_total)
    train_X, test_X = X[perm[n_test:]], X[perm[:n_test]]
    train_Y, test_Y = Y[perm[n_test:]], Y[perm[:n_test]]

    with stage(f"build_det_feats (seed={seed})"):
        det_train = build_det_feats(train_X)
        det_test = build_det_feats(test_X)

    return {
        "det_train": det_train, "det_test": det_test,
        "err_feats": torch.from_numpy(cached_bg.error_weights.reshape(-1, 1)).float(),
        "ei_d2e": ei_d2e, "ei_e2d": ei_e2d, "err_w": err_w, "obs_mask": obs_mask,
        "train_Y": train_Y, "test_Y": test_Y,
        "train_X": train_X, "test_X": test_X,
        "n_det": n_det, "N_det_graph": N_det_graph,
    }


# ── Model builder ────────────────────────────────────────────────────

def build_model(data, seed, num_k_bins=12):
    from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
    from qec_noise_factory.ml.bench.density_prior import build_k_prior_table
    from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count

    n_det = data["n_det"]
    K_train = compute_syndrome_count(data["train_X"])
    y_train_bool = data["train_Y"].astype(bool).ravel()
    train_Y_2d = data["train_Y"].astype(np.float32)
    if train_Y_2d.ndim == 1:
        train_Y_2d = train_Y_2d.reshape(-1, 1)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FactorGraphDecoderV1(
        det_input_dim=2, err_input_dim=1, output_dim=train_Y_2d.shape[1],
        hidden_dim=48, num_mp_layers=3, readout="mean_max", dropout=0.1,
        loss_fn="focal", focal_gamma=2.0, use_density_residualization=True,
    )
    model.use_density_prior_final = True
    model.fg_use_scrambler_nullspace_loss = True
    model.fg_use_alpha_kbin = True
    model.fg_use_kcs_standardization = True
    model.fg_use_leakage_penalty = True
    model.fg_lambda_leak = 0.3
    model.fg_use_tanh_clamp = True
    model.fg_clamp_c = 5.0
    model.fg_stopgrad_kcs = True
    model.fg_use_iso_k_loss = True
    model.fg_lambda_iso = 0.05
    model.fg_use_scr_baseline_centering = True
    model.fg_scr_baseline_mode = "bin"
    model.fg_sigma_floor = 0.1
    model.fg_use_k_ortho = False
    model._adaptive_soft_shield = None
    model._k_ortho_probeset_synced = None
    model._k_ortho_epoch_rolling = None
    model._k_ortho_frozen = None
    model._k_ortho_module = None
    model._k_ortho_window = None
    model._last_ortho_info = None
    model._current_epoch = 1
    model._frozen_beta = None
    model._ortho_active = False
    model._ortho_alignment_logged = False
    model._enable_grad_shield = False
    model._split_residual_head = None

    prior_table = build_k_prior_table(K_train, y_train_bool, alpha=1.0)
    model.set_density_prior(prior_table, n_det=n_det)
    model.setup_alpha_kbin(K_train, num_bins=num_k_bins, lambda_tv=0.05, alpha_init=0.0)
    model.setup_kcs(K_train, num_bins=num_k_bins)
    return model, K_train, train_Y_2d


# ── Training with gradient accumulation ──────────────────────────────

def train_one_epoch(model, data, K_train, train_Y_2d, optimizer, focal,
                    iso_loss_fn, iso_active, seed, epoch,
                    micro_bs=MICRO_BATCH_SIZE, accum_steps=GRAD_ACCUM_STEPS):
    from qec_noise_factory.ml.bench.density_scrambler import scramble_detector_syndromes
    device = next(model.parameters()).device
    train_Y_t = torch.from_numpy(train_Y_2d).to(device)
    model.train()
    model._current_epoch = epoch

    # Move shared graph tensors to device once
    err_feats_d = data["err_feats"].to(device)
    ei_d2e_d = data["ei_d2e"].to(device)
    ei_e2d_d = data["ei_e2d"].to(device)
    err_w_d = data["err_w"].to(device)
    obs_mask_d = data["obs_mask"].to(device)

    N = data["det_train"].shape[0]
    ep_perm = torch.randperm(N)

    ep_loss, n_global_steps = 0.0, 0
    nan_detected = False
    pair_total_acc, pair_used_acc = [], []
    no_pair_count, total_active_batches = 0, 0
    violation_acc, zgap_acc = [], []
    iso_loss_acc, bce_loss_acc = [], []
    unique_bins_acc = []

    for g_start in range(0, N, GLOBAL_BATCH_SIZE):
        g_end = min(g_start + GLOBAL_BATCH_SIZE, N)
        g_indices = ep_perm[g_start:g_end]

        optimizer.zero_grad()

        actual_micros = 0
        for m_start in range(0, len(g_indices), micro_bs):
            m_end = min(m_start + micro_bs, len(g_indices))
            bi = g_indices[m_start:m_end]
            det_batch = data["det_train"][bi].to(device)
            actual_micros += 1

            split = model.forward_split(det_batch, err_feats_d,
                                        ei_d2e_d, ei_e2d_d,
                                        error_weights=err_w_d,
                                        observable_mask=obs_mask_d)

            bce_loss = focal(split['logit_final'], train_Y_t[bi])
            if torch.isnan(bce_loss) or torch.isinf(bce_loss):
                nan_detected = True
            # DETACH immediately to float
            bce_loss_acc.append(float(bce_loss.item()))
            loss = bce_loss

            z_g1 = split['logit_residual_norm']
            K_batch = split.get('K')
            Y_batch = train_Y_t[bi].squeeze(-1)

            if iso_loss_fn is not None and iso_active and K_batch is not None:
                iso_info = iso_loss_fn(z_g1, K_batch, Y_batch)
                loss = loss + iso_info['loss']
                # DETACH all iso metrics immediately
                _iso_l = iso_info['loss']
                iso_loss_acc.append(float(_iso_l.item()) if hasattr(_iso_l, 'item') else float(_iso_l))
                pair_total_acc.append(int(iso_info['pair_count_total']))
                pair_used_acc.append(int(iso_info['pair_count_used']))
                unique_bins_acc.append(int(iso_info.get('unique_bins_with_pairs', 0)))
                if iso_info['no_pair']:
                    no_pair_count += 1
                else:
                    violation_acc.append(float(iso_info['violation_rate']))
                    zgap_acc.append(float(iso_info['zgap_mean']))
                total_active_batches += 1

            det_scr = scramble_detector_syndromes(det_batch.cpu(), seed=seed + (epoch - 1) * 1000 + g_start + m_start).to(device)
            null_result = model.compute_centered_nullspace_loss(
                det_scr, err_feats_d, ei_d2e_d, ei_e2d_d,
                error_weights=err_w_d, observable_mask=obs_mask_d)
            loss = loss + 2.0 * null_result['loss']

            if K_batch is not None:
                y_batch = train_Y_t[bi].squeeze(-1)
                existing_iso = model.compute_iso_k_loss(split['logit_residual_norm'], K_batch, y_batch)
                loss = loss + 0.05 * existing_iso
                leak_loss = model.compute_leakage_penalty(split['logit_residual_norm'], K_batch)
                loss = loss + 0.3 * leak_loss

            loss = loss + model.compute_corr_penalty() + model.compute_tv_penalty()

            scaled_loss = loss / actual_micros if actual_micros > 1 else loss
            scaled_loss.backward()
            ep_loss += float(loss.item())

            # Free intermediate tensors
            del split, det_batch, det_scr, null_result, loss, scaled_loss, bce_loss
            if K_batch is not None:
                del z_g1, K_batch, Y_batch

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n_global_steps += 1

    # Free device tensors
    del err_feats_d, ei_d2e_d, ei_e2d_d, err_w_d, obs_mask_d, train_Y_t

    active = max(total_active_batches, 1)
    return {
        'loss': ep_loss / max(n_global_steps * GRAD_ACCUM_STEPS, 1),
        'nan_detected': nan_detected,
        'pair_coverage_rate': 1.0 - (no_pair_count / active) if iso_loss_fn and iso_active else 0.0,
        'no_pair_rate': no_pair_count / active if iso_loss_fn and iso_active else 0.0,
        'mean_pairs_total': float(np.mean(pair_total_acc)) if pair_total_acc else 0.0,
        'mean_pairs_used': float(np.mean(pair_used_acc)) if pair_used_acc else 0.0,
        'mean_violation_rate': float(np.mean(violation_acc)) if violation_acc else 0.0,
        'mean_zgap': float(np.mean(zgap_acc)) if zgap_acc else 0.0,
        'iso_bce_ratio': float(np.mean(iso_loss_acc)) / (float(np.mean(bce_loss_acc)) + 1e-12) if iso_loss_acc else 0.0,
        'mean_unique_bins': float(np.mean(unique_bins_acc)) if unique_bins_acc else 0.0,
    }


# ── Topology evaluation ──────────────────────────────────────────────

def evaluate_topology(model, data):
    from qec_noise_factory.ml.diagnostics.exact_k_scrambler import compute_exact_k_scrambler_metric
    from qec_noise_factory.ml.bench.density_scrambler import scramble_detector_syndromes
    from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count, compute_topology_gain
    from qec_noise_factory.ml.diagnostics.exact_k_slice import compute_exact_k_slice_auroc
    from qec_noise_factory.ml.metrics.ranking import compute_auroc

    device = next(model.parameters()).device
    err_feats_d = data["err_feats"].to(device)
    ei_d2e_d = data["ei_d2e"].to(device)
    ei_e2d_d = data["ei_e2d"].to(device)
    err_w_d = data["err_w"].to(device)
    obs_mask_d = data["obs_mask"].to(device)

    model.eval()
    with torch.no_grad():
        split_clean = model.forward_split(
            data["det_test"].to(device), err_feats_d, ei_d2e_d, ei_e2d_d,
            error_weights=err_w_d, observable_mask=obs_mask_d)
        z_clean = split_clean['logit_residual_norm'].detach().cpu().numpy().ravel()
        logit_final_cpu = split_clean['logit_final'].detach().cpu()
        del split_clean

        det_scr = scramble_detector_syndromes(data["det_test"], seed=99999).to(device)
        split_scr = model.forward_split(
            det_scr, err_feats_d, ei_d2e_d, ei_e2d_d,
            error_weights=err_w_d, observable_mask=obs_mask_d)
        z_scr = split_scr['logit_residual_norm'].detach().cpu().numpy().ravel()
        del split_scr, det_scr

    del err_feats_d, ei_d2e_d, ei_e2d_d, err_w_d, obs_mask_d

    y = data["test_Y"].ravel()
    K = compute_syndrome_count(data["test_X"])
    topo = compute_exact_k_scrambler_metric(y, z_clean, z_scr, K)
    probs_clean = torch.sigmoid(logit_final_cpu).numpy().ravel()
    slice_result = compute_exact_k_slice_auroc(y, probs_clean, K)
    auroc_clean = slice_result.get('mean_slice_auroc')
    K_norm = (K - K.mean()) / (K.std() + 1e-8)
    auroc_density = compute_auroc(y.astype(bool), K_norm)
    tg = compute_topology_gain(auroc_clean, auroc_density)

    val_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logit_final_cpu,
        torch.from_numpy(data["test_Y"].astype(np.float32).reshape(-1, 1)),
        reduction='mean').item()
    del logit_final_cpu

    return {
        "slice_clean": float(topo.get("mean_slice_clean_exactK")) if topo.get("mean_slice_clean_exactK") is not None else None,
        "mean_drop": float(topo.get("mean_drop_exactK")) if topo.get("mean_drop_exactK") is not None else None,
        "fraction_pass": float(topo.get("fraction_pass")) if topo.get("fraction_pass") is not None else None,
        "TG": float(tg) if tg is not None else None,
        "val_bce": float(val_bce),
    }


def check_topology_collapse(ctrl_topos, arm_topos):
    ctrl_drops = [t["mean_drop"] for t in ctrl_topos if t.get("mean_drop") is not None]
    arm_drops = [t["mean_drop"] for t in arm_topos if t.get("mean_drop") is not None]
    collapsed, reasons = False, []
    if ctrl_drops and arm_drops:
        c_med, a_med = float(np.median(ctrl_drops)), float(np.median(arm_drops))
        if c_med - a_med > TOPO_COLLAPSE_DROP_DELTA:
            collapsed = True
            reasons.append(f"drop collapse: ctrl={c_med:.3f} -> arm={a_med:.3f}")
    return collapsed, reasons


# ── Save checkpoint ──────────────────────────────────────────────────

def save_checkpoint(model, arm_name, seed, epoch):
    ckpt_dir = CHECKPOINT_DIR / arm_name / str(seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"epoch{epoch:02d}.pt"
    torch.save(model.state_dict(), path)
    return str(path)


# ── Incremental artifact save ────────────────────────────────────────

def save_incremental_artifacts(all_results, alignment_reports, gate_report,
                                best_epoch_data, per_seed_summaries,
                                completed_seeds, memory_log):
    """Save current state of all artifacts (crash-safe)."""
    try:
        exp_data = {
            'all_results': all_results,
            'alignment_reports': alignment_reports,
            'per_seed_summaries': per_seed_summaries,
            'gate_report': gate_report,
            'best_epoch_data': best_epoch_data,
            'completed_seeds': completed_seeds,
            'memory_log': memory_log,
            'stage_log': _stage_log,
        }
        (ARTIFACT_DIR / "all_results.json").write_text(
            json.dumps(all_results, indent=2, default=_convert))
        (ARTIFACT_DIR / "alignment_report.json").write_text(
            json.dumps(alignment_reports, indent=2, default=_convert))
        (ARTIFACT_DIR / "gate_report.json").write_text(
            json.dumps(gate_report, indent=2, default=_convert))
        (ARTIFACT_DIR / "best_epoch_candidates.json").write_text(
            json.dumps(best_epoch_data, indent=2, default=_convert))
        (ARTIFACT_DIR / "per_seed_summaries.json").write_text(
            json.dumps(per_seed_summaries, indent=2, default=_convert))
        (ARTIFACT_DIR / "memory_log.json").write_text(
            json.dumps(memory_log, indent=2, default=_convert))
        (ARTIFACT_DIR / "stage_log.json").write_text(
            json.dumps(_stage_log, indent=2))
        print(f"  [SAVE] Incremental artifacts saved ({len(completed_seeds)} seeds)", flush=True)
    except Exception as e:
        print(f"  [SAVE] WARNING: incremental save failed: {e}", flush=True)


# ── Run experiment ────────────────────────────────────────────────────

def run_experiment(seeds):
    from qec_noise_factory.ml.diagnostics.g1_probe import (
        generate_probe_set, evaluate_g1_aligned)
    from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss

    print(f"\n[PREPROCESS] Building d={DISTANCE} bipartite graph (one-time)...", flush=True)
    cached_bg = get_bipartite_graph_cached()
    print(f"[PREPROCESS] Graph ready: {cached_bg.num_detectors} det, "
          f"{cached_bg.num_errors} err, {cached_bg.edge_index_d2e.shape[1]} edges\n", flush=True)

    all_results = {}
    alignment_reports = []
    per_seed_summaries = []
    gate_report = []
    best_epoch_data = {}
    any_nan = False
    any_alignment_fail = False
    total_alignment_checks = 0
    alignment_pass_count = 0
    completed_seeds = []
    memory_log = []

    log_memory("before_seeds")

    for si, seed in enumerate(seeds):
        seed_t0 = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"  Seed {seed}  ({si+1}/{len(seeds)})")
        print(f"{'='*60}", flush=True)

        mem_before = log_memory(f"seed_{seed}_start")

        data = build_data(seed, cached_bg)

        with stage(f"generate_probe_set (seed={seed})"):
            probe_set = generate_probe_set(
                distance=DISTANCE, p=P, basis=BASIS,
                n_probe=N_PROBE, seed=seed, corr_strength=CORR_STRENGTH,
                cached_bg=cached_bg)

        seed_g1 = {}

        for arm_name, arm_cfg in ARMS.items():
            print(f"\n  --- {arm_name} (seed={seed}) ---", flush=True)

            with stage(f"build_model ({arm_name}, seed={seed})"):
                model, K_train, train_Y_2d = build_model(data, seed)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            print(f"    [device] model on {device}", flush=True)

            from qec_noise_factory.ml.models.factor_graph import FocalLoss
            focal = FocalLoss(gamma=2.0, pos_weight=None)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            iso_loss_fn = None
            if arm_cfg['lambda_base'] > 0:
                iso_loss_fn = IsoKRankingLoss(
                    lambda_iso=arm_cfg['lambda_base'],
                    margin=arm_cfg['margin'],
                    delta_k=arm_cfg['delta_k'],
                    max_pairs=MAX_PAIRS,
                    use_safe_std=False)

            epoch_records = []
            epoch_candidates = []

            for epoch in range(1, EPOCHS + 1):
                ortho_state = {
                    'beta_E': 0.0, 'muK_probe': 0.0, 'w': 0.0,
                    'eff_corr_expected': False,
                }
                g1 = evaluate_g1_aligned(model, probe_set, ortho_state)

                total_alignment_checks += 1
                if g1['alignment_invariant_pass']:
                    alignment_pass_count += 1
                else:
                    any_alignment_fail = True

                # DETACH: store only primitives
                g1_raw = float(g1['G1_raw_probe'])
                g1_align = bool(g1['alignment_invariant_pass'])

                alignment_reports.append({
                    'seed': seed, 'arm': arm_name, 'epoch': epoch,
                    'G1_raw_probe': g1_raw,
                    'alignment_invariant_pass': g1_align,
                })

                lambda_epoch = get_lambda_for_arm(arm_cfg, epoch)
                iso_active = iso_loss_fn is not None and lambda_epoch > 0

                if iso_loss_fn is not None:
                    iso_loss_fn.lambda_iso = lambda_epoch

                train_info = train_one_epoch(
                    model, data, K_train, train_Y_2d, optimizer, focal,
                    iso_loss_fn, iso_active, seed, epoch)
                if train_info['nan_detected']:
                    any_nan = True

                topo = {}
                ckpt_path = None
                if epoch >= 6:
                    topo = evaluate_topology(model, data)
                    ckpt_path = save_checkpoint(model, arm_name, seed, epoch)

                    topo_safe = topo.get('mean_drop') is not None and topo.get('mean_drop', -1) > -0.05
                    epoch_candidates.append({
                        'epoch': epoch, 'val_bce': topo.get('val_bce'),
                        'G1_aligned': g1_raw,
                        'mean_drop': topo.get('mean_drop'),
                        'slice_clean': topo.get('slice_clean'),
                        'topo_safe': topo_safe,
                        'checkpoint': ckpt_path,
                    })

                record = {
                    'epoch': epoch, 'loss': float(train_info['loss']),
                    'G1_raw_probe': g1_raw,
                    'alignment_ok': g1_align,
                    'nan_detected': train_info['nan_detected'],
                    'iso_active': iso_active,
                    'lambda_epoch': float(lambda_epoch),
                    'pair_coverage_rate': float(train_info['pair_coverage_rate']),
                    'mean_pairs_used': float(train_info['mean_pairs_used']),
                    'mean_violation_rate': float(train_info['mean_violation_rate']),
                    'mean_zgap': float(train_info['mean_zgap']),
                    'iso_bce_ratio': float(train_info['iso_bce_ratio']),
                    'mean_unique_bins': float(train_info['mean_unique_bins']),
                    **{f'topo_{k}': v for k, v in topo.items()},
                }
                epoch_records.append(record)

                gate_report.append({
                    'seed': seed, 'arm': arm_name, 'epoch': epoch,
                    'G1': g1_raw, 'iso_active': iso_active,
                    'lambda': float(lambda_epoch),
                    'topo_TG': topo.get('TG'),
                    'topo_mean_drop': topo.get('mean_drop'),
                })

                if epoch >= 6:
                    md_s = f"{topo['mean_drop']:.3f}" if topo.get('mean_drop') is not None else "N/A"
                    tg_s = f"{topo['TG']:.4f}" if topo.get('TG') is not None else "N/A"
                    pair_str = ""
                    if iso_active:
                        pair_str = (f" λ={lambda_epoch:.4f}"
                                    f" pairs={train_info['mean_pairs_used']:.0f}"
                                    f" viol={train_info['mean_violation_rate']:.2f}"
                                    f" gap={train_info['mean_zgap']:.3f}")
                    print(f"    Ep{epoch}: G1={g1_raw:.4f} "
                          f"loss={train_info['loss']:.4f}{pair_str}"
                          f" MD={md_s} TG={tg_s}", flush=True)

            key = f"{arm_name}_{seed}"
            all_results[key] = epoch_records

            # Best epoch selection
            constrained = [c for c in epoch_candidates
                           if c['G1_aligned'] <= 0.015 and c['topo_safe']]
            if constrained:
                best = min(constrained, key=lambda c: c['val_bce'])
            else:
                best = min(epoch_candidates, key=lambda c: c['val_bce']) if epoch_candidates else None
            best_epoch_data[key] = {
                'candidates': epoch_candidates,
                'best_epoch': best,
                'constrained_available': len(constrained) > 0,
            }

            late = [r for r in epoch_records if r['epoch'] >= 6]
            seed_g1[arm_name] = float(np.median([r['G1_raw_probe'] for r in late]))

            # ── CLEANUP: free model/optimizer per arm ──
            del model, optimizer, focal, iso_loss_fn
            del K_train, train_Y_2d, epoch_records, epoch_candidates
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Per-seed summaries
        for arm_name in ["ExactK_Tuned_Prod", "ExactK_EarlyCutoff"]:
            ctrl_recs = all_results[f"Control_{seed}"]
            arm_recs = all_results[f"{arm_name}_{seed}"]
            ctrl_t = [{'mean_drop': r.get('topo_mean_drop')}
                      for r in ctrl_recs if r['epoch'] >= 6 and r.get('topo_mean_drop') is not None]
            arm_t = [{'mean_drop': r.get('topo_mean_drop')}
                     for r in arm_recs if r['epoch'] >= 6 and r.get('topo_mean_drop') is not None]
            collapsed, reasons = check_topology_collapse(ctrl_t, arm_t)
            ctrl_g1 = seed_g1["Control"]
            arm_g1 = seed_g1[arm_name]
            reduction = (ctrl_g1 - arm_g1) / max(ctrl_g1, 1e-6)
            per_seed_summaries.append({
                'seed': seed, 'arm': arm_name,
                'ctrl_G1': float(ctrl_g1), 'arm_G1': float(arm_g1),
                'reduction': float(reduction),
                'topo_collapsed': collapsed, 'topo_reasons': reasons,
            })

        completed_seeds.append(seed)
        seed_elapsed = time.perf_counter() - seed_t0
        mem_after = log_memory(f"seed_{seed}_end ({seed_elapsed:.0f}s)")
        memory_log.append({
            'seed': seed, 'elapsed_s': round(seed_elapsed, 1),
            'mem_before': mem_before, 'mem_after': mem_after,
        })

        # ── CLEANUP: free data/probe per seed ──
        del data, probe_set, seed_g1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── INCREMENTAL SAVE (crash-safe) ──
        save_incremental_artifacts(
            all_results, alignment_reports, gate_report,
            best_epoch_data, per_seed_summaries,
            completed_seeds, memory_log)

    return {
        'all_results': all_results, 'alignment_reports': alignment_reports,
        'per_seed_summaries': per_seed_summaries, 'gate_report': gate_report,
        'best_epoch_data': best_epoch_data,
        'any_nan': any_nan, 'any_alignment_fail': any_alignment_fail,
        'total_alignment_checks': total_alignment_checks,
        'alignment_pass_count': alignment_pass_count,
        'memory_log': memory_log,
        'stage_log': _stage_log,
    }


# ── Verdict ──────────────────────────────────────────────────────────

def compute_verdict(exp_data, seeds):
    all_results = exp_data['all_results']
    # Only compute over seeds that completed
    available_seeds = [s for s in seeds if f"Control_{s}" in all_results]
    if not available_seeds:
        return {'verdict': 'NO_DATA', 'gen_verdict': 'NO_DATA'}

    agg = {}
    any_topo = False
    collapse_details = []

    for arm in ARMS:
        g1s = []
        for seed in available_seeds:
            key = f"{arm}_{seed}"
            if key not in all_results:
                continue
            recs = all_results[key]
            late = [r for r in recs if r['epoch'] >= 6]
            if late:
                g1s.append(float(np.median([r['G1_raw_probe'] for r in late])))
        agg[arm] = {'med_G1': float(np.median(g1s)) if g1s else 999.0, 'per_seed_G1': g1s}

    for ss in exp_data.get('per_seed_summaries', []):
        if ss['topo_collapsed']:
            any_topo = True
            collapse_details.append(f"{ss['arm']}/seed={ss['seed']}: {ss['topo_reasons']}")

    ctrl_g1 = agg["Control"]["med_G1"]
    crit_1 = exp_data['alignment_pass_count'] == exp_data['total_alignment_checks']
    crit_3 = not any_topo
    crit_4 = not exp_data['any_nan']

    best_arm, best_red = None, -999.0
    for arm in ["ExactK_Tuned_Prod", "ExactK_EarlyCutoff"]:
        if arm in agg:
            r = (ctrl_g1 - agg[arm]["med_G1"]) / max(ctrl_g1, 1e-6)
            if r > best_red:
                best_red, best_arm = r, arm

    all_ok = crit_1 and crit_3 and crit_4
    if all_ok and best_red >= 0.30:
        verdict = "PASS"
    elif all_ok and best_red >= 0.10:
        verdict = "PARTIAL"
    elif not crit_3:
        verdict = "FAIL_TOPOLOGY"
    elif not crit_4:
        verdict = "FAIL_NAN"
    elif not crit_1:
        verdict = "INVALID_MEASUREMENT"
    else:
        verdict = "FAIL"

    if all_ok and best_red >= 0.20:
        gen_verdict = "GENERALIZATION_PASS"
    elif all_ok and best_red >= 0.10:
        gen_verdict = "GENERALIZATION_PARTIAL"
    elif not crit_3:
        gen_verdict = "FAIL_TOPOLOGY"
    else:
        gen_verdict = "FAIL"

    return {
        'verdict': verdict, 'gen_verdict': gen_verdict,
        'best_arm': best_arm, 'best_reduction': float(best_red),
        'seeds_completed': len(available_seeds),
        'seeds_total': len(seeds),
        'criteria': {
            'alignment': {'pass': crit_1, 'count': f"{exp_data['alignment_pass_count']}/{exp_data['total_alignment_checks']}"},
            'g1_reduction': {'reduction': float(best_red)},
            'topology': {'pass': crit_3, 'details': collapse_details},
            'stability': {'pass': crit_4},
        },
        'ctrl_G1': float(ctrl_g1), 'agg': agg,
    }


def print_results(v, exp_data, seeds):
    agg = v['agg']
    print(f"\n{'='*70}")
    print(f"  DAY 70 — d=7 GENERALIZATION RESULTS (epochs >= 6, {v.get('seeds_completed', '?')}/{len(seeds)} seeds)")
    print(f"{'='*70}")
    for arm in ARMS:
        g1s = agg.get(arm, {}).get('per_seed_G1', [])
        med = agg.get(arm, {}).get('med_G1', -1)
        print(f"\n  {arm}: med G1 = {med:.4f}")
        for i, g in enumerate(g1s):
            print(f"    G1={g:.4f}")

    print(f"\n  PER-SEED SUMMARY")
    print(f"  {'Seed':<8} {'Ctrl':<10} {'Prod':<10} {'Δ_P%':<10} "
          f"{'Cutoff':<10} {'Δ_C%':<10} {'Topo'}")
    print(f"  {'─'*70}")
    for ss_pair in zip(
        [s for s in exp_data.get('per_seed_summaries', []) if s['arm'] == 'ExactK_Tuned_Prod'],
        [s for s in exp_data.get('per_seed_summaries', []) if s['arm'] == 'ExactK_EarlyCutoff'],
    ):
        p, e = ss_pair
        topo = "OK"
        if p['topo_collapsed']: topo = "FAIL_P"
        if e['topo_collapsed']: topo = "FAIL_E"
        print(f"  {p['seed']:<8} {p['ctrl_G1']:<10.4f} {p['arm_G1']:<10.4f} {p['reduction']:+<10.1%} "
              f"{e['arm_G1']:<10.4f} {e['reduction']:+<10.1%} {topo}")

    print(f"\n{'='*70}")
    print(f"  VERDICT:     {v['verdict']}")
    print(f"  OOD VERDICT: {v['gen_verdict']}")
    print(f"  Best arm:    {v.get('best_arm')} (Δ={v.get('best_reduction', 0):+.1%})")
    print(f"  ALIGNMENT:   {'100% PASS' if v['criteria']['alignment']['pass'] else 'FAIL'} ({v['criteria']['alignment']['count']})")
    print(f"  TOPOLOGY:    {'NONE ✓' if v['criteria']['topology']['pass'] else 'COLLAPSE ✗'}")
    for d in v['criteria']['topology']['details']:
        print(f"     {d}")
    print(f"  NaN/Inf:     {'NONE ✓' if v['criteria']['stability']['pass'] else 'DETECTED ✗'}")
    print(f"{'='*70}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        'day': 70, 'mechanism': 'exactk_d7_generalization',
        'distance': DISTANCE, 'p': P, 'basis': BASIS,
        'arms': {k: v for k, v in ARMS.items()},
        'seeds': SEEDS, 'epochs': EPOCHS, 'warmup': WARMUP_EPOCHS,
        'global_batch_size': GLOBAL_BATCH_SIZE,
        'micro_batch_size': MICRO_BATCH_SIZE,
        'grad_accum_steps': GRAD_ACCUM_STEPS,
        'max_pairs': MAX_PAIRS,
        'lambda_decay_rate': LAMBDA_DECAY_RATE,
        'lambda_decay_start': LAMBDA_DECAY_START_EPOCH,
    }
    (ARTIFACT_DIR / "params_used.json").write_text(json.dumps(params, indent=2, default=_convert))

    print(f"[Day70] ExactK d={DISTANCE} Generalization + EarlyCutoff")
    print(f"[Day70] Config: d={DISTANCE}, p={P}, B={GLOBAL_BATCH_SIZE} (micro={MICRO_BATCH_SIZE}×{GRAD_ACCUM_STEPS})")
    print(f"[Day70] Arms: {list(ARMS.keys())}")
    print(f"[Day70] Seeds: {SEEDS}")
    print(f"[Day70] faulthandler watchdog every 600s", flush=True)

    exp_data = run_experiment(SEEDS)
    verdict = compute_verdict(exp_data, SEEDS)
    print_results(verdict, exp_data, SEEDS)

    elapsed = time.time() - start_time
    verdict['elapsed_s'] = elapsed

    (ARTIFACT_DIR / "decision_report.json").write_text(json.dumps(verdict, indent=2, default=_convert))
    (ARTIFACT_DIR / "all_results.json").write_text(json.dumps(exp_data['all_results'], indent=2, default=_convert))
    (ARTIFACT_DIR / "alignment_report.json").write_text(json.dumps(exp_data['alignment_reports'], indent=2, default=_convert))
    (ARTIFACT_DIR / "gate_report.json").write_text(json.dumps(exp_data['gate_report'], indent=2, default=_convert))
    (ARTIFACT_DIR / "best_epoch_candidates.json").write_text(json.dumps(exp_data['best_epoch_data'], indent=2, default=_convert))
    (ARTIFACT_DIR / "stage_log.json").write_text(json.dumps(_stage_log, indent=2))
    (ARTIFACT_DIR / "memory_log.json").write_text(json.dumps(exp_data.get('memory_log', []), indent=2, default=_convert))

    checksums = {}
    for f in ARTIFACT_DIR.iterdir():
        if f.is_file() and f.name != "checksums.sha256":
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            checksums[f.name] = h
    (ARTIFACT_DIR / "checksums.sha256").write_text(
        "\n".join(f"{v}  {k}" for k, v in sorted(checksums.items())) + "\n")

    print(f"\n  ARTIFACTS: {ARTIFACT_DIR}/")
    print(f"  Elapsed: {elapsed:.1f}s")
    for f in sorted(ARTIFACT_DIR.iterdir()):
        if f.is_file():
            print(f"    {f.name}")

    return 0


def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


if __name__ == "__main__":
    sys.exit(main())
