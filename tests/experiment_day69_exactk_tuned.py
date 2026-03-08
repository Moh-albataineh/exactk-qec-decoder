"""
Day 69 — ExactK Tuning: margin↓ + late-epoch λ decay + rZK_Y1 do-no-harm gate

Fix Seed 49200's late-epoch G1 explosion to push from PARTIAL (+26.6%)
to PASS (>=30% median G1 reduction across 10 seeds).

3 arms:
  A) Control:               No auxiliary iso-K loss
  B) ExactK_Tuned:          margin=0.30, λ decay after epoch 8
  C) ExactK_Tuned_Gated:    same + rZK_Y1 hysteresis gate (primary safety bet)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch

# ── Config ──────────────────────────────────────────────────────────────

ARMS = {
    "Control":             {"lambda_base": 0.0,  "delta_k": 0, "margin": 0.30, "use_safe_std": False, "gated": False},
    "ExactK_Tuned":        {"lambda_base": 0.10, "delta_k": 0, "margin": 0.30, "use_safe_std": False, "gated": False},
    "ExactK_Tuned_Gated":  {"lambda_base": 0.10, "delta_k": 0, "margin": 0.30, "use_safe_std": False, "gated": True},
}
SEEDS = [47000, 49000, 49200, 50000, 51000, 52000, 53000, 54000, 55000, 56000]
EPOCHS = 12
WARMUP_EPOCHS = 5
N_TRAIN = 2048
N_PROBE = 4096
DISTANCE = 5
P = 0.04
BASIS = "X"
CORR_STRENGTH = 0.5
BATCH_SIZE = 128
MAX_PAIRS = 256
GRAD_SAMPLE_EVERY = 20
TOPO_COLLAPSE_DROP_DELTA = 0.10
TOPO_COLLAPSE_FPASS_DELTA = 0.30

# Day 69 tuning knobs
LAMBDA_BASE = 0.10
MARGIN = 0.30
LAMBDA_DECAY_RATE = 0.85
LAMBDA_DECAY_START_EPOCH = 8
GATE_ON_THRESHOLD = 0.20
GATE_OFF_THRESHOLD = 0.12
GATE_MULT = 0.25
GATE_MIN_Y1_COUNT = 64

ARTIFACT_DIR = Path("ml_artifacts/day69_exactk_tuned_margin_decay_gate")


# ── Lambda schedule ──────────────────────────────────────────────────

def compute_lambda_epoch(lambda_base, epoch):
    """Compute λ for a given epoch with late-epoch decay."""
    if epoch > LAMBDA_DECAY_START_EPOCH:
        return lambda_base * (LAMBDA_DECAY_RATE ** (epoch - LAMBDA_DECAY_START_EPOCH))
    return lambda_base


# ── rZK_Y1 gate ──────────────────────────────────────────────────────

def compute_rzk_y1_probe(model, probe_set, min_count=GATE_MIN_Y1_COUNT):
    """Compute corr(Z_g1, K | Y=1) on ProbeSet in eval mode."""
    from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        split = model.forward_split(
            probe_set['det_feats'].to(device), probe_set['err_feats'].to(device),
            probe_set['ei_d2e'].to(device), probe_set['ei_e2d'].to(device),
            error_weights=probe_set.get('err_w').to(device),
            observable_mask=probe_set.get('obs_mask').to(device))
        z_g1 = split['logit_residual_norm'].detach().cpu().numpy().ravel()
        K = split.get('K')
        if K is not None:
            K = K.detach().cpu().numpy().ravel()
        else:
            K = compute_syndrome_count(probe_set['X_raw'])

    Y = probe_set['Y_raw'].ravel().astype(bool)
    result = {'rZK_Y0': None, 'rZK_Y1': None, 'n_Y1': int(Y.sum()),
              'n_Y0': int((~Y).sum()), 'insufficient_count': False}

    # Y=1 correlation
    mask_1 = Y
    if mask_1.sum() >= min_count:
        z1, k1 = z_g1[mask_1], K[mask_1].astype(float)
        if z1.std() > 1e-10 and k1.std() > 1e-10:
            result['rZK_Y1'] = float(np.corrcoef(z1, k1)[0, 1])
        else:
            result['rZK_Y1'] = 0.0
    else:
        result['insufficient_count'] = True

    # Y=0 correlation
    mask_0 = ~Y
    if mask_0.sum() >= min_count:
        z0, k0 = z_g1[mask_0], K[mask_0].astype(float)
        if z0.std() > 1e-10 and k0.std() > 1e-10:
            result['rZK_Y0'] = float(np.corrcoef(z0, k0)[0, 1])
        else:
            result['rZK_Y0'] = 0.0

    model.train()
    return result


def apply_hysteresis_gate(rzk_y1_abs, prev_gate_on):
    """Hysteresis gate: ON if |rZK_Y1| >= 0.20, OFF if <= 0.12."""
    if rzk_y1_abs >= GATE_ON_THRESHOLD:
        return True
    elif rzk_y1_abs <= GATE_OFF_THRESHOLD:
        return False
    else:
        return prev_gate_on  # stay in current state


# ── Data builder (same as Day 68) ─────────────────────────────────────

def build_data(distance, p, basis, n_samples, corr_strength, seed):
    from qec_noise_factory.ml.bench.regime_lock import RegimeLock, generate_locked_data
    from qec_noise_factory.ml.graph.dem_bipartite import build_bipartite_graph, bipartite_graph_to_tensors
    lock = RegimeLock(distance=distance, target_p=p, basis=basis,
                      require_generated=True, n_samples=n_samples,
                      corr_strength=corr_strength, seed=seed)
    X, Y = generate_locked_data(lock)
    n_det = X.shape[1]
    bg = build_bipartite_graph(distance=distance, rounds=distance, p=p,
                               basis=basis, noise_model="correlated_crosstalk_like")
    ei_d2e, ei_e2d, err_w, obs_mask = bipartite_graph_to_tensors(bg)
    N_det_graph = bg.num_detectors

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
    return {
        "det_train": build_det_feats(train_X), "det_test": build_det_feats(test_X),
        "err_feats": torch.from_numpy(bg.error_weights.reshape(-1, 1)).float(),
        "ei_d2e": ei_d2e, "ei_e2d": ei_e2d, "err_w": err_w, "obs_mask": obs_mask,
        "train_Y": train_Y, "test_Y": test_Y,
        "train_X": train_X, "test_X": test_X,
        "n_det": n_det, "N_det_graph": N_det_graph,
    }


# ── Model builder (same as Day 68) ────────────────────────────────────

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


# ── Training loop ─────────────────────────────────────────────────────

def train_one_epoch(model, data, K_train, train_Y_2d, optimizer, focal,
                    iso_loss_fn, iso_active, seed, epoch,
                    batch_size=BATCH_SIZE):
    from qec_noise_factory.ml.bench.density_scrambler import scramble_detector_syndromes
    device = next(model.parameters()).device
    train_Y_t = torch.from_numpy(train_Y_2d).to(device)
    model.train()
    model._current_epoch = epoch
    
    err_feats_d = data["err_feats"].to(device)
    ei_d2e_d = data["ei_d2e"].to(device)
    ei_e2d_d = data["ei_e2d"].to(device)
    err_w_d = data["err_w"].to(device)
    obs_mask_d = data["obs_mask"].to(device)
    
    ep_perm = torch.randperm(data["det_train"].shape[0])

    ep_loss = 0.0
    n_b = 0
    nan_detected = False

    pair_total_acc, pair_used_acc = [], []
    no_pair_count = 0
    violation_acc, zgap_acc = [], []
    iso_loss_acc, bce_loss_acc = [], []
    grad_ratio_acc = []
    z_var_global_acc, z_var_intra_k_acc = [], []
    unique_bins_acc = []
    alpha_z_scale_acc = []
    batch_count = 0

    for start in range(0, data["det_train"].shape[0], batch_size):
        end = min(start + batch_size, data["det_train"].shape[0])
        bi = ep_perm[start:end]
        det_batch = data["det_train"][bi].to(device)

        split = model.forward_split(det_batch, err_feats_d,
                                    ei_d2e_d, ei_e2d_d,
                                    error_weights=err_w_d,
                                    observable_mask=obs_mask_d)

        bce_loss = focal(split['logit_final'], train_Y_t[bi])
        if torch.isnan(bce_loss) or torch.isinf(bce_loss):
            nan_detected = True
        bce_loss_acc.append(float(bce_loss.detach()))
        loss = bce_loss

        z_g1 = split['logit_residual_norm']
        K_batch = split.get('K')
        Y_batch = train_Y_t[bi].squeeze(-1)

        if iso_loss_fn is not None and iso_active and K_batch is not None:
            iso_info = iso_loss_fn(z_g1, K_batch, Y_batch)
            loss = loss + iso_info['loss']
            iso_loss_acc.append(float(iso_info['loss'].detach()) if hasattr(iso_info['loss'], 'detach') else float(iso_info['loss']))
            pair_total_acc.append(iso_info['pair_count_total'])
            pair_used_acc.append(iso_info['pair_count_used'])
            z_var_global_acc.append(iso_info.get('z_var_global', 0.0))
            z_var_intra_k_acc.append(iso_info.get('z_var_intra_k', 0.0))
            unique_bins_acc.append(iso_info.get('unique_bins_with_pairs', 0))
            if iso_info['no_pair']:
                no_pair_count += 1
            else:
                violation_acc.append(iso_info['violation_rate'])
                zgap_acc.append(iso_info['zgap_mean'])

            with torch.no_grad():
                alpha_val = split.get('alpha', 1.0)
                if isinstance(alpha_val, torch.Tensor):
                    alpha_v = alpha_val.detach()
                else:
                    alpha_v = alpha_val
                az = (alpha_v * z_g1.detach()).abs().mean()
                lp = split['logit_prior'].detach().abs().mean() + 1e-12
                alpha_z_scale_acc.append(float(az / lp))

            batch_count += 1
            if not iso_info['no_pair'] and batch_count % GRAD_SAMPLE_EVERY == 0:
                z_for_grad = z_g1.detach().requires_grad_(True)
                iso_sample = iso_loss_fn(z_for_grad, K_batch, Y_batch)
                if iso_sample['loss'].requires_grad:
                    g_iso = torch.autograd.grad(iso_sample['loss'], z_for_grad,
                                                retain_graph=False, create_graph=False)[0]
                    iso_grad_norm = float(g_iso.norm().detach())
                else:
                    iso_grad_norm = 0.0
                z_for_bce = z_g1.detach().requires_grad_(True)
                alpha_val = split.get('alpha', 1.0)
                if isinstance(alpha_val, torch.Tensor):
                    logit_approx = split['logit_prior'].detach() + alpha_val.detach() * z_for_bce
                else:
                    logit_approx = split['logit_prior'].detach() + alpha_val * z_for_bce
                bce_approx = focal(logit_approx, train_Y_t[bi])
                g_bce = torch.autograd.grad(bce_approx, z_for_bce,
                                            retain_graph=False, create_graph=False)[0]
                bce_grad_norm = float(g_bce.norm().detach())
                if bce_grad_norm > 1e-12:
                    grad_ratio_acc.append(iso_grad_norm / bce_grad_norm)
        else:
            batch_count += 1

        det_scr = scramble_detector_syndromes(det_batch.cpu(), seed=seed + (epoch - 1) * 1000 + start).to(device)
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

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_loss += loss.item()
        n_b += 1

    active_batches = max(batch_count, 1)
    return {
        'loss': ep_loss / max(n_b, 1),
        'nan_detected': nan_detected,
        'pair_coverage_rate': 1.0 - (no_pair_count / active_batches) if iso_loss_fn and iso_active else 0.0,
        'no_pair_rate': no_pair_count / active_batches if iso_loss_fn and iso_active else 0.0,
        'mean_pairs_total': float(np.mean(pair_total_acc)) if pair_total_acc else 0.0,
        'mean_pairs_used': float(np.mean(pair_used_acc)) if pair_used_acc else 0.0,
        'mean_violation_rate': float(np.mean(violation_acc)) if violation_acc else 0.0,
        'mean_zgap': float(np.mean(zgap_acc)) if zgap_acc else 0.0,
        'iso_bce_ratio': float(np.mean(iso_loss_acc)) / (float(np.mean(bce_loss_acc)) + 1e-12) if iso_loss_acc else 0.0,
        'grad_ratio_median': float(np.median(grad_ratio_acc)) if grad_ratio_acc else 0.0,
        'mean_z_var_global': float(np.mean(z_var_global_acc)) if z_var_global_acc else 0.0,
        'mean_z_var_intra_k': float(np.mean(z_var_intra_k_acc)) if z_var_intra_k_acc else 0.0,
        'mean_unique_bins': float(np.mean(unique_bins_acc)) if unique_bins_acc else 0.0,
        'mean_alpha_z_scale_ratio': float(np.mean(alpha_z_scale_acc)) if alpha_z_scale_acc else 0.0,
    }


# ── Topology evaluation (same as Day 68) ─────────────────────────────

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
        det_scr = scramble_detector_syndromes(data["det_test"], seed=99999).to(device)
        split_scr = model.forward_split(
            det_scr, err_feats_d, ei_d2e_d, ei_e2d_d,
            error_weights=err_w_d, observable_mask=obs_mask_d)
        z_scr = split_scr['logit_residual_norm'].detach().cpu().numpy().ravel()

    y = data["test_Y"].ravel()
    K = compute_syndrome_count(data["test_X"])
    topo = compute_exact_k_scrambler_metric(y, z_clean, z_scr, K)
    probs_clean = torch.sigmoid(split_clean['logit_final']).detach().cpu().numpy().ravel()
    slice_result = compute_exact_k_slice_auroc(y, probs_clean, K)
    auroc_clean = slice_result.get('mean_slice_auroc')
    K_norm = (K - K.mean()) / (K.std() + 1e-8)
    auroc_density = compute_auroc(y.astype(bool), K_norm)
    tg = compute_topology_gain(auroc_clean, auroc_density)

    # Intra-class correlations
    intra = {}
    for cls_mask, label in [(y.astype(bool), 'Y1'), (~y.astype(bool), 'Y0')]:
        if cls_mask.sum() >= 8:
            zc, kc = z_clean[cls_mask], K[cls_mask].astype(float)
            if zc.std() > 1e-10 and kc.std() > 1e-10:
                intra[f'corr_Z_K_{label}'] = float(np.corrcoef(zc, kc)[0, 1])
            else:
                intra[f'corr_Z_K_{label}'] = 0.0
        else:
            intra[f'corr_Z_K_{label}'] = None

    # Iso-K pair accuracy
    iso_acc = _iso_pair_acc(z_clean, K, y, delta_k=0)

    return {
        "slice_clean": topo.get("mean_slice_clean_exactK"),
        "mean_drop": topo.get("mean_drop_exactK"),
        "fraction_pass": topo.get("fraction_pass"),
        "n_slices": topo.get("n_slices", 0),
        "TG": tg,
        "iso_pair_acc_exact": iso_acc,
        **intra,
    }


def _iso_pair_acc(z_np, K_np, Y_np, delta_k=0):
    z = np.asarray(z_np).ravel()
    K = np.asarray(K_np).ravel()
    Y = np.asarray(Y_np).ravel().astype(bool)
    pos_idx, neg_idx = np.where(Y)[0], np.where(~Y)[0]
    if len(pos_idx) < 1 or len(neg_idx) < 1:
        return None
    rng = np.random.RandomState(0)
    correct = total = 0
    for _ in range(5000):
        pi, ni = rng.choice(pos_idx), rng.choice(neg_idx)
        if abs(K[pi] - K[ni]) <= delta_k:
            total += 1
            if z[pi] > z[ni]:
                correct += 1
    return correct / total if total > 0 else None


def check_topology_collapse(ctrl_topos, arm_topos):
    ctrl_drops = [t["mean_drop"] for t in ctrl_topos if t["mean_drop"] is not None]
    arm_drops = [t["mean_drop"] for t in arm_topos if t["mean_drop"] is not None]
    ctrl_fpass = [t["fraction_pass"] for t in ctrl_topos if t["fraction_pass"] is not None]
    arm_fpass = [t["fraction_pass"] for t in arm_topos if t["fraction_pass"] is not None]
    collapsed, reasons = False, []
    if ctrl_drops and arm_drops:
        c_med, a_med = np.median(ctrl_drops), np.median(arm_drops)
        if c_med - a_med > TOPO_COLLAPSE_DROP_DELTA:
            collapsed = True
            reasons.append(f"drop collapse: ctrl={c_med:.3f} -> arm={a_med:.3f}")
    if ctrl_fpass and arm_fpass:
        c_med, a_med = np.median(ctrl_fpass), np.median(arm_fpass)
        if c_med - a_med > TOPO_COLLAPSE_FPASS_DELTA:
            collapsed = True
            reasons.append(f"frac_pass collapse: ctrl={c_med:.3f} -> arm={a_med:.3f}")
    return collapsed, reasons


# ── Run experiment ────────────────────────────────────────────────────

def run_experiment(seeds):
    from qec_noise_factory.ml.diagnostics.g1_probe import (
        generate_probe_set, evaluate_g1_aligned)
    from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count
    from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss

    all_results = {}
    alignment_reports = []
    per_seed_summaries = []
    gate_report = []

    any_nan = False
    any_alignment_fail = False
    total_alignment_checks = 0
    alignment_pass_count = 0

    for si, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  Seed {seed}  ({si+1}/{len(seeds)})")
        print(f"{'='*60}")

        data = build_data(DISTANCE, P, BASIS, N_TRAIN, CORR_STRENGTH, seed)
        probe_set = generate_probe_set(
            distance=DISTANCE, p=P, basis=BASIS,
            n_probe=N_PROBE, seed=seed, corr_strength=CORR_STRENGTH)

        seed_g1 = {}

        for arm_name, arm_cfg in ARMS.items():
            print(f"\n  --- {arm_name} (seed={seed}) ---")
            model, K_train, train_Y_2d = build_model(data, seed)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
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
                    use_safe_std=arm_cfg['use_safe_std'])

            epoch_records = []
            gate_on = False  # hysteresis state

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

                alignment_reports.append({
                    'seed': seed, 'arm': arm_name, 'epoch': epoch,
                    'G1_raw_probe': g1['G1_raw_probe'],
                    'alignment_invariant_pass': g1['alignment_invariant_pass'],
                })

                iso_active = iso_loss_fn is not None and epoch > WARMUP_EPOCHS

                # --- Day 69: Compute lambda_epoch with decay ---
                lambda_epoch = compute_lambda_epoch(arm_cfg['lambda_base'], epoch)

                # --- Day 69: rZK_Y1 gate (Arm C only) ---
                rzk_probe = {'rZK_Y0': None, 'rZK_Y1': None, 'insufficient_count': False}
                gate_mult = 1.0
                gate_state_str = "OFF"
                lambda_epoch_final = lambda_epoch

                if iso_active and arm_cfg['gated']:
                    rzk_probe = compute_rzk_y1_probe(model, probe_set)
                    if rzk_probe['insufficient_count']:
                        gate_on = False
                        gate_mult = 1.0
                        gate_state_str = "OFF_insufficient"
                    else:
                        rzk_y1_abs = abs(rzk_probe['rZK_Y1']) if rzk_probe['rZK_Y1'] is not None else 0.0
                        gate_on = apply_hysteresis_gate(rzk_y1_abs, gate_on)
                        gate_mult = GATE_MULT if gate_on else 1.0
                        gate_state_str = "ON" if gate_on else "OFF"
                    lambda_epoch_final = lambda_epoch * gate_mult
                elif iso_active:
                    # Arm B: compute rZK for telemetry but no gating
                    rzk_probe = compute_rzk_y1_probe(model, probe_set)
                    lambda_epoch_final = lambda_epoch

                # Dynamically set lambda on loss function
                if iso_loss_fn is not None and iso_active:
                    iso_loss_fn.lambda_iso = lambda_epoch_final

                train_info = train_one_epoch(
                    model, data, K_train, train_Y_2d, optimizer, focal,
                    iso_loss_fn, iso_active, seed, epoch)
                if train_info['nan_detected']:
                    any_nan = True

                topo = {}
                if epoch >= 6:
                    topo = evaluate_topology(model, data)

                record = {
                    'epoch': epoch, 'loss': train_info['loss'],
                    'G1_raw_probe': g1['G1_raw_probe'],
                    'alignment_ok': g1['alignment_invariant_pass'],
                    'nan_detected': train_info['nan_detected'],
                    'iso_active': iso_active,
                    'lambda_epoch': lambda_epoch,
                    'lambda_epoch_final': lambda_epoch_final,
                    'gate_state': gate_state_str,
                    'gate_mult': gate_mult,
                    'rZK_Y1_probe': rzk_probe.get('rZK_Y1'),
                    'rZK_Y0_probe': rzk_probe.get('rZK_Y0'),
                    'pair_coverage_rate': train_info['pair_coverage_rate'],
                    'no_pair_rate': train_info['no_pair_rate'],
                    'mean_pairs_total': train_info['mean_pairs_total'],
                    'mean_pairs_used': train_info['mean_pairs_used'],
                    'mean_violation_rate': train_info['mean_violation_rate'],
                    'mean_zgap': train_info['mean_zgap'],
                    'iso_bce_ratio': train_info['iso_bce_ratio'],
                    'grad_ratio_median': train_info['grad_ratio_median'],
                    'mean_z_var_global': train_info['mean_z_var_global'],
                    'mean_z_var_intra_k': train_info['mean_z_var_intra_k'],
                    'mean_unique_bins': train_info['mean_unique_bins'],
                    'mean_alpha_z_scale_ratio': train_info['mean_alpha_z_scale_ratio'],
                    **{f'topo_{k}': v for k, v in topo.items()},
                }
                epoch_records.append(record)

                gate_report.append({
                    'seed': seed, 'arm': arm_name, 'epoch': epoch,
                    'G1': g1['G1_raw_probe'],
                    'alignment_ok': g1['alignment_invariant_pass'],
                    'iso_active': iso_active,
                    'lambda_epoch': lambda_epoch,
                    'lambda_epoch_final': lambda_epoch_final,
                    'gate_state': gate_state_str,
                    'gate_mult': gate_mult,
                    'rZK_Y1_probe': rzk_probe.get('rZK_Y1'),
                    'rZK_Y0_probe': rzk_probe.get('rZK_Y0'),
                    'nan': train_info['nan_detected'],
                    'loss': train_info['loss'],
                    'pair_coverage': train_info['pair_coverage_rate'],
                    'topo_TG': topo.get('TG'),
                    'topo_mean_drop': topo.get('mean_drop'),
                })

                if epoch >= 6:
                    topo_str = ""
                    if topo:
                        tg_s = f"{topo['TG']:.4f}" if topo.get('TG') is not None else "N/A"
                        md_s = f"{topo['mean_drop']:.3f}" if topo.get('mean_drop') is not None else "N/A"
                        topo_str = f" MD={md_s} TG={tg_s}"

                    pair_str = ""
                    if iso_active:
                        gate_s = f" gate={gate_state_str}" if arm_cfg['gated'] else ""
                        rzk_s = f" rY1={rzk_probe['rZK_Y1']:.3f}" if rzk_probe.get('rZK_Y1') is not None else ""
                        pair_str = (f" λ={lambda_epoch_final:.4f}{gate_s}{rzk_s}"
                                    f" pairs={train_info['mean_pairs_used']:.0f}"
                                    f" viol={train_info['mean_violation_rate']:.2f}"
                                    f" gap={train_info['mean_zgap']:.3f}")

                    print(f"    Ep{epoch}: G1={g1['G1_raw_probe']:.4f} "
                          f"loss={train_info['loss']:.4f}{pair_str}{topo_str}")

            key = f"{arm_name}_{seed}"
            all_results[key] = epoch_records

            late = [r for r in epoch_records if r['epoch'] >= 6]
            seed_g1[arm_name] = {
                'med_G1_raw': float(np.median([r['G1_raw_probe'] for r in late])),
            }

        for arm_name in ["ExactK_Tuned", "ExactK_Tuned_Gated"]:
            ctrl_recs = all_results[f"Control_{seed}"]
            arm_recs = all_results[f"{arm_name}_{seed}"]
            ctrl_t = [{'mean_drop': r.get('topo_mean_drop'),
                       'fraction_pass': r.get('topo_fraction_pass')}
                      for r in ctrl_recs if r['epoch'] >= 6
                      and r.get('topo_mean_drop') is not None]
            arm_t = [{'mean_drop': r.get('topo_mean_drop'),
                      'fraction_pass': r.get('topo_fraction_pass')}
                     for r in arm_recs if r['epoch'] >= 6
                     and r.get('topo_mean_drop') is not None]
            collapsed, reasons = check_topology_collapse(ctrl_t, arm_t)
            ctrl_g1 = seed_g1["Control"]["med_G1_raw"]
            arm_g1 = seed_g1[arm_name]["med_G1_raw"]
            reduction = (ctrl_g1 - arm_g1) / max(ctrl_g1, 1e-6)

            per_seed_summaries.append({
                'seed': seed, 'arm': arm_name,
                'ctrl_G1': ctrl_g1, 'arm_G1': arm_g1,
                'reduction': reduction,
                'topo_collapsed': collapsed, 'topo_reasons': reasons,
            })

    return {
        'all_results': all_results,
        'alignment_reports': alignment_reports,
        'per_seed_summaries': per_seed_summaries,
        'gate_report': gate_report,
        'any_nan': any_nan,
        'any_alignment_fail': any_alignment_fail,
        'total_alignment_checks': total_alignment_checks,
        'alignment_pass_count': alignment_pass_count,
    }


# ── Verdict logic (same structure as Day 68) ──────────────────────────

def compute_verdict(exp_data, seeds):
    all_results = exp_data['all_results']
    per_seed_summaries = exp_data['per_seed_summaries']

    agg_report = {}
    any_topo_collapse = False
    collapse_details = []

    for arm_name in ARMS:
        arm_g1s = []
        for seed in seeds:
            recs = all_results[f"{arm_name}_{seed}"]
            late = [r for r in recs if r['epoch'] >= 6]
            arm_g1s.append(np.median([r['G1_raw_probe'] for r in late]))
        med_g1 = float(np.median(arm_g1s))
        agg_report[arm_name] = {
            'med_G1_raw': med_g1,
            'per_seed_G1': [float(v) for v in arm_g1s],
        }

    for ss in per_seed_summaries:
        if ss['topo_collapsed']:
            any_topo_collapse = True
            collapse_details.append(f"{ss['arm']}/seed={ss['seed']}: {ss['topo_reasons']}")

    ctrl_g1_agg = agg_report["Control"]["med_G1_raw"]

    best_arm = None
    best_reduction = -999.0
    for arm_name in ["ExactK_Tuned", "ExactK_Tuned_Gated"]:
        arm_med = agg_report[arm_name]["med_G1_raw"]
        red = (ctrl_g1_agg - arm_med) / max(ctrl_g1_agg, 1e-6)
        if red > best_reduction:
            best_reduction = red
            best_arm = arm_name

    crit_1 = exp_data['alignment_pass_count'] == exp_data['total_alignment_checks']
    crit_2_strong = best_reduction >= 0.30
    crit_3 = not any_topo_collapse
    crit_4 = not exp_data['any_nan']

    all_pass = crit_1 and crit_3 and crit_4
    if all_pass and crit_2_strong:
        verdict = "PASS"
    elif all_pass and best_reduction >= 0.10:
        verdict = "PARTIAL"
    elif not crit_3:
        verdict = "FAIL_TOPOLOGY"
    elif not crit_4:
        verdict = "FAIL_NAN"
    elif not crit_1:
        verdict = "INVALID_MEASUREMENT"
    else:
        verdict = "FAIL"

    return {
        'verdict': verdict,
        'best_arm': best_arm,
        'best_reduction': float(best_reduction),
        'criteria': {
            '1_alignment': {'pass': crit_1, 'count': f"{exp_data['alignment_pass_count']}/{exp_data['total_alignment_checks']}"},
            '2_g1_reduction': {'strong': crit_2_strong, 'reduction': float(best_reduction)},
            '3_topology': {'pass': crit_3, 'collapse_details': collapse_details},
            '4_stability': {'pass': crit_4},
        },
        'ctrl_G1': float(ctrl_g1_agg),
        'agg_report': agg_report,
        'collapse_details': collapse_details,
    }


def print_results(verdict_data, exp_data, seeds):
    agg = verdict_data['agg_report']
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS (epochs >= 6, 10 seeds)")
    print(f"{'='*70}")

    for arm_name in ARMS:
        med_g1 = agg[arm_name]['med_G1_raw']
        print(f"\n  {arm_name}: med G1 = {med_g1:.4f}")
        for i, seed in enumerate(seeds):
            print(f"    Seed {seed}: G1={agg[arm_name]['per_seed_G1'][i]:.4f}")

    print(f"\n  PER-SEED SUMMARY")
    print(f"  {'Seed':<8} {'Ctrl':<10} {'Tuned':<10} {'Δ_B%':<10} "
          f"{'Gated':<10} {'Δ_C%':<10} {'Topo'}")
    print(f"  {'─'*70}")
    for si, seed in enumerate(seeds):
        c = agg["Control"]["per_seed_G1"][si]
        sB = [s for s in exp_data['per_seed_summaries']
              if s['seed'] == seed and s['arm'] == 'ExactK_Tuned']
        sC = [s for s in exp_data['per_seed_summaries']
              if s['seed'] == seed and s['arm'] == 'ExactK_Tuned_Gated']
        sB = sB[0] if sB else None
        sC = sC[0] if sC else None
        gB = sB['arm_G1'] if sB else 0
        gC = sC['arm_G1'] if sC else 0
        dB = sB['reduction'] if sB else 0
        dC = sC['reduction'] if sC else 0
        topo_ok = "OK"
        if sB and sB['topo_collapsed']:
            topo_ok = "FAIL_B"
        if sC and sC['topo_collapsed']:
            topo_ok = "FAIL_C"
        print(f"  {seed:<8} {c:<10.4f} {gB:<10.4f} {dB:+<10.1%} "
              f"{gC:<10.4f} {dC:+<10.1%} {topo_ok}")

    v = verdict_data
    print(f"\n{'='*70}")
    print(f"  A) VERDICT: {v['verdict']}")
    print(f"     Best arm: {v['best_arm']} (Δ={v['best_reduction']:+.1%})")
    print(f"  D) ALIGNMENT: {'100% PASS' if v['criteria']['1_alignment']['pass'] else 'FAIL'} "
          f"({v['criteria']['1_alignment']['count']})")
    print(f"  E) TOPOLOGY:  {'NONE ✓' if v['criteria']['3_topology']['pass'] else 'COLLAPSE ✗'}")
    for d in v['collapse_details']:
        print(f"     {d}")
    print(f"  NaN/Inf: {'NONE ✓' if v['criteria']['4_stability']['pass'] else 'DETECTED ✗'}")
    print(f"{'='*70}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    params_used = {
        'day': 69, 'mechanism': 'exactk_tuned_margin_decay_gate',
        'arms': {k: v for k, v in ARMS.items()},
        'seeds': SEEDS,
        'epochs': EPOCHS, 'warmup_epochs': WARMUP_EPOCHS,
        'n_train': N_TRAIN, 'n_probe': N_PROBE,
        'distance': DISTANCE, 'p': P, 'basis': BASIS,
        'corr_strength': CORR_STRENGTH, 'batch_size': BATCH_SIZE,
        'max_pairs': MAX_PAIRS,
        'margin': MARGIN, 'lambda_base': LAMBDA_BASE,
        'lambda_decay_rate': LAMBDA_DECAY_RATE,
        'lambda_decay_start_epoch': LAMBDA_DECAY_START_EPOCH,
        'gate_on_threshold': GATE_ON_THRESHOLD,
        'gate_off_threshold': GATE_OFF_THRESHOLD,
        'gate_mult': GATE_MULT,
        'gate_min_y1_count': GATE_MIN_Y1_COUNT,
    }
    (ARTIFACT_DIR / "params_used.json").write_text(
        json.dumps(params_used, indent=2, default=_convert))

    print(f"[Day69] ExactK_Tuned — margin↓ + λ decay + rZK_Y1 gate")
    print(f"[Day69] Config: d={DISTANCE}, p={P}, {EPOCHS} epochs, {len(ARMS)} arms, B={BATCH_SIZE}")
    print(f"[Day69] margin={MARGIN}, λ_base={LAMBDA_BASE}, decay={LAMBDA_DECAY_RATE}^(ep-{LAMBDA_DECAY_START_EPOCH})")
    print(f"[Day69] Gate: ON≥{GATE_ON_THRESHOLD}, OFF≤{GATE_OFF_THRESHOLD}, mult={GATE_MULT}")
    print(f"[Day69] Seeds: {SEEDS}")

    exp_data = run_experiment(SEEDS)
    verdict_data = compute_verdict(exp_data, SEEDS)
    print_results(verdict_data, exp_data, SEEDS)

    elapsed = time.time() - start_time
    verdict_data['elapsed_s'] = elapsed

    (ARTIFACT_DIR / "decision_report.json").write_text(
        json.dumps(verdict_data, indent=2, default=_convert))
    (ARTIFACT_DIR / "all_results.json").write_text(
        json.dumps(exp_data['all_results'], indent=2, default=_convert))
    (ARTIFACT_DIR / "alignment_report.json").write_text(
        json.dumps(exp_data['alignment_reports'], indent=2, default=_convert))
    (ARTIFACT_DIR / "gate_report.json").write_text(
        json.dumps(exp_data['gate_report'], indent=2, default=_convert))

    # Telemetry summary
    telemetry = {}
    for arm_name in ["ExactK_Tuned", "ExactK_Tuned_Gated"]:
        arm_recs = []
        for seed in SEEDS:
            recs = exp_data['all_results'].get(f"{arm_name}_{seed}", [])
            late = [r for r in recs if r['epoch'] >= 6]
            arm_recs.extend(late)
        if arm_recs:
            telemetry[arm_name] = {
                'med_violation_rate': float(np.median([r['mean_violation_rate'] for r in arm_recs])),
                'med_zgap': float(np.median([r['mean_zgap'] for r in arm_recs])),
                'med_z_var_global': float(np.median([r['mean_z_var_global'] for r in arm_recs])),
                'med_z_var_intra_k': float(np.median([r['mean_z_var_intra_k'] for r in arm_recs])),
                'med_alpha_z_scale': float(np.median([r['mean_alpha_z_scale_ratio'] for r in arm_recs])),
                'med_lambda_final': float(np.median([r['lambda_epoch_final'] for r in arm_recs])),
                'gate_on_fraction': float(np.mean([1 if r['gate_state'] == 'ON' else 0 for r in arm_recs])),
            }
    (ARTIFACT_DIR / "telemetry_summary.json").write_text(
        json.dumps(telemetry, indent=2, default=_convert))

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
        print(f"    {f.name}")

    return 0 if verdict_data['verdict'] in ("PASS", "PARTIAL") else 1


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
