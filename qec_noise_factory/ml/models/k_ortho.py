"""
Day 57 — Vector K-Orthogonalization Module (v2)

Removes the linear component of K from Z_g1 using per-channel vector β
with magnitude cap, correlation-floor gating, and ramp schedule.

    z_ortho_i = z_i - w * beta_cap_i * k0      (for each channel i)

Day 56 failure modes addressed:
  - β explosion → magnitude cap + hard clamp [-2, 2]
  - Noise injection → corr-floor (|corr| < corr_min → β=0)
  - Cold-start shock → ramp schedule (w=0 epochs 1-2, ramp 3-6, w=1 epoch 7+)
  - Evasion windup → frozen-epoch β variant (stationary target)

Safety invariants:
  - Memory: all window entries are CPU Python floats / lists (no graph refs)
  - Train/eval: window updates only during training
  - Numerical: eps-safe denominators always
  - No in-place: z_ortho is always out-of-place
  - Device/dtype: beta moved to z.device before subtraction
  - Shape: k0 broadcast-safe with z
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import math

BETA_HARD_CLAMP = 2.0
ETA_DEFAULT = 0.15        # magnitude cap fraction
CORR_MIN_DEFAULT = 0.01   # correlation floor
N_MIN_DEFAULT = 512       # minimum samples before applying


def get_ramp_weight(epoch: int) -> float:
    """Deterministic ramp schedule.

    Epochs 1-2: w=0 (no correction)
    Epochs 3-6: linear ramp 0→1
    Epoch ≥ 7: w=1 (full correction)
    """
    if epoch <= 2:
        return 0.0
    elif epoch <= 6:
        return (epoch - 2) / 4.0  # 3→0.25, 4→0.50, 5→0.75, 6→1.0
    else:
        return 1.0


class VectorKBetaWindow:
    """Rolling window for per-channel vector β = Cov(z_i, k) / Var(k).

    Stores per-batch sufficient statistics as CPU Python floats/lists.
    Supports D-dimensional z (D channels).

    All stored values are Python floats/lists — never tensors with grad.
    """

    def __init__(self, max_batches: int = 64, eps: float = 1e-6,
                 beta_hard_clamp: float = BETA_HARD_CLAMP,
                 eta: float = ETA_DEFAULT,
                 corr_min: float = CORR_MIN_DEFAULT,
                 n_min: int = N_MIN_DEFAULT):
        self.max_batches = max_batches
        self.eps = eps
        self.beta_hard_clamp = beta_hard_clamp
        self.eta = eta
        self.corr_min = corr_min
        self.n_min = n_min
        self._buffer: deque = deque(maxlen=max_batches)
        self._unique_k_values: set = set()
        self._D: Optional[int] = None  # feature dim, set on first update

    def reset(self):
        """Reset window (call at epoch start if reset_each_epoch=True)."""
        self._buffer.clear()
        self._unique_k_values.clear()

    def update(self, z: torch.Tensor, K: torch.Tensor,
               training: bool = True):
        """Add a batch to the window.

        Args:
            z: (B,) or (B, D) — features. Detached + stored as CPU floats.
            K: (B,) — syndrome count. Detached + stored as CPU floats.
            training: must be True to update (eval guard).
        """
        if not training:
            return

        # Detach + CPU + flatten K
        z_d = z.detach().cpu().float()
        K_d = K.detach().cpu().float().reshape(-1)

        # Ensure z is 2D: (B, D)
        if z_d.ndim == 1:
            z_d = z_d.unsqueeze(1)
        B, D = z_d.shape

        if B < 2:
            return

        # Set/verify feature dim
        if self._D is None:
            self._D = D
        # If D changed (shouldn't happen), skip
        if D != self._D:
            return

        # Store as Python floats/lists only
        self._buffer.append({
            'n': int(B),
            'sum_k': float(K_d.sum().item()),
            'sum_k2': float((K_d ** 2).sum().item()),
            'sum_z': [float(v) for v in z_d.sum(dim=0).tolist()],
            'sum_z2': [float(v) for v in (z_d ** 2).sum(dim=0).tolist()],
            'sum_zk': [float(v) for v in (z_d * K_d.unsqueeze(1)).sum(dim=0).tolist()],
        })

        # Track unique K (rounded ints)
        self._unique_k_values.update(K_d.round().long().tolist())

    @property
    def unique_K_count(self) -> int:
        return len(self._unique_k_values)

    @property
    def total_n(self) -> int:
        return sum(b['n'] for b in self._buffer)

    def compute_vector_beta(self) -> Dict[str, object]:
        """Compute per-channel beta with magnitude cap + corr-floor.

        Returns dict with:
          beta_cap: list[float] of length D (capped, clamped, floor-gated)
          beta_raw: list[float] of length D
          sigma_z: list[float], sigma_k: float
          corr: list[float] per channel
          var_k0: float, mean_k: float
          unique_K: int, total_n: int
          no_op: bool
          std_corr_ratio: float
          clamp_hit_rate: float
          corr_min_hit_rate: float
          beta_vec_std: float
        """
        D = self._D or 1
        empty = {
            'beta_cap': [0.0] * D, 'beta_raw': [0.0] * D,
            'sigma_z': [0.0] * D, 'sigma_k': 0.0,
            'corr': [0.0] * D,
            'var_k0': 0.0, 'mean_k': 0.0,
            'unique_K': 0, 'total_n': 0,
            'no_op': True,
            'std_corr_ratio': 0.0, 'clamp_hit_rate': 0.0,
            'corr_min_hit_rate': 0.0, 'beta_vec_std': 0.0,
        }

        if len(self._buffer) == 0:
            return empty

        N = self.total_n
        unique_K = self.unique_K_count

        # NO_OP: not enough data or K diversity
        if N < self.n_min or unique_K < 2:
            empty['total_n'] = N
            empty['unique_K'] = unique_K
            return empty

        # Aggregate sufficient statistics
        sum_k = sum(b['sum_k'] for b in self._buffer)
        sum_k2 = sum(b['sum_k2'] for b in self._buffer)
        sum_z = [sum(b['sum_z'][i] for b in self._buffer) for i in range(D)]
        sum_z2 = [sum(b['sum_z2'][i] for b in self._buffer) for i in range(D)]
        sum_zk = [sum(b['sum_zk'][i] for b in self._buffer) for i in range(D)]

        mean_k = sum_k / N
        var_k0 = sum_k2 / N - mean_k ** 2

        # NO_OP: near-zero K variance
        if var_k0 < self.eps:
            empty['total_n'] = N
            empty['unique_K'] = unique_K
            empty['var_k0'] = var_k0
            empty['mean_k'] = mean_k
            return empty

        sigma_k = math.sqrt(max(var_k0, 0.0))

        # Per-channel stats
        mean_z = [sum_z[i] / N for i in range(D)]
        var_z = [max(sum_z2[i] / N - mean_z[i] ** 2, 0.0) for i in range(D)]
        sigma_z = [math.sqrt(v) for v in var_z]
        cov = [sum_zk[i] / N - mean_z[i] * mean_k for i in range(D)]

        # Per-channel beta, correlation, cap, floor
        beta_raw = []
        beta_cap = []
        corr_vals = []
        cap_hits = 0
        floor_hits = 0

        for i in range(D):
            # Raw beta (eps-safe)
            b_raw = cov[i] / (var_k0 + self.eps)
            beta_raw.append(b_raw)

            # Correlation (eps-safe)
            denom_corr = (sigma_z[i] * sigma_k + self.eps)
            corr_i = cov[i] / denom_corr
            corr_vals.append(corr_i)

            # Corr-floor gating: if too small, zero out
            if abs(corr_i) < self.corr_min:
                beta_cap.append(0.0)
                floor_hits += 1
                continue

            # Magnitude cap: sigma_corr = |beta| * sigma_k <= eta * sigma_z
            sigma_corr = abs(b_raw) * sigma_k
            eta_limit = self.eta * sigma_z[i]
            if sigma_corr > eta_limit + self.eps:
                scale = eta_limit / (sigma_corr + self.eps)
                b_capped = b_raw * scale
                cap_hits += 1
            else:
                b_capped = b_raw

            # Hard clamp
            b_capped = max(-self.beta_hard_clamp,
                           min(self.beta_hard_clamp, b_capped))

            beta_cap.append(b_capped)

        # Telemetry
        std_corr_ratio_vals = [abs(beta_raw[i]) * sigma_k / (sigma_z[i] + self.eps)
                               for i in range(D)]
        std_corr_ratio = sum(std_corr_ratio_vals) / D if D > 0 else 0.0
        clamp_hit_rate = cap_hits / D if D > 0 else 0.0
        corr_min_hit_rate = floor_hits / D if D > 0 else 0.0
        beta_mean = sum(beta_cap) / D if D > 0 else 0.0
        beta_vec_std = math.sqrt(sum((b - beta_mean) ** 2 for b in beta_cap) / max(D, 1))

        return {
            'beta_cap': beta_cap, 'beta_raw': beta_raw,
            'sigma_z': sigma_z, 'sigma_k': sigma_k,
            'corr': corr_vals,
            'var_k0': var_k0, 'mean_k': mean_k,
            'unique_K': unique_K, 'total_n': N,
            'no_op': False,
            'std_corr_ratio': std_corr_ratio,
            'clamp_hit_rate': clamp_hit_rate,
            'corr_min_hit_rate': corr_min_hit_rate,
            'beta_vec_std': beta_vec_std,
        }


def k_orthogonalize_v2(z: torch.Tensor, K: torch.Tensor,
                       window: VectorKBetaWindow,
                       epoch: int = 1,
                       training: bool = True,
                       frozen_beta: Optional[List[float]] = None,
                       ) -> tuple:
    """Apply vector K-orthogonalization with magnitude cap + corr-floor + ramp.

    Args:
        z: (B,) or (B, D) — Z_g1. NOT modified in-place.
        K: (B,) — syndrome count
        window: VectorKBetaWindow
        epoch: current epoch (for ramp schedule)
        training: if True, update window
        frozen_beta: if provided, use this beta instead of computing from window
                     (for frozen-epoch arm; still applies cap/floor from window stats)

    Returns:
        (z_ortho, info_dict). z_ortho has same shape/device/dtype as z.
    """
    # Update window (training only)
    window.update(z, K, training=training)

    # Ramp weight
    w = get_ramp_weight(epoch)

    # Compute stats (always, for telemetry)
    stats = window.compute_vector_beta()

    info = {
        'beta_cap': stats['beta_cap'],
        'beta_raw': stats['beta_raw'],
        'beta_norm': sum(abs(b) for b in stats['beta_cap']) / max(len(stats['beta_cap']), 1),
        'var_k0': stats['var_k0'],
        'mean_k': stats['mean_k'],
        'unique_K_in_beta_window': stats['unique_K'],
        'total_n_in_window': stats['total_n'],
        'NO_OP_K_ORTHO': stats['no_op'] or w == 0.0,
        'k_ortho_weight': w,
        'epoch': epoch,
        'std_corr_ratio': stats['std_corr_ratio'],
        'clamp_hit_rate': stats['clamp_hit_rate'],
        'corr_min_hit_rate': stats['corr_min_hit_rate'],
        'beta_vec_std': stats['beta_vec_std'],
    }

    # No-op conditions: ramp=0, or window says no-op
    if stats['no_op'] or w == 0.0:
        return z, info

    # Use frozen beta if provided (frozen-epoch arm)
    beta_list = frozen_beta if frozen_beta is not None else stats['beta_cap']

    # Build beta tensor on z's device/dtype
    beta_t = torch.tensor(beta_list, dtype=z.dtype, device=z.device)

    # k0 = K - mean_k (detached, on z's device)
    k0 = (K.detach().float() - stats['mean_k']).to(device=z.device, dtype=z.dtype)

    # Shape alignment
    z_2d = z if z.ndim >= 2 else z.unsqueeze(1)
    k0_bc = k0.unsqueeze(1)  # (B, 1) for broadcast with (B, D)
    beta_bc = beta_t.unsqueeze(0)  # (1, D) for broadcast with (B, D)

    # Out-of-place: z_ortho = z - w * beta * k0
    z_ortho_2d = z_2d - w * beta_bc * k0_bc

    # Restore original shape
    z_ortho = z_ortho_2d if z.ndim >= 2 else z_ortho_2d.squeeze(1)

    info['NO_OP_K_ORTHO'] = False
    return z_ortho, info


# ── Legacy compat (Day 56) ──────────────────────────────────────────────
# Keep old names importable for Day 56 tests
KBetaWindow = VectorKBetaWindow
k_orthogonalize = k_orthogonalize_v2


# ═══════════════════════════════════════════════════════════════════════
# Day 58 — Cross-Epoch EMA Vector β + Global Predictive Gate
# ═══════════════════════════════════════════════════════════════════════

R_GLOBAL_MIN_DEFAULT = 0.10  # global predictive gate threshold
EMA_ALPHA_DEFAULT = 0.05     # EMA momentum


class EMAKOrtho(torch.nn.Module):
    """Cross-epoch EMA-based K-orthogonalization with global predictive gate.

    Replaces VectorKBetaWindow (Day 57). Key differences:
      - EMA sufficient stats (no rolling window, no per-epoch reset)
      - register_buffer for device safety (moves with .to(device))
      - First-batch exact init (no zero-bias lag)
      - Global predictive gate (R_global = |Corr(Z·β, k0)| ≥ threshold)
      - No per-channel corr-floor (incompatible with distributed leakage)
      - Magnitude cap + hard clamp (same safety as Day 57)

    Usage:
      ortho = EMAKOrtho(D=1)
      z_out, info = ortho(z, K, epoch=5, training=True)
    """

    def __init__(self, D: int = 1,
                 ema_alpha: float = EMA_ALPHA_DEFAULT,
                 eps: float = 1e-6,
                 beta_hard_clamp: float = BETA_HARD_CLAMP,
                 eta: float = ETA_DEFAULT,
                 r_global_min: float = R_GLOBAL_MIN_DEFAULT):
        super().__init__()
        self.D = D
        self.ema_alpha = ema_alpha
        self.eps = eps
        self.beta_hard_clamp = beta_hard_clamp
        self.eta = eta
        self.r_global_min = r_global_min

        # EMA buffers — registered for device/dtype tracking
        self.register_buffer('cov_zk_ema', torch.zeros(D))
        self.register_buffer('var_k_ema', torch.zeros(1))
        self.register_buffer('mu_z_ema', torch.zeros(D))
        self.register_buffer('mu_k_ema', torch.zeros(1))
        self.register_buffer('sigma_z_ema', torch.zeros(D))
        self.register_buffer('n_seen', torch.zeros(1, dtype=torch.long))
        self._initialized = False

    def reset(self):
        """Reset EMA state (normally NOT called — cross-epoch by design)."""
        self.cov_zk_ema.zero_()
        self.var_k_ema.zero_()
        self.mu_z_ema.zero_()
        self.mu_k_ema.zero_()
        self.sigma_z_ema.zero_()
        self.n_seen.zero_()
        self._initialized = False

    def _init_from_batch(self, z: torch.Tensor, k: torch.Tensor):
        """Initialize EMA with exact stats from the first batch (no zero-bias)."""
        # z: (B, D), k: (B,)
        B = z.shape[0]
        mu_z = z.mean(dim=0)       # (D,)
        mu_k = k.mean()            # scalar
        z_c = z - mu_z.unsqueeze(0)
        k_c = k - mu_k
        cov = (z_c * k_c.unsqueeze(1)).mean(dim=0)  # (D,)
        var_k = k_c.var(correction=0)                 # scalar
        sigma_z = z.std(dim=0, correction=0)           # (D,)

        self.mu_z_ema.copy_(mu_z.detach())
        self.mu_k_ema.fill_(mu_k.detach().item())
        self.cov_zk_ema.copy_(cov.detach())
        self.var_k_ema.fill_(var_k.detach().item())
        self.sigma_z_ema.copy_(sigma_z.detach())
        self.n_seen.fill_(B)
        self._initialized = True

    def _ema_update(self, z: torch.Tensor, k: torch.Tensor):
        """Update EMA sufficient stats with a new batch."""
        alpha = self.ema_alpha
        B = z.shape[0]

        mu_z_batch = z.mean(dim=0).detach()       # (D,)
        mu_k_batch = k.mean().detach()             # scalar
        z_c = z - mu_z_batch.unsqueeze(0)
        k_c = k - mu_k_batch
        cov_batch = (z_c * k_c.unsqueeze(1)).mean(dim=0).detach()  # (D,)
        var_k_batch = k_c.var(correction=0).detach()                # scalar
        sigma_z_batch = z.std(dim=0, correction=0).detach()         # (D,)

        self.mu_z_ema.mul_(1 - alpha).add_(mu_z_batch * alpha)
        self.mu_k_ema.mul_(1 - alpha).add_(mu_k_batch * alpha)
        self.cov_zk_ema.mul_(1 - alpha).add_(cov_batch * alpha)
        self.var_k_ema.mul_(1 - alpha).add_(var_k_batch * alpha)
        self.sigma_z_ema.mul_(1 - alpha).add_(sigma_z_batch * alpha)
        self.n_seen.add_(B)

    def compute_beta(self) -> torch.Tensor:
        """Compute beta_ema = cov / (var + eps). Returns (D,) tensor."""
        return self.cov_zk_ema / (self.var_k_ema + self.eps)

    def apply_magnitude_cap(self, beta: torch.Tensor) -> tuple:
        """Apply per-channel magnitude cap + hard clamp.

        Returns (beta_capped, clamp_hit_rate).
        """
        sigma_k = torch.sqrt(torch.clamp(self.var_k_ema, min=0.0)) + self.eps
        sigma_z = torch.clamp(self.sigma_z_ema, min=self.eps)

        # Per-channel cap: |beta_i| * sigma_k <= eta * sigma_z_i
        sigma_corr = beta.abs() * sigma_k          # (D,)
        eta_limit = self.eta * sigma_z              # (D,)
        needs_cap = sigma_corr > eta_limit
        scale = torch.where(needs_cap,
                            eta_limit / (sigma_corr + self.eps),
                            torch.ones_like(beta))
        beta_capped = beta * scale

        # Hard clamp
        beta_capped = torch.clamp(beta_capped,
                                  min=-self.beta_hard_clamp,
                                  max=self.beta_hard_clamp)

        clamp_hit = needs_cap.float().mean().item()
        return beta_capped, clamp_hit

    def compute_r_global(self, z: torch.Tensor, k0: torch.Tensor,
                         beta: torch.Tensor) -> float:
        """Safe Pearson: R_global = |Corr(Z·β, k0)|.

        Uses clamped stds to prevent NaN from zero-variance inputs.
        """
        # k_pred = (Z_centered * beta).sum(dim=1) → prediction of k
        k_pred = (z * beta.unsqueeze(0)).sum(dim=1)  # (B,)

        # Safe std (clamp at 1e-8 to prevent division by zero)
        std_pred = torch.clamp(torch.std(k_pred), min=1e-8)
        std_k0 = torch.clamp(torch.std(k0), min=1e-8)

        # Covariance
        cov_pred_k0 = ((k_pred - k_pred.mean()) * (k0 - k0.mean())).mean()

        r_global = torch.abs(cov_pred_k0 / (std_pred * std_k0))
        return r_global.detach().item()

    def forward(self, z: torch.Tensor, K: torch.Tensor,
                epoch: int = 1,
                training: bool = True,
                frozen_beta: Optional[List[float]] = None,
                ) -> tuple:
        """Apply EMA K-orthogonalization with global gate + cap + ramp.

        Args:
            z: (B,) or (B, D) — Z_g1. NOT modified in-place.
            K: (B,) — syndrome count
            epoch: current epoch (for ramp schedule)
            training: if True, update EMA
            frozen_beta: if provided, use instead of live EMA beta

        Returns:
            (z_ortho, info_dict). z_ortho same shape/device/dtype as z.
        """
        # Ensure 2D
        z_2d = z if z.ndim >= 2 else z.unsqueeze(1)
        B, D = z_2d.shape
        k_flat = K.detach().float().reshape(-1)

        # k0 = K - mu_k (use EMA mean if initialized, else batch mean)
        if self._initialized:
            mean_k = self.mu_k_ema.item()
        else:
            mean_k = k_flat.mean().item()
        k0 = (k_flat - mean_k).to(device=z.device, dtype=z.dtype)

        # Schedule weight
        w = get_ramp_weight(epoch)

        # Determine NO_OP reason
        no_op_reason = None
        if w == 0.0:
            no_op_reason = 'NO_OP_SCHEDULE_OFF'
        elif not self._initialized and not training:
            no_op_reason = 'NO_OP_NOT_INITIALIZED'

        # EMA update (training only, batch must have B >= 2)
        if training and B >= 2:
            z_detached = z_2d.detach()
            k_detached = k_flat.detach().to(device=z.device, dtype=z.dtype)
            if not self._initialized:
                self._init_from_batch(z_detached, k_detached)
            else:
                self._ema_update(z_detached, k_detached)

        # Compute beta
        if frozen_beta is not None:
            beta = torch.tensor(frozen_beta, dtype=z.dtype, device=z.device)
        elif self._initialized:
            beta = self.compute_beta()
        else:
            beta = torch.zeros(D, dtype=z.dtype, device=z.device)

        # Magnitude cap
        beta_cap, clamp_hit = self.apply_magnitude_cap(beta)

        # Global predictive gate
        r_global = 0.0
        gate_triggered = False
        if self._initialized and w > 0.0 and B >= 2:
            r_global = self.compute_r_global(z_2d.detach(), k0.detach(), beta_cap.detach())
            if r_global < self.r_global_min:
                no_op_reason = 'NO_OP_R_GLOBAL_LOW'
                gate_triggered = True

        # Check var_k
        var_k = self.var_k_ema.item() if self._initialized else 0.0
        if var_k < self.eps and no_op_reason is None and w > 0.0:
            no_op_reason = 'NO_OP_VAR0'

        is_no_op = no_op_reason is not None

        # Telemetry
        beta_list = beta_cap.detach().cpu().tolist()
        beta_raw_list = beta.detach().cpu().tolist() if self._initialized else [0.0] * D
        info = {
            'beta_cap': beta_list,
            'beta_raw': beta_raw_list,
            'beta_norm': sum(abs(b) for b in beta_list) / max(D, 1),
            'var_k0': var_k,
            'mean_k': mean_k,
            'unique_K_in_beta_window': 0,  # not tracked in EMA
            'total_n_in_window': int(self.n_seen.item()),
            'NO_OP_K_ORTHO': is_no_op,
            'no_op_reason': no_op_reason or '',
            'k_ortho_weight': w,
            'epoch': epoch,
            'r_global_pre': r_global,
            'r_global_gate_triggered': gate_triggered,
            'clamp_hit_rate': clamp_hit,
            'beta_vec_std': float(torch.std(beta_cap).item()) if D > 1 else 0.0,
            'ema_initialized': self._initialized,
            'n_seen': int(self.n_seen.item()),
        }

        # Compute eff_corr_ratio
        if not is_no_op:
            correction = w * beta_cap.unsqueeze(0) * k0.unsqueeze(1)
            eff_corr = correction.norm() / (z_2d.detach().norm() + self.eps)
            info['eff_corr_ratio'] = eff_corr.item()
        else:
            info['eff_corr_ratio'] = 0.0

        # Apply or skip
        if is_no_op:
            return z, info

        # Out-of-place: z_ortho = z - w * beta_cap * k0
        k0_bc = k0.unsqueeze(1)        # (B, 1)
        beta_bc = beta_cap.unsqueeze(0)  # (1, D)
        z_ortho_2d = z_2d - w * beta_bc * k0_bc

        # Restore original shape
        z_ortho = z_ortho_2d if z.ndim >= 2 else z_ortho_2d.squeeze(1)

        return z_ortho, info


# ═══════════════════════════════════════════════════════════════════════
# Day 59 — Frozen-Beta K-Orthogonalization (StatsSet-Based)
# ═══════════════════════════════════════════════════════════════════════

class FrozenBetaKOrtho:
    """Day 59: Frozen per-epoch β computed from a dedicated OrthoStatSet.

    Key differences from Day 57/58:
      - β computed ONCE per epoch from a large fixed dataset (N=4096)
        under torch.no_grad() with ortho disabled → no batch noise
      - No rolling window, no EMA, no per-batch gate
      - All stored state is Python floats (no graph refs, no OOM)
      - Magnitude cap + hard clamp (same safety invariants)
      - k0 clamped to ±3σ_k to prevent outlier-driven over-correction

    Usage at epoch boundary:
      frozen_ortho.compute_from_statset(model, statset_data, device)
    Usage per batch:
      z_out, info = frozen_ortho.apply(z, K, w)
    """

    def __init__(self, D: int = 1,
                 eta: float = ETA_DEFAULT,
                 eps: float = 1e-6,
                 beta_hard_clamp: float = BETA_HARD_CLAMP,
                 k0_clamp_sigma: float = 3.0):
        self.D = D
        self.eta = eta
        self.eps = eps
        self.beta_hard_clamp = beta_hard_clamp
        self.k0_clamp_sigma = k0_clamp_sigma

        # Frozen state (all Python floats/lists — no graph refs)
        self.beta_cap: List[float] = [0.0] * D
        self.beta_raw: List[float] = [0.0] * D
        self.mu_k: float = 0.0
        self.sigma_k: float = 1.0
        self.sigma_z: List[float] = [1.0] * D
        self.var_k: float = 1.0
        self.n_stat: int = 0
        self._computed = False

        # Telemetry from last compute
        self.last_telemetry: Dict = {}

    def compute_from_statset(
        self,
        model: torch.nn.Module,
        statset_data: dict,
        device: torch.device,
    ) -> Dict:
        """Compute frozen β from the OrthoStatSet at epoch boundary.

        Runs model.forward_split() on entire statset under torch.no_grad()
        and model.eval() with ortho disabled. Collects global moments,
        computes β = Cov(z,k) / (Var(k)+ε), applies magnitude cap.

        Args:
            model: FactorGraphDecoderV1 (temporarily set to eval, ortho OFF)
            statset_data: dict with det_feats, err_feats, ei_d2e, ei_e2d,
                          error_weights, observable_mask, K
            device: torch device

        Returns:
            telemetry dict
        """
        was_training = model.training
        # Temporarily disable all ortho paths
        old_frozen = getattr(model, '_k_ortho_frozen', None)
        model._k_ortho_frozen = None
        old_mod = getattr(model, '_k_ortho_module', None)
        model._k_ortho_module = None
        old_win = getattr(model, '_k_ortho_window', None)
        model._k_ortho_window = None
        old_use = getattr(model, 'fg_use_k_ortho', False)
        model.fg_use_k_ortho = False

        model.eval()
        with torch.no_grad():
            ew = statset_data.get('err_w', statset_data.get('error_weights'))
            om = statset_data.get('obs_mask', statset_data.get('observable_mask'))
            split = model.forward_split(
                statset_data['det_feats'].to(device),
                statset_data['err_feats'].to(device),
                statset_data['ei_d2e'].to(device),
                statset_data['ei_e2d'].to(device),
                error_weights=ew.to(device) if ew is not None else None,
                observable_mask=om.to(device) if om is not None else None,
            )
            z = split['logit_residual_norm'].detach().cpu().float()
            K_float = split['K'].detach().cpu().float()

        # Restore model state
        model.fg_use_k_ortho = old_use
        model._k_ortho_frozen = old_frozen
        model._k_ortho_module = old_mod
        model._k_ortho_window = old_win
        if was_training:
            model.train()

        # Ensure 2D
        if z.ndim == 1:
            z = z.unsqueeze(1)
        K_flat = K_float.reshape(-1)
        N, D = z.shape

        # Global moments
        mu_k = K_flat.mean().item()
        var_k = K_flat.var(correction=0).item()
        sigma_k = math.sqrt(max(var_k, 0.0))
        mu_z = z.mean(dim=0)
        sigma_z = z.std(dim=0, correction=0)

        # k0 = K - mu_k
        k0 = K_flat - mu_k

        # Per-channel Cov(z, k) / (Var(k) + eps)
        z_c = z - mu_z.unsqueeze(0)
        cov_zk = (z_c * k0.unsqueeze(1)).mean(dim=0)

        beta_raw = []
        beta_cap = []
        cap_hits = 0
        scale_vals = []

        for i in range(D):
            b_raw = cov_zk[i].item() / (var_k + self.eps)
            beta_raw.append(b_raw)

            sz_i = max(sigma_z[i].item(), self.eps)

            # Magnitude cap: |beta| * sigma_k <= eta * sigma_z
            sigma_corr = abs(b_raw) * sigma_k
            eta_limit = self.eta * sz_i
            if sigma_corr > eta_limit + self.eps:
                s = eta_limit / (sigma_corr + self.eps)
                b_capped = b_raw * s
                cap_hits += 1
                scale_vals.append(s)
            else:
                b_capped = b_raw
                scale_vals.append(1.0)

            # Hard clamp
            b_capped = max(-self.beta_hard_clamp,
                           min(self.beta_hard_clamp, b_capped))
            beta_cap.append(b_capped)

        # Store frozen state
        self.beta_cap = beta_cap
        self.beta_raw = beta_raw
        self.mu_k = mu_k
        self.sigma_k = sigma_k
        self.sigma_z = [sigma_z[i].item() for i in range(D)]
        self.var_k = var_k
        self.n_stat = N
        self._computed = True
        self.D = D

        # Telemetry
        beta_l2 = math.sqrt(sum(b ** 2 for b in beta_cap))
        mean_scale_s = sum(scale_vals) / max(len(scale_vals), 1)
        clamp_hit_rate = cap_hits / max(D, 1)
        beta_mean = sum(beta_cap) / max(D, 1)
        beta_std = (math.sqrt(sum((b - beta_mean) ** 2
                                  for b in beta_cap) / max(D, 1))
                    if D > 1 else 0.0)

        self.last_telemetry = {
            'beta_cap': list(beta_cap),
            'beta_raw': list(beta_raw),
            'beta_l2': beta_l2,
            'beta_std': beta_std,
            'clamp_hit_rate': clamp_hit_rate,
            'mean_scale_s': mean_scale_s,
            'sigma_k': sigma_k,
            'var_k': var_k,
            'mu_k': mu_k,
            'sigma_z': [sigma_z[i].item() for i in range(D)],
            'n_stat': N,
            'eta': self.eta,
        }
        return self.last_telemetry

    def apply(self, z: torch.Tensor, K: torch.Tensor,
              w: float) -> tuple:
        """Apply frozen β correction.

        Args:
            z: (B,) or (B, D) — Z_g1. NOT modified in-place.
            K: (B,) — syndrome count
            w: ramp schedule weight (0.0 to 1.0)

        Returns:
            (z_ortho, info_dict)
        """
        z_2d = z if z.ndim >= 2 else z.unsqueeze(1)
        B, D = z_2d.shape
        k_flat = K.detach().float().reshape(-1)

        # Build k0 (centered, clamped)
        k0 = (k_flat - self.mu_k).to(device=z.device, dtype=z.dtype)
        k0_limit = self.k0_clamp_sigma * max(self.sigma_k, self.eps)
        k0_clamped = torch.clamp(k0, min=-k0_limit, max=k0_limit)
        k0_clamp_rate = (k0.abs() > k0_limit).float().mean().item()

        # Build beta tensor
        beta_t = torch.tensor(self.beta_cap, dtype=z.dtype, device=z.device)

        # Check for effective NO_OP
        beta_norm = beta_t.abs().sum().item()
        is_no_op = (not self._computed) or (w == 0.0) or (beta_norm < self.eps)

        info = {
            'beta_cap': list(self.beta_cap),
            'beta_raw': list(self.beta_raw),
            'beta_norm': beta_norm / max(D, 1),
            'var_k0': self.var_k,
            'mean_k': self.mu_k,
            'NO_OP_K_ORTHO': is_no_op,
            'no_op_reason': ('NOT_COMPUTED' if not self._computed
                             else 'SCHEDULE_OFF' if w == 0.0
                             else 'BETA_ZERO' if beta_norm < self.eps
                             else ''),
            'k_ortho_weight': w,
            'k0_clamp_rate': k0_clamp_rate,
            'eff_corr_ratio': 0.0,
            'frozen_beta_mode': True,
        }

        if is_no_op:
            return z, info

        # Out-of-place correction
        k0_bc = k0_clamped.unsqueeze(1)    # (B, 1)
        beta_bc = beta_t.unsqueeze(0)      # (1, D)
        delta = w * beta_bc * k0_bc        # (B, D)
        z_ortho_2d = z_2d - delta

        # eff_corr_ratio
        z_norm_val = z_2d.detach().norm().item()
        delta_norm = delta.detach().norm().item()
        info['eff_corr_ratio'] = delta_norm / (z_norm_val + self.eps)
        info['NO_OP_K_ORTHO'] = False
        info['no_op_reason'] = ''

        # Restore shape
        z_ortho = z_ortho_2d if z.ndim >= 2 else z_ortho_2d.squeeze(1)
        return z_ortho, info


def generate_ortho_statset(distance: int, p: float, basis: str,
                           n_ortho: int, seed: int,
                           corr_strength: float = 0.5) -> dict:
    """Generate a dedicated OrthoStatSet, disjoint from ProbeSet.

    Uses ortho_seed = (seed + 99991) + 13337 for deterministic
    separation from probe_seed = seed + 99991.

    Follows exact same graph construction as generate_probe_set to ensure
    det_feats dimensions match edge indices (N_det_graph, not n_det).

    Returns dict with det_feats, err_feats, edge indices, K, and metadata.
    """
    import numpy as np
    from qec_noise_factory.ml.bench.regime_lock import RegimeLock, generate_locked_data
    from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count
    from qec_noise_factory.ml.graph.dem_bipartite import (
        build_bipartite_graph, bipartite_graph_to_tensors)

    ortho_seed = (seed + 99991) + 13337

    # Generate data with ortho_seed
    lock = RegimeLock(
        distance=distance, target_p=p, basis=basis,
        require_generated=True, n_samples=n_ortho,
        corr_strength=corr_strength, seed=ortho_seed,
    )
    X, Y = generate_locked_data(lock)
    n_det = X.shape[1]

    # Build bipartite graph (same as probe_set and build_data)
    bg = build_bipartite_graph(
        distance=distance, rounds=distance, p=p,
        basis=basis, noise_model="correlated_crosstalk_like")
    ei_d2e, ei_e2d, err_w, obs_mask = bipartite_graph_to_tensors(bg)
    N_det_graph = bg.num_detectors

    # Build det_feats with graph-sized padding (critical for edge index compat)
    B = X.shape[0]
    feats = np.zeros((B, N_det_graph, 2), dtype=np.float32)
    feats[:, :n_det, 0] = X.astype(np.float32)
    feats[:, -1, 1] = 1.0
    det_feats = torch.from_numpy(feats)

    K = compute_syndrome_count(X)

    return {
        'det_feats': det_feats,
        'err_feats': torch.from_numpy(bg.error_weights.reshape(-1, 1)).float(),
        'ei_d2e': ei_d2e,
        'ei_e2d': ei_e2d,
        'err_w': err_w,
        'obs_mask': obs_mask,
        'error_weights': err_w,
        'observable_mask': obs_mask,
        'K': torch.from_numpy(K.astype(np.float32)),
        'X_raw': X,
        'Y_raw': Y,
        'ortho_seed': ortho_seed,
        'n_ortho': B,
        'n_det': n_det,
        'N_det_graph': N_det_graph,
        'distance': distance,
        'p': p,
        'basis': basis,
        'corr_strength': corr_strength,
    }


# ═══════════════════════════════════════════════════════════════════════
# Day 60 — Epoch-Rolling Scalar K-Orthogonalization + Do-No-Harm Gate
# ═══════════════════════════════════════════════════════════════════════

R2_GATE_THRESHOLD = 0.015   # Do-No-Harm: β=0 when R²_probe ≤ this
BETA_MIN_DEFAULT = 1e-4     # |β| below this counts as NO_OP
ECR_MIN_DEFAULT = 1e-4      # eff_corr_ratio below this counts as NO_OP


class EpochRollingKOrtho:
    """Day 60: Epoch-rolling β with Do-No-Harm gate.

    Key differences from Day 59 (FrozenBetaKOrtho):
      - β recomputed EVERY epoch from fresh forward_split() on OrthoStatSet
        → no stale-beta trap (Day 59 root cause)
      - Do-No-Harm gate: R²_probe ≤ 0.015 → β=0, need_ortho=False
        → safe when model is already clean
      - Per-sample delta magnitude cap: clamp(delta, -η*std_z, +η*std_z)
      - All stored state is Python floats (no graph refs, no OOM)

    Usage at epoch boundary (after epoch E-1):
      ortho.compute_from_statset(model, statset_data, device)
    Usage per batch during epoch E (training only):
      z_out, info = ortho.apply(z, K, w)
    """

    def __init__(self, D: int = 1,
                 eta: float = ETA_DEFAULT,
                 eps: float = 1e-6,
                 beta_hard_clamp: float = BETA_HARD_CLAMP,
                 r2_gate_threshold: float = R2_GATE_THRESHOLD,
                 beta_min: float = BETA_MIN_DEFAULT,
                 ecr_min: float = ECR_MIN_DEFAULT):
        self.D = D
        self.eta = eta
        self.eps = eps
        self.beta_hard_clamp = beta_hard_clamp
        self.r2_gate_threshold = r2_gate_threshold
        self.beta_min = beta_min
        self.ecr_min = ecr_min

        # Epoch-rolling state (all Python floats — no graph refs)
        self.beta_E: float = 0.0
        self.beta_raw: float = 0.0
        self.mu_k: float = 0.0
        self.var_k: float = 1.0
        self.std_z: float = 1.0
        self.R2_probe: float = 0.0
        self.need_ortho: bool = False
        self.n_stat: int = 0
        self._computed: bool = False

        # Telemetry from last compute
        self.last_telemetry: Dict = {}

    def compute_from_statset(
        self,
        model: torch.nn.Module,
        statset_data: dict,
        device: torch.device,
    ) -> Dict:
        """Recompute β at epoch boundary from the OrthoStatSet.

        Runs model.forward_split() on entire statset under torch.no_grad()
        and model.eval() with ortho disabled. Computes fresh β and R²_probe,
        then applies the Do-No-Harm gate.

        Args:
            model: FactorGraphDecoderV1 (temporarily set to eval, ortho OFF)
            statset_data: dict with det_feats, err_feats, ei_d2e, ei_e2d,
                          error_weights, observable_mask, K
            device: torch device

        Returns:
            telemetry dict
        """
        was_training = model.training
        # Temporarily disable ALL ortho paths (Day 57-60)
        old_epoch_rolling = getattr(model, '_k_ortho_epoch_rolling', None)
        model._k_ortho_epoch_rolling = None
        old_frozen = getattr(model, '_k_ortho_frozen', None)
        model._k_ortho_frozen = None
        old_mod = getattr(model, '_k_ortho_module', None)
        model._k_ortho_module = None
        old_win = getattr(model, '_k_ortho_window', None)
        model._k_ortho_window = None
        old_use = getattr(model, 'fg_use_k_ortho', False)
        model.fg_use_k_ortho = False

        model.eval()
        with torch.no_grad():
            ew = statset_data.get('err_w', statset_data.get('error_weights'))
            om = statset_data.get('obs_mask', statset_data.get('observable_mask'))
            split = model.forward_split(
                statset_data['det_feats'].to(device),
                statset_data['err_feats'].to(device),
                statset_data['ei_d2e'].to(device),
                statset_data['ei_e2d'].to(device),
                error_weights=ew.to(device) if ew is not None else None,
                observable_mask=om.to(device) if om is not None else None,
            )
            z = split['logit_residual_norm'].detach().cpu().float()
            K_float = split['K'].detach().cpu().float()

        # Restore model state
        model.fg_use_k_ortho = old_use
        model._k_ortho_epoch_rolling = old_epoch_rolling
        model._k_ortho_frozen = old_frozen
        model._k_ortho_module = old_mod
        model._k_ortho_window = old_win
        if was_training:
            model.train()

        # Flatten for scalar case
        z_flat = z.reshape(-1)
        K_flat = K_float.reshape(-1)
        N = z_flat.shape[0]

        # Global moments
        mu_k = float(K_flat.mean().item())
        var_k = float(K_flat.var(correction=0).item())
        mu_z = float(z_flat.mean().item())
        std_z = float(z_flat.std(correction=0).item())

        # k0 = K - mu_k
        k0 = K_flat - mu_k

        # Cov(Z_g1, K) and beta = Cov / (Var(K) + eps)
        z_c = z_flat - mu_z
        cov_zk = float((z_c * k0).mean().item())
        beta_raw = cov_zk / (var_k + self.eps)

        # R²_probe = Corr(Z_g1, K)² = (Cov / (std_z * std_k))²
        sigma_k = math.sqrt(max(var_k, 0.0))
        denom_corr = (max(std_z, self.eps) * max(sigma_k, self.eps))
        corr = cov_zk / denom_corr
        R2_probe = corr ** 2

        # ── Do-No-Harm Gate ──
        if R2_probe <= self.r2_gate_threshold:
            # Model is already clean → do not apply
            beta_E = 0.0
            need_ortho = False
            gate_action = 'GATE_OFF'
        else:
            # Leakage detected → apply with magnitude cap
            need_ortho = True
            gate_action = 'GATE_ON'

            # Magnitude cap: |β| * σ_k ≤ η * std_z
            sigma_corr = abs(beta_raw) * sigma_k
            eta_limit = self.eta * max(std_z, self.eps)
            if sigma_corr > eta_limit + self.eps:
                scale = eta_limit / (sigma_corr + self.eps)
                beta_E = beta_raw * scale
            else:
                beta_E = beta_raw

            # Hard clamp
            beta_E = max(-self.beta_hard_clamp,
                         min(self.beta_hard_clamp, beta_E))

        # Store epoch-rolling state (all Python floats)
        self.beta_E = float(beta_E)
        self.beta_raw = float(beta_raw)
        self.mu_k = mu_k
        self.var_k = var_k
        self.std_z = max(std_z, self.eps)
        self.R2_probe = float(R2_probe)
        self.need_ortho = need_ortho
        self.n_stat = N
        self._computed = True

        # Telemetry
        self.last_telemetry = {
            'beta_E': self.beta_E,
            'beta_raw': self.beta_raw,
            'R2_probe': self.R2_probe,
            'need_ortho': self.need_ortho,
            'gate_action': gate_action,
            'corr': float(corr),
            'mu_k': mu_k,
            'var_k': var_k,
            'std_z': std_z,
            'sigma_k': sigma_k,
            'n_stat': N,
            'eta': self.eta,
            'r2_gate_threshold': self.r2_gate_threshold,
        }
        return self.last_telemetry

    def apply(self, z: torch.Tensor, K: torch.Tensor,
              w: float) -> tuple:
        """Apply epoch-rolling β correction with Do-No-Harm gate.

        Args:
            z: (B,) or (B, D) — Z_g1. NOT modified in-place.
            K: (B,) — syndrome count
            w: ramp schedule weight (0.0 to 1.0)

        Returns:
            (z_ortho, info_dict)
        """
        z_2d = z if z.ndim >= 2 else z.unsqueeze(1)
        B = z_2d.shape[0]
        k_flat = K.detach().float().reshape(-1)

        # Determine NO_OP reason
        no_op_reason = ''
        if not self._computed:
            no_op_reason = 'NOT_COMPUTED'
        elif w == 0.0:
            no_op_reason = 'SCHEDULE_OFF'
        elif not self.need_ortho:
            no_op_reason = 'DO_NO_HARM'
        elif abs(self.beta_E) < self.beta_min:
            no_op_reason = 'BETA_ZERO'

        is_no_op = bool(no_op_reason)

        info = {
            'beta_E': self.beta_E,
            'beta_raw': self.beta_raw,
            'R2_probe': self.R2_probe,
            'need_ortho': self.need_ortho,
            'var_k0': self.var_k,
            'mean_k': self.mu_k,
            'std_z': self.std_z,
            'NO_OP_K_ORTHO': is_no_op,
            'no_op_reason': no_op_reason,
            'k_ortho_weight': w,
            'eff_corr_ratio': 0.0,
            'epoch_rolling_mode': True,
        }

        if is_no_op:
            return z, info

        # k0 = K_batch - mu_k (detached)
        k0 = (k_flat - self.mu_k).to(device=z.device, dtype=z.dtype)

        # delta = w * beta_E * k0 (both detached scalars, no graph)
        beta_val = torch.tensor(self.beta_E, dtype=z.dtype, device=z.device)
        delta_raw = w * beta_val * k0  # (B,)

        # Magnitude cap (blast shield): clamp per-sample
        cap = self.eta * self.std_z
        delta = torch.clamp(delta_raw, min=-cap, max=cap)

        # Reshape for subtraction
        delta_2d = delta.unsqueeze(1)  # (B, 1)

        # Out-of-place: Z_g1_ortho = Z_g1 - delta
        z_ortho_2d = z_2d - delta_2d

        # eff_corr_ratio = ||delta|| / (||Z_g1|| + eps)
        delta_norm = float(delta.detach().norm().item())
        z_norm_val = float(z_2d.detach().norm().item())
        ecr = delta_norm / (z_norm_val + self.eps)
        info['eff_corr_ratio'] = ecr

        # Check for effective NO_OP via eff_corr_ratio
        if ecr < self.ecr_min:
            info['NO_OP_K_ORTHO'] = True
            info['no_op_reason'] = 'ECR_TOO_LOW'
            return z, info

        info['NO_OP_K_ORTHO'] = False

        # Restore original shape
        z_ortho = z_ortho_2d if z.ndim >= 2 else z_ortho_2d.squeeze(1)
        return z_ortho, info


# ═══════════════════════════════════════════════════════════════════════
# Day 61 — OrthoGradientShield (autograd.Function)
# ═══════════════════════════════════════════════════════════════════════

class OrthoGradientShield(torch.autograd.Function):
    """Remove K-collinear component from gradients in backward pass.

    Forward: identity (Z_post passes through unchanged).
    Backward: grad_Z = grad - proj * K_cent
        where proj = mean(grad * K_cent) / (mean(K_cent²) + eps)

    This prevents the optimizer from "pre-cancelling" the K-ortho
    correction by learning to add K-collinear signal back into Z.

    Day 62: Added grad_proj_ratio telemetry.
    """

    @staticmethod
    def forward(ctx, z_post: torch.Tensor,
                k_cent: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(k_cent)
        return z_post

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        k_cent, = ctx.saved_tensors
        # k_cent shape: (B,) or (B, 1); match grad shape
        if k_cent.ndim < grad_output.ndim:
            k_cent = k_cent.unsqueeze(-1)
        # Project out K-collinear component
        k2_mean = (k_cent ** 2).mean() + 1e-6
        proj = (grad_output * k_cent).mean() / k2_mean
        removed_component = proj * k_cent
        grad_clean = grad_output - removed_component
        # Store telemetry
        OrthoGradientShield.last_proj_mag = float(proj.detach().abs().item())
        # Day 62: grad_proj_ratio = |removed| / |grad|
        grad_norm = float(grad_output.detach().norm().item()) + 1e-8
        removed_norm = float(removed_component.detach().norm().item())
        OrthoGradientShield.last_grad_proj_ratio = removed_norm / grad_norm
        return grad_clean, None  # no grad for k_cent


OrthoGradientShield.last_proj_mag = 0.0
OrthoGradientShield.last_grad_proj_ratio = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Day 64 — AdaptiveSoftGradientShield (autograd.Function)
# ═══════════════════════════════════════════════════════════════════════

class AdaptiveSoftGradientShield(torch.autograd.Function):
    """Remove a FRACTION (lam) of the K-collinear gradient component.

    Forward: identity (z passes through unchanged).
    Backward: g_new = g - lam * proj_K(g)
        where proj_K(g) = (g·Kc).sum() / (Kc²).sum() * Kc

    Batch var_K safety skip: if var_K_batch / (var_K_probe + eps) < 0.05,
    gradient passes through unchanged.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, k_cent: torch.Tensor,
                lam: float, var_K_probe: float) -> torch.Tensor:
        ctx.save_for_backward(k_cent)
        ctx.lam = lam
        ctx.var_K_probe = var_K_probe
        return z

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        k_cent, = ctx.saved_tensors
        lam = ctx.lam
        var_K_probe = ctx.var_K_probe

        if k_cent.ndim < grad_output.ndim:
            k_cent = k_cent.unsqueeze(-1)

        # Batch var_K safety skip
        var_K_batch = float((k_cent ** 2).mean().item())
        ratio = var_K_batch / (var_K_probe + 1e-8)
        if ratio < 0.05:
            AdaptiveSoftGradientShield.last_low_varK_skip = True
            AdaptiveSoftGradientShield.last_grad_proj_ratio = 0.0
            AdaptiveSoftGradientShield.last_proj_mag = 0.0
            AdaptiveSoftGradientShield.last_var_K_batch_ratio = ratio
            return grad_output, None, None, None

        # Scalar 1D batch projection
        k2_sum = (k_cent ** 2).sum().clamp(min=1e-8)
        proj_scalar = (grad_output * k_cent).sum() / k2_sum
        g_collinear = proj_scalar * k_cent
        grad_clean = grad_output - lam * g_collinear

        # Telemetry
        grad_norm = float(grad_output.detach().norm().item()) + 1e-8
        removed_norm = float((lam * g_collinear).detach().norm().item())
        AdaptiveSoftGradientShield.last_proj_mag = float(proj_scalar.detach().abs().item())
        AdaptiveSoftGradientShield.last_grad_proj_ratio = removed_norm / grad_norm
        AdaptiveSoftGradientShield.last_low_varK_skip = False
        AdaptiveSoftGradientShield.last_var_K_batch_ratio = ratio
        return grad_clean, None, None, None


AdaptiveSoftGradientShield.last_proj_mag = 0.0
AdaptiveSoftGradientShield.last_grad_proj_ratio = 0.0
AdaptiveSoftGradientShield.last_low_varK_skip = False
AdaptiveSoftGradientShield.last_var_K_batch_ratio = 0.0


class AdaptiveSoftShieldController:
    """Epoch-level hysteresis gate + batch-level soft shield for Day 64.

    Gate logic (compute at end of epoch E-1, apply during epoch E):
      - Warmup (epoch <= warmup_epochs): shield OFF unconditionally
      - ON  if G1_raw_probe > gate_on_thresh  (default 0.020)
      - OFF if G1_raw_probe < gate_off_thresh (default 0.010)
      - HOLD otherwise (keep previous state)
    """

    def __init__(self, lam: float = 0.50,
                 gate_on_thresh: float = 0.020,
                 gate_off_thresh: float = 0.010,
                 warmup_epochs: int = 5):
        assert 0.0 < lam <= 1.0, f"lam must be in (0, 1], got {lam}"
        assert gate_off_thresh < gate_on_thresh
        self.lam = lam
        self.gate_on_thresh = gate_on_thresh
        self.gate_off_thresh = gate_off_thresh
        self.warmup_epochs = warmup_epochs

        self.shield_state = 'OFF'
        self.var_K_probe = 0.0
        self._current_epoch = 0

        self._epoch_proj_ratios = []
        self._epoch_low_varK_skips = 0
        self._epoch_active_batches = 0
        self._epoch_total_batches = 0
        self.last_gate_info = {}
        self.last_epoch_telem = {}

    def update_gate(self, epoch: int, G1_raw_probe: float, var_K_probe: float):
        """Update shield gate state based on aligned epoch probe leakage."""
        self._current_epoch = epoch
        self.var_K_probe = var_K_probe
        prev_state = self.shield_state

        if epoch <= self.warmup_epochs:
            self.shield_state = 'OFF'
            reason = 'WARMUP_OFF'
        elif G1_raw_probe > self.gate_on_thresh:
            self.shield_state = 'ON'
            reason = 'ON' if prev_state != 'ON' else 'HOLD_ON'
        elif G1_raw_probe < self.gate_off_thresh:
            self.shield_state = 'OFF'
            reason = 'OFF' if prev_state != 'OFF' else 'HOLD_OFF'
        else:
            reason = f'HOLD_{self.shield_state}'

        self.last_gate_info = {
            'epoch': epoch,
            'G1_raw_probe': G1_raw_probe,
            'var_K_probe': var_K_probe,
            'shield_state_prev': prev_state,
            'shield_state_new': self.shield_state,
            'shield_gate_reason': reason,
            'shield_active_epoch': 1 if self.shield_state == 'ON' else 0,
            'lam': self.lam,
        }

        self._epoch_proj_ratios = []
        self._epoch_low_varK_skips = 0
        self._epoch_active_batches = 0
        self._epoch_total_batches = 0

        return self.last_gate_info

    def apply_to_tensor(self, z: torch.Tensor, K: torch.Tensor):
        """Apply adaptive soft shield. Forward: identity. Backward: partial proj.

        Returns (z_out, info_dict).
        """
        self._epoch_total_batches += 1

        K_float = K.float() if K.dtype != torch.float32 else K
        K_cent = K_float - K_float.mean()

        info = {
            'shield_state': self.shield_state,
            'lam': self.lam,
            'epoch': self._current_epoch,
            'grad_shield_on': False,
            'low_varK_skip': False,
            'grad_proj_ratio': 0.0,
            'var_K_batch_ratio': 0.0,
        }

        if self.shield_state != 'ON' or not z.requires_grad:
            return z, info

        z_out = AdaptiveSoftGradientShield.apply(
            z, K_cent, self.lam, self.var_K_probe)

        info['grad_shield_on'] = True
        info['low_varK_skip'] = AdaptiveSoftGradientShield.last_low_varK_skip
        info['grad_proj_ratio'] = AdaptiveSoftGradientShield.last_grad_proj_ratio
        info['var_K_batch_ratio'] = AdaptiveSoftGradientShield.last_var_K_batch_ratio

        self._epoch_active_batches += 1
        if info['low_varK_skip']:
            self._epoch_low_varK_skips += 1
        else:
            self._epoch_proj_ratios.append(info['grad_proj_ratio'])

        return z_out, info

    def get_epoch_telemetry(self):
        """Summarize per-epoch projection and gate stats."""
        n_proj = len(self._epoch_proj_ratios)
        proj_arr = np.array(self._epoch_proj_ratios) if n_proj > 0 else np.array([0.0])
        active = max(self._epoch_active_batches, 1)

        telem = {
            **self.last_gate_info,
            'total_batches': self._epoch_total_batches,
            'active_batches': self._epoch_active_batches,
            'low_varK_skip_rate': self._epoch_low_varK_skips / active
                                  if self._epoch_active_batches > 0 else 0.0,
            'active_batch_projection_rate': (
                (self._epoch_active_batches - self._epoch_low_varK_skips) / active
            ) if self._epoch_active_batches > 0 else 0.0,
            'grad_proj_ratio_median': float(np.median(proj_arr)) if n_proj > 0 else 0.0,
            'grad_proj_ratio_p90': float(np.percentile(proj_arr, 90)) if n_proj > 0 else 0.0,
        }
        self.last_epoch_telem = telem
        return telem


# ═══════════════════════════════════════════════════════════════════════
# Day 61 — ProbeSet-Synced Epoch β + Simulated-Post Sanity Check
# ═══════════════════════════════════════════════════════════════════════

G1_ORGANIC_CLEAN_THRESH = 0.015  # G1_raw_probe ≤ this → already clean
G1_SANITY_MARGIN = 1e-4          # VETO if G1_sim_post ≥ G1_raw - this


class ProbeSetSyncedKOrtho:
    """Day 61/62: β computed from the exact ProbeSet + honest_g1 ridge CV.

    Key differences from Day 60 (EpochRollingKOrtho):
      - β uses the SAME ProbeSet and SAME ridge CV as the G1 verdict
        → eliminates OrthoStatSet distribution mismatch
      - Simulated-post sanity check: G1_sim_post checked BEFORE apply
      - Three-way gate:
        (a) Organic clean: G1_raw ≤ 0.015 → β=0
        (b) VETO: G1_sim_post ≥ G1_raw - 1e-4 → β=0
        (c) Apply: β = clamp(beta_cand, [-2, 2])
      - Optional gradient shield to block pre-cancellation

    Day 62 additions:
      - mode='SHIELD_ONLY': no forward beta, only gradient shield
      - mode='SHIELD_AND_BETA': full beta correction + shield
      - Out-of-sample (OOS) A/B split beta sanity check
      - R2_probe_raw tracking
      - grad_proj_ratio telemetry

    Usage at epoch boundary:
      ortho.compute_from_probeset(model, probe_set, device)
    Usage per batch:
      z_out, info = ortho.apply(z, K, w, enable_grad_shield)
    """

    def __init__(self, D: int = 1,
                 mode: str = 'SHIELD_AND_BETA',
                 beta_hard_clamp: float = BETA_HARD_CLAMP,
                 organic_clean_thresh: float = G1_ORGANIC_CLEAN_THRESH,
                 sanity_margin: float = G1_SANITY_MARGIN,
                 beta_min: float = BETA_MIN_DEFAULT,
                 ecr_min: float = ECR_MIN_DEFAULT,
                 eps: float = 1e-6):
        assert mode in ('SHIELD_ONLY', 'SHIELD_AND_BETA'), \
            f"Invalid mode: {mode}"
        self.D = D
        self.mode = mode
        self.beta_hard_clamp = beta_hard_clamp
        self.organic_clean_thresh = organic_clean_thresh
        self.sanity_margin = sanity_margin
        self.beta_min = beta_min
        self.ecr_min = ecr_min
        self.eps = eps

        # Epoch-rolling state (all Python floats)
        self.beta_E: float = 0.0
        self.beta_cand: float = 0.0
        self.muK_probe: float = 0.0
        self.G1_raw_probe: float = 0.0
        self.G1_sim_post: float = 0.0
        self.R2_probe_raw: float = 0.0
        self.R2_sim_post_oos: float = 0.0
        self.need_ortho: bool = False
        self.veto_reason: str = ''
        self._computed: bool = False

        # Telemetry
        self.last_telemetry: Dict = {}

    # ── Day 62: OOS A/B split beta sanity ───────────────────────────
    def _oos_beta_sanity(
        self,
        Z_raw: np.ndarray,  # (N, 1)
        K_probe: np.ndarray,  # (N,)
        probe_seed: int,
    ) -> Dict:
        """Out-of-sample A/B split beta sanity check.

        Prevents trivial in-sample OLS zero-correlation by computing
        beta on A, evaluating on B and vice versa.

        Returns dict with R2_sim_post_oos, beta_A, beta_B, veto.
        """
        from qec_noise_factory.ml.diagnostics.g1_probe import run_linear_probe_cv
        rng = np.random.RandomState(probe_seed + 77777)
        n = len(K_probe)
        perm = rng.permutation(n)
        half = n // 2
        idx_A, idx_B = perm[:half], perm[half:]

        def _ols_beta(Z_sub, K_sub):
            Z_f = Z_sub.ravel()
            muK = float(K_sub.mean())
            Kc = K_sub - muK
            Zc = Z_f - float(Z_f.mean())
            var_Kc = float((Kc ** 2).mean())
            cov_ZK = float((Zc * Kc).mean())
            return cov_ZK / (var_Kc + self.eps), muK

        # Compute beta on A, simulate on B
        beta_A, muK_A = _ols_beta(Z_raw[idx_A], K_probe[idx_A])
        Kc_B = (K_probe[idx_B] - muK_A).reshape(-1, 1)
        Z_sim_B = Z_raw[idx_B] - beta_A * Kc_B
        r2_sim_B = run_linear_probe_cv(
            Z_sim_B, K_probe[idx_B], cv=5, seed=probe_seed)['r2_score']

        # Compute beta on B, simulate on A
        beta_B, muK_B = _ols_beta(Z_raw[idx_B], K_probe[idx_B])
        Kc_A = (K_probe[idx_A] - muK_B).reshape(-1, 1)
        Z_sim_A = Z_raw[idx_A] - beta_B * Kc_A
        r2_sim_A = run_linear_probe_cv(
            Z_sim_A, K_probe[idx_A], cv=5, seed=probe_seed)['r2_score']

        R2_sim_post_oos = float(np.mean([r2_sim_A, r2_sim_B]))

        # R2_probe_raw for the full set
        R2_probe_raw_full = run_linear_probe_cv(
            Z_raw, K_probe, cv=5, seed=probe_seed)['r2_score']

        # Veto if OOS doesn't improve
        oos_margin = 1e-4
        oos_veto = R2_sim_post_oos >= R2_probe_raw_full - oos_margin

        return {
            'R2_sim_post_oos': R2_sim_post_oos,
            'R2_probe_raw': R2_probe_raw_full,
            'beta_A': float(beta_A),
            'beta_B': float(beta_B),
            'r2_sim_A': float(r2_sim_A),
            'r2_sim_B': float(r2_sim_B),
            'oos_veto': oos_veto,
        }

    def compute_from_probeset(
        self,
        model: torch.nn.Module,
        probe_set: dict,
        device: torch.device,
        probe_seed: int = None,
    ) -> Dict:
        """Recompute β from ProbeSet using honest_g1 ridge CV.

        1. Forward pass on ProbeSet → Z_probe_raw
        2. Compute G1_raw_probe via ridge CV
        3. Compute OLS beta_cand from (Z, K) covariance
        4. Simulate Z_sim = Z - beta_cand * Kc
        5. Compute G1_sim_post via ridge CV on Z_sim
        6. Day 62: OOS A/B split sanity check (SHIELD_AND_BETA only)
        7. Gate: organic clean / VETO / OOS_VETO / apply
        """
        from qec_noise_factory.ml.diagnostics.g1_probe import (
            build_probe_features, run_linear_probe_cv)

        if probe_seed is None:
            probe_seed = probe_set['probe_seed']

        was_training = model.training

        # Temporarily disable ALL ortho paths
        saved = {}
        for attr in ['_k_ortho_probeset_synced', '_k_ortho_epoch_rolling',
                      '_k_ortho_frozen', '_k_ortho_module', '_k_ortho_window']:
            saved[attr] = getattr(model, attr, None)
            setattr(model, attr, None)
        old_use = getattr(model, 'fg_use_k_ortho', False)
        model.fg_use_k_ortho = False

        # Step 1: Extract Z_probe_raw using exact same path as honest_g1
        model.eval()
        Z_probe_raw = build_probe_features(
            model,
            probe_set['det_feats'], probe_set['err_feats'],
            probe_set['ei_d2e'], probe_set['ei_e2d'],
            error_weights=probe_set.get('err_w'),
            observable_mask=probe_set.get('obs_mask'),
        )  # (N, 1) numpy float64

        # Restore model state
        model.fg_use_k_ortho = old_use
        for attr, val in saved.items():
            setattr(model, attr, val)
        if was_training:
            model.train()

        K_probe = np.asarray(probe_set['K'], dtype=np.float64).ravel()

        # Step 2: G1_raw_probe
        raw_result = run_linear_probe_cv(
            Z_probe_raw, K_probe, cv=5, seed=probe_seed)
        G1_raw_probe = raw_result['r2_score']  # mean + std
        R2_probe_raw = G1_raw_probe  # alias for Day 62 telemetry

        # Step 3: Compute OLS beta_cand
        Z_flat = Z_probe_raw.ravel()
        muK = float(K_probe.mean())
        muZ = float(Z_flat.mean())
        Kc = K_probe - muK
        Zc = Z_flat - muZ
        var_Kc = float((Kc ** 2).mean())
        cov_ZK = float((Zc * Kc).mean())
        beta_cand = cov_ZK / (var_Kc + self.eps)

        # SHIELD_ONLY mode: skip beta computation, just record probe state
        if self.mode == 'SHIELD_ONLY':
            self.beta_E = 0.0
            self.beta_cand = float(beta_cand)
            self.muK_probe = muK
            self.G1_raw_probe = float(G1_raw_probe)
            self.G1_sim_post = float(G1_raw_probe)  # no correction
            self.R2_probe_raw = float(R2_probe_raw)
            self.R2_sim_post_oos = 0.0
            self.need_ortho = R2_probe_raw > self.organic_clean_thresh
            self.veto_reason = '' if self.need_ortho else 'ORGANIC_CLEAN'
            self._computed = True

            self.last_telemetry = {
                'beta_E': 0.0,
                'beta_cand': self.beta_cand,
                'muK_probe': muK,
                'G1_raw_probe': self.G1_raw_probe,
                'G1_sim_post': self.G1_sim_post,
                'G1_improvement': 0.0,
                'R2_probe_raw': self.R2_probe_raw,
                'R2_sim_post_oos': 0.0,
                'need_ortho': self.need_ortho,
                'veto_reason': self.veto_reason,
                'var_Kc': var_Kc,
                'cov_ZK': cov_ZK,
                'probe_seed': probe_seed,
                'n_probe': len(K_probe),
                'mode': 'SHIELD_ONLY',
                'beta_gate_reason': 'SHIELD_ONLY_NO_BETA',
            }
            return self.last_telemetry

        # SHIELD_AND_BETA mode: full pipeline
        # Step 4: Simulate post
        Z_sim = Z_probe_raw - beta_cand * Kc.reshape(-1, 1)

        # Step 5: G1_sim_post
        sim_result = run_linear_probe_cv(
            Z_sim, K_probe, cv=5, seed=probe_seed)
        G1_sim_post = sim_result['r2_score']

        # Step 6: Day 62 OOS A/B split sanity check
        oos_result = self._oos_beta_sanity(Z_probe_raw, K_probe, probe_seed)
        R2_sim_post_oos = oos_result['R2_sim_post_oos']
        oos_veto = oos_result['oos_veto']

        # Step 7: Gate logic
        beta_gate_reason = 'BETA_ON'
        if G1_raw_probe <= self.organic_clean_thresh:
            beta_E = 0.0
            need_ortho = False
            veto_reason = 'ORGANIC_CLEAN'
            beta_gate_reason = 'CLEAN_GATE_OFF'
        elif oos_veto:
            beta_E = 0.0
            need_ortho = False
            veto_reason = 'BETA_VETO_OOS_SANITY'
            beta_gate_reason = 'BETA_VETO_OOS_SANITY'
        elif G1_sim_post >= G1_raw_probe - self.sanity_margin:
            beta_E = 0.0
            need_ortho = False
            veto_reason = 'VETO_NO_IMPROVEMENT'
            beta_gate_reason = 'VETO_NO_IMPROVEMENT'
        else:
            beta_E = max(-self.beta_hard_clamp,
                         min(self.beta_hard_clamp, beta_cand))
            need_ortho = True
            veto_reason = ''
            beta_gate_reason = 'BETA_ON'

        # Store all as Python floats
        self.beta_E = float(beta_E)
        self.beta_cand = float(beta_cand)
        self.muK_probe = muK
        self.G1_raw_probe = float(G1_raw_probe)
        self.G1_sim_post = float(G1_sim_post)
        self.R2_probe_raw = float(R2_probe_raw)
        self.R2_sim_post_oos = float(R2_sim_post_oos)
        self.need_ortho = need_ortho
        self.veto_reason = veto_reason
        self._computed = True

        self.last_telemetry = {
            'beta_E': self.beta_E,
            'beta_cand': self.beta_cand,
            'muK_probe': muK,
            'G1_raw_probe': self.G1_raw_probe,
            'G1_sim_post': self.G1_sim_post,
            'G1_improvement': float(G1_raw_probe - G1_sim_post),
            'R2_probe_raw': self.R2_probe_raw,
            'R2_sim_post_oos': self.R2_sim_post_oos,
            'need_ortho': need_ortho,
            'veto_reason': veto_reason,
            'var_Kc': var_Kc,
            'cov_ZK': cov_ZK,
            'probe_seed': probe_seed,
            'n_probe': len(K_probe),
            'mode': 'SHIELD_AND_BETA',
            'beta_gate_reason': beta_gate_reason,
            'oos_beta_A': oos_result['beta_A'],
            'oos_beta_B': oos_result['beta_B'],
            'oos_r2_sim_A': oos_result['r2_sim_A'],
            'oos_r2_sim_B': oos_result['r2_sim_B'],
            'oos_veto': oos_veto,
        }
        return self.last_telemetry

    def apply(self, z: torch.Tensor, K: torch.Tensor,
              w: float, enable_grad_shield: bool = False) -> tuple:
        """Apply ProbeSet-synced correction.

        Args:
            z: (B,) or (B, D) — Z_g1. NOT modified in-place.
            K: (B,) — syndrome count
            w: ramp schedule weight
            enable_grad_shield: wrap output in OrthoGradientShield

        Returns:
            (z_post, info_dict)

        Day 62 behavior:
          SHIELD_ONLY: forward value = z (identity), backward = K-proj removed
          SHIELD_AND_BETA: forward value = z - correction, backward = K-proj removed
        """
        z_2d = z if z.ndim >= 2 else z.unsqueeze(1)
        k_flat = K.detach().float().reshape(-1)

        # Build K_cent used by both beta correction and gradient shield
        k_cent = (k_flat - self.muK_probe).to(
            device=z.device, dtype=z.dtype).detach()

        # ── SHIELD_ONLY mode: no forward beta ───────────────────────
        if self.mode == 'SHIELD_ONLY':
            shield_gate_reason = 'WARMUP_OFF'
            shield_on = False

            if not self._computed:
                shield_gate_reason = 'NOT_COMPUTED'
            elif w == 0.0:
                shield_gate_reason = 'WARMUP_OFF'
            elif not self.need_ortho:
                shield_gate_reason = 'CLEAN_GATE_OFF'
            else:
                shield_gate_reason = 'ACTIVE_ON'

            z_out_2d = z_2d  # forward value is identity

            # Apply gradient shield if active
            if (shield_gate_reason == 'ACTIVE_ON'
                    and enable_grad_shield and z.requires_grad):
                k_cent_for_shield = k_cent.unsqueeze(1)  # (B, 1)
                z_out_2d = OrthoGradientShield.apply(
                    z_out_2d, k_cent_for_shield)
                shield_on = True

            z_out = z_out_2d if z.ndim >= 2 else z_out_2d.squeeze(1)
            info = {
                'beta_E': 0.0,
                'beta_cand': self.beta_cand,
                'muK_probe': self.muK_probe,
                'G1_raw_probe': self.G1_raw_probe,
                'G1_sim_post': self.G1_sim_post,
                'R2_probe_raw': self.R2_probe_raw,
                'need_ortho': self.need_ortho,
                'veto_reason': self.veto_reason,
                'NO_OP_K_ORTHO': False,  # shield-only: always valid
                'no_op_reason': '',
                'k_ortho_weight': w,
                'eff_corr_ratio': 0.0,  # no forward correction
                'probeset_synced_mode': True,
                'mode': 'SHIELD_ONLY',
                'grad_shield_on': shield_on,
                'shield_gate_reason': shield_gate_reason,
                'proj_mag': OrthoGradientShield.last_proj_mag if shield_on else 0.0,
                'grad_proj_ratio': OrthoGradientShield.last_grad_proj_ratio if shield_on else 0.0,
            }
            return z_out, info

        # ── SHIELD_AND_BETA mode: original behavior + Day 62 telemetry ──
        # NO_OP checks
        no_op_reason = ''
        beta_gate_reason = 'BETA_ON'
        if not self._computed:
            no_op_reason = 'NOT_COMPUTED'
            beta_gate_reason = 'NOT_COMPUTED'
        elif w == 0.0:
            no_op_reason = 'SCHEDULE_OFF'
            beta_gate_reason = 'WARMUP_OFF'
        elif not self.need_ortho:
            no_op_reason = self.veto_reason or 'GATE_OFF'
            beta_gate_reason = self.veto_reason or 'CLEAN_GATE_OFF'
        elif abs(self.beta_E) < self.beta_min:
            no_op_reason = 'BETA_ZERO'
            beta_gate_reason = 'BETA_ZERO'

        is_no_op = bool(no_op_reason)

        info = {
            'beta_E': self.beta_E,
            'beta_cand': self.beta_cand,
            'muK_probe': self.muK_probe,
            'G1_raw_probe': self.G1_raw_probe,
            'G1_sim_post': self.G1_sim_post,
            'R2_probe_raw': self.R2_probe_raw,
            'R2_sim_post_oos': self.R2_sim_post_oos,
            'need_ortho': self.need_ortho,
            'veto_reason': self.veto_reason,
            'NO_OP_K_ORTHO': is_no_op,
            'no_op_reason': no_op_reason,
            'k_ortho_weight': w,
            'eff_corr_ratio': 0.0,
            'probeset_synced_mode': True,
            'mode': 'SHIELD_AND_BETA',
            'beta_gate_reason': beta_gate_reason,
            'grad_shield_on': False,
            'shield_gate_reason': 'INACTIVE',
            'proj_mag': 0.0,
            'grad_proj_ratio': 0.0,
        }

        if is_no_op:
            # Even if beta is off, still apply shield if needed
            z_out_2d = z_2d
            if (enable_grad_shield and z.requires_grad
                    and self._computed and w > 0
                    and self.R2_probe_raw > self.organic_clean_thresh):
                k_cent_for_shield = k_cent.unsqueeze(1)
                z_out_2d = OrthoGradientShield.apply(
                    z_out_2d, k_cent_for_shield)
                info['grad_shield_on'] = True
                info['shield_gate_reason'] = 'ACTIVE_ON'
                info['proj_mag'] = OrthoGradientShield.last_proj_mag
                info['grad_proj_ratio'] = OrthoGradientShield.last_grad_proj_ratio
            z_out = z_out_2d if z.ndim >= 2 else z_out_2d.squeeze(1)
            return z_out, info

        # correction = (w * beta_E * K_cent).detach()
        correction = (w * self.beta_E * k_cent).detach()

        # Reshape for subtraction
        correction_2d = correction.unsqueeze(1)  # (B, 1)

        # Z_post = Z_g1 - correction (out-of-place)
        z_post_2d = z_2d - correction_2d

        # eff_corr_ratio
        corr_norm = float(correction.norm().item())
        z_norm_val = float(z_2d.detach().norm().item())
        ecr = corr_norm / (z_norm_val + self.eps)
        info['eff_corr_ratio'] = ecr

        if ecr < self.ecr_min:
            info['NO_OP_K_ORTHO'] = True
            info['no_op_reason'] = 'ECR_TOO_LOW'
            return z, info

        info['NO_OP_K_ORTHO'] = False

        # Apply gradient shield
        if enable_grad_shield and z.requires_grad:
            k_cent_for_shield = k_cent.unsqueeze(1)  # (B, 1)
            z_post_2d = OrthoGradientShield.apply(
                z_post_2d, k_cent_for_shield)
            info['grad_shield_on'] = True
            info['shield_gate_reason'] = 'ACTIVE_ON'
            info['proj_mag'] = OrthoGradientShield.last_proj_mag
            info['grad_proj_ratio'] = OrthoGradientShield.last_grad_proj_ratio

        # Restore shape
        z_post = z_post_2d if z.ndim >= 2 else z_post_2d.squeeze(1)
        return z_post, info


# ═══════════════════════════════════════════════════════════════════════
# Day 65: Split Residual Head + Nuisance Siphon
# ═══════════════════════════════════════════════════════════════════════

class SplitResidualHead(torch.nn.Module):
    """Day 65: Split a scalar residual into topology + nuisance channels.

    Takes z_norm [B, 1] (post-KCS scalar) and produces:
        z_topo [B, 1]: topology scalar — ONLY this goes to final logits
        z_aux  [B, 1]: nuisance siphon — absorbs K-related variation

    Total parameters added: 4 (Linear(1, 2) = 2 weights + 2 biases).
    No in-place ops. Both outputs carry gradients.
    """

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 2, bias=True)
        # Initialize: topo channel ≈ identity, aux ≈ zero
        with torch.no_grad():
            self.proj.weight[0, 0] = 1.0   # z_topo ≈ z_norm at init
            self.proj.weight[1, 0] = 0.0   # z_aux  ≈ 0 at init
            self.proj.bias[0] = 0.0
            self.proj.bias[1] = 0.0

    def forward(self, z_norm: torch.Tensor) -> tuple:
        """Split z_norm [B, 1] → (z_topo [B, 1], z_aux [B, 1])."""
        out = self.proj(z_norm)       # [B, 2]
        z_topo = out[:, 0:1]         # [B, 1] — slice, not index (preserves dim)
        z_aux = out[:, 1:2]          # [B, 1]
        return z_topo, z_aux


class NuisanceSiphonLoss(torch.nn.Module):
    """Day 65: Auxiliary losses for nuisance siphon (Arm C only).

    Two soft loss terms (no gradient projection / no shielding):
    1) Aux K-prediction loss: MSE(z_aux, K_normalized)
       → Encourages z_aux to absorb K-related variation.
    2) Topology decorrelation penalty: squared Pearson corr(z_topo, K)
       → Soft pressure for z_topo to be uncorrelated with K.

    Both losses are eps-safe and use no in-place operations.
    """

    def __init__(self, lambda_aux: float = 0.1, lambda_decor: float = 0.05):
        super().__init__()
        self.lambda_aux = lambda_aux
        self.lambda_decor = lambda_decor

    def forward(self, z_topo: torch.Tensor, z_aux: torch.Tensor,
                K: torch.Tensor) -> dict:
        """Compute auxiliary losses.

        Args:
            z_topo: [B, 1] topology scalar (from SplitResidualHead)
            z_aux:  [B, 1] nuisance scalar (from SplitResidualHead)
            K:      [B] or [B, 1] syndrome count (float)

        Returns:
            dict with: aux_loss, decor_loss (weighted),
                       aux_loss_raw, decor_loss_raw (unweighted, detached),
                       corr_z_topo_K, corr_z_aux_K (detached telemetry)
        """
        eps = 1e-6

        # Flatten to [B]
        z_t = z_topo.squeeze(-1)       # [B]
        z_a = z_aux.squeeze(-1)        # [B]
        K_flat = K.squeeze(-1).float() # [B]

        # --- 1) Aux K prediction loss: MSE(z_aux, K_normalized) ---
        K_mean = K_flat.mean()
        K_std = K_flat.std() + eps
        K_norm = (K_flat - K_mean) / K_std
        aux_loss_raw = torch.nn.functional.mse_loss(z_a, K_norm)

        # --- 2) Decorrelation penalty: squared Pearson corr(z_topo, K) ---
        z_t_centered = z_t - z_t.mean()
        K_centered = K_flat - K_mean
        cov = (z_t_centered * K_centered).mean()
        corr_topo_K = cov / (z_t_centered.std() * K_std + eps)
        decor_loss_raw = corr_topo_K ** 2

        # --- Telemetry: corr(z_aux, K) ---
        z_a_centered = z_a - z_a.mean()
        corr_aux_K = (z_a_centered * K_centered).mean() / (
            z_a_centered.std() * K_std + eps)

        return {
            'aux_loss': self.lambda_aux * aux_loss_raw,
            'decor_loss': self.lambda_decor * decor_loss_raw,
            'aux_loss_raw': float(aux_loss_raw.detach()),
            'decor_loss_raw': float(decor_loss_raw.detach()),
            'corr_z_topo_K': float(corr_topo_K.detach()),
            'corr_z_aux_K': float(corr_aux_K.detach()),
        }


# ═══════════════════════════════════════════════════════════════════════
# Day 66 — Decorrelation-Only Regularization (forward-pass only)
# ═══════════════════════════════════════════════════════════════════════


class DecorrelationOnlyLoss(torch.nn.Module):
    """Day 66: Squared Pearson correlation penalty on Z_g1 vs K.

    Forward-pass regularizer only — no gradient surgery, no split head.
    Applied directly to the same aligned Z_g1 scalar used in final logits.

    Loss = lambda_decorr * corr(Z_g1, K)^2

    Safety:
        - VarK skip (absolute + ratio to probe reference)
        - Eps-safe denominators
        - Out-of-place operations only
    """

    def __init__(
        self,
        lambda_decorr: float = 0.05,
        varK_abs_eps: float = 1e-6,
        varK_ratio_min: float = 0.05,
    ):
        super().__init__()
        self.lambda_decorr = lambda_decorr
        self.varK_abs_eps = varK_abs_eps
        self.varK_ratio_min = varK_ratio_min

    def forward(
        self,
        z_g1: torch.Tensor,
        K: torch.Tensor,
        var_K_probe: float,
    ) -> dict:
        """Compute decorrelation penalty.

        Args:
            z_g1: [B, 1] aligned residual scalar (same tensor as logit_residual_norm).
            K: [B] or [B, 1] syndrome counts.
            var_K_probe: Probe-level var(K) for ratio-based safety skip.

        Returns:
            dict with: loss (scaled), corr2 (raw), varK_batch, varZ_batch,
                       skipped (bool), and all telemetry.
        """
        eps = 1e-8

        Z = z_g1.view(-1).float()
        K_flat = K.view(-1).float()

        varK_batch = float(K_flat.var().detach())
        varZ_batch = float(Z.var().detach())

        # --- VarK safety skip ---
        varK_ratio = varK_batch / (var_K_probe + 1e-12)
        skip = (varK_batch < self.varK_abs_eps) or (varK_ratio < self.varK_ratio_min)

        if skip:
            zero = torch.tensor(0.0, device=z_g1.device, requires_grad=False)
            return {
                'loss': zero,
                'corr2': 0.0,
                'varK_batch': varK_batch,
                'varZ_batch': varZ_batch,
                'varK_ratio': varK_ratio,
                'skipped': True,
            }

        # --- Squared Pearson correlation ---
        Z_centered = Z - Z.mean()
        K_centered = K_flat - K_flat.mean()

        cov = (Z_centered * K_centered).mean()
        var_Z = (Z_centered ** 2).mean() + eps
        var_K = (K_centered ** 2).mean() + eps

        corr2 = (cov ** 2) / (var_Z * var_K)

        loss = self.lambda_decorr * corr2

        return {
            'loss': loss,
            'corr2': float(corr2.detach()),
            'varK_batch': varK_batch,
            'varZ_batch': varZ_batch,
            'varK_ratio': varK_ratio,
            'skipped': False,
        }


class HysteresisDecorGate:
    """Day 66: Epoch-level hysteresis gate for adaptive decorrelation.

    State machine:
        - Warmup (epoch <= warmup_epochs): always OFF
        - Active: ON if G1_probe > on_thresh, OFF if G1_probe < off_thresh, HOLD otherwise
        - State persists across epochs (no flicker)
    """

    def __init__(
        self,
        on_thresh: float = 0.025,
        off_thresh: float = 0.015,
        warmup_epochs: int = 5,
    ):
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh
        self.warmup_epochs = warmup_epochs
        self.state = 'OFF'
        self._history: list = []

    def update(self, epoch: int, g1_probe: float) -> dict:
        """Update gate state for this epoch.

        Returns:
            dict with: active (bool), state (str), transition (str),
                       epoch, g1_probe.
        """
        prev_state = self.state

        if epoch <= self.warmup_epochs:
            self.state = 'OFF'
            transition = 'WARMUP'
        elif g1_probe > self.on_thresh:
            self.state = 'ON'
            transition = 'ON' if prev_state != 'ON' else 'HOLD_ON'
        elif g1_probe < self.off_thresh:
            self.state = 'OFF'
            transition = 'OFF' if prev_state != 'OFF' else 'HOLD_OFF'
        else:
            # Hysteresis: keep previous state
            transition = f'HOLD_{self.state}'

        record = {
            'epoch': epoch,
            'g1_probe': g1_probe,
            'state': self.state,
            'prev_state': prev_state,
            'transition': transition,
            'active': self.state == 'ON',
        }
        self._history.append(record)
        return record

    @property
    def active(self) -> bool:
        return self.state == 'ON'

    @property
    def history(self) -> list:
        return list(self._history)


# ═══════════════════════════════════════════════════════════════════════
# Day 67 — Iso-K Local Ranking Loss (forward-pass auxiliary only)
# ═══════════════════════════════════════════════════════════════════════


class IsoKRankingLoss(torch.nn.Module):
    """Day 67/68: Hinge margin ranking within iso-K neighborhoods.

    Mines (pos, neg) pairs where |K_pos - K_neg| <= delta_k, then applies:
        L_iso = lambda_iso * mean( relu(margin - (z_pos - z_neg)) )

    Day 68 addition: `use_safe_std` toggle.
      - True (SafeStd, default): Z is batch-standardized with detached scale
        for stable margin tuning. Prevents scale-gaming.
      - False (Base): hinge operates on raw Z_g1 directly.

    Forward-pass auxiliary only — no gradient surgery.
    """

    def __init__(
        self,
        lambda_iso: float = 0.10,
        margin: float = 0.50,
        delta_k: int = 1,
        max_pairs: int = 128,
        use_safe_std: bool = True,
    ):
        super().__init__()
        self.lambda_iso = lambda_iso
        self.margin = margin
        self.delta_k = delta_k
        self.max_pairs = max_pairs
        self.use_safe_std = use_safe_std

    def forward(
        self,
        z_g1: torch.Tensor,
        K: torch.Tensor,
        Y: torch.Tensor,
    ) -> dict:
        """Compute iso-K ranking loss.

        Args:
            z_g1: [B, 1] aligned residual scalar.
            K: [B] or [B, 1] syndrome counts.
            Y: [B] or [B, 1] binary labels.

        Returns:
            dict with: loss, pair_count_total, pair_count_used, no_pair,
                       violation_rate, zgap_mean, zgap_median,
                       z_var_global, z_var_intra_k, unique_bins_with_pairs.
        """
        eps = 1e-6
        device = z_g1.device

        Z = z_g1.view(-1).float()
        K_flat = K.view(-1).float()
        Y_flat = Y.view(-1).float()

        # --- Scale-gaming telemetry (always computed) ---
        with torch.no_grad():
            z_var_global = float(Z.var().detach())
            # Intra-K variance: mean of per-K-bin variances
            unique_K = torch.unique(K_flat)
            intra_vars = []
            for kv in unique_K:
                mask_k = K_flat == kv
                if mask_k.sum() >= 2:
                    intra_vars.append(float(Z[mask_k].var().detach()))
            z_var_intra_k = float(np.mean(intra_vars)) if intra_vars else 0.0

        # --- Optionally batch-standardize Z with detached scale ---
        if self.use_safe_std:
            z_mean = Z.mean()
            z_std_scale = Z.std(unbiased=False).detach() + eps
            Z_for_loss = (Z - z_mean) / z_std_scale
        else:
            Z_for_loss = Z

        # --- Find positive and negative indices ---
        pos_mask = Y_flat > 0.5
        neg_mask = Y_flat <= 0.5
        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0]

        n_pos = pos_idx.shape[0]
        n_neg = neg_idx.shape[0]

        # No-pair safety
        _zero_result = {
            'loss': torch.tensor(0.0, device=device, requires_grad=False),
            'pair_count_total': 0,
            'pair_count_used': 0,
            'no_pair': True,
            'violation_rate': 0.0,
            'zgap_mean': 0.0,
            'zgap_median': 0.0,
            'z_var_global': z_var_global,
            'z_var_intra_k': z_var_intra_k,
            'unique_bins_with_pairs': 0,
        }
        if n_pos < 1 or n_neg < 1:
            return _zero_result

        # --- Mine pairs with |ΔK| <= delta_k ---
        K_pos = K_flat[pos_idx]  # [n_pos]
        K_neg = K_flat[neg_idx]  # [n_neg]

        # Compute pairwise K distance: [n_pos, n_neg]
        delta_K = (K_pos.unsqueeze(1) - K_neg.unsqueeze(0)).abs()
        valid_mask = delta_K <= self.delta_k  # [n_pos, n_neg]

        # Get valid pair indices
        valid_pairs = torch.where(valid_mask)
        pair_count_total = valid_pairs[0].shape[0]

        if pair_count_total == 0:
            return _zero_result

        # --- Count unique K-bins that contributed pairs ---
        with torch.no_grad():
            paired_K_pos = K_pos[valid_pairs[0]]
            paired_K_neg = K_neg[valid_pairs[1]]
            all_paired_K = torch.cat([paired_K_pos, paired_K_neg])
            unique_bins_with_pairs = int(torch.unique(all_paired_K).shape[0])

        # --- Cap pairs (uniform random sampling) ---
        if pair_count_total > self.max_pairs:
            perm = torch.randperm(pair_count_total, device=device)[:self.max_pairs]
            pi = valid_pairs[0][perm]
            ni = valid_pairs[1][perm]
            pair_count_used = self.max_pairs
        else:
            pi = valid_pairs[0]
            ni = valid_pairs[1]
            pair_count_used = pair_count_total

        # Map back to batch indices
        batch_pos_idx = pos_idx[pi]
        batch_neg_idx = neg_idx[ni]

        # --- Hinge margin loss ---
        z_pos = Z_for_loss[batch_pos_idx]
        z_neg = Z_for_loss[batch_neg_idx]
        gaps = z_pos - z_neg  # want this > margin

        violations = torch.relu(self.margin - gaps)
        loss = self.lambda_iso * violations.mean()

        # --- Telemetry ---
        with torch.no_grad():
            violation_rate = float((gaps < self.margin).float().mean())
            zgap_mean = float(gaps.mean())
            zgap_median = float(gaps.median())

        return {
            'loss': loss,
            'pair_count_total': pair_count_total,
            'pair_count_used': pair_count_used,
            'no_pair': False,
            'violation_rate': violation_rate,
            'zgap_mean': zgap_mean,
            'zgap_median': zgap_median,
            'z_var_global': z_var_global,
            'z_var_intra_k': z_var_intra_k,
            'unique_bins_with_pairs': unique_bins_with_pairs,
        }
