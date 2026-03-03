"""
Factor-Graph Decoder v0 — Day 32

Bipartite message-passing decoder for QEC using DEM factor graphs.
Preserves k>2 hyperedge structure (no clique expansion loss).

Architecture:
    Detector nodes (syndrome bits) ↔ Error nodes (static weights)
    Bipartite MP: D→E aggregation + E→D aggregation (2-4 layers)
    Readout: pool error nodes with observable_mask=1 → MLP → logit

No dependency on torch_geometric — manual scatter-based message passing.
"""
from __future__ import annotations

import numpy as np

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BipartiteMPLayer(nn.Module):
    """Single bipartite message-passing layer (D→E then E→D).

    D→E: hE' = Norm(hE + dropout(ReLU(W_e_self·hE + W_d2e·AGG_d(hD) + b_e)))
    E→D: hD' = Norm(hD + dropout(ReLU(W_d_self·hD + W_e2d·AGG_e(hE) + b_d)))

    Aggregation uses weighted mean scatter (edge weights applied to D→E messages).
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        # D→E transform
        self.W_d2e = nn.Linear(dim, dim, bias=False)
        self.W_e_self = nn.Linear(dim, dim, bias=False)
        self.b_e = nn.Parameter(torch.zeros(dim))
        self.norm_e = nn.LayerNorm(dim)

        # E→D transform
        self.W_e2d = nn.Linear(dim, dim, bias=False)
        self.W_d_self = nn.Linear(dim, dim, bias=False)
        self.b_d = nn.Parameter(torch.zeros(dim))
        self.norm_d = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hD: torch.Tensor,             # (B, N_d, H)
        hE: torch.Tensor,             # (B, N_e, H)
        edge_d2e: torch.Tensor,        # (2, E) long — (det_idx, err_idx)
        edge_e2d: torch.Tensor,        # (2, E) long — (err_idx, det_idx)
        error_weights: Optional[torch.Tensor] = None,  # (N_e,) float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_d, H = hD.shape
        _, N_e, _ = hE.shape

        # --- D→E aggregation ---
        d_src, e_dst = edge_d2e[0], edge_d2e[1]  # (E,)
        hD_src = hD[:, d_src, :]                   # (B, E, H)
        msg_d2e = self.W_d2e(hD_src)               # (B, E, H)

        # Weight messages by error weight (sigmoid transform)
        if error_weights is not None:
            # error_weights are per-error-node, index by destination error
            w_per_edge = torch.sigmoid(error_weights[e_dst])  # (E,)
            msg_d2e = msg_d2e * w_per_edge.unsqueeze(0).unsqueeze(-1)

        # Scatter-mean to error nodes
        agg_d2e = torch.zeros(B, N_e, H, device=hD.device, dtype=hD.dtype)
        e_dst_exp = e_dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, H)
        agg_d2e.scatter_add_(1, e_dst_exp, msg_d2e)

        count_e = torch.zeros(B, N_e, 1, device=hD.device, dtype=hD.dtype)
        e_dst_c = e_dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        count_e.scatter_add_(1, e_dst_c, torch.ones_like(e_dst_c, dtype=hD.dtype))
        count_e = count_e.clamp(min=1.0)
        agg_d2e = agg_d2e / count_e

        # Update E
        hE_new = self.norm_e(
            hE + self.dropout(F.relu(self.W_e_self(hE) + agg_d2e + self.b_e))
        )

        # --- E→D aggregation ---
        e_src, d_dst = edge_e2d[0], edge_e2d[1]  # (E,)
        hE_src = hE_new[:, e_src, :]               # (B, E, H)
        msg_e2d = self.W_e2d(hE_src)               # (B, E, H)

        # Scatter-mean to detector nodes
        agg_e2d = torch.zeros(B, N_d, H, device=hE.device, dtype=hE.dtype)
        d_dst_exp = d_dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, H)
        agg_e2d.scatter_add_(1, d_dst_exp, msg_e2d)

        count_d = torch.zeros(B, N_d, 1, device=hE.device, dtype=hE.dtype)
        d_dst_c = d_dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        count_d.scatter_add_(1, d_dst_c, torch.ones_like(d_dst_c, dtype=hE.dtype))
        count_d = count_d.clamp(min=1.0)
        agg_e2d = agg_e2d / count_d

        # Update D
        hD_new = self.norm_d(
            hD + self.dropout(F.relu(self.W_d_self(hD) + agg_e2d + self.b_d))
        )

        return hD_new, hE_new


class FactorGraphDecoderV0(nn.Module):
    """Factor-Graph Decoder v0 — Day 32.

    Bipartite message-passing decoder that preserves k>2 DEM hyperedge
    structure.  Two disjoint node sets (Detectors, Errors) exchange messages
    via scatter-based aggregation.

    Architecture:
        D input projection (F_det → H)
        E input projection (F_err → H)
        K bipartite MP layers (D→E then E→D)
        Readout: pool error nodes where observable_mask=True
            concat(mean, max) → MLP head → logit

    Important:
        observable_mask is NOT used as a message-passing feature.
        It only controls which error nodes contribute to readout pooling.

    Args:
        det_input_dim:  feature dim per detector node (typically 1 = syndrome bit)
        err_input_dim:  feature dim per error node (typically 1 = weight)
        output_dim:     number of observables
        hidden_dim:     hidden dimension for MP layers
        num_mp_layers:  number of bipartite MP rounds
        readout:        "mean_max" | "mean"
        dropout:        dropout rate
    """
    needs_graph = True
    uses_edge_weight = True
    model_name = "factor_graph_v0"

    def __init__(
        self,
        det_input_dim: int = 1,
        err_input_dim: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 32,
        num_mp_layers: int = 3,
        readout: str = "mean_max",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.det_input_dim = det_input_dim
        self.err_input_dim = err_input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.readout_type = readout

        # Input projections
        self.det_proj = nn.Linear(det_input_dim, hidden_dim)
        self.err_proj = nn.Linear(err_input_dim, hidden_dim)

        # Bipartite MP layers
        self.mp_layers = nn.ModuleList()
        for _ in range(num_mp_layers):
            self.mp_layers.append(BipartiteMPLayer(hidden_dim, dropout))

        # Readout head
        if readout == "mean_max":
            head_in = hidden_dim * 2
        else:
            head_in = hidden_dim

        # Day 50: Reverted bias=True (Day 49.3 bias=False caused tanh saturation)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        det_features: torch.Tensor,         # (B, N_d, F_det)
        err_features: torch.Tensor,         # (B, N_e, F_err)  or (N_e, F_err) broadcast
        edge_index_d2e: torch.Tensor,       # (2, E) long
        edge_index_e2d: torch.Tensor,       # (2, E) long
        error_weights: Optional[torch.Tensor] = None,  # (N_e,) float
        observable_mask: Optional[torch.Tensor] = None, # (N_e,) bool
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            det_features:    (B, N_d, F_det) — syndrome bits + optional features
            err_features:    (B, N_e, F_err) or (N_e, F_err) — static error features
            edge_index_d2e:  (2, E) — detector→error edges
            edge_index_e2d:  (2, E) — error→detector edges
            error_weights:   (N_e,) — matching weights for edge weighting
            observable_mask: (N_e,) bool — which errors affect observable (readout only)

        Returns:
            logits: (B, output_dim)
        """
        B = det_features.shape[0]

        # Broadcast error features if needed: (N_e, F) → (B, N_e, F)
        if err_features.dim() == 2:
            err_features = err_features.unsqueeze(0).expand(B, -1, -1)

        # Input projection
        hD = F.relu(self.det_proj(det_features))  # (B, N_d, H)
        hE = F.relu(self.err_proj(err_features))  # (B, N_e, H)

        # Message passing
        for mp in self.mp_layers:
            hD, hE = mp(hD, hE, edge_index_d2e, edge_index_e2d, error_weights)

        # Readout: pool ONLY error nodes with observable_mask=True
        if observable_mask is not None and observable_mask.any():
            hE_obs = hE[:, observable_mask, :]  # (B, N_obs, H)
        else:
            # Fallback: pool all error nodes
            hE_obs = hE

        if self.readout_type == "mean_max":
            graph_emb = torch.cat([
                hE_obs.mean(dim=1),
                hE_obs.max(dim=1).values,
            ], dim=-1)  # (B, 2H)
        else:
            graph_emb = hE_obs.mean(dim=1)  # (B, H)

        return self.head(graph_emb)  # (B, output_dim)


class FocalLoss(nn.Module):
    """Focal Loss for binary classification — Day 33.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces loss for well-classified examples, focusing on hard negatives.
    Particularly effective for reducing false positives in imbalanced QEC data.

    Args:
        gamma: focusing parameter (default 2.0 — standard focal loss)
        pos_weight: weight for positive class (like BCE pos_weight)
    """

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw logits
            targets: (B, C) binary targets
        """
        # BCE without reduction
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], dtype=logits.dtype, device=logits.device)
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pw, reduction="none"
            )
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )

        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        return (focal_weight * bce).mean()


class ParityChannel(nn.Module):
    """Local parity aggregation — Day 35.

    Computes a parity-like signal via product-reduce at E->D edges:
        llr = llr_proj(hE)                         # (B, N_e, 1)
        pt  = -tanh(llr / 2)  clamped to [-0.999, 0.999]
        parity_agg = scatter_prod(pt, det_idx)     # (B, N_d, 1)

    Product-reduce is implemented in log-space for numerical stability:
        sign = sign(pt), abs_pt = |pt|
        log_abs = scatter_add(log(abs_pt), det_idx)
        sign_agg = scatter_prod(sign, det_idx)  -- via count-of-negatives mod 2
        parity_agg = sign_agg * exp(log_abs)

    The parity signal is mixed into detector embeddings, then scattered
    back to error nodes via D->E edges so it reaches the readout.
    """

    def __init__(self, hidden_dim: int, alpha: float = 0.1):
        super().__init__()
        self.llr_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.mix_mlp_d = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # D->E scatter-back: project parity-enriched detector info to error nodes
        self.scatter_back = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.alpha = alpha

    def forward(
        self,
        hD: torch.Tensor,          # (B, N_d, H)
        hE: torch.Tensor,          # (B, N_e, H)
        edge_e2d: torch.Tensor,    # (2, E) long -- (err_idx, det_idx)
        edge_d2e: torch.Tensor,    # (2, E) long -- (det_idx, err_idx)
    ) -> tuple:
        """Compute parity aggregation and propagate to both D and E nodes.

        Returns:
            (hD_updated, hE_updated) -- both with parity mixed in
        """
        B, N_d, H = hD.shape
        _, N_e, _ = hE.shape

        # LLR projection from error node embeddings
        llr = self.llr_proj(hE)                    # (B, N_e, 1)

        # Parity term: -tanh(llr/2), clamped for stability
        pt = -torch.tanh(llr / 2.0)
        pt = pt.clamp(-0.999, 0.999)               # (B, N_e, 1)

        # Product-reduce via log-space for numerical stability
        e_src, d_dst = edge_e2d[0], edge_e2d[1]    # (E,)
        pt_edges = pt[:, e_src, :]                  # (B, E, 1)

        # Decompose into sign and log-magnitude
        sign_edges = torch.sign(pt_edges)           # (B, E, 1)
        abs_edges = pt_edges.abs().clamp(min=1e-8)  # avoid log(0)
        log_abs = torch.log(abs_edges)              # (B, E, 1)

        # Scatter-add log magnitudes to detector nodes
        d_dst_exp = d_dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        log_agg = torch.zeros(B, N_d, 1, device=hD.device, dtype=hD.dtype)
        log_agg.scatter_add_(1, d_dst_exp, log_abs)

        # Sign aggregation: count negatives per detector, parity = (-1)^count
        is_neg = (sign_edges < 0).float()           # (B, E, 1)
        neg_count = torch.zeros(B, N_d, 1, device=hD.device, dtype=hD.dtype)
        neg_count.scatter_add_(1, d_dst_exp, is_neg)
        sign_agg = 1.0 - 2.0 * (neg_count % 2)     # (-1)^count

        # Reconstruct product
        parity_agg = sign_agg * torch.exp(log_agg.clamp(max=20.0))  # (B, N_d, 1)

        # NaN guard
        parity_agg = torch.where(
            torch.isfinite(parity_agg), parity_agg,
            torch.zeros_like(parity_agg)
        )

        # Mix into detector embedding
        mixed_d = self.mix_mlp_d(torch.cat([hD, parity_agg], dim=-1))  # (B, N_d, H)
        hD_new = hD + self.alpha * mixed_d

        # Scatter parity-enriched detector info back to error nodes (D->E)
        d_src_de, e_dst_de = edge_d2e[0], edge_d2e[1]  # (E,)
        hD_msg = self.scatter_back(hD_new[:, d_src_de, :])  # (B, E, H)
        e_dst_exp = e_dst_de.unsqueeze(0).unsqueeze(-1).expand(B, -1, H)
        agg_d2e = torch.zeros(B, N_e, H, device=hE.device, dtype=hE.dtype)
        agg_d2e.scatter_add_(1, e_dst_exp, hD_msg)
        count_e = torch.zeros(B, N_e, 1, device=hE.device, dtype=hE.dtype)
        e_c = e_dst_de.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        count_e.scatter_add_(1, e_c, torch.ones_like(e_c, dtype=hE.dtype))
        count_e = count_e.clamp(min=1.0)
        agg_d2e = agg_d2e / count_e
        hE_new = hE + self.alpha * agg_d2e

        return hD_new, hE_new


class FactorGraphDecoderV1(FactorGraphDecoderV0):
    """Factor-Graph Decoder v1 — Day 33 + Day 35 parity channel.

    Same bipartite MP architecture as v0 (3 layers, residual, LayerNorm).
    Improvements:
        - Configurable loss: 'bce' (weighted BCE) or 'focal' (gamma=2)
        - model_name = 'factor_graph_v1' for tracking
        - Higher default hidden_dim (48 vs 32)
        - Day 35: Optional local parity channel (use_parity_channel flag)

    The actual precision/FPR improvements come from:
        - Focal loss reducing loss for easy-to-classify examples
        - BRQL calibration (base-rate quantile lock threshold selection)
        - Stronger collapse guards in the benchmark harness
        - Day 35: Parity channel reduces density shortcut reliance
    """
    model_name = "factor_graph_v1"

    def __init__(
        self,
        det_input_dim: int = 1,
        err_input_dim: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 48,
        num_mp_layers: int = 3,
        readout: str = "mean_max",
        dropout: float = 0.1,
        loss_fn: str = "focal",
        focal_gamma: float = 2.0,
        use_parity_channel: bool = False,
        parity_alpha: float = 0.1,
        use_density_residualization: bool = False,
        fg_use_bp_checknode: bool = False,
        bp_beta: float = 0.2,
    ):
        super().__init__(
            det_input_dim=det_input_dim,
            err_input_dim=err_input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_mp_layers=num_mp_layers,
            readout=readout,
            dropout=dropout,
        )
        self.loss_fn_name = loss_fn
        self.focal_gamma = focal_gamma
        self.use_parity_channel = use_parity_channel
        self.use_density_residualization = use_density_residualization
        self.fg_use_bp_checknode = fg_use_bp_checknode

        # Day 40+41: Explicit prior+residual recombination flag
        self.use_density_prior_final = False  # set externally
        self.alpha_residual = 1.0  # Day 41: scaling for residual in recombined mode

        # Day 43: Null-space scrambler loss flags
        self.fg_use_scrambler_nullspace_loss = False
        self.fg_lambda_nullspace = 2.0

        # Day 43: K-binned alpha table
        self.fg_use_alpha_kbin = False
        self._alpha_kbin_table = None  # set via setup_alpha_kbin()

        # Day 44+45: K-Conditional Standardization + GRL adversary
        self.fg_use_kcs_standardization = False
        self.fg_use_grl_k_adversary = False
        self.fg_lambda_adv = 0.05  # Day 45: lowered default from 0.25
        self.fg_adv_warmup_epochs = 2  # Day 45: extended warmup
        self._kcs = None         # set via setup_kcs()
        self._k_adversary = None  # set via setup_adversary()
        self._current_epoch = 0   # tracked externally

        # Day 45: Iso-K topology preservation loss
        self.fg_use_iso_k_loss = False
        self.fg_lambda_iso = 0.05
        self.fg_iso_margin = 0.2

        # Day 46: Targeted leakage penalty (non-adversarial)
        self.fg_use_leakage_penalty = False
        self.fg_lambda_leak = 0.3

        # Day 47: Fix normalizer gaming
        self.fg_use_tanh_clamp = False
        self.fg_clamp_c = 5.0
        self.fg_stopgrad_kcs = False
        self.fg_kcs_center_only = False  # Arm C: mean-center only, no divide
        self.fg_use_envelope_penalty = False
        self.fg_lambda_env = 2.0

        # Day 49.1: Pre-clamp null-space + per-bin scrambled bias kill
        self.fg_use_preclamp_nullspace = False  # when True, null-space uses z_pre (no tanh)
        self.fg_lambda_scr_bias = 0.0           # per-bin scrambled mean^2 penalty weight

        # Day 49.2: Debiased null-space (learnable baseline subtraction)
        self.fg_use_baseline_debias = False
        self._scr_baseline = None  # nn.Embedding, set by setup_scr_baseline()
        self._scr_num_bins = 12

        # Day 49.3: Seed-robust G4 fix (kept for backward compat, disabled)
        self.fg_use_ema_centering = False
        self.fg_use_shielded_null = False
        self._ema_mu_scr = None       # buffer: [num_k_bins]
        self._ema_momentum = 0.9
        self._ema_num_bins = 12

        # Day 50: Baseline-centered null-space
        self.fg_use_scr_baseline_centering = False
        self.fg_scr_baseline_mode = "batch"  # "batch" or "bin"
        self.fg_sigma_floor = 0.1

        # Day 51: Sigma EMA (scale-only KCS)
        self.fg_use_sigma_ema = False
        self._sigma_ema = None       # buffer: [num_k_bins]
        self._sigma_ema_momentum = 0.9
        self._sigma_ema_num_bins = 12
        self._sigma_ema_min_count = 3

        # Day 53: Envelope independence penalty
        self.fg_use_env_penalty = False
        self.fg_lambda_env_max = 2.0
        self.fg_env_ramp_epochs = 4   # used by experiment schedule

        # Day 35: Local parity channel (optional)
        if use_parity_channel:
            self.parity = ParityChannel(hidden_dim, alpha=parity_alpha)

        # Day 39: BP check-node channel (optional)
        if fg_use_bp_checknode:
            from qec_noise_factory.ml.models.bp_checknode import BPCheckNode
            self.bp_checknode = BPCheckNode(hidden_dim, beta=bp_beta)

        # Day 37: Density residualization prior (frozen)
        # Set via set_density_prior() after construction
        self._density_prior_table = None
        self._density_prior_logits = None  # tensor buffer
        self._density_n_det = None

        # Day 37.1: Residual orthogonality guard
        self.use_residual_projection_removal = use_density_residualization  # auto-enable with residualization
        self.lambda_corr = 1.0  # correlation penalty weight
        self._last_residual_logit = None  # cached for penalty computation
        self._last_K_float = None  # cached for penalty computation

    def set_density_prior(self, prior_table: dict, n_det: int) -> None:
        """Store frozen K->logit prior for density residualization.

        Args:
            prior_table: output of build_k_prior_table.
            n_det: number of real detectors (excluding boundary).
        """
        self._density_prior_table = prior_table
        self._density_n_det = n_det

        # Build dense lookup tensor for fast batched access
        mapping = prior_table["mapping"]
        k_max = max(mapping.keys()) if mapping else 0
        # Extend to n_det in case test data has higher K
        size = max(k_max + 1, n_det + 1)
        lookup = torch.full((size,), prior_table["global_logit"], dtype=torch.float32)
        for k_val, logit in mapping.items():
            if k_val < size:
                lookup[k_val] = logit
        # Register as buffer (saved with model, no gradients)
        self.register_buffer("_prior_lookup", lookup)

    def compute_corr_penalty(self) -> torch.Tensor:
        """Compute correlation penalty between residual logit and K.

        Returns lambda_corr * corr(residual, K)^2. Call after forward().
        Returns 0 if no cached values available.
        """
        if self._last_residual_logit is None or self._last_K_float is None:
            return torch.tensor(0.0)

        r = self._last_residual_logit.detach().squeeze()
        k = self._last_K_float.detach().squeeze()

        if r.numel() < 3 or r.std() < 1e-8 or k.std() < 1e-8:
            return torch.tensor(0.0)

        # Pearson correlation (differentiable through residual)
        r_live = self._last_residual_logit.squeeze()  # keep gradient
        r_c = r_live - r_live.mean()
        k_c = k - k.mean()
        corr = (r_c * k_c).sum() / (
            (r_c * r_c).sum().sqrt() * (k_c * k_c).sum().sqrt() + 1e-8
        )
        return self.lambda_corr * corr * corr

    @staticmethod
    def _project_out_k(residual: torch.Tensor, K_float: torch.Tensor) -> torch.Tensor:
        """Remove linear K component from residual logits (batch-wise).

        residual: (B, O) logits
        K_float: (B,) syndrome counts as float

        Returns residual with K-linear component removed.
        """
        K_c = K_float - K_float.mean()  # (B,)
        K_c_2d = K_c.unsqueeze(-1)  # (B, 1)
        denom = (K_c * K_c).sum().clamp(min=1e-8)
        proj_coeff = (residual * K_c_2d).sum(dim=0) / denom  # (O,)
        return residual - K_c_2d * proj_coeff.unsqueeze(0)

    def _compute_backbone(self, det_features, err_features, edge_index_d2e,
                           edge_index_e2d, error_weights, observable_mask):
        """Day 49.3: MP layers + readout pooling → graph_emb (no head)."""
        B = det_features.shape[0]

        if err_features.dim() == 2:
            err_features = err_features.unsqueeze(0).expand(B, -1, -1)

        hD = F.relu(self.det_proj(det_features))
        hE = F.relu(self.err_proj(err_features))

        for mp in self.mp_layers:
            hD, hE = mp(hD, hE, edge_index_d2e, edge_index_e2d, error_weights)

        # Day 39: BP check-node messages
        if self.fg_use_bp_checknode:
            n_det_bp = det_features.shape[1] - 1
            syndrome = det_features[:, :n_det_bp, 0]
            if syndrome.shape[1] < hD.shape[1]:
                pad = torch.zeros(B, hD.shape[1] - syndrome.shape[1], device=hD.device)
                syndrome = torch.cat([syndrome, pad], dim=1)
            bp_msg = self.bp_checknode(hE, hD, syndrome, edge_index_e2d, edge_index_d2e)
            hE = hE + bp_msg

        if self.use_parity_channel:
            hD, hE = self.parity(hD, hE, edge_index_e2d, edge_index_d2e)

        if observable_mask is not None and observable_mask.any():
            hE_obs = hE[:, observable_mask, :]
        else:
            hE_obs = hE

        if self.readout_type == "mean_max":
            graph_emb = torch.cat([
                hE_obs.mean(dim=1),
                hE_obs.max(dim=1).values,
            ], dim=-1)
        else:
            graph_emb = hE_obs.mean(dim=1)

        return graph_emb

    def _compute_graph_embedding(self, det_features, err_features, edge_index_d2e,
                                   edge_index_e2d, error_weights, observable_mask):
        """Shared computation: backbone + head → residual_logit."""
        graph_emb = self._compute_backbone(
            det_features, err_features, edge_index_d2e,
            edge_index_e2d, error_weights, observable_mask)
        return self.head(graph_emb)

    def _apply_residualization(self, residual_logit, det_features, B):
        """Apply K-projection removal, cache, and return (residual, prior, K_float)."""
        n_det = self._density_n_det if self._density_n_det else det_features.shape[1] - 1
        K = det_features[:, :n_det, 0].sum(dim=1).long()
        K = K.clamp(0, self._prior_lookup.shape[0] - 1)
        K_float = K.float()

        # Day 37.1: Projection removal on residual ONLY
        if self.use_residual_projection_removal and B > 1:
            residual_logit = self._project_out_k(residual_logit, K_float)

        # Cache for correlation penalty
        self._last_residual_logit = residual_logit
        self._last_K_float = K_float

        prior_logit = self._prior_lookup[K].unsqueeze(-1).expand_as(residual_logit)
        return residual_logit, prior_logit.detach(), K_float

    def forward(
        self,
        det_features: torch.Tensor,
        err_features: torch.Tensor,
        edge_index_d2e: torch.Tensor,
        edge_index_e2d: torch.Tensor,
        error_weights: Optional[torch.Tensor] = None,
        observable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Day 41 fix:
          use_density_prior_final=OFF  → return logit_residual (no prior)
          use_density_prior_final=ON   → return logit_prior + alpha_residual * logit_residual
        """
        B = det_features.shape[0]
        residual_logit = self._compute_graph_embedding(
            det_features, err_features, edge_index_d2e, edge_index_e2d,
            error_weights, observable_mask)

        if self.use_density_residualization and hasattr(self, '_prior_lookup'):
            residual_logit, prior_logit, K_float = self._apply_residualization(
                residual_logit, det_features, B)
            # Day 49.2: subtract baseline (detached) on real branch
            z_for_kcs = residual_logit
            if self.fg_use_baseline_debias and self._scr_baseline is not None:
                b_real = self._get_baseline(K_float).detach()
                z_for_kcs = residual_logit - b_real
            # Day 49.3: tanh clamp → EMA centering → KCS
            z_clamped = self._apply_tanh_clamp(z_for_kcs)
            if self.fg_use_ema_centering and self._ema_mu_scr is not None:
                z_clamped = self._apply_ema_centering(z_clamped, K_float)
            z_for_recombo = self._apply_kcs_no_clamp(z_clamped, K_float)
            # Day 41: flag controls whether prior is added
            if self.use_density_prior_final:
                alpha = self._get_alpha(K_float)
                return prior_logit + alpha * z_for_recombo
            return z_for_recombo  # baseline: residual only

        return residual_logit

    def forward_split(
        self,
        det_features: torch.Tensor,
        err_features: torch.Tensor,
        edge_index_d2e: torch.Tensor,
        edge_index_e2d: torch.Tensor,
        error_weights: Optional[torch.Tensor] = None,
        observable_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Day 40: Forward returning split logits for separate metrics.

        Returns dict with 'logit_residual', 'logit_prior', 'logit_final', 'K',
        'z_centered' (Day 49.3).
        """
        B = det_features.shape[0]
        residual_logit = self._compute_graph_embedding(
            det_features, err_features, edge_index_d2e, edge_index_e2d,
            error_weights, observable_mask)

        if self.use_density_residualization and hasattr(self, '_prior_lookup'):
            residual_logit, prior_logit, K_float = self._apply_residualization(
                residual_logit, det_features, B)
            # Day 49.2: subtract baseline (detached) on real branch
            z_for_kcs = residual_logit
            if self.fg_use_baseline_debias and self._scr_baseline is not None:
                b_real = self._get_baseline(K_float).detach()
                z_for_kcs = residual_logit - b_real
            # Day 49.3: tanh clamp → EMA centering → KCS (no double-clamp)
            z_clamped = self._apply_tanh_clamp(z_for_kcs)
            z_centered = z_clamped
            if self.fg_use_ema_centering and self._ema_mu_scr is not None:
                z_centered = self._apply_ema_centering(z_clamped, K_float)
            z_norm = self._apply_kcs_no_clamp(z_centered, K_float)
            # Day 65: Split residual head (after KCS, before logit composition)
            _split_head = getattr(self, '_split_residual_head', None)
            if _split_head is not None:
                z_topo, z_aux = _split_head(z_norm)
            else:
                z_topo = z_norm
                z_aux = None
            # Day 56-61: K-orthogonalization on Z_g1 (exact G1 probe input)
            if getattr(self, 'fg_use_k_ortho', False):
                # E1: One-time alignment debug print
                if not getattr(self, '_ortho_alignment_logged', False):
                    print(f"[Day61-E1] G1 probe consumes: logit_residual_norm "
                          f"shape={z_norm.shape}")
                    print(f"[Day61-E1] K-ortho applies to: logit_residual_norm "
                          f"shape={z_norm.shape}")
                    self._ortho_alignment_logged = True

                # Day 64: AdaptiveSoftShieldController (checked first)
                adaptive_shield = getattr(self, '_adaptive_soft_shield', None)
                if adaptive_shield is not None:
                    z_norm, ortho_info = adaptive_shield.apply_to_tensor(
                        z_norm, K_float)
                    self._last_ortho_info = ortho_info
                # Day 61: ProbeSetSyncedKOrtho (checked second)
                elif (ps_ortho := getattr(self, '_k_ortho_probeset_synced', None)) is not None:
                    from qec_noise_factory.ml.models.k_ortho import get_ramp_weight
                    epoch = getattr(self, '_current_epoch', 1)
                    w = get_ramp_weight(epoch)
                    enable_shield = getattr(
                        self, '_enable_grad_shield', False)
                    if w > 0:
                        z_norm, ortho_info = ps_ortho.apply(
                            z_norm, K_float, w,
                            enable_grad_shield=enable_shield)
                    else:
                        ortho_info = {
                            'NO_OP_K_ORTHO': True,
                            'no_op_reason': 'SCHEDULE_OFF',
                            'k_ortho_weight': w,
                            'probeset_synced_mode': True,
                            'beta_E': ps_ortho.beta_E,
                            'beta_cand': ps_ortho.beta_cand,
                            'G1_raw_probe': ps_ortho.G1_raw_probe,
                            'G1_sim_post': ps_ortho.G1_sim_post,
                            'need_ortho': ps_ortho.need_ortho,
                            'veto_reason': ps_ortho.veto_reason,
                            'eff_corr_ratio': 0.0,
                            'grad_shield_on': False,
                            'proj_mag': 0.0,
                        }
                    self._last_ortho_info = ortho_info
                # Day 60: EpochRollingKOrtho (legacy)
                elif getattr(self, '_k_ortho_epoch_rolling', None) is not None:
                    epoch_rolling_ortho = self._k_ortho_epoch_rolling
                    from qec_noise_factory.ml.models.k_ortho import get_ramp_weight
                    epoch = getattr(self, '_current_epoch', 1)
                    w = get_ramp_weight(epoch)
                    ortho_active = getattr(self, '_ortho_active', True)
                    if ortho_active and w > 0:
                        z_norm, ortho_info = epoch_rolling_ortho.apply(
                            z_norm, K_float, w)
                    else:
                        ortho_info = {
                            'NO_OP_K_ORTHO': True,
                            'no_op_reason': ('SCHEDULE_OFF' if w == 0
                                              else 'ORTHO_INACTIVE'),
                            'k_ortho_weight': w,
                            'epoch_rolling_mode': True,
                            'beta_E': epoch_rolling_ortho.beta_E,
                            'beta_raw': epoch_rolling_ortho.beta_raw,
                            'R2_probe': epoch_rolling_ortho.R2_probe,
                            'need_ortho': epoch_rolling_ortho.need_ortho,
                            'var_k0': epoch_rolling_ortho.var_k,
                            'mean_k': epoch_rolling_ortho.mu_k,
                            'eff_corr_ratio': 0.0,
                        }
                    self._last_ortho_info = ortho_info
                # Day 59: FrozenBetaKOrtho (legacy)
                elif getattr(self, '_k_ortho_frozen', None) is not None:
                    frozen_ortho = self._k_ortho_frozen
                    from qec_noise_factory.ml.models.k_ortho import get_ramp_weight
                    epoch = getattr(self, '_current_epoch', 1)
                    w = get_ramp_weight(epoch)
                    ortho_active = getattr(self, '_ortho_active', True)
                    if ortho_active and w > 0:
                        z_norm, ortho_info = frozen_ortho.apply(
                            z_norm, K_float, w)
                    else:
                        ortho_info = {
                            'NO_OP_K_ORTHO': True,
                            'no_op_reason': ('SCHEDULE_OFF' if w == 0
                                              else 'ORTHO_INACTIVE'),
                            'k_ortho_weight': w,
                            'frozen_beta_mode': True,
                            'beta_cap': frozen_ortho.beta_cap,
                            'beta_raw': frozen_ortho.beta_raw,
                            'beta_norm': sum(abs(b) for b in frozen_ortho.beta_cap)
                                         / max(len(frozen_ortho.beta_cap), 1),
                            'var_k0': frozen_ortho.var_k,
                            'mean_k': frozen_ortho.mu_k,
                            'eff_corr_ratio': 0.0,
                        }
                    self._last_ortho_info = ortho_info
                else:
                    # Day 58/57 legacy paths
                    ortho_mod = getattr(self, '_k_ortho_module', None)
                    ortho_win = getattr(self, '_k_ortho_window', None)
                    if ortho_mod is not None:
                        z_norm, ortho_info = ortho_mod(
                            z_norm, K_float,
                            epoch=getattr(self, '_current_epoch', 1),
                            training=self.training,
                            frozen_beta=getattr(self, '_frozen_beta', None))
                        self._last_ortho_info = ortho_info
                    elif ortho_win is not None:
                        from qec_noise_factory.ml.models.k_ortho import k_orthogonalize_v2
                        z_norm, ortho_info = k_orthogonalize_v2(
                            z_norm, K_float, ortho_win,
                            epoch=getattr(self, '_current_epoch', 1),
                            training=self.training,
                            frozen_beta=getattr(self, '_frozen_beta', None))
                        self._last_ortho_info = ortho_info
            alpha = self._get_alpha(K_float)
            return {
                'logit_residual': residual_logit,  # Z_pre (pre-clamp)
                'z_pre': residual_logit,            # Day 49.1: alias
                'z_deb': z_for_kcs,                 # Day 49.2: debiased residual
                'z_centered': z_centered,           # Day 49.3: EMA centered
                'logit_residual_norm': z_topo,      # Day 65: topo scalar (G1 probe input)
                'z_aux': z_aux,                     # Day 65: nuisance channel (None for control)
                'logit_prior': prior_logit,
                'logit_final': prior_logit + alpha * z_topo,
                'K': K_float,
                'alpha': alpha,
            }

        # No prior available — residual IS final
        return {
            'logit_residual': residual_logit,
            'z_pre': residual_logit,
            'z_deb': residual_logit,
            'z_centered': residual_logit,
            'logit_residual_norm': residual_logit,
            'logit_prior': torch.zeros_like(residual_logit),
            'logit_final': residual_logit,
            'K': None,
            'alpha': None,
        }

    # ── Day 43 helpers ──────────────────────────────────────────────────

    def _get_alpha(self, K_float: torch.Tensor) -> torch.Tensor:
        """Return alpha value(s). Uses K-binned table if enabled, else scalar."""
        if self.fg_use_alpha_kbin and self._alpha_kbin_table is not None:
            # Returns (B,) -> unsqueeze to (B, 1) for broadcasting
            return self._alpha_kbin_table(K_float).unsqueeze(-1)
        return self.alpha_residual

    def setup_alpha_kbin(self, K_train: np.ndarray, num_bins: int = 12,
                         lambda_tv: float = 0.05, alpha_init: float = 0.0):
        """Initialize K-binned alpha table from training K values."""
        from qec_noise_factory.ml.models.alpha_k_bin import AlphaKBinTable
        self._alpha_kbin_table = AlphaKBinTable(
            num_bins=num_bins, alpha_init=alpha_init, lambda_tv=lambda_tv)
        self._alpha_kbin_table.fit_bins(K_train)

    def compute_nullspace_loss(
        self,
        det_features_scrambled: torch.Tensor,
        err_features: torch.Tensor,
        edge_index_d2e: torch.Tensor,
        edge_index_e2d: torch.Tensor,
        error_weights: Optional[torch.Tensor] = None,
        observable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Day 43/48: Null-space loss on scrambled twin batch.

        Day 48 lesson: targets Z_raw_scr (with tanh clamp only, NOT KCS).
        Applying KCS(real stats) to scrambled creates systematic -1.0 offset
        because KCS subtracts real mu from near-zero scrambled residual.

        L_null = mean(Z_raw_clamped_scr^2)
        """
        if not self.fg_use_scrambler_nullspace_loss:
            return torch.tensor(0.0)

        B = det_features_scrambled.shape[0]
        residual_scr = self._compute_graph_embedding(
            det_features_scrambled, err_features,
            edge_index_d2e, edge_index_e2d,
            error_weights, observable_mask)

        if self.use_density_residualization and hasattr(self, '_prior_lookup'):
            residual_scr, _, _ = self._apply_residualization(
                residual_scr, det_features_scrambled, B)

        # Day 48: apply tanh clamp only (NOT KCS) to scrambled
        residual_scr = self._apply_tanh_clamp(residual_scr)
        return (residual_scr ** 2).mean()

    def compute_nullspace_per_bin_mean(
        self,
        det_features_scrambled: torch.Tensor,
        err_features: torch.Tensor,
        edge_index_d2e: torch.Tensor,
        edge_index_e2d: torch.Tensor,
        error_weights: Optional[torch.Tensor] = None,
        observable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Day 48: Per-bin scrambled mean penalty to kill constant-offset loophole.

        L_scr_mean = Σ_bins mean(Z_raw_clamped_scr | bin)^2
        Uses Z_raw (tanh clamped only, NOT KCS) — same as compute_nullspace_loss.
        """
        if not self.fg_use_scrambler_nullspace_loss:
            return torch.tensor(0.0)

        B = det_features_scrambled.shape[0]
        residual_scr = self._compute_graph_embedding(
            det_features_scrambled, err_features,
            edge_index_d2e, edge_index_e2d,
            error_weights, observable_mask)

        if self.use_density_residualization and hasattr(self, '_prior_lookup'):
            residual_scr, _, K_float_scr = self._apply_residualization(
                residual_scr, det_features_scrambled, B)

        # Apply tanh clamp only (NOT KCS) — Day 48 lesson
        residual_scr = self._apply_tanh_clamp(residual_scr)
        z_flat = residual_scr.squeeze(-1) if residual_scr.dim() > 1 else residual_scr

        # Per-bin mean penalty
        if self._kcs is not None and self._kcs._bin_edges is not None:
            bin_idx = self._kcs._k_to_bin(K_float_scr)
            penalties = []
            for b in range(self._kcs.num_bins):
                mask = bin_idx == b
                if mask.sum() < 2:
                    continue
                penalties.append(z_flat[mask].mean() ** 2)
            if penalties:
                return torch.stack(penalties).mean()
        return (z_flat.mean() ** 2)

    def compute_nullspace_loss_preclamp(
        self,
        det_features_scrambled: torch.Tensor,
        err_features: torch.Tensor,
        edge_index_d2e: torch.Tensor,
        edge_index_e2d: torch.Tensor,
        error_weights: Optional[torch.Tensor] = None,
        observable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Day 49.1: Null-space loss on PRE-CLAMP logits (z_pre).

        Unlike compute_nullspace_loss which applies tanh clamp first,
        this operates on raw unbounded logits so gradients can actually
        push the scrambled residual to zero even when tanh saturates.

        L_null_pre = mean(z_pre_scr^2)
        """
        if not self.fg_use_scrambler_nullspace_loss:
            return torch.tensor(0.0)

        B = det_features_scrambled.shape[0]
        residual_scr = self._compute_graph_embedding(
            det_features_scrambled, err_features,
            edge_index_d2e, edge_index_e2d,
            error_weights, observable_mask)

        if self.use_density_residualization and hasattr(self, '_prior_lookup'):
            residual_scr, _, _ = self._apply_residualization(
                residual_scr, det_features_scrambled, B)

        # Day 49.1: NO tanh clamp — use raw logits for full gradient flow
        return (residual_scr ** 2).mean()

    def compute_scrambled_bias_loss(
        self,
        det_features_scrambled: torch.Tensor,
        err_features: torch.Tensor,
        edge_index_d2e: torch.Tensor,
        edge_index_e2d: torch.Tensor,
        error_weights: Optional[torch.Tensor] = None,
        observable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Day 49.1: Per-bin scrambled mean² penalty on z_pre (pre-clamp).

        Kills the constant-bias attractor observed in Day 49 where per-bin
        scrambled means were all ≈ -1.18.  Operates on z_pre (no tanh) so
        gradients can push the per-bin mean to exactly 0.

        L_scr_bias = Σ_bins mean(z_pre_scr | bin)²
        """
        if not self.fg_use_scrambler_nullspace_loss:
            return torch.tensor(0.0)

        B = det_features_scrambled.shape[0]
        residual_scr = self._compute_graph_embedding(
            det_features_scrambled, err_features,
            edge_index_d2e, edge_index_e2d,
            error_weights, observable_mask)

        if self.use_density_residualization and hasattr(self, '_prior_lookup'):
            residual_scr, _, K_float_scr = self._apply_residualization(
                residual_scr, det_features_scrambled, B)

        # Day 49.1: NO tanh clamp — raw logits for unimpeded gradient
        z_flat = residual_scr.squeeze(-1) if residual_scr.dim() > 1 else residual_scr

        # Per-bin mean² penalty
        if self._kcs is not None and self._kcs._bin_edges is not None:
            bin_idx = self._kcs._k_to_bin(K_float_scr)
            penalties = []
            for b in range(self._kcs.num_bins):
                mask = bin_idx == b
                if mask.sum() < 2:
                    continue
                penalties.append(z_flat[mask].mean() ** 2)
            if penalties:
                return torch.stack(penalties).mean()
        return (z_flat.mean() ** 2)

    def compute_tv_penalty(self) -> torch.Tensor:
        """Day 43: TV penalty from alpha K-bin table."""
        if self._alpha_kbin_table is not None and self.fg_use_alpha_kbin:
            return self._alpha_kbin_table.compute_tv_penalty()
        return torch.tensor(0.0)

    # ── Day 44 helpers ────────────────────────────────────────────────────────

    def _apply_tanh_clamp(self, Z_raw: torch.Tensor) -> torch.Tensor:
        """Day 47: Bounded tanh clamp to prevent magnitude blowup."""
        if not self.fg_use_tanh_clamp:
            return Z_raw
        c = self.fg_clamp_c
        return c * torch.tanh(Z_raw / c)

    def _apply_kcs(self, residual_logit: torch.Tensor,
                   K_float: torch.Tensor) -> torch.Tensor:
        """Apply tanh clamp + KCS. Returns Z_norm or Z_raw."""
        # Day 47: tanh clamp before KCS
        residual_logit = self._apply_tanh_clamp(residual_logit)
        if self.fg_use_kcs_standardization and self._kcs is not None:
            return self._kcs(residual_logit, K_float,
                            center_only=self.fg_kcs_center_only)
        return residual_logit

    def _apply_kcs_no_clamp(self, z_already_clamped: torch.Tensor,
                             K_float: torch.Tensor) -> torch.Tensor:
        """Day 49.3: Apply KCS WITHOUT tanh clamp (input already clamped+centered)."""
        if self.fg_use_kcs_standardization and self._kcs is not None:
            return self._kcs(z_already_clamped, K_float,
                            center_only=self.fg_kcs_center_only)
        return z_already_clamped

    def setup_kcs(self, K_train: np.ndarray, num_bins: int = 12):
        """Initialize KCS from training K values."""
        from qec_noise_factory.ml.models.k_conditional_standardizer import KConditionalStandardizer
        self._kcs = KConditionalStandardizer(num_bins=num_bins,
                                            stopgrad=self.fg_stopgrad_kcs)
        self._kcs.fit_bins(K_train)

    # ── Day 49.3 helpers ─────────────────────────────────────────────────────

    def setup_ema_centering(self, num_bins: int = 12, momentum: float = 0.9):
        """Day 49.3: Init per-K-bin EMA tracker for scrambled centering."""
        # Remove plain attribute from __init__ before register_buffer
        if hasattr(self, '_ema_mu_scr') and '_ema_mu_scr' not in self._buffers:
            delattr(self, '_ema_mu_scr')
        self.register_buffer('_ema_mu_scr', torch.zeros(num_bins))
        self._ema_momentum = momentum
        self._ema_num_bins = num_bins

    def _apply_ema_centering(self, z_raw: torch.Tensor,
                              K_float: torch.Tensor) -> torch.Tensor:
        """Day 49.3: Subtract per-K-bin EMA mean (always detached)."""
        if self._ema_mu_scr is None:
            return z_raw
        if self._kcs is not None and self._kcs._bin_edges is not None:
            bin_idx = self._kcs._k_to_bin(K_float)  # (B,)
            mu = self._ema_mu_scr[bin_idx].unsqueeze(-1)  # (B, 1)
        else:
            mu = self._ema_mu_scr[0].unsqueeze(0).unsqueeze(0).expand_as(z_raw)
        return z_raw - mu.detach()

    def update_ema(self, z_raw_scr: torch.Tensor, K_float_scr: torch.Tensor):
        """Day 49.3: Update EMA tracker from scrambled batch (call during training).

        Only bins with samples in this batch are updated.
        z_raw_scr should be detached (no grad needed).
        """
        if self._ema_mu_scr is None:
            return
        m = self._ema_momentum
        z_flat = z_raw_scr.detach().squeeze(-1) if z_raw_scr.dim() > 1 else z_raw_scr.detach()
        if self._kcs is not None and self._kcs._bin_edges is not None:
            bin_idx = self._kcs._k_to_bin(K_float_scr)
            for b in range(self._ema_num_bins):
                mask = bin_idx == b
                if mask.sum() > 0:
                    batch_mean = z_flat[mask].mean()
                    self._ema_mu_scr[b] = m * self._ema_mu_scr[b] + (1 - m) * batch_mean
        else:
            self._ema_mu_scr[0] = m * self._ema_mu_scr[0] + (1 - m) * z_flat.mean()

    def compute_shielded_nullspace_loss(
        self,
        det_features_scrambled: torch.Tensor,
        err_features: torch.Tensor,
        edge_index_d2e: torch.Tensor,
        edge_index_e2d: torch.Tensor,
        error_weights: Optional[torch.Tensor] = None,
        observable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Day 49.3: Null-space loss with backbone shielding.

        Backbone embedding is detached before the head, so L_null gradients
        ONLY update the head weights — backbone features encoding topology
        are protected.

        Pipeline:
        1. backbone(G_scr) → graph_emb_scr
        2. head(graph_emb_scr.detach()) → z_pre_scr  (backbone shielded)
        3. residualization → tanh clamp → EMA centering → KCS sigma
        4. L = mean(z_norm_scr^2)
        """
        if not self.fg_use_scrambler_nullspace_loss:
            return torch.tensor(0.0)

        B = det_features_scrambled.shape[0]

        # Step 1: backbone (full graph traversal)
        graph_emb_scr = self._compute_backbone(
            det_features_scrambled, err_features,
            edge_index_d2e, edge_index_e2d,
            error_weights, observable_mask)

        # Step 2: head on DETACHED backbone embedding
        z_pre_scr = self.head(graph_emb_scr.detach())

        # Step 3: residualization + pipeline
        if self.use_density_residualization and hasattr(self, '_prior_lookup'):
            z_pre_scr, _, K_float_scr = self._apply_residualization(
                z_pre_scr, det_features_scrambled, B)
        else:
            K_float_scr = torch.zeros(B)

        # Tanh clamp
        z_raw_scr = self._apply_tanh_clamp(z_pre_scr)

        # Update EMA with this batch's scrambled z_raw (before centering)
        if self.training:
            self.update_ema(z_raw_scr, K_float_scr)

        # EMA centering
        z_centered_scr = z_raw_scr
        if self.fg_use_ema_centering and self._ema_mu_scr is not None:
            z_centered_scr = self._apply_ema_centering(z_raw_scr, K_float_scr)

        # KCS sigma normalization (no clamp — already clamped)
        z_norm_scr = self._apply_kcs_no_clamp(z_centered_scr, K_float_scr)

        return (z_norm_scr ** 2).mean()

    # ── Day 50 helpers ───────────────────────────────────────────────────────

    def compute_scr_baseline(self, z_pre_scr: torch.Tensor,
                              K_float_scr: Optional[torch.Tensor] = None
                              ) -> torch.Tensor:
        """Day 50: Compute per-step detached baseline from scrambled residual.

        Returns a detached baseline (scalar or per-sample) that when subtracted
        from z_pre centers the scrambled residual around 0.

        Args:
            z_pre_scr: (B, 1) scrambled residual pre-clamp
            K_float_scr: (B,) K values for per-bin mode
        """
        z_flat = z_pre_scr.detach().squeeze(-1)

        if self.fg_scr_baseline_mode == "bin" and self._kcs is not None \
                and hasattr(self._kcs, '_bin_edges') and self._kcs._bin_edges is not None \
                and K_float_scr is not None:
            bin_idx = self._kcs._k_to_bin(K_float_scr)
            b = torch.zeros_like(z_flat)
            for i in range(self._kcs.num_bins):
                mask = bin_idx == i
                if mask.sum() > 0:
                    b[mask] = z_flat[mask].mean()
                # empty bins: leave as 0 (safe fallback)
            return b.unsqueeze(-1).detach()

        # Default: batch scalar baseline
        return z_flat.mean().detach()

    def compute_centered_nullspace_loss(
        self,
        det_features_scrambled: torch.Tensor,
        err_features: torch.Tensor,
        edge_index_d2e: torch.Tensor,
        edge_index_e2d: torch.Tensor,
        error_weights: Optional[torch.Tensor] = None,
        observable_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Day 50: Null-space loss with per-step baseline centering.

        Pipeline:
        1. backbone(G_scr) → graph_emb_scr
        2. head(graph_emb_scr.detach()) → z_pre_scr  (backbone shielded)
        3. residualize
        4. b = compute_scr_baseline(z_pre_scr)  (detached, per-step)
        5. z_centered = z_pre_scr - b
        6. z_scr_used = c * tanh(z_centered / c)
        7. L_null = mean(z_scr_used^2)

        Returns:
            dict with 'loss', 'z_scr_used', 'baseline_mean'
        """
        if not self.fg_use_scrambler_nullspace_loss:
            return {'loss': torch.tensor(0.0), 'z_scr_used': None, 'baseline_mean': 0.0}

        B = det_features_scrambled.shape[0]

        # Step 1: backbone
        graph_emb_scr = self._compute_backbone(
            det_features_scrambled, err_features,
            edge_index_d2e, edge_index_e2d,
            error_weights, observable_mask)

        # Step 2: head on DETACHED backbone (shields backbone from L_null)
        z_pre_scr = self.head(graph_emb_scr.detach())

        # Step 3: residualize
        K_float_scr = None
        if self.use_density_residualization and hasattr(self, '_prior_lookup'):
            z_pre_scr, _, K_float_scr = self._apply_residualization(
                z_pre_scr, det_features_scrambled, B)

        # Step 4: compute per-step baseline (detached)
        b = self.compute_scr_baseline(z_pre_scr, K_float_scr)

        # Step 5: center
        z_centered = z_pre_scr - b

        # Step 6: tanh clamp AFTER centering
        z_scr_used = self._apply_tanh_clamp(z_centered)

        # Step 7: null-space MSE loss
        loss = (z_scr_used ** 2).mean()

        b_mean = b.mean().item() if isinstance(b, torch.Tensor) else float(b)
        return {'loss': loss, 'z_scr_used': z_scr_used, 'baseline_mean': b_mean}

    # ── Day 51 helpers ───────────────────────────────────────────────────────

    def setup_sigma_ema(self, num_bins: int = 12, momentum: float = 0.9):
        """Day 51: Init per-K-bin sigma EMA for scale-only KCS.

        Sigma initialized to 1.0 (safe neutral scale).
        """
        if hasattr(self, '_sigma_ema') and '_sigma_ema' not in self._buffers:
            delattr(self, '_sigma_ema')
        self.register_buffer('_sigma_ema', torch.ones(num_bins))
        self._sigma_ema_momentum = momentum
        self._sigma_ema_num_bins = num_bins

    def update_sigma_ema(self, z_centered_real: torch.Tensor,
                          K_float: torch.Tensor):
        """Day 51: Update sigma EMA from DETACHED real-branch stats.

        Only updates bins with >= min_count samples.
        """
        if self._sigma_ema is None:
            return
        z_flat = z_centered_real.detach().squeeze(-1)
        m = self._sigma_ema_momentum
        min_count = self._sigma_ema_min_count

        if self._kcs is not None and hasattr(self._kcs, '_bin_edges') \
                and self._kcs._bin_edges is not None:
            bin_idx = self._kcs._k_to_bin(K_float)
            for b in range(self._sigma_ema_num_bins):
                mask = bin_idx == b
                if mask.sum() >= min_count:
                    batch_std = z_flat[mask].std()
                    if not torch.isnan(batch_std) and batch_std > 0:
                        self._sigma_ema[b] = m * self._sigma_ema[b] + (1 - m) * batch_std
        else:
            # Global fallback
            if z_flat.numel() >= min_count:
                batch_std = z_flat.std()
                if not torch.isnan(batch_std) and batch_std > 0:
                    self._sigma_ema[0] = m * self._sigma_ema[0] + (1 - m) * batch_std

    def apply_scale_only_kcs(self, z_centered: torch.Tensor,
                              K_float: torch.Tensor) -> torch.Tensor:
        """Day 51: Scale-only KCS — divide by sigma_ema, NO mean subtraction.

        sigma = max(sigma_ema[Kbin].detach(), sigma_floor)
        z_norm = z_centered / sigma
        """
        if self._sigma_ema is None:
            return z_centered

        if self._kcs is not None and hasattr(self._kcs, '_bin_edges') \
                and self._kcs._bin_edges is not None:
            bin_idx = self._kcs._k_to_bin(K_float)
            sigma = self._sigma_ema[bin_idx].detach().unsqueeze(-1)
        else:
            sigma = self._sigma_ema[0].detach().unsqueeze(0).unsqueeze(0).expand_as(z_centered)

        sigma = torch.clamp(sigma, min=self.fg_sigma_floor)
        return z_centered / sigma

    def forward_day51(
        self,
        det_features: torch.Tensor,
        err_features: torch.Tensor,
        edge_index_d2e: torch.Tensor,
        edge_index_e2d: torch.Tensor,
        error_weights: Optional[torch.Tensor] = None,
        observable_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Day 51: Forward with center→scale→clamp pipeline.

        Pipeline:
        1. backbone → head → residualize → z_pre
        2. C_bin baseline center: z_centered = z_pre - b
        3. Scale-only KCS: z_scaled = z_centered / sigma_ema
        4. Tanh clamp: z_used = c * tanh(z_scaled / c)  (AFTER division)
        5. Return dict (experiment handles recombination with warmup)
        """
        B = det_features.shape[0]
        residual_logit = self._compute_graph_embedding(
            det_features, err_features, edge_index_d2e, edge_index_e2d,
            error_weights, observable_mask)

        if not (self.use_density_residualization and hasattr(self, '_prior_lookup')):
            return {
                'z_pre': residual_logit, 'z_centered': residual_logit,
                'z_scaled': residual_logit, 'z_used': residual_logit,
                'logit_prior': torch.zeros_like(residual_logit),
                'K': None, 'alpha': None,
            }

        z_pre, prior_logit, K_float = self._apply_residualization(
            residual_logit, det_features, B)

        # Step 2: C_bin baseline center
        b = self.compute_scr_baseline(z_pre, K_float)
        z_centered = z_pre - b

        # Step 3: scale-only KCS (divide by sigma_ema)
        if self.fg_use_sigma_ema and self._sigma_ema is not None:
            z_scaled = self.apply_scale_only_kcs(z_centered, K_float)
        else:
            z_scaled = z_centered

        # Step 4: tanh clamp AFTER division
        z_used = self._apply_tanh_clamp(z_scaled)

        alpha = self._get_alpha(K_float)
        return {
            'z_pre': z_pre,
            'z_centered': z_centered,
            'z_scaled': z_scaled,
            'z_used': z_used,
            'logit_prior': prior_logit,
            'K': K_float,
            'alpha': alpha,
            'baseline_mean': b.mean().item() if isinstance(b, torch.Tensor) else float(b),
        }

    def compute_day51_nullspace_loss(
        self,
        det_features_scrambled: torch.Tensor,
        err_features: torch.Tensor,
        edge_index_d2e: torch.Tensor,
        edge_index_e2d: torch.Tensor,
        error_weights: Optional[torch.Tensor] = None,
        observable_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Day 51: Null-space with center→scale(REAL sigma)→clamp pipeline.

        Uses REAL sigma_ema for scrambled normalization. Backbone shielded.
        """
        if not self.fg_use_scrambler_nullspace_loss:
            return {'loss': torch.tensor(0.0), 'z_scr_used': None, 'baseline_mean': 0.0}

        B = det_features_scrambled.shape[0]

        # Backbone shielded
        graph_emb_scr = self._compute_backbone(
            det_features_scrambled, err_features,
            edge_index_d2e, edge_index_e2d,
            error_weights, observable_mask)
        z_pre_scr = self.head(graph_emb_scr.detach())

        # Residualize
        K_float_scr = None
        if self.use_density_residualization and hasattr(self, '_prior_lookup'):
            z_pre_scr, _, K_float_scr = self._apply_residualization(
                z_pre_scr, det_features_scrambled, B)

        # Center
        b = self.compute_scr_baseline(z_pre_scr, K_float_scr)
        z_centered = z_pre_scr - b

        # Scale using REAL sigma_ema (not scrambled stats)
        if self.fg_use_sigma_ema and self._sigma_ema is not None and K_float_scr is not None:
            z_scaled = self.apply_scale_only_kcs(z_centered, K_float_scr)
        else:
            z_scaled = z_centered

        # Clamp after scaling
        z_scr_used = self._apply_tanh_clamp(z_scaled)

        loss = (z_scr_used ** 2).mean()
        b_mean = b.mean().item() if isinstance(b, torch.Tensor) else float(b)
        return {'loss': loss, 'z_scr_used': z_scr_used, 'baseline_mean': b_mean}

    def setup_scr_baseline(self, num_bins: int = 12):
        """Day 49.2: Init per-K-bin learnable baseline for scrambled debias.

        Baseline is initialized to zero. It is trained ONLY by the scrambled
        null-space loss (detached on real branch).
        """
        self._scr_baseline = nn.Embedding(num_bins, 1)
        nn.init.zeros_(self._scr_baseline.weight)
        self._scr_num_bins = num_bins

    def _get_baseline(self, K_float: torch.Tensor) -> torch.Tensor:
        """Day 49.2: Look up per-K-bin baseline. Returns (B, 1)."""
        if self._kcs is not None and self._kcs._bin_edges is not None:
            bin_idx = self._kcs._k_to_bin(K_float)
            return self._scr_baseline(bin_idx)  # (B, 1)
        # Fallback: use bin 0 for all
        return self._scr_baseline(torch.zeros_like(K_float.long()))

    def compute_debiased_nullspace_loss(
        self,
        det_features_scrambled: torch.Tensor,
        err_features: torch.Tensor,
        edge_index_d2e: torch.Tensor,
        edge_index_e2d: torch.Tensor,
        error_weights: Optional[torch.Tensor] = None,
        observable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Day 49.2: Null-space loss on debiased scrambled residual.

        z_deb_scr = z_raw_scr - b(kbin)  where b is TRAINABLE (not detached)
        L = mean(z_deb_scr^2)

        The baseline b learns to absorb the constant offset so that
        z_deb_scr → 0 even if z_raw_scr has a rest-state bias.
        """
        if not self.fg_use_scrambler_nullspace_loss:
            return torch.tensor(0.0)
        if self._scr_baseline is None:
            # Fallback to standard null-space if baseline not set up
            return self.compute_nullspace_loss(
                det_features_scrambled, err_features,
                edge_index_d2e, edge_index_e2d,
                error_weights, observable_mask)

        B = det_features_scrambled.shape[0]
        residual_scr = self._compute_graph_embedding(
            det_features_scrambled, err_features,
            edge_index_d2e, edge_index_e2d,
            error_weights, observable_mask)

        if self.use_density_residualization and hasattr(self, '_prior_lookup'):
            residual_scr, _, K_float_scr = self._apply_residualization(
                residual_scr, det_features_scrambled, B)

        # Apply tanh clamp → z_raw_scr
        residual_scr = self._apply_tanh_clamp(residual_scr)

        # Subtract TRAINABLE baseline (NOT detached) → z_deb_scr
        b_scr = self._get_baseline(K_float_scr)  # trainable
        z_deb_scr = residual_scr - b_scr

        return (z_deb_scr ** 2).mean()

    def setup_adversary(self, K_train: np.ndarray, hidden_dim: int = 16):
        """Initialize K adversary."""
        from qec_noise_factory.ml.models.grl import KAdversary
        self._k_adversary = KAdversary(input_dim=1, hidden_dim=hidden_dim)
        self._k_adversary.set_k_stats(K_train)

    def freeze_kcs_stats(self, det_train, err_features, ei_d2e, ei_e2d,
                         error_weights=None, observable_mask=None):
        """Compute and freeze KCS stats on full training set (call after training)."""
        if not self.fg_use_kcs_standardization or self._kcs is None:
            return
        self.eval()
        with torch.no_grad():
            split = self.forward_split(det_train, err_features, ei_d2e, ei_e2d,
                                       error_weights=error_weights,
                                       observable_mask=observable_mask)
        self._kcs.freeze_stats(split['logit_residual'], split['K'])

    def compute_adversary_loss(self, Z_norm: torch.Tensor,
                                K_float: torch.Tensor) -> torch.Tensor:
        """Day 44+45: Adversary loss with GRL + linear warmup schedule."""
        if not self.fg_use_grl_k_adversary or self._k_adversary is None:
            return torch.tensor(0.0)
        from qec_noise_factory.ml.models.grl import compute_warmup_lambda
        lam_eff = compute_warmup_lambda(
            self._current_epoch, self.fg_adv_warmup_epochs, self.fg_lambda_adv)
        if lam_eff < 1e-8:
            return torch.tensor(0.0)
        return self._k_adversary.compute_loss(Z_norm, K_float, lambda_adv=lam_eff)

    def compute_iso_k_loss(self, Z_norm: torch.Tensor, K: torch.Tensor,
                            y: torch.Tensor) -> torch.Tensor:
        """Day 45: Iso-K pairwise margin loss for topology preservation."""
        if not self.fg_use_iso_k_loss:
            return torch.tensor(0.0)
        from qec_noise_factory.ml.models.iso_k_loss import compute_iso_k_loss
        return compute_iso_k_loss(Z_norm, K, y, margin=self.fg_iso_margin)

    def compute_leakage_penalty(self, Z_norm: torch.Tensor,
                                 K: torch.Tensor) -> torch.Tensor:
        """Day 46: Non-adversarial leakage penalty (moment-matching + envelope)."""
        if not self.fg_use_leakage_penalty:
            return torch.tensor(0.0)
        if self._kcs is None or self._kcs._bin_edges is None:
            return torch.tensor(0.0)
        from qec_noise_factory.ml.models.leakage_penalty import compute_full_leakage_penalty
        return compute_full_leakage_penalty(
            Z_norm, K, self._kcs._bin_edges.to(Z_norm.device), self._kcs.num_bins)

    def compute_envelope_leakage_penalty(self, Z_norm: torch.Tensor,
                                          K: torch.Tensor) -> torch.Tensor:
        """Day 47: Envelope-only leakage penalty — Corr(|Z|, K)²."""
        if not self.fg_use_envelope_penalty:
            return torch.tensor(0.0)
        from qec_noise_factory.ml.models.leakage_penalty import compute_envelope_penalty
        if self._kcs is not None and self._kcs._bin_edges is not None:
            return compute_envelope_penalty(
                Z_norm, K, self._kcs._bin_edges.to(Z_norm.device), self._kcs.num_bins)
        return compute_envelope_penalty(Z_norm, K, None, 1)

