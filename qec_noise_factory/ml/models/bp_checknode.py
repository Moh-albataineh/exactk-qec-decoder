"""
Day 39 — BP Check-Node Message Passing

Numerically stable belief-propagation-style check-node update using
the log-tanh / sign-logabs trick.  Computes extrinsic messages from
detector (check) nodes to error (variable) nodes.

This module is used when ``fg_use_bp_checknode=True`` in FactorGraphDecoderV1.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

_LLR_CLAMP = 15.0
_TANH_CLAMP = 0.9999


class BPCheckNode(nn.Module):
    """Belief-propagation check-node update (numerically stable).

    For each detector (check) node, computes extrinsic leave-one-out
    messages to incident error (variable) nodes using:

        t_e = tanh(llr_e / 2)
        product_at_d = prod t_e  (over edges incident to detector d)
        extrinsic_e = product_at_d / t_e   (leave-one-out)
        msg_e = 2 * atanh(syndrome_sign * extrinsic_e)

    With damping toward previous messages (default beta=0.2).

    Args:
        hidden_dim: dimension of error node embeddings (for llr projection).
        beta: damping coefficient (0 = no damping, 1 = fully damped).
    """

    def __init__(self, hidden_dim: int, beta: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta

        # Project error embedding -> scalar LLR per edge
        self.to_llr = nn.Linear(hidden_dim, 1)

        # Project BP messages back to hidden_dim for concat
        self.msg_proj = nn.Linear(1, hidden_dim)

        self._prev_msg = None  # for damping

    def reset_damping(self):
        """Reset previous messages (call at start of new input)."""
        self._prev_msg = None

    def forward(
        self,
        hE: torch.Tensor,            # (B, N_e, H)
        hD: torch.Tensor,            # (B, N_d, H)
        syndrome: torch.Tensor,      # (B, N_d) binary {0, 1}
        edge_e2d: torch.Tensor,      # (2, E) long -- (err_idx, det_idx)
        edge_d2e: torch.Tensor,      # (2, E) long -- (det_idx, err_idx)
    ) -> torch.Tensor:
        """Compute BP check-node messages and aggregate to error nodes.

        Returns:
            bp_msg_E: (B, N_e, H) -- BP messages aggregated at error nodes,
            ready to be concatenated with neural MP messages.
        """
        B, N_e, H = hE.shape
        _, N_d, _ = hD.shape
        E = edge_e2d.shape[1]
        device = hE.device

        e_src, d_dst = edge_e2d[0], edge_e2d[1]  # (E,)
        d_src, e_dst = edge_d2e[0], edge_d2e[1]  # (E,)

        # --- Step 1: Compute LLR per edge from error embeddings ---
        hE_at_edge = hE[:, e_src, :]      # (B, E, H)
        llr = self.to_llr(hE_at_edge).squeeze(-1)  # (B, E)
        llr = llr.clamp(-_LLR_CLAMP, _LLR_CLAMP)

        # --- Step 2: tanh half-LLR ---
        t = torch.tanh(llr / 2.0)        # (B, E)
        t = t.clamp(-_TANH_CLAMP, _TANH_CLAMP)

        # --- Step 3: Log-domain product at each detector node ---
        sign = torch.sign(t)             # (B, E) -- +/-1
        log_abs = torch.log(t.abs().clamp(min=1e-30))  # (B, E)

        # Sign product: use (-1)^count_negatives per detector
        is_neg = (sign < 0).float()       # (B, E)
        neg_count = torch.zeros(B, N_d, device=device)
        d_dst_exp = d_dst.unsqueeze(0).expand(B, -1)
        neg_count.scatter_add_(1, d_dst_exp, is_neg)

        # Total sign per detector: (-1)^neg_count
        sign_product = 1.0 - 2.0 * (neg_count % 2)  # (B, N_d)

        # Log-abs sum per detector
        logabs_sum = torch.zeros(B, N_d, device=device)
        logabs_sum.scatter_add_(1, d_dst_exp, log_abs)

        # --- Step 4: Extrinsic leave-one-out per edge ---
        # For edge e->d: extrinsic = (total product at d) / t_e
        # In log domain: logabs_ext = logabs_sum[d] - log_abs[e]
        logabs_ext = logabs_sum[:, d_dst] - log_abs  # (B, E)

        # Sign extrinsic: sign_product[d] * sign[e] (dividing out this edge's sign)
        sign_ext = sign_product[:, d_dst] * sign      # (B, E)

        # --- Step 5: Syndrome sign flip ---
        # syndrome = 1 means error detected -> flip sign
        syn_sign = 1.0 - 2.0 * syndrome[:, d_dst]  # (B, E), +1 if syn=0, -1 if syn=1
        sign_ext = sign_ext * syn_sign

        # --- Step 6: Invert back to LLR domain ---
        ext_t = sign_ext * torch.exp(logabs_ext)    # (B, E)
        ext_t = ext_t.clamp(-_TANH_CLAMP, _TANH_CLAMP)
        msg = 2.0 * torch.atanh(ext_t)              # (B, E)

        # Clamp for numerical safety
        msg = msg.clamp(-_LLR_CLAMP, _LLR_CLAMP)

        # --- Step 7: Damping ---
        if self._prev_msg is not None and self._prev_msg.shape == msg.shape:
            msg = (1.0 - self.beta) * msg + self.beta * self._prev_msg
        self._prev_msg = msg.detach()

        # --- Step 8: Aggregate messages at error nodes (scatter mean) ---
        msg_2d = msg.unsqueeze(-1)  # (B, E, 1)

        # Map edges back to error nodes using d2e edges (d_src->e_dst)
        bp_agg = torch.zeros(B, N_e, 1, device=device)
        e_dst_exp = e_dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        bp_agg.scatter_add_(1, e_dst_exp, msg_2d)

        count_e = torch.zeros(B, N_e, 1, device=device)
        bp_count_idx = e_dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        count_e.scatter_add_(1, bp_count_idx, torch.ones_like(bp_count_idx, dtype=hE.dtype))
        count_e = count_e.clamp(min=1.0)
        bp_agg = bp_agg / count_e  # (B, N_e, 1)

        # Project to hidden_dim
        bp_msg_E = self.msg_proj(bp_agg)  # (B, N_e, H)

        return bp_msg_E
