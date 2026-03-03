"""
Day 45 — Iso-K Pairwise Margin Loss

Within-bin pairwise margin loss on Z_norm (real graphs only).
Protects topology signal by enforcing that Z_norm(pos) > Z_norm(neg) within same K-bin.
"""
from __future__ import annotations

import torch


def compute_iso_k_loss(Z_norm: torch.Tensor, K: torch.Tensor, y: torch.Tensor,
                       margin: float = 0.2, max_pairs_per_bin: int = 8) -> torch.Tensor:
    """Compute iso-K pairwise margin loss.

    For each K-bin containing both y=1 and y=0 samples:
      L = mean(relu(margin - (Z_pos - Z_neg)))

    Args:
        Z_norm: (B, 1) standardized residual logits.
        K: (B,) syndrome counts.
        y: (B,) binary labels (0/1 or bool).
        margin: margin for hinge loss.
        max_pairs_per_bin: max sampled pairs per bin.

    Returns:
        Scalar loss (0.0 if no valid pairs).
    """
    z = Z_norm.squeeze(-1) if Z_norm.dim() > 1 else Z_norm
    y_bool = y.bool() if y.dtype != torch.bool else y
    K_int = K.long()
    unique_k = K_int.unique()

    losses = []
    for k_val in unique_k:
        mask_k = K_int == k_val
        pos_mask = mask_k & y_bool
        neg_mask = mask_k & (~y_bool)

        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()
        if n_pos < 1 or n_neg < 1:
            continue

        z_pos = z[pos_mask]
        z_neg = z[neg_mask]

        # Sample pairs (cross product, limited)
        n_pairs = min(max_pairs_per_bin, n_pos * n_neg)
        if n_pos * n_neg <= max_pairs_per_bin:
            # Full cross product
            diff = z_pos.unsqueeze(1) - z_neg.unsqueeze(0)  # (n_pos, n_neg)
            pair_loss = torch.relu(margin - diff)
            losses.append(pair_loss.mean())
        else:
            # Random subsample
            idx_p = torch.randint(n_pos, (n_pairs,), device=z.device)
            idx_n = torch.randint(n_neg, (n_pairs,), device=z.device)
            diff = z_pos[idx_p] - z_neg[idx_n]
            pair_loss = torch.relu(margin - diff)
            losses.append(pair_loss.mean())

    if len(losses) == 0:
        return torch.tensor(0.0, device=Z_norm.device)

    return torch.stack(losses).mean()
