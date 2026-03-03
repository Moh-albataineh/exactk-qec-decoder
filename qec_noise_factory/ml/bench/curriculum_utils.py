"""
Day 38 — Curriculum Transfer Utilities

Freezing controls and scrambler regularization for curriculum transfer.
Only used by experiment_day38_curriculum_transfer.py — no default pipeline changes.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Phase B: Freezing Controls
# ============================================================

def freeze_backbone(model: nn.Module) -> None:
    """Freeze all MP layers + input projections. Only head remains trainable.

    Freezes: det_proj, err_proj, all mp_layers, parity (if exists).
    Leaves trainable: head, density prior (already frozen buffer).
    """
    for name, param in model.named_parameters():
        if any(k in name for k in ["det_proj", "err_proj", "mp_layers", "parity"]):
            param.requires_grad = False


def unfreeze_last_mp(model: nn.Module) -> None:
    """Unfreeze only the last MP layer (keep earlier layers frozen).

    Call after freeze_backbone() to partially unfreeze.
    Leaves trainable: last mp_layer + head.
    """
    if not hasattr(model, "mp_layers"):
        return
    n_layers = len(model.mp_layers)
    if n_layers == 0:
        return
    last_idx = n_layers - 1
    for name, param in model.named_parameters():
        if f"mp_layers.{last_idx}" in name:
            param.requires_grad = True


def get_trainable_params(model: nn.Module):
    """Return only parameters with requires_grad=True (for optimizer)."""
    return [p for p in model.parameters() if p.requires_grad]


# ============================================================
# Phase C: Scrambler Regularization
# ============================================================

def compute_scrambler_reg_loss(
    logits_real: torch.Tensor,
    logits_scrambled: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Scrambler-based regularization loss.

    For y=1 graphs only:
        loss = relu(margin - (logit_real - logit_scrambled))

    This encourages the model to assign higher logits to real (structured)
    syndrome patterns vs scrambled (structure-destroyed) patterns,
    ensuring it learns topology rather than just density.

    Args:
        logits_real: (B, O) logits from real data.
        logits_scrambled: (B, O) logits from scrambled data.
        targets: (B, O) binary targets.
        margin: required gap between real and scrambled logits.

    Returns:
        Scalar loss (mean over y=1 samples). Returns 0 if no y=1 samples.
    """
    # Mask: only y=1 samples
    pos_mask = targets > 0.5  # (B, O)
    if not pos_mask.any():
        return torch.tensor(0.0, device=logits_real.device, requires_grad=True)

    gap = logits_real - logits_scrambled  # (B, O)
    loss = F.relu(margin - gap)  # (B, O)

    # Average only over positive samples
    return loss[pos_mask].mean()
