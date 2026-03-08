# Reproduction Policy

This document explains how to interpret reproduction runs and what level of numerical agreement to expect.

## Canonical Environment

The results cited in `RESULTS.md`, `DAYS.md`, and the paper were produced in the following environments:

| | Development (Days 1–69, d=5) | Production (Days 70–75, d=7) |
|---|---|---|
| **OS** | Windows 11 | Ubuntu 24.04 (RunPod Docker) |
| **CPU** | Intel Core i7-14700HX | Intel Xeon 6952P |
| **RAM** | 32 GB DDR5 | 188 GB |
| **GPU** | NVIDIA RTX 4060 (8 GB) | NVIDIA RTX PRO 6000 (96 GB) |
| **Compute** | CPU-only (Day 69) | GPU (Days 70–75) |
| **Python** | 3.10+ | 3.10+ |
| **PyTorch** | 2.x (CPU) | 2.x (CUDA 12.4) |

> [!NOTE]
> Day 69 (d=5) canonical results were produced on **CPU**. Days 70 and 75 (d=7) canonical results were produced on **GPU**. The compute device affects floating-point ordering and therefore the exact training trajectory.

## Expected Variance Sources

When re-running experiments on different hardware, the following factors can cause numerical differences:

| Source | Impact | Mitigation |
|--------|--------|------------|
| **GPU vs CPU execution** | High — different RNG streams, reduction order | Use same device type as canonical run |
| **Different GPU architecture** | Medium — different cuDNN algorithm selection | `cudnn.deterministic=True` (already set) |
| **CUDA / cuDNN version** | Low-Medium — kernel implementation differences | Pin versions if bit-exact needed |
| **PyTorch version** | Low — internal implementation changes | Pin version |
| **Sparse/GNN message passing** | Medium — non-associative floating-point sums | Inherent; cannot be fully eliminated |
| **OS / compiler** | Low — floating-point rounding differences | Negligible in practice |

## Deterministic Settings

All experiment scripts include:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

These ensure that **repeated runs on the same hardware** produce identical results. They do **not** guarantee agreement across different GPUs, CUDA versions, or CPU vs GPU.

## How to Judge a Successful Reproduction

A reproduction is considered **successful** if it preserves:

1. **Same high-level verdict** — PASS/PARTIAL/FAIL classification matches or is within one tier
2. **Same safety conclusion** — zero topology collapses, alignment invariant pass, no NaN/Inf
3. **Same best arm** — the same ExactK variant wins (or both are close)
4. **Qualitatively similar effect size** — the direction and approximate magnitude of improvement are consistent (e.g., both show +20–35% G1 reduction, not one showing +30% and another −50%)

A reproduction is **not required** to:

- Match every decimal in canonical tables
- Produce identical per-seed G1 values
- Cross the exact same threshold (e.g., 30% PASS vs 28% PARTIAL is acceptable variance)

## What Is Guaranteed

| Guarantee | Scope |
|-----------|-------|
| Same hardware, same software → **bit-identical** | ✅ Yes (deterministic settings) |
| Same GPU model, same CUDA → **very close** | ✅ Expected |
| Different GPU or CPU vs GPU → **qualitatively consistent** | ✅ Expected |
| Any hardware → **exact decimal match** | ❌ Not guaranteed |

## Which Numbers Are Canonical

The **canonical reference numbers** are those in:

- [`docs/RESULTS.md`](RESULTS.md) — all result tables
- [`docs/DAYS.md`](DAYS.md) — per-day experiment summaries
- [`README.md`](../README.md) — V1.0 headline KPIs

These were produced in the canonical environments listed above and are the authoritative values for the paper. Reproduction runs that show consistent safety invariants and similar effect sizes **confirm** the scientific claims even if exact numbers differ.
