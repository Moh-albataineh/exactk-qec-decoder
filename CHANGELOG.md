# Changelog

All notable changes to ExactK QEC Decoder are documented in this file.

## [1.0.0] — 2026-03-03

### Release: ExactK V1.0

**Training Configuration**
- ExactK iso-K hinge loss: ΔK=0, λ=0.10, margin=0.30, decay=0.85^(ep−8)
- Factor-Graph v1 bipartite message-passing decoder
- FocalLoss (γ=2) with BRQL calibration
- Effective batch size B=256 (micro=64, grad_accum=4 for d=7)

**Selector v6 Policy**
- Survival: `tg_roll ≥ −0.015` (drop_slice_floor mode)
- CLEAN pool: `g1roll ≤ 0.025` AND `g1_inst ≤ 0.035` → argmax(tg_roll)
- LEAKY pool: argmin(g1roll)
- No silent defaults — missing receipt fields raise KeyError

**MLOps Infrastructure**
- JSONL Write-Ahead Logging with `os.fsync()` per epoch
- Progressive checkpointing for epochs ≥ 6
- Post-training selector v6 + receipt generation

**Validated Results**
- Day 69 (d=5): +31.2% median G1 reduction — PASS
- Day 70 (d=7): +23.4% OOD generalization — GEN_PASS
- Day 75 (holdout, d=7, seeds 60000–60009):
  - Science Δ: +45.0% (target ≥ 20%)
  - Safe Yield: 80% (target ≥ 80%)
  - TOPO_FAIL: 0/10 (target ≤ 10%)
  - KPI-B: +15.1% leaky cohort improvement (83% wins)
  - Do-No-Harm: 0 violations
  - KPI-A: Prod tg_roll=0.0664 vs Ctrl=0.0800 (−17%, informational)

**Deprecations**
- Selected-G1 Δ% permanently deprecated (Asymmetric Selection Paradox)
