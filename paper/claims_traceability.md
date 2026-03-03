# Claims Traceability — ExactK V1.0

This document maps each major paper claim to the reproduction script, test file, and documentation section that provides evidence.

---

## Core Claims

| # | Claim | Script | Test | Docs |
|---|-------|--------|------|------|
| 1 | ExactK eliminates all K-confounding within iso-K pairs | `tests/experiment_day69_exactk_tuned.py` | `tests/test_ml_day67_iso_k_ranking.py` | `docs/METHODS.md` § Days 67–69 |
| 2 | +31.2% median G1 reduction at d=5 (10 seeds) | `scripts/reproduce/repro_day69_d5.sh` | `tests/test_ml_day69_exactk_tuned.py` | `docs/RESULTS.md` § Day 69 |
| 3 | +23.4% OOD generalization at d=7 (10 seeds) | `scripts/reproduce/repro_day70_d7.sh` | `tests/test_ml_day70_exactk_d7.py` | `docs/RESULTS.md` § Day 70 |
| 4 | +45.0% science delta on holdout (seeds 60000–60009) | `scripts/reproduce/repro_day75_holdout.sh` | `tests/test_ops_day75_2_v1_release_kpis.py` | `docs/RESULTS.md` § Day 75 |
| 5 | 80% safe yield (8/10 seeds in CLEAN pool) | `scripts/reproduce/run_selector_v6.sh` | `tests/test_ops_checkpoint_selection.py` | `docs/RESULTS.md` § Day 75 |
| 6 | 0% TOPO_FAIL on holdout | `scripts/reproduce/repro_day75_holdout.sh` | — | `docs/RESULTS.md` § Day 75 |
| 7 | 0 Do-No-Harm violations | `scripts/day75_2_compute_v1_release_kpis.py` | `tests/test_ops_day75_2_v1_release_kpis.py` | `docs/RESULTS.md` § Day 75 |

---

## Mechanism Claims

| # | Claim | Evidence |
|---|-------|----------|
| 8 | NearK (ΔK=1) reintroduces confounding; ExactK (ΔK=0) does not | `tests/test_ml_day67_iso_k_ranking.py`: pair mining tests with delta_k=0 vs 1 |
| 9 | 14 mechanisms tested, only ExactK succeeded | `docs/DAYS.md` Days 42–69 (complete narrative); `docs/METHODS.md` Days 42–69 |
| 10 | Gradient shielding closed as research direction | `docs/LIMITATIONS.md` § Day 64; `docs/METHODS.md` § Day 64 |
| 11 | Asymmetric Selection Paradox invalidates Selected-G1 Δ% | `docs/RESULTS.md` § Metric Deprecations; `docs/MLOPS_POLICY.md` § Deprecation |

---

## MLOps Claims

| # | Claim | Evidence |
|---|-------|----------|
| 12 | Selector v6: drop_slice_floor eliminates TOPO_FAIL | `scripts/day73_select_checkpoints_v6.py` retroactive; `docs/RESULTS.md` § Day 73 |
| 13 | JSONL WAL is crash-safe | `docs/MLOPS_POLICY.md` § JSONL WAL; source: `qec_noise_factory/ml/ops/checkpoint_selection.py` |
| 14 | Receipt schema has no silent defaults | `tests/test_ops_day75_2_v1_release_kpis.py` :: `TestExtractRequiredFloat` |

---

## How to Verify

1. **Unit tests**: `python -m pytest tests/ -v` — validates claims 1, 2, 3, 5, 7, 8, 14
2. **Repro scripts**: `scripts/reproduce/repro_day69_d5.sh` (claim 2), `repro_day70_d7.sh` (claim 3), `repro_day75_holdout.sh` (claims 4, 5, 6, 7)
3. **Documentation**: Full narrative in `docs/DAYS.md` (claims 9, 10, 11)
