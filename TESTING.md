# Testing Guide

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_ml_day67_iso_k_ranking.py -v     # ExactK core
python -m pytest tests/test_ops_checkpoint_selection.py -v    # Selector v6
python -m pytest tests/test_ops_day75_2_v1_release_kpis.py -v  # KPI computation
```

## Test Categories

| Category | Files | What they test |
|----------|-------|----------------|
| **ExactK core** | `test_ml_day67_*`, `test_ml_day68_*`, `test_ml_day69_*` | Iso-K hinge loss, pair mining, ExactK tuned config |
| **d=7 OOD** | `test_ml_day70_*` | Out-of-distribution generalization |
| **Selector/KPI** | `test_ops_*` | Selector v6, receipt schema, KPI computation |
| **FG model** | `test_ml_day32.py`, `test_ml_day33.py` | Factor-graph decoder v0/v1 |
| **G1 probe** | `test_ml_day55_*` | Linear Ridge CV K-leakage measurement |
| **Infrastructure** | `test_ml_metrics.py`, `test_ml_splits.py`, `test_ml_day27_*` | Metrics, splits, DEM graph |
| **Physics** | `test_physics_engine.py` | Noise models, Pauli channels |
| **Experiments** | `experiment_day69_*`, `experiment_day70_*`, `experiment_day75_*` | Full reproduction scripts |

## Expected Results

```
tests/ — 21 files, 200+ individual tests
Expected: all PASSED
Typical runtime: ~30s (unit tests only)
```

## Experiment Scripts (Longer Runs)

The `experiment_day*.py` files are full reproduction scripts that require GPU and significant compute time. See `REPRODUCIBILITY.md` and `scripts/reproduce/` for instructions.
