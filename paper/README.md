# Paper Companion — ExactK V1.0

This directory contains artifacts for reproducing the results presented in the paper.

## Directory Structure

```
paper/
├── README.md                    # This file
├── claims_traceability.md       # Claim → script → docs mapping
├── figures/                     # Placeholder for generated figures
└── tables/                      # Placeholder for exported tables
```

## How to Use

1. **Verify claims**: See `claims_traceability.md` for a mapping of each paper claim to its reproduction script and test.

2. **Reproduce figures**: Run the experiment scripts (see `../REPRODUCIBILITY.md`), then use the generated artifacts in `ml_artifacts/` to create plots.

3. **Export tables**: KPI results are exported as CSV by `scripts/day75_2_compute_v1_release_kpis.py` → `ml_artifacts/day75_2_v1_release_closure/v1_release_kpis.csv`.

## Key Scripts

| Script | What it produces |
|--------|-----------------|
| `scripts/reproduce/repro_day69_d5.sh` | d=5 per-seed G1 results |
| `scripts/reproduce/repro_day70_d7.sh` | d=7 OOD per-seed results |
| `scripts/reproduce/repro_day75_holdout.sh` | Holdout validation data |
| `scripts/day75_2_compute_v1_release_kpis.py` | V1.0 KPI JSON/CSV/markdown |
