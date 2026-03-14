# Repository Layout

```
exactk-qec-decoder/
├── README.md                    # Project overview and quick start
├── LICENSE                      # MIT license
├── CITATION.cff                 # Machine-readable citation metadata
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # Contribution guidelines
├── SECURITY.md                  # Security policy
├── TESTING.md                   # Test commands and categories
├── REPRODUCIBILITY.md           # Seeds, hardware, reproduction guide
├── requirements.txt             # Python dependencies
│
├── qec_noise_factory/           # Core Python package
│   ├── ml/                      # ML decoder pipeline
│   │   ├── models/              #   Factor-graph, GNN, MLP decoders
│   │   ├── ops/                 #   Checkpoint selection (selector v6)
│   │   ├── diagnostics/         #   G1 probe, K-leakage diagnostics
│   │   ├── graph/               #   DEM bipartite graph builder
│   │   ├── train/               #   Training loop
│   │   ├── eval/                #   Evaluation utilities
│   │   └── bench/               #   Benchmarking tools
│   ├── physics/                 # Noise models (SD6, SI1000, correlated)
│   ├── factory/                 # Stim circuit builders
│   ├── verify/                  # Quality gates
│   └── utils/                   # Hashing utilities
│
├── tests/                       # Test suite (18 files, 200+ tests)
│   ├── experiment_day*.py       #   Key experiment scripts
│   └── test_ml_day*.py          #   Unit tests per development day
│
├── scripts/                     # Analytics and reproduction
│   ├── reproduce/               #   Shell scripts for key experiments
│   └── *.py                     #   Selector, KPI computation
│
├── paper/                       # Paper companion artifacts
│   ├── overleaf_package/        #   Final LaTeX source (canonical)
│   │   ├── main.tex             #     Document root
│   │   ├── references.bib       #     Bibliography
│   │   ├── figures/             #     Final figure PDFs
│   │   └── sections/            #     Section .tex files
│   ├── plot_figures.py          #   Figure generation script
│   ├── compute_bootstrap_cis.py #   Bootstrap CI computation
│   ├── claims_traceability.md   #   Claim → evidence mapping
│   ├── references.bib           #   Bibliography (top-level copy)
│   └── tables/                  #     Bootstrap CI JSON
│
├── docs/                        # Documentation
│   ├── AI_DISCLOSURE.md         #   AI-assisted development disclosure
│   ├── DAYS.md                  #   Full daily development log (75 days)
│   ├── RESULTS.md               #   Benchmark + V1.0 release KPIs
│   ├── METHODS.md               #   Technical methods
│   ├── LIMITATIONS.md           #   Known constraints
│   ├── MLOPS_POLICY.md          #   Production deployment policy
│   ├── REPRO_POLICY.md          #   Canonical vs reproduction run policy
│   └── REPO_LAYOUT.md           #   This file
```
