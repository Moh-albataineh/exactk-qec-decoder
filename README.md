# ExactK: Iso-$K$ Hinge Loss for Robust K-Leakage Reduction in QEC Neural Decoders

An auxiliary training loss for quantum error correction neural decoders that eliminates syndrome-count confounding by constraining ranking comparisons to sample pairs with identical syndrome count ($\Delta K = 0$).

---

## AI-Assisted Development Disclosure

This repository was developed through a heavily AI-assisted workflow directed by the author. Large language model tools were used extensively for planning support, code generation and revision, debugging, documentation drafting, manuscript editing, and LaTeX/PDF preparation. The author's role was to direct the workflow, review outputs, validate final results used in the release, integrate the final artifacts, and assume responsibility for the released claims.

See [`docs/AI_DISCLOSURE.md`](docs/AI_DISCLOSURE.md) for details.

---

## Status

This repository is a **research artifact and paper companion** for the ExactK manuscript. It contains:

- The complete source code for the bipartite factor-graph decoder and ExactK loss
- Reproduction scripts for all key experiments (d=5, d=7, holdout)
- The final manuscript LaTeX source and figures
- 75 days of development documentation and provenance

---

## Main Results (V1.0 Holdout — Day 75)

| KPI | Value | Target | Status |
|-----|-------|--------|--------|
| Science Δ (epoch-median G₁) | **+45.0%** | ≥ 20% | Pass |
| Safe Yield (Prod CLEAN) | **80%** (8/10) | ≥ 80% | Pass |
| TOPO_FAIL | **0/10** | ≤ 10% | Pass |
| KPI-B: Leaky cohort improvement | **+15.1%** (83% wins) | > 0 | Pass |
| Do-No-Harm violations | **0** | 0 | Pass |

Validated on 10 unseen holdout seeds (60000–60009) at d=7, p=0.04, correlated crosstalk noise.

**Prior to ExactK**, 14 mitigation mechanisms across 5 families (global decorrelation, null-space alignment, K-orthogonalization, gradient shielding, architectural factorization) were tested over 28 development days. None reliably reduced K-leakage while preserving topology signal across seeds at the required effect size.

---

## What is ExactK?

Syndrome count $K$ (the number of triggered detectors) is a strong statistical predictor of logical error probability in surface codes. Neural decoders can exploit this as a shortcut, encoding density information into representations intended for spatial topology — a phenomenon we call **K-leakage**.

ExactK addresses this by adding an auxiliary hinge-margin ranking loss that mines training pairs exclusively from samples sharing identical $K$. Because within-pair $K$ variation is zero by construction, the model can only improve on this loss by learning genuine spatial structure, not syndrome-count shortcuts.

Key properties:
- **No gradient surgery** — operates entirely in the forward pass
- **No architectural modification** — a pair-mining step within each batch
- **No adversarial training** — no min-max optimization
- **Strict $\Delta K = 0$ constraint** — relaxing to $\Delta K = 1$ reintroduces confounding

---

## Quick Start

### Installation

```bash
git clone https://github.com/Moh-albataineh/exactk-qec-decoder.git
cd exactk-qec-decoder
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Reproduce Key Results

| Experiment | Command | Time (GPU) |
|------------|---------|------------|
| d=5 (10 seeds) | `bash scripts/reproduce/repro_day69_d5.sh` | ~10-15 min |
| d=7 OOD (10 seeds) | `bash scripts/reproduce/repro_day70_d7.sh` | ~4 hours |
| Holdout (10 seeds) | `bash scripts/reproduce/repro_day75_holdout.sh` | ~4 hours |
| Selector + KPIs | `bash scripts/reproduce/run_selector_v6.sh` | ~1 min |

> **Note**: Canonical results are from the validated reference environment. Re-running on different hardware may yield numerically different results due to floating-point non-determinism. A reproduction is successful if it preserves the same verdict, safety invariants, and qualitatively similar effect size. See [`docs/REPRO_POLICY.md`](docs/REPRO_POLICY.md).

---

## Paper

**Final PDF**: [`paper/ExactK_Final.pdf`](paper/ExactK_Final.pdf)

The manuscript LaTeX source is in [`paper/overleaf_package/`](paper/overleaf_package/):

| File | Description |
|------|-------------|
| `paper/overleaf_package/main.tex` | Document root |
| `paper/overleaf_package/sections/` | All section `.tex` files |
| `paper/overleaf_package/figures/` | Final figure PDFs |
| `paper/overleaf_package/references.bib` | Bibliography |
| `paper/plot_figures.py` | Figure generation script |
| `paper/compute_bootstrap_cis.py` | Bootstrap CI computation |
| `paper/claims_traceability.md` | Claim → evidence mapping |

---

## Repository Layout

See [`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md) for the full annotated tree. Key directories:

| Directory | Purpose |
|-----------|---------|
| `qec_noise_factory/` | Core Python package (decoder, training, evaluation) |
| `tests/` | 18 test files, 200+ tests |
| `scripts/reproduce/` | Shell scripts for reproducing key experiments |
| `paper/overleaf_package/` | Final canonical LaTeX source |
| `docs/` | Methods, results, limitations, AI disclosure, provenance |

---

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/DAYS.md`](docs/DAYS.md) | Full daily log — 75 days of development |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Benchmark results + V1.0 release KPIs |
| [`docs/METHODS.md`](docs/METHODS.md) | Technical methods: ExactK, selector v6, MLOps |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | Known constraints and caveats |
| [`docs/AI_DISCLOSURE.md`](docs/AI_DISCLOSURE.md) | AI-assisted development disclosure |
| [`docs/REPRO_POLICY.md`](docs/REPRO_POLICY.md) | Canonical vs reproduction run policy |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Seeds, hardware, reproduction guide |

---

## How to Cite

```bibtex
@software{albataineh2026exactk,
  title   = {ExactK: Iso-K Hinge Loss for K-Leakage-Free QEC Decoding},
  author  = {Albataineh, Mohammad},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/Moh-albataineh/exactk-qec-decoder}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

---

## License

MIT — see [`LICENSE`](LICENSE).
