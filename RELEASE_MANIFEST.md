# Release Manifest — ExactK V1.0

## Included (Whitelist)

| Directory/File | Description |
|---------------|-------------|
| `qec_noise_factory/ml/` | Full ML pipeline (models, bench, data, diagnostics, eval, graph, metrics, ops, stim, train) |
| `qec_noise_factory/physics/` | Noise models (Pauli channels) |
| `qec_noise_factory/factory/` | Stim circuit builders (subset: circuits_qc, noise_models, circuit_demo) |
| `qec_noise_factory/verify/` | Quality gates (data_quality, reason_codes) |
| `qec_noise_factory/utils/` | Hashing utilities |
| `tests/` | 21 selected test files (ExactK, KPI, FG, infrastructure) |
| `scripts/` | 4 analytics scripts + 5 reproduction scripts |
| `docs/` | DAYS.md, RESULTS.md, METHODS.md, LIMITATIONS.md, MLOPS_POLICY.md |
| `context_pack/` | CURRENT_STATUS.md, SCHEMAS.md |
| `paper/` | Claims traceability, figure/table placeholders |

## Excluded (Always)

| Pattern | Reason |
|---------|--------|
| `ml_artifacts/` | Generated at runtime |
| `*.pt`, `*.pth`, `*.ckpt` | Model checkpoints |
| `*.jsonl` | Training logs |
| `*.bin` | Binary shard data |
| `shards/`, `packs/generated/` | Large data dumps |
| `output/` | Generated output |
| `__pycache__/` | Python bytecode cache |
| `tasks_archive/` | Internal archives |
| `.env`, `.env.local` | Environment secrets |
| `google-genai` | Private dependency (removed from requirements.txt) |
| `qec_noise_factory/orchestrator/` | Factory infrastructure (not needed for ML) |
| `qec_noise_factory/sandbox/` | Docker execution (not needed for ML) |
| `qec_noise_factory/jobs/` | Job management (not needed for ML) |
| `qec_noise_factory/memory/` | SQLite provenance DB (not needed for ML) |

## Security Scan Results

| Check | Status |
|-------|--------|
| API keys / secrets | ✅ None found |
| Personal paths (`C:\Users\...`) | ✅ None in code files (docs may contain historical references) |
| `.pt` / `.pth` files | ✅ None included |
| `ml_artifacts/` | ✅ Not included |
| Files > 50 MB | ✅ None |
| `google-genai` dependency | ✅ Removed from requirements.txt |

## Size Audit

All files are source code, documentation, or shell scripts. No binary artifacts included.
