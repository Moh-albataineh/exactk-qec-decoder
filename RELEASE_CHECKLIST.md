# Release Checklist — ExactK V1.0

## Definition of Done

### Code
- [x] Source code whitelist-copied (ml, physics, factory subset, verify, utils)
- [x] No `__pycache__` directories
- [x] `requirements.txt` cleaned (no broken packages, no private deps)

### Tests
- [x] 21 test files included
- [ ] All tests pass: `python -m pytest tests/ -v`

### Documentation
- [x] `docs/DAYS.md` — final patched (Day 75.3 included)
- [x] `docs/RESULTS.md` — KPI-A corrected, "Spike violations (Prod)" clarified
- [x] `docs/METHODS.md` — Days 1–75 complete
- [x] `docs/LIMITATIONS.md` — Days 1–75 complete
- [x] `docs/MLOPS_POLICY.md` — selector v6, JSONL WAL, receipt schema
- [x] `context_pack/CURRENT_STATUS.md` — Day 75.3

### Release Metadata
- [x] `README.md` — project overview, quickstart, results, citation
- [x] `LICENSE` — MIT
- [x] `CITATION.cff` — v1.0.0
- [x] `CHANGELOG.md` — v1.0.0 entry
- [x] `CONTRIBUTING.md` — contribution guide
- [x] `SECURITY.md` — minimal policy
- [x] `TESTING.md` — test commands
- [x] `REPRODUCIBILITY.md` — seeds, hardware, instructions
- [x] `.gitignore` — excludes artifacts

### Release Guides
- [x] `REPO_SPLIT_GUIDE.md` — git commands for new repo + subtree
- [x] `RELEASE_MANIFEST.md` — include/exclude list + security scan
- [x] `RELEASE_CHECKLIST.md` — this file

### Paper Artifacts
- [x] `paper/claims_traceability.md` — claim → evidence mapping
- [x] `paper/README.md` — paper companion guide
- [x] `paper/figures/.gitkeep` — placeholder
- [x] `paper/tables/.gitkeep` — placeholder

### Reproduction Scripts
- [x] `scripts/reproduce/smoke_test.sh`
- [x] `scripts/reproduce/repro_day69_d5.sh`
- [x] `scripts/reproduce/repro_day70_d7.sh`
- [x] `scripts/reproduce/repro_day75_holdout.sh`
- [x] `scripts/reproduce/run_selector_v6.sh`

### Security / Privacy
- [ ] No API keys in any file
- [ ] No personal absolute paths in `.py` files
- [ ] No `.pt` / `.pth` files
- [ ] No `ml_artifacts/` directory
- [ ] No files > 50 MB
- [ ] `google-genai` not in requirements.txt

### Size Audit
- [ ] Total size < 20 MB (code + docs only)
