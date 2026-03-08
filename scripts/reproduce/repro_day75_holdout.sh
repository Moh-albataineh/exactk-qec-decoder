#!/usr/bin/env bash
# Reproduce Day 75 — V1.0 Holdout validation (d=7, seeds 60000-60009)
# Usage: bash scripts/reproduce/repro_day75_holdout.sh
# Requires GPU with >= 16 GB VRAM
set -euo pipefail

echo "============================================="
echo "  ExactK V1.0 — Reproduce Day 75 (Holdout)"
echo "============================================="
echo ""

echo "Mode: FULL (10 holdout seeds, GPU required)"
echo "Distance: 7"
echo "Effective B: 256 (micro=64, grad_accum=4)"
echo ""

# Experiment writes to its hardcoded artifact dir
ARTIFACT_DIR="ml_artifacts/day75_holdout_d7_v1"
LOGDIR="ml_artifacts/repro_day75_holdout"
mkdir -p "$LOGDIR"

echo "[1/3] Running holdout experiment..."
python tests/experiment_day75_holdout_d7.py 2>&1 | tee "$LOGDIR/run.log"

echo ""
echo "[2/3] Running selector v6..."
python scripts/run_selector_v6_from_jsonl.py \
    --artifact_dir "$ARTIFACT_DIR" \
    2>&1 | tee -a "$LOGDIR/run.log"

echo ""
echo "[3/3] Computing V1.0 KPIs..."
python scripts/day75_2_compute_v1_release_kpis.py \
    --artifact_dir "$ARTIFACT_DIR" \
    --out_dir "$ARTIFACT_DIR/v1_release_closure" \
    2>&1 | tee -a "$LOGDIR/run.log"

echo ""
echo "Results written to: $ARTIFACT_DIR/"
echo "Run log: $LOGDIR/run.log"
echo ""
echo "============================================="
echo "  Day 75 holdout reproduction complete"
echo "============================================="
