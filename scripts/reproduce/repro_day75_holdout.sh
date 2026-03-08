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

OUTDIR="ml_artifacts/repro_day75_holdout"
mkdir -p "$OUTDIR"

echo "[1/3] Running holdout experiment..."
python tests/experiment_day75_holdout_d7.py 2>&1 | tee "$OUTDIR/run.log"

echo ""
echo "[2/3] Running selector v6..."
python scripts/run_selector_v6_from_jsonl.py \
    --artifact_dir "$OUTDIR" \
    2>&1 | tee -a "$OUTDIR/run.log"

echo ""
echo "[3/3] Computing V1.0 KPIs..."
python scripts/day75_2_compute_v1_release_kpis.py \
    --artifact_dir "$OUTDIR" \
    --out_dir "$OUTDIR/v1_release_closure" \
    2>&1 | tee -a "$OUTDIR/run.log"

echo ""
echo "Results written to: $OUTDIR/"
echo ""
echo "============================================="
echo "  Day 75 holdout reproduction complete"
echo "============================================="
