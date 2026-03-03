#!/usr/bin/env bash
# Reproduce Day 75 — V1.0 Holdout validation (d=7, seeds 60000-60009)
# Usage: bash scripts/reproduce/repro_day75_holdout.sh [--full]
# Requires GPU with ≥ 16 GB VRAM
set -euo pipefail

FULL=false
if [[ "${1:-}" == "--full" ]]; then FULL=true; fi

echo "============================================="
echo "  ExactK V1.0 — Reproduce Day 75 (Holdout)"
echo "============================================="
echo ""

if $FULL; then
    echo "Mode: FULL (10 holdout seeds, ~4 hours, GPU required)"
    SEEDS="60000 60001 60002 60003 60004 60005 60006 60007 60008 60009"
else
    echo "Mode: QUICK (2 holdout seeds, ~45 min, GPU recommended)"
    SEEDS="60000 60001"
fi

echo "Seeds: $SEEDS"
echo "Distance: 7"
echo "Effective B: 256 (micro=64, grad_accum=4)"
echo ""

OUTDIR="ml_artifacts/repro_day75_holdout"
mkdir -p "$OUTDIR"

echo "[1/3] Running holdout experiment..."
python tests/experiment_day75_holdout_d7.py \
    --d 7 \
    --shots 4096 \
    --epochs 12 \
    --micro-batch 64 \
    --grad-accum 4 \
    --seeds $SEEDS \
    --output-dir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/run.log"

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
