#!/usr/bin/env bash
# Reproduce Day 70 — ExactK d=7 OOD generalization
# Usage: bash scripts/reproduce/repro_day70_d7.sh [--full]
# Requires GPU with ≥ 16 GB VRAM
set -euo pipefail

FULL=false
if [[ "${1:-}" == "--full" ]]; then FULL=true; fi

echo "============================================="
echo "  ExactK V1.0 — Reproduce Day 70 (d=7 OOD)"
echo "============================================="
echo ""

if $FULL; then
    echo "Mode: FULL (10 seeds, ~4 hours, GPU required)"
    SEEDS="47000 49000 49200 50000 51000 52000 53000 54000 55000 56000"
else
    echo "Mode: QUICK (2 seeds, ~45 min, GPU recommended)"
    SEEDS="47000 49000"
fi

echo "Seeds: $SEEDS"
echo "Distance: 7"
echo "Effective B: 256 (micro=64, grad_accum=4)"
echo ""

OUTDIR="ml_artifacts/repro_day70_d7"
mkdir -p "$OUTDIR"

echo "[1/2] Running experiment..."
python tests/experiment_day70_exactk_d7_generalization.py \
    --d 7 \
    --shots 4096 \
    --epochs 12 \
    --micro-batch 64 \
    --grad-accum 4 \
    --seeds $SEEDS \
    --output-dir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/run.log"

echo ""
echo "[2/2] Results written to: $OUTDIR/"
echo ""
echo "============================================="
echo "  Day 70 reproduction complete"
echo "============================================="
