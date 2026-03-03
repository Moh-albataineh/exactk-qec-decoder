#!/usr/bin/env bash
# Reproduce Day 69 — ExactK d=5 results
# Usage: bash scripts/reproduce/repro_day69_d5.sh [--full]
set -euo pipefail

FULL=false
if [[ "${1:-}" == "--full" ]]; then FULL=true; fi

echo "============================================="
echo "  ExactK V1.0 — Reproduce Day 69 (d=5)"
echo "============================================="
echo ""

if $FULL; then
    echo "Mode: FULL (10 seeds, ~2 hours)"
    SEEDS="47000 49000 49200 50000 51000 52000 53000 54000 55000 56000"
else
    echo "Mode: QUICK (2 seeds, ~10 min)"
    SEEDS="47000 49000"
fi

echo "Seeds: $SEEDS"
echo "Distance: 5"
echo "Config: ExactK ΔK=0, λ=0.10, margin=0.30"
echo ""

OUTDIR="ml_artifacts/repro_day69_d5"
mkdir -p "$OUTDIR"

echo "[1/2] Running experiment..."
python tests/experiment_day69_exactk_tuned.py \
    --d 5 \
    --shots 4096 \
    --epochs 12 \
    --seeds $SEEDS \
    --output-dir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/run.log"

echo ""
echo "[2/2] Results written to: $OUTDIR/"
echo ""
echo "============================================="
echo "  Day 69 reproduction complete"
echo "============================================="
