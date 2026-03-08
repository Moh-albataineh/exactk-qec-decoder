#!/usr/bin/env bash
# Reproduce Day 69 — ExactK d=5 results
# Usage: bash scripts/reproduce/repro_day69_d5.sh
set -euo pipefail

echo "============================================="
echo "  ExactK V1.0 — Reproduce Day 69 (d=5)"
echo "============================================="
echo ""

echo "Mode: FULL (10 seeds)"
echo "Distance: 5"
echo "Config: ExactK ΔK=0, λ=0.10, margin=0.30"
echo ""

OUTDIR="ml_artifacts/repro_day69_d5"
mkdir -p "$OUTDIR"

echo "[1/2] Running experiment..."
python tests/experiment_day69_exactk_tuned.py 2>&1 | tee "$OUTDIR/run.log"

echo ""
echo "[2/2] Results written to: $OUTDIR/"
echo ""
echo "============================================="
echo "  Day 69 reproduction complete"
echo "============================================="
