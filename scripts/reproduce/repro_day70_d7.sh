#!/usr/bin/env bash
# Reproduce Day 70 — ExactK d=7 OOD generalization
# Usage: bash scripts/reproduce/repro_day70_d7.sh
# Requires GPU with >= 16 GB VRAM
set -euo pipefail

echo "============================================="
echo "  ExactK V1.0 — Reproduce Day 70 (d=7 OOD)"
echo "============================================="
echo ""

echo "Mode: FULL (10 seeds, GPU required)"
echo "Distance: 7"
echo "Effective B: 256 (micro=64, grad_accum=4)"
echo ""

OUTDIR="ml_artifacts/repro_day70_d7"
mkdir -p "$OUTDIR"

echo "[1/2] Running experiment..."
python tests/experiment_day70_exactk_d7_generalization.py 2>&1 | tee "$OUTDIR/run.log"

echo ""
echo "[2/2] Results written to: $OUTDIR/"
echo ""
echo "============================================="
echo "  Day 70 reproduction complete"
echo "============================================="
