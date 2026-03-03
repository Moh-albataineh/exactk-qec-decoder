#!/usr/bin/env bash
# Run Selector v6 + V1.0 KPI computation from existing JSONL/receipts
# Usage: bash scripts/reproduce/run_selector_v6.sh [RESULTS_DIR]
set -euo pipefail

RESULTS_DIR="${1:-ml_artifacts/repro_day75_holdout}"

echo "============================================="
echo "  ExactK V1.0 — Selector v6 + KPI Pipeline"
echo "============================================="
echo ""
echo "Results dir: $RESULTS_DIR"
echo ""

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: $RESULTS_DIR does not exist."
    echo "Run repro_day75_holdout.sh first."
    exit 1
fi

echo "[1/3] Running selector v6 on JSONL logs..."
python scripts/run_selector_v6_from_jsonl.py \
    --artifact_dir "$RESULTS_DIR"
echo ""

echo "[2/3] Regenerating receipts..."
python scripts/day75_3_regenerate_receipts.py \
    --artifact_dir "$RESULTS_DIR"
echo ""

echo "[3/3] Computing V1.0 release KPIs..."
OUTDIR="$RESULTS_DIR/v1_release_closure"
mkdir -p "$OUTDIR"
python scripts/day75_2_compute_v1_release_kpis.py \
    --artifact_dir "$RESULTS_DIR" \
    --out_dir "$OUTDIR"
echo ""

echo "Output files:"
ls -la "$OUTDIR/"
echo ""
echo "============================================="
echo "  Selector v6 + KPI pipeline complete"
echo "============================================="
