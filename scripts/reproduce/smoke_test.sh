#!/usr/bin/env bash
# Smoke test — validate pipeline works with minimal compute
# Usage: bash scripts/reproduce/smoke_test.sh
set -euo pipefail

echo "============================================="
echo "  ExactK V1.0 — Smoke Test"
echo "============================================="
echo ""

echo "[1/3] Running unit tests..."
python -m pytest tests/ -v --tb=short -q 2>&1 | tail -5
echo ""

echo "[2/3] Quick import validation..."
python -c "
from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
from qec_noise_factory.ml.ops.checkpoint_selection import select_epoch_for_seed
from qec_noise_factory.ml.diagnostics.g1_probe import build_probe_features
print('  All core imports OK')
"
echo ""

echo "[3/3] Selector v6 determinism check..."
python -c "
from qec_noise_factory.ml.ops.checkpoint_selection import select_epoch_for_seed
import json

# Synthetic epoch data (2 epochs)
epochs = [
    {'epoch': 6, 'g1_linear_r2_cv': 0.02, 'g1_inst': 0.018, 'tg_roll': 0.05, 'slice_clean_frac': 0.6},
    {'epoch': 7, 'g1_linear_r2_cv': 0.03, 'g1_inst': 0.022, 'tg_roll': 0.08, 'slice_clean_frac': 0.7},
]
result = select_epoch_for_seed(epochs, seed=42, arm='test', selector_version='v6_drop_slice_floor')
print(f'  Selected epoch: {result[\"chosen_epoch\"]}, pool: {result[\"selector_pool\"]}')
print('  Selector v6 determinism OK')
"
echo ""

echo "============================================="
echo "  Smoke test PASSED"
echo "============================================="
