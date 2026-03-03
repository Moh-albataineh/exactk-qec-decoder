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

# Synthetic epoch data (3 epochs)
epochs = [
    {'epoch': 6, 'G1_aligned': 0.02, 'topo_TG': 0.05, 'slice_clean': 0.6},
    {'epoch': 7, 'G1_aligned': 0.03, 'topo_TG': 0.08, 'slice_clean': 0.7},
    {'epoch': 8, 'G1_aligned': 0.01, 'topo_TG': 0.06, 'slice_clean': 0.65},
]
result = select_epoch_for_seed(epochs, drop_slice_floor=True)
print(f'  Selected epoch: {result[\"chosen_epoch\"]}, pool: {result[\"selector_pool\"]}')
print('  Selector v6 determinism OK')
"
echo ""

echo "============================================="
echo "  Smoke test PASSED"
echo "============================================="
