"""
Day 55 — G1 Probe Stabilization Tests

Tests:
  1. Determinism: same (seed, probe_seed) → identical r2_mean/std
  2. No-grad: probe features have requires_grad=False
  3. Feature parity: linear probe uses same feature builder as MLP
  4. Linear probe: correlated data → high R², independent → low R²
  5. Linear probe: small input → returns zero
  6. MLP telemetry: returns dict with expected keys
  7. ProbeSet isolation: probe_seed differs from train seed
  8. ProbeSet determinism: same seed → same data
  9. evaluate_g1: returns all required fields
  10. evaluate_g1: pass threshold logic
  11. Linear CV stability: fold R²s have low variance
  12. Strict detach: X_np is float64 numpy, not tensor
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class TestDeterminism:
    def test_same_seed_same_result(self):
        """Same (seed, probe_seed) → identical linear probe results."""
        from qec_noise_factory.ml.diagnostics.g1_probe import run_linear_probe_cv
        rng = np.random.RandomState(42)
        X = rng.randn(200, 1)
        y = rng.randn(200)

        r1 = run_linear_probe_cv(X, y, cv=5, seed=100)
        r2 = run_linear_probe_cv(X, y, cv=5, seed=100)
        assert r1["r2_mean"] == r2["r2_mean"]
        assert r1["r2_std"] == r2["r2_std"]
        assert r1["r2_score"] == r2["r2_score"]
        assert r1["fold_r2s"] == r2["fold_r2s"]


class TestNoGrad:
    def test_features_detached(self):
        """build_probe_features returns numpy (no gradients possible)."""
        from qec_noise_factory.ml.diagnostics.g1_probe import build_probe_features
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1

        model = FactorGraphDecoderV1(det_input_dim=2, err_input_dim=1,
                                     output_dim=1, hidden_dim=16, num_mp_layers=1)
        # Create minimal tensor inputs
        det = torch.randn(4, 10, 2)
        err = torch.randn(5, 1)
        ei = torch.zeros(2, 0, dtype=torch.long)

        X_np = build_probe_features(model, det, err, ei, ei)

        assert isinstance(X_np, np.ndarray)
        assert X_np.dtype == np.float64
        # Verify no torch tensor
        assert not isinstance(X_np, torch.Tensor)


class TestFeatureParity:
    def test_same_feature_builder(self):
        """Linear probe uses same forward_split → logit_residual_norm."""
        from qec_noise_factory.ml.diagnostics.g1_probe import build_probe_features
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1

        model = FactorGraphDecoderV1(det_input_dim=2, err_input_dim=1,
                                     output_dim=1, hidden_dim=16, num_mp_layers=1)
        det = torch.randn(4, 10, 2)
        err = torch.randn(5, 1)
        ei = torch.zeros(2, 0, dtype=torch.long)

        X_np = build_probe_features(model, det, err, ei, ei)

        # Compare with direct forward_split
        model.eval()
        with torch.no_grad():
            split = model.forward_split(det, err, ei, ei)
            z_direct = split['logit_residual_norm'].detach().cpu().numpy().reshape(-1, 1)

        np.testing.assert_allclose(X_np, z_direct.astype(np.float64), atol=1e-6)


class TestLinearProbe:
    def test_correlated_high_r2(self):
        """Correlated data → high R²."""
        from qec_noise_factory.ml.diagnostics.g1_probe import run_linear_probe_cv
        rng = np.random.RandomState(42)
        K = rng.uniform(0, 20, size=500)
        z = 0.5 * K + rng.randn(500) * 0.1  # strong linear correlation
        result = run_linear_probe_cv(z.reshape(-1, 1), K, cv=5, seed=42)
        assert result["r2_mean"] > 0.9

    def test_independent_low_r2(self):
        """Independent data → near-zero R²."""
        from qec_noise_factory.ml.diagnostics.g1_probe import run_linear_probe_cv
        rng = np.random.RandomState(42)
        z = rng.randn(500, 1)
        K = rng.randn(500)
        result = run_linear_probe_cv(z, K, cv=5, seed=42)
        assert result["r2_score"] < 0.05

    def test_small_input_returns_zero(self):
        """Tiny input → zero R²."""
        from qec_noise_factory.ml.diagnostics.g1_probe import run_linear_probe_cv
        result = run_linear_probe_cv(np.zeros((3, 1)), np.zeros(3), cv=5, seed=42)
        assert result["r2_mean"] == 0.0
        assert result["r2_score"] == 0.0


class TestMLPTelemetry:
    def test_returns_expected_keys(self):
        from qec_noise_factory.ml.diagnostics.g1_probe import run_mlp_telemetry
        rng = np.random.RandomState(42)
        X = rng.randn(100, 1)
        K = rng.randn(100)
        result = run_mlp_telemetry(X, K, seed=42)
        assert "diag_r2_mlp" in result
        assert "diag_r2_linear_simple" in result
        assert "nonlinear_leakage_suspected" in result


class TestProbeSet:
    def test_isolation_seed(self):
        """ProbeSet uses different seed from training seed."""
        from qec_noise_factory.ml.diagnostics.g1_probe import generate_probe_set
        seed = 47000
        probe = generate_probe_set(distance=3, p=0.04, basis="X",
                                   n_probe=100, seed=seed)
        assert probe["probe_seed"] == seed + 99991
        assert probe["probe_seed"] != seed

    def test_determinism(self):
        """Same seed → same ProbeSet."""
        from qec_noise_factory.ml.diagnostics.g1_probe import generate_probe_set
        p1 = generate_probe_set(distance=3, p=0.04, basis="X",
                                n_probe=100, seed=47000)
        p2 = generate_probe_set(distance=3, p=0.04, basis="X",
                                n_probe=100, seed=47000)
        np.testing.assert_array_equal(p1["X_raw"], p2["X_raw"])
        np.testing.assert_array_equal(p1["K"], p2["K"])


class TestEvaluateG1:
    def test_returns_all_fields(self):
        from qec_noise_factory.ml.diagnostics.g1_probe import evaluate_g1
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
        # Minimal setup
        model = FactorGraphDecoderV1(det_input_dim=2, err_input_dim=1,
                                     output_dim=1, hidden_dim=16, num_mp_layers=1)
        probe = {
            "det_feats": torch.randn(50, 10, 2),
            "err_feats": torch.randn(5, 1),
            "ei_d2e": torch.zeros(2, 0, dtype=torch.long),
            "ei_e2d": torch.zeros(2, 0, dtype=torch.long),
            "err_w": None, "obs_mask": None,
            "K": np.random.randint(0, 10, size=50).astype(np.float64),
            "probe_seed": 99991,
        }
        result = evaluate_g1(model, probe, cv=3)
        assert "G1_pass" in result
        assert "G1_r2_linear_mean" in result
        assert "G1_r2_linear_std" in result
        assert "G1_r2_linear_score" in result
        assert "diag_r2_mlp" in result
        assert "probe_n" in result
        assert "ridge_alpha" in result
        assert "fold_r2s" in result

    def test_pass_threshold(self):
        from qec_noise_factory.ml.diagnostics.g1_probe import run_linear_probe_cv, THRESH_G1
        rng = np.random.RandomState(42)
        z = rng.randn(500, 1)
        K = rng.randn(500)
        result = run_linear_probe_cv(z, K, cv=5, seed=42)
        # Independent → score should be near 0, below threshold
        assert result["r2_score"] <= THRESH_G1 or result["r2_score"] < 0.05


class TestCVStability:
    def test_fold_r2s_low_variance(self):
        """CV folds should have consistent R² on well-behaved data."""
        from qec_noise_factory.ml.diagnostics.g1_probe import run_linear_probe_cv
        rng = np.random.RandomState(42)
        K = rng.uniform(0, 20, size=1000)
        z = 0.3 * K + rng.randn(1000) * 2.0
        result = run_linear_probe_cv(z.reshape(-1, 1), K, cv=5, seed=42)
        fold_r2s = result["fold_r2s"]
        assert len(fold_r2s) == 5
        # Variance across folds should be moderate
        assert np.std(fold_r2s) < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
