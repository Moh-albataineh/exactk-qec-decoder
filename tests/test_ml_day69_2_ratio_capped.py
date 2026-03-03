"""
Day 69.2 — Unit tests for RatioCapped per-batch iso pressure cap.

Tests:
  1) Cap triggers when iso > 20% of BCE
  2) Cap does not trigger when iso <= 20%
  3) Cap scales iso correctly
  4) Gradient flows through capped iso
  5) Zero/tiny BCE edge cases
  6) Lambda schedule still works with cap
  7) Alignment invariant preserved
"""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np


class TestCapTriggering(unittest.TestCase):
    """Cap triggers when iso_weighted > cap_ratio * bce."""

    def test_triggers_above_threshold(self):
        from experiment_day69_2_ratio_capped import apply_ratio_cap
        iso = torch.tensor(0.50, requires_grad=True)
        bce = 1.0  # cap = 0.20
        capped, triggered, pre, post = apply_ratio_cap(iso, bce, 0.20)
        self.assertTrue(triggered)
        self.assertAlmostEqual(float(capped.detach()), 0.20, places=3)

    def test_does_not_trigger_below(self):
        from experiment_day69_2_ratio_capped import apply_ratio_cap
        iso = torch.tensor(0.10, requires_grad=True)
        bce = 1.0  # cap = 0.20
        capped, triggered, pre, post = apply_ratio_cap(iso, bce, 0.20)
        self.assertFalse(triggered)
        self.assertAlmostEqual(float(capped.detach()), 0.10, places=3)

    def test_exact_boundary_float32_triggers(self):
        """At exact boundary, float32 tensor is slightly above float64 cap → triggers."""
        from experiment_day69_2_ratio_capped import apply_ratio_cap
        iso = torch.tensor(0.20, requires_grad=True)
        bce = 1.0
        capped, triggered, pre, post = apply_ratio_cap(iso, bce, 0.20)
        # float32(0.20) > float64(0.20) due to precision
        self.assertTrue(triggered)

    def test_large_iso_capped_to_20pct(self):
        from experiment_day69_2_ratio_capped import apply_ratio_cap
        iso = torch.tensor(5.0, requires_grad=True)
        bce = 2.0  # cap = 0.40
        capped, triggered, pre, post = apply_ratio_cap(iso, bce, 0.20)
        self.assertTrue(triggered)
        self.assertAlmostEqual(float(capped.detach()), 0.40, places=3)


class TestCapScaling(unittest.TestCase):
    """Capped iso = ceiling / iso_val * iso_tensor."""

    def test_scale_factor_correct(self):
        from experiment_day69_2_ratio_capped import apply_ratio_cap
        iso = torch.tensor(1.0, requires_grad=True)
        bce = 1.0  # ceiling = 0.20
        capped, _, _, _ = apply_ratio_cap(iso, bce, 0.20)
        self.assertAlmostEqual(float(capped.detach()), 0.20, places=4)

    def test_pre_post_ratio(self):
        from experiment_day69_2_ratio_capped import apply_ratio_cap
        iso = torch.tensor(0.50, requires_grad=True)
        bce = 1.0
        _, _, pre, post = apply_ratio_cap(iso, bce, 0.20)
        self.assertAlmostEqual(pre, 0.50, places=2)
        self.assertAlmostEqual(post, 0.20, places=2)

    def test_uncapped_ratios_equal(self):
        from experiment_day69_2_ratio_capped import apply_ratio_cap
        iso = torch.tensor(0.10, requires_grad=True)
        bce = 1.0
        _, _, pre, post = apply_ratio_cap(iso, bce, 0.20)
        self.assertAlmostEqual(pre, post, places=4)


class TestGradientFlow(unittest.TestCase):
    """Gradient flows through capped iso loss."""

    def test_capped_has_grad(self):
        from experiment_day69_2_ratio_capped import apply_ratio_cap
        iso = torch.tensor(1.0, requires_grad=True)
        bce = 1.0
        capped, _, _, _ = apply_ratio_cap(iso, bce, 0.20)
        capped.backward()
        self.assertIsNotNone(iso.grad)
        self.assertGreater(abs(float(iso.grad)), 0)

    def test_uncapped_has_grad(self):
        from experiment_day69_2_ratio_capped import apply_ratio_cap
        iso = torch.tensor(0.05, requires_grad=True)
        bce = 1.0
        capped, _, _, _ = apply_ratio_cap(iso, bce, 0.20)
        capped.backward()
        self.assertIsNotNone(iso.grad)
        self.assertAlmostEqual(float(iso.grad), 1.0, places=4)

    def test_capped_grad_is_scaled(self):
        from experiment_day69_2_ratio_capped import apply_ratio_cap
        iso = torch.tensor(1.0, requires_grad=True)
        bce = 1.0  # ceiling=0.20, scale=0.20
        capped, _, _, _ = apply_ratio_cap(iso, bce, 0.20)
        capped.backward()
        # grad should be scale factor = 0.20
        self.assertAlmostEqual(float(iso.grad), 0.20, places=2)


class TestEdgeCases(unittest.TestCase):
    """Zero/tiny BCE edge cases."""

    def test_zero_bce_no_crash(self):
        from experiment_day69_2_ratio_capped import apply_ratio_cap
        iso = torch.tensor(0.10, requires_grad=True)
        bce = 0.0
        capped, triggered, _, _ = apply_ratio_cap(iso, bce, 0.20)
        # ceiling = 0, iso > 0 but ceiling is tiny → no trigger (ceiling < 1e-10)
        self.assertFalse(triggered)

    def test_zero_iso_no_trigger(self):
        from experiment_day69_2_ratio_capped import apply_ratio_cap
        iso = torch.tensor(0.0, requires_grad=True)
        bce = 1.0
        capped, triggered, _, _ = apply_ratio_cap(iso, bce, 0.20)
        self.assertFalse(triggered)
        self.assertAlmostEqual(float(capped.detach()), 0.0, places=6)

    def test_tiny_bce(self):
        from experiment_day69_2_ratio_capped import apply_ratio_cap
        iso = torch.tensor(0.001, requires_grad=True)
        bce = 0.001
        capped, triggered, _, _ = apply_ratio_cap(iso, bce, 0.20)
        if triggered:
            self.assertLessEqual(float(capped.detach()), 0.20 * 0.001 + 1e-6)


class TestLambdaWithCap(unittest.TestCase):
    """Lambda schedule still works alongside cap."""

    def test_decay_schedule(self):
        from experiment_day69_2_ratio_capped import compute_lambda_epoch
        self.assertAlmostEqual(compute_lambda_epoch(0.10, 8), 0.10, places=6)
        self.assertAlmostEqual(compute_lambda_epoch(0.10, 9), 0.10 * 0.85, places=6)

    def test_cap_applied_after_lambda(self):
        """iso = lambda * raw_iso_loss, then cap applied."""
        from experiment_day69_2_ratio_capped import apply_ratio_cap, compute_lambda_epoch
        lam = compute_lambda_epoch(0.10, 9)  # 0.085
        raw_iso = 3.0  # large
        iso_weighted = torch.tensor(lam * raw_iso, requires_grad=True)
        bce = 1.0
        capped, triggered, _, _ = apply_ratio_cap(iso_weighted, bce, 0.20)
        self.assertTrue(triggered)
        self.assertAlmostEqual(float(capped.detach()), 0.20, places=3)


class TestAlignmentPreserved(unittest.TestCase):
    """RatioCap does not break alignment invariant."""

    def test_alignment(self):
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
        from qec_noise_factory.ml.bench.density_prior import build_k_prior_table
        from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count

        B, N_d, N_e = 16, 120, 60
        rng = np.random.RandomState(42)
        X = rng.randint(0, 2, (128, N_d)).astype(np.uint8)
        Yd = rng.randint(0, 2, (128,)).astype(np.uint8)
        K_train = compute_syndrome_count(X)

        model = FactorGraphDecoderV1(
            det_input_dim=2, err_input_dim=1,
            hidden_dim=16, num_mp_layers=2,
            use_density_residualization=True,
        )
        prior = build_k_prior_table(K_train, Yd.astype(bool), alpha=1.0)
        model.set_density_prior(prior, n_det=N_d)
        model.use_density_prior_final = True
        model.fg_use_tanh_clamp = True
        model.fg_use_kcs_standardization = True
        model.setup_kcs(K_train, num_bins=12)
        model._split_residual_head = None

        det_feats = torch.randn(B, N_d + 1, 2)
        err_feats = torch.randn(N_e, 1)
        ei_d2e = torch.stack([torch.randint(0, N_d+1, (N_e*2,)), torch.arange(N_e).repeat(2)])
        ei_e2d = torch.stack([ei_d2e[1], ei_d2e[0]])

        model.train()
        result = model.forward_split(det_feats, err_feats, ei_d2e, ei_e2d)
        z_g1 = result['logit_residual_norm']
        alpha = result['alpha']
        expected = result['logit_prior'] + alpha * z_g1
        self.assertTrue(torch.allclose(result['logit_final'], expected, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
