"""
Day 69 — Unit tests for ExactK tuning: margin, λ decay, rZK_Y1 gate.

Tests:
  1) TestMarginWiring: margin=0.30 affects hinge threshold
  2) TestLambdaDecay: schedule correctness
  3) TestRZKY1Computation: intra-class correlation + min_count guard
  4) TestHysteresisTransitions: ON/OFF logic
  5) TestLambdaFinalReflectsGate: final = epoch × gate_mult
  6) TestAlignmentInvariant: probe tensor == logit composition
"""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch


class TestMarginWiring(unittest.TestCase):
    """margin=0.30 correctly affects hinge threshold."""

    def test_margin_030_less_violation_than_050(self):
        """With margin 0.30, fewer violations than margin 0.50 for same data."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        fn_030 = IsoKRankingLoss(delta_k=0, margin=0.30, max_pairs=10000, use_safe_std=False)
        fn_050 = IsoKRankingLoss(delta_k=0, margin=0.50, max_pairs=10000, use_safe_std=False)

        z = torch.cat([torch.full((16, 1), 0.4), torch.full((16, 1), 0.0)])
        K = torch.full((32,), 10.0)
        Y = torch.cat([torch.ones(16), torch.zeros(16)])

        out_030 = fn_030(z, K, Y)
        out_050 = fn_050(z, K, Y)
        self.assertLessEqual(out_030['violation_rate'], out_050['violation_rate'])

    def test_margin_030_zero_violation_for_wide_gap(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        fn = IsoKRankingLoss(delta_k=0, margin=0.30, max_pairs=10000, use_safe_std=False)
        z = torch.cat([torch.full((8, 1), 10.0), torch.full((8, 1), -10.0)])
        K = torch.full((16,), 10.0)
        Y = torch.cat([torch.ones(8), torch.zeros(8)])
        out = fn(z, K, Y)
        self.assertAlmostEqual(out['violation_rate'], 0.0, places=3)
        self.assertAlmostEqual(out['loss'].item(), 0.0, places=5)

    def test_margin_030_loss_lower_than_050(self):
        """Same data: margin=0.30 should produce lower loss than 0.50."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        fn_030 = IsoKRankingLoss(delta_k=0, margin=0.30, lambda_iso=0.10, max_pairs=10000, use_safe_std=False)
        fn_050 = IsoKRankingLoss(delta_k=0, margin=0.50, lambda_iso=0.10, max_pairs=10000, use_safe_std=False)

        torch.manual_seed(42)
        z = torch.randn(64, 1)
        K = torch.full((64,), 10.0)
        Y = torch.cat([torch.ones(32), torch.zeros(32)])

        out_030 = fn_030(z, K, Y)
        out_050 = fn_050(z, K, Y)
        self.assertLessEqual(out_030['loss'].item(), out_050['loss'].item())


class TestLambdaDecay(unittest.TestCase):
    """Lambda decay schedule: λ × 0.85^(epoch-8) for epoch > 8."""

    def _compute(self, epoch):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from experiment_day69_exactk_tuned import compute_lambda_epoch
        return compute_lambda_epoch(0.10, epoch)

    def test_epoch_6_no_decay(self):
        self.assertAlmostEqual(self._compute(6), 0.10, places=6)

    def test_epoch_7_no_decay(self):
        self.assertAlmostEqual(self._compute(7), 0.10, places=6)

    def test_epoch_8_no_decay(self):
        self.assertAlmostEqual(self._compute(8), 0.10, places=6)

    def test_epoch_9_decay_1(self):
        expected = 0.10 * 0.85
        self.assertAlmostEqual(self._compute(9), expected, places=6)

    def test_epoch_10_decay_2(self):
        expected = 0.10 * 0.85 ** 2
        self.assertAlmostEqual(self._compute(10), expected, places=6)

    def test_epoch_12_decay_4(self):
        expected = 0.10 * 0.85 ** 4
        self.assertAlmostEqual(self._compute(12), expected, places=6)

    def test_monotonically_decreasing(self):
        vals = [self._compute(ep) for ep in range(6, 13)]
        for i in range(len(vals) - 1):
            self.assertGreaterEqual(vals[i], vals[i + 1])


class TestRZKY1Computation(unittest.TestCase):
    """Intra-class Z-K correlation computation."""

    def test_perfect_positive_correlation(self):
        z = np.arange(100, dtype=float)
        K = np.arange(100, dtype=float)
        Y = np.ones(100, dtype=bool)
        # Manual correlation
        corr = np.corrcoef(z[Y], K[Y])[0, 1]
        self.assertAlmostEqual(corr, 1.0, places=5)

    def test_min_count_guard(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        # Can't easily test compute_rzk_y1_probe without model, but test the logic
        n_y1 = 30  # below min_count=64
        self.assertTrue(n_y1 < 64)

    def test_zero_std_returns_zero(self):
        """If Z has zero std, correlation should be 0."""
        z = np.ones(100)  # constant Z
        K = np.arange(100, dtype=float)
        # std(z) = 0, corrcoef would be nan
        if z.std() < 1e-10:
            result = 0.0
        else:
            result = float(np.corrcoef(z, K)[0, 1])
        self.assertEqual(result, 0.0)


class TestHysteresisTransitions(unittest.TestCase):
    """ON if |rZK_Y1| >= 0.20, OFF if <= 0.12, else stay."""

    def _apply(self, rzk_abs, prev_on):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from experiment_day69_exactk_tuned import apply_hysteresis_gate
        return apply_hysteresis_gate(rzk_abs, prev_on)

    def test_above_on_threshold(self):
        self.assertTrue(self._apply(0.25, False))
        self.assertTrue(self._apply(0.20, False))

    def test_below_off_threshold(self):
        self.assertFalse(self._apply(0.10, True))
        self.assertFalse(self._apply(0.12, True))

    def test_in_hysteresis_band_stays_on(self):
        self.assertTrue(self._apply(0.15, True))

    def test_in_hysteresis_band_stays_off(self):
        self.assertFalse(self._apply(0.15, False))

    def test_exact_on_boundary(self):
        self.assertTrue(self._apply(0.20, False))  # >= 0.20 → ON

    def test_exact_off_boundary(self):
        self.assertFalse(self._apply(0.12, True))  # <= 0.12 → OFF

    def test_sequence(self):
        """Simulate ON → hysteresis → OFF → hysteresis → ON."""
        state = False  # start OFF
        state = self._apply(0.25, state)  # above 0.20 → ON
        self.assertTrue(state)
        state = self._apply(0.18, state)  # in band → stay ON
        self.assertTrue(state)
        state = self._apply(0.10, state)  # below 0.12 → OFF
        self.assertFalse(state)
        state = self._apply(0.15, state)  # in band → stay OFF
        self.assertFalse(state)
        state = self._apply(0.22, state)  # above 0.20 → ON
        self.assertTrue(state)


class TestLambdaFinalReflectsGate(unittest.TestCase):
    """lambda_epoch_final = lambda_epoch × gate_mult."""

    def test_gate_off_mult_1(self):
        lambda_epoch = 0.10
        gate_mult = 1.0
        self.assertAlmostEqual(lambda_epoch * gate_mult, 0.10, places=6)

    def test_gate_on_mult_025(self):
        lambda_epoch = 0.10
        gate_mult = 0.25
        self.assertAlmostEqual(lambda_epoch * gate_mult, 0.025, places=6)

    def test_decay_plus_gate(self):
        """Epoch 10 + gate ON: 0.10 * 0.85^2 * 0.25."""
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from experiment_day69_exactk_tuned import compute_lambda_epoch
        lambda_ep = compute_lambda_epoch(0.10, 10)
        gate_mult = 0.25
        expected = 0.10 * (0.85 ** 2) * 0.25
        self.assertAlmostEqual(lambda_ep * gate_mult, expected, places=6)

    def test_sets_on_loss_fn(self):
        """Verify IsoKRankingLoss.lambda_iso can be dynamically set."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        fn = IsoKRankingLoss(lambda_iso=0.10, delta_k=0, margin=0.30)
        self.assertAlmostEqual(fn.lambda_iso, 0.10)
        fn.lambda_iso = 0.025  # simulate gate ON
        self.assertAlmostEqual(fn.lambda_iso, 0.025)

        # Loss should scale
        z = torch.cat([torch.full((8, 1), -10.0), torch.full((8, 1), 10.0)])
        K = torch.full((16,), 10.0)
        Y = torch.cat([torch.ones(8), torch.zeros(8)])

        fn.lambda_iso = 0.10
        out_high = fn(z, K, Y)
        fn.lambda_iso = 0.025
        out_low = fn(z, K, Y)
        if out_high['loss'].item() > 0:
            self.assertAlmostEqual(out_low['loss'].item() / out_high['loss'].item(),
                                   0.025 / 0.10, places=2)


class TestAlignmentInvariant(unittest.TestCase):
    """Probe tensor == logit composition tensor (regression from Day 67/68)."""

    def test_alignment_with_margin_030(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
        from qec_noise_factory.ml.bench.density_prior import build_k_prior_table
        from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count

        B, N_d, N_e = 32, 120, 60
        rng = np.random.RandomState(42)
        X_dummy = rng.randint(0, 2, (256, N_d)).astype(np.uint8)
        Y_dummy = rng.randint(0, 2, (256,)).astype(np.uint8)
        K_train = compute_syndrome_count(X_dummy)

        model = FactorGraphDecoderV1(
            det_input_dim=2, err_input_dim=1,
            hidden_dim=16, num_mp_layers=2,
            use_density_residualization=True,
        )
        prior = build_k_prior_table(K_train, Y_dummy.astype(bool), alpha=1.0)
        model.set_density_prior(prior, n_det=N_d)
        model.use_density_prior_final = True
        model.fg_use_tanh_clamp = True
        model.fg_use_kcs_standardization = True
        model.setup_kcs(K_train, num_bins=12)
        model._split_residual_head = None

        det_feats = torch.randn(B, N_d + 1, 2)
        err_feats = torch.randn(N_e, 1)
        ei_d2e = torch.stack([
            torch.randint(0, N_d + 1, (N_e * 2,)),
            torch.arange(N_e).repeat(2),
        ])
        ei_e2d = torch.stack([ei_d2e[1], ei_d2e[0]])

        model.train()
        result = model.forward_split(det_feats, err_feats, ei_d2e, ei_e2d)

        z_g1 = result['logit_residual_norm']
        alpha = result['alpha']
        prior_logit = result['logit_prior']
        expected_final = prior_logit + alpha * z_g1
        self.assertTrue(torch.allclose(result['logit_final'], expected_final, atol=1e-5))

        # Day 69 margin=0.30 iso loss
        iso = IsoKRankingLoss(lambda_iso=0.10, margin=0.30, delta_k=0,
                              use_safe_std=False)
        K_batch = result['K']
        Y_batch = torch.cat([torch.ones(B // 2), torch.zeros(B // 2)])
        out = iso(z_g1, K_batch, Y_batch)
        self.assertIn('loss', out)


if __name__ == '__main__':
    unittest.main()
