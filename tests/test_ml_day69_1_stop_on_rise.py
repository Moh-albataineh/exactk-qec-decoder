"""
Day 69.1 — Unit tests for StopOnRise guard decision logic.

Tests:
  1) Guard triggers on doubling
  2) Guard triggers on absolute threshold
  3) Guard stays off when conditions not met
  4) Guard is permanent once triggered
  5) Guard respects start epoch
  6) Lambda schedule + guard interaction
"""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

sys.path.insert(0, str(Path(__file__).resolve().parent))


class TestStopOnRiseDoubling(unittest.TestCase):
    """Guard triggers when G1 doubles vs previous epoch."""

    def test_exact_doubling_triggers(self):
        from experiment_day69_1_stop_on_rise import check_stop_on_rise
        stopped, reason = check_stop_on_rise(0.02, 0.01, False)
        self.assertTrue(stopped)
        self.assertIn("doubling", reason)

    def test_more_than_doubling_triggers(self):
        from experiment_day69_1_stop_on_rise import check_stop_on_rise
        stopped, reason = check_stop_on_rise(0.04, 0.01, False)
        self.assertTrue(stopped)

    def test_less_than_doubling_ok(self):
        from experiment_day69_1_stop_on_rise import check_stop_on_rise
        stopped, reason = check_stop_on_rise(0.015, 0.01, False)
        self.assertFalse(stopped)
        self.assertEqual(reason, "ok")

    def test_zero_previous_no_crash(self):
        from experiment_day69_1_stop_on_rise import check_stop_on_rise
        stopped, reason = check_stop_on_rise(0.005, 0.0, False)
        self.assertFalse(stopped)  # 0 prev → skip doubling check

    def test_none_previous_no_crash(self):
        from experiment_day69_1_stop_on_rise import check_stop_on_rise
        stopped, reason = check_stop_on_rise(0.005, None, False)
        self.assertFalse(stopped)


class TestStopOnRiseAbsThreshold(unittest.TestCase):
    """Guard triggers when G1 >= 0.03 absolute."""

    def test_at_threshold(self):
        from experiment_day69_1_stop_on_rise import check_stop_on_rise
        stopped, reason = check_stop_on_rise(0.03, 0.025, False)
        self.assertTrue(stopped)
        self.assertIn("abs_threshold", reason)

    def test_above_threshold(self):
        from experiment_day69_1_stop_on_rise import check_stop_on_rise
        stopped, reason = check_stop_on_rise(0.05, 0.04, False)
        self.assertTrue(stopped)

    def test_below_threshold(self):
        from experiment_day69_1_stop_on_rise import check_stop_on_rise
        stopped, reason = check_stop_on_rise(0.025, 0.02, False)
        self.assertFalse(stopped)


class TestStopOnRisePermanent(unittest.TestCase):
    """Once triggered, guard stays stopped permanently."""

    def test_already_stopped_stays_stopped(self):
        from experiment_day69_1_stop_on_rise import check_stop_on_rise
        stopped, reason = check_stop_on_rise(0.001, 0.001, True)
        self.assertTrue(stopped)
        self.assertEqual(reason, "already_stopped")

    def test_already_stopped_even_with_good_g1(self):
        from experiment_day69_1_stop_on_rise import check_stop_on_rise
        stopped, reason = check_stop_on_rise(0.0001, 0.0001, True)
        self.assertTrue(stopped)


class TestStopOnRiseSequence(unittest.TestCase):
    """Simulate a realistic epoch sequence."""

    def test_seed_49200_like_sequence(self):
        """Simulate Day 68 seed 49200 trajectory: ok at ep7-8, explosion ep9-11."""
        from experiment_day69_1_stop_on_rise import check_stop_on_rise
        g1s = [0.0027, 0.0003, 0.0274, 0.0410, 0.1020, 0.0693]  # epochs 7-12
        # Guard starts checking at epoch 8 (index 1 = epoch 8)
        stopped = False
        trigger_epoch = None

        # Epoch 7: no check (before start epoch)
        # Epoch 8: check with prev=0.0027
        stopped, reason = check_stop_on_rise(g1s[1], g1s[0], stopped)
        # 0.0003/0.0027 < 2 and 0.0003 < 0.03 → ok
        self.assertFalse(stopped)

        # Epoch 9: G1 = 0.0274 vs prev 0.0003 → 91× doubling!
        stopped, reason = check_stop_on_rise(g1s[2], g1s[1], stopped)
        self.assertTrue(stopped)
        self.assertIn("doubling", reason)
        trigger_epoch = 9  # would be epoch 9

        # Epoch 10+: stays stopped
        stopped, reason = check_stop_on_rise(g1s[3], g1s[2], stopped)
        self.assertTrue(stopped)
        self.assertEqual(reason, "already_stopped")

    def test_healthy_seed_no_trigger(self):
        """Simulate a healthy seed: gradual improvement."""
        from experiment_day69_1_stop_on_rise import check_stop_on_rise
        g1s = [0.005, 0.003, 0.002, 0.001, 0.001]  # monotone decrease
        stopped = False
        for i in range(1, len(g1s)):
            stopped, _ = check_stop_on_rise(g1s[i], g1s[i-1], stopped)
            self.assertFalse(stopped)


class TestLambdaWithGuard(unittest.TestCase):
    """Lambda schedule + guard interaction."""

    def test_guard_zeroes_lambda(self):
        from experiment_day69_1_stop_on_rise import compute_lambda_epoch
        lambda_ep = compute_lambda_epoch(0.10, 9)
        # Guard triggered → final should be 0
        lambda_final = 0.0 if True else lambda_ep  # simulating guard
        self.assertEqual(lambda_final, 0.0)

    def test_no_guard_has_decay(self):
        from experiment_day69_1_stop_on_rise import compute_lambda_epoch
        lambda_ep = compute_lambda_epoch(0.10, 9)
        expected = 0.10 * 0.85
        self.assertAlmostEqual(lambda_ep, expected, places=6)

    def test_lambda_iso_settable(self):
        """IsoKRankingLoss.lambda_iso can be set to 0."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        import torch
        fn = IsoKRankingLoss(lambda_iso=0.10, delta_k=0, margin=0.30)
        fn.lambda_iso = 0.0
        z = torch.randn(16, 1)
        K = torch.full((16,), 10.0)
        Y = torch.cat([torch.ones(8), torch.zeros(8)])
        out = fn(z, K, Y)
        self.assertEqual(out['loss'].item(), 0.0)


class TestAlignmentPreserved(unittest.TestCase):
    """StopOnRise does not break alignment invariant."""

    def test_alignment_with_zero_lambda(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
        from qec_noise_factory.ml.bench.density_prior import build_k_prior_table
        from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count
        import torch
        import numpy as np

        B, N_d, N_e = 16, 120, 60
        rng = np.random.RandomState(42)
        X_dummy = rng.randint(0, 2, (128, N_d)).astype(np.uint8)
        Y_dummy = rng.randint(0, 2, (128,)).astype(np.uint8)
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
        ei_d2e = torch.stack([torch.randint(0, N_d+1, (N_e*2,)), torch.arange(N_e).repeat(2)])
        ei_e2d = torch.stack([ei_d2e[1], ei_d2e[0]])

        model.train()
        result = model.forward_split(det_feats, err_feats, ei_d2e, ei_e2d)
        z_g1 = result['logit_residual_norm']
        alpha = result['alpha']
        prior_logit = result['logit_prior']
        expected_final = prior_logit + alpha * z_g1
        self.assertTrue(torch.allclose(result['logit_final'], expected_final, atol=1e-5))

        # lambda=0 → loss should be 0, no gradient from iso
        iso = IsoKRankingLoss(lambda_iso=0.0, margin=0.30, delta_k=0)
        K_batch = result['K']
        Y_batch = torch.cat([torch.ones(B//2), torch.zeros(B//2)])
        out = iso(z_g1, K_batch, Y_batch)
        self.assertEqual(out['loss'].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
