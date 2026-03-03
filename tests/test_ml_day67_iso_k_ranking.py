"""
Day 67 — Unit tests for IsoKRankingLoss.

Tests:
  - Pair mining: exact-K, |ΔK|<=1, cap enforcement, no-pair, class-missing
  - Z-standardization: detached scale, eps safety
  - Gradient flow, alignment invariant
  - Warmup no-op (tested at experiment level)
"""
from __future__ import annotations

import unittest
import numpy as np
import torch


class TestIsoKRankingLossKeys(unittest.TestCase):
    """Output shape and key checks."""

    def test_output_keys(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=1)
        z = torch.randn(32, 1)
        K = torch.randint(5, 15, (32,)).float()
        Y = torch.cat([torch.ones(16), torch.zeros(16)])
        out = loss_fn(z, K, Y)
        for key in ['loss', 'pair_count_total', 'pair_count_used',
                     'no_pair', 'violation_rate', 'zgap_mean', 'zgap_median']:
            self.assertIn(key, out, f"Missing key: {key}")

    def test_loss_is_scalar(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=1)
        z = torch.randn(32, 1)
        K = torch.randint(5, 15, (32,)).float()
        Y = torch.cat([torch.ones(16), torch.zeros(16)])
        out = loss_fn(z, K, Y)
        self.assertEqual(out['loss'].shape, ())


class TestIsoKRankingLossPairMiningExactK(unittest.TestCase):
    """Exact-K pair mining (delta_k=0)."""

    def test_exact_k_finds_pairs(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0, max_pairs=1000)
        # All samples have K=10, mix of pos/neg
        z = torch.randn(20, 1)
        K = torch.full((20,), 10.0)
        Y = torch.cat([torch.ones(10), torch.zeros(10)])
        out = loss_fn(z, K, Y)
        self.assertFalse(out['no_pair'])
        self.assertEqual(out['pair_count_total'], 100)  # 10 * 10

    def test_exact_k_no_pairs_different_K(self):
        """No pairs when pos and neg have different K."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0)
        z = torch.randn(20, 1)
        K = torch.cat([torch.full((10,), 5.0), torch.full((10,), 15.0)])
        Y = torch.cat([torch.ones(10), torch.zeros(10)])  # pos=K5, neg=K15
        out = loss_fn(z, K, Y)
        self.assertTrue(out['no_pair'])
        self.assertEqual(out['loss'].item(), 0.0)


class TestIsoKRankingLossPairMiningNearK(unittest.TestCase):
    """|ΔK|<=1 pair mining."""

    def test_near_k_finds_more_pairs(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        exact = IsoKRankingLoss(delta_k=0, max_pairs=10000)
        near = IsoKRankingLoss(delta_k=1, max_pairs=10000)
        z = torch.randn(40, 1)
        K = torch.arange(40).float() % 5  # K ∈ {0,1,2,3,4}
        Y = torch.cat([torch.ones(20), torch.zeros(20)])
        out_exact = exact(z, K, Y)
        out_near = near(z, K, Y)
        self.assertGreaterEqual(out_near['pair_count_total'],
                                out_exact['pair_count_total'])

    def test_delta_k_1_includes_adjacent(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=1, max_pairs=10000)
        z = torch.randn(10, 1)
        # Pos: K=10, Neg: K=11 → |ΔK|=1 should pair
        K = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0,
                          11.0, 11.0, 11.0, 11.0, 11.0])
        Y = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0,
                          0.0, 0.0, 0.0, 0.0, 0.0])
        out = loss_fn(z, K, Y)
        self.assertFalse(out['no_pair'])
        self.assertEqual(out['pair_count_total'], 25)  # 5 * 5


class TestIsoKRankingLossCapEnforcement(unittest.TestCase):
    """Pair cap enforcement."""

    def test_cap_limits_pairs(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=1, max_pairs=10)
        z = torch.randn(64, 1)
        K = torch.full((64,), 10.0)
        Y = torch.cat([torch.ones(32), torch.zeros(32)])
        out = loss_fn(z, K, Y)
        self.assertEqual(out['pair_count_used'], 10)
        self.assertGreater(out['pair_count_total'], 10)


class TestIsoKRankingLossNoPairSafety(unittest.TestCase):

    def test_all_positive(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss()
        z = torch.randn(16, 1)
        K = torch.randint(0, 10, (16,)).float()
        Y = torch.ones(16)  # no negatives
        out = loss_fn(z, K, Y)
        self.assertTrue(out['no_pair'])
        self.assertEqual(out['loss'].item(), 0.0)

    def test_all_negative(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss()
        z = torch.randn(16, 1)
        K = torch.randint(0, 10, (16,)).float()
        Y = torch.zeros(16)  # no positives
        out = loss_fn(z, K, Y)
        self.assertTrue(out['no_pair'])

    def test_single_sample_each(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0)
        z = torch.tensor([[1.0], [-1.0]])
        K = torch.tensor([10.0, 10.0])
        Y = torch.tensor([1.0, 0.0])
        out = loss_fn(z, K, Y)
        self.assertFalse(out['no_pair'])
        self.assertEqual(out['pair_count_total'], 1)


class TestIsoKRankingLossKShapes(unittest.TestCase):
    """K and Y can be [B] or [B,1]."""

    def test_K_2d_Y_2d(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=1)
        z = torch.randn(16, 1)
        K = torch.randint(5, 10, (16, 1)).float()
        Y = torch.cat([torch.ones(8, 1), torch.zeros(8, 1)])
        out = loss_fn(z, K, Y)
        self.assertIn('loss', out)


class TestIsoKRankingLossGradient(unittest.TestCase):

    def test_gradient_flows(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=1, lambda_iso=0.10, margin=0.50)
        z = torch.randn(32, 1, requires_grad=True)
        K = torch.full((32,), 10.0)
        Y = torch.cat([torch.ones(16), torch.zeros(16)])
        out = loss_fn(z, K, Y)
        out['loss'].backward()
        self.assertIsNotNone(z.grad)

    def test_no_gradient_when_no_pairs(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0)
        z = torch.randn(16, 1, requires_grad=True)
        K = torch.cat([torch.full((8,), 5.0), torch.full((8,), 15.0)])
        Y = torch.cat([torch.ones(8), torch.zeros(8)])  # pos=K5, neg=K15
        out = loss_fn(z, K, Y)
        # No pairs → detached zero
        self.assertTrue(out['no_pair'])
        self.assertFalse(out['loss'].requires_grad)


class TestIsoKRankingLossViolationRate(unittest.TestCase):

    def test_perfect_ranking(self):
        """When pos z > neg z by more than margin, violation_rate should be 0."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0, margin=0.50, lambda_iso=0.10)
        # Pos z = 10, Neg z = -10 → gap >> margin
        z = torch.cat([torch.full((8, 1), 10.0), torch.full((8, 1), -10.0)])
        K = torch.full((16,), 10.0)
        Y = torch.cat([torch.ones(8), torch.zeros(8)])
        out = loss_fn(z, K, Y)
        self.assertAlmostEqual(out['violation_rate'], 0.0, places=3)
        self.assertAlmostEqual(out['loss'].item(), 0.0, places=5)

    def test_inverted_ranking(self):
        """When neg z > pos z, violation_rate should be 1."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0, margin=0.50, lambda_iso=0.10)
        z = torch.cat([torch.full((8, 1), -10.0), torch.full((8, 1), 10.0)])
        K = torch.full((16,), 10.0)
        Y = torch.cat([torch.ones(8), torch.zeros(8)])
        out = loss_fn(z, K, Y)
        self.assertAlmostEqual(out['violation_rate'], 1.0, places=3)
        self.assertGreater(out['loss'].item(), 0.0)


class TestIsoKRankingLossAlignment(unittest.TestCase):
    """Verify iso-K loss uses exact same tensor as logit_final."""

    def test_iso_k_on_aligned_z_g1(self):
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
        K_batch = result['K']
        Y_batch = torch.cat([torch.ones(B // 2), torch.zeros(B // 2)])

        # Verify alignment
        alpha = result['alpha']
        prior_logit = result['logit_prior']
        expected_final = prior_logit + alpha * z_g1
        self.assertTrue(torch.allclose(result['logit_final'], expected_final, atol=1e-5))

        # Apply IsoKRankingLoss
        iso = IsoKRankingLoss(lambda_iso=0.10, margin=0.50, delta_k=1)
        out = iso(z_g1, K_batch, Y_batch)
        self.assertIn('loss', out)

        # Gradient flows from iso loss through z_g1 to model
        if not out['no_pair']:
            out['loss'].backward()
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in model.parameters() if p.requires_grad)
            self.assertTrue(has_grad)


if __name__ == '__main__':
    unittest.main()
