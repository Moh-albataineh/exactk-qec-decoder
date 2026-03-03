"""
Day 68 — Unit tests for IsoKRankingLoss ExactK enhancements.

Tests:
  1) ExactK mining correctness: only ΔK=0 pairs, labels correct
  2) No-pair batch behavior: iso-loss=0, no crash
  3) Pair cap enforcement: max_pairs respected
  4) SafeStd correctness: std is detached, logits use raw Z_g1
  5) Alignment invariant regression: probe consumes correct tensor
"""
from __future__ import annotations

import unittest
import numpy as np
import torch


class TestExactKMiningCorrectness(unittest.TestCase):
    """Only ΔK=0 pairs should be mined; ΔK=1 must be excluded."""

    def test_exact_k_only_same_K_pairs(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0, max_pairs=10000, use_safe_std=True)
        z = torch.randn(20, 1)
        K = torch.full((20,), 10.0)
        Y = torch.cat([torch.ones(10), torch.zeros(10)])
        out = loss_fn(z, K, Y)
        self.assertFalse(out['no_pair'])
        self.assertEqual(out['pair_count_total'], 100)  # 10 pos × 10 neg

    def test_exact_k_rejects_delta_1(self):
        """ΔK=1 pairs must NOT be found when delta_k=0."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0, max_pairs=10000)
        z = torch.randn(20, 1)
        # pos has K=10, neg has K=11 → ΔK=1 → should find 0 pairs
        K = torch.cat([torch.full((10,), 10.0), torch.full((10,), 11.0)])
        Y = torch.cat([torch.ones(10), torch.zeros(10)])
        out = loss_fn(z, K, Y)
        self.assertTrue(out['no_pair'])
        self.assertEqual(out['pair_count_total'], 0)

    def test_exact_k_mixed_K_bins(self):
        """Multiple K bins, only same-K pairs counted."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0, max_pairs=10000)
        # K=5: 3 pos, 2 neg → 6 pairs; K=10: 2 pos, 3 neg → 6 pairs; K=15: 0 pairs
        z = torch.randn(15, 1)
        K = torch.tensor([5,5,5,  10,10,   5,5,  10,10,10,  15,15,15,15,15], dtype=torch.float)
        Y = torch.tensor([1,1,1,   1, 1,   0, 0,  0, 0, 0,   0, 0, 0, 0, 0], dtype=torch.float)
        out = loss_fn(z, K, Y)
        self.assertFalse(out['no_pair'])
        self.assertEqual(out['pair_count_total'], 12)  # 6 + 6

    def test_labels_correct_pos_first(self):
        """Verify pos z > neg z induces zero violation for wide gap."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0, margin=0.50, lambda_iso=0.10)
        z = torch.cat([torch.full((8, 1), 10.0), torch.full((8, 1), -10.0)])
        K = torch.full((16,), 10.0)
        Y = torch.cat([torch.ones(8), torch.zeros(8)])
        out = loss_fn(z, K, Y)
        self.assertAlmostEqual(out['violation_rate'], 0.0, places=3)
        self.assertAlmostEqual(out['loss'].item(), 0.0, places=5)


class TestNoPairBatchBehavior(unittest.TestCase):
    """No-pair scenarios: loss=0, no crash, no_pair=True."""

    def test_all_positive(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0)
        z = torch.randn(16, 1)
        K = torch.full((16,), 10.0)
        Y = torch.ones(16)
        out = loss_fn(z, K, Y)
        self.assertTrue(out['no_pair'])
        self.assertEqual(out['loss'].item(), 0.0)

    def test_all_negative(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0)
        z = torch.randn(16, 1)
        K = torch.full((16,), 10.0)
        Y = torch.zeros(16)
        out = loss_fn(z, K, Y)
        self.assertTrue(out['no_pair'])

    def test_disjoint_K_bins(self):
        """Pos and neg in different K bins → no pairs."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0)
        z = torch.randn(16, 1)
        K = torch.cat([torch.full((8,), 5.0), torch.full((8,), 15.0)])
        Y = torch.cat([torch.ones(8), torch.zeros(8)])
        out = loss_fn(z, K, Y)
        self.assertTrue(out['no_pair'])
        self.assertEqual(out['loss'].item(), 0.0)

    def test_no_pair_has_telemetry(self):
        """Even with no pairs, telemetry keys are present."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0)
        z = torch.randn(16, 1)
        K = torch.full((16,), 10.0)
        Y = torch.ones(16)
        out = loss_fn(z, K, Y)
        for key in ['z_var_global', 'z_var_intra_k', 'unique_bins_with_pairs']:
            self.assertIn(key, out)


class TestPairCapEnforcement(unittest.TestCase):
    """max_pairs must be respected."""

    def test_cap_limits_used_pairs(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0, max_pairs=10)
        z = torch.randn(64, 1)
        K = torch.full((64,), 10.0)
        Y = torch.cat([torch.ones(32), torch.zeros(32)])
        out = loss_fn(z, K, Y)
        self.assertEqual(out['pair_count_used'], 10)
        self.assertGreater(out['pair_count_total'], 10)

    def test_cap_256(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0, max_pairs=256)
        z = torch.randn(128, 1)
        K = torch.full((128,), 10.0)
        Y = torch.cat([torch.ones(64), torch.zeros(64)])
        out = loss_fn(z, K, Y)
        self.assertLessEqual(out['pair_count_used'], 256)

    def test_under_cap_uses_all(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        loss_fn = IsoKRankingLoss(delta_k=0, max_pairs=10000)
        z = torch.randn(10, 1)
        K = torch.full((10,), 10.0)
        Y = torch.cat([torch.ones(5), torch.zeros(5)])
        out = loss_fn(z, K, Y)
        self.assertEqual(out['pair_count_used'], 25)  # 5 × 5
        self.assertEqual(out['pair_count_total'], 25)


class TestSafeStdCorrectness(unittest.TestCase):
    """SafeStd: std is detached. Base: uses raw Z."""

    def test_safe_std_detaches_scale(self):
        """In SafeStd mode, changing Z magnitude doesn't affect normalized gap direction."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        fn1 = IsoKRankingLoss(delta_k=0, margin=0.50, use_safe_std=True, max_pairs=10000)
        fn2 = IsoKRankingLoss(delta_k=0, margin=0.50, use_safe_std=True, max_pairs=10000)

        K = torch.full((16,), 10.0)
        Y = torch.cat([torch.ones(8), torch.zeros(8)])

        # Small scale
        z_small = torch.cat([torch.full((8, 1), 0.5), torch.full((8, 1), -0.5)])
        out_small = fn1(z_small, K, Y)

        # Large scale (10× magnitude)
        z_large = torch.cat([torch.full((8, 1), 5.0), torch.full((8, 1), -5.0)])
        out_large = fn2(z_large, K, Y)

        # SafeStd should produce similar violation rates regardless of scale
        self.assertAlmostEqual(out_small['violation_rate'], out_large['violation_rate'], places=2)

    def test_base_mode_is_scale_sensitive(self):
        """In Base mode (no std), large scale Z has fewer violations."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        fn_base = IsoKRankingLoss(delta_k=0, margin=0.50, use_safe_std=False, max_pairs=10000)

        K = torch.full((16,), 10.0)
        Y = torch.cat([torch.ones(8), torch.zeros(8)])

        # Large positive gap → should have 0% violations
        z_big = torch.cat([torch.full((8, 1), 10.0), torch.full((8, 1), -10.0)])
        out = fn_base(z_big, K, Y)
        self.assertAlmostEqual(out['violation_rate'], 0.0, places=3)

    def test_safe_std_gradient_flows(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        fn = IsoKRankingLoss(delta_k=0, margin=0.50, use_safe_std=True)
        z = torch.randn(32, 1, requires_grad=True)
        K = torch.full((32,), 10.0)
        Y = torch.cat([torch.ones(16), torch.zeros(16)])
        out = fn(z, K, Y)
        out['loss'].backward()
        self.assertIsNotNone(z.grad)

    def test_base_gradient_flows(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        fn = IsoKRankingLoss(delta_k=0, margin=0.50, use_safe_std=False)
        z = torch.randn(32, 1, requires_grad=True)
        K = torch.full((32,), 10.0)
        Y = torch.cat([torch.ones(16), torch.zeros(16)])
        out = fn(z, K, Y)
        out['loss'].backward()
        self.assertIsNotNone(z.grad)

    def test_new_telemetry_keys_present(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        for use_safe in [True, False]:
            fn = IsoKRankingLoss(delta_k=0, use_safe_std=use_safe)
            z = torch.randn(32, 1)
            K = torch.full((32,), 10.0)
            Y = torch.cat([torch.ones(16), torch.zeros(16)])
            out = fn(z, K, Y)
            for key in ['z_var_global', 'z_var_intra_k', 'unique_bins_with_pairs']:
                self.assertIn(key, out, f"Missing {key} with use_safe_std={use_safe}")


class TestAlignmentInvariant(unittest.TestCase):
    """Probe consumes exact logit_residual_norm tensor from forward_split."""

    def test_probe_tensor_matches_logit_composition(self):
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
        self.assertTrue(torch.allclose(result['logit_final'], expected_final, atol=1e-5),
                        "Alignment invariant FAILED: logit_final != prior + alpha * z_g1")

        # ExactK IsoKRankingLoss on z_g1
        K_batch = result['K']
        Y_batch = torch.cat([torch.ones(B // 2), torch.zeros(B // 2)])

        for use_safe_std in [True, False]:
            iso = IsoKRankingLoss(lambda_iso=0.10, margin=0.50, delta_k=0,
                                  use_safe_std=use_safe_std)
            out = iso(z_g1, K_batch, Y_batch)
            self.assertIn('loss', out)

            # Gradient flows back to model
            if not out['no_pair']:
                out['loss'].backward(retain_graph=True)
                has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                              for p in model.parameters() if p.requires_grad)
                self.assertTrue(has_grad,
                                f"No gradient flow with use_safe_std={use_safe_std}")
                model.zero_grad()


if __name__ == '__main__':
    unittest.main()
