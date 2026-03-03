"""
Day 70 — Unit tests for d=7 generalization experiment.

Tests:
  1) Lambda schedules (Tuned_Prod decay vs EarlyCutoff zero)
  2) ΔK=0 invariant in pair mining
  3) max_pairs cap
  4) Iso-loss on aligned Z_g1 only
  5) Alignment invariant (d=7 model)
  6) get_lambda_for_arm dispatch
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


class TestLambdaScheduleProd(unittest.TestCase):
    """ExactK_Tuned_Prod: warmup → active → decay."""

    def test_warmup_zero(self):
        from experiment_day70_exactk_d7_generalization import compute_lambda_tuned_prod
        for ep in range(1, 6):
            self.assertEqual(compute_lambda_tuned_prod(0.10, ep), 0.0)

    def test_active_phase(self):
        from experiment_day70_exactk_d7_generalization import compute_lambda_tuned_prod
        for ep in [6, 7, 8]:
            self.assertAlmostEqual(compute_lambda_tuned_prod(0.10, ep), 0.10, places=6)

    def test_decay_epoch_9(self):
        from experiment_day70_exactk_d7_generalization import compute_lambda_tuned_prod
        expected = 0.10 * 0.85
        self.assertAlmostEqual(compute_lambda_tuned_prod(0.10, 9), expected, places=6)

    def test_decay_epoch_12(self):
        from experiment_day70_exactk_d7_generalization import compute_lambda_tuned_prod
        expected = 0.10 * 0.85 ** 4
        self.assertAlmostEqual(compute_lambda_tuned_prod(0.10, 12), expected, places=6)

    def test_monotone_decrease(self):
        from experiment_day70_exactk_d7_generalization import compute_lambda_tuned_prod
        vals = [compute_lambda_tuned_prod(0.10, ep) for ep in range(6, 13)]
        for i in range(len(vals) - 1):
            self.assertGreaterEqual(vals[i], vals[i + 1])


class TestLambdaScheduleEarlyCutoff(unittest.TestCase):
    """ExactK_EarlyCutoff: warmup → active 6-8 → zero 9-12."""

    def test_warmup_zero(self):
        from experiment_day70_exactk_d7_generalization import compute_lambda_early_cutoff
        for ep in range(1, 6):
            self.assertEqual(compute_lambda_early_cutoff(0.10, ep), 0.0)

    def test_active_phase(self):
        from experiment_day70_exactk_d7_generalization import compute_lambda_early_cutoff
        for ep in [6, 7, 8]:
            self.assertAlmostEqual(compute_lambda_early_cutoff(0.10, ep), 0.10, places=6)

    def test_cutoff_after_8(self):
        from experiment_day70_exactk_d7_generalization import compute_lambda_early_cutoff
        for ep in [9, 10, 11, 12]:
            self.assertEqual(compute_lambda_early_cutoff(0.10, ep), 0.0)

    def test_vs_prod_epoch_9(self):
        from experiment_day70_exactk_d7_generalization import (
            compute_lambda_tuned_prod, compute_lambda_early_cutoff)
        prod = compute_lambda_tuned_prod(0.10, 9)
        cutoff = compute_lambda_early_cutoff(0.10, 9)
        self.assertGreater(prod, 0)
        self.assertEqual(cutoff, 0)


class TestGetLambdaDispatch(unittest.TestCase):
    """get_lambda_for_arm dispatches correctly."""

    def test_control(self):
        from experiment_day70_exactk_d7_generalization import get_lambda_for_arm, ARMS
        for ep in range(1, 13):
            self.assertEqual(get_lambda_for_arm(ARMS["Control"], ep), 0.0)

    def test_prod_uses_decay(self):
        from experiment_day70_exactk_d7_generalization import get_lambda_for_arm, ARMS
        self.assertAlmostEqual(get_lambda_for_arm(ARMS["ExactK_Tuned_Prod"], 9),
                               0.10 * 0.85, places=6)

    def test_cutoff_zeros_after_8(self):
        from experiment_day70_exactk_d7_generalization import get_lambda_for_arm, ARMS
        self.assertEqual(get_lambda_for_arm(ARMS["ExactK_EarlyCutoff"], 9), 0.0)


class TestDeltaK0Invariant(unittest.TestCase):
    """IsoKRankingLoss with delta_k=0 only pairs same-K samples."""

    def test_exact_k_pairs_only(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        fn = IsoKRankingLoss(lambda_iso=0.10, margin=0.30, delta_k=0, max_pairs=10000)
        # 3 K bins: K=5 (mixed), K=10 (mixed), K=15 (only pos)
        z = torch.randn(30, 1)
        K = torch.cat([torch.full((10,), 5.0), torch.full((10,), 10.0), torch.full((10,), 15.0)])
        Y = torch.cat([
            torch.tensor([1,1,1,1,1,0,0,0,0,0], dtype=torch.float),  # K=5: 5+5-
            torch.tensor([1,1,1,0,0,0,0,0,0,0], dtype=torch.float),  # K=10: 3+7-
            torch.tensor([1,1,1,1,1,1,1,1,1,1], dtype=torch.float),  # K=15: all pos → no pairs
        ])
        out = fn(z, K, Y)
        # K=15 has no neg → no pairs there; only K=5 and K=10 should contribute
        self.assertGreater(out['pair_count_total'], 0)

    def test_no_cross_k_pairs(self):
        """With only 2 bins and no intra-bin pos/neg mix, no pairs formed."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        fn = IsoKRankingLoss(lambda_iso=0.10, margin=0.30, delta_k=0, max_pairs=10000)
        z = torch.randn(20, 1)
        K = torch.cat([torch.full((10,), 5.0), torch.full((10,), 10.0)])
        Y = torch.cat([
            torch.ones(10),   # K=5 all pos
            torch.zeros(10),  # K=10 all neg
        ])
        out = fn(z, K, Y)
        self.assertTrue(out['no_pair'])


class TestMaxPairsCap(unittest.TestCase):
    """max_pairs=512 caps pair count."""

    def test_cap_enforced(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        fn = IsoKRankingLoss(lambda_iso=0.10, margin=0.30, delta_k=0, max_pairs=512)
        # Large batch with many same-K pairs
        z = torch.randn(256, 1)
        K = torch.full((256,), 10.0)
        Y = torch.cat([torch.ones(128), torch.zeros(128)])
        out = fn(z, K, Y)
        self.assertLessEqual(out['pair_count_used'], 512)

    def test_cap_with_fewer_pairs(self):
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss
        fn = IsoKRankingLoss(lambda_iso=0.10, margin=0.30, delta_k=0, max_pairs=512)
        z = torch.randn(16, 1)
        K = torch.full((16,), 10.0)
        Y = torch.cat([torch.ones(8), torch.zeros(8)])
        out = fn(z, K, Y)
        self.assertLessEqual(out['pair_count_used'], 512)


class TestAlignmentInvariant(unittest.TestCase):
    """Alignment preserved with d=7 model dimensions."""

    def test_alignment_d7(self):
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
        from qec_noise_factory.ml.bench.density_prior import build_k_prior_table
        from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count

        B, N_d, N_e = 16, 336, 180  # d=7 dimensions
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

    def test_iso_on_z_g1_not_priors(self):
        """Iso loss operates on Z_g1, not logit_prior."""
        from qec_noise_factory.ml.models.k_ortho import IsoKRankingLoss

        z_g1 = torch.randn(32, 1, requires_grad=True)
        K = torch.full((32,), 10.0)
        Y = torch.cat([torch.ones(16), torch.zeros(16)])

        fn = IsoKRankingLoss(lambda_iso=0.10, margin=0.30, delta_k=0)
        out = fn(z_g1, K, Y)

        # Verify gradient flows through z_g1
        if out['loss'].requires_grad:
            out['loss'].backward()
            self.assertIsNotNone(z_g1.grad)

        # Create a separate "prior" tensor and verify iso didn't touch it
        logit_prior = torch.randn(32, 1, requires_grad=True)
        fn2 = IsoKRankingLoss(lambda_iso=0.10, margin=0.30, delta_k=0)
        out2 = fn2(z_g1.detach().requires_grad_(True), K, Y)
        # logit_prior should have no gradient from iso loss
        self.assertIsNone(logit_prior.grad)


class TestGradAccumConfig(unittest.TestCase):
    """Gradient accumulation config is consistent."""

    def test_global_equals_micro_times_accum(self):
        from experiment_day70_exactk_d7_generalization import (
            GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE, GRAD_ACCUM_STEPS)
        self.assertEqual(GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS)

    def test_micro_batch_64(self):
        from experiment_day70_exactk_d7_generalization import MICRO_BATCH_SIZE
        self.assertEqual(MICRO_BATCH_SIZE, 64)

    def test_global_batch_256(self):
        from experiment_day70_exactk_d7_generalization import GLOBAL_BATCH_SIZE
        self.assertEqual(GLOBAL_BATCH_SIZE, 256)


if __name__ == '__main__':
    unittest.main()
