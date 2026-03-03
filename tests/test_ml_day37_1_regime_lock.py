"""
Day 37.1 Unit Tests — Regime Lock + Orthogonality Guard + Blocking Gates

Tests:
  1. RegimeLock: fails on real data, fails on wrong p
  2. Projection removal: K-correlated residual becomes ~0
  3. Corr penalty: correct computation, edge cases
  4. Iso-density: density-only baseline yields ~0.5 within K-buckets
  5. Gate wiring: correct reason codes on failure
  6. Reason codes exist
  7. Forward pass with projection removal
"""
import pytest
import numpy as np
import torch


# ============================================================
# Test 1: RegimeLock
# ============================================================
class TestRegimeLock:
    def test_fails_on_real_data(self):
        from qec_noise_factory.ml.bench.regime_lock import RegimeLock, check_regime, RegimeLockError
        lock = RegimeLock(require_generated=True)
        with pytest.raises(RegimeLockError) as exc_info:
            check_regime(lock, data_source="shard_dir:d5")
        assert exc_info.value.reason_code == "ERR_REGIME_LOCK_REQUIRED"

    def test_fails_on_wrong_p(self):
        from qec_noise_factory.ml.bench.regime_lock import RegimeLock, check_regime, RegimeLockError
        lock = RegimeLock(target_p=0.04, require_generated=True)
        with pytest.raises(RegimeLockError) as exc_info:
            check_regime(lock, data_source="generated_stim", p_used=0.01)
        assert exc_info.value.reason_code == "ERR_TARGET_P_MISSING"

    def test_passes_on_correct_generated(self):
        from qec_noise_factory.ml.bench.regime_lock import RegimeLock, check_regime
        lock = RegimeLock(target_p=0.04, require_generated=True)
        # Should not raise
        check_regime(lock, data_source="generated_stim_p0.04", p_used=0.04)

    def test_stim_required_raises(self):
        from qec_noise_factory.ml.bench.regime_lock import RegimeLock, generate_locked_data, RegimeLockError
        lock = RegimeLock()
        # stim is not installed in this environment
        try:
            import stim  # noqa
            pytest.skip("stim is installed, can't test missing stim")
        except ImportError:
            with pytest.raises(RegimeLockError) as exc_info:
                generate_locked_data(lock)
            assert exc_info.value.reason_code == "ERR_STIM_REQUIRED_FOR_DECISION"

    def test_regime_lock_default_values(self):
        from qec_noise_factory.ml.bench.regime_lock import RegimeLock
        lock = RegimeLock()
        assert lock.distance == 5
        assert lock.target_p == 0.04
        assert lock.require_generated is True
        assert lock.n_samples == 4096


# ============================================================
# Test 2: Projection Removal
# ============================================================
class TestProjectionRemoval:
    def test_k_correlated_residual_zeroed(self):
        """If residual = a*K + b, projection removal should zero the K-linear component."""
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
        K = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float)
        residual = K.unsqueeze(-1) * 2.5 + 1.0  # linear in K: (8, 1)
        result = FactorGraphDecoderV1._project_out_k(residual, K)
        # After projection removal, result should have zero correlation with K
        K_c = K - K.mean()
        r_c = result.squeeze() - result.squeeze().mean()
        corr = (r_c * K_c).sum() / (r_c.norm() * K_c.norm() + 1e-8)
        assert abs(corr.item()) < 1e-4, f"Expected ~0 corr after projection, got {corr.item()}"

    def test_orthogonal_residual_unchanged(self):
        """If residual is orthogonal to K, projection removal should not change it."""
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
        K = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float)
        # Create residual orthogonal to K_centered via Gram-Schmidt
        K_c = K - K.mean()
        v = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
        v_orth = v - (v @ K_c) / (K_c @ K_c) * K_c  # project out K component
        residual = v_orth.unsqueeze(-1)  # (8, 1)
        # Verify orthogonality
        dot = (residual.squeeze() * K_c).sum()
        assert abs(dot.item()) < 1e-4  # indeed orthogonal
        result = FactorGraphDecoderV1._project_out_k(residual, K)
        torch.testing.assert_close(result, residual, atol=1e-4, rtol=1e-4)

    def test_batch_size_1_no_crash(self):
        """Batch size 1: projection should handle gracefully (B>1 guard in forward)."""
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
        # B=1 would have constant K, so denom=0 clamped to 1e-8
        K = torch.tensor([5.0])
        residual = torch.tensor([[2.0]])
        result = FactorGraphDecoderV1._project_out_k(residual, K)
        # K_c = [0], denom clamps to 1e-8, proj_coeff = 0
        torch.testing.assert_close(result, residual, atol=1e-4, rtol=1e-4)


# ============================================================
# Test 3: Correlation Penalty
# ============================================================
class TestCorrPenalty:
    def _make_model(self):
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
        from qec_noise_factory.ml.bench.density_prior import build_k_prior_table
        torch.manual_seed(42)
        m = FactorGraphDecoderV1(
            det_input_dim=2, err_input_dim=1, output_dim=1,
            hidden_dim=16, num_mp_layers=2,
            use_density_residualization=True,
        )
        K_train = np.array([0, 1, 2, 3, 4, 5])
        y_train = np.array([0, 0, 1, 0, 1, 1], dtype=bool)
        prior = build_k_prior_table(K_train, y_train)
        m.set_density_prior(prior, n_det=5)
        return m

    def test_corr_penalty_no_cache(self):
        m = self._make_model()
        # Before any forward call, penalty should be 0
        assert m.compute_corr_penalty().item() == 0.0

    def test_corr_penalty_after_forward(self):
        m = self._make_model()
        det = torch.zeros(8, 6, 2)
        det[:, :5, 0] = torch.rand(8, 5) > 0.5
        det[:, -1, 1] = 1.0
        err = torch.randn(10, 1)
        ei = torch.tensor([[0,1,2], [3,4,5]], dtype=torch.long)
        m.eval()
        with torch.no_grad():
            m(det, err, ei, ei)
        penalty = m.compute_corr_penalty()
        assert penalty.item() >= 0.0  # corr^2 is always >= 0

    def test_constant_k_returns_zero(self):
        m = self._make_model()
        # All same K: std=0 → penalty=0
        m._last_residual_logit = torch.randn(10, 1)
        m._last_K_float = torch.ones(10) * 3.0
        assert m.compute_corr_penalty().item() == 0.0


# ============================================================
# Test 4: Iso-density baseline ~0.5
# ============================================================
class TestIsoDensityBaseline:
    def test_random_pred_near_half(self):
        """If predictions are random, within-K AUROC should be ~0.5."""
        from qec_noise_factory.ml.bench.density_prior import compute_iso_density_auroc
        rng = np.random.RandomState(42)
        N = 400
        K = np.repeat([0, 5], N // 2)  # 200 each
        y = rng.random(N) > 0.5
        probs = rng.random(N)  # random predictions
        result = compute_iso_density_auroc(y, probs, K, n_min=80)
        assert result["n_qualified"] == 2
        # Random predictions should give AUROC near 0.5
        for b in result["bucket_list"]:
            assert 0.3 < b["auroc"] < 0.7  # within reasonable range of 0.5


# ============================================================
# Test 5: Blocking Gates
# ============================================================
class TestBlockingGates:
    def test_all_pass(self):
        from qec_noise_factory.ml.bench.blocking_gates import run_blocking_gates
        metrics = {
            "topology_gain": 0.05,
            "scrambler_delta": 0.15,
            "residual_k_corr_clean": 0.05,
            "residual_k_corr_scrambled": 0.03,
            "iso_density": {"macro_auroc": 0.60, "n_qualified": 3, "bucket_list": []},
            "tpr": 0.3,
        }
        result = run_blocking_gates(metrics)
        assert result["all_pass"] is True
        assert result["n_fail"] == 0

    def test_topology_gain_fail(self):
        from qec_noise_factory.ml.bench.blocking_gates import run_blocking_gates
        metrics = {
            "topology_gain": 0.01,
            "scrambler_delta": 0.15,
            "residual_k_corr_clean": 0.05,
            "tpr": 0.3,
        }
        result = run_blocking_gates(metrics)
        failed = [g for g in result["gate_results"] if g["status"] == "FAIL"]
        assert any(g["gate"] == "topology_gain" for g in failed)

    def test_residual_corr_fail(self):
        from qec_noise_factory.ml.bench.blocking_gates import run_blocking_gates
        metrics = {
            "topology_gain": 0.05,
            "scrambler_delta": 0.15,
            "residual_k_corr_clean": 0.50,  # too high
            "tpr": 0.3,
        }
        result = run_blocking_gates(metrics)
        failed = [g for g in result["gate_results"] if g["status"] == "FAIL"]
        assert any(g["gate"] == "residual_k_corr_clean" for g in failed)
        assert any(g["reason_code"] == "ERR_RESIDUAL_K_CORR_HIGH" for g in failed)

    def test_scrambler_shortcut_fail(self):
        from qec_noise_factory.ml.bench.blocking_gates import run_blocking_gates
        metrics = {
            "topology_gain": 0.05,
            "scrambler_delta": -0.01,
            "residual_k_corr_clean": 0.05,
            "tpr": 0.3,
        }
        result = run_blocking_gates(metrics)
        assert result["all_pass"] is False

    def test_dump_bundle_written(self, tmp_path):
        from qec_noise_factory.ml.bench.blocking_gates import run_blocking_gates
        metrics = {"topology_gain": 0.001, "tpr": 0.3}
        result = run_blocking_gates(metrics, artifact_dir=tmp_path)
        assert result["all_pass"] is False
        # Check dump bundle was written
        fail_dirs = list(tmp_path.glob("FAIL_*"))
        assert len(fail_dirs) == 1
        assert (fail_dirs[0] / "gate_report.json").exists()
        assert (fail_dirs[0] / "metrics_dump.json").exists()


# ============================================================
# Test 6: Reason Codes
# ============================================================
class TestDay371ReasonCodes:
    def test_codes_exist(self):
        from qec_noise_factory.ml.bench import reason_codes as RC
        assert RC.ERR_REGIME_LOCK_REQUIRED == "ERR_REGIME_LOCK_REQUIRED"
        assert RC.ERR_TARGET_P_MISSING == "ERR_TARGET_P_MISSING"
        assert RC.ERR_STIM_REQUIRED_FOR_DECISION == "ERR_STIM_REQUIRED_FOR_DECISION"


# ============================================================
# Test 7: FG v1 with projection removal in forward pass
# ============================================================
class TestFGV1ProjectionForward:
    def test_output_shape_with_projection(self):
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
        from qec_noise_factory.ml.bench.density_prior import build_k_prior_table
        torch.manual_seed(42)
        m = FactorGraphDecoderV1(
            det_input_dim=2, err_input_dim=1, output_dim=1,
            hidden_dim=16, num_mp_layers=2,
            use_density_residualization=True,
        )
        K_train = np.array([0, 1, 2, 3, 4, 5, 0, 1])
        y_train = np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype=bool)
        prior = build_k_prior_table(K_train, y_train)
        m.set_density_prior(prior, n_det=5)

        det = torch.zeros(8, 6, 2)
        det[:, :5, 0] = torch.rand(8, 5) > 0.5
        det[:, -1, 1] = 1.0
        err = torch.randn(10, 1)
        ei = torch.tensor([[0,1,2], [3,4,5]], dtype=torch.long)

        out = m(det, err, ei, ei)
        assert out.shape == (8, 1)

    def test_projection_reduces_k_correlation(self):
        """With projection ON, residual-K correlation should be near 0."""
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
        from qec_noise_factory.ml.bench.density_prior import build_k_prior_table, compute_residual_k_correlation
        torch.manual_seed(42)
        m = FactorGraphDecoderV1(
            det_input_dim=2, err_input_dim=1, output_dim=1,
            hidden_dim=16, num_mp_layers=2,
            use_density_residualization=True,
        )
        K_train = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
        y_train = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1], dtype=bool)
        prior = build_k_prior_table(K_train, y_train)
        m.set_density_prior(prior, n_det=5)

        det = torch.zeros(16, 6, 2)
        # Varying K values
        for i in range(16):
            n_active = i % 6
            det[i, :n_active, 0] = 1.0
        det[:, -1, 1] = 1.0
        err = torch.randn(10, 1)
        ei = torch.tensor([[0,1,2], [3,4,5]], dtype=torch.long)

        m.eval()
        with torch.no_grad():
            m(det, err, ei, ei)

        # Check cached residual
        if m._last_residual_logit is not None and m._last_K_float is not None:
            resid = m._last_residual_logit.numpy().ravel()
            K_vals = m._last_K_float.numpy().ravel()
            corr = compute_residual_k_correlation(resid, K_vals)
            # Projection should make correlation near 0
            assert abs(corr) < 0.15, f"Expected low corr with projection, got {corr}"
