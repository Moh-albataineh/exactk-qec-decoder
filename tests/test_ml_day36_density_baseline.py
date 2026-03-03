"""
Day 36 Unit Tests — Density Baseline + p-Regime Hygiene

Tests:
  1. syndrome_count: shape, values, edge cases
  2. density_only_auroc: AUROC in [0,1], single-class, perfect separation
  3. topology_gain: computation and None handling
  4. extract_p_distribution: correct extraction
  5. check_p_regime: match/mismatch detection
  6. Reason codes: exist and correct values
  7. Artifact schema: required keys present
"""
import pytest
import numpy as np


# ============================================================
# Test 1: syndrome_count
# ============================================================
class TestSyndromeCount:
    def test_basic_shape(self):
        from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count
        X = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
        K = compute_syndrome_count(X)
        assert K.shape == (3,)
        assert list(K) == [2, 0, 4]

    def test_all_zeros(self):
        from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count
        X = np.zeros((5, 10), dtype=int)
        K = compute_syndrome_count(X)
        assert K.shape == (5,)
        assert (K == 0).all()

    def test_all_ones(self):
        from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count
        X = np.ones((3, 8), dtype=int)
        K = compute_syndrome_count(X)
        assert list(K) == [8, 8, 8]

    def test_1d_input(self):
        from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count
        X = np.array([1, 0, 1, 1])
        K = compute_syndrome_count(X)
        assert K.shape == (1,)
        assert K[0] == 3

    def test_bool_input(self):
        from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count
        X = np.array([[True, False, True], [False, False, False]])
        K = compute_syndrome_count(X)
        assert list(K) == [2, 0]


# ============================================================
# Test 2: density_only_auroc
# ============================================================
class TestDensityOnlyAuroc:
    def test_returns_auroc_in_range(self):
        from qec_noise_factory.ml.bench.density_baseline import density_only_auroc
        rng = np.random.RandomState(42)
        # Higher K -> more likely positive
        K = np.concatenate([rng.randint(0, 3, 50), rng.randint(5, 10, 50)])
        y = np.array([False] * 50 + [True] * 50)
        result = density_only_auroc(K, y)
        assert result["auroc_density"] is not None
        assert 0.0 <= result["auroc_density"] <= 1.0
        assert result["auroc_density"] > 0.7  # should be high with clear separation

    def test_single_class_all_positive(self):
        from qec_noise_factory.ml.bench.density_baseline import density_only_auroc
        K = np.array([1, 2, 3, 4, 5])
        y = np.array([True, True, True, True, True])
        result = density_only_auroc(K, y)
        assert result["auroc_density"] is None
        assert result["method"] == "single_class"

    def test_single_class_all_negative(self):
        from qec_noise_factory.ml.bench.density_baseline import density_only_auroc
        K = np.array([1, 2, 3, 4, 5])
        y = np.array([False, False, False, False, False])
        result = density_only_auroc(K, y)
        assert result["auroc_density"] is None

    def test_random_labels(self):
        from qec_noise_factory.ml.bench.density_baseline import density_only_auroc
        rng = np.random.RandomState(99)
        K = rng.randint(0, 10, 100)
        y = rng.random(100) > 0.5
        result = density_only_auroc(K, y)
        assert result["auroc_density"] is not None
        assert 0.0 <= result["auroc_density"] <= 1.0

    def test_pr_auc_present(self):
        from qec_noise_factory.ml.bench.density_baseline import density_only_auroc
        rng = np.random.RandomState(42)
        K = np.concatenate([rng.randint(0, 3, 50), rng.randint(5, 10, 50)])
        y = np.array([False] * 50 + [True] * 50)
        result = density_only_auroc(K, y)
        assert "pr_auc_density" in result
        if result["pr_auc_density"] is not None:
            assert 0.0 <= result["pr_auc_density"] <= 1.0


# ============================================================
# Test 3: topology_gain
# ============================================================
class TestTopologyGain:
    def test_basic_computation(self):
        from qec_noise_factory.ml.bench.density_baseline import compute_topology_gain
        gain = compute_topology_gain(0.85, 0.70)
        assert abs(gain - 0.15) < 1e-8

    def test_negative_gain(self):
        from qec_noise_factory.ml.bench.density_baseline import compute_topology_gain
        gain = compute_topology_gain(0.60, 0.70)
        assert gain < 0  # model worse than density-only

    def test_none_clean(self):
        from qec_noise_factory.ml.bench.density_baseline import compute_topology_gain
        gain = compute_topology_gain(None, 0.70)
        assert gain is None

    def test_none_density(self):
        from qec_noise_factory.ml.bench.density_baseline import compute_topology_gain
        gain = compute_topology_gain(0.85, None)
        assert gain is None

    def test_both_none(self):
        from qec_noise_factory.ml.bench.density_baseline import compute_topology_gain
        gain = compute_topology_gain(None, None)
        assert gain is None


# ============================================================
# Test 4: extract_p_distribution
# ============================================================
class TestExtractPDistribution:
    def test_with_mock_meta(self):
        from qec_noise_factory.ml.bench.density_baseline import extract_p_distribution
        import json

        class MockMeta:
            def __init__(self, p):
                self.params_canonical = json.dumps({"circuit": {"p": p}})

        metas = [MockMeta(0.001), MockMeta(0.01), MockMeta(0.001), MockMeta(0.02)]
        result = extract_p_distribution(metas)
        assert result["p_min"] == 0.001
        assert result["p_max"] == 0.02
        assert result["p_count"] == 4
        assert len(result["p_values"]) == 3  # unique

    def test_empty_meta(self):
        from qec_noise_factory.ml.bench.density_baseline import extract_p_distribution
        result = extract_p_distribution([])
        assert result["p_count"] == 0
        assert result["p_min"] is None


# ============================================================
# Test 5: check_p_regime
# ============================================================
class TestCheckPRegime:
    def test_match(self):
        from qec_noise_factory.ml.bench.density_baseline import check_p_regime
        import json

        class MockMeta:
            def __init__(self, p):
                self.params_canonical = json.dumps({"circuit": {"p": p}})

        metas = [MockMeta(0.04)]
        result = check_p_regime(metas, target_p=0.04)
        assert result["match"] is True
        assert result["closest_p"] == 0.04

    def test_mismatch(self):
        from qec_noise_factory.ml.bench.density_baseline import check_p_regime
        import json

        class MockMeta:
            def __init__(self, p):
                self.params_canonical = json.dumps({"circuit": {"p": p}})

        metas = [MockMeta(0.001), MockMeta(0.01)]
        result = check_p_regime(metas, target_p=0.04, tolerance=0.5)
        assert result["match"] is False

    def test_close_enough(self):
        from qec_noise_factory.ml.bench.density_baseline import check_p_regime
        import json

        class MockMeta:
            def __init__(self, p):
                self.params_canonical = json.dumps({"circuit": {"p": p}})

        metas = [MockMeta(0.035)]
        # rel_err = |0.035 - 0.04| / 0.04 = 0.125 < 0.5
        result = check_p_regime(metas, target_p=0.04, tolerance=0.5)
        assert result["match"] is True

    def test_empty(self):
        from qec_noise_factory.ml.bench.density_baseline import check_p_regime
        result = check_p_regime([], target_p=0.04)
        assert result["match"] is False


# ============================================================
# Test 6: Reason codes
# ============================================================
class TestReasonCodes:
    def test_day36_codes_exist(self):
        from qec_noise_factory.ml.bench import reason_codes as RC
        assert hasattr(RC, 'ERR_TOPOLOGY_GAIN_WARN')
        assert hasattr(RC, 'ERR_P_REGIME_MISMATCH')
        assert RC.ERR_TOPOLOGY_GAIN_WARN == "ERR_TOPOLOGY_GAIN_WARN"
        assert RC.ERR_P_REGIME_MISMATCH == "ERR_P_REGIME_MISMATCH"


# ============================================================
# Test 7: syndrome_count matches detector sum
# ============================================================
class TestSyndromeCountMatchesDetSum:
    def test_matches_manual_sum(self):
        from qec_noise_factory.ml.bench.density_baseline import compute_syndrome_count
        rng = np.random.RandomState(42)
        X = (rng.random((20, 50)) > 0.7).astype(int)
        K = compute_syndrome_count(X)
        expected = X.sum(axis=1)
        np.testing.assert_array_equal(K, expected)
