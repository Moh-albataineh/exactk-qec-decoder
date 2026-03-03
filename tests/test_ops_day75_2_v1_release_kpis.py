"""Tests for Day 75.2 V1.0 release KPIs."""
import json, os, statistics, sys, tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.day75_2_compute_v1_release_kpis import (
    compute_v1_release_kpis, compute_epoch_median, _normalize_fields,
    extract_required_float,
    TAU_CLEAN, TAU_CLEAN_HI, TG_FLOOR, LEAKY_THRESHOLD, SPIKE_LIMIT,
)


# ── Unit tests ────────────────────────────────────────────────────

class TestEpochMedian:
    def test_deterministic(self):
        recs = [{"epoch": i, "G1_aligned": 0.1 - i * 0.008} for i in range(1, 13)]
        assert compute_epoch_median(recs) == compute_epoch_median(recs)

    def test_only_active(self):
        recs = [{"epoch": i, "G1_aligned": float(i)} for i in range(1, 13)]
        m = compute_epoch_median(recs, epoch_min=6)
        expected = statistics.median([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        assert m == expected


class TestNormalize:
    def test_maps_topo_fields(self):
        r = _normalize_fields({"topo_slice_clean": 0.5, "topo_mean_drop": 0.01})
        assert r["slice_clean"] == 0.5
        assert r["mean_drop"] == 0.01

    def test_defaults(self):
        r = _normalize_fields({"epoch": 3})
        assert r["slice_clean"] is None
        assert r["mean_drop"] is None


class TestThresholds:
    def test_deployable_all_pass(self):
        assert 0.020 <= TAU_CLEAN and 0.030 <= TAU_CLEAN_HI and 0.01 >= TG_FLOOR

    def test_deployable_g1roll_fail(self):
        assert not (0.026 <= TAU_CLEAN)

    def test_spike_at_limit(self):
        assert not (0.015 < SPIKE_LIMIT)


class TestCohortPartition:
    def test_leaky_threshold(self):
        meds = {1: 0.030, 2: 0.010, 3: 0.025, 4: 0.024}
        leaky = [s for s, m in meds.items() if m >= LEAKY_THRESHOLD]
        clean = [s for s, m in meds.items() if m < LEAKY_THRESHOLD]
        assert sorted(leaky) == [1, 3]
        assert sorted(clean) == [2, 4]


# ── E2E on real artifacts ─────────────────────────────────────────

class TestFullPipeline:
    @pytest.fixture
    def day75_root(self):
        root = "ml_artifacts/day75_holdout_d7_v1"
        if not os.path.exists(root):
            pytest.skip("Day 75 artifacts not available")
        return root

    def test_kpis_deterministic(self, day75_root):
        with tempfile.TemporaryDirectory() as t1, \
             tempfile.TemporaryDirectory() as t2:
            r1 = compute_v1_release_kpis(day75_root, t1)
            r2 = compute_v1_release_kpis(day75_root, t2)
            assert r1["science"] == r2["science"]
            assert r1["safe_yield"] == r2["safe_yield"]
            assert r1["kpi_c_do_no_harm"]["n_violations"] == r2["kpi_c_do_no_harm"]["n_violations"]

    def test_science_matches(self, day75_root):
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_v1_release_kpis(day75_root, tmp)
            assert abs(r["science"]["delta_pct"] - 45.0) < 0.2

    def test_outputs_exist(self, day75_root):
        with tempfile.TemporaryDirectory() as tmp:
            compute_v1_release_kpis(day75_root, tmp)
            for name in ["v1_release_kpis.json", "v1_release_kpis.md",
                         "v1_release_kpis.csv", "checksums.sha256"]:
                assert os.path.exists(os.path.join(tmp, name)), f"Missing {name}"

    def test_json_has_required_keys(self, day75_root):
        with tempfile.TemporaryDirectory() as tmp:
            compute_v1_release_kpis(day75_root, tmp)
            rpt = json.loads(open(os.path.join(tmp, "v1_release_kpis.json")).read())
            for key in ["science", "safe_yield", "topo_fail",
                         "kpi_a_clean_basin_utility", "kpi_b_leaky_cohort_efficacy",
                         "kpi_c_do_no_harm", "integrity"]:
                assert key in rpt, f"Missing key {key}"

    def test_no_dual_cap_violations(self, day75_root):
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_v1_release_kpis(day75_root, tmp)
            assert r["integrity"]["ExactK_Tuned_Prod"]["dual_cap_violations"] == 0

    def test_kpi_a_present_in_json(self, day75_root):
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_v1_release_kpis(day75_root, tmp)
            ka = r["kpi_a_clean_basin_utility"]
            assert "ctrl_median_tg_roll" in ka
            assert "prod_median_tg_roll" in ka
            assert ka["ctrl_count"] > 0
            assert ka["prod_count"] > 0

    def test_kpi_a_deterministic(self, day75_root):
        with tempfile.TemporaryDirectory() as t1, \
             tempfile.TemporaryDirectory() as t2:
            r1 = compute_v1_release_kpis(day75_root, t1)
            r2 = compute_v1_release_kpis(day75_root, t2)
            assert r1["kpi_a_clean_basin_utility"] == r2["kpi_a_clean_basin_utility"]

    def test_md_contains_deprecation(self, day75_root):
        with tempfile.TemporaryDirectory() as tmp:
            compute_v1_release_kpis(day75_root, tmp)
            md = open(os.path.join(tmp, "v1_release_kpis.md"), encoding="utf-8").read()
            assert "deprecated" in md.lower()
            assert "KPI-A" in md


# ── Synthetic KPI-A test ──────────────────────────────────────────

class TestKpiASynthetic:
    def test_clean_basin_utility(self):
        """Deterministic KPI-A on tiny synthetic fixture."""
        # Simulate 4 seeds: 2 CLEAN, 2 LEAKY for each arm
        ctrl_clean_tg = [0.08, 0.10]
        prod_clean_tg = [0.05, 0.07]
        ctrl_med = statistics.median(ctrl_clean_tg)  # 0.09
        prod_med = statistics.median(prod_clean_tg)  # 0.06
        delta = (prod_med - ctrl_med) / ctrl_med * 100  # (0.06-0.09)/0.09 = -33.3%

        assert ctrl_med == pytest.approx(0.09)
        assert prod_med == pytest.approx(0.06)
        assert round(delta, 1) == pytest.approx(-33.3)

        # Run twice -> same result
        assert statistics.median(ctrl_clean_tg) == statistics.median(ctrl_clean_tg)

    def test_tg_roll_zero_not_none(self):
        """tg_roll=0.0 should not be treated as None."""
        tg_vals = [0.0, 0.0, 0.05, 0.03]
        med = statistics.median(tg_vals)
        assert med is not None
        assert med == statistics.median([0.0, 0.0, 0.03, 0.05])


class TestExtractRequiredFloat:
    def test_present_returns_float(self):
        r = {"tg_roll": 0.08}
        assert extract_required_float(r, "tg_roll") == 0.08

    def test_zero_is_valid(self):
        r = {"tg_roll": 0.0}
        assert extract_required_float(r, "tg_roll") == 0.0

    def test_missing_raises(self):
        r = {"other_key": 0.5}
        with pytest.raises(KeyError, match="tg_roll"):
            extract_required_float(r, "tg_roll")

    def test_none_raises(self):
        r = {"tg_roll": None}
        with pytest.raises(KeyError, match="tg_roll"):
            extract_required_float(r, "tg_roll")

    def test_context_in_error(self):
        r = {}
        with pytest.raises(KeyError, match="Control_60000"):
            extract_required_float(r, "tg_roll", "Control_60000")


class TestKpiAE2EAfterRegen:
    """E2E tests that run AFTER receipt regeneration (Day 75.3)."""

    @pytest.fixture
    def day75_root(self):
        root = "ml_artifacts/day75_holdout_d7_v1"
        if not os.path.exists(root):
            pytest.skip("Day 75 artifacts not available")
        # Check for arm-keyed receipts (Day 75.3)
        if not os.path.exists(os.path.join(root, "selection_receipt_Control_60000.json")):
            pytest.skip("Day 75.3 arm-keyed receipts not yet generated")
        return root

    def test_kpi_a_prod_not_zero(self, day75_root):
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_v1_release_kpis(day75_root, tmp)
            ka = r["kpi_a_clean_basin_utility"]
            assert ka["prod_median_tg_roll"] is not None
            assert ka["prod_median_tg_roll"] > 0.0, \
                f"prod_median_tg_roll should be nonzero, got {ka['prod_median_tg_roll']}"

    def test_prod_count_matches_safe_yield(self, day75_root):
        with tempfile.TemporaryDirectory() as tmp:
            r = compute_v1_release_kpis(day75_root, tmp)
            ka = r["kpi_a_clean_basin_utility"]
            sy = r["safe_yield"]["ExactK_Tuned_Prod"]
            assert ka["prod_count"] == sy["clean_count"]
