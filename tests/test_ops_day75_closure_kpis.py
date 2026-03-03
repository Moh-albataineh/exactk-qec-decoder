"""Tests for Day 75.1 closure KPIs."""
import json, os, statistics, sys, tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.day75_1_compute_closure_kpis import (
    compute_epoch_median, discover_day75_files, _normalize_fields,
    compute_kpis, TAU_CLEAN, TAU_CLEAN_HI, TG_FLOOR, LEAKY_THRESHOLD,
    DNH_THRESHOLD, SPIKE_LIMIT,
)


def _make_epoch_records(g1_values, start_epoch=1):
    """Helper: build minimal epoch records."""
    recs = []
    for i, g1 in enumerate(g1_values):
        ep = start_epoch + i
        r = {
            "epoch": ep,
            "G1_aligned": g1,
            "loss": 0.5,
            "topo_slice_clean": 0.55 if ep >= 6 else None,
            "topo_mean_drop": 0.01 if ep >= 6 else None,
            "topo_TG": 0.05 if ep >= 6 else None,
        }
        recs.append(r)
    return recs


# ── Test: determinism ─────────────────────────────────────────────

class TestDeterminism:
    def test_epoch_median_deterministic(self):
        recs = _make_epoch_records([0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01,
                                    0.015, 0.012, 0.011, 0.009, 0.008])
        m1 = compute_epoch_median(recs, epoch_min=6)
        m2 = compute_epoch_median(recs, epoch_min=6)
        assert m1 == m2

    def test_epoch_median_correct(self):
        # epochs 1-12, values 0.1 down. Active (>=6) = [0.05,0.04,0.03,0.02,0.01,0.005,0.001]
        g1s = [0.1, 0.09, 0.08, 0.07, 0.06,
               0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]
        recs = _make_epoch_records(g1s)
        m = compute_epoch_median(recs, epoch_min=6)
        active_vals = g1s[5:]  # [0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]
        assert m == statistics.median(active_vals)


# ── Test: cohort partition ────────────────────────────────────────

class TestCohortPartition:
    def test_leaky_vs_clean(self):
        """Seeds with Control epoch-median >= 0.025 are leaky."""
        ctrl_medians = {
            100: 0.030,  # leaky
            101: 0.010,  # clean
            102: 0.050,  # leaky
            103: 0.024,  # clean (< 0.025)
            104: 0.025,  # leaky (>= 0.025)
        }
        leaky = [s for s, m in ctrl_medians.items() if m >= LEAKY_THRESHOLD]
        clean = [s for s, m in ctrl_medians.items() if m < LEAKY_THRESHOLD]
        assert sorted(leaky) == [100, 102, 104]
        assert sorted(clean) == [101, 103]


# ── Test: SafeYield logic ────────────────────────────────────────

class TestSafeYield:
    def test_deployable_all_conditions(self):
        """Must pass all 3 conditions: g1roll <= tau, g1_inst <= tau_hi, tg >= floor."""
        cases = [
            # (g1roll, g1_inst, tg_roll, expected)
            (0.020, 0.030, 0.01, True),     # all pass
            (0.026, 0.030, 0.01, False),    # g1roll > tau_clean
            (0.020, 0.036, 0.01, False),    # g1_inst > tau_clean_hi
            (0.020, 0.030, -0.020, False),  # tg_roll < floor
            (0.025, 0.035, -0.015, True),   # exact boundary: pass
            (0.000, 0.000, 0.000, True),    # ideal
        ]
        for g1r, g1i, tgr, expected in cases:
            deployable = (g1r <= TAU_CLEAN and g1i <= TAU_CLEAN_HI and tgr >= TG_FLOOR)
            assert deployable == expected, f"g1r={g1r}, g1i={g1i}, tgr={tgr}"


# ── Test: spike constraint ────────────────────────────────────────

class TestSpikeConstraint:
    def test_spike_below_limit(self):
        assert abs(0.010) < SPIKE_LIMIT

    def test_spike_at_limit(self):
        # >= 0.015 is a violation
        assert not (abs(0.015) < SPIKE_LIMIT)

    def test_spike_above_limit(self):
        assert not (abs(0.020) < SPIKE_LIMIT)


# ── Test: field normalization ─────────────────────────────────────

class TestNormalization:
    def test_topo_fields_mapped(self):
        r = {"epoch": 6, "topo_slice_clean": 0.55, "topo_mean_drop": 0.01, "topo_TG": 0.05}
        n = _normalize_fields(r)
        assert n["slice_clean"] == 0.55
        assert n["mean_drop"] == 0.01
        assert n["topo_TG"] == 0.05

    def test_defaults_for_warmup(self):
        r = {"epoch": 3, "G1_aligned": 0.1}
        n = _normalize_fields(r)
        assert n["slice_clean"] is None
        assert n["mean_drop"] is None
        assert n["topo_TG"] is None


# ── Test: full pipeline on real artifacts ─────────────────────────

class TestFullPipeline:
    @pytest.fixture
    def day75_root(self):
        root = "ml_artifacts/day75_holdout_d7_v1"
        if not os.path.exists(root):
            pytest.skip("Day 75 artifacts not available")
        return root

    def test_discover(self, day75_root):
        disc = discover_day75_files(day75_root)
        assert disc["all_results_path"] is not None
        assert "ExactK_Tuned_Prod" in disc["arms"]
        assert "Control" in disc["arms"]
        assert len(disc["arms"]["ExactK_Tuned_Prod"]) == 10

    def test_kpis_deterministic(self, day75_root):
        with tempfile.TemporaryDirectory() as tmp1, \
             tempfile.TemporaryDirectory() as tmp2:
            kpi1 = compute_kpis(day75_root, tmp1)
            kpi2 = compute_kpis(day75_root, tmp2)
            assert kpi1["science"]["delta_pct"] == kpi2["science"]["delta_pct"]
            assert kpi1["safe_yield"] == kpi2["safe_yield"]
            assert kpi1["do_no_harm"]["n_violations"] == kpi2["do_no_harm"]["n_violations"]

    def test_science_matches_day75(self, day75_root):
        with tempfile.TemporaryDirectory() as tmp:
            kpi = compute_kpis(day75_root, tmp)
            assert abs(kpi["science"]["delta_pct"] - 45.0) < 0.2
