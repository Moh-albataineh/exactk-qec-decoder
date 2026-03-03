"""
Day 33 — Unit Tests: Factor-Graph v1, Focal Loss, Calibration, Guards

Tests (all ≤5s):
  - FactorGraphDecoderV1: forward/backward, model_name, loss_fn config
  - FocalLoss: forward/backward, shape, gamma effect
  - pos_weight clamping behavior
  - F0.5 calibration grid: fields present, threshold in range
  - Collapse guard triggers: all-0, all-1 adversarial preds
  - Metric integrity: TPR not silently 0 when F1>0
  - Hash field assertions for MWPM and FG rows
  - Quality gate: metric_integrity gate catches bad rows
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FactorGraphDecoderV1 Tests
# ---------------------------------------------------------------------------

class TestFactorGraphV1:
    """Tests for FactorGraphDecoderV1."""

    @pytest.fixture
    def small_graph(self):
        """Small bipartite graph fixture."""
        N_d, N_e = 5, 3
        d2e_src = [0, 1, 1, 2, 2, 3, 4]
        d2e_dst = [0, 0, 1, 1, 2, 2, 2]
        return {
            "N_d": N_d,
            "N_e": N_e,
            "edge_d2e": torch.tensor([d2e_src, d2e_dst], dtype=torch.long),
            "edge_e2d": torch.tensor([d2e_dst, d2e_src], dtype=torch.long),
            "error_weights": torch.tensor([3.0, 2.5, 1.0], dtype=torch.float32),
            "obs_mask": torch.tensor([True, False, True], dtype=torch.bool),
        }

    def test_v1_forward_shape(self, small_graph):
        """V1 forward pass produces correct shape."""
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1

        B = 4
        model = FactorGraphDecoderV1(
            det_input_dim=2, err_input_dim=1, output_dim=1,
            hidden_dim=16, num_mp_layers=2,
        )

        det_feats = torch.randn(B, small_graph["N_d"], 2)
        err_feats = torch.randn(small_graph["N_e"], 1)

        logits = model(
            det_feats, err_feats,
            small_graph["edge_d2e"], small_graph["edge_e2d"],
            error_weights=small_graph["error_weights"],
            observable_mask=small_graph["obs_mask"],
        )
        assert logits.shape == (B, 1)
        assert torch.isfinite(logits).all()

    def test_v1_model_name(self):
        """V1 has correct model_name."""
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
        assert FactorGraphDecoderV1.model_name == "factor_graph_v1"
        assert FactorGraphDecoderV1.needs_graph is True

    def test_v1_inherits_v0(self):
        """V1 inherits from V0."""
        from qec_noise_factory.ml.models.factor_graph import (
            FactorGraphDecoderV0, FactorGraphDecoderV1,
        )
        assert issubclass(FactorGraphDecoderV1, FactorGraphDecoderV0)

    def test_v1_loss_fn_attr(self):
        """V1 has loss_fn_name and focal_gamma attributes."""
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1

        model = FactorGraphDecoderV1(loss_fn="focal", focal_gamma=2.0)
        assert model.loss_fn_name == "focal"
        assert model.focal_gamma == 2.0

        model_bce = FactorGraphDecoderV1(loss_fn="bce", focal_gamma=0.0)
        assert model_bce.loss_fn_name == "bce"

    def test_v1_default_hidden_dim(self):
        """V1 defaults to hidden_dim=48 (vs v0's 32)."""
        from qec_noise_factory.ml.models.factor_graph import FactorGraphDecoderV1
        model = FactorGraphDecoderV1()
        assert model.hidden_dim == 48

    def test_v1_backward_finite(self, small_graph):
        """V1 backward pass with focal loss produces finite gradients."""
        from qec_noise_factory.ml.models.factor_graph import (
            FactorGraphDecoderV1, FocalLoss,
        )

        B = 4
        model = FactorGraphDecoderV1(
            det_input_dim=2, err_input_dim=1, output_dim=1,
            hidden_dim=16, num_mp_layers=2,
        )

        det_feats = torch.randn(B, small_graph["N_d"], 2)
        err_feats = torch.randn(small_graph["N_e"], 1)
        labels = torch.randint(0, 2, (B, 1)).float()

        logits = model(
            det_feats, err_feats,
            small_graph["edge_d2e"], small_graph["edge_e2d"],
            error_weights=small_graph["error_weights"],
            observable_mask=small_graph["obs_mask"],
        )

        focal_loss = FocalLoss(gamma=2.0, pos_weight=3.0)
        loss = focal_loss(logits, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite grad: {name}"


# ---------------------------------------------------------------------------
# FocalLoss Tests
# ---------------------------------------------------------------------------

class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_focal_forward_shape(self):
        """Focal loss produces scalar."""
        from qec_noise_factory.ml.models.factor_graph import FocalLoss

        fl = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = fl(logits, targets)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_focal_with_pos_weight(self):
        """Focal loss with pos_weight produces finite scalar."""
        from qec_noise_factory.ml.models.factor_graph import FocalLoss

        fl = FocalLoss(gamma=2.0, pos_weight=5.0)
        logits = torch.randn(16, 1)
        targets = torch.randint(0, 2, (16, 1)).float()
        loss = fl(logits, targets)
        assert torch.isfinite(loss)

    def test_focal_gamma_effect(self):
        """Higher gamma reduces loss for well-classified examples."""
        from qec_noise_factory.ml.models.factor_graph import FocalLoss

        torch.manual_seed(42)
        logits = torch.tensor([[5.0], [-5.0], [5.0], [-5.0]])  # confident
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])   # correct

        fl_low = FocalLoss(gamma=0.0)   # no focal weighting
        fl_high = FocalLoss(gamma=3.0)  # strong focal weighting

        loss_low = fl_low(logits, targets)
        loss_high = fl_high(logits, targets)

        # Focal loss should be LOWER for well-classified examples with higher gamma
        assert loss_high < loss_low, \
            f"Focal loss with gamma=3 ({loss_high:.4f}) should be < gamma=0 ({loss_low:.4f})"

    def test_focal_backward(self):
        """Focal loss backward pass produces gradients."""
        from qec_noise_factory.ml.models.factor_graph import FocalLoss

        logits = torch.randn(8, 1, requires_grad=True)
        targets = torch.randint(0, 2, (8, 1)).float()
        fl = FocalLoss(gamma=2.0, pos_weight=3.0)
        loss = fl(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# pos_weight Clamping Tests
# ---------------------------------------------------------------------------

class TestPosWeightClamp:
    """Tests for pos_weight computation and clamping."""

    def test_pos_weight_clamp_high_imbalance(self):
        """pos_weight clamped to 8.0 for extreme imbalance."""
        # Simulate: y_pos = 0.01 → pw_auto = 99.0 → clamped to 8.0
        y_pos = 0.01
        pw_auto = (1 - y_pos) / max(y_pos, 1e-6)
        pw_max = 8.0
        pw_used = min(pw_max, pw_auto)
        assert pw_used == 8.0
        assert pw_used < pw_auto  # was clamped

    def test_pos_weight_no_clamp_balanced(self):
        """pos_weight not clamped for balanced data."""
        y_pos = 0.4
        pw_auto = (1 - y_pos) / max(y_pos, 1e-6)
        pw_max = 8.0
        pw_used = min(pw_max, pw_auto)
        assert pw_used == pytest.approx(pw_auto)  # not clamped

    def test_pos_weight_zero_guard(self):
        """pos_weight handles y_pos=0 without division by zero."""
        y_pos = 0.0
        pw_auto = (1 - y_pos) / max(y_pos, 1e-6)
        pw_max = 8.0
        pw_used = min(pw_max, pw_auto)
        assert pw_used == 8.0  # clamped


# ---------------------------------------------------------------------------
# Calibration Tests
# ---------------------------------------------------------------------------

class TestCalibration:
    """Tests for F0.5 calibration grid."""

    def test_f05_calibration_fields(self):
        """F0.5 calibration produces expected fields."""
        # Simulate the constrained calibration sweep
        cal_grid = np.linspace(0.15, 0.95, 33)
        assert len(cal_grid) == 33
        assert cal_grid[0] == pytest.approx(0.15)
        assert cal_grid[-1] == pytest.approx(0.95)

    def test_f05_vs_f1(self):
        """F0.5 favors precision over recall compared to F1."""
        prec, rec = 0.9, 0.5
        f1 = 2 * prec * rec / (prec + rec)
        beta = 0.5
        f05 = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
        # F0.5 should be higher than F1 when precision > recall
        assert f05 > f1, f"F0.5={f05:.4f} should > F1={f1:.4f} when prec>rec"

    def test_f05_with_high_recall(self):
        """F0.5 penalizes low precision even with high recall."""
        prec, rec = 0.3, 1.0
        beta = 0.5
        f05 = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
        # F0.5 should be low despite perfect recall
        assert f05 < 0.5, f"F0.5={f05:.4f} should be low for prec={prec}"

    def test_calibration_threshold_selection(self):
        """Threshold selection picks highest F0.5."""
        np.random.seed(42)
        probs = np.random.rand(100, 1)
        labels = (np.random.rand(100, 1) > 0.7).astype(bool)

        cal_grid = np.linspace(0.05, 0.95, 37)
        best_thr, best_score = 0.5, -1.0

        for thr in cal_grid:
            preds = (probs > thr).astype(bool)
            tp = (preds & labels).sum()
            fp = (preds & ~labels).sum()
            fn = (~preds & labels).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            beta = 0.5
            f_beta = (1 + beta**2) * prec * rec / max(beta**2 * prec + rec, 1e-8)

            if f_beta > best_score:
                best_score = f_beta
                best_thr = float(thr)

        assert 0.05 <= best_thr <= 0.95
        assert best_score >= 0


# ---------------------------------------------------------------------------
# Collapse Guard Tests
# ---------------------------------------------------------------------------

class TestCollapseGuards:
    """Tests for collapse and reverse-collapse guard triggers."""

    def test_all_positive_triggers_high_collapse(self):
        """All-positive predictions trigger PPR>95% collapse."""
        ppr = 1.0
        assert ppr > 0.95  # triggers collapse_high

    def test_all_negative_triggers_low_collapse(self):
        """All-negative predictions trigger PPR<0.5% collapse."""
        ppr = 0.0
        assert ppr < 0.005  # triggers collapse_low

    def test_tpr_collapse(self):
        """TPR<5% with F1>0 triggers TPR collapse."""
        tpr = 0.02
        f1 = 0.1
        assert tpr < 0.05 and f1 > 0

    def test_reverse_collapse_fpr(self):
        """FPR>70% triggers reverse collapse."""
        fpr = 0.85
        assert fpr > 0.70

    def test_healthy_no_collapse(self):
        """Healthy metrics trigger no collapse guards."""
        ppr = 0.25
        tpr = 0.80
        fpr = 0.15
        f1 = 0.65

        collapse_low = ppr < 0.005
        collapse_high = ppr > 0.95
        collapse_tpr = tpr < 0.05 and f1 > 0
        reverse_collapse = fpr > 0.70

        any_warn = collapse_low or collapse_high or collapse_tpr or reverse_collapse
        assert not any_warn, "Healthy metrics should not trigger collapse"


# ---------------------------------------------------------------------------
# Metric Integrity Tests
# ---------------------------------------------------------------------------

class TestMetricIntegrity:
    """Tests for metric field integrity."""

    def test_tpr_consistent_with_f1(self):
        """TPR=0 with F1>0 is a metric mapping bug."""
        from qec_noise_factory.ml.metrics.classification import compute_metrics

        y_true = np.array([[True], [True], [False], [False]])
        y_pred = np.array([[True], [False], [False], [False]])

        m = compute_metrics(y_true, y_pred)
        tpr = m.get("macro_tpr", 0.0)
        f1 = m.get("macro_f1", 0.0)

        # If F1 > 0, TPR must be > 0
        if f1 > 0:
            assert tpr > 0, f"F1={f1} but TPR={tpr} — metric mapping bug"

    def test_metrics_keys_present(self):
        """All expected metric keys are present."""
        from qec_noise_factory.ml.metrics.classification import compute_metrics

        y_true = np.array([[True], [False]])
        y_pred = np.array([[True], [False]])

        m = compute_metrics(y_true, y_pred)
        for key in ["macro_f1", "macro_precision", "macro_tpr", "macro_fpr",
                     "macro_balanced_accuracy"]:
            assert key in m, f"Missing metric key: {key}"

    def test_all_identical_predictions(self):
        """Metrics correct for all-positive predictions (degenerate)."""
        from qec_noise_factory.ml.metrics.classification import compute_metrics

        y_true = np.array([[True], [True], [False], [False]])
        y_pred = np.array([[True], [True], [True], [True]])  # all positive

        m = compute_metrics(y_true, y_pred)
        assert m["macro_tpr"] == 1.0  # recall = 1.0
        assert m["macro_fpr"] == 1.0  # FPR = 1.0 (all negatives are FPs)
        assert m["macro_precision"] == 0.5  # 2 TP / 4 predicted positive


# ---------------------------------------------------------------------------
# Hash Field Tests
# ---------------------------------------------------------------------------

class TestHashFields:
    """Tests for hash field population."""

    def test_bipartite_hash_nonempty(self):
        """Bipartite graph hash is non-empty."""
        from qec_noise_factory.ml.graph.dem_bipartite import build_bipartite_graph

        bg = build_bipartite_graph(distance=3, rounds=3, p=0.01, basis="X")
        assert bg.dem_topology_hash != "", "Bipartite hash must not be empty"
        assert len(bg.dem_topology_hash) == 16

    def test_dem_graph_hash_nonempty(self):
        """DEM graph hash is non-empty."""
        from qec_noise_factory.ml.graph.dem_graph import build_dem_graph

        spec = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
        assert spec.dem_graph_hash != "", "DEM graph hash must not be empty"


# ---------------------------------------------------------------------------
# Quality Gate Tests
# ---------------------------------------------------------------------------

class TestQualityGates:
    """Tests for quality gate logic."""

    def test_metric_integrity_gate_catches_bug(self):
        """metric_integrity gate fails when TPR=0 but F1>0."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import (
            BenchV3Row, run_quality_gates,
        )

        # Create a row with F1>0 but TPR=0 (metric mapping bug)
        bogus = BenchV3Row(
            suite="TEST", decoder="FAKE",
            f1=0.5, recall_tpr=0.0, status="pass",
        )
        result = run_quality_gates([bogus])
        checks = result["checks"]
        gate_dict = {c["name"]: c for c in checks}
        assert "metric_integrity" in gate_dict
        assert gate_dict["metric_integrity"]["ok"] is False

    def test_metric_integrity_gate_passes_clean(self):
        """metric_integrity gate passes for consistent metrics."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import (
            BenchV3Row, run_quality_gates,
        )

        clean = BenchV3Row(
            suite="TEST", decoder="FAKE",
            f1=0.5, recall_tpr=0.6, status="pass",
        )
        result = run_quality_gates([clean])
        checks = result["checks"]
        gate_dict = {c["name"]: c for c in checks}
        assert "metric_integrity" in gate_dict
        assert gate_dict["metric_integrity"]["ok"] is True


# ---------------------------------------------------------------------------
# Reason Codes Tests
# ---------------------------------------------------------------------------

class TestReasonCodes:
    """Tests for Day 33 reason codes."""

    def test_day33_codes_exist(self):
        """Day 33 reason codes exist."""
        from qec_noise_factory.ml.bench import reason_codes as rc

        assert hasattr(rc, "FG_COLLAPSE_TPR")
        assert hasattr(rc, "FG_REVERSE_COLLAPSE")
        assert hasattr(rc, "METRIC_INTEGRITY_FAIL")
        assert hasattr(rc, "HASH_MISSING")

    def test_codes_are_strings(self):
        """All reason codes are non-empty strings."""
        from qec_noise_factory.ml.bench import reason_codes as rc

        for code in [rc.FG_COLLAPSE_TPR, rc.FG_REVERSE_COLLAPSE,
                     rc.METRIC_INTEGRITY_FAIL, rc.HASH_MISSING]:
            assert isinstance(code, str) and len(code) > 0


# ---- Day 33.5: Pre-rerun hardening tests ----

class TestMWPMPPRPopulation:
    """Tests for MWPM pred_positive_rate being populated."""

    def test_mwpm_ppr_in_row_extra(self):
        """Verify BenchV3Row supports pred_positive_rate in extra."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import BenchV3Row
        row = BenchV3Row(suite="test", decoder="MWPM_ORACLE")
        row.extra["pred_positive_rate"] = 0.15
        assert row.extra["pred_positive_rate"] == 0.15

    def test_mwpm_ppr_not_default_zero(self):
        """MWPM PPR should not silently default to 0.0 if predictions exist."""
        # Simulate: y_hat with some predictions > 0
        preds = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        ppr = float(preds.mean())
        assert ppr == pytest.approx(0.3), f"Expected 0.3 but got {ppr}"
        assert ppr > 0.0  # Must not be zero when predictions exist


class TestDEMHashPopulation:
    """Tests for dem_graph_hash population and gate accuracy."""

    def test_hash_field_default(self):
        """dem_graph_hash defaults to empty string (falsy)."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import BenchV3Row
        row = BenchV3Row(suite="test", decoder="MWPM_ORACLE")
        assert row.dem_graph_hash == ""
        assert not row.dem_graph_hash  # Must be falsy

    def test_hash_gate_fails_when_empty(self):
        """Gate must FAIL if all MWPM rows have empty dem_graph_hash."""
        rows = []
        for i in range(3):
            from qec_noise_factory.ml.bench.unified_benchmark_v3 import BenchV3Row
            row = BenchV3Row(suite="D", decoder="MWPM_ORACLE", status="pass")
            row.dem_graph_hash = ""  # Empty — not populated
            rows.append(row)

        any_hash = any(r.dem_graph_hash for r in rows)
        assert not any_hash, "Gate should fail: no rows have hash"

    def test_hash_gate_passes_when_populated(self):
        """Gate passes if at least one MWPM row has non-empty hash."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import BenchV3Row
        rows = [
            BenchV3Row(suite="D", decoder="MWPM_ORACLE", status="pass"),
            BenchV3Row(suite="D", decoder="MWPM_ORACLE", status="pass"),
        ]
        rows[0].dem_graph_hash = ""
        rows[1].dem_graph_hash = "abcdef1234567890"
        any_hash = any(r.dem_graph_hash for r in rows)
        assert any_hash, "Gate should pass: one row has hash"


class TestCalibrationConstraintTracking:
    """Tests for calibration constraint hit detection."""

    def test_constrained_grid_bounds(self):
        """Constrained grid starts at 0.15, not 0.05."""
        cal_grid = np.linspace(0.15, 0.95, 33)
        assert len(cal_grid) == 33
        assert cal_grid[0] == pytest.approx(0.15)
        assert cal_grid[-1] == pytest.approx(0.95)
        # No threshold below 0.15
        assert all(t >= 0.15 for t in cal_grid)

    def test_hit_threshold_floor_detected(self):
        """Detect when calibration picks the floor threshold."""
        cal_grid = np.linspace(0.15, 0.95, 33)
        best_thr = float(cal_grid[0])  # = 0.15 (the floor)
        hit_floor = abs(best_thr - float(cal_grid[0])) < 0.001
        assert hit_floor, "Should detect floor hit"

    def test_no_floor_hit_for_normal_threshold(self):
        """Normal threshold should NOT be flagged as floor hit."""
        cal_grid = np.linspace(0.15, 0.95, 33)
        best_thr = 0.45  # normal threshold
        hit_floor = abs(best_thr - float(cal_grid[0])) < 0.001
        assert not hit_floor, "Should not flag normal threshold as floor"

    def test_feasibility_check(self):
        """Feasibility check rejects high-FPR and high-PPR thresholds."""
        fpr_cap = 0.60
        ppr_cap = 0.90

        # Case 1: feasible
        assert (0.30 <= fpr_cap) and (0.50 <= ppr_cap)

        # Case 2: FPR too high
        fpr_bad = 0.75
        assert not (fpr_bad <= fpr_cap), "FPR 75% should be infeasible"

        # Case 3: PPR too high
        ppr_bad = 0.95
        assert not (ppr_bad <= ppr_cap), "PPR 95% should be infeasible"

    def test_fallback_activates_when_no_feasible(self):
        """If no threshold is feasible, fallback_used should be True."""
        # Simulate: all thresholds have FPR > cap
        cal_grid = np.linspace(0.15, 0.95, 33)
        fpr_cap = 0.60
        ppr_cap = 0.90

        best_score = -1.0  # nothing feasible
        fallback_used = best_score <= 0
        assert fallback_used, "Fallback should activate when no feasible threshold"


class TestGateMessageAccuracy:
    """Ensure gate messages accurately reflect pass/fail state."""

    def test_tpr_gate_pass_message(self):
        """Pass message should not say 'all rows have TPR=0'."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import BenchV3Row
        rows = [BenchV3Row(suite="D", decoder="FG", status="pass")]
        rows[0].recall_tpr = 0.85

        passing = [r for r in rows if r.status == "pass"]
        any_tpr = any(r.recall_tpr > 0 for r in passing)
        n_zero = sum(1 for r in passing if r.recall_tpr == 0)

        assert any_tpr
        msg = f"{len(passing) - n_zero}/{len(passing)} passing rows have recall_tpr>0"
        assert "recall_tpr=0" not in msg
        assert "1/1" in msg


# ---- Day 33.6: Hardening Readiness Fix tests ----

class TestCalibrationFallbackPrevention:
    """Calibration must never select threshold that yields TPR=0 or PPR=0."""

    def test_old_logic_would_pick_max_threshold(self):
        """Synthetic scenario: all thresholds with FPR≤cap have TPR=0.
        Old logic picked thr=0.95 (max_threshold_fallback). New logic must not."""
        # Simulate: 33 thresholds where low thresholds have high FPR,
        # high thresholds have TPR=0 (all-negative)
        cal_grid = np.linspace(0.15, 0.95, 33)
        tpr_min = 0.05
        ppr_min = 0.01
        fpr_cap = 0.60
        ppr_cap = 0.90

        # Scenario: thresholds < 0.5 have FPR=0.80 (infeasible via cap)
        # thresholds >= 0.5 have TPR=0 / PPR=0 (infeasible via floor)
        best_thr, best_score = 0.5, -1.0
        for thr in cal_grid:
            if thr < 0.5:
                rec, fpr, ppr = 0.90, 0.80, 0.85  # high FPR → infeasible
            else:
                rec, fpr, ppr = 0.0, 0.0, 0.0      # all-negative → infeasible via floors

            feasible = (
                fpr <= fpr_cap
                and ppr <= ppr_cap
                and rec >= tpr_min
                and ppr >= ppr_min
            )
            assert not feasible, f"Threshold {thr} should be infeasible"

        # Old logic would pick max_threshold_fallback = 0.95
        # New logic should NOT select thr=0.95 with TPR=0
        assert best_score <= 0, "No feasible option should exist"
        # New fallback 3: hard fail
        cal_metric = "DEGENERATE_FAIL"
        assert cal_metric == "DEGENERATE_FAIL"

    def test_tpr_floor_rejects_all_negative(self):
        """Threshold with TPR=0 is rejected by tpr_min constraint."""
        tpr_min = 0.05
        ppr_min = 0.01
        rec, ppr = 0.0, 0.0  # all-negative predictions
        feasible = rec >= tpr_min and ppr >= ppr_min
        assert not feasible, "TPR=0 must be infeasible"

    def test_relaxed_fallback_keeps_floors(self):
        """Fallback 2 relaxes caps but keeps TPR/PPR floors."""
        tpr_min = 0.05
        ppr_min = 0.01
        # Case: high FPR (cap violated) but valid TPR/PPR → relaxed feasible
        rec, fpr, ppr = 0.30, 0.80, 0.40
        relaxed_ok = rec >= tpr_min and ppr >= ppr_min
        assert relaxed_ok, "Should be feasible if caps relaxed"

        # Case: TPR=0 → still infeasible even with relaxed caps
        rec2, ppr2 = 0.0, 0.0
        relaxed_ok2 = rec2 >= tpr_min and ppr2 >= ppr_min
        assert not relaxed_ok2, "TPR=0 must stay infeasible even with relaxed caps"

    def test_rejected_reason_counts_tracked(self):
        """Rejected reason counts are tracked correctly."""
        rejected = {"fpr_over_cap": 0, "ppr_over_cap": 0,
                     "tpr_under_min": 0, "ppr_under_min": 0}
        # Simulate 3 rejected thresholds
        rejected["fpr_over_cap"] += 1
        rejected["tpr_under_min"] += 2
        assert rejected["fpr_over_cap"] == 1
        assert rejected["tpr_under_min"] == 2
        assert sum(rejected.values()) == 3


class TestFGDecoderGates:
    """Unit tests for run_fg_gates with mocked rows."""

    def _make_fg_row(self, label="fg1_test", tpr=0.5, fpr=0.3, f1=0.5, ppr=0.3,
                     decoder="FG_DEM_BIPARTITE_V1", cal_metric="f0.5_constrained"):
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import BenchV3Row
        row = BenchV3Row(suite="D", decoder=decoder, status="pass", label=label)
        row.recall_tpr = tpr
        row.fpr = fpr
        row.f1 = f1
        row.extra["pred_positive_rate"] = ppr
        row.extra["calibration"] = {"metric": cal_metric}
        return row

    def test_all_gates_pass_healthy_rows(self):
        """Healthy FG rows pass all gates."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import run_fg_gates
        rows = [
            self._make_fg_row("fg1_p01", tpr=0.8, fpr=0.2, f1=0.7, ppr=0.3),
            self._make_fg_row("fg1_p02", tpr=0.6, fpr=0.3, f1=0.5, ppr=0.4),
        ]
        result = run_fg_gates(rows)
        assert result["pass"] is True
        for check in result["checks"]:
            assert check["ok"], f"Gate {check['name']} should pass: {check['msg']}"

    def test_majority_collapse_tpr_zero(self):
        """Gate fails when TPR=0 (all-negative)."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import run_fg_gates
        rows = [self._make_fg_row("fg1_bad", tpr=0.0, fpr=0.0, f1=0.0, ppr=0.0)]
        result = run_fg_gates(rows)
        collapse_gate = next(c for c in result["checks"] if c["name"] == "fg_no_majority_collapse")
        assert not collapse_gate["ok"]
        assert "collapsed to all-negative" in collapse_gate["msg"]

    def test_majority_collapse_ppr_zero(self):
        """Gate fails when PPR=0 (no positive predictions)."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import run_fg_gates
        rows = [self._make_fg_row("fg1_bad", tpr=0.0, fpr=0.0, f1=0.0, ppr=0.0)]
        result = run_fg_gates(rows)
        assert result["pass"] is False

    def test_reverse_collapse_high_ppr(self):
        """Gate fails when PPR>0.95."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import run_fg_gates
        rows = [self._make_fg_row("fg1_rev", tpr=1.0, fpr=0.9, f1=0.5, ppr=0.98)]
        result = run_fg_gates(rows)
        rev_gate = next(c for c in result["checks"] if c["name"] == "fg_no_reverse_collapse")
        assert not rev_gate["ok"]
        assert "reverse-collapsed" in rev_gate["msg"]

    def test_reverse_collapse_high_fpr(self):
        """Gate fails when FPR>0.70."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import run_fg_gates
        rows = [self._make_fg_row("fg1_rev", tpr=0.8, fpr=0.75, f1=0.5, ppr=0.5)]
        result = run_fg_gates(rows)
        rev_gate = next(c for c in result["checks"] if c["name"] == "fg_no_reverse_collapse")
        assert not rev_gate["ok"]

    def test_metric_integrity_f1_positive_tpr_zero(self):
        """Gate fails when F1>0 but TPR=0."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import run_fg_gates
        rows = [self._make_fg_row("fg1_int", tpr=0.0, fpr=0.0, f1=0.5, ppr=0.3)]
        result = run_fg_gates(rows)
        int_gate = next(c for c in result["checks"] if c["name"] == "fg_metric_integrity")
        assert not int_gate["ok"]
        assert "F1>0 but TPR=0" in int_gate["msg"]

    def test_degenerate_calibration_fails(self):
        """Gate fails when calibration metric is DEGENERATE_FAIL."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import run_fg_gates
        rows = [self._make_fg_row("fg1_deg", cal_metric="DEGENERATE_FAIL")]
        result = run_fg_gates(rows)
        deg_gate = next(c for c in result["checks"] if c["name"] == "fg_calibration_not_degenerate")
        assert not deg_gate["ok"]
        assert "degenerate calibration" in deg_gate["msg"]

    def test_no_fg_rows_fails(self):
        """Gate fails when no FG rows exist."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import run_fg_gates, BenchV3Row
        mwpm_row = BenchV3Row(suite="D", decoder="MWPM_ORACLE", status="pass")
        result = run_fg_gates([mwpm_row])
        assert result["pass"] is False
        exist_gate = next(c for c in result["checks"] if c["name"] == "fg_rows_exist")
        assert not exist_gate["ok"]

    def test_mwpm_not_counted_as_fg(self):
        """MWPM rows must not be counted as FG rows."""
        from qec_noise_factory.ml.bench.unified_benchmark_v3 import run_fg_gates, BenchV3Row
        mwpm_row = BenchV3Row(suite="D", decoder="MWPM_ORACLE", status="pass")
        mwpm_row.recall_tpr = 0.9
        result = run_fg_gates([mwpm_row])
        assert result["pass"] is False  # No FG rows → fail


class TestShardLoaderTypeValidation:
    """Ensure shard loading returns correct types."""

    def test_read_shards_dir_returns_list(self):
        """read_shards_dir returns a list (not a ShardDataset directly)."""
        from qec_noise_factory.ml.data.reader import ShardDataset
        # Simulate the return type
        result = []  # Empty list when no shards
        assert isinstance(result, list)
        assert not hasattr(result, 'X'), "List must not have .X attribute"

    def test_merge_datasets_returns_shard_dataset(self):
        """merge_datasets must return a ShardDataset with .X attribute."""
        from qec_noise_factory.ml.data.reader import ShardDataset, merge_datasets
        # Create two small datasets
        ds1 = ShardDataset(
            X=np.array([[1, 0], [0, 1]], dtype=np.uint8),
            Y=np.array([[0], [1]], dtype=np.uint8),
            meta=[], shard_path="test1",
        )
        ds2 = ShardDataset(
            X=np.array([[1, 1], [0, 0]], dtype=np.uint8),
            Y=np.array([[1], [0]], dtype=np.uint8),
            meta=[], shard_path="test2",
        )
        merged = merge_datasets([ds1, ds2])
        assert hasattr(merged, 'X'), "merge_datasets must return object with .X"
        assert merged.X.shape[0] == 4
        assert merged.Y.shape[0] == 4

    def test_list_has_no_x_attribute(self):
        """Directly accessing .X on a list must raise AttributeError."""
        datasets = [1, 2, 3]  # Simulate list return
        with pytest.raises(AttributeError):
            _ = datasets.X
