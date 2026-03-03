"""
Unit tests for Day 74: Selector v6 + MLOps (JSONL, checkpoints, receipts).
"""

import json
import os
import tempfile
import pytest
import numpy as np
from pathlib import Path
from qec_noise_factory.ml.ops.checkpoint_selection import (
    select_epoch_for_seed,
    rolling_median,
    compute_slice_clean_baseline,
    compute_slice_clean_null_p05,
    DNHGate,
    EpochLogger,
    load_day_artifacts_auto,
    write_selection_receipt,
    copy_best_checkpoint,
    cleanup_unselected_checkpoints,
    _normalize_epoch,
)


# ---- Helpers ----

def _ep(epoch, g1, sc, md, tg=0.05, loss=1.0):
    return {
        "epoch": epoch, "G1_aligned": g1, "slice_clean": sc,
        "mean_drop": md, "topo_TG": tg, "loss": loss,
    }


# ====================================================================
# DNH Gate (unchanged)
# ====================================================================

class TestDNHGate:

    def test_initial_off(self):
        assert DNHGate().state == 0

    def test_warmup_forces_off(self):
        g = DNHGate(warmup_epochs=5)
        g.update(0.05, epoch=3)
        assert g.state == 0

    def test_on_above_tau(self):
        g = DNHGate(tau_on=0.025)
        g.update(0.03, epoch=6)
        assert g.state == 1

    def test_off_below_tau(self):
        g = DNHGate(tau_off=0.020, tau_on=0.025)
        g.update(0.03, epoch=6)
        g.update(0.015, epoch=7)
        assert g.state == 0

    def test_deadband_stays(self):
        g = DNHGate(tau_off=0.020, tau_on=0.025)
        g.update(0.03, epoch=6)
        g.update(0.022, epoch=7)
        assert g.state == 1

    def test_iso_weight_decay(self):
        g = DNHGate(lambda_base=0.10, decay_rate=0.85, decay_start_epoch=8)
        g.update(0.03, epoch=6)
        assert g.compute_iso_weight(9) == pytest.approx(0.10 * 0.85)


# ====================================================================
# Rolling median (SC + TG)
# ====================================================================

class TestRollingMedian:

    def test_slice_clean_roll(self):
        metrics = [_ep(6, 0, sc=0.45, md=0), _ep(7, 0, sc=0.55, md=0),
                   _ep(8, 0, sc=0.50, md=0)]
        assert rolling_median(metrics, key="slice_clean", window=3)[8] == pytest.approx(0.50)

    def test_tg_roll(self):
        metrics = [_ep(6, 0, 0, 0, tg=0.02), _ep(7, 0, 0, 0, tg=0.09),
                   _ep(8, 0, 0, 0, tg=0.05)]
        assert rolling_median(metrics, key="topo_TG", window=3)[8] == pytest.approx(0.05)

    def test_g1_roll(self):
        metrics = [_ep(6, 0.10, 0, 0), _ep(7, 0.20, 0, 0), _ep(8, 0.30, 0, 0)]
        assert rolling_median(metrics, key="G1_aligned", window=3)[8] == pytest.approx(0.20)


# ====================================================================
# Selector v6: CLEAN → max(tg_roll)
# ====================================================================

class TestCleanMaxTGRoll:

    def test_picks_best_tg_roll(self):
        metrics = [_ep(6, 0.01, 0.55, 0.01, 0.06), _ep(7, 0.02, 0.56, 0.01, 0.09),
                   _ep(8, 0.01, 0.54, 0.01, 0.07)]
        result = select_epoch_for_seed(metrics, slice_floor=0.500, tg_floor=-0.015)
        assert result["selection_mode"] == "CLEAN_MAX_TG"
        assert "tg_roll" in result


# ====================================================================
# LEAKY → argmin(g1roll)
# ====================================================================

class TestLeakyMinG1Roll:

    def test_uses_g1roll(self):
        metrics = [_ep(6, 0.04, 0.55, 0.01, 0.05), _ep(7, 0.04, 0.56, 0.01, 0.06),
                   _ep(8, 0.03, 0.57, 0.01, 0.07)]
        result = select_epoch_for_seed(metrics, tau_clean=0.025, tau_clean_hi=0.035,
                                        slice_floor=0.500, tg_floor=-0.015)
        assert result["selection_mode"] == "LEAKY_MIN_G1ROLL"


# ====================================================================
# TOPO_FAIL → max(tg_roll)
# ====================================================================

class TestTopoFailMaxTGRoll:

    def test_picks_max_tg_roll(self):
        metrics = [_ep(6, 0.01, 0.40, 0.01, 0.02), _ep(7, 0.05, 0.42, 0.01, 0.09),
                   _ep(8, 0.03, 0.41, 0.01, 0.05)]
        result = select_epoch_for_seed(metrics, slice_floor=0.500, tg_floor=-0.015)
        assert result["selection_mode"] == "TOPO_FAIL_MAX_TG"

    def test_never_argmin_g1(self):
        metrics = [_ep(6, 0.001, 0.40, 0.01, 0.02), _ep(7, 0.05, 0.40, 0.01, 0.09)]
        result = select_epoch_for_seed(metrics, slice_floor=0.500)
        assert "MIN_G1" not in result["selection_mode"]


# ====================================================================
# Dual-cap
# ====================================================================

class TestDualCap:

    def test_g1_spike_excludes_clean(self):
        metrics = [_ep(6, 0.01, 0.55, 0.01, 0.05), _ep(7, 0.01, 0.56, 0.01, 0.06),
                   _ep(8, 0.04, 0.57, 0.01, 0.07)]
        result = select_epoch_for_seed(metrics, tau_clean=0.025, tau_clean_hi=0.035,
                                        slice_floor=0.500, tg_floor=-0.015)
        if result["selection_mode"] == "CLEAN_MAX_TG":
            assert result["G1_aligned"] <= 0.035


# ====================================================================
# Smoothed survival
# ====================================================================

class TestSmoothedSurvival:

    def test_tg_roll_passes_when_instant_fails(self):
        metrics = [_ep(6, 0.01, 0.55, 0.01, 0.05), _ep(7, 0.01, 0.55, 0.01, -0.02),
                   _ep(8, 0.01, 0.55, 0.01, 0.05)]
        result = select_epoch_for_seed(metrics, slice_floor=0.500, tg_floor=-0.015)
        assert result["n_surviving"] >= 1


# ====================================================================
# Empirical null floor
# ====================================================================

class TestEmpiricalNull:

    def test_deterministic(self):
        rng = np.random.RandomState(42)
        Y = rng.randint(0, 2, size=500).astype(bool)
        K = rng.randint(5, 15, size=500)
        f1 = compute_slice_clean_null_p05(Y, K, n_shuffles=50, seed=0)
        f2 = compute_slice_clean_null_p05(Y, K, n_shuffles=50, seed=0)
        assert f1 == f2

    def test_capped_at_0500(self):
        rng = np.random.RandomState(123)
        Y = rng.randint(0, 2, size=200).astype(bool)
        K = rng.randint(5, 15, size=200)
        assert compute_slice_clean_null_p05(Y, K, n_shuffles=50, seed=0) <= 0.500


# ====================================================================
# Spike delta
# ====================================================================

class TestSpikeDelta:

    def test_computed_correctly(self):
        metrics = [_ep(6, 0.01, 0.55, 0.01, 0.05), _ep(7, 0.05, 0.56, 0.01, 0.06)]
        result = select_epoch_for_seed(metrics, slice_floor=0.500)
        assert abs(result["g1_spike_delta"] - (result["G1_aligned"] - result["g1_roll"])) < 1e-5


# ====================================================================
# JSONL Logger
# ====================================================================

class TestEpochLogger:

    def test_writes_valid_json_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                          delete=False) as f:
            path = f.name
        try:
            with EpochLogger(path) as logger:
                logger.log_epoch({"epoch": 1, "loss": 0.5, "arm": "Control", "seed": 42})
                logger.log_epoch({"epoch": 2, "loss": 0.3, "arm": "Control", "seed": 42})

            with open(path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            assert len(lines) == 2
            for line in lines:
                rec = json.loads(line)
                assert "epoch" in rec
                assert "loss" in rec
        finally:
            os.unlink(path)

    def test_flushes_immediately(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                          delete=False) as f:
            path = f.name
        try:
            logger = EpochLogger(path)
            logger.log_epoch({"epoch": 1, "val": 1.0})
            # Should be readable before close
            with open(path, "r") as f:
                assert len(f.readlines()) == 1
            logger.close()
        finally:
            os.unlink(path)


# ====================================================================
# JSONL format loading
# ====================================================================

class TestJSONLLoading:

    def test_load_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jpath = Path(tmpdir) / "metrics_42.jsonl"
            with open(jpath, "w") as f:
                for ep in range(1, 8):
                    rec = {"epoch": ep, "arm": "Control", "seed": 42,
                           "G1_aligned": 0.01 * ep, "slice_clean": 0.5,
                           "mean_drop": 0.0, "topo_TG": 0.05, "loss": 1.0}
                    f.write(json.dumps(rec) + "\n")
            merged = load_day_artifacts_auto(tmpdir)
            assert "Control_42" in merged
            assert len(merged["Control_42"]) == 7


# ====================================================================
# Day 69 format normalization
# ====================================================================

class TestDay69Format:

    def test_normalize_g1_raw_probe(self):
        raw = {"epoch": 6, "G1_raw_probe": 0.042, "topo_slice_clean": 0.55,
               "topo_mean_drop": 0.02, "topo_TG": 0.08, "loss": 1.0}
        norm = _normalize_epoch(raw)
        assert norm["G1_aligned"] == 0.042
        assert norm["slice_clean"] == 0.55
        assert norm["mean_drop"] == 0.02


# ====================================================================
# Selection receipt
# ====================================================================

class TestSelectionReceipt:

    def test_write_receipt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            receipt = {"selector_version": "v6", "chosen_epoch": 8, "seed": 42}
            path = write_selection_receipt(receipt, tmpdir, seed=42)
            assert path.exists()
            loaded = json.loads(path.read_text())
            assert loaded["chosen_epoch"] == 8


# ====================================================================
# Checkpoint copy + cleanup
# ====================================================================

class TestCheckpointUtils:

    def test_copy_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "ckpts"
            ckpt_dir.mkdir()
            # Create fake checkpoints
            for ep in [6, 7, 8]:
                (ckpt_dir / f"ckpt_42_ep{ep}.pt").write_text(f"model_ep{ep}")

            # Copy
            dst = copy_best_checkpoint(ckpt_dir, tmpdir, seed=42, chosen_epoch=7)
            assert dst.exists()
            assert dst.read_text() == "model_ep7"

            # Cleanup
            deleted = cleanup_unselected_checkpoints(ckpt_dir, 42, chosen_epoch=7)
            assert len(deleted) == 2
            assert (ckpt_dir / "ckpt_42_ep7.pt").exists()  # kept
            assert not (ckpt_dir / "ckpt_42_ep6.pt").exists()  # deleted

    def test_missing_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                copy_best_checkpoint(tmpdir, tmpdir, seed=42, chosen_epoch=99)


# ====================================================================
# Determinism
# ====================================================================

class TestDeterminism:

    def test_same_output(self):
        metrics = [_ep(6, 0.01, 0.55, 0.01, 0.05), _ep(7, 0.02, 0.58, 0.01, 0.09),
                   _ep(8, 0.005, 0.53, 0.01, 0.07)]
        results = [select_epoch_for_seed(metrics) for _ in range(10)]
        assert all(r == results[0] for r in results)


# ====================================================================
# Baseline sanity
# ====================================================================

class TestBaseline:

    def test_near_050(self):
        assert 0.48 <= compute_slice_clean_baseline() <= 0.52


# ====================================================================
# Active epoch
# ====================================================================

class TestActiveEpoch:

    def test_warmup_ignored(self):
        metrics = [_ep(1, 0.001, 0.60, 0.02, 0.10), _ep(6, 0.02, 0.55, 0.01)]
        assert select_epoch_for_seed(metrics, active_epoch_min=6)["epoch"] == 6

    def test_no_active_raises(self):
        with pytest.raises(ValueError):
            select_epoch_for_seed([_ep(1, 0.001, 0.60, 0.02)], active_epoch_min=6)
