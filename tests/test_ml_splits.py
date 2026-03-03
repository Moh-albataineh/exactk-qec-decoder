"""
Tests for ML Split Policies — Day 15

Covers:
  - WITHIN_MODEL: random split, all blocks assigned
  - CROSS_MODEL: no physics_hash leakage
  - OOD_P_RANGE: no p-bucket leakage
  - Edge cases: single block, empty groups
"""
from __future__ import annotations

import json
import pytest

from qec_noise_factory.ml.data.schema import ShardMeta
from qec_noise_factory.ml.data.splits import (
    SplitPolicy,
    SplitResult,
    split_blocks,
    split_within_model,
    split_cross_model,
    split_ood_p_range,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(
    sample_key: str = "k",
    p: float = 0.05,
    physics_hash: str = "hash_A",
    record_count: int = 100,
    pack_name: str = "test",
) -> ShardMeta:
    return ShardMeta(
        schema_version=1,
        pack_name=pack_name,
        sample_key=sample_key,
        attempt_id="att",
        seed=42,
        shots=record_count,
        num_detectors=2,
        num_observables=1,
        det_bytes_per_shot=1,
        obs_bytes_per_shot=1,
        record_start=0,
        record_count=record_count,
        params_canonical=json.dumps({"circuit": {"p": p}}),
        p=p,
        physics_model_name="sym",
        physics_hash=physics_hash,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWithinModel:
    def test_all_blocks_assigned(self):
        blocks = [_make_block(f"k{i}") for i in range(10)]
        result = split_within_model(blocks, train_ratio=0.8)
        assert len(result.train_blocks) + len(result.test_blocks) == 10
        assert result.policy == SplitPolicy.WITHIN_MODEL

    def test_deterministic(self):
        blocks = [_make_block(f"k{i}") for i in range(10)]
        r1 = split_within_model(blocks, seed=42)
        r2 = split_within_model(blocks, seed=42)
        assert [b.sample_key for b in r1.train_blocks] == [b.sample_key for b in r2.train_blocks]

    def test_train_ratio(self):
        blocks = [_make_block(f"k{i}") for i in range(100)]
        result = split_within_model(blocks, train_ratio=0.7)
        assert len(result.train_blocks) == 70
        assert len(result.test_blocks) == 30


class TestCrossModel:
    def test_no_hash_leakage(self):
        blocks = [
            _make_block("k1", physics_hash="hash_A"),
            _make_block("k2", physics_hash="hash_A"),
            _make_block("k3", physics_hash="hash_B"),
            _make_block("k4", physics_hash="hash_B"),
            _make_block("k5", physics_hash="hash_C"),
        ]
        result = split_cross_model(blocks, train_ratio=0.6)
        assert result.leakage_check(), "Cross-model split has leakage!"

        train_hashes = {b.physics_hash for b in result.train_blocks}
        test_hashes = {b.physics_hash for b in result.test_blocks}
        assert len(train_hashes & test_hashes) == 0

    def test_all_blocks_assigned(self):
        blocks = [_make_block(f"k{i}", physics_hash=f"h{i%3}") for i in range(9)]
        result = split_cross_model(blocks, train_ratio=0.7)
        total = len(result.train_blocks) + len(result.test_blocks)
        assert total == 9


class TestOODPRange:
    def test_no_p_overlap(self):
        blocks = [
            _make_block("k1", p=0.01),
            _make_block("k2", p=0.02),
            _make_block("k3", p=0.15),
            _make_block("k4", p=0.3),
            _make_block("k5", p=0.5),
        ]
        result = split_ood_p_range(blocks, test_p_lo=0.1, test_p_hi=0.6)
        assert result.leakage_check()

        # Train should have p < 0.1 blocks
        for b in result.train_blocks:
            assert b.p < 0.1

        # Test should have 0.1 <= p <= 0.6
        for b in result.test_blocks:
            assert 0.1 <= b.p <= 0.6

    def test_fallback_when_all_same_range(self):
        blocks = [_make_block(f"k{i}", p=0.05) for i in range(5)]
        # All p=0.05 < test_p_lo=0.1, so no test blocks → fallback
        result = split_ood_p_range(blocks, test_p_lo=0.1, test_p_hi=0.6)
        assert len(result.train_blocks) + len(result.test_blocks) == 5


class TestSplitDispatch:
    def test_dispatch_within(self):
        blocks = [_make_block(f"k{i}") for i in range(5)]
        result = split_blocks(blocks, SplitPolicy.WITHIN_MODEL)
        assert result.policy == SplitPolicy.WITHIN_MODEL

    def test_dispatch_cross(self):
        blocks = [_make_block(f"k{i}", physics_hash=f"h{i}") for i in range(5)]
        result = split_blocks(blocks, SplitPolicy.CROSS_MODEL)
        assert result.policy == SplitPolicy.CROSS_MODEL

    def test_dispatch_ood(self):
        blocks = [
            _make_block("k1", p=0.01),
            _make_block("k2", p=0.3),
        ]
        result = split_blocks(blocks, SplitPolicy.OOD_P_RANGE, test_p_lo=0.1, test_p_hi=0.6)
        assert result.policy == SplitPolicy.OOD_P_RANGE
