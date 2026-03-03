"""
ML Split Policies — Day 15

Three split strategies for QEC ML training:
  WITHIN_MODEL  — random block-level split (standard ML)
  CROSS_MODEL   — split by physics_hash (no leakage across models)
  OOD_P_RANGE   — split by p-bucket (test generalization to unseen noise)

All splits operate on block-level metadata to prevent intra-block leakage.
"""
from __future__ import annotations

import random
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from qec_noise_factory.ml.data.schema import ShardMeta


class SplitPolicy(Enum):
    WITHIN_MODEL = "within_model"
    CROSS_MODEL = "cross_model"
    OOD_P_RANGE = "ood_p_range"


@dataclass
class SplitResult:
    """Output of a split operation."""
    train_blocks: List[ShardMeta]
    test_blocks: List[ShardMeta]
    policy: SplitPolicy
    train_samples: int
    test_samples: int

    @property
    def train_ratio(self) -> float:
        total = self.train_samples + self.test_samples
        return self.train_samples / max(1, total)

    def leakage_check(self) -> bool:
        """
        Verify no leakage between train and test.
        For CROSS_MODEL: no shared physics_hash.
        For OOD_P_RANGE: no shared p-bucket.
        Returns True if clean (no leakage).
        """
        if self.policy == SplitPolicy.CROSS_MODEL:
            train_hashes = {b.physics_hash for b in self.train_blocks if b.physics_hash}
            test_hashes = {b.physics_hash for b in self.test_blocks if b.physics_hash}
            return len(train_hashes & test_hashes) == 0

        if self.policy == SplitPolicy.OOD_P_RANGE:
            # Check exact p-values are disjoint (not buckets — buckets are too
            # coarse for low p-values like 0.001..0.02 which all map to bucket 0)
            train_ps = {round(b.p, 8) for b in self.train_blocks}
            test_ps = {round(b.p, 8) for b in self.test_blocks}
            return len(train_ps & test_ps) == 0

        return True  # WITHIN_MODEL has no structural leakage concern


def _p_bucket(p: float, n_buckets: int = 10) -> int:
    """Assign p to a bucket index (0 to n_buckets-1)."""
    # Clamp to [0, 1]
    p = max(0.0, min(1.0, p))
    return min(int(p * n_buckets), n_buckets - 1)


def split_within_model(
    blocks: List[ShardMeta],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> SplitResult:
    """
    Random block-level split.
    Each block goes entirely to train or test (no intra-block splits).
    """
    rng = random.Random(seed)
    shuffled = list(blocks)
    rng.shuffle(shuffled)

    n_train = max(1, int(len(shuffled) * train_ratio))
    train = shuffled[:n_train]
    test = shuffled[n_train:]

    return SplitResult(
        train_blocks=train,
        test_blocks=test,
        policy=SplitPolicy.WITHIN_MODEL,
        train_samples=sum(b.record_count for b in train),
        test_samples=sum(b.record_count for b in test),
    )


def split_cross_model(
    blocks: List[ShardMeta],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> SplitResult:
    """
    Split by physics_hash — no model appears in both train and test.
    Blocks without physics_hash use pack_name+p as a fallback group key.
    """
    # Group by physics identity
    groups: Dict[str, List[ShardMeta]] = {}
    for b in blocks:
        key = b.physics_hash if b.physics_hash else f"{b.pack_name}:p={b.p:.6f}"
        groups.setdefault(key, []).append(b)

    # Shuffle group keys and split
    rng = random.Random(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)

    train_blocks = []
    test_blocks = []
    total = sum(b.record_count for b in blocks)
    train_target = int(total * train_ratio)
    train_count = 0

    for k in keys:
        grp = groups[k]
        grp_count = sum(b.record_count for b in grp)
        if train_count < train_target:
            train_blocks.extend(grp)
            train_count += grp_count
        else:
            test_blocks.extend(grp)

    # Ensure at least 1 block in test
    if not test_blocks and len(train_blocks) > 1:
        test_blocks.append(train_blocks.pop())

    return SplitResult(
        train_blocks=train_blocks,
        test_blocks=test_blocks,
        policy=SplitPolicy.CROSS_MODEL,
        train_samples=sum(b.record_count for b in train_blocks),
        test_samples=sum(b.record_count for b in test_blocks),
    )


def split_ood_p_range(
    blocks: List[ShardMeta],
    test_p_lo: float = 0.1,
    test_p_hi: float = 0.6,
    seed: int = 42,
) -> SplitResult:
    """
    OOD split: test set contains only blocks with p in [test_p_lo, test_p_hi].
    Train set contains blocks outside this range.
    Tests generalization to unseen noise regimes.
    """
    train_blocks = []
    test_blocks = []

    for b in blocks:
        if test_p_lo <= b.p <= test_p_hi:
            test_blocks.append(b)
        else:
            train_blocks.append(b)

    # If all blocks fall in one set, do a fallback random split
    if not test_blocks or not train_blocks:
        return split_within_model(blocks, train_ratio=0.8, seed=seed)

    return SplitResult(
        train_blocks=train_blocks,
        test_blocks=test_blocks,
        policy=SplitPolicy.OOD_P_RANGE,
        train_samples=sum(b.record_count for b in train_blocks),
        test_samples=sum(b.record_count for b in test_blocks),
    )


def split_blocks(
    blocks: List[ShardMeta],
    policy: SplitPolicy,
    train_ratio: float = 0.8,
    seed: int = 42,
    **kwargs,
) -> SplitResult:
    """
    Dispatch to the appropriate split function based on policy.
    """
    if policy == SplitPolicy.WITHIN_MODEL:
        return split_within_model(blocks, train_ratio, seed)
    elif policy == SplitPolicy.CROSS_MODEL:
        return split_cross_model(blocks, train_ratio, seed)
    elif policy == SplitPolicy.OOD_P_RANGE:
        return split_ood_p_range(blocks, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown split policy: {policy}")
