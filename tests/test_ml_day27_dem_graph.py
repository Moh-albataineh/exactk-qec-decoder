"""
Tests for DEM Graph Builder — Day 27

Unit tests for DEM extraction, weight formula, edge merging,
boundary handling, canonical ordering, deterministic hashing.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qec_noise_factory.ml.graph.dem_graph import (
    matching_weight,
    merge_probabilities,
    build_dem_graph,
    build_dem_graph_from_meta,
    compute_dem_graph_hash,
    save_dem_graph,
    load_dem_graph,
    DemBuildKey,
    DemGraphSpec,
    _extract_dem_edges,
)
from qec_noise_factory.ml.stim.rebuild import (
    rebuild_stim_circuit,
    params_from_canonical,
    circuit_cache_key,
)


# ---------------------------------------------------------------------------
# Weight formula
# ---------------------------------------------------------------------------

class TestWeightFormula:
    def test_finite_low_p(self):
        w = matching_weight(1e-6)
        assert math.isfinite(w)
        assert w > 0

    def test_finite_medium_p(self):
        w = matching_weight(0.01)
        assert math.isfinite(w)
        assert w > 0

    def test_finite_high_p(self):
        w = matching_weight(0.2)
        assert math.isfinite(w)
        assert w > 0

    def test_extreme_low(self):
        w = matching_weight(1e-15)
        assert math.isfinite(w)

    def test_extreme_high(self):
        w = matching_weight(0.999999)
        assert math.isfinite(w)

    def test_monotone_decreasing(self):
        """Lower error probability → higher matching weight."""
        w1 = matching_weight(0.001)
        w2 = matching_weight(0.01)
        w3 = matching_weight(0.1)
        assert w1 > w2 > w3


# ---------------------------------------------------------------------------
# Edge merging
# ---------------------------------------------------------------------------

class TestEdgeMerge:
    def test_single_prob(self):
        assert merge_probabilities([0.1]) == pytest.approx(0.1)

    def test_two_probs(self):
        # p_total = 1 - (1-0.1)*(1-0.2) = 1 - 0.72 = 0.28
        assert merge_probabilities([0.1, 0.2]) == pytest.approx(0.28)

    def test_three_probs(self):
        p = merge_probabilities([0.1, 0.1, 0.1])
        expected = 1.0 - (0.9 ** 3)
        assert p == pytest.approx(expected)

    def test_merged_weight_finite(self):
        p = merge_probabilities([0.01, 0.02, 0.03])
        w = matching_weight(p)
        assert math.isfinite(w)
        assert w > 0


# ---------------------------------------------------------------------------
# DEM graph build
# ---------------------------------------------------------------------------

class TestBuildDemGraph:
    def test_d3_basic(self):
        g = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
        assert g.num_detectors > 0
        assert g.edge_index.shape[0] == 2
        assert g.edge_index.shape[1] > 0
        assert g.edge_weight.shape[0] == g.edge_index.shape[1]
        assert g.edge_prob.shape[0] == g.edge_index.shape[1]

    def test_d5_basic(self):
        g = build_dem_graph(distance=5, rounds=5, p=0.001, basis="X")
        assert g.num_detectors == 120  # d=5 surface code
        assert g.edge_index.shape[1] > 0

    def test_weights_finite(self):
        g = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
        assert np.all(np.isfinite(g.edge_weight))
        assert np.all(np.isfinite(g.edge_prob))

    def test_probs_in_range(self):
        g = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
        assert np.all(g.edge_prob > 0)
        assert np.all(g.edge_prob < 1)

    def test_canonical_ordering(self):
        """All edges should have u < v."""
        g = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
        us = g.edge_index[0]
        vs = g.edge_index[1]
        assert np.all(us < vs), "Edges must be stored with u < v"

    def test_sorted_edges(self):
        """Edges should be lexicographically sorted."""
        g = build_dem_graph(distance=3, rounds=3, p=0.01, basis="Z")
        E = g.edge_index.shape[1]
        for i in range(E - 1):
            u1, v1 = g.edge_index[0, i], g.edge_index[1, i]
            u2, v2 = g.edge_index[0, i + 1], g.edge_index[1, i + 1]
            assert (u1, v1) <= (u2, v2), f"Edge {i} not sorted: ({u1},{v1}) > ({u2},{v2})"

    def test_boundary_handling(self):
        """DEM should produce boundary edges (1-detector terms)."""
        g = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
        # Boundary node = num_detectors
        boundary = g.num_detectors
        boundary_edges = np.any(g.edge_index == boundary, axis=0)
        assert boundary_edges.sum() > 0, "Should have boundary edges"
        assert g.has_boundary_node
        assert g.node_count == g.num_detectors + 1

    def test_dem_info_populated(self):
        g = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
        assert "num_terms" in g.dem_info
        assert "num_boundary_edges" in g.dem_info
        assert g.dem_info["num_terms"] > 0


# ---------------------------------------------------------------------------
# Deterministic hashing
# ---------------------------------------------------------------------------

class TestDeterministicHash:
    def test_same_params_same_hash(self):
        g1 = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
        g2 = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
        assert g1.dem_graph_hash == g2.dem_graph_hash
        assert len(g1.dem_graph_hash) == 64  # SHA-256

    def test_different_p_different_hash(self):
        g1 = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
        g2 = build_dem_graph(distance=3, rounds=3, p=0.02, basis="X")
        assert g1.dem_graph_hash != g2.dem_graph_hash

    def test_different_basis_different_hash(self):
        g1 = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
        g2 = build_dem_graph(distance=3, rounds=3, p=0.01, basis="Z")
        assert g1.dem_graph_hash != g2.dem_graph_hash

    def test_hash_stable_across_build_save_load(self):
        g = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
        with tempfile.TemporaryDirectory() as tmp:
            save_dem_graph(g, Path(tmp))
            # Rebuild and compare
            g2 = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
            assert g.dem_graph_hash == g2.dem_graph_hash


# ---------------------------------------------------------------------------
# Storage roundtrip
# ---------------------------------------------------------------------------

class TestStorage:
    def test_save_load_roundtrip(self):
        g = build_dem_graph(distance=3, rounds=3, p=0.01, basis="X")
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = save_dem_graph(g, Path(tmp))
            loaded = load_dem_graph(npz_path)
            assert loaded.dem_graph_hash == g.dem_graph_hash
            assert loaded.node_count == g.node_count
            assert loaded.num_detectors == g.num_detectors
            np.testing.assert_array_equal(loaded.edge_index, g.edge_index)
            np.testing.assert_array_almost_equal(loaded.edge_weight, g.edge_weight)
            np.testing.assert_array_almost_equal(loaded.edge_prob, g.edge_prob)


# ---------------------------------------------------------------------------
# Build from meta
# ---------------------------------------------------------------------------

class TestBuildFromMeta:
    def test_from_meta(self):
        meta = json.dumps({
            "circuit": {
                "type": "surface_code_rotated_memory",
                "distance": 3, "rounds": 3, "p": 0.01,
                "basis": "X", "noise_model": "baseline_symmetric",
                "physics_hash": "abc123",
            }
        })
        g = build_dem_graph_from_meta(meta)
        assert g.num_detectors > 0
        assert g.build_key.physics_hash == "abc123"


# ---------------------------------------------------------------------------
# Shared rebuild utility (Day 27 refactor)
# ---------------------------------------------------------------------------

class TestSharedRebuild:
    def test_rebuild_deterministic(self):
        c1 = rebuild_stim_circuit(3, 3, 0.01, "X")
        c2 = rebuild_stim_circuit(3, 3, 0.01, "X")
        assert str(c1) == str(c2)

    def test_params_from_canonical(self):
        meta = json.dumps({"circuit": {"distance": 5, "rounds": 5, "p": 0.001, "basis": "z", "noise_model": "baseline_symmetric"}})
        params = params_from_canonical(meta)
        assert params["distance"] == 5
        assert params["basis"] == "Z"

    def test_cache_key(self):
        meta = json.dumps({"circuit": {"type": "sc", "distance": 3, "rounds": 3, "basis": "X", "noise_model": "bs", "p": 0.01}})
        key = circuit_cache_key(meta)
        assert key == ("sc", 3, 3, "X", "bs", 0.01)
