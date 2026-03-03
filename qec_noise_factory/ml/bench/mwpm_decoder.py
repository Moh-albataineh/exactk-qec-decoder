"""
MWPM Decoder Wrapper — Day 26 (refactored Day 27)

Wraps PyMatching for fair comparison with GNN decoder.
Rebuilds Stim circuits from shard metadata, extracts DEM, decodes batches.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import stim

# Day 27: use shared rebuild utility (single source of truth)
from qec_noise_factory.ml.stim.rebuild import (
    rebuild_stim_circuit,
    params_from_canonical as params_from_meta,
    circuit_cache_key as circuit_key_from_meta,
)



class MWPMDecoder:
    """
    MWPM decoder using PyMatching.

    Builds matching graph from Stim DEM, decodes detection event batches.
    Caches matching graphs per circuit configuration.

    Usage:
        decoder = MWPMDecoder()
        decoder.build(distance=5, rounds=5, p=0.001, basis="X")
        y_hat = decoder.decode_batch(X)  # X: (N, num_detectors) bool
    """

    def __init__(self):
        self._cache: Dict[Tuple, Any] = {}
        self._current_matching = None
        self._current_key = None
        self._build_times: Dict[Tuple, float] = {}
        self._num_detectors: int = 0
        self._num_observables: int = 0

    def build(
        self,
        distance: int,
        rounds: int,
        p: float,
        basis: str,
        noise_model: str = "baseline_symmetric",
    ) -> float:
        """
        Build matching graph for given circuit parameters.

        Returns build time in seconds. Uses cache if available.
        """
        import pymatching

        key = ("surface_code_rotated_memory", distance, rounds, basis.upper(), noise_model, p)

        if key in self._cache:
            self._current_matching = self._cache[key]["matching"]
            self._current_key = key
            self._num_detectors = self._cache[key]["num_detectors"]
            self._num_observables = self._cache[key]["num_observables"]
            return 0.0  # cached, no build time

        t0 = time.perf_counter()
        circuit = rebuild_stim_circuit(distance, rounds, p, basis, noise_model)
        dem = circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        build_time = time.perf_counter() - t0

        self._num_detectors = circuit.num_detectors
        self._num_observables = circuit.num_observables

        self._cache[key] = {
            "matching": matching,
            "num_detectors": self._num_detectors,
            "num_observables": self._num_observables,
            "build_time": build_time,
        }
        self._current_matching = matching
        self._current_key = key
        self._build_times[key] = build_time

        return build_time

    def build_from_meta(self, params_canonical: str) -> float:
        """Build matching graph from shard metadata params_canonical."""
        params = params_from_meta(params_canonical)
        if params["distance"] == 0:
            raise ValueError(f"Cannot extract valid circuit params from: {params_canonical}")
        return self.build(**params)

    def decode_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Decode a batch of detection events.

        Args:
            X: (N, num_detectors) bool array

        Returns:
            y_hat: (N, num_observables) bool array — predicted observable flips
        """
        if self._current_matching is None:
            raise RuntimeError("Must call build() before decode_batch()")

        N = X.shape[0]
        y_hat = np.zeros((N, self._num_observables), dtype=bool)

        for i in range(N):
            prediction = self._current_matching.decode(X[i].astype(np.uint8))
            y_hat[i] = prediction[:self._num_observables]

        return y_hat

    def decode_batch_fast(self, X: np.ndarray) -> np.ndarray:
        """
        Decode using PyMatching's batch decode (if available).

        Falls back to per-sample decode if batch not supported.
        """
        if self._current_matching is None:
            raise RuntimeError("Must call build() before decode_batch_fast()")

        try:
            # PyMatching 2.x supports decode_batch
            predictions = self._current_matching.decode_batch(X.astype(np.uint8))
            return predictions[:, :self._num_observables].astype(bool)
        except (AttributeError, TypeError):
            return self.decode_batch(X)

    @property
    def num_detectors(self) -> int:
        return self._num_detectors

    @property
    def num_observables(self) -> int:
        return self._num_observables

    @property
    def graph_build_time(self) -> float:
        if self._current_key and self._current_key in self._build_times:
            return self._build_times[self._current_key]
        return 0.0

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self):
        self._cache.clear()
        self._build_times.clear()
        self._current_matching = None
        self._current_key = None
