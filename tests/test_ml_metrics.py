"""
Tests for ML Metrics — Day 16

Verifies accuracy, BER, balanced_accuracy on known cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from qec_noise_factory.ml.metrics.classification import compute_metrics


class TestAllCorrect:
    def test_perfect_accuracy(self):
        y_true = np.array([[True], [False], [True], [False]])
        y_pred = np.array([[True], [False], [True], [False]])
        m = compute_metrics(y_true, y_pred)
        assert m["macro_accuracy"] == 1.0
        assert m["macro_ber"] == 0.0

    def test_balanced_accuracy_perfect(self):
        y_true = np.array([[True], [False], [True], [False]])
        y_pred = np.array([[True], [False], [True], [False]])
        m = compute_metrics(y_true, y_pred)
        assert m["macro_balanced_accuracy"] == 1.0


class TestAllWrong:
    def test_zero_accuracy(self):
        y_true = np.array([[True], [True], [True], [True]])
        y_pred = np.array([[False], [False], [False], [False]])
        m = compute_metrics(y_true, y_pred)
        assert m["macro_accuracy"] == 0.0
        assert m["macro_ber"] == 1.0

    def test_balanced_accuracy_all_wrong(self):
        y_true = np.array([[True], [False], [True], [False]])
        y_pred = np.array([[False], [True], [False], [True]])
        m = compute_metrics(y_true, y_pred)
        assert m["macro_balanced_accuracy"] == 0.0


class TestHalfCorrect:
    def test_50_percent(self):
        y_true = np.array([[True], [True], [False], [False]])
        y_pred = np.array([[True], [False], [False], [True]])
        m = compute_metrics(y_true, y_pred)
        assert m["macro_accuracy"] == pytest.approx(0.5)
        assert m["macro_ber"] == pytest.approx(0.5)


class TestMultiObservable:
    def test_two_observables(self):
        y_true = np.array([[True, False], [False, True]])
        y_pred = np.array([[True, True], [False, False]])
        m = compute_metrics(y_true, y_pred)
        # obs_0: perfect, obs_1: all wrong
        assert m["obs_0_accuracy"] == 1.0
        assert m["obs_1_accuracy"] == 0.0
        assert m["macro_accuracy"] == pytest.approx(0.5)
        assert m["num_observables"] == 2


class TestEdgeCases:
    def test_1d_input(self):
        y_true = np.array([True, False, True])
        y_pred = np.array([True, False, True])
        m = compute_metrics(y_true, y_pred)
        assert m["macro_accuracy"] == 1.0

    def test_all_negative_true(self):
        """When all true labels are False, trivial predict-zero gets 100%."""
        y_true = np.array([[False], [False], [False]])
        y_pred = np.array([[False], [False], [False]])
        m = compute_metrics(y_true, y_pred)
        assert m["macro_accuracy"] == 1.0
        assert m["obs_0_tnr"] == 1.0
