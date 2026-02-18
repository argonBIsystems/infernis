"""Tests for model evaluator and drift detection."""

import numpy as np
import pytest

from infernis.training.evaluator import ModelEvaluator


@pytest.fixture
def evaluator():
    return ModelEvaluator()


class TestEvaluator:
    def test_perfect_predictions(self, evaluator):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        metrics = evaluator.evaluate(y_true, y_prob)
        assert metrics["auc_roc"] == 1.0
        assert metrics["n_samples"] == 6
        assert metrics["n_positive"] == 3

    def test_random_predictions_auc_around_half(self, evaluator):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.rand(1000)
        metrics = evaluator.evaluate(y_true, y_prob)
        # Random predictions should have AUC-ROC near 0.5
        assert 0.4 <= metrics["auc_roc"] <= 0.6

    def test_threshold_metrics_included(self, evaluator):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        metrics = evaluator.evaluate(y_true, y_prob)
        assert "0.5" in metrics["threshold_metrics"]
        assert "precision" in metrics["threshold_metrics"]["0.5"]

    def test_brier_score_bounded(self, evaluator):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        metrics = evaluator.evaluate(y_true, y_prob)
        assert 0.0 <= metrics["brier_score"] <= 1.0

    def test_calibration_curve(self, evaluator):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.15, 0.2, 0.25, 0.7, 0.75, 0.8, 0.85])
        curve = evaluator.compute_calibration_curve(y_true, y_prob, n_bins=5)
        assert len(curve["bin_edges"]) == 6
        assert len(curve["predicted_mean"]) == 5
        assert len(curve["actual_fraction"]) == 5


class TestDriftDetection:
    def test_no_drift_good_model(self, evaluator):
        # Good AUC and low ECE: pass metrics directly to check_drift
        good_metrics = {"auc_roc": 0.92, "ece": 0.03}
        drift = evaluator.check_drift(good_metrics)
        assert drift["drift_detected"] is False

    def test_drift_detected_low_auc(self, evaluator):
        bad_metrics = {"auc_roc": 0.60, "ece": 0.02}
        drift = evaluator.check_drift(bad_metrics)
        assert drift["drift_detected"] is True
        assert any("AUC-ROC" in a for a in drift["alerts"])

    def test_drift_detected_bad_calibration(self, evaluator):
        bad_metrics = {"auc_roc": 0.90, "ece": 0.10}
        drift = evaluator.check_drift(bad_metrics)
        assert drift["drift_detected"] is True
        assert any("Calibration" in a for a in drift["alerts"])

    def test_relative_drift_from_baseline(self, evaluator):
        current = {"auc_roc": 0.80, "ece": 0.03}
        baseline = {"auc_roc": 0.92}
        drift = evaluator.check_drift(current, baseline)
        assert drift["drift_detected"] is True
        assert any("dropped" in a for a in drift["alerts"])
