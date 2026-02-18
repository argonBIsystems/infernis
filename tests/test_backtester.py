"""Tests for the historical backtesting framework."""

import numpy as np
import pytest

from infernis.training.backtester import HistoricalBacktester
from infernis.training.evaluator import ModelEvaluator


@pytest.fixture
def synthetic_data():
    """Create synthetic data spanning 3 years with separable classes."""
    rng = np.random.default_rng(42)
    n_per_year = 200
    years_list = [2020, 2021, 2022]

    X_parts, y_parts, year_parts, zone_parts = [], [], [], []
    zones = ["CWH", "BWBS", "SBS"]

    for year in years_list:
        n_pos = 20
        n_neg = n_per_year - n_pos
        X_pos = rng.standard_normal((n_pos, 10)).astype(np.float32) + 1.5
        X_neg = rng.standard_normal((n_neg, 10)).astype(np.float32)
        X_parts.append(np.vstack([X_pos, X_neg]))
        y_parts.append(np.concatenate([np.ones(n_pos), np.zeros(n_neg)]))
        year_parts.append(np.full(n_per_year, year))
        zone_parts.append(rng.choice(zones, size=n_per_year))

    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.float32)
    years = np.concatenate(year_parts).astype(np.int32)
    bec_zones = np.concatenate(zone_parts)

    return X, y, years, bec_zones


class TestTemporalCV:
    def test_produces_per_year_results(self, synthetic_data):
        X, y, years, bec_zones = synthetic_data
        bt = HistoricalBacktester()

        results = bt.temporal_cv(
            X,
            y,
            years,
            train_start=2020,
            test_years=[2021, 2022],
            n_rounds=20,
        )

        assert len(results) == 2
        assert results[0]["test_year"] == 2021
        assert results[1]["test_year"] == 2022

    def test_metrics_present(self, synthetic_data):
        X, y, years, bec_zones = synthetic_data
        bt = HistoricalBacktester()

        results = bt.temporal_cv(
            X,
            y,
            years,
            train_start=2020,
            test_years=[2022],
            n_rounds=20,
        )

        assert len(results) == 1
        metrics = results[0]["metrics"]
        assert "auc_roc" in metrics
        assert "pr_auc" in metrics
        assert "brier_score" in metrics
        assert metrics["auc_roc"] > 0.5  # Better than random

    def test_with_bec_zones(self, synthetic_data):
        X, y, years, bec_zones = synthetic_data
        bt = HistoricalBacktester()

        results = bt.temporal_cv(
            X,
            y,
            years,
            bec_zones=bec_zones,
            train_start=2020,
            test_years=[2022],
            n_rounds=20,
        )

        assert "per_zone" in results[0]

    def test_insufficient_data_skipped(self):
        """Year with <10 samples should be skipped."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((25, 5)).astype(np.float32)
        y = np.concatenate([np.ones(5), np.zeros(20)]).astype(np.float32)
        years = np.array([2020] * 20 + [2021] * 5)  # Only 5 test samples

        bt = HistoricalBacktester()
        results = bt.temporal_cv(
            X,
            y,
            years,
            train_start=2020,
            test_years=[2021],
            n_rounds=10,
        )

        assert len(results) == 0  # Skipped due to insufficient data


class TestPerZoneBreakdown:
    def test_produces_zone_results(self, synthetic_data):
        X, y, years, bec_zones = synthetic_data
        bt = HistoricalBacktester()

        # Use raw predictions (simulated)
        rng = np.random.default_rng(42)
        preds = rng.random(len(y)).astype(np.float32)

        breakdown = bt.per_zone_breakdown(y, preds, bec_zones, min_samples=10)

        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0
        for zone, metrics in breakdown.items():
            assert "auc_roc" in metrics
            assert "n_samples" in metrics
            assert "n_fires" in metrics

    def test_empty_zone_skipped(self):
        """Zone with < min_samples excluded."""
        y = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] * 10, dtype=np.float32)
        preds = np.random.default_rng(42).random(len(y)).astype(np.float32)
        zones = np.array(["BIG"] * (len(y) - 3) + ["TINY"] * 3)

        bt = HistoricalBacktester()
        breakdown = bt.per_zone_breakdown(y, preds, zones, min_samples=50)

        assert "TINY" not in breakdown
        assert "BIG" in breakdown


class TestCompareModels:
    def test_produces_comparison(self, synthetic_data):
        X, y, years, bec_zones = synthetic_data
        bt = HistoricalBacktester()

        rng = np.random.default_rng(42)
        preds_a = rng.random(len(y)).astype(np.float32)
        preds_b = rng.random(len(y)).astype(np.float32)

        comparison = bt.compare_models(
            y,
            preds_a,
            preds_b,
            label_a="5km",
            label_b="1km",
        )

        assert "5km" in comparison
        assert "1km" in comparison
        assert "deltas" in comparison
        assert "auc_roc" in comparison["deltas"]

    def test_with_zones(self, synthetic_data):
        X, y, years, bec_zones = synthetic_data
        bt = HistoricalBacktester()

        rng = np.random.default_rng(42)
        preds_a = rng.random(len(y)).astype(np.float32)
        preds_b = rng.random(len(y)).astype(np.float32)

        comparison = bt.compare_models(
            y,
            preds_a,
            preds_b,
            bec_zones=bec_zones,
        )

        assert "per_zone" in comparison


class TestReportGeneration:
    def test_report_structure(self, synthetic_data):
        X, y, years, bec_zones = synthetic_data
        bt = HistoricalBacktester()

        results = bt.temporal_cv(
            X,
            y,
            years,
            bec_zones=bec_zones,
            train_start=2020,
            test_years=[2021, 2022],
            n_rounds=20,
        )

        report = bt.generate_report(results, model_label="test_model", grid_resolution_km=5.0)

        assert "metadata" in report
        assert report["metadata"]["model_label"] == "test_model"
        assert report["metadata"]["grid_resolution_km"] == 5.0
        assert "overall" in report
        assert "per_year" in report
        assert "per_zone" in report
        assert len(report["per_year"]) == 2

    def test_empty_results(self):
        bt = HistoricalBacktester()
        report = bt.generate_report([], model_label="empty")

        assert report["overall"] == {}
        assert report["per_year"] == []
        assert report["per_zone"] == {}


class TestEvaluatorExtensions:
    """Test the new evaluate_per_zone and compare_models on ModelEvaluator."""

    def test_evaluate_per_zone(self):
        rng = np.random.default_rng(42)
        n = 300
        y = np.concatenate([np.ones(30), np.zeros(270)]).astype(np.float32)
        preds = rng.random(n).astype(np.float32)
        zones = np.array(["CWH"] * 150 + ["BWBS"] * 150)

        evaluator = ModelEvaluator()
        results = evaluator.evaluate_per_zone(y, preds, zones, min_samples=10)

        assert isinstance(results, dict)
        for zone, metrics in results.items():
            assert "auc_roc" in metrics
            assert "n_samples" in metrics

    def test_compare_models_evaluator(self):
        rng = np.random.default_rng(42)
        n = 200
        y = np.concatenate([np.ones(20), np.zeros(180)]).astype(np.float32)
        preds_a = rng.random(n).astype(np.float32)
        preds_b = rng.random(n).astype(np.float32)

        evaluator = ModelEvaluator()
        comparison = evaluator.compare_models(y, preds_a, preds_b, label_a="A", label_b="B")

        assert "A" in comparison
        assert "B" in comparison
        assert "deltas" in comparison
        assert "summary" in comparison
        assert "auc_roc_improved" in comparison["summary"]
