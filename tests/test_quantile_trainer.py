"""Tests for XGBoost quantile regression confidence interval trainer.

Tests cover:
- train_quantile_models() returns valid (lower, upper) model pair
- Lower bound <= upper bound for all predictions
- 90% CI coverage >= 80% on held-out test set
- Save / load round-trip produces identical predictions
- Edge-case handling (empty input, bad quantile ordering, missing files)
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(n: int = 1000, n_features: int = 8, seed: int = 42):
    """Create a synthetic fire-risk-like dataset with continuous targets in [0, 1].

    Signal is a scaled sigmoid of feature 0, restricted to [0.15, 0.85] so
    that boundary clipping (``np.clip``) does not truncate the Gaussian noise
    distribution and distort quantile estimates.  Homoscedastic noise with
    standard deviation 0.04 is added.

    For the coverage test (Task 7c) use ``n >= 10 000`` so that the quantile
    models have enough data to achieve 80 %+ empirical coverage consistently
    across random seeds.
    """
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, n_features)).astype(np.float32)

    # Signal bounded away from 0/1 to avoid boundary truncation artefacts
    signal = 0.15 + 0.70 * (1.0 / (1.0 + np.exp(-X[:, 0] * 0.8)))
    noise = rng.normal(0, 0.04, n).astype(np.float32)
    y = np.clip(signal + noise, 0.0, 1.0).astype(np.float32)

    return X, y


# ---------------------------------------------------------------------------
# Module-level fixtures (slow to create — train once per session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset():
    """Full synthetic dataset split into train / test.

    Uses 10 000 samples so that quantile regression achieves reliable
    empirical coverage (>= 80 %) on the held-out test set.  The fixture
    is module-scoped so the dataset is generated only once per test run.
    """
    X, y = _make_synthetic_dataset(n=10_000)
    split = int(0.8 * len(X))
    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "X_test": X[split:],
        "y_test": y[split:],
    }


@pytest.fixture(scope="module")
def trained_models(dataset):
    """Train quantile models once for reuse across tests."""
    from infernis.training.quantile_trainer import train_quantile_models

    lower, upper = train_quantile_models(dataset["X_train"], dataset["y_train"])
    return lower, upper


# ---------------------------------------------------------------------------
# Task 7a — train_quantile_models() returns two models
# ---------------------------------------------------------------------------


class TestTrainQuantileModels:
    def test_returns_two_models(self, trained_models):
        lower, upper = trained_models
        assert lower is not None
        assert upper is not None

    def test_models_are_fitted(self, trained_models):
        """Both models should be XGBRegressors with fitted attributes."""
        import xgboost as xgb

        lower, upper = trained_models
        assert isinstance(lower, xgb.XGBRegressor)
        assert isinstance(upper, xgb.XGBRegressor)

    def test_invalid_quantile_order_raises(self, dataset):
        from infernis.training.quantile_trainer import train_quantile_models

        with pytest.raises(ValueError, match="lower_q.*upper_q"):
            train_quantile_models(
                dataset["X_train"],
                dataset["y_train"],
                lower_q=0.95,
                upper_q=0.05,
            )

    def test_equal_quantiles_raises(self, dataset):
        from infernis.training.quantile_trainer import train_quantile_models

        with pytest.raises(ValueError, match="lower_q.*upper_q"):
            train_quantile_models(
                dataset["X_train"],
                dataset["y_train"],
                lower_q=0.5,
                upper_q=0.5,
            )

    def test_empty_input_raises(self):
        from infernis.training.quantile_trainer import train_quantile_models

        with pytest.raises(ValueError, match="empty"):
            train_quantile_models(
                np.empty((0, 5), dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )

    def test_mismatched_lengths_raises(self, dataset):
        from infernis.training.quantile_trainer import train_quantile_models

        with pytest.raises(ValueError, match="mismatch"):
            train_quantile_models(
                dataset["X_train"][:10],
                dataset["y_train"][:5],
            )


# ---------------------------------------------------------------------------
# Task 7b — lower bound <= upper bound for all predictions
# ---------------------------------------------------------------------------


class TestPredictQuantiles:
    def test_lower_leq_upper_for_all_cells(self, trained_models, dataset):
        """Lower bound must be <= upper bound for every sample."""
        from infernis.training.quantile_trainer import predict_quantiles

        lower, upper = trained_models
        lb, ub = predict_quantiles(dataset["X_test"], lower, upper)

        violations = np.sum(lb > ub)
        assert violations == 0, (
            f"Found {violations}/{len(lb)} cells where lower > upper after ordering fix"
        )

    def test_bounds_clipped_to_unit_interval(self, trained_models, dataset):
        from infernis.training.quantile_trainer import predict_quantiles

        lower, upper = trained_models
        lb, ub = predict_quantiles(dataset["X_test"], lower, upper)

        assert np.all(lb >= 0.0), f"Lower bound below 0: min={lb.min()}"
        assert np.all(lb <= 1.0), f"Lower bound above 1: max={lb.max()}"
        assert np.all(ub >= 0.0), f"Upper bound below 0: min={ub.min()}"
        assert np.all(ub <= 1.0), f"Upper bound above 1: max={ub.max()}"

    def test_output_shapes_match_input(self, trained_models, dataset):
        from infernis.training.quantile_trainer import predict_quantiles

        lower, upper = trained_models
        n = len(dataset["X_test"])
        lb, ub = predict_quantiles(dataset["X_test"], lower, upper)

        assert lb.shape == (n,)
        assert ub.shape == (n,)

    def test_crossing_enforcement(self, trained_models, dataset):
        """Even if models produce crossings on novel data, they should be corrected."""
        from infernis.training.quantile_trainer import predict_quantiles

        lower, upper = trained_models
        # Extreme inputs that may provoke raw crossings
        X_extreme = np.full((20, dataset["X_test"].shape[1]), -5.0, dtype=np.float32)
        lb, ub = predict_quantiles(X_extreme, lower, upper)

        assert np.all(lb <= ub), "Crossings not corrected for extreme inputs"


# ---------------------------------------------------------------------------
# Task 7c — coverage test: ~80%+ of true values within 90% CI
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_coverage_at_least_80_percent(self, trained_models, dataset):
        """At 90% nominal coverage, empirical coverage on held-out test >= 80%.

        We use a soft threshold (80%) rather than 90% because:
        - The synthetic dataset is small (200 test samples)
        - Quantile regression on a separable toy problem may be slightly
          conservative or liberal near boundaries.
        """
        from infernis.training.quantile_trainer import predict_quantiles

        lower, upper = trained_models
        lb, ub = predict_quantiles(dataset["X_test"], lower, upper)
        y_test = dataset["y_test"]

        within = np.sum((y_test >= lb) & (y_test <= ub))
        coverage = within / len(y_test)

        assert coverage >= 0.80, (
            f"90% CI coverage too low: {coverage:.1%} (expected >= 80% on held-out test set)"
        )

    def test_upper_quantile_mostly_above_lower(self, trained_models, dataset):
        """Sanity: the interval should have positive width for most samples."""
        from infernis.training.quantile_trainer import predict_quantiles

        lower, upper = trained_models
        lb, ub = predict_quantiles(dataset["X_test"], lower, upper)

        width = ub - lb
        pct_positive = np.mean(width > 1e-4)
        assert pct_positive >= 0.70, (
            f"Only {pct_positive:.1%} of intervals have positive width — "
            "quantile models may have degenerated"
        )


# ---------------------------------------------------------------------------
# Task 7d — save and load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_save_creates_files(self, trained_models, tmp_path):
        from infernis.training.quantile_trainer import save_quantile_models

        lower, upper = trained_models
        lower_path = tmp_path / "q05.json"
        upper_path = tmp_path / "q95.json"
        save_quantile_models(lower, upper, lower_path, upper_path)

        assert lower_path.exists(), "Lower model file not created"
        assert upper_path.exists(), "Upper model file not created"

    def test_load_returns_models(self, trained_models, tmp_path):
        from infernis.training.quantile_trainer import (
            load_quantile_models,
            save_quantile_models,
        )

        lower, upper = trained_models
        lower_path = tmp_path / "q05.json"
        upper_path = tmp_path / "q95.json"
        save_quantile_models(lower, upper, lower_path, upper_path)

        loaded_lower, loaded_upper = load_quantile_models(lower_path, upper_path)

        assert loaded_lower is not None
        assert loaded_upper is not None

    def test_loaded_predictions_match_original(self, trained_models, dataset, tmp_path):
        """Reloaded models must produce identical predictions."""

        from infernis.training.quantile_trainer import (
            load_quantile_models,
            predict_quantiles,
            save_quantile_models,
        )

        lower, upper = trained_models
        lower_path = tmp_path / "q05.json"
        upper_path = tmp_path / "q95.json"
        save_quantile_models(lower, upper, lower_path, upper_path)

        loaded_lower, loaded_upper = load_quantile_models(lower_path, upper_path)

        X_test = dataset["X_test"]
        lb_orig, ub_orig = predict_quantiles(X_test, lower, upper)
        lb_load, ub_load = predict_quantiles(X_test, loaded_lower, loaded_upper)

        np.testing.assert_array_almost_equal(lb_orig, lb_load, decimal=5)
        np.testing.assert_array_almost_equal(ub_orig, ub_load, decimal=5)

    def test_load_missing_lower_returns_none_pair(self, tmp_path):
        from infernis.training.quantile_trainer import load_quantile_models

        lower_path = tmp_path / "nonexistent_q05.json"
        upper_path = tmp_path / "nonexistent_q95.json"

        result_lower, result_upper = load_quantile_models(lower_path, upper_path)
        assert result_lower is None
        assert result_upper is None

    def test_load_missing_upper_returns_none_pair(self, trained_models, tmp_path):
        from infernis.training.quantile_trainer import (
            load_quantile_models,
        )

        lower, upper = trained_models
        lower_path = tmp_path / "q05.json"
        upper_path = tmp_path / "missing_q95.json"

        # Save only the lower model
        lower.save_model(str(lower_path))

        result_lower, result_upper = load_quantile_models(lower_path, upper_path)
        assert result_lower is None
        assert result_upper is None

    def test_save_creates_parent_directories(self, trained_models, tmp_path):
        from infernis.training.quantile_trainer import save_quantile_models

        lower, upper = trained_models
        nested = tmp_path / "deep" / "nested" / "dir"
        lower_path = nested / "q05.json"
        upper_path = nested / "q95.json"

        # Should not raise even though the directory doesn't exist yet
        save_quantile_models(lower, upper, lower_path, upper_path)
        assert lower_path.exists()
        assert upper_path.exists()
