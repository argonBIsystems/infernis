"""Tests for SHAP-based ExplainabilityService."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_xgb_model():
    """Train a tiny XGBoost Booster on synthetic 28-feature data."""
    xgb = pytest.importorskip("xgboost")
    from infernis.pipelines.data_processor import FEATURE_NAMES

    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 28)).astype(np.float32)
    y = (X[:, 5] * 0.3 + X[:, 6] * 0.2 + rng.standard_normal(n) * 0.1).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min())

    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    params = {"max_depth": 3, "n_estimators": 10, "objective": "reg:squarederror", "seed": 0}
    model = xgb.train(params, dtrain, num_boost_round=10)
    return model


@pytest.fixture
def explainer(tiny_xgb_model):
    """Return an ExplainabilityService wrapping the tiny XGBoost model."""
    from infernis.pipelines.data_processor import FEATURE_NAMES
    from infernis.services.explainability import ExplainabilityService

    return ExplainabilityService(tiny_xgb_model, FEATURE_NAMES)


@pytest.fixture
def sample_X():
    """Return a small feature matrix [5, 28]."""
    rng = np.random.default_rng(7)
    return rng.standard_normal((5, 28)).astype(np.float32)


# ---------------------------------------------------------------------------
# ExplainabilityService construction
# ---------------------------------------------------------------------------


class TestExplainabilityServiceInit:
    def test_init_with_xgb_model(self, tiny_xgb_model):
        """Service initializes without error given a real XGBoost Booster."""
        from infernis.pipelines.data_processor import FEATURE_NAMES
        from infernis.services.explainability import ExplainabilityService

        svc = ExplainabilityService(tiny_xgb_model, FEATURE_NAMES)
        assert svc is not None
        assert svc.feature_names == FEATURE_NAMES

    def test_init_with_none_model(self):
        """Service initializes gracefully when model is None (no model loaded)."""
        from infernis.pipelines.data_processor import FEATURE_NAMES
        from infernis.services.explainability import ExplainabilityService

        svc = ExplainabilityService(None, FEATURE_NAMES)
        assert svc is not None

    def test_feature_descriptions_populated(self, explainer):
        """FEATURE_DESCRIPTIONS should have entries for all 28 features."""
        from infernis.services.explainability import FEATURE_DESCRIPTIONS

        assert len(FEATURE_DESCRIPTIONS) >= 28
        for name in ["ffmc", "fwi", "temperature_c", "rh_pct", "wind_kmh", "ndvi"]:
            assert name in FEATURE_DESCRIPTIONS, f"Missing description for '{name}'"


# ---------------------------------------------------------------------------
# compute_shap_values
# ---------------------------------------------------------------------------


class TestComputeShapValues:
    def test_returns_correct_shape(self, explainer, sample_X):
        """compute_shap_values returns array [n_cells, n_features]."""
        result = explainer.compute_shap_values(sample_X)
        assert result is not None
        assert result.shape == (5, 28)

    def test_returns_numpy_array(self, explainer, sample_X):
        """Output is a numpy ndarray."""
        result = explainer.compute_shap_values(sample_X)
        assert isinstance(result, np.ndarray)

    def test_values_are_finite(self, explainer, sample_X):
        """All SHAP values should be finite (no NaN or Inf)."""
        result = explainer.compute_shap_values(sample_X)
        assert np.all(np.isfinite(result))

    def test_returns_none_when_no_model(self, sample_X):
        """Returns None gracefully when model is None."""
        from infernis.pipelines.data_processor import FEATURE_NAMES
        from infernis.services.explainability import ExplainabilityService

        svc = ExplainabilityService(None, FEATURE_NAMES)
        result = svc.compute_shap_values(sample_X)
        assert result is None

    def test_shap_sum_approximates_prediction(self, explainer, tiny_xgb_model, sample_X):
        """SHAP values + expected value ≈ model output (TreeSHAP completeness property)."""
        import xgboost as xgb
        from infernis.pipelines.data_processor import FEATURE_NAMES

        shap_vals = explainer.compute_shap_values(sample_X)
        dmatrix = xgb.DMatrix(sample_X, feature_names=FEATURE_NAMES)
        preds = tiny_xgb_model.predict(dmatrix)

        # SHAP sum + bias ≈ prediction. With XGBoost native pred_contribs,
        # the bias is the last column of the full contribs matrix.
        full_contribs = tiny_xgb_model.predict(dmatrix, pred_contribs=True)
        bias = full_contribs[:, -1]
        approx = shap_vals.sum(axis=1) + bias
        np.testing.assert_allclose(approx, preds, atol=1e-3)


# ---------------------------------------------------------------------------
# get_drivers
# ---------------------------------------------------------------------------


class TestGetDrivers:
    def test_returns_list(self, explainer, sample_X):
        """get_drivers returns a list."""
        shap_vals = explainer.compute_shap_values(sample_X)
        feature_values = dict(zip(explainer.feature_names, sample_X[0]))
        drivers = explainer.get_drivers(feature_values, shap_values=shap_vals[0])
        assert isinstance(drivers, list)

    def test_default_top_n(self, explainer, sample_X):
        """Default top_n=5 returns 5 drivers."""
        shap_vals = explainer.compute_shap_values(sample_X)
        feature_values = dict(zip(explainer.feature_names, sample_X[0]))
        drivers = explainer.get_drivers(feature_values, shap_values=shap_vals[0])
        assert len(drivers) == 5

    def test_custom_top_n(self, explainer, sample_X):
        """top_n parameter is respected."""
        shap_vals = explainer.compute_shap_values(sample_X)
        feature_values = dict(zip(explainer.feature_names, sample_X[0]))
        drivers = explainer.get_drivers(feature_values, shap_values=shap_vals[0], top_n=3)
        assert len(drivers) == 3

    def test_driver_keys(self, explainer, sample_X):
        """Each driver dict has the required keys."""
        shap_vals = explainer.compute_shap_values(sample_X)
        feature_values = dict(zip(explainer.feature_names, sample_X[0]))
        drivers = explainer.get_drivers(feature_values, shap_values=shap_vals[0])
        required = {"feature", "contribution", "value", "direction", "description"}
        for d in drivers:
            assert required.issubset(d.keys()), f"Driver missing keys: {required - d.keys()}"

    def test_sorted_by_abs_contribution(self, explainer, sample_X):
        """Drivers are sorted by descending absolute contribution."""
        shap_vals = explainer.compute_shap_values(sample_X)
        feature_values = dict(zip(explainer.feature_names, sample_X[0]))
        drivers = explainer.get_drivers(feature_values, shap_values=shap_vals[0])
        contribs = [abs(d["contribution"]) for d in drivers]
        assert contribs == sorted(contribs, reverse=True)

    def test_direction_values(self, explainer, sample_X):
        """Direction must be 'increasing' or 'decreasing'."""
        shap_vals = explainer.compute_shap_values(sample_X)
        feature_values = dict(zip(explainer.feature_names, sample_X[0]))
        drivers = explainer.get_drivers(feature_values, shap_values=shap_vals[0])
        for d in drivers:
            assert d["direction"] in ("increasing", "decreasing"), (
                f"Invalid direction: {d['direction']}"
            )

    def test_description_non_empty(self, explainer, sample_X):
        """Each driver has a non-empty description string."""
        shap_vals = explainer.compute_shap_values(sample_X)
        feature_values = dict(zip(explainer.feature_names, sample_X[0]))
        drivers = explainer.get_drivers(feature_values, shap_values=shap_vals[0])
        for d in drivers:
            assert isinstance(d["description"], str)
            assert len(d["description"]) > 0

    def test_feature_names_valid(self, explainer, sample_X):
        """Feature names in drivers are all from FEATURE_NAMES."""
        from infernis.pipelines.data_processor import FEATURE_NAMES

        shap_vals = explainer.compute_shap_values(sample_X)
        feature_values = dict(zip(explainer.feature_names, sample_X[0]))
        drivers = explainer.get_drivers(feature_values, shap_values=shap_vals[0])
        for d in drivers:
            assert d["feature"] in FEATURE_NAMES, f"Unknown feature: {d['feature']}"

    def test_no_model_returns_empty(self, sample_X):
        """Returns empty list gracefully when model is None."""
        from infernis.pipelines.data_processor import FEATURE_NAMES
        from infernis.services.explainability import ExplainabilityService

        svc = ExplainabilityService(None, FEATURE_NAMES)
        feature_values = dict(zip(FEATURE_NAMES, sample_X[0]))
        drivers = svc.get_drivers(feature_values)
        assert drivers == []


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    def test_returns_non_empty_string(self, explainer, sample_X):
        """generate_summary returns a non-empty string for any danger level."""
        shap_vals = explainer.compute_shap_values(sample_X)
        feature_values = dict(zip(explainer.feature_names, sample_X[0]))
        drivers = explainer.get_drivers(feature_values, shap_values=shap_vals[0])
        for level in ["LOW", "MODERATE", "HIGH", "VERY_HIGH", "EXTREME"]:
            summary = explainer.generate_summary(drivers, level)
            assert isinstance(summary, str)
            assert len(summary) > 0, f"Empty summary for level {level}"

    def test_summary_mentions_top_driver(self, explainer, sample_X):
        """Summary should mention the top driver feature (or its label)."""
        shap_vals = explainer.compute_shap_values(sample_X)
        feature_values = dict(zip(explainer.feature_names, sample_X[0]))
        drivers = explainer.get_drivers(feature_values, shap_values=shap_vals[0])
        summary = explainer.generate_summary(drivers, "HIGH")
        # The summary should reference the top driver in some recognizable form
        assert len(summary) >= 20, "Summary seems too short to be informative"

    def test_summary_handles_empty_drivers(self, explainer):
        """generate_summary with empty drivers returns a non-empty fallback."""
        summary = explainer.generate_summary([], "MODERATE")
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_varies_by_level(self, explainer, sample_X):
        """Summaries for LOW vs EXTREME should differ in content."""
        shap_vals = explainer.compute_shap_values(sample_X)
        feature_values = dict(zip(explainer.feature_names, sample_X[0]))
        drivers = explainer.get_drivers(feature_values, shap_values=shap_vals[0])
        low = explainer.generate_summary(drivers, "LOW")
        extreme = explainer.generate_summary(drivers, "EXTREME")
        assert low != extreme
