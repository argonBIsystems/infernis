"""Tests for the daily pipeline orchestrator."""

from datetime import date
from unittest.mock import patch

import numpy as np
import pytest

from infernis.pipelines.daily_pipeline import DailyPipeline


@pytest.fixture
def pipeline():
    return DailyPipeline()


def _synthetic_weather(n):
    """Return synthetic weather dict matching DailyPipeline._fetch_weather fallback."""
    return {
        "temperature_c": np.full(n, 22.0),
        "rh_pct": np.full(n, 45.0),
        "wind_kmh": np.full(n, 12.0),
        "precip_24h_mm": np.zeros(n),
        "soil_moisture_1": np.full(n, 0.25),
        "soil_moisture_2": np.full(n, 0.28),
        "soil_moisture_3": np.full(n, 0.30),
        "soil_moisture_4": np.full(n, 0.32),
        "evapotrans_mm": np.full(n, 3.0),
        "wind_dir_deg": np.full(n, 225.0),
    }


def _synthetic_satellite(n):
    return {"ndvi": np.full(n, 0.5), "snow": np.zeros(n, dtype=bool), "lai": np.full(n, 2.0)}


def _synthetic_lightning(n):
    return {"lightning_24h": np.zeros(n), "lightning_72h": np.zeros(n)}


class TestDailyPipeline:
    def test_empty_grid_returns_empty(self, pipeline, sample_grid_df):
        """Pipeline returns empty dict with no grid cells."""
        result = pipeline.run(target_date=date(2025, 7, 15), grid_df=None)
        assert result == {}

    def test_run_with_dummy_predictions(self, pipeline, sample_grid_df):
        """Pipeline should produce predictions using dummy model and synthetic data."""
        n = len(sample_grid_df)
        with (
            patch.object(pipeline, "_fetch_weather", return_value=_synthetic_weather(n)),
            patch.object(pipeline, "_fetch_satellite", return_value=_synthetic_satellite(n)),
            patch.object(pipeline, "_fetch_lightning", return_value=_synthetic_lightning(n)),
        ):
            result = pipeline.run(target_date=date(2025, 7, 15), grid_df=sample_grid_df)
        assert len(result) == 3

        for cell_id, pred in result.items():
            assert 0.0 <= pred["score"] <= 1.0
            assert pred["level"] in ["VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH", "EXTREME"]
            assert "timestamp" in pred
            assert "ffmc" in pred
            assert "dmc" in pred
            assert "dc" in pred
            assert "temperature_c" in pred
            assert "ndvi" in pred
            assert isinstance(pred["snow_cover"], bool)

    def test_fwi_state_persists(self, pipeline, sample_grid_df):
        """Previous FWI state should carry forward between runs."""
        n = len(sample_grid_df)
        with (
            patch.object(pipeline, "_fetch_weather", return_value=_synthetic_weather(n)),
            patch.object(pipeline, "_fetch_satellite", return_value=_synthetic_satellite(n)),
            patch.object(pipeline, "_fetch_lightning", return_value=_synthetic_lightning(n)),
        ):
            pipeline.run(target_date=date(2025, 7, 15), grid_df=sample_grid_df)
        assert len(pipeline._prev_fwi_state) == 3
        for cell_id, state in pipeline._prev_fwi_state.items():
            assert "ffmc" in state
            assert "dmc" in state
            assert "dc" in state

    def test_assemble_features_shape(self, pipeline, sample_grid_df, sample_weather_arrays):
        """Feature matrix should have correct shape (28 features)."""
        from infernis.services.fwi_service import FWIService

        svc = FWIService()
        # Use vectorized FWI computation (returns arrays, not list of dicts)
        ffmc, dmc, dc, isi, bui, fwi = svc.compute_daily_vec(
            temp=sample_weather_arrays["temperature_c"],
            rh=sample_weather_arrays["rh_pct"],
            wind=sample_weather_arrays["wind_kmh"],
            precip=sample_weather_arrays["precip_24h_mm"],
            month=7,
            prev_ffmc=np.full(3, svc.DEFAULT_FFMC),
            prev_dmc=np.full(3, svc.DEFAULT_DMC),
            prev_dc=np.full(3, svc.DEFAULT_DC),
        )
        fwi_results = {
            "ffmc": ffmc,
            "dmc": dmc,
            "dc": dc,
            "isi": isi,
            "bui": bui,
            "fwi": fwi,
        }

        satellite = {
            "ndvi": np.array([0.5, 0.6, 0.4]),
            "snow": np.array([False, False, True]),
            "lai": np.array([2.0, 3.0, 1.5]),
        }

        features = pipeline._assemble_features(
            sample_weather_arrays,
            fwi_results,
            satellite,
            sample_grid_df,
            date(2025, 7, 15),
        )
        # 6 FWI + 10 weather + 3 veg + 5 topo/infra + 2 temporal + 2 lightning = 28
        assert features.shape == (3, 28)

    def test_cnn_returns_none_without_model(self, pipeline, sample_grid_df, sample_weather_arrays):
        """CNN predict should return None when no model is loaded."""
        fwi_results = {
            "ffmc": np.array([85.0, 85.0, 85.0]),
            "dmc": np.array([6.0, 6.0, 6.0]),
            "dc": np.array([15.0, 15.0, 15.0]),
            "isi": np.array([3.0, 3.0, 3.0]),
            "bui": np.array([8.0, 8.0, 8.0]),
            "fwi": np.array([5.0, 5.0, 5.0]),
        }
        satellite = {
            "ndvi": np.array([0.5, 0.6, 0.4]),
            "snow": np.array([False, False, True]),
            "lai": np.array([2.0, 3.0, 1.5]),
        }
        result = pipeline._predict_cnn(
            sample_weather_arrays,
            fwi_results,
            satellite,
            sample_grid_df,
            date(2025, 7, 15),
        )
        assert result is None

    def test_risk_fuser_xgb_only(self, pipeline, sample_grid_df):
        """Risk fuser with no CNN should return calibrated XGB scores."""
        from infernis.training.risk_fuser import RiskFuser

        pipeline._risk_fuser = RiskFuser()
        xgb_scores = np.array([0.1, 0.5, 0.9])
        result = pipeline._apply_risk_fuser(xgb_scores, None, sample_grid_df)
        assert result.shape == (3,)
        # Default fuser is identity transform (weight=1.0, bias=0.0)
        np.testing.assert_allclose(result, xgb_scores, atol=1e-6)

    def test_risk_fuser_with_cnn(self, pipeline, sample_grid_df):
        """Risk fuser with CNN scores should use full fusion."""
        from infernis.training.risk_fuser import RiskFuser

        pipeline._risk_fuser = RiskFuser()
        xgb_scores = np.array([0.1, 0.5, 0.9])
        cnn_scores = np.array([0.2, 0.4, 0.8])
        result = pipeline._apply_risk_fuser(xgb_scores, cnn_scores, sample_grid_df)
        assert result.shape == (3,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestPredict:
    def test_dummy_scores_in_range(self, pipeline):
        """Dummy predictions should be clipped to [0, 1]."""
        features = np.random.rand(10, 28)
        scores = pipeline._predict(features)
        assert scores.shape == (10,)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)
