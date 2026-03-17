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


class TestDiurnalIntegration:
    """Verify that diurnal FFMC adjustment is correctly integrated into the pipeline."""

    def _make_fwi_results(self, n: int) -> dict:
        """Return a minimal fwi_results dict mimicking _compute_fwi output."""
        from infernis.services.fwi_service import FWIService

        svc = FWIService()
        ffmc, dmc, dc, isi, bui, fwi = svc.compute_daily_vec(
            temp=np.full(n, 28.0),
            rh=np.full(n, 30.0),
            wind=np.full(n, 15.0),
            precip=np.zeros(n),
            month=7,
            prev_ffmc=np.full(n, svc.DEFAULT_FFMC),
            prev_dmc=np.full(n, svc.DEFAULT_DMC),
            prev_dc=np.full(n, svc.DEFAULT_DC),
        )
        return {"ffmc": ffmc, "dmc": dmc, "dc": dc, "isi": isi, "bui": bui, "fwi": fwi}

    def test_diurnal_ffmc_higher_than_daily_at_14h(self, pipeline):
        """After adjustment at hour=14, FFMC should exceed the daily value (dry conditions)."""
        n = 3
        fwi_results = self._make_fwi_results(n)
        daily_ffmc = fwi_results["ffmc"].copy()

        weather = {
            "temperature_c": np.full(n, 30.0),
            "rh_pct": np.full(n, 20.0),
            "wind_kmh": np.full(n, 15.0),
        }
        adjusted = pipeline._apply_diurnal_adjustment(fwi_results, weather, pipeline_hour=14)

        assert np.all(adjusted["ffmc"] > daily_ffmc), (
            "Diurnal-adjusted FFMC should exceed daily FFMC at 14:00 under dry conditions"
        )

    def test_diurnal_isi_fwi_recomputed(self, pipeline):
        """ISI and FWI should change after diurnal adjustment; DMC, DC, BUI unchanged."""
        n = 3
        fwi_results = self._make_fwi_results(n)

        weather = {
            "temperature_c": np.full(n, 30.0),
            "rh_pct": np.full(n, 20.0),
            "wind_kmh": np.full(n, 15.0),
        }
        adjusted = pipeline._apply_diurnal_adjustment(fwi_results, weather, pipeline_hour=14)

        # ISI and FWI should be higher (more FFMC → more ISI → more FWI)
        assert np.all(adjusted["isi"] > fwi_results["isi"]), "ISI should increase with FFMC"
        assert np.all(adjusted["fwi"] > fwi_results["fwi"]), "FWI should increase with ISI"

        # Daily-scale components are unchanged
        np.testing.assert_array_equal(adjusted["dmc"], fwi_results["dmc"])
        np.testing.assert_array_equal(adjusted["dc"], fwi_results["dc"])
        np.testing.assert_array_equal(adjusted["bui"], fwi_results["bui"])

    def test_prev_fwi_state_not_diurnally_adjusted(self, pipeline, sample_grid_df):
        """The FWI carry-forward state must store unadjusted daily FFMC."""
        n = len(sample_grid_df)
        with (
            patch.object(pipeline, "_fetch_weather", return_value=_synthetic_weather(n)),
            patch.object(pipeline, "_fetch_satellite", return_value=_synthetic_satellite(n)),
            patch.object(pipeline, "_fetch_lightning", return_value=_synthetic_lightning(n)),
        ):
            result = pipeline.run(target_date=date(2025, 7, 15), grid_df=sample_grid_df)

        # Build what the daily (unadjusted) FFMC would be
        from infernis.services.fwi_service import FWIService

        svc = FWIService()
        weather = _synthetic_weather(n)
        ffmc_daily, *_ = svc.compute_daily_vec(
            temp=weather["temperature_c"],
            rh=weather["rh_pct"],
            wind=weather["wind_kmh"],
            precip=weather["precip_24h_mm"],
            month=7,
            prev_ffmc=np.full(n, svc.DEFAULT_FFMC),
            prev_dmc=np.full(n, svc.DEFAULT_DMC),
            prev_dc=np.full(n, svc.DEFAULT_DC),
        )

        for i, cell_id in enumerate(sample_grid_df["cell_id"]):
            stored_ffmc = pipeline._prev_fwi_state[cell_id]["ffmc"]
            np.testing.assert_allclose(
                stored_ffmc, float(ffmc_daily[i]), rtol=1e-5,
                err_msg=f"Cell {cell_id}: carry-forward FFMC should be daily (unadjusted)",
            )

    def test_diurnal_output_in_valid_range(self, pipeline):
        """Diurnal-adjusted FFMC, ISI, and FWI should all be non-negative."""
        n = 5
        fwi_results = self._make_fwi_results(n)
        weather = {
            "temperature_c": np.full(n, 35.0),
            "rh_pct": np.full(n, 10.0),
            "wind_kmh": np.full(n, 20.0),
        }
        adjusted = pipeline._apply_diurnal_adjustment(fwi_results, weather, pipeline_hour=14)

        assert np.all(adjusted["ffmc"] >= 0.0) and np.all(adjusted["ffmc"] <= 101.0)
        assert np.all(adjusted["isi"] >= 0.0)
        assert np.all(adjusted["fwi"] >= 0.0)

    def test_run_includes_diurnal_adjustment(self, pipeline, sample_grid_df):
        """End-to-end: predictions should reflect diurnally adjusted FFMC at 14:00."""
        n = len(sample_grid_df)
        weather = _synthetic_weather(n)
        with (
            patch.object(pipeline, "_fetch_weather", return_value=weather),
            patch.object(pipeline, "_fetch_satellite", return_value=_synthetic_satellite(n)),
            patch.object(pipeline, "_fetch_lightning", return_value=_synthetic_lightning(n)),
        ):
            result = pipeline.run(target_date=date(2025, 7, 15), grid_df=sample_grid_df)

        # The reported FFMC should be the diurnally adjusted value (higher than raw daily
        # under these hot/dry conditions: temp=22, rh=45 → moderate adjustment)
        from infernis.services.fwi_service import FWIService

        svc = FWIService()
        ffmc_daily, *_ = svc.compute_daily_vec(
            temp=weather["temperature_c"],
            rh=weather["rh_pct"],
            wind=weather["wind_kmh"],
            precip=weather["precip_24h_mm"],
            month=7,
            prev_ffmc=np.full(n, svc.DEFAULT_FFMC),
            prev_dmc=np.full(n, svc.DEFAULT_DMC),
            prev_dc=np.full(n, svc.DEFAULT_DC),
        )

        for i, cell_id in enumerate(sample_grid_df["cell_id"]):
            reported_ffmc = result[cell_id]["ffmc"]
            daily_ffmc = float(ffmc_daily[i])
            # Under temp=22, rh=45 conditions at 14:00 the adjustment is positive
            assert reported_ffmc > daily_ffmc, (
                f"Cell {cell_id}: reported FFMC {reported_ffmc:.1f} should exceed "
                f"daily {daily_ffmc:.1f} after 14:00 diurnal adjustment"
            )


class TestPredict:
    def test_dummy_scores_in_range(self, pipeline):
        """Dummy predictions should be clipped to [0, 1]."""
        features = np.random.rand(10, 28)
        scores = pipeline._predict(features)
        assert scores.shape == (10,)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


class TestConfidenceIntervals:
    """Tests for quantile-based confidence interval integration in DailyPipeline."""

    def test_predict_quantiles_returns_none_without_models(self, pipeline):
        """_predict_quantiles() returns (None, None) when quantile models are not loaded."""
        features = np.random.rand(5, 28).astype(np.float32)
        lb, ub = pipeline._predict_quantiles(features)
        assert lb is None
        assert ub is None

    def test_confidence_interval_none_when_no_quantile_models(
        self, pipeline, sample_grid_df
    ):
        """Each prediction cell should have confidence_interval=None when quantile
        models are absent (the default case for a fresh pipeline)."""
        n = len(sample_grid_df)
        with (
            patch.object(pipeline, "_fetch_weather", return_value=_synthetic_weather(n)),
            patch.object(pipeline, "_fetch_satellite", return_value=_synthetic_satellite(n)),
            patch.object(pipeline, "_fetch_lightning", return_value=_synthetic_lightning(n)),
        ):
            result = pipeline.run(target_date=date(2025, 7, 15), grid_df=sample_grid_df)

        for cell_id, pred in result.items():
            assert "confidence_interval" in pred, (
                f"Cell {cell_id}: confidence_interval key missing from prediction"
            )
            assert pred["confidence_interval"] is None, (
                f"Cell {cell_id}: confidence_interval should be None without quantile models"
            )

    def test_predict_quantiles_with_injected_models(self, pipeline, sample_grid_df):
        """When quantile models are injected, _predict_quantiles() returns valid bounds."""
        from infernis.training.quantile_trainer import train_quantile_models

        # Train tiny quantile models for injection
        rng = np.random.default_rng(0)
        X_tiny = rng.standard_normal((200, 28)).astype(np.float32)
        y_tiny = rng.uniform(0.1, 0.9, 200).astype(np.float32)
        lower_model, upper_model = train_quantile_models(X_tiny, y_tiny)

        pipeline._quantile_lower = lower_model
        pipeline._quantile_upper = upper_model

        features = rng.standard_normal((5, 28)).astype(np.float32)
        lb, ub = pipeline._predict_quantiles(features)

        assert lb is not None and ub is not None
        assert lb.shape == (5,)
        assert ub.shape == (5,)
        assert np.all(lb >= 0.0) and np.all(lb <= 1.0)
        assert np.all(ub >= 0.0) and np.all(ub <= 1.0)
        assert np.all(lb <= ub), "Lower bound must be <= upper bound"

        # Clean up
        pipeline._quantile_lower = None
        pipeline._quantile_upper = None

    def test_confidence_interval_populated_when_quantile_models_injected(
        self, pipeline, sample_grid_df
    ):
        """When quantile models are loaded, each prediction cell gets a CI dict."""
        from infernis.training.quantile_trainer import train_quantile_models

        rng = np.random.default_rng(1)
        X_tiny = rng.standard_normal((200, 28)).astype(np.float32)
        y_tiny = rng.uniform(0.1, 0.9, 200).astype(np.float32)
        lower_model, upper_model = train_quantile_models(X_tiny, y_tiny)

        pipeline._quantile_lower = lower_model
        pipeline._quantile_upper = upper_model

        n = len(sample_grid_df)
        with (
            patch.object(pipeline, "_fetch_weather", return_value=_synthetic_weather(n)),
            patch.object(pipeline, "_fetch_satellite", return_value=_synthetic_satellite(n)),
            patch.object(pipeline, "_fetch_lightning", return_value=_synthetic_lightning(n)),
        ):
            result = pipeline.run(target_date=date(2025, 7, 15), grid_df=sample_grid_df)

        for cell_id, pred in result.items():
            ci = pred.get("confidence_interval")
            assert ci is not None, f"Cell {cell_id}: confidence_interval should not be None"
            assert "lower" in ci
            assert "upper" in ci
            assert "level" in ci
            assert ci["level"] == 0.90
            assert 0.0 <= ci["lower"] <= 1.0
            assert 0.0 <= ci["upper"] <= 1.0
            assert ci["lower"] <= ci["upper"], (
                f"Cell {cell_id}: lower ({ci['lower']}) > upper ({ci['upper']})"
            )

        # Clean up
        pipeline._quantile_lower = None
        pipeline._quantile_upper = None


class TestFBPIntegration:
    """Verify FBP fire behaviour is computed and attached in the daily pipeline."""

    _REQUIRED_FB_KEYS = {
        "rate_of_spread_mpm",
        "head_fire_intensity_kwm",
        "fire_type",
        "crown_fraction_burned",
        "flame_length_m",
    }

    def test_fire_behaviour_key_present_in_all_cells(self, pipeline, sample_grid_df):
        """Every prediction cell must have a fire_behaviour key (may be dict or None)."""
        n = len(sample_grid_df)
        with (
            patch.object(pipeline, "_fetch_weather", return_value=_synthetic_weather(n)),
            patch.object(pipeline, "_fetch_satellite", return_value=_synthetic_satellite(n)),
            patch.object(pipeline, "_fetch_lightning", return_value=_synthetic_lightning(n)),
        ):
            result = pipeline.run(target_date=date(2025, 7, 15), grid_df=sample_grid_df)

        for cell_id, pred in result.items():
            assert "fire_behaviour" in pred, (
                f"Cell {cell_id}: 'fire_behaviour' key missing from prediction"
            )

    def test_fire_behaviour_populated_when_fuel_type_available(self, pipeline, sample_grid_df):
        """When fuel_type column is present, fire_behaviour dicts should be populated."""
        n = len(sample_grid_df)
        # sample_grid_df has fuel_type column (C3, C5, D1)
        with (
            patch.object(pipeline, "_fetch_weather", return_value=_synthetic_weather(n)),
            patch.object(pipeline, "_fetch_satellite", return_value=_synthetic_satellite(n)),
            patch.object(pipeline, "_fetch_lightning", return_value=_synthetic_lightning(n)),
        ):
            result = pipeline.run(target_date=date(2025, 7, 15), grid_df=sample_grid_df)

        # All cells should have fire_behaviour (fuel types are all combustible in fixture)
        for cell_id, pred in result.items():
            fb = pred["fire_behaviour"]
            assert fb is not None, (
                f"Cell {cell_id}: fire_behaviour should not be None when fuel_type present"
            )
            assert self._REQUIRED_FB_KEYS.issubset(fb.keys()), (
                f"Cell {cell_id}: fire_behaviour missing keys: "
                f"{self._REQUIRED_FB_KEYS - fb.keys()}"
            )

    def test_fire_behaviour_values_are_non_negative(self, pipeline, sample_grid_df):
        """All numeric FBP fields should be >= 0."""
        n = len(sample_grid_df)
        with (
            patch.object(pipeline, "_fetch_weather", return_value=_synthetic_weather(n)),
            patch.object(pipeline, "_fetch_satellite", return_value=_synthetic_satellite(n)),
            patch.object(pipeline, "_fetch_lightning", return_value=_synthetic_lightning(n)),
        ):
            result = pipeline.run(target_date=date(2025, 7, 15), grid_df=sample_grid_df)

        for cell_id, pred in result.items():
            fb = pred.get("fire_behaviour")
            if fb is None:
                continue
            assert fb["rate_of_spread_mpm"] >= 0.0, f"Cell {cell_id}: ROS < 0"
            assert fb["head_fire_intensity_kwm"] >= 0.0, f"Cell {cell_id}: HFI < 0"
            assert 0.0 <= fb["crown_fraction_burned"] <= 1.0, (
                f"Cell {cell_id}: CFB out of range"
            )
            assert fb["flame_length_m"] >= 0.0, f"Cell {cell_id}: flame_length < 0"

    def test_fire_behaviour_fire_type_valid(self, pipeline, sample_grid_df):
        """fire_type must be one of the three valid string values."""
        valid_types = {"surface", "intermittent_crown", "active_crown"}
        n = len(sample_grid_df)
        with (
            patch.object(pipeline, "_fetch_weather", return_value=_synthetic_weather(n)),
            patch.object(pipeline, "_fetch_satellite", return_value=_synthetic_satellite(n)),
            patch.object(pipeline, "_fetch_lightning", return_value=_synthetic_lightning(n)),
        ):
            result = pipeline.run(target_date=date(2025, 7, 15), grid_df=sample_grid_df)

        for cell_id, pred in result.items():
            fb = pred.get("fire_behaviour")
            if fb is None:
                continue
            assert fb["fire_type"] in valid_types, (
                f"Cell {cell_id}: unexpected fire_type '{fb['fire_type']}'"
            )

    def test_fire_behaviour_none_when_no_fuel_type_column(self, pipeline):
        """fire_behaviour should be None for all cells when fuel_type column is absent."""
        import pandas as pd

        grid_no_fuel = pd.DataFrame(
            {
                "cell_id": ["BC-TEST-000"],
                "lat": [51.0],
                "lon": [-122.0],
                "elevation_m": [500.0],
                "slope_deg": [10.0],
                "aspect_deg": [180.0],
                "hillshade": [128.0],
                "distance_to_road_km": [10.0],
                "bec_zone": ["IDF"],
                # No fuel_type column
            }
        )
        n = 1
        with (
            patch.object(pipeline, "_fetch_weather", return_value=_synthetic_weather(n)),
            patch.object(pipeline, "_fetch_satellite", return_value=_synthetic_satellite(n)),
            patch.object(pipeline, "_fetch_lightning", return_value=_synthetic_lightning(n)),
        ):
            result = pipeline.run(target_date=date(2025, 7, 15), grid_df=grid_no_fuel)

        for cell_id, pred in result.items():
            assert "fire_behaviour" in pred
            assert pred["fire_behaviour"] is None, (
                f"Cell {cell_id}: fire_behaviour should be None without fuel_type column"
            )
