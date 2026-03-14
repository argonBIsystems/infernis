"""Multi-day forecast pipeline orchestrator.

Combines HRDPS (days 1-2) and GDPS (days 3-10) weather forecasts
with FWI roll-forward to produce multi-day fire risk predictions.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from infernis.config import settings
from infernis.models.enums import DangerLevel
from infernis.services.fwi_service import FWIService

logger = logging.getLogger(__name__)


class ForecastPipeline:
    """Orchestrates multi-day fire risk forecasting."""

    def __init__(self):
        self.fwi_service = FWIService()
        self._model = None
        self.confidence_decay = settings.forecast_confidence_decay
        self.max_days = settings.forecast_max_days
        # Observed state from today's daily pipeline (carried forward)
        self._observed_ndvi: np.ndarray | None = None
        self._observed_snow: np.ndarray | None = None
        self._observed_lai: np.ndarray | None = None
        self._observed_soil_moisture: dict[str, np.ndarray] | None = None
        self._prefetched_weather: dict[int, dict[str, np.ndarray]] | None = None

    def load_model(self, model_path: str | None = None):
        """Load XGBoost model for forecast inference."""
        path = model_path or settings.model_path
        from pathlib import Path

        if not Path(path).exists():
            logger.warning("Model not found at %s — using dummy predictions", path)
            self._model = None
            return

        import xgboost as xgb

        self._model = xgb.Booster()
        self._model.load_model(path)
        logger.info("Forecast pipeline loaded model from %s", path)

    def run(
        self,
        grid_df: pd.DataFrame,
        current_fwi_state: dict[str, dict],
        target_date: date | None = None,
    ) -> dict[str, list[dict]]:
        """Execute the multi-day forecast pipeline.

        Args:
            grid_df: Grid DataFrame with cell_id, lat, lon, bec_zone, etc.
            current_fwi_state: dict cell_id → {ffmc, dmc, dc} from today's pipeline.
            target_date: Base date for the forecast. Default today.

        Returns:
            dict mapping cell_id to list of forecast day dicts.
        """
        target_date = target_date or date.today()
        logger.info("=== Starting forecast pipeline for %s ===", target_date)

        cell_ids = grid_df["cell_id"].values
        grid_lats = grid_df["lat"].values
        grid_lons = grid_df["lon"].values
        n_cells = len(cell_ids)

        # Step 1: Fetch forecast weather (Open-Meteo primary, GRIB2 fallback)
        all_weather = self._get_forecast_weather(target_date, grid_lats, grid_lons)

        # Step 2: Roll forward FWI and predict for each day
        forecasts: dict[str, list[dict]] = {cid: [] for cid in cell_ids}

        # Start with current FWI state
        fwi_state = {}
        default_fwi = {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0}
        for cid in cell_ids:
            fwi_state[cid] = current_fwi_state.get(cid, dict(default_fwi))

        for lead_day in range(1, self.max_days + 1):
            valid_date = target_date + timedelta(days=lead_day)

            # Get weather for this day
            if lead_day in all_weather:
                weather = all_weather[lead_day]
                source = "GEM" if lead_day <= 2 else "GEM_GLOBAL"
            else:
                logger.warning("No weather data for day+%d, stopping forecast", lead_day)
                break

            # Roll forward FWI codes
            month = valid_date.month
            fwi_results = self._compute_fwi_vectorized(cell_ids, weather, month, fwi_state)

            # Build feature matrix and predict
            features = self._build_features(weather, fwi_results, grid_df, valid_date, n_cells)
            raw_scores = self._predict(features)

            # Apply confidence decay
            confidence = self.confidence_decay**lead_day
            decayed_scores = raw_scores * confidence

            # Build forecast day for each cell
            for i, cid in enumerate(cell_ids):
                score = float(np.clip(decayed_scores[i], 0.0, 1.0))
                level = DangerLevel.from_score(score)
                danger_level_num = list(DangerLevel).index(level) + 1
                forecasts[cid].append(
                    {
                        "valid_date": valid_date.isoformat(),
                        "lead_day": lead_day,
                        "risk_score": round(score, 4),
                        "danger_level": danger_level_num,
                        "danger_label": level.value,
                        "confidence": round(confidence, 4),
                        "data_source": source,
                        "fwi": {
                            "ffmc": round(fwi_results["ffmc"][i], 1),
                            "dmc": round(fwi_results["dmc"][i], 1),
                            "dc": round(fwi_results["dc"][i], 1),
                            "isi": round(fwi_results["isi"][i], 2),
                            "bui": round(fwi_results["bui"][i], 1),
                            "fwi": round(fwi_results["fwi"][i], 2),
                        },
                        "temperature_c": round(
                            float(weather.get("temperature_c", np.zeros(n_cells))[i]), 1
                        ),
                        "rh_pct": round(float(weather.get("rh_pct", np.zeros(n_cells))[i]), 1),
                        "wind_kmh": round(float(weather.get("wind_kmh", np.zeros(n_cells))[i]), 0),
                        "precip_24h_mm": round(
                            float(weather.get("precip_24h_mm", np.zeros(n_cells))[i]), 1
                        ),
                    }
                )

                # Update FWI state for next day's roll-forward
                fwi_state[cid] = {
                    "ffmc": float(fwi_results["ffmc"][i]),
                    "dmc": float(fwi_results["dmc"][i]),
                    "dc": float(fwi_results["dc"][i]),
                }

            logger.info(
                "Forecast day+%d (%s, %s): mean=%.4f, max=%.4f, confidence=%.2f",
                lead_day,
                valid_date,
                source,
                decayed_scores.mean(),
                decayed_scores.max(),
                confidence,
            )

        total_days = max((len(days) for days in forecasts.values()), default=0)
        logger.info(
            "=== Forecast pipeline complete: %d cells x %d days ===",
            n_cells,
            total_days,
        )
        return forecasts

    def _get_forecast_weather(
        self, target_date: date, grid_lats: np.ndarray, grid_lons: np.ndarray
    ) -> dict[int, dict[str, np.ndarray]]:
        """Fetch forecast weather. Uses prefetched data if available, else Open-Meteo, else GRIB2."""
        # Check if daily pipeline already fetched forecast weather (combined fetch)
        if self._prefetched_weather and len(self._prefetched_weather) >= self.max_days:
            logger.info(
                "Forecast weather: reusing prefetched Open-Meteo data (%d days)",
                len(self._prefetched_weather),
            )
            return self._prefetched_weather

        # Primary: Open-Meteo API (lightweight JSON, same GEM model data)
        try:
            from infernis.pipelines.openmeteo_pipeline import OpenMeteoPipeline

            openmeteo = OpenMeteoPipeline(max_days=self.max_days)
            weather = openmeteo.fetch_forecast_weather(grid_lats, grid_lons, self.max_days)
            if weather and len(weather) >= self.max_days:
                logger.info("Forecast weather: Open-Meteo (GEM seamless) — %d days", len(weather))
                return weather
            logger.warning("Open-Meteo returned incomplete data (%d days)", len(weather))
        except Exception as e:
            logger.warning("Open-Meteo failed: %s — trying GRIB2 fallback", e)

        # Fallback: GRIB2 downloads from MSC Datamart
        all_weather: dict[int, dict[str, np.ndarray]] = {}
        all_weather.update(self._get_hrdps_weather_grib2(target_date, grid_lats, grid_lons))
        all_weather.update(self._get_gdps_weather_grib2(target_date, grid_lats, grid_lons))

        if all_weather:
            logger.info("Forecast weather: GRIB2 fallback — %d days", len(all_weather))
            return all_weather

        # No weather data available — abort rather than produce fake forecasts
        logger.error("All forecast weather sources failed — skipping forecast (no synthetic data)")
        return {}

    def _get_hrdps_weather_grib2(
        self, target_date: date, grid_lats: np.ndarray, grid_lons: np.ndarray
    ) -> dict[int, dict[str, np.ndarray]]:
        """Download and process HRDPS GRIB2 data (fallback)."""
        try:
            from infernis.pipelines.hrdps_pipeline import HRDPSPipeline

            hrdps = HRDPSPipeline()
            for run_hour in [12, 6, 0]:
                try:
                    files = hrdps.download_run(run_hour=run_hour, target_date=target_date)
                    if files:
                        run_dir = files[0].parent
                        return hrdps.process_for_grid(run_dir, grid_lats, grid_lons)
                except Exception as e:
                    logger.warning("HRDPS %02dZ GRIB2 failed: %s", run_hour, e)
        except Exception as e:
            logger.warning("HRDPS GRIB2 pipeline failed: %s", e)
        return {}

    def _get_gdps_weather_grib2(
        self, target_date: date, grid_lats: np.ndarray, grid_lons: np.ndarray
    ) -> dict[int, dict[str, np.ndarray]]:
        """Download and process GDPS GRIB2 data (fallback)."""
        try:
            from infernis.pipelines.gdps_pipeline import GDPSPipeline

            gdps = GDPSPipeline()
            for run_hour in [0, 12]:
                try:
                    files = gdps.download_run(run_hour=run_hour, target_date=target_date)
                    if files:
                        run_dir = files[0].parent
                        return gdps.process_for_grid(
                            run_dir,
                            grid_lats,
                            grid_lons,
                            start_day=3,
                            max_day=self.max_days,
                        )
                except Exception as e:
                    logger.warning("GDPS %02dZ GRIB2 failed: %s", run_hour, e)
        except Exception as e:
            logger.warning("GDPS GRIB2 pipeline failed: %s", e)
        return {}

    # Synthetic weather removed — forecasts should never use hardcoded fake data.
    # If both Open-Meteo and GRIB2 fail, the pipeline returns {} and the forecast
    # is skipped for that day rather than producing misleading predictions.

    def _compute_fwi_vectorized(
        self,
        cell_ids: np.ndarray,
        weather: dict[str, np.ndarray],
        month: int,
        fwi_state: dict[str, dict],
    ) -> dict[str, np.ndarray]:
        """Compute FWI for all cells using vectorized method.

        Returns dict with keys: ffmc, dmc, dc, isi, bui, fwi — each a 1D array.
        """
        n = len(cell_ids)

        prev_ffmc = np.array([fwi_state[cid].get("ffmc", 85.0) for cid in cell_ids])
        prev_dmc = np.array([fwi_state[cid].get("dmc", 6.0) for cid in cell_ids])
        prev_dc = np.array([fwi_state[cid].get("dc", 15.0) for cid in cell_ids])

        ffmc, dmc, dc, isi, bui, fwi = self.fwi_service.compute_daily_vec(
            temp=weather.get("temperature_c", np.full(n, 20.0)),
            rh=weather.get("rh_pct", np.full(n, 50.0)),
            wind=weather.get("wind_kmh", np.full(n, 10.0)),
            precip=weather.get("precip_24h_mm", np.zeros(n)),
            month=month,
            prev_ffmc=prev_ffmc,
            prev_dmc=prev_dmc,
            prev_dc=prev_dc,
        )
        return {
            "ffmc": ffmc,
            "dmc": dmc,
            "dc": dc,
            "isi": isi,
            "bui": bui,
            "fwi": fwi,
        }

    def _build_features(
        self,
        weather: dict[str, np.ndarray],
        fwi: dict[str, np.ndarray],
        grid_df: pd.DataFrame,
        valid_date: date,
        n_cells: int,
    ) -> np.ndarray:
        """Build the 28-feature matrix for XGBoost inference.

        For forecast days, satellite and lightning data use defaults
        since we don't have future observations.
        """
        doy = valid_date.timetuple().tm_yday
        doy_sin = np.sin(2 * np.pi * doy / 365)
        doy_cos = np.cos(2 * np.pi * doy / 365)

        elevation = grid_df.get("elevation_m", pd.Series(np.zeros(n_cells))).fillna(0).values
        slope = grid_df.get("slope_deg", pd.Series(np.zeros(n_cells))).fillna(0).values
        aspect = grid_df.get("aspect_deg", pd.Series(np.zeros(n_cells))).fillna(0).values
        hillshade = grid_df.get("hillshade", pd.Series(np.full(n_cells, 128))).fillna(128).values
        road_dist = (
            grid_df.get("distance_to_road_km", pd.Series(np.full(n_cells, 50.0)))
            .fillna(50.0)
            .values
        )

        feature_matrix = np.column_stack(
            [
                # FWI components (6)
                fwi["ffmc"],
                fwi["dmc"],
                fwi["dc"],
                fwi["isi"],
                fwi["bui"],
                fwi["fwi"],
                # Weather (10)
                weather.get("temperature_c", np.full(n_cells, 20.0)),
                weather.get("rh_pct", np.full(n_cells, 50.0)),
                weather.get("wind_kmh", np.full(n_cells, 10.0)),
                weather.get("wind_dir_deg", np.full(n_cells, 225.0)),
                weather.get("precip_24h_mm", np.zeros(n_cells)),
                self._get_soil_moisture(weather, "soil_moisture_1", n_cells),
                self._get_soil_moisture(weather, "soil_moisture_2", n_cells),
                self._get_soil_moisture(weather, "soil_moisture_3", n_cells),
                self._get_soil_moisture(weather, "soil_moisture_4", n_cells),
                weather.get("evapotrans_mm", np.full(n_cells, 2.0)),
                # Vegetation — use today's observed values if available (3)
                self._observed_ndvi if self._observed_ndvi is not None else np.full(n_cells, 0.5),
                self._observed_snow if self._observed_snow is not None else np.zeros(n_cells),
                self._observed_lai if self._observed_lai is not None else np.full(n_cells, 2.0),
                # Topography / Infrastructure (5)
                elevation,
                slope,
                aspect,
                hillshade,
                road_dist,
                # Temporal (2)
                np.full(n_cells, doy_sin),
                np.full(n_cells, doy_cos),
                # Lightning defaults for forecast (2)
                np.zeros(n_cells),  # lightning_24h
                np.zeros(n_cells),  # lightning_72h
            ]
        )

        return feature_matrix

    def _get_soil_moisture(
        self, weather: dict[str, np.ndarray], key: str, n_cells: int
    ) -> np.ndarray:
        """Get soil moisture: prefer forecast weather, fall back to daily ERA5 observed."""
        defaults = {
            "soil_moisture_1": 0.25,
            "soil_moisture_2": 0.28,
            "soil_moisture_3": 0.30,
            "soil_moisture_4": 0.32,
        }
        # Check if forecast weather has real values (not NaN-filled defaults)
        forecast_val = weather.get(key)
        if forecast_val is not None:
            # Check if it's the NaN-fill default (meaning Open-Meteo returned None)
            default = defaults.get(key, 0.3)
            if not np.allclose(forecast_val, default, atol=0.001):
                return forecast_val

        # Fall back to today's ERA5 observed values
        if self._observed_soil_moisture and key in self._observed_soil_moisture:
            val = self._observed_soil_moisture[key]
            if val is not None:
                return val

        return np.full(n_cells, defaults.get(key, 0.3))

    def _predict(self, features: np.ndarray) -> np.ndarray:
        """Run XGBoost inference."""
        if self._model is not None:
            import xgboost as xgb

            from infernis.pipelines.data_processor import FEATURE_NAMES

            model_features = self._model.feature_names
            if model_features and set(model_features) != set(FEATURE_NAMES):
                # Model was trained on a subset of features (e.g. 5km model
                # uses 24 features vs 28 in the full pipeline).  Select only
                # the columns the model expects.
                idx = [FEATURE_NAMES.index(f) for f in model_features]
                features = features[:, idx]
                dmatrix = xgb.DMatrix(features, feature_names=model_features)
            else:
                dmatrix = xgb.DMatrix(features, feature_names=FEATURE_NAMES)
            scores = self._model.predict(dmatrix)
            return np.clip(scores, 0.0, 1.0)

        # Dummy predictions
        temp = features[:, 6]
        fwi_val = features[:, 5]
        temp_norm = np.clip((temp - 10) / 30, 0, 1)
        fwi_norm = np.clip(fwi_val / 40, 0, 1)
        scores = 0.5 * temp_norm + 0.5 * fwi_norm
        return np.clip(scores, 0.0, 1.0)
