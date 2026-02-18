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

        # Step 1: Download and process HRDPS (days 1-2)
        hrdps_weather = self._get_hrdps_weather(target_date, grid_lats, grid_lons)

        # Step 2: Download and process GDPS (days 3-10)
        gdps_weather = self._get_gdps_weather(target_date, grid_lats, grid_lons)

        # Step 3: Roll forward FWI and predict for each day
        forecasts: dict[str, list[dict]] = {cid: [] for cid in cell_ids}

        # Start with current FWI state
        fwi_state = {}
        default_fwi = {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0}
        for cid in cell_ids:
            fwi_state[cid] = current_fwi_state.get(cid, dict(default_fwi))

        for lead_day in range(1, self.max_days + 1):
            valid_date = target_date + timedelta(days=lead_day)

            # Get weather for this day
            if lead_day <= 2 and lead_day in hrdps_weather:
                weather = hrdps_weather[lead_day]
                source = "HRDPS"
            elif lead_day in gdps_weather:
                weather = gdps_weather[lead_day]
                source = "GDPS"
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

    def _get_hrdps_weather(
        self, target_date: date, grid_lats: np.ndarray, grid_lons: np.ndarray
    ) -> dict[int, dict[str, np.ndarray]]:
        """Download and process HRDPS data."""
        try:
            from infernis.pipelines.hrdps_pipeline import HRDPSPipeline

            hrdps = HRDPSPipeline()

            # Try 12Z run first, then 06Z
            for run_hour in [12, 6, 0]:
                try:
                    files = hrdps.download_run(run_hour=run_hour, target_date=target_date)
                    if files:
                        run_dir = files[0].parent
                        return hrdps.process_for_grid(run_dir, grid_lats, grid_lons)
                except Exception as e:
                    logger.warning("HRDPS %02dZ failed: %s", run_hour, e)

        except Exception as e:
            logger.error("HRDPS pipeline failed: %s", e)

        logger.warning("No HRDPS data available — using synthetic weather for days 1-2")
        return self._synthetic_weather(grid_lats, days=[1, 2])

    def _get_gdps_weather(
        self, target_date: date, grid_lats: np.ndarray, grid_lons: np.ndarray
    ) -> dict[int, dict[str, np.ndarray]]:
        """Download and process GDPS data."""
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
                    logger.warning("GDPS %02dZ failed: %s", run_hour, e)

        except Exception as e:
            logger.error("GDPS pipeline failed: %s", e)

        logger.warning(
            "No GDPS data available — using synthetic weather for days 3-%d", self.max_days
        )
        return self._synthetic_weather(grid_lats, days=list(range(3, self.max_days + 1)))

    def _synthetic_weather(
        self, grid_lats: np.ndarray, days: list[int]
    ) -> dict[int, dict[str, np.ndarray]]:
        """Generate synthetic weather fallback."""
        n = len(grid_lats)
        result = {}
        for day in days:
            result[day] = {
                "temperature_c": np.full(n, 22.0),
                "rh_pct": np.full(n, 45.0),
                "wind_kmh": np.full(n, 12.0),
                "wind_dir_deg": np.full(n, 225.0),
                "precip_24h_mm": np.zeros(n),
            }
        return result

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
                weather.get("soil_moisture_1", np.full(n_cells, 0.3)),
                weather.get("soil_moisture_2", np.full(n_cells, 0.3)),
                weather.get("soil_moisture_3", np.full(n_cells, 0.3)),
                weather.get("soil_moisture_4", np.full(n_cells, 0.3)),
                weather.get("evapotrans_mm", np.full(n_cells, 2.0)),
                # Vegetation defaults for forecast (3)
                np.full(n_cells, 0.5),  # ndvi
                np.zeros(n_cells),  # snow_cover
                np.full(n_cells, 2.0),  # lai
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

    def _predict(self, features: np.ndarray) -> np.ndarray:
        """Run XGBoost inference."""
        if self._model is not None:
            import xgboost as xgb

            from infernis.pipelines.data_processor import FEATURE_NAMES

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
