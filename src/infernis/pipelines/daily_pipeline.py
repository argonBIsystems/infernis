"""Daily prediction pipeline orchestrator.

Runs at 14:00 PT daily:
1. Fetch ERA5 weather data
2. Compute FWI components
3. Fetch GEE satellite data (NDVI, snow, LAI)
4. Assemble feature matrix (28 features)
5. Run XGBoost inference
6. Run CNN heatmap inference (if model available)
7. Fuse scores via Risk Fuser with BEC-zone calibration
8. Write results to cache
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from infernis.config import settings
from infernis.services.c_haines import compute_c_haines
from infernis.services.diurnal_ffmc import adjust_ffmc_diurnal
from infernis.services.fwi_service import FWIService

logger = logging.getLogger(__name__)


class DailyPipeline:
    """Orchestrates the daily fire risk prediction pipeline."""

    # Fire season in BC: April 1 - October 31
    FIRE_SEASON_START_MONTH = 4
    FIRE_SEASON_END_MONTH = 10

    # Standard FWI startup defaults (CFFDRS specification)
    FWI_DEFAULTS = {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0}

    def __init__(self):
        self.fwi_service = FWIService()
        self._model = None
        self._cnn_trainer = None  # HeatmapTrainer (holds CNN model)
        self._cnn_stats = None  # channel normalisation stats
        self._risk_fuser = None
        self._quantile_lower = None  # XGBRegressor for 5th percentile
        self._quantile_upper = None  # XGBRegressor for 95th percentile
        self._prev_fwi_state: dict[str, dict] = {}  # cell_id -> {ffmc, dmc, dc}
        self._pipeline_status = "success"  # tracks partial failures
        self._static_features: dict[str, np.ndarray] | None = None  # cached per-run
        self._forecast_max_days = settings.forecast_max_days  # for combined Open-Meteo fetch
        self._openmeteo_forecast_weather: dict | None = None  # shared with forecast pipeline

    def load_model(self, model_path: str | None = None):
        """Load the XGBoost model, CNN heatmap model, and Risk Fuser."""
        path = model_path or settings.model_path
        model_dir = Path(path).parent if path else Path("models")

        # --- XGBoost ---
        if not Path(path).exists():
            logger.warning("Model file not found at %s - using dummy predictions", path)
            self._model = None
        else:
            import xgboost as xgb

            self._model = xgb.Booster()
            self._model.load_model(path)
            logger.info("Loaded XGBoost model from %s", path)

        # --- CNN heatmap model ---
        cnn_path = model_dir / "heatmap_v1.pt"
        if cnn_path.exists():
            try:
                from infernis.training.heatmap_model import FireUNet, HeatmapTrainer

                # Detect base_filters from state dict
                state = _peek_base_filters(cnn_path)
                model = FireUNet(base_filters=state)

                # Use MPS (Apple Silicon GPU) when available, else CPU
                import torch

                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
                trainer = HeatmapTrainer(model=model, device=device)
                trainer.load(cnn_path)
                self._cnn_trainer = trainer
                logger.info("Loaded CNN heatmap model from %s (base_filters=%d)", cnn_path, state)

                # Load normalisation stats
                stats_path = Path("data/processed/heatmap/channel_stats.json")
                if stats_path.exists():
                    with open(stats_path) as f:
                        self._cnn_stats = json.load(f)
                    logger.info("Loaded CNN channel stats")
                else:
                    logger.warning("CNN channel_stats.json not found — CNN will use raw values")
            except Exception as e:
                logger.warning("CNN model load failed: %s. Running XGBoost-only.", e)
                self._cnn_trainer = None
        else:
            logger.info("No CNN model at %s — XGBoost-only mode", cnn_path)

        # --- Risk Fuser ---
        try:
            from infernis.training.risk_fuser import RiskFuser

            # Try BEC calibration first, then fuser_weights
            bec_path = model_dir / "bec_calibration.json"
            fuser_path = model_dir / "fuser_weights.json"

            if bec_path.exists():
                self._risk_fuser = RiskFuser(weights_path=bec_path)
                logger.info("Loaded BEC calibration from %s", bec_path)
            elif fuser_path.exists():
                self._risk_fuser = RiskFuser(weights_path=fuser_path)
                logger.info("Loaded fuser weights from %s", fuser_path)
            else:
                self._risk_fuser = RiskFuser()
                logger.info("Risk Fuser initialized with default weights")
        except Exception as e:
            logger.warning("Risk Fuser init failed: %s. Using raw scores.", e)
            self._risk_fuser = None

        # --- Quantile models (optional — CI unavailable when files absent) ---
        try:
            from infernis.training.quantile_trainer import load_quantile_models

            self._quantile_lower, self._quantile_upper = load_quantile_models(
                settings.quantile_lower_path,
                settings.quantile_upper_path,
            )
            if self._quantile_lower is not None:
                logger.info(
                    "Loaded quantile models from %s / %s",
                    settings.quantile_lower_path,
                    settings.quantile_upper_path,
                )
        except Exception as e:
            logger.warning("Quantile model load failed: %s — CI will be None", e)
            self._quantile_lower = None
            self._quantile_upper = None

    def run(self, target_date: date | None = None, grid_df=None) -> dict:
        """Execute the full daily pipeline. Returns predictions dict keyed by cell_id."""
        target_date = target_date or date.today()
        self._pipeline_status = "success"
        logger.info("=== Starting daily pipeline for %s ===", target_date)

        if grid_df is None or len(grid_df) == 0:
            logger.error("No grid cells provided")
            return {}

        grid_lats = grid_df["lat"].values
        grid_lons = grid_df["lon"].values
        cell_ids = grid_df["cell_id"].values
        n_cells = len(cell_ids)

        # Initialize FWI state for fire season start
        self._init_fwi_season(cell_ids, target_date)

        # Step 1: Fetch ERA5 weather
        weather = self._fetch_weather(target_date, grid_lats, grid_lons)
        # Store soil moisture for forecast pipeline (Open-Meteo GEM doesn't provide it)
        self._last_weather = weather

        # Step 1b: Fetch pressure-level data for C-Haines (best-effort, non-blocking)
        c_haines_arr = self._fetch_c_haines(grid_lats, grid_lons)

        # Step 2: Compute FWI for each cell
        fwi_results = self._compute_fwi(cell_ids, weather, target_date)

        # Step 2b: Apply diurnal FFMC adjustment for 14:00 PT pipeline run time.
        # _prev_fwi_state already holds the unadjusted daily FFMC (set inside
        # _compute_fwi before this call), so carry-forward state is unaffected.
        # We adjust FFMC → recompute ISI → recompute FWI for today's prediction.
        # DMC, DC, and BUI are daily-scale indices and are NOT recomputed.
        fwi_results = self._apply_diurnal_adjustment(fwi_results, weather, pipeline_hour=14)

        # Step 3: Fetch satellite data (NDVI, snow, LAI)
        satellite = self._fetch_satellite(target_date, grid_lats, grid_lons)
        # Store for forecast pipeline to carry forward (avoids hardcoded defaults)
        self._last_satellite = satellite

        # Step 3b: Fetch lightning data
        lightning = self._fetch_lightning(target_date, grid_lats, grid_lons)

        # Step 4: Assemble 28-feature matrix for XGBoost
        features = self._assemble_features(
            weather,
            fwi_results,
            satellite,
            grid_df,
            target_date,
            lightning=lightning,
        )

        # Step 5: Run XGBoost inference
        xgb_scores = self._predict(features)

        # Step 5b: Compute SHAP values for all cells (optional — graceful on failure)
        shap_matrix = self._compute_shap(features)

        # Step 5c: Run quantile inference for confidence intervals (optional)
        ci_lower, ci_upper = self._predict_quantiles(features)

        # Step 6: Run CNN heatmap inference (if available)
        cnn_scores = self._predict_cnn(
            weather,
            fwi_results,
            satellite,
            grid_df,
            target_date,
        )

        # Step 7: Fuse scores with BEC-zone calibration
        scores = self._apply_risk_fuser(xgb_scores, cnn_scores, grid_df)

        # Step 8: Build predictions dict (vectorized pre-computation)
        now = datetime.now(timezone.utc).isoformat()

        # Vectorized danger level computation (thresholds from DangerLevel.from_score)
        scores_rounded = np.round(scores, 4)
        levels = np.empty(n_cells, dtype=object)
        levels[:] = "EXTREME"
        levels[scores < 0.80] = "VERY_HIGH"
        levels[scores < 0.60] = "HIGH"
        levels[scores < 0.35] = "MODERATE"
        levels[scores < 0.15] = "LOW"
        levels[scores < 0.05] = "VERY_LOW"

        # Pre-round all arrays once (not per-cell)
        temp_r = np.round(weather.get("temperature_c", np.zeros(n_cells)), 1)
        rh_r = np.round(weather.get("rh_pct", np.zeros(n_cells)), 1)
        wind_r = np.round(weather.get("wind_kmh", np.zeros(n_cells)), 1)
        precip_r = np.round(weather.get("precip_24h_mm", np.zeros(n_cells)), 1)
        sm_r = np.round(weather.get("soil_moisture_1", np.zeros(n_cells)), 4)
        ndvi_r = np.round(satellite["ndvi"], 3)
        snow_bool = satellite["snow"].astype(bool)

        ffmc_r = np.round(fwi_results["ffmc"], 1)
        dmc_r = np.round(fwi_results["dmc"], 1)
        dc_r = np.round(fwi_results["dc"], 1)
        isi_r = np.round(fwi_results["isi"], 1)
        bui_r = np.round(fwi_results["bui"], 1)
        fwi_r = np.round(fwi_results["fwi"], 1)

        # C-Haines: round to 2 dp if available, else None per cell
        c_haines_r = None
        if c_haines_arr is not None:
            c_haines_r = np.round(c_haines_arr, 2)

        # Pre-build SHAP feature-name list (same order as feature matrix columns)
        from infernis.pipelines.data_processor import FEATURE_NAMES as _FEATURE_NAMES

        predictions = {}
        for i in range(n_cells):
            # Confidence interval (None when quantile models not loaded)
            if ci_lower is not None and ci_upper is not None:
                confidence_interval = {
                    "lower": round(float(ci_lower[i]), 4),
                    "upper": round(float(ci_upper[i]), 4),
                    "level": 0.90,
                }
            else:
                confidence_interval = None

            # Per-cell SHAP dict {feature_name: contribution}
            if shap_matrix is not None:
                shap_row = shap_matrix[i]
                cell_shap = {
                    _FEATURE_NAMES[j]: round(float(shap_row[j]), 6)
                    for j in range(len(_FEATURE_NAMES))
                }
            else:
                cell_shap = None

            predictions[cell_ids[i]] = {
                "score": float(scores_rounded[i]),
                "level": levels[i],
                "timestamp": now,
                "ffmc": float(ffmc_r[i]),
                "dmc": float(dmc_r[i]),
                "dc": float(dc_r[i]),
                "isi": float(isi_r[i]),
                "bui": float(bui_r[i]),
                "fwi": float(fwi_r[i]),
                "temperature_c": float(temp_r[i]),
                "rh_pct": float(rh_r[i]),
                "wind_kmh": float(wind_r[i]),
                "precip_24h_mm": float(precip_r[i]),
                "soil_moisture": float(sm_r[i]),
                "ndvi": float(ndvi_r[i]),
                "snow_cover": bool(snow_bool[i]),
                "c_haines": float(c_haines_r[i]) if c_haines_r is not None else None,
                "confidence_interval": confidence_interval,
                "shap_values": cell_shap,
                "next_update": "",
                "fire_behaviour": None,
            }

        # Step 9: Compute FBP fire behaviour per cell (only when fuel_type available)
        self._compute_fbp(predictions, grid_df, fwi_results, weather, target_date)

        status_msg = f"Pipeline {self._pipeline_status}: {len(predictions)} cells processed"
        logger.info("=== %s ===", status_msg)
        return predictions

    @property
    def pipeline_status(self) -> str:
        """Returns 'success' or 'partial' based on data source failures."""
        return self._pipeline_status

    def _init_fwi_season(self, cell_ids, target_date: date):
        """Initialize FWI codes at fire season start or for cells without state.

        At fire season startup (April 1), or for any cell without previous
        state, use the CFFDRS standard defaults: FFMC=85.0, DMC=6.0, DC=15.0.
        """
        existing = set(self._prev_fwi_state.keys())
        needed = set(cell_ids) - existing

        if needed:
            defaults = dict(self.FWI_DEFAULTS)
            for cid in needed:
                self._prev_fwi_state[cid] = dict(defaults)
            logger.info(
                "Initialized FWI state for %d cells with defaults (FFMC=%.1f, DMC=%.1f, DC=%.1f)",
                len(needed),
                self.FWI_DEFAULTS["ffmc"],
                self.FWI_DEFAULTS["dmc"],
                self.FWI_DEFAULTS["dc"],
            )

    def _apply_risk_fuser(
        self,
        xgb_scores: np.ndarray,
        cnn_scores: np.ndarray | None,
        grid_df,
    ) -> np.ndarray:
        """Apply Risk Fuser for BEC-zone calibrated scoring.

        When CNN scores are available, uses full fuse() for XGB+CNN ensemble.
        Otherwise, uses fuse_xgb_only() for XGB-only calibration.
        """
        if self._risk_fuser is None:
            if cnn_scores is not None:
                # Simple average without calibration
                return np.clip(0.65 * xgb_scores + 0.35 * cnn_scores, 0.0, 1.0)
            return xgb_scores

        bec_zones = grid_df.get("bec_zone", pd.Series(["IDF"] * len(grid_df))).values
        bec_zones = np.array([str(z) if z else "IDF" for z in bec_zones])

        if cnn_scores is not None:
            return self._risk_fuser.fuse(xgb_scores, cnn_scores, bec_zones)
        return self._risk_fuser.fuse_xgb_only(xgb_scores, bec_zones)

    def _fetch_weather(self, target_date, grid_lats, grid_lons) -> dict:
        """Fetch weather data for the grid.

        Uses a coarse weather grid (~25km, ~1,400 points) and interpolates
        to the 5km prediction grid. This reduces Open-Meteo API calls from
        282 batches to ~5, eliminating rate-limit issues on the free tier.

        Primary: Open-Meteo GEM (same-day, ECCC's operational model).
        Fallback: ERA5 reanalysis (5-day lag, but has soil moisture).
        """
        weather = None

        # Primary: Open-Meteo GEM via coarse weather grid
        try:
            from infernis.pipelines.openmeteo_pipeline import OpenMeteoPipeline
            from infernis.pipelines.weather_grid import (
                generate_weather_grid,
                interpolate_forecast_to_prediction_grid,
            )

            # Generate coarse weather grid (~1,400 points vs 84,535)
            wx_lats, wx_lons = generate_weather_grid()

            om = OpenMeteoPipeline(max_days=self._forecast_max_days)
            all_weather_coarse = om.fetch_forecast_weather(
                wx_lats,
                wx_lons,
                forecast_days=self._forecast_max_days,
                include_today=True,
            )

            if all_weather_coarse and 0 in all_weather_coarse:
                # Interpolate from coarse weather grid to prediction grid
                all_weather = interpolate_forecast_to_prediction_grid(
                    all_weather_coarse, wx_lats, wx_lons, grid_lats, grid_lons
                )
                weather = all_weather[0]
                # Store forecast days for the forecast pipeline to reuse
                self._openmeteo_forecast_weather = {
                    k: v for k, v in all_weather.items() if k >= 1
                }
                logger.info(
                    "Weather: Open-Meteo GEM via coarse grid (%d wx points → %d cells)",
                    len(wx_lats),
                    len(grid_lats),
                )
            else:
                logger.warning("Open-Meteo returned no data for today")
        except Exception as e:
            logger.warning("Open-Meteo failed: %s — falling back to ERA5", e)

        # Fallback: ERA5 (has everything including soil moisture, but 5-day lag)
        if weather is None:
            weather = self._fetch_era5_weather(target_date, grid_lats, grid_lons)

        # Always merge ERA5 soil moisture (Open-Meteo GEM doesn't provide it)
        self._merge_era5_soil_moisture(weather, target_date, grid_lats, grid_lons)

        return weather

    def _fetch_c_haines(
        self,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> np.ndarray | None:
        """Fetch pressure-level data and compute C-Haines for all cells.

        Uses Open-Meteo hourly T850, T500, Td850 for today. Returns an array
        of C-Haines values (range 0-13) per cell, or None on failure so that
        the pipeline remains fully functional without pressure-level data.

        A NaN in the raw pressure data (failed cell) maps to None-equivalent
        in the output — the per-cell None is handled in the predictions loop.
        """
        try:
            from infernis.pipelines.openmeteo_pipeline import OpenMeteoPipeline
            from infernis.pipelines.weather_grid import (
                generate_weather_grid,
                interpolate_to_prediction_grid,
            )

            # Use coarse grid for pressure levels too
            wx_lats, wx_lons = generate_weather_grid()
            om = OpenMeteoPipeline()
            pl_coarse = om.fetch_pressure_levels(wx_lats, wx_lons)

            # Interpolate to prediction grid
            pl = interpolate_to_prediction_grid(
                pl_coarse, wx_lats, wx_lons, grid_lats, grid_lons
            )

            t850 = pl["t850"]
            t500 = pl["t500"]
            td850 = pl["td850"]

            # Cells where any pressure-level input is missing: leave as NaN
            valid = np.isfinite(t850) & np.isfinite(t500) & np.isfinite(td850)
            n = len(grid_lats)
            c_haines_arr = np.full(n, np.nan)
            if valid.any():
                c_haines_arr[valid] = compute_c_haines(t850[valid], t500[valid], td850[valid])
                logger.info(
                    "C-Haines: computed for %d/%d cells (mean=%.2f, max=%.2f)",
                    int(valid.sum()),
                    n,
                    float(np.nanmean(c_haines_arr)),
                    float(np.nanmax(c_haines_arr)),
                )
            else:
                logger.warning("C-Haines: no valid pressure-level data — all cells will be None")
            return c_haines_arr

        except Exception as e:
            logger.warning("C-Haines fetch failed: %s — c_haines will be None for all cells", e)
            return None

    def _fetch_era5_weather(self, target_date, grid_lats, grid_lons) -> dict:
        """Fetch ERA5 reanalysis weather (fallback, 5-day lag)."""
        from datetime import timedelta

        try:
            from infernis.pipelines.era5_pipeline import ERA5Pipeline

            era5 = ERA5Pipeline()

            for days_back in range(0, 8):
                try_date = target_date - timedelta(days=days_back)
                try:
                    filepath = era5.fetch_day(try_date)
                    weather = era5.process_for_grid(filepath, grid_lats, grid_lons)
                    if days_back > 0:
                        logger.warning(
                            "ERA5 fallback: using data from %s (%d days old)",
                            try_date,
                            days_back,
                        )
                    else:
                        logger.info("Weather: ERA5 (same-day)")
                    self._pipeline_status = "partial"
                    return weather
                except Exception as e:
                    logger.debug("ERA5 fetch failed for %s: %s", try_date, e)
                    continue

            logger.error("ERA5 fetch failed for all dates")
        except Exception as e:
            logger.error("ERA5 pipeline init failed: %s", e)

        # Last resort: no weather data at all — use conservative defaults
        logger.error("All weather sources failed — using defaults")
        self._pipeline_status = "partial"
        n = len(grid_lats)
        return {
            "temperature_c": np.full(n, 15.0),
            "rh_pct": np.full(n, 60.0),
            "wind_kmh": np.full(n, 10.0),
            "precip_24h_mm": np.zeros(n),
            "evapotrans_mm": np.full(n, 2.0),
            "wind_dir_deg": np.full(n, 225.0),
        }

    def _merge_era5_soil_moisture(self, weather, target_date, grid_lats, grid_lons):
        """Fetch soil moisture from ERA5 and merge into weather dict.

        ERA5 is the only reliable source for soil moisture at 4 depths.
        The 5-day lag is acceptable because soil moisture changes slowly.
        """
        sm_keys = ["soil_moisture_1", "soil_moisture_2", "soil_moisture_3", "soil_moisture_4"]

        # If weather already has soil moisture (from ERA5 fallback), keep it
        if all(k in weather for k in sm_keys):
            return

        try:
            era5_weather = self._fetch_era5_weather(target_date, grid_lats, grid_lons)
            for key in sm_keys:
                if key in era5_weather:
                    weather[key] = era5_weather[key]
            logger.info("Soil moisture: ERA5 (merged into Open-Meteo weather)")
        except Exception as e:
            logger.warning("ERA5 soil moisture failed: %s — using defaults", e)
            n = len(grid_lats)
            defaults = {
                "soil_moisture_1": 0.25,
                "soil_moisture_2": 0.28,
                "soil_moisture_3": 0.30,
                "soil_moisture_4": 0.32,
            }
            for key, val in defaults.items():
                if key not in weather:
                    weather[key] = np.full(n, val)

    def _compute_fwi(self, cell_ids, weather, target_date) -> dict[str, np.ndarray]:
        """Compute FWI for all cells using vectorized numpy operations.

        Uses compute_daily_vec() for a single vectorized call instead of
        2.1M scalar calls. Returns dict of arrays instead of list of dicts.
        """
        n = len(cell_ids)

        # Extract previous state as aligned arrays (defaults for missing cells)
        prev_ffmc = np.full(n, self.FWI_DEFAULTS["ffmc"])
        prev_dmc = np.full(n, self.FWI_DEFAULTS["dmc"])
        prev_dc = np.full(n, self.FWI_DEFAULTS["dc"])

        for i, cid in enumerate(cell_ids):
            prev = self._prev_fwi_state.get(cid)
            if prev:
                prev_ffmc[i] = prev["ffmc"]
                prev_dmc[i] = prev["dmc"]
                prev_dc[i] = prev["dc"]

        # Single vectorized FWI call (fwi_service.py:201-223)
        ffmc, dmc, dc, isi, bui, fwi = self.fwi_service.compute_daily_vec(
            temp=weather.get("temperature_c", np.zeros(n)),
            rh=weather.get("rh_pct", np.full(n, 50)),
            wind=weather.get("wind_kmh", np.full(n, 10)),
            precip=weather.get("precip_24h_mm", np.zeros(n)),
            month=target_date.month,
            prev_ffmc=prev_ffmc,
            prev_dmc=prev_dmc,
            prev_dc=prev_dc,
        )

        # Update state dict for persistence (dict lookups only, ~2s for 2.1M)
        for i, cid in enumerate(cell_ids):
            self._prev_fwi_state[cid] = {
                "ffmc": float(ffmc[i]),
                "dmc": float(dmc[i]),
                "dc": float(dc[i]),
            }

        return {
            "ffmc": ffmc,
            "dmc": dmc,
            "dc": dc,
            "isi": isi,
            "bui": bui,
            "fwi": fwi,
        }

    def _apply_diurnal_adjustment(
        self,
        fwi_results: dict[str, np.ndarray],
        weather: dict,
        pipeline_hour: int = 14,
    ) -> dict[str, np.ndarray]:
        """Apply diurnal FFMC correction and propagate through ISI → FWI.

        Uses the hour-of-day adjustment from diurnal_ffmc.py (Red Book 2018).
        Only FFMC, ISI, and FWI are updated; DMC, DC, and BUI are daily-scale
        and do not change.

        The FWI carry-forward state (_prev_fwi_state) is intentionally NOT
        touched here — it was written with the unadjusted daily FFMC inside
        _compute_fwi() and must remain that way for next-day continuity.

        Parameters
        ----------
        fwi_results:
            Dict returned by _compute_fwi() with arrays for each FWI component.
        weather:
            Weather dict; temperature_c and rh_pct are used for the adjustment.
        pipeline_hour:
            Local hour at which the pipeline runs (default 14 for 14:00 PT).

        Returns
        -------
        Updated fwi_results dict with adjusted ffmc, isi, and fwi arrays.
        """
        n = len(fwi_results["ffmc"])
        temp = weather.get("temperature_c", np.full(n, 20.0))
        rh = weather.get("rh_pct", np.full(n, 40.0))

        adj_ffmc = adjust_ffmc_diurnal(fwi_results["ffmc"], temp, rh, hour=pipeline_hour)
        adj_isi = self.fwi_service._vec_isi(weather.get("wind_kmh", np.full(n, 10.0)), adj_ffmc)
        adj_fwi = self.fwi_service._vec_fwi(adj_isi, fwi_results["bui"])

        logger.debug(
            "Diurnal FFMC adjustment at hour=%d: mean Δffmc=%.2f, mean Δisi=%.2f, mean Δfwi=%.2f",
            pipeline_hour,
            float(np.mean(adj_ffmc - fwi_results["ffmc"])),
            float(np.mean(adj_isi - fwi_results["isi"])),
            float(np.mean(adj_fwi - fwi_results["fwi"])),
        )

        return {
            **fwi_results,
            "ffmc": adj_ffmc,
            "isi": adj_isi,
            "fwi": adj_fwi,
        }

    def _fetch_satellite(self, target_date, grid_lats, grid_lons) -> dict:
        """Fetch NDVI, snow cover, and LAI from GEE.

        Returns dict with keys: ndvi, snow, lai.
        """
        n = len(grid_lats)
        result = {
            "ndvi": np.full(n, 0.5),
            "snow": np.zeros(n, dtype=bool),
            "lai": np.full(n, 2.0),
        }
        try:
            from infernis.pipelines.gee_pipeline import GEEPipeline

            gee = GEEPipeline()
            result["ndvi"] = gee.fetch_ndvi(grid_lats, grid_lons, target_date)
            result["snow"] = gee.fetch_snow_cover(grid_lats, grid_lons, target_date)
            # LAI if available
            if hasattr(gee, "fetch_lai"):
                result["lai"] = gee.fetch_lai(grid_lats, grid_lons, target_date)
        except Exception as e:
            logger.error("GEE fetch failed: %s. Using defaults.", e)
            self._pipeline_status = "partial"
        return result

    def _fetch_lightning(self, target_date, grid_lats, grid_lons) -> dict:
        """Fetch lightning density data from MSC Datamart."""
        n = len(grid_lats)
        try:
            from infernis.pipelines.lightning_pipeline import LightningPipeline

            lp = LightningPipeline()
            result = lp.fetch_lightning_density(grid_lats, grid_lons, target_date)
            lp.close()
            return result
        except Exception as e:
            logger.error("Lightning fetch failed: %s. Using zeros.", e)
            return {
                "lightning_24h": np.zeros(n),
                "lightning_72h": np.zeros(n),
            }

    def _get_static_features(self, grid_df) -> dict[str, np.ndarray]:
        """Get static features (terrain, road distance). Cached per-run."""
        if self._static_features is not None:
            return self._static_features

        n = len(grid_df)
        self._static_features = {
            "elevation_m": grid_df.get("elevation_m", pd.Series(np.zeros(n))).fillna(0).values,
            "slope_deg": grid_df.get("slope_deg", pd.Series(np.zeros(n))).fillna(0).values,
            "aspect_deg": grid_df.get("aspect_deg", pd.Series(np.zeros(n))).fillna(0).values,
            "hillshade": grid_df.get("hillshade", pd.Series(np.full(n, 128))).fillna(128).values,
            "distance_to_road_km": grid_df.get("distance_to_road_km", pd.Series(np.full(n, 50.0)))
            .fillna(50.0)
            .values,
        }
        return self._static_features

    def _assemble_features(
        self, weather, fwi_results, satellite, grid_df, target_date, lightning=None
    ):
        """Assemble the 28-feature matrix for XGBoost inference.

        Feature order must match FEATURE_NAMES in data_processor.py.
        """
        n = len(grid_df)
        doy = target_date.timetuple().tm_yday
        doy_sin = np.sin(2 * np.pi * doy / 365)
        doy_cos = np.cos(2 * np.pi * doy / 365)

        # Lightning data
        if lightning is None:
            lightning = {}
        lightning_24h = lightning.get("lightning_24h", np.zeros(n))
        lightning_72h = lightning.get("lightning_72h", np.zeros(n))

        static = self._get_static_features(grid_df)

        feature_matrix = np.column_stack(
            [
                # FWI components (6) — arrays from vectorized _compute_fwi()
                fwi_results["ffmc"],
                fwi_results["dmc"],
                fwi_results["dc"],
                fwi_results["isi"],
                fwi_results["bui"],
                fwi_results["fwi"],
                # Weather (10)
                weather.get("temperature_c", np.zeros(n)),
                weather.get("rh_pct", np.full(n, 50)),
                weather.get("wind_kmh", np.full(n, 10)),
                weather.get("wind_dir_deg", np.zeros(n)),
                weather.get("precip_24h_mm", np.zeros(n)),
                weather.get("soil_moisture_1", np.full(n, 0.3)),
                weather.get("soil_moisture_2", np.full(n, 0.3)),
                weather.get("soil_moisture_3", np.full(n, 0.3)),
                weather.get("soil_moisture_4", np.full(n, 0.3)),
                weather.get("evapotrans_mm", np.full(n, 2)),
                # Vegetation (3)
                satellite["ndvi"],
                satellite["snow"].astype(np.float64),
                satellite["lai"],
                # Topography / Infrastructure (5)
                static["elevation_m"],
                static["slope_deg"],
                static["aspect_deg"],
                static["hillshade"],
                static["distance_to_road_km"],
                # Temporal (2)
                np.full(n, doy_sin),
                np.full(n, doy_cos),
                # Lightning (2)
                lightning_24h,
                lightning_72h,
            ]
        )

        return feature_matrix

    def _predict(self, features: np.ndarray) -> np.ndarray:
        """Run XGBoost inference. Returns array of scores [0, 1]."""
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
        else:
            # Dummy predictions based on feature heuristics
            n = features.shape[0]
            temp = features[:, 6]  # temperature_c
            soil = features[:, 11]  # soil_moisture_1
            wind = features[:, 8]  # wind_kmh
            fwi_val = features[:, 5]  # FWI

            temp_norm = np.clip((temp - 10) / 30, 0, 1)
            soil_norm = np.clip(1 - soil / 0.5, 0, 1)
            wind_norm = np.clip(wind / 40, 0, 1)
            fwi_norm = np.clip(fwi_val / 40, 0, 1)

            scores = 0.3 * temp_norm + 0.3 * soil_norm + 0.15 * wind_norm + 0.25 * fwi_norm
            scores += np.random.normal(0, 0.02, n)
            return np.clip(scores, 0.0, 1.0)

    def _compute_shap(self, features: np.ndarray) -> np.ndarray | None:
        """Compute SHAP values for all cells using TreeSHAP (optional, graceful).

        Returns a [n_cells, n_features] array of per-feature SHAP contributions,
        or None if the model is not a real XGBoost Booster, shap is not installed,
        or computation fails for any reason.
        """
        if self._model is None:
            return None

        try:
            from infernis.pipelines.data_processor import FEATURE_NAMES
            from infernis.services.explainability import ExplainabilityService

            svc = ExplainabilityService(self._model, FEATURE_NAMES)
            shap_matrix = svc.compute_shap_values(features)
            if shap_matrix is not None:
                logger.info(
                    "SHAP: computed %d × %d values (mean |shap|=%.4f)",
                    shap_matrix.shape[0],
                    shap_matrix.shape[1],
                    float(np.mean(np.abs(shap_matrix))),
                )
            return shap_matrix
        except Exception as e:
            logger.warning("SHAP computation failed (non-fatal): %s", e)
            return None

    def _predict_quantiles(
        self, features: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Run quantile regression inference for confidence interval bounds.

        Returns ``(lower_bounds, upper_bounds)`` arrays clipped to ``[0, 1]``
        with guaranteed ``lower[i] <= upper[i]``, or ``(None, None)`` when
        quantile models are not loaded.
        """
        if self._quantile_lower is None or self._quantile_upper is None:
            return None, None

        try:
            from infernis.training.quantile_trainer import predict_quantiles

            lower, upper = predict_quantiles(features, self._quantile_lower, self._quantile_upper)
            logger.info(
                "Quantile CI: lower=[%.3f, %.3f] upper=[%.3f, %.3f]",
                float(lower.min()),
                float(lower.max()),
                float(upper.min()),
                float(upper.max()),
            )
            return lower, upper
        except Exception as e:
            logger.warning("Quantile inference failed: %s — CI will be None", e)
            return None, None

    def _compute_fbp(
        self,
        predictions: dict,
        grid_df,
        fwi_results: dict[str, np.ndarray],
        weather: dict,
        target_date,
    ) -> None:
        """Compute FBP fire behaviour for each grid cell and attach to predictions.

        Only runs when fuel_type column is present in grid_df. Per-cell failures
        are skipped silently (fire_behaviour remains None for that cell).

        Parameters
        ----------
        predictions:
            Dict mapping cell_id → prediction dict (mutated in place).
        grid_df:
            Grid DataFrame containing fuel_type, slope_deg, aspect_deg, etc.
        fwi_results:
            Dict of FWI arrays from _compute_fwi() / _apply_diurnal_adjustment().
        weather:
            Weather dict from _fetch_weather().
        target_date:
            Pipeline run date (used for Julian day computation).
        """
        if "fuel_type" not in grid_df.columns:
            logger.debug("FBP skipped: fuel_type column not present in grid_df")
            return

        from infernis.services.fbp_service import compute_fire_behaviour

        cell_ids = grid_df["cell_id"].values
        fuel_types = grid_df["fuel_type"].values
        lats = grid_df["lat"].values
        lons = grid_df["lon"].values
        slopes = grid_df.get("slope_deg", pd.Series(np.zeros(len(grid_df)))).fillna(0).values
        aspects = grid_df.get("aspect_deg", pd.Series(np.zeros(len(grid_df)))).fillna(0).values
        elevations = (
            grid_df.get("elevation_m", pd.Series(np.zeros(len(grid_df)))).fillna(0).values
        )

        bui_arr = fwi_results["bui"]
        ffmc_arr = fwi_results["ffmc"]
        wind_arr = weather.get("wind_kmh", np.full(len(cell_ids), 10.0))
        wind_dir_arr = weather.get("wind_dir_deg", np.full(len(cell_ids), 0.0))

        month = target_date.month
        day = target_date.day

        fbp_ok = 0
        for i, cell_id in enumerate(cell_ids):
            if cell_id not in predictions:
                continue
            fuel = str(fuel_types[i]) if fuel_types[i] else "NF"
            try:
                fb = compute_fire_behaviour(
                    fuel_type=fuel,
                    ffmc=float(ffmc_arr[i]),
                    bui=float(bui_arr[i]),
                    wind_speed=float(wind_arr[i]),
                    wind_direction=float(wind_dir_arr[i]),
                    slope=float(slopes[i]),
                    aspect=float(aspects[i]),
                    latitude=float(lats[i]),
                    longitude=float(lons[i]),
                    elevation=float(elevations[i]),
                    month=month,
                    day=day,
                )
                predictions[cell_id]["fire_behaviour"] = fb
                fbp_ok += 1
            except Exception as exc:
                logger.debug("FBP skipped for cell %s: %s", cell_id, exc)

        logger.info(
            "FBP: computed for %d/%d cells", fbp_ok, len(cell_ids)
        )

    def _predict_cnn(
        self,
        weather: dict,
        fwi_results: dict[str, np.ndarray],
        satellite: dict,
        grid_df,
        target_date: date,
    ) -> np.ndarray | None:
        """Run CNN heatmap inference. Returns per-cell scores or None if unavailable."""
        if self._cnn_trainer is None:
            return None

        try:
            from infernis.training.heatmap_data import (
                INPUT_CHANNELS,
                LAT_MAX,
                LAT_STEP,
                LON_MIN,
                LON_STEP,
                RASTER_H,
                RASTER_W,
            )

            n = len(grid_df)
            grid_lats = grid_df["lat"].values
            grid_lons = grid_df["lon"].values

            # Map grid cells to raster pixels
            rows = np.clip(((LAT_MAX - grid_lats) / LAT_STEP).astype(int), 0, RASTER_H - 1)
            cols = np.clip(((grid_lons - LON_MIN) / LON_STEP).astype(int), 0, RASTER_W - 1)

            # Build [C, H, W] input raster
            raster = np.zeros((INPUT_CHANNELS, RASTER_H, RASTER_W), dtype=np.float32)

            doy = target_date.timetuple().tm_yday
            doy_sin = np.sin(2 * np.pi * doy / 365)

            static = self._get_static_features(grid_df)

            # Channel mapping matches CHANNEL_NAMES in heatmap_model.py
            channel_data = {
                0: weather.get("temperature_c", np.zeros(n)),  # temperature_c
                1: weather.get("rh_pct", np.full(n, 50.0)),  # rh_pct
                2: weather.get("wind_kmh", np.full(n, 10.0)),  # wind_kmh
                3: weather.get("soil_moisture_1", np.full(n, 0.3)),  # soil_moisture_1
                4: fwi_results["fwi"],  # fwi
                5: satellite["ndvi"],  # ndvi
                6: satellite["snow"].astype(np.float32),  # snow_cover
                7: static["elevation_m"],  # elevation_m
                8: static["slope_deg"],  # slope_deg
                # 9: fuel_type_encoded — zeros (not available yet)
                # 10: bec_zone_encoded — zeros (not available yet)
                11: np.full(n, doy_sin, dtype=np.float32),  # doy_sin
            }

            for ch, values in channel_data.items():
                vals = np.asarray(values, dtype=np.float32)
                valid = np.isfinite(vals) & (vals > -9999)
                raster[ch, rows[valid], cols[valid]] = vals[valid]

            # Apply normalisation if stats are available
            if self._cnn_stats is not None:
                mean = np.array(self._cnn_stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
                std = np.array(self._cnn_stats["std"], dtype=np.float32).reshape(-1, 1, 1)
                raster = (raster - mean) / std

            # Run inference
            heatmap = self._cnn_trainer.predict(raster)  # [H, W]

            # Extract per-cell scores from heatmap
            cnn_scores = heatmap[rows, cols]
            logger.info(
                "CNN heatmap: min=%.4f, mean=%.4f, max=%.4f",
                cnn_scores.min(),
                cnn_scores.mean(),
                cnn_scores.max(),
            )
            return np.clip(cnn_scores, 0.0, 1.0).astype(np.float64)

        except Exception as e:
            logger.error("CNN inference failed: %s. Using XGBoost-only.", e)
            return None


def _peek_base_filters(model_path: Path) -> int:
    """Detect base_filters from a saved FireUNet state dict."""
    import torch

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    # enc1.block.0.weight has shape [base_filters, in_channels, 3, 3]
    key = "enc1.block.0.weight"
    if key in state:
        return state[key].shape[0]
    return 64  # default
