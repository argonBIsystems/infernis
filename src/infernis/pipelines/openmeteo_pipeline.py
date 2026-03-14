"""Open-Meteo forecast weather provider.

Fetches forecast weather data from the Open-Meteo API using the GEM model
(HRDPS for days 1-2, GEM Global for days 3+). This replaces GRIB2 downloads
from MSC Datamart which are too heavy for container deployments.

Open-Meteo serves the same underlying GEM/HRDPS/GDPS data as MSC Datamart
but as lightweight JSON instead of raw GRIB2 files.

License: CC BY 4.0 (free for commercial and non-commercial use with attribution).
Rate limit: 10,000 calls/day, 600 calls/min (free tier, no API key required).
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

# Open-Meteo forecast endpoint
BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Daily variables needed for FWI computation + feature matrix
DAILY_VARIABLES = [
    "temperature_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    # Soil moisture as daily means (avoid hourly data which 4x's response size)
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "soil_moisture_28_to_100cm_mean",
    "soil_moisture_100_to_255cm_mean",
]

# Max coordinates per batch request (tested: 350 works, 400 hits URI limit)
BATCH_SIZE = 300

# Pause between batches — 600 req/min limit = 100ms min, use 300ms for safety
BATCH_DELAY_S = 0.3

# Retry config for rate limiting (429)
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 5.0  # 5s, 10s, 20s backoff


class OpenMeteoPipeline:
    """Fetches forecast weather from Open-Meteo for BC grid cells."""

    def __init__(self, max_days: int = 10):
        self.max_days = max_days

    def fetch_forecast_weather(
        self,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        forecast_days: int | None = None,
        include_today: bool = False,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Fetch forecast weather for all grid cells.

        Args:
            grid_lats: Array of latitudes for each grid cell.
            grid_lons: Array of longitudes for each grid cell.
            forecast_days: Number of forecast days (default: self.max_days).
            include_today: If True, also populate result[0] with today's weather
                          (index 0 from Open-Meteo response). Used by the daily
                          pipeline for same-day weather instead of stale ERA5.

        Returns:
            dict mapping day index to weather feature dict.
            Keys are 1-based (1=tomorrow, 2=day after, etc.).
            If include_today=True, key 0 = today's weather.
        """
        import httpx

        forecast_days = forecast_days or self.max_days
        n_cells = len(grid_lats)

        # Pre-allocate result arrays for each day
        result: dict[int, dict[str, np.ndarray]] = {}
        start_day = 0 if include_today else 1
        for day in range(start_day, forecast_days + 1):
            result[day] = {
                "temperature_c": np.full(n_cells, np.nan),
                "rh_pct": np.full(n_cells, np.nan),
                "wind_kmh": np.full(n_cells, np.nan),
                "wind_dir_deg": np.full(n_cells, np.nan),
                "precip_24h_mm": np.full(n_cells, np.nan),
                "evapotrans_mm": np.full(n_cells, np.nan),
                "soil_moisture_1": np.full(n_cells, np.nan),
                "soil_moisture_2": np.full(n_cells, np.nan),
                "soil_moisture_3": np.full(n_cells, np.nan),
                "soil_moisture_4": np.full(n_cells, np.nan),
            }

        # Process in batches
        n_batches = (n_cells + BATCH_SIZE - 1) // BATCH_SIZE
        cells_fetched = 0
        cells_failed = 0

        logger.info(
            "Open-Meteo: fetching %d-day forecast for %d cells in %d batches (batch_size=%d)",
            forecast_days,
            n_cells,
            n_batches,
            BATCH_SIZE,
        )

        with httpx.Client(timeout=60.0) as client:
            for batch_idx in range(n_batches):
                start = batch_idx * BATCH_SIZE
                end = min(start + BATCH_SIZE, n_cells)
                batch_lats = grid_lats[start:end]
                batch_lons = grid_lons[start:end]

                success = False
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        batch_data = self._fetch_batch(
                            client, batch_lats, batch_lons, forecast_days
                        )
                        self._fill_result(result, batch_data, start, end, forecast_days, start_day)
                        cells_fetched += end - start
                        success = True
                        break
                    except Exception as e:
                        is_rate_limit = "429" in str(e)
                        if is_rate_limit and attempt < MAX_RETRIES:
                            delay = RETRY_BASE_DELAY_S * (2**attempt)
                            logger.warning(
                                "Open-Meteo batch %d/%d rate-limited (attempt %d/%d), "
                                "retrying in %.0fs",
                                batch_idx + 1,
                                n_batches,
                                attempt + 1,
                                MAX_RETRIES + 1,
                                delay,
                            )
                            time.sleep(delay)
                        else:
                            logger.warning(
                                "Open-Meteo batch %d/%d failed: %s",
                                batch_idx + 1,
                                n_batches,
                                e,
                            )
                            cells_failed += end - start
                            break

                # Progress logging every 50 batches
                if (batch_idx + 1) % 50 == 0 or batch_idx == n_batches - 1:
                    logger.info(
                        "Open-Meteo progress: %d/%d batches (%d cells fetched, %d failed)",
                        batch_idx + 1,
                        n_batches,
                        cells_fetched,
                        cells_failed,
                    )

                # Rate limit pause between batches
                if batch_idx < n_batches - 1 and success:
                    time.sleep(BATCH_DELAY_S)

        # Fill any NaN cells with reasonable defaults
        nan_total = 0
        for day in range(start_day, forecast_days + 1):
            weather = result[day]
            for key, default in [
                ("temperature_c", 15.0),
                ("rh_pct", 60.0),
                ("wind_kmh", 10.0),
                ("wind_dir_deg", 225.0),
                ("precip_24h_mm", 0.0),
                ("evapotrans_mm", 2.0),
                ("soil_moisture_1", 0.25),
                ("soil_moisture_2", 0.28),
                ("soil_moisture_3", 0.30),
                ("soil_moisture_4", 0.32),
            ]:
                nan_mask = np.isnan(weather[key])
                if nan_mask.any():
                    weather[key][nan_mask] = default
                    if day == 1 and key == "temperature_c":
                        nan_total = int(nan_mask.sum())

        if nan_total > 0:
            logger.warning(
                "Open-Meteo: %d/%d cells (%.1f%%) fell back to NaN defaults",
                nan_total,
                n_cells,
                nan_total / n_cells * 100,
            )

        success_pct = cells_fetched / n_cells * 100 if n_cells > 0 else 0
        logger.info(
            "Open-Meteo: complete — %d/%d cells (%.1f%%), %d failed",
            cells_fetched,
            n_cells,
            success_pct,
            cells_failed,
        )

        return result

    def _fetch_batch(
        self,
        client,
        lats: np.ndarray,
        lons: np.ndarray,
        forecast_days: int,
    ) -> list[dict]:
        """Fetch forecast for a batch of coordinates.

        Returns list of per-location daily forecast dicts.
        """
        params = {
            "latitude": ",".join(f"{lat:.4f}" for lat in lats),
            "longitude": ",".join(f"{lon:.4f}" for lon in lons),
            "daily": ",".join(DAILY_VARIABLES),
            "forecast_days": min(
                forecast_days + 3, 16
            ),  # +3 buffer (day 0 is today, GEM edge days may be None)
            "models": "gem_seamless",
            "timezone": "UTC",
        }

        resp = client.get(BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        # Single location returns a dict, multiple returns a list
        if isinstance(data, dict):
            if "error" in data and data["error"]:
                raise RuntimeError(f"Open-Meteo error: {data.get('reason', 'unknown')}")
            return [data]
        return data

    def _fill_result(
        self,
        result: dict[int, dict[str, np.ndarray]],
        batch_data: list[dict],
        start: int,
        end: int,
        forecast_days: int,
        start_day: int = 1,
    ):
        """Fill result arrays from a batch of Open-Meteo responses."""
        for i, location_data in enumerate(batch_data):
            cell_idx = start + i
            if cell_idx >= end:
                break

            daily = location_data.get("daily")
            if not daily:
                continue

            # Open-Meteo returns forecast_days+1 values (today + N days)
            # We want lead_day 1 = tomorrow, so skip index 0 (today)
            temps = daily.get("temperature_2m_max", [])
            rhs = daily.get("relative_humidity_2m_min", [])
            winds = daily.get("wind_speed_10m_max", [])
            wdirs = daily.get("wind_direction_10m_dominant", [])
            precips = daily.get("precipitation_sum", [])
            ets = daily.get("et0_fao_evapotranspiration", [])

            # Soil moisture from daily means
            sm_arrays = [
                daily.get("soil_moisture_0_to_7cm_mean", []),
                daily.get("soil_moisture_7_to_28cm_mean", []),
                daily.get("soil_moisture_28_to_100cm_mean", []),
                daily.get("soil_moisture_100_to_255cm_mean", []),
            ]
            sm_keys = ["soil_moisture_1", "soil_moisture_2", "soil_moisture_3", "soil_moisture_4"]

            for day in range(start_day, forecast_days + 1):
                # Index into the daily arrays: day 0 = index 0 (today), day 1 = index 1, etc.
                idx = day
                if idx < len(temps) and temps[idx] is not None:
                    result[day]["temperature_c"][cell_idx] = temps[idx]
                if idx < len(rhs) and rhs[idx] is not None:
                    result[day]["rh_pct"][cell_idx] = rhs[idx]
                if idx < len(winds) and winds[idx] is not None:
                    result[day]["wind_kmh"][cell_idx] = winds[idx]
                if idx < len(wdirs) and wdirs[idx] is not None:
                    result[day]["wind_dir_deg"][cell_idx] = wdirs[idx]
                if idx < len(precips) and precips[idx] is not None:
                    result[day]["precip_24h_mm"][cell_idx] = precips[idx]
                if idx < len(ets) and ets[idx] is not None:
                    result[day]["evapotrans_mm"][cell_idx] = ets[idx]
                for sm_arr, sm_key in zip(sm_arrays, sm_keys):
                    if idx < len(sm_arr) and sm_arr[idx] is not None:
                        result[day][sm_key][cell_idx] = sm_arr[idx]
