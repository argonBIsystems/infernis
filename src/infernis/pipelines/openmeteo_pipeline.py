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

# Pressure-level variables for C-Haines computation
PRESSURE_LEVEL_VARIABLES = [
    "temperature_850hPa",
    "temperature_500hPa",
    "dewpoint_850hPa",
]

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

# Open-Meteo free tier rate limits (weighted):
#   weight = nLocations × (nDays/14) × (nVars/10)
#   Limits: 600/min, 5000/hour, 10000/day
#
# Forecast: 300 coords × (13/14 days) × (10/10 vars) ≈ 279 weight/request
#   → 5000/279 ≈ 17 safe batches per hour
# Pressure: 300 coords × (1/14 days) × (3/10 vars) ≈ 6.4 weight/request
#   → basically unlimited within the hour
#
# Strategy: process 15 batches (with safety margin), then pause until
# the hourly window resets. Total: 282/15 = 19 hours × ~4 min pause ≈ 4.5h.
# Better than retry storms that waste all attempts.
HOURLY_BATCH_BUDGET = 15  # batches before hourly pause
HOURLY_PAUSE_S = 3600.0  # 1 hour pause to reset the hourly budget
BATCH_DELAY_S = 5.0  # 5s between batches within an hourly window

# Retry config for rate limiting (429) — in case we still hit limits
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 60.0  # 60s, 120s, 240s backoff

# After hitting a 429, wait for hourly reset
RATE_LIMIT_COOLDOWN_S = 3600.0


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

        hourly_count = 0  # batches sent in current hourly window

        with httpx.Client(timeout=60.0) as client:
            for batch_idx in range(n_batches):
                start = batch_idx * BATCH_SIZE
                end = min(start + BATCH_SIZE, n_cells)
                batch_lats = grid_lats[start:end]
                batch_lons = grid_lons[start:end]

                # Proactive hourly budget: pause before hitting the limit
                if hourly_count >= HOURLY_BATCH_BUDGET:
                    remaining = n_batches - batch_idx
                    logger.info(
                        "Open-Meteo: hourly budget (%d batches) spent. "
                        "Pausing %.0fs before continuing (%d batches remaining)",
                        HOURLY_BATCH_BUDGET,
                        HOURLY_PAUSE_S,
                        remaining,
                    )
                    time.sleep(HOURLY_PAUSE_S)
                    hourly_count = 0

                success = False
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        batch_data = self._fetch_batch(
                            client, batch_lats, batch_lons, forecast_days
                        )
                        self._fill_result(result, batch_data, start, end, forecast_days, start_day)
                        cells_fetched += end - start
                        success = True
                        hourly_count += 1
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
                            if is_rate_limit:
                                # Hit hourly limit — reset budget and pause
                                logger.info(
                                    "Open-Meteo: rate limit hit, pausing %.0fs",
                                    RATE_LIMIT_COOLDOWN_S,
                                )
                                time.sleep(RATE_LIMIT_COOLDOWN_S)
                                hourly_count = 0
                            break

                # Progress logging every 15 batches (matches hourly budget)
                if (batch_idx + 1) % HOURLY_BATCH_BUDGET == 0 or batch_idx == n_batches - 1:
                    logger.info(
                        "Open-Meteo progress: %d/%d batches (%d cells fetched, %d failed)",
                        batch_idx + 1,
                        n_batches,
                        cells_fetched,
                        cells_failed,
                    )

                # Brief pause between batches within the hourly window
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

    def fetch_pressure_levels(
        self,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Fetch pressure-level temperatures for C-Haines computation.

        Fetches hourly T850, T500, and Td850 for today and returns the
        afternoon representative value (mean of 12–18 UTC hours) per cell.

        Args:
            grid_lats: Array of latitudes for each grid cell.
            grid_lons: Array of longitudes for each grid cell.

        Returns:
            dict with keys 't850', 't500', 'td850' — arrays of shape (n_cells,).
            Values are NaN where data is unavailable.
        """
        import httpx

        n_cells = len(grid_lats)
        t850 = np.full(n_cells, np.nan)
        t500 = np.full(n_cells, np.nan)
        td850 = np.full(n_cells, np.nan)

        n_batches = (n_cells + BATCH_SIZE - 1) // BATCH_SIZE
        cells_fetched = 0
        cells_failed = 0

        logger.info(
            "Open-Meteo: fetching pressure levels for C-Haines (%d cells, %d batches)",
            n_cells,
            n_batches,
        )

        # Pressure levels are very lightweight (1 day, 3 vars, weight ≈ 6.4/batch)
        # so we can process ~780 batches per hour — no hourly budgeting needed.
        # But share the overall budget with forecast, so if the forecast just ran
        # we may still be in the rate-limited window. Use shorter retries and
        # give up faster — C-Haines is optional (pipeline continues without it).

        consecutive_failures = 0

        with httpx.Client(timeout=60.0) as client:
            for batch_idx in range(n_batches):
                start = batch_idx * BATCH_SIZE
                end = min(start + BATCH_SIZE, n_cells)
                batch_lats = grid_lats[start:end]
                batch_lons = grid_lons[start:end]

                success = False
                # Shorter retries for pressure levels (C-Haines is optional)
                pl_max_retries = 2
                pl_retry_delay = 30.0  # 30s, 60s — then give up
                for attempt in range(pl_max_retries + 1):
                    try:
                        batch_data = self._fetch_pressure_level_batch(
                            client, batch_lats, batch_lons
                        )
                        self._fill_pressure_levels(batch_data, t850, t500, td850, start, end)
                        cells_fetched += end - start
                        success = True
                        break
                    except Exception as e:
                        is_rate_limit = "429" in str(e)
                        if is_rate_limit and attempt < pl_max_retries:
                            delay = pl_retry_delay * (2**attempt)
                            logger.warning(
                                "Open-Meteo pressure-level batch %d/%d rate-limited "
                                "(attempt %d/%d), retrying in %.0fs",
                                batch_idx + 1,
                                n_batches,
                                attempt + 1,
                                pl_max_retries + 1,
                                delay,
                            )
                            time.sleep(delay)
                        else:
                            logger.warning(
                                "Open-Meteo pressure-level batch %d/%d failed: %s",
                                batch_idx + 1,
                                n_batches,
                                e,
                            )
                            cells_failed += end - start
                            break

                # If 2+ consecutive batches fail from rate limiting, bail out early.
                # C-Haines is optional — no point blocking the pipeline for 20+ min.
                if not success:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0

                if consecutive_failures >= 2:
                    remaining = n_batches - batch_idx - 1
                    logger.warning(
                        "Open-Meteo pressure levels: 2 consecutive failures, "
                        "skipping remaining %d batches (C-Haines is optional)",
                        remaining,
                    )
                    cells_failed += sum(
                        min((b + 1) * BATCH_SIZE, n_cells) - b * BATCH_SIZE
                        for b in range(batch_idx + 1, n_batches)
                    )
                    break

                if batch_idx < n_batches - 1 and success:
                    time.sleep(BATCH_DELAY_S)

        logger.info(
            "Open-Meteo pressure levels: %d/%d cells fetched, %d failed",
            cells_fetched,
            n_cells,
            cells_failed,
        )

        return {"t850": t850, "t500": t500, "td850": td850}

    def _fetch_pressure_level_batch(
        self,
        client,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> list[dict]:
        """Fetch hourly pressure-level data for a batch of coordinates.

        Fetches today only (forecast_days=1) to minimise response size.
        """
        params = {
            "latitude": ",".join(f"{lat:.4f}" for lat in lats),
            "longitude": ",".join(f"{lon:.4f}" for lon in lons),
            "hourly": ",".join(PRESSURE_LEVEL_VARIABLES),
            "forecast_days": 1,
            "models": "gem_seamless",
            "timezone": "UTC",
        }

        resp = client.get(BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict):
            if "error" in data and data["error"]:
                raise RuntimeError(
                    f"Open-Meteo pressure-level error: {data.get('reason', 'unknown')}"
                )
            return [data]
        return data

    def _fill_pressure_levels(
        self,
        batch_data: list[dict],
        t850: np.ndarray,
        t500: np.ndarray,
        td850: np.ndarray,
        start: int,
        end: int,
    ):
        """Fill pressure-level arrays using the afternoon mean (12–18 UTC)."""
        for i, location_data in enumerate(batch_data):
            cell_idx = start + i
            if cell_idx >= end:
                break

            hourly = location_data.get("hourly")
            if not hourly:
                continue

            times = hourly.get("time", [])
            t850_h = hourly.get("temperature_850hPa", [])
            t500_h = hourly.get("temperature_500hPa", [])
            td850_h = hourly.get("dewpoint_850hPa", [])

            # Select afternoon hours (12–18 UTC) for representative instability
            t850_vals, t500_vals, td850_vals = [], [], []
            for j, ts in enumerate(times):
                # ts format: "2026-03-15T12:00"
                try:
                    hour = int(ts[11:13])
                except (IndexError, ValueError):
                    continue
                if 12 <= hour <= 18:
                    if j < len(t850_h) and t850_h[j] is not None:
                        t850_vals.append(t850_h[j])
                    if j < len(t500_h) and t500_h[j] is not None:
                        t500_vals.append(t500_h[j])
                    if j < len(td850_h) and td850_h[j] is not None:
                        td850_vals.append(td850_h[j])

            if t850_vals:
                t850[cell_idx] = np.mean(t850_vals)
            if t500_vals:
                t500[cell_idx] = np.mean(t500_vals)
            if td850_vals:
                td850[cell_idx] = np.mean(td850_vals)
