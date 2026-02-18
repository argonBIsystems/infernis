"""GDPS (Global Deterministic Prediction System) pipeline.

Downloads and processes 15km GRIB2 forecast data from MSC Datamart
for 10-day weather predictions over BC. Used for days 3-10 of the
multi-day forecast (HRDPS covers days 1-2 at higher resolution).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np

from infernis.config import settings

logger = logging.getLogger(__name__)

# MSC Datamart base URL for GDPS
BASE_URL = "https://dd.weather.gc.ca/model_gdps/15km"


class GDPSPipeline:
    """Downloads and processes GDPS GRIB2 forecasts for BC grid cells."""

    # GDPS provides 3-hourly output to 240h (10 days)
    FORECAST_HOURS = list(range(3, 241, 3))

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or settings.gdps_data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_run(self, run_hour: int = 0, target_date: date | None = None) -> list[Path]:
        """Download GDPS GRIB2 files for a model run.

        Args:
            run_hour: Model run hour UTC (0 or 12). Default 00Z.
            target_date: Date of the model run. Default today.

        Returns:
            List of paths to downloaded GRIB2 files.
        """
        import requests

        target_date = target_date or date.today()
        date_str = target_date.strftime("%Y%m%d")
        run_dir = self.data_dir / f"{date_str}" / f"{run_hour:02d}Z"
        run_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []
        variables = ["TMP_TGL_2", "RH_TGL_2", "WIND_TGL_10", "WDIR_TGL_10", "APCP_SFC_0"]

        for fh in self.FORECAST_HOURS:
            for var in variables:
                filename = f"{date_str}T{run_hour:02d}Z_MSC_GDPS_{var}_LatLon0.15_PT{fh:03d}H.grib2"
                filepath = run_dir / filename

                if filepath.exists() and filepath.stat().st_size > 0:
                    downloaded.append(filepath)
                    continue

                url = f"{BASE_URL}/{run_hour:02d}/{fh:03d}/{filename}"
                try:
                    resp = requests.get(url, timeout=60)
                    resp.raise_for_status()
                    filepath.write_bytes(resp.content)
                    downloaded.append(filepath)
                except Exception as e:
                    logger.warning("Failed to download %s: %s", filename, e)

        logger.info(
            "GDPS download: %d files for %s %02dZ",
            len(downloaded),
            date_str,
            run_hour,
        )
        return downloaded

    def process_for_grid(
        self,
        grib_dir: Path,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        start_day: int = 3,
        max_day: int = 10,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Process GDPS GRIB2 files and interpolate to grid.

        Groups 3-hourly forecast hours into 24h periods:
        - Hours 3-24 → day+1, hours 27-48 → day+2, etc.
        - Only returns days from start_day to max_day.

        Returns:
            dict mapping lead_day to weather feature dict.
        """
        import glob

        import xarray as xr

        grib_files = sorted(glob.glob(str(grib_dir / "*.grib2")))
        if not grib_files:
            logger.warning("No GRIB2 files found in %s", grib_dir)
            return {}

        # Collect hourly values
        from infernis.pipelines.hrdps_pipeline import HRDPS_VARIABLES

        hourly_data: dict[int, dict[str, np.ndarray]] = {}
        for fpath in grib_files:
            try:
                ds = xr.open_dataset(fpath, engine="cfgrib")
                fh = self._extract_forecast_hour(ds, fpath)
                if fh is None:
                    ds.close()
                    continue

                if fh not in hourly_data:
                    hourly_data[fh] = {}

                for grib_var, our_var in HRDPS_VARIABLES.items():
                    if grib_var.lower() in [v.lower() for v in ds.data_vars]:
                        actual_var = [v for v in ds.data_vars if v.lower() == grib_var.lower()][0]
                        values = self._interpolate_to_grid(ds[actual_var], grid_lats, grid_lons)
                        hourly_data[fh][our_var] = values

                ds.close()
            except Exception as e:
                logger.warning("Failed to process %s: %s", fpath, e)

        if not hourly_data:
            return {}

        # Aggregate into daily periods
        from infernis.pipelines.hrdps_pipeline import HRDPSPipeline

        aggregator = HRDPSPipeline()
        result = {}
        for lead_day in range(start_day, max_day + 1):
            hour_start = (lead_day - 1) * 24 + 1
            hour_end = lead_day * 24
            day_hours = {
                h: hourly_data[h] for h in range(hour_start, hour_end + 1) if h in hourly_data
            }
            if day_hours:
                result[lead_day] = aggregator._aggregate_daily(day_hours, len(grid_lats))

        logger.info(
            "GDPS processed: %d forecast days (days %d-%d)", len(result), start_day, max_day
        )
        return result

    def _interpolate_to_grid(
        self, data_array, grid_lats: np.ndarray, grid_lons: np.ndarray
    ) -> np.ndarray:
        """Nearest-neighbor interpolation from GDPS grid to BC grid."""
        import xarray as xr

        lat_dim = None
        lon_dim = None
        for dim in data_array.dims:
            if "lat" in dim.lower() or dim == "y":
                lat_dim = dim
            elif "lon" in dim.lower() or dim == "x":
                lon_dim = dim

        if lat_dim is None or lon_dim is None:
            dims = [d for d in data_array.dims if d not in ("time", "step")]
            if len(dims) >= 2:
                lat_dim, lon_dim = dims[0], dims[1]
            else:
                return np.full(len(grid_lats), np.nan)

        result = data_array.interp(
            {
                lat_dim: xr.DataArray(grid_lats, dims="cell"),
                lon_dim: xr.DataArray(grid_lons, dims="cell"),
            },
            method="nearest",
        )
        return result.values

    @staticmethod
    def _extract_forecast_hour(ds, filepath: str) -> int | None:
        """Extract forecast hour from dataset or filename."""
        if "step" in ds.coords:
            step = ds.coords["step"].values
            if hasattr(step, "astype"):
                hours = step.astype("timedelta64[h]").astype(int)
                return int(hours)

        import re

        match = re.search(r"PT(\d{3})H", str(filepath))
        if match:
            return int(match.group(1))
        return None
