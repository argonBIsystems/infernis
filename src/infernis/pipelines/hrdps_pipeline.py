"""HRDPS (High Resolution Deterministic Prediction System) pipeline.

Downloads and processes 2.5km GRIB2 forecast data from MSC Datamart
for 48-hour weather predictions over BC.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np

from infernis.config import settings

logger = logging.getLogger(__name__)

# HRDPS variable mappings: GRIB shortName → our variable name
HRDPS_VARIABLES = {
    "TMP": "temperature_k",
    "RH": "rh_pct",
    "WIND": "wind_ms",
    "WDIR": "wind_dir_deg",
    "APCP": "precip_mm",
}

# MSC Datamart base URL for HRDPS continental domain
BASE_URL = "https://dd.weather.gc.ca/model_hrdps/continental/2.5km"


class HRDPSPipeline:
    """Downloads and processes HRDPS GRIB2 forecasts for BC grid cells."""

    FORECAST_HOURS = list(range(1, 49))  # 1h to 48h

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or settings.hrdps_data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_run(self, run_hour: int = 12, target_date: date | None = None) -> list[Path]:
        """Download HRDPS GRIB2 files for a model run.

        Args:
            run_hour: Model run hour UTC (0, 6, 12, 18). Default 12Z.
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
                filename = (
                    f"{date_str}T{run_hour:02d}Z_MSC_HRDPS_{var}_RLatLon0.0225_PT{fh:03d}H.grib2"
                )
                filepath = run_dir / filename

                if filepath.exists() and filepath.stat().st_size > 0:
                    downloaded.append(filepath)
                    continue

                url = f"{BASE_URL}/{run_hour:02d}/{fh:03d}/{filename}"
                try:
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    filepath.write_bytes(resp.content)
                    downloaded.append(filepath)
                except Exception as e:
                    logger.warning("Failed to download %s: %s", filename, e)

        logger.info(
            "HRDPS download: %d files for %s %02dZ",
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
    ) -> dict[int, dict[str, np.ndarray]]:
        """Process HRDPS GRIB2 files and interpolate to grid.

        Groups forecast hours into 24h periods:
        - Hours 1-24 → day+1
        - Hours 25-48 → day+2

        For each day, aggregates to daily values:
        - Temperature: max
        - RH: min
        - Wind: max speed
        - Precipitation: sum
        - Wind direction: mode (from max wind hour)

        Returns:
            dict mapping lead_day (1 or 2) to weather feature dict.
        """
        import glob

        import xarray as xr

        grib_files = sorted(glob.glob(str(grib_dir / "*.grib2")))
        if not grib_files:
            logger.warning("No GRIB2 files found in %s", grib_dir)
            return {}

        # Collect hourly values per variable
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
        result = {}
        for lead_day, hour_range in [(1, range(1, 25)), (2, range(25, 49))]:
            day_hours = {h: hourly_data[h] for h in hour_range if h in hourly_data}
            if not day_hours:
                continue
            result[lead_day] = self._aggregate_daily(day_hours, len(grid_lats))

        logger.info("HRDPS processed: %d forecast days", len(result))
        return result

    def _aggregate_daily(
        self, hourly: dict[int, dict[str, np.ndarray]], n_cells: int
    ) -> dict[str, np.ndarray]:
        """Aggregate hourly forecast values to daily FWI-convention values."""
        temps = []
        rhs = []
        winds = []
        wdirs = []
        precips = []

        for h, data in sorted(hourly.items()):
            if "temperature_k" in data:
                temps.append(data["temperature_k"])
            if "rh_pct" in data:
                rhs.append(data["rh_pct"])
            if "wind_ms" in data:
                winds.append(data["wind_ms"])
            if "wind_dir_deg" in data:
                wdirs.append(data["wind_dir_deg"])
            if "precip_mm" in data:
                precips.append(data["precip_mm"])

        result: dict[str, np.ndarray] = {}

        # Temperature: max (K → C)
        if temps:
            result["temperature_c"] = np.max(temps, axis=0) - 273.15
        else:
            result["temperature_c"] = np.full(n_cells, 20.0)

        # RH: min
        if rhs:
            result["rh_pct"] = np.min(rhs, axis=0)
        else:
            result["rh_pct"] = np.full(n_cells, 50.0)

        # Wind: max speed (m/s → km/h)
        if winds:
            max_wind = np.max(winds, axis=0)
            result["wind_kmh"] = max_wind * 3.6
        else:
            result["wind_kmh"] = np.full(n_cells, 10.0)

        # Wind direction: from the hour with max wind
        if wdirs and winds:
            max_idx = np.argmax(winds, axis=0)
            wdir_stack = np.stack(wdirs)
            result["wind_dir_deg"] = wdir_stack[max_idx, np.arange(n_cells)]
        else:
            result["wind_dir_deg"] = np.full(n_cells, 225.0)

        # Precip: sum (already in mm)
        if precips:
            result["precip_24h_mm"] = np.maximum(np.sum(precips, axis=0), 0.0)
        else:
            result["precip_24h_mm"] = np.zeros(n_cells)

        return result

    def _interpolate_to_grid(
        self, data_array, grid_lats: np.ndarray, grid_lons: np.ndarray
    ) -> np.ndarray:
        """Nearest-neighbor interpolation from HRDPS grid to BC grid."""
        import xarray as xr

        # HRDPS uses latitude/longitude dimensions
        lat_dim = None
        lon_dim = None
        for dim in data_array.dims:
            if "lat" in dim.lower() or dim == "y":
                lat_dim = dim
            elif "lon" in dim.lower() or dim == "x":
                lon_dim = dim

        if lat_dim is None or lon_dim is None:
            # Fallback: use first two spatial dims
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
        # Try from dataset attributes
        if "step" in ds.coords:
            step = ds.coords["step"].values
            if hasattr(step, "astype"):
                hours = step.astype("timedelta64[h]").astype(int)
                return int(hours)

        # Try from filename: PT{NNN}H pattern
        import re

        match = re.search(r"PT(\d{3})H", str(filepath))
        if match:
            return int(match.group(1))

        return None
