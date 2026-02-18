"""ERA5 weather data pipeline - fetches and processes ERA5 reanalysis for BC grid."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from infernis.config import settings

logger = logging.getLogger(__name__)

# ERA5 variables to fetch
ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "potential_evaporation",
]

# Variable short names in the NetCDF files
ERA5_SHORT_NAMES = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_precipitation": "tp",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",
    "volumetric_soil_water_layer_3": "swvl3",
    "volumetric_soil_water_layer_4": "swvl4",
    "potential_evaporation": "pev",
}


class ERA5Pipeline:
    """Fetches ERA5 reanalysis data and extracts weather features for BC grid cells."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path("data/raw/era5")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_day(self, target_date: date) -> Path:
        """Fetch ERA5 data for a single day via cdsapi. Returns path to NetCDF file.

        The CDS API returns a ZIP archive containing separate NetCDF files for
        instantaneous and accumulated variables. This method extracts and merges
        them into a single NetCDF file.
        """
        import cdsapi

        filename = f"era5_bc_{target_date.isoformat()}.nc"
        filepath = self.data_dir / filename

        if filepath.exists() and not self._is_zip(filepath):
            logger.info("ERA5 file already exists: %s", filepath)
            return filepath

        client = cdsapi.Client(url=settings.cds_url, key=settings.cds_key)

        request = {
            "product_type": ["reanalysis"],
            "variable": ERA5_VARIABLES,
            "year": str(target_date.year),
            "month": f"{target_date.month:02d}",
            "day": f"{target_date.day:02d}",
            "time": "20:00",  # 12:00 PT = 20:00 UTC
            "data_format": "netcdf",
            "area": [
                settings.bc_bbox_north,
                settings.bc_bbox_west,
                settings.bc_bbox_south,
                settings.bc_bbox_east,
            ],
        }

        logger.info("Fetching ERA5 data for %s", target_date)
        zip_path = self.data_dir / f"era5_bc_{target_date.isoformat()}.zip"
        client.retrieve("reanalysis-era5-single-levels", request, str(zip_path))

        # CDS API returns ZIP with separate NetCDFs for instant/accum variables
        filepath = self._extract_and_merge(zip_path, filepath)
        logger.info("ERA5 data saved to %s", filepath)
        return filepath

    @staticmethod
    def _is_zip(path: Path) -> bool:
        """Check if a file is actually a ZIP archive (not a valid NetCDF)."""
        try:
            with open(path, "rb") as f:
                return f.read(2) == b"PK"
        except Exception:
            return False

    def _extract_and_merge(self, zip_path: Path, output_path: Path) -> Path:
        """Extract NetCDF files from CDS ZIP and merge into single dataset."""
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            nc_files = [n for n in zf.namelist() if n.endswith(".nc")]

            if len(nc_files) == 1:
                # Single file — just extract it
                with zf.open(nc_files[0]) as src, open(output_path, "wb") as dst:
                    dst.write(src.read())
            else:
                # Multiple files (instant + accumulated) — merge them
                extract_dir = zip_path.parent / f"_tmp_{zip_path.stem}"
                extract_dir.mkdir(exist_ok=True)
                zf.extractall(extract_dir)

                datasets = []
                for nc_file in nc_files:
                    ds = xr.open_dataset(extract_dir / nc_file)
                    datasets.append(ds)

                merged = xr.merge(datasets)
                merged.to_netcdf(output_path)
                merged.close()
                for ds in datasets:
                    ds.close()

                # Clean up temp files
                import shutil

                shutil.rmtree(extract_dir, ignore_errors=True)

        # Remove the zip
        zip_path.unlink(missing_ok=True)
        return output_path

    def process_for_grid(
        self, filepath: Path, grid_lats: np.ndarray, grid_lons: np.ndarray
    ) -> dict:
        """Extract weather values at grid cell locations from an ERA5 NetCDF file.

        Returns dict of feature arrays, each shape [n_cells].
        """
        ds = xr.open_dataset(filepath)

        # Handle potential time dimension (CDS API uses 'valid_time' or 'time')
        for time_dim in ("valid_time", "time"):
            if time_dim in ds.dims:
                ds = ds.isel({time_dim: 0})
                break

        features = {}

        # Temperature: K -> C
        t2m = self._interpolate_to_grid(ds["t2m"], grid_lats, grid_lons)
        features["temperature_c"] = t2m - 273.15

        # Dewpoint -> Relative Humidity
        d2m = self._interpolate_to_grid(ds["d2m"], grid_lats, grid_lons)
        features["rh_pct"] = self._calc_rh(t2m, d2m)

        # Wind speed from u, v components (m/s -> km/h)
        u10 = self._interpolate_to_grid(ds["u10"], grid_lats, grid_lons)
        v10 = self._interpolate_to_grid(ds["v10"], grid_lats, grid_lons)
        wind_ms = np.sqrt(u10**2 + v10**2)
        features["wind_kmh"] = wind_ms * 3.6
        features["wind_dir_deg"] = (np.degrees(np.arctan2(-u10, -v10)) + 360) % 360

        # Precipitation (m -> mm, accumulated)
        tp = self._interpolate_to_grid(ds["tp"], grid_lats, grid_lons)
        features["precip_24h_mm"] = np.maximum(tp * 1000.0, 0.0)

        # Soil moisture (volumetric, m^3/m^3)
        features["soil_moisture_1"] = self._interpolate_to_grid(ds["swvl1"], grid_lats, grid_lons)
        features["soil_moisture_2"] = self._interpolate_to_grid(ds["swvl2"], grid_lats, grid_lons)
        if "swvl3" in ds:
            features["soil_moisture_3"] = self._interpolate_to_grid(
                ds["swvl3"], grid_lats, grid_lons
            )
        if "swvl4" in ds:
            features["soil_moisture_4"] = self._interpolate_to_grid(
                ds["swvl4"], grid_lats, grid_lons
            )

        # Potential evaporation (m -> mm)
        pev = self._interpolate_to_grid(ds["pev"], grid_lats, grid_lons)
        features["evapotrans_mm"] = np.abs(pev) * 1000.0

        ds.close()
        return features

    def _interpolate_to_grid(
        self, data_array: xr.DataArray, grid_lats: np.ndarray, grid_lons: np.ndarray
    ) -> np.ndarray:
        """Nearest-neighbor interpolation from ERA5 grid to BC grid cell locations."""
        result = data_array.interp(
            latitude=xr.DataArray(grid_lats, dims="cell"),
            longitude=xr.DataArray(grid_lons, dims="cell"),
            method="nearest",
        )
        return result.values

    @staticmethod
    def _calc_rh(t2m_k: np.ndarray, d2m_k: np.ndarray) -> np.ndarray:
        """Calculate relative humidity from temperature and dewpoint (both in K)."""
        t_c = t2m_k - 273.15
        td_c = d2m_k - 273.15
        rh = 100.0 * np.exp(17.625 * td_c / (243.04 + td_c)) / np.exp(17.625 * t_c / (243.04 + t_c))
        return np.clip(rh, 0.0, 100.0)
