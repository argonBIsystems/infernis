"""Data processing ETL pipeline.

Transforms raw downloaded data into grid-aligned, training-ready features.
Reads from data/raw/ and writes processed parquet files to data/processed/.

Workflow:
1. Load the BC grid (5km cells)
2. For each date in the training period:
   a. Extract ERA5 weather at grid cells (NetCDF -> numpy)
   b. Compute FWI codes with daily carry-forward
   c. Sample satellite rasters at grid cells (GeoTIFF -> numpy)
   d. Load static features (elevation, fuel type, BEC zone)
3. Write daily feature matrices to parquet
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from infernis.services.fwi_service import FWIService

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# ERA5 variable short names in NetCDF
ERA5_VARS = {
    "t2m": "temperature_k",
    "d2m": "dewpoint_k",
    "u10": "u_wind",
    "v10": "v_wind",
    "tp": "total_precip_m",
    "swvl1": "soil_moisture_1",
    "swvl2": "soil_moisture_2",
    "swvl3": "soil_moisture_3",
    "swvl4": "soil_moisture_4",
    "pev": "pot_evaporation_m",
}


class DataProcessor:
    """Processes raw data sources into training-ready feature matrices."""

    def __init__(
        self,
        raw_dir: Path | None = None,
        processed_dir: Path | None = None,
    ):
        self.raw_dir = raw_dir or RAW_DIR
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.fwi_service = FWIService()

    def process_era5_month(
        self,
        year: int,
        month: int,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Process one ERA5 monthly file, returning weather features per grid cell.

        Returns dict mapping variable name -> array of shape [n_days, n_cells].
        """
        filepath = self.raw_dir / "era5" / f"era5_bc_{year}_{month:02d}.nc"
        if not filepath.exists():
            logger.warning("ERA5 file missing: %s", filepath)
            return {}

        ds = xr.open_dataset(filepath)

        # Normalize time dimension name to "time"
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})

        # Merge supplementary precip file if available
        precip_path = self.raw_dir / "era5" / f"era5_bc_{year}_{month:02d}_precip.nc"
        if precip_path.exists():
            ds_precip = xr.open_dataset(precip_path)
            if "valid_time" in ds_precip.dims:
                ds_precip = ds_precip.rename({"valid_time": "time"})
            for var in ds_precip.data_vars:
                if var not in ds:
                    ds[var] = ds_precip[var]
            ds_precip.close()

        # Determine available time steps
        if "time" in ds.dims:
            n_times = ds.sizes["time"]
        else:
            # Single time step
            n_times = 1

        n_cells = len(grid_lats)
        features = {}

        # Pre-compute grid query points for scipy interpolation
        from scipy.interpolate import RegularGridInterpolator

        grid_points = np.column_stack([grid_lats, grid_lons])

        for era5_var, feat_name in ERA5_VARS.items():
            if era5_var not in ds:
                logger.warning("Variable %s not found in %s", era5_var, filepath.name)
                continue

            data = ds[era5_var]
            values = np.zeros((n_times, n_cells), dtype=np.float32)

            # Extract ERA5 coordinate arrays (regular grid)
            era5_lats = data.coords["latitude"].values.astype(np.float64)
            era5_lons = data.coords["longitude"].values.astype(np.float64)

            # RegularGridInterpolator requires ascending axes
            lat_ascending = era5_lats[0] < era5_lats[-1]

            for t in range(n_times):
                slice_data = data.isel(time=t) if "time" in data.dims else data
                data_2d = slice_data.values.astype(np.float64)

                # Flip latitude axis if descending
                if not lat_ascending:
                    interp_lats = era5_lats[::-1]
                    data_2d = data_2d[::-1]
                else:
                    interp_lats = era5_lats

                interp = RegularGridInterpolator(
                    (interp_lats, era5_lons),
                    data_2d,
                    method="nearest",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                values[t] = interp(grid_points).astype(np.float32)

            features[feat_name] = values

        # Derive computed features
        if "temperature_k" in features:
            features["temperature_c"] = features["temperature_k"] - 273.15

        if "temperature_k" in features and "dewpoint_k" in features:
            t_c = features["temperature_k"] - 273.15
            td_c = features["dewpoint_k"] - 273.15
            rh = (
                100.0
                * np.exp(17.625 * td_c / (243.04 + td_c))
                / np.exp(17.625 * t_c / (243.04 + t_c))
            )
            features["rh_pct"] = np.clip(rh, 0.0, 100.0)

        if "u_wind" in features and "v_wind" in features:
            u = features["u_wind"]
            v = features["v_wind"]
            features["wind_kmh"] = np.sqrt(u**2 + v**2) * 3.6
            features["wind_dir_deg"] = (np.degrees(np.arctan2(-u, -v)) + 360) % 360

        if "total_precip_m" in features:
            features["precip_24h_mm"] = np.maximum(features["total_precip_m"] * 1000.0, 0.0)
        elif "temperature_k" in features:
            # CDS sometimes omits accumulated variables — default to 0 (dry, conservative for fire risk)
            logger.debug("Precipitation missing, defaulting to 0 mm")
            features["precip_24h_mm"] = np.zeros_like(features["temperature_k"])

        if "pot_evaporation_m" in features:
            features["evapotrans_mm"] = np.abs(features["pot_evaporation_m"]) * 1000.0
        elif "temperature_k" in features:
            logger.debug("Evapotranspiration missing, defaulting to 3 mm")
            features["evapotrans_mm"] = np.full_like(features["temperature_k"], 3.0)

        # Extract time coordinates
        if "time" in ds.dims:
            features["_times"] = pd.DatetimeIndex(ds["time"].values)

        ds.close()
        return features

    def sample_raster_at_grid(
        self,
        tif_path: Path,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        band: int = 1,
    ) -> np.ndarray:
        """Sample a GeoTIFF raster at grid cell locations using nearest-neighbor.

        Returns array of shape [n_cells] with sampled values.
        """
        import rasterio
        from rasterio.transform import rowcol

        if not tif_path.exists():
            logger.warning("Raster file missing: %s", tif_path)
            return np.full(len(grid_lats), np.nan)

        with rasterio.open(tif_path) as src:
            data = src.read(band)
            nodata = src.nodata
            transform = src.transform

            # Vectorized sampling — rowcol supports array inputs
            rows, cols = rowcol(transform, grid_lons, grid_lats)
            rows = np.asarray(rows, dtype=np.intp)
            cols = np.asarray(cols, dtype=np.intp)

            valid = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])

            values = np.full(len(grid_lats), np.nan, dtype=np.float32)
            values[valid] = data[rows[valid], cols[valid]]

            if nodata is not None:
                values[values == nodata] = np.nan

        return values

    def process_satellite_year(
        self,
        year: int,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Sample all GEE satellite rasters for a given year.

        Returns dict mapping feature name -> array [n_cells].
        """
        features = {}

        # MODIS NDVI
        ndvi_path = self.raw_dir / "gee" / "modis_ndvi" / f"modis_ndvi_bc_{year}.tif"
        ndvi = self.sample_raster_at_grid(ndvi_path, grid_lats, grid_lons)
        # MODIS NDVI scale factor: values are stored as raw integers * 10000
        features["ndvi"] = np.where(np.isnan(ndvi), 0.5, ndvi * 0.0001)

        # MODIS Snow Cover
        snow_path = self.raw_dir / "gee" / "modis_snow" / f"modis_snow_bc_{year}.tif"
        snow = self.sample_raster_at_grid(snow_path, grid_lats, grid_lons)
        features["snow_cover"] = np.where(np.isnan(snow), 0, (snow > 50).astype(np.float32))

        # MODIS LAI (Leaf Area Index)
        lai_path = self.raw_dir / "gee" / "modis_lai" / f"modis_lai_bc_{year}.tif"
        lai = self.sample_raster_at_grid(lai_path, grid_lats, grid_lons)
        # MODIS LAI scale factor: stored as int * 10, valid range 0-100
        features["lai"] = np.where(np.isnan(lai), 2.0, lai * 0.1)

        return features

    def process_static_features(
        self,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Process static features: DEM (elevation, slope, aspect, hillshade).

        Returns dict mapping feature name -> array [n_cells].
        """
        features = {}

        # CDEM elevation
        cdem_path = self.raw_dir / "gee" / "cdem" / "cdem_bc.tif"
        elevation = self.sample_raster_at_grid(cdem_path, grid_lats, grid_lons)

        # DEM from separate download (fallback)
        if np.all(np.isnan(elevation)):
            dem_path = self.raw_dir / "dem" / "cdem_bc.tif"
            if dem_path.exists():
                elevation = self.sample_raster_at_grid(dem_path, grid_lats, grid_lons)

        features["elevation_m"] = np.nan_to_num(elevation, nan=500.0)

        # Derive slope, aspect, hillshade from DEM raster if available
        slope, aspect, hillshade = self._derive_terrain_from_dem(cdem_path, grid_lats, grid_lons)
        features["slope_deg"] = slope
        features["aspect_deg"] = aspect
        features["hillshade"] = hillshade

        # Distance to nearest road
        features["distance_to_road_km"] = self._compute_distance_to_road(grid_lats, grid_lons)

        return features

    def _derive_terrain_from_dem(
        self,
        dem_path: Path,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derive slope, aspect, and hillshade from a DEM raster.

        Returns (slope_deg, aspect_deg, hillshade) arrays of shape [n_cells].
        """
        n = len(grid_lats)
        default_slope = np.zeros(n, dtype=np.float32)
        default_aspect = np.zeros(n, dtype=np.float32)
        default_hillshade = np.full(n, 128.0, dtype=np.float32)

        if not dem_path.exists():
            # Also try fallback DEM path
            fallback = self.raw_dir / "dem" / "cdem_bc.tif"
            if not fallback.exists():
                return default_slope, default_aspect, default_hillshade
            dem_path = fallback

        try:
            import rasterio
            from rasterio.transform import rowcol

            with rasterio.open(dem_path) as src:
                dem = src.read(1).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    dem[dem == nodata] = np.nan
                transform = src.transform
                res = abs(transform[0])  # pixel resolution in degrees

                # Compute gradient (slope and aspect) from DEM
                # Convert resolution to approximate meters for slope calculation
                dy, dx = np.gradient(np.nan_to_num(dem, nan=0.0), res * 111000)
                slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
                slope_deg_raster = np.degrees(slope_rad)
                aspect_rad = np.arctan2(-dy, dx)
                aspect_deg_raster = (np.degrees(aspect_rad) + 360) % 360

                # Hillshade (sun azimuth 315 deg, altitude 45 deg)
                sun_az = np.radians(315)
                sun_alt = np.radians(45)
                hs = np.cos(sun_alt) * np.cos(slope_rad) + np.sin(sun_alt) * np.sin(
                    slope_rad
                ) * np.cos(sun_az - aspect_rad)
                hillshade_raster = np.clip(hs * 255, 0, 255).astype(np.float32)

                # Sample at grid points (vectorized)
                rows, cols = rowcol(transform, grid_lons, grid_lats)
                rows = np.asarray(rows, dtype=np.intp)
                cols = np.asarray(cols, dtype=np.intp)

                valid = (rows >= 0) & (rows < dem.shape[0]) & (cols >= 0) & (cols < dem.shape[1])

                slopes = np.zeros(n, dtype=np.float32)
                aspects = np.zeros(n, dtype=np.float32)
                hillshades = np.full(n, 128.0, dtype=np.float32)

                slopes[valid] = slope_deg_raster[rows[valid], cols[valid]]
                aspects[valid] = aspect_deg_raster[rows[valid], cols[valid]]
                hillshades[valid] = hillshade_raster[rows[valid], cols[valid]]

                logger.info(
                    "Terrain derived: slope %.1f-%.1f, aspect %.0f-%.0f, hillshade %.0f-%.0f (%d valid)",
                    slopes[valid].min(),
                    slopes[valid].max(),
                    aspects[valid].min(),
                    aspects[valid].max(),
                    hillshades[valid].min(),
                    hillshades[valid].max(),
                    valid.sum(),
                )
                return slopes, aspects, hillshades

        except Exception as e:
            logger.warning("Terrain derivation failed: %s. Using defaults.", e)
            return default_slope, default_aspect, default_hillshade

    _MAX_ROAD_DISTANCE_KM = 200.0  # cap for cells outside road data coverage

    def _compute_distance_to_road(
        self,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> np.ndarray:
        """Compute distance from each grid cell to the nearest road (km).

        Uses a KDTree on road vertices for fast nearest-neighbor lookup.
        Distances are capped at _MAX_ROAD_DISTANCE_KM since road data may
        not cover all of BC (current dataset covers southern Interior only).
        """
        from scipy.spatial import KDTree

        n = len(grid_lats)
        road_path = self.raw_dir / "bc_roads" / "bc_roads.geojson"
        if not road_path.exists():
            logger.warning("Road data not found: %s — defaulting to 50 km", road_path)
            return np.full(n, 50.0, dtype=np.float32)

        try:
            import geopandas as gpd

            logger.info("Loading road network...")
            roads = gpd.read_file(road_path)
            if roads.crs and roads.crs.to_epsg() != 4326:
                roads = roads.to_crs("EPSG:4326")

            # Extract all vertices from LineString and MultiLineString geometries
            road_points = []
            for geom in roads.geometry:
                if geom is None:
                    continue
                if geom.geom_type == "MultiLineString":
                    for part in geom.geoms:
                        for lon, lat, *_ in part.coords:
                            road_points.append((lat, lon))
                elif hasattr(geom, "coords"):
                    for lon, lat, *_ in geom.coords:
                        road_points.append((lat, lon))

            if not road_points:
                logger.warning("No road coordinates extracted")
                return np.full(n, 50.0, dtype=np.float32)

            road_arr = np.array(road_points)
            logger.info("Road network: %d vertices", len(road_arr))

            # Log coverage bounds to flag incomplete data
            road_lat_range = (road_arr[:, 0].min(), road_arr[:, 0].max())
            road_lon_range = (road_arr[:, 1].min(), road_arr[:, 1].max())  # noqa: F841
            grid_lat_range = (grid_lats.min(), grid_lats.max())
            grid_lon_range = (grid_lons.min(), grid_lons.max())  # noqa: F841
            if (road_lat_range[1] - road_lat_range[0]) < 0.5 * (
                grid_lat_range[1] - grid_lat_range[0]
            ):
                logger.warning(
                    "Road data covers only lat %.1f–%.1f (grid: %.1f–%.1f); "
                    "distances capped at %.0f km for uncovered areas",
                    *road_lat_range,
                    *grid_lat_range,
                    self._MAX_ROAD_DISTANCE_KM,
                )

            # Scale coordinates to approximate km for distance
            # At BC latitudes (~54°N): 1° lat ≈ 111 km, 1° lon ≈ 65 km
            lat_scale = 111.0
            lon_scale = 111.0 * np.cos(np.radians(54.0))

            road_scaled = road_arr * [lat_scale, lon_scale]
            grid_scaled = np.column_stack([grid_lats * lat_scale, grid_lons * lon_scale])

            tree = KDTree(road_scaled)
            distances, _ = tree.query(grid_scaled)
            distances = np.clip(distances, 0.0, self._MAX_ROAD_DISTANCE_KM)

            logger.info(
                "Distance to road: min=%.1f km, median=%.1f km, max=%.1f km",
                distances.min(),
                np.median(distances),
                distances.max(),
            )
            return distances.astype(np.float32)

        except Exception as e:
            logger.warning("Road distance computation failed: %s — defaulting to 50 km", e)
            return np.full(n, 50.0, dtype=np.float32)

    def compute_fwi_season(
        self,
        weather_features: dict[str, np.ndarray],
        cell_ids: np.ndarray,
        start_idx: int = 0,
        prev_fwi_state: dict | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute FWI codes for a season of weather data (vectorized).

        Args:
            weather_features: dict with arrays of shape [n_days, n_cells]
            cell_ids: array of cell IDs
            start_idx: day index within the month to start from
            prev_fwi_state: dict[cell_id -> {ffmc, dmc, dc}] from previous period

        Returns:
            dict mapping component name -> array of shape [n_days, n_cells]
        """
        prev_state = prev_fwi_state or {}
        n_days = weather_features["temperature_c"].shape[0]
        n_cells = len(cell_ids)
        times = weather_features.get("_times")

        # Initialize carry-forward arrays from previous state
        prev_ffmc = np.full(n_cells, self.fwi_service.DEFAULT_FFMC, dtype=np.float64)
        prev_dmc = np.full(n_cells, self.fwi_service.DEFAULT_DMC, dtype=np.float64)
        prev_dc = np.full(n_cells, self.fwi_service.DEFAULT_DC, dtype=np.float64)

        for i, cid in enumerate(cell_ids):
            prev = prev_state.get(cid, {})
            if "ffmc" in prev:
                prev_ffmc[i] = prev["ffmc"]
            if "dmc" in prev:
                prev_dmc[i] = prev["dmc"]
            if "dc" in prev:
                prev_dc[i] = prev["dc"]

        # Output arrays: [n_days, n_cells]
        out = {
            k: np.zeros((n_days, n_cells), dtype=np.float32)
            for k in ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]
        }

        for day in range(n_days):
            month = times[day].month if times is not None else 7

            temp = weather_features["temperature_c"][day].astype(np.float64)
            rh = weather_features["rh_pct"][day].astype(np.float64)
            wind = weather_features["wind_kmh"][day].astype(np.float64)
            precip = weather_features["precip_24h_mm"][day].astype(np.float64)

            ffmc, dmc, dc, isi, bui, fwi = self.fwi_service.compute_daily_vec(
                temp, rh, wind, precip, month, prev_ffmc, prev_dmc, prev_dc
            )

            prev_ffmc, prev_dmc, prev_dc = ffmc, dmc, dc
            out["ffmc"][day] = ffmc
            out["dmc"][day] = dmc
            out["dc"][day] = dc
            out["isi"][day] = isi
            out["bui"][day] = bui
            out["fwi"][day] = fwi

        # Save final state back
        for i, cid in enumerate(cell_ids):
            prev_state[cid] = {
                "ffmc": float(prev_ffmc[i]),
                "dmc": float(prev_dmc[i]),
                "dc": float(prev_dc[i]),
            }

        return out

    def build_daily_features(
        self,
        target_date: date,
        weather: dict[str, np.ndarray],
        fwi_day: dict[str, np.ndarray],
        satellite: dict[str, np.ndarray],
        static: dict[str, np.ndarray],
        grid_df: pd.DataFrame,
        lightning: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Assemble the full feature vector for one day.

        Args:
            fwi_day: dict mapping FWI component name -> array[n_cells]

        Returns array of shape [n_cells, len(FEATURE_NAMES)].
        """
        n = len(grid_df)
        doy = target_date.timetuple().tm_yday
        doy_sin = np.sin(2 * np.pi * doy / 365)
        doy_cos = np.cos(2 * np.pi * doy / 365)

        if lightning is None:
            lightning = {}

        feature_matrix = np.column_stack(
            [
                # FWI components (6)
                fwi_day.get("ffmc", np.full(n, 85.0)),
                fwi_day.get("dmc", np.full(n, 6.0)),
                fwi_day.get("dc", np.full(n, 15.0)),
                fwi_day.get("isi", np.zeros(n)),
                fwi_day.get("bui", np.zeros(n)),
                fwi_day.get("fwi", np.zeros(n)),
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
                satellite.get("ndvi", np.full(n, 0.5)),
                satellite.get("snow_cover", np.zeros(n)),
                satellite.get("lai", np.full(n, 2.0)),
                # Topography + infrastructure (5)
                static.get("elevation_m", np.full(n, 500)),
                static.get("slope_deg", np.zeros(n)),
                static.get("aspect_deg", np.zeros(n)),
                static.get("hillshade", np.full(n, 128.0)),
                static.get("distance_to_road_km", np.full(n, 50.0)),
                # Temporal (2)
                np.full(n, doy_sin),
                np.full(n, doy_cos),
                # Lightning (2)
                lightning.get("lightning_24h", np.zeros(n)),
                lightning.get("lightning_72h", np.zeros(n)),
            ]
        )

        return feature_matrix.astype(np.float32)

    def process_training_period(
        self,
        grid_df: pd.DataFrame,
        start_year: int = 2015,
        end_year: int = 2024,
        fire_season_only: bool = True,
        chunk_days: int = 0,
    ) -> Path:
        """Process all raw data for the training period.

        Writes daily feature parquet files to data/processed/features/.

        Args:
            chunk_days: If > 0, split monthly output into chunks of this many days.
                Output files: features_{year}_{month}_w{chunk}.parquet.
                If 0, write one file per month (backward compatible).

        Returns the output directory.
        """
        output_dir = self.processed_dir / "features"
        output_dir.mkdir(parents=True, exist_ok=True)

        grid_lats = grid_df["lat"].values
        grid_lons = grid_df["lon"].values
        cell_ids = grid_df["cell_id"].values

        # Load static features once
        static = self.process_static_features(grid_lats, grid_lons)
        logger.info("Static features loaded: %d cells", len(grid_lats))

        prev_fwi_state: dict = {}

        for year in range(start_year, end_year + 1):
            # Load satellite data for this year
            satellite = self.process_satellite_year(year, grid_lats, grid_lons)
            logger.info("Satellite data loaded for %d", year)

            # Determine months to process
            months = range(4, 11) if fire_season_only else range(1, 13)

            for month in months:
                # Check if already processed (monthly or chunked)
                out_file = output_dir / f"features_{year}_{month:02d}.parquet"
                chunk_marker = output_dir / f"features_{year}_{month:02d}_w0.parquet"
                if out_file.exists() or (chunk_days > 0 and chunk_marker.exists()):
                    logger.info("Skipping (exists): %d-%02d", year, month)
                    continue

                # Process ERA5 weather for this month
                weather = self.process_era5_month(year, month, grid_lats, grid_lons)
                if not weather:
                    logger.warning("No ERA5 data for %d-%02d, skipping", year, month)
                    continue

                times = weather.get("_times")
                if times is None:
                    logger.warning("No time index for %d-%02d", year, month)
                    continue

                n_days = len(times)
                n_cells = len(cell_ids)
                logger.info("Processing %d-%02d: %d days x %d cells", year, month, n_days, n_cells)

                # Compute FWI for each day with carry-forward
                fwi_season = self.compute_fwi_season(
                    weather, cell_ids, prev_fwi_state=prev_fwi_state
                )

                # Stream day-by-day, flush to parquet every chunk_days
                # to avoid allocating (n_days x n_cells x 28) in memory
                effective_chunk = chunk_days if chunk_days > 0 else n_days
                chunk_buf: list[np.ndarray] = []
                chunk_dates: list[str] = []
                chunk_idx = 0

                for day_idx in range(n_days):
                    day_date = times[day_idx].date()

                    day_weather = {}
                    for key, arr in weather.items():
                        if key.startswith("_"):
                            continue
                        day_weather[key] = arr[day_idx] if arr.ndim == 2 else arr

                    fwi_day = {k: fwi_season[k][day_idx] for k in fwi_season}

                    features = self.build_daily_features(
                        target_date=day_date,
                        weather=day_weather,
                        fwi_day=fwi_day,
                        satellite=satellite,
                        static=static,
                        grid_df=grid_df,
                    )
                    chunk_buf.append(features)
                    chunk_dates.append(day_date.isoformat())

                    # Flush when chunk is full or last day
                    if len(chunk_buf) >= effective_chunk or day_idx == n_days - 1:
                        chunk_n = len(chunk_buf)
                        chunk_flat = np.vstack(chunk_buf).astype(np.float16)  # half storage

                        cell_ids_rep = np.tile(cell_ids, chunk_n)
                        dates_rep = np.repeat(chunk_dates, n_cells)
                        lats_rep = np.tile(grid_lats, chunk_n)
                        lons_rep = np.tile(grid_lons, chunk_n)

                        df = pd.DataFrame(chunk_flat, columns=FEATURE_NAMES)
                        df.insert(0, "cell_id", cell_ids_rep)
                        df.insert(1, "date", dates_rep)
                        df.insert(2, "lat", lats_rep)
                        df.insert(3, "lon", lons_rep)

                        if chunk_days > 0:
                            chunk_file = (
                                output_dir / f"features_{year}_{month:02d}_w{chunk_idx}.parquet"
                            )
                        else:
                            chunk_file = out_file

                        df.to_parquet(chunk_file, index=False)
                        logger.info("Wrote %s: %d rows", chunk_file.name, len(df))
                        chunk_idx += 1

                        # Free memory
                        chunk_buf.clear()
                        chunk_dates.clear()
                        del df, chunk_flat

        return output_dir


# Feature column names matching the feature vector
# 6 FWI + 10 Weather + 3 Vegetation + 5 Topography/Infra + 2 Temporal + 2 Lightning = 28
FEATURE_NAMES = [
    "ffmc",
    "dmc",
    "dc",
    "isi",
    "bui",
    "fwi",
    "temperature_c",
    "rh_pct",
    "wind_kmh",
    "wind_dir_deg",
    "precip_24h_mm",
    "soil_moisture_1",
    "soil_moisture_2",
    "soil_moisture_3",
    "soil_moisture_4",
    "evapotrans_mm",
    "ndvi",
    "snow_cover",
    "lai",
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "hillshade",
    "distance_to_road_km",
    "doy_sin",
    "doy_cos",
    "lightning_24h",
    "lightning_72h",
]
