"""Lightning density pipeline - fetches flash data from MSC Datamart.

The Canadian Lightning Detection Network (CLDN) data is available via
Environment and Climate Change Canada's MSC Datamart at dd.weather.gc.ca.

Lightning density grids are provided as GeoTIFF at 2.5km resolution with
10-minute temporal updates. The pipeline aggregates flash counts within
each 5km grid cell over 24h and 72h windows.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx
import numpy as np

from infernis.config import settings

logger = logging.getLogger(__name__)

# MSC Datamart URLs for lightning data (restructured late 2025)
# /today/ has rolling ~24h; archived days under /{YYYYMMDD}/WXO-DD/lightning/
MSC_TODAY_URL = "https://dd.weather.gc.ca/today/lightning"
MSC_ARCHIVE_URL = "https://dd.weather.gc.ca/{date_str}/WXO-DD/lightning"

# BC bounding box for filtering
BC_SOUTH = settings.bc_bbox_south
BC_NORTH = settings.bc_bbox_north
BC_WEST = settings.bc_bbox_west
BC_EAST = settings.bc_bbox_east

# GeoTIFF nodata value
NODATA = -999.0


class LightningPipeline:
    """Fetches and aggregates lightning strike data for BC grid cells."""

    def __init__(self, data_dir: str = "data/raw/lightning"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(timeout=30.0)

    def fetch_lightning_density(
        self,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        target_date: date,
    ) -> dict:
        """Fetch lightning flash counts for grid cells.

        Returns dict with keys:
        - lightning_24h: flash count per cell in last 24 hours
        - lightning_72h: flash count per cell in last 72 hours
        """
        n_cells = len(grid_lats)

        try:
            density_24h = self._fetch_window(target_date, grid_lats, grid_lons, hours_back=24)
            density_72h = self._fetch_window(target_date, grid_lats, grid_lons, hours_back=72)

            total_24h = int(density_24h.sum())
            total_72h = int(density_72h.sum())
            logger.info(
                "Lightning data: %d flashes (24h), %d flashes (72h)",
                total_24h,
                total_72h,
            )

            return {
                "lightning_24h": density_24h,
                "lightning_72h": density_72h,
            }

        except Exception as e:
            logger.error("Lightning fetch failed: %s. Using zeros.", e)
            return {
                "lightning_24h": np.zeros(n_cells),
                "lightning_72h": np.zeros(n_cells),
            }

    def _fetch_window(
        self,
        target_date: date,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        hours_back: int,
    ) -> np.ndarray:
        """Fetch and aggregate lightning GeoTIFFs over a time window."""
        n_cells = len(grid_lats)
        counts = np.zeros(n_cells, dtype=np.float64)

        end_dt = datetime.combine(target_date + timedelta(days=1), datetime.min.time())
        end_dt = end_dt.replace(tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(hours=hours_back)

        # Iterate over each day in the window
        current_date = start_dt.date()
        files_processed = 0
        while current_date <= end_dt.date():
            timestamps = self._generate_timestamps(current_date, start_dt, end_dt)
            for ts in timestamps:
                density = self._fetch_and_sample_tif(ts, grid_lats, grid_lons)
                if density is not None:
                    counts += density
                    files_processed += 1
            current_date += timedelta(days=1)

        logger.debug("Lightning: processed %d TIF files for %dh window", files_processed, hours_back)
        return counts

    def _generate_timestamps(
        self, day: date, window_start: datetime, window_end: datetime
    ) -> list[str]:
        """Generate 10-minute timestamp strings for a day within the time window."""
        timestamps = []
        for hour in range(24):
            for minute in range(0, 60, 10):
                dt = datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)
                if window_start <= dt < window_end:
                    timestamps.append(dt.strftime("%Y%m%dT%H%MZ"))
        return timestamps

    def _fetch_and_sample_tif(
        self, timestamp: str, grid_lats: np.ndarray, grid_lons: np.ndarray
    ) -> np.ndarray | None:
        """Download a lightning GeoTIFF and sample it at grid cell locations."""
        import rasterio
        from rasterio.transform import rowcol

        filename = f"{timestamp}_MSC_Lightning_2.5km.tif"

        # Try cache first
        cached = self.data_dir / filename
        if cached.exists() and cached.stat().st_size > 0:
            return self._read_tif(cached, grid_lats, grid_lons)

        # Determine URL: today's data vs archive
        date_str = timestamp[:8]
        today_str = date.today().strftime("%Y%m%d")
        if date_str == today_str:
            url = f"{MSC_TODAY_URL}/{filename}"
        else:
            url = f"{MSC_ARCHIVE_URL.format(date_str=date_str)}/{filename}"

        try:
            response = self._client.get(url)
            if response.status_code == 200:
                cached.write_bytes(response.content)
                return self._read_tif(cached, grid_lats, grid_lons)
        except httpx.HTTPError:
            pass

        return None

    def _read_tif(
        self, filepath: Path, grid_lats: np.ndarray, grid_lons: np.ndarray
    ) -> np.ndarray:
        """Read a lightning GeoTIFF and sample at grid cell locations."""
        import rasterio

        n_cells = len(grid_lats)
        counts = np.zeros(n_cells, dtype=np.float64)

        with rasterio.open(filepath) as ds:
            data = ds.read(1)
            transform = ds.transform

            # Convert grid lat/lon to row/col in the raster
            rows, cols = rasterio.transform.rowcol(transform, grid_lons, grid_lats)
            rows = np.array(rows)
            cols = np.array(cols)

            # Filter to valid indices within raster bounds
            valid = (
                (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
            )

            values = np.zeros(n_cells, dtype=np.float64)
            if valid.any():
                values[valid] = data[rows[valid], cols[valid]]

            # Replace nodata with 0, clip negatives
            values[values <= NODATA] = 0.0
            values = np.maximum(values, 0.0)

        return values

    def close(self):
        """Close the HTTP client."""
        self._client.close()
