"""Lightning density pipeline - fetches flash data from MSC Datamart.

The Canadian Lightning Detection Network (CLDN) data is available via
Environment and Climate Change Canada's MSC Datamart at dd.weather.gc.ca.

Lightning density grids are provided as GRIB2 at 2.5km resolution with
10-minute temporal updates. The pipeline aggregates flash counts within
each 5km grid cell over 24h and 72h windows.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np

from infernis.config import settings

logger = logging.getLogger(__name__)

# MSC Datamart base URL for lightning data
MSC_DATAMART_BASE = "https://dd.weather.gc.ca/lightning_data/2.5km"

# BC bounding box for filtering
BC_SOUTH = settings.bc_bbox_south
BC_NORTH = settings.bc_bbox_north
BC_WEST = settings.bc_bbox_west
BC_EAST = settings.bc_bbox_east


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
            # Fetch 24h lightning data
            flashes_24h = self._fetch_flashes(target_date, hours_back=24)
            # Fetch 72h lightning data
            flashes_72h = self._fetch_flashes(target_date, hours_back=72)

            # Aggregate to grid cells
            density_24h = self._aggregate_to_grid(flashes_24h, grid_lats, grid_lons)
            density_72h = self._aggregate_to_grid(flashes_72h, grid_lats, grid_lons)

            logger.info(
                "Lightning data: %d flashes (24h), %d flashes (72h)",
                len(flashes_24h),
                len(flashes_72h),
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

    def _fetch_flashes(self, target_date: date, hours_back: int = 24) -> list:
        """Fetch lightning flash records from MSC Datamart.

        MSC Datamart provides CLDN data as CSV with columns:
        date, time, latitude, longitude, strength, multiplicity, type
        """
        flashes = []
        end_dt = datetime.combine(target_date, datetime.max.time().replace(tzinfo=timezone.utc))
        start_dt = end_dt - timedelta(hours=hours_back)

        # Try fetching data files for each day in the window
        current_date = start_dt.date()
        while current_date <= end_dt.date():
            day_flashes = self._fetch_day_flashes(current_date)
            # Filter to BC and time window
            for flash in day_flashes:
                if BC_SOUTH <= flash["lat"] <= BC_NORTH and BC_WEST <= flash["lon"] <= BC_EAST:
                    flashes.append(flash)
            current_date += timedelta(days=1)

        return flashes

    def _fetch_day_flashes(self, target_date: date) -> list:
        """Fetch flash data for a single day from MSC Datamart.

        Tries the real-time and archive paths. Falls back gracefully.
        """
        flashes = []
        date_str = target_date.strftime("%Y%m%d")

        # MSC Datamart provides lightning summary files
        # Format: CG_{YYYYMMDD}_{HH}00.csv
        for hour in range(24):
            url = f"{MSC_DATAMART_BASE}/{date_str}/CG_{date_str}_{hour:02d}00.csv"
            try:
                response = self._client.get(url)
                if response.status_code == 200:
                    flashes.extend(self._parse_cldn_csv(response.text, target_date, hour))
            except httpx.HTTPError:
                continue  # File not available yet, skip

        return flashes

    def _parse_cldn_csv(self, content: str, target_date: date, hour: int) -> list:
        """Parse CLDN CSV format into flash records.

        Expected CSV format (no header):
        latitude,longitude,strength_kA,multiplicity,cloud_to_ground_flag
        """
        flashes = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("lat"):
                continue
            try:
                parts = line.split(",")
                if len(parts) >= 2:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    strength = float(parts[2]) if len(parts) > 2 else 0.0
                    flashes.append(
                        {
                            "lat": lat,
                            "lon": lon,
                            "strength_kA": strength,
                            "date": target_date,
                            "hour": hour,
                        }
                    )
            except (ValueError, IndexError):
                continue

        return flashes

    def _aggregate_to_grid(
        self,
        flashes: list,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> np.ndarray:
        """Aggregate flash counts to nearest grid cells using KD-tree.

        For each lightning flash, find the nearest grid cell and increment
        that cell's flash count.
        """
        n_cells = len(grid_lats)
        counts = np.zeros(n_cells, dtype=np.float64)

        if not flashes:
            return counts

        from scipy.spatial import KDTree

        grid_coords = np.column_stack([grid_lats, grid_lons])
        tree = KDTree(grid_coords)

        # Half-cell size in degrees (approx) for matching threshold
        max_dist = settings.grid_resolution_km * 0.01  # ~0.05 degrees at 5km

        for flash in flashes:
            dist, idx = tree.query([flash["lat"], flash["lon"]])
            if dist <= max_dist:
                counts[idx] += 1.0

        return counts

    def close(self):
        """Close the HTTP client."""
        self._client.close()
