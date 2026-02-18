"""Training data assembler.

Builds labeled training datasets by combining:
- Processed feature matrices (from data_processor.py)
- Fire history (CNFDB + BC fire incidents) for positive samples
- Non-fire cells for negative samples with spatiotemporal buffering

Output: A single parquet file with features + binary label (fire=1, no_fire=0).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from infernis.pipelines.data_processor import FEATURE_NAMES

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

# Spatiotemporal buffer for negative sampling:
# Exclude cells within this distance (km) and time window (days) of a fire
SPATIAL_BUFFER_KM = 10.0
TEMPORAL_BUFFER_DAYS = 7

# Target negative:positive ratio for training
NEG_POS_RATIO = 10


class FeatureBuilder:
    """Assembles labeled training datasets from processed features and fire history."""

    def __init__(
        self,
        processed_dir: Path | None = None,
        raw_dir: Path | None = None,
    ):
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.raw_dir = raw_dir or RAW_DIR

    def load_fire_history(self) -> pd.DataFrame:
        """Load fire occurrence records from available sources.

        Returns DataFrame with columns: lat, lon, date, size_ha, source.
        """
        fires = []

        # Source 1: CNFDB (Canadian National Fire Database)
        cnfdb_dir = self.raw_dir / "cnfdb"
        if cnfdb_dir.exists():
            fires.extend(self._load_cnfdb(cnfdb_dir))

        # Source 2: BC Fire Incidents
        bc_incidents_dir = self.raw_dir / "bc_fire_incidents"
        if bc_incidents_dir.exists():
            fires.extend(self._load_bc_incidents(bc_incidents_dir))

        # Source 3: BC Fire Perimeters
        bc_perimeters_dir = self.raw_dir / "bc_fire_perimeters"
        if bc_perimeters_dir.exists():
            fires.extend(self._load_bc_perimeters(bc_perimeters_dir))

        if not fires:
            logger.warning("No fire history data found")
            return pd.DataFrame(columns=["lat", "lon", "date", "size_ha", "source"])

        df = pd.DataFrame(fires)
        df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
        df = df.dropna(subset=["lat", "lon", "date"])
        logger.info("Loaded %d fire records from %d sources", len(df), df["source"].nunique())
        return df

    def _load_cnfdb(self, cnfdb_dir: Path) -> list[dict]:
        """Parse CNFDB shapefile/CSV fire records."""
        records = []
        try:
            import geopandas as gpd

            for shp in list(cnfdb_dir.glob("*.shp")) + list(cnfdb_dir.glob("*.geojson")):
                gdf = gpd.read_file(shp)
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs("EPSG:4326")
                # CNFDB fields vary; try common column names
                lat_col = next(
                    (c for c in gdf.columns if c.lower() in ("lat", "latitude", "rep_date")), None
                )
                lon_col = next(
                    (c for c in gdf.columns if c.lower() in ("lon", "longitude", "long")), None
                )
                date_col = next(
                    (c for c in gdf.columns if "date" in c.lower() or "rep_date" in c.lower()), None
                )
                size_col = next(
                    (c for c in gdf.columns if "size" in c.lower() or "area" in c.lower()), None
                )

                for _, row in gdf.iterrows():
                    lat = row.geometry.y if hasattr(row.geometry, "y") else row.get(lat_col)
                    lon = row.geometry.x if hasattr(row.geometry, "x") else row.get(lon_col)
                    records.append(
                        {
                            "lat": float(lat) if lat is not None else None,
                            "lon": float(lon) if lon is not None else None,
                            "date": str(row.get(date_col, "")) if date_col else None,
                            "size_ha": float(row.get(size_col, 0)) if size_col else 0,
                            "source": "cnfdb",
                        }
                    )
        except Exception as e:
            logger.warning("Error loading CNFDB: %s", e)

        # Also try CSV format
        for csv_file in cnfdb_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
                lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
                date_col = next((c for c in df.columns if "date" in c.lower()), None)
                size_col = next(
                    (c for c in df.columns if "size" in c.lower() or "area" in c.lower()), None
                )

                if lat_col and lon_col:
                    for _, row in df.iterrows():
                        records.append(
                            {
                                "lat": float(row[lat_col]) if pd.notna(row[lat_col]) else None,
                                "lon": float(row[lon_col]) if pd.notna(row[lon_col]) else None,
                                "date": str(row.get(date_col, "")) if date_col else None,
                                "size_ha": float(row.get(size_col, 0))
                                if size_col and pd.notna(row.get(size_col))
                                else 0,
                                "source": "cnfdb",
                            }
                        )
            except Exception as e:
                logger.warning("Error loading CNFDB CSV %s: %s", csv_file.name, e)

        return records

    def _load_bc_incidents(self, bc_dir: Path) -> list[dict]:
        """Parse BC Wildfire Service incident data (shp or geojson)."""
        records = []
        try:
            import geopandas as gpd

            for f in list(bc_dir.glob("*.shp")) + list(bc_dir.glob("*.geojson")):
                gdf = gpd.read_file(f)
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs("EPSG:4326")
                for _, row in gdf.iterrows():
                    if row.geometry is None:
                        continue
                    lat = row.geometry.y if hasattr(row.geometry, "y") else None
                    lon = row.geometry.x if hasattr(row.geometry, "x") else None
                    fire_date = (
                        row.get("IGNITION_DATE")
                        or row.get("FIRE_DATE")
                        or row.get("IGN_DATE")
                        or row.get("START_DATE")
                    )
                    size = (
                        row.get("CURRENT_SIZE") or row.get("SIZE_HA") or row.get("FIRE_SIZE") or 0
                    )
                    records.append(
                        {
                            "lat": float(lat) if lat is not None else None,
                            "lon": float(lon) if lon is not None else None,
                            "date": str(fire_date) if fire_date else None,
                            "size_ha": float(size) if size else 0,
                            "source": "bc_incidents",
                        }
                    )
        except Exception as e:
            logger.warning("Error loading BC incidents: %s", e)

        return records

    def _load_bc_perimeters(self, bc_dir: Path) -> list[dict]:
        """Parse BC fire perimeter polygons (use centroids as point locations)."""
        records = []
        try:
            import geopandas as gpd

            for f in list(bc_dir.glob("*.shp")) + list(bc_dir.glob("*.geojson")):
                gdf = gpd.read_file(f)
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs("EPSG:4326")
                for _, row in gdf.iterrows():
                    if row.geometry is None:
                        continue
                    centroid = row.geometry.centroid
                    fire_date = (
                        row.get("FIRE_DATE")
                        or row.get("IGN_DATE")
                        or row.get("IGNITION_DATE")
                        or row.get("FIRE_YEAR")
                    )
                    size = (
                        row.get("FIRE_SIZE_HECTARES")
                        or row.get("SIZE_HA")
                        or row.get("FIRE_SIZE_HA")
                        or row.get("CURRENT_SIZE")
                        or 0
                    )
                    records.append(
                        {
                            "lat": centroid.y,
                            "lon": centroid.x,
                            "date": str(fire_date) if fire_date else None,
                            "size_ha": float(size) if size else 0,
                            "source": "bc_perimeters",
                        }
                    )
        except Exception as e:
            logger.warning("Error loading BC perimeters: %s", e)

        return records

    def assign_fires_to_grid(
        self,
        fires: pd.DataFrame,
        grid_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Map fire occurrences to nearest grid cells.

        Returns DataFrame with columns: cell_id, date, fire=True.
        """
        if fires.empty:
            return pd.DataFrame(columns=["cell_id", "date", "fire"])

        grid_lats = grid_df["lat"].values
        grid_lons = grid_df["lon"].values
        cell_ids = grid_df["cell_id"].values

        # Build KD-tree on grid centroids
        tree = KDTree(np.column_stack([grid_lats, grid_lons]))

        # Find nearest cell for each fire
        fire_coords = np.column_stack([fires["lat"].values, fires["lon"].values])
        distances, indices = tree.query(fire_coords)

        # Filter out invalid indices (NaN coords produce sentinel index == n)
        valid = indices < len(cell_ids)
        indices = indices[valid]
        distances = distances[valid]
        fire_dates = fires["date"].values[valid]

        fire_cells = pd.DataFrame(
            {
                "cell_id": cell_ids[indices],
                "date": fire_dates,
                "fire": True,
                "distance_deg": distances,
            }
        )

        # Filter: only keep fires that are within ~50km of a grid cell
        # At BC latitudes, 1 degree ~ 80km
        fire_cells = fire_cells[fire_cells["distance_deg"] < 0.6]
        fire_cells = fire_cells.drop(columns=["distance_deg"])

        # Deduplicate: one fire per cell per day
        fire_cells = fire_cells.drop_duplicates(subset=["cell_id", "date"])

        logger.info(
            "Assigned %d fires to %d unique grid cells",
            len(fire_cells),
            fire_cells["cell_id"].nunique(),
        )
        return fire_cells

    def sample_negatives(
        self,
        fire_cells: pd.DataFrame,
        grid_df: pd.DataFrame,
        features_dir: Path,
        ratio: int = NEG_POS_RATIO,
    ) -> pd.DataFrame:
        """Generate negative samples (no-fire cell-days) with spatiotemporal buffering.

        Excludes cells within SPATIAL_BUFFER_KM of any fire within TEMPORAL_BUFFER_DAYS.
        Returns DataFrame with same columns as fire_cells but fire=False.
        """
        if fire_cells.empty:
            logger.warning("No positive samples to base negative sampling on")
            return pd.DataFrame(columns=["cell_id", "date", "fire"])

        n_negatives = len(fire_cells) * ratio
        cell_ids = grid_df["cell_id"].values

        # Build a set of (cell_id, date) pairs to exclude
        # Include the fire cells themselves + spatiotemporal buffer
        exclude_set = set()

        # Add the fire cells themselves
        for _, row in fire_cells.iterrows():
            exclude_set.add((row["cell_id"], pd.Timestamp(row["date"]).date()))

        # Build spatial buffer using KD-tree
        grid_lats = grid_df["lat"].values
        grid_lons = grid_df["lon"].values
        tree = KDTree(np.column_stack([grid_lats, grid_lons]))
        cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

        for _, row in fire_cells.iterrows():
            fire_idx = cell_id_to_idx.get(row["cell_id"])
            if fire_idx is None:
                continue
            fire_coord = np.array([grid_lats[fire_idx], grid_lons[fire_idx]])

            # ~1 degree latitude = 111 km
            buffer_deg = SPATIAL_BUFFER_KM / 111.0
            nearby_indices = tree.query_ball_point(fire_coord, buffer_deg)

            fire_date = pd.Timestamp(row["date"])
            for offset in range(-TEMPORAL_BUFFER_DAYS, TEMPORAL_BUFFER_DAYS + 1):
                buffered_date = (fire_date + pd.Timedelta(days=offset)).date()
                for idx in nearby_indices:
                    exclude_set.add((cell_ids[idx], buffered_date))

        logger.info("Excluding %d cell-day pairs from negative sampling", len(exclude_set))

        # Collect dates from feature files (just unique dates, not all rows)
        date_pool = []
        for parquet_file in sorted(features_dir.glob("features_*.parquet")):
            df = pd.read_parquet(parquet_file, columns=["date"])
            unique_dates = df["date"].unique()
            date_pool.extend(unique_dates)
        date_pool = sorted(set(date_pool))

        if not date_pool:
            logger.warning("No available cell-day pairs for negative sampling")
            return pd.DataFrame(columns=["cell_id", "date", "fire"])

        logger.info("Sampling negatives from %d dates x %d cells", len(date_pool), len(cell_ids))

        # Sample by randomly picking (cell_id, date) pairs and rejecting exclusions
        rng = np.random.default_rng(42)
        selected = []
        attempts = 0
        max_attempts = n_negatives * 5

        while len(selected) < n_negatives and attempts < max_attempts:
            batch_size = min(n_negatives * 2, max_attempts - attempts)
            rand_cell_idx = rng.integers(0, len(cell_ids), size=batch_size)
            rand_date_idx = rng.integers(0, len(date_pool), size=batch_size)

            for i in range(batch_size):
                cid = cell_ids[rand_cell_idx[i]]
                d = pd.Timestamp(date_pool[rand_date_idx[i]]).date()
                if (cid, d) not in exclude_set:
                    selected.append((cid, d))
                    if len(selected) >= n_negatives:
                        break
            attempts += batch_size

        if not selected:
            logger.warning("No available cell-day pairs for negative sampling")
            return pd.DataFrame(columns=["cell_id", "date", "fire"])

        negatives = pd.DataFrame(selected, columns=["cell_id", "date"])
        negatives["fire"] = False
        negatives["date"] = pd.to_datetime(negatives["date"])

        logger.info(
            "Sampled %d negative examples (ratio %.1f:1)",
            len(negatives),
            len(negatives) / max(len(fire_cells), 1),
        )
        return negatives

    def build_training_dataset(
        self,
        grid_df: pd.DataFrame,
        start_year: int = 2015,
        end_year: int = 2024,
    ) -> Path:
        """Build the complete labeled training dataset.

        Steps:
        1. Load fire history -> assign to grid cells (positive samples)
        2. Sample negative examples with spatiotemporal buffering
        3. Join with processed feature matrices
        4. Write to data/processed/training_data.parquet

        Returns path to the output file.
        """
        features_dir = self.processed_dir / "features"
        output_path = self.processed_dir / "training_data.parquet"

        # Step 1: Load and assign fires
        fires = self.load_fire_history()
        if not fires.empty:
            # Filter to training period
            fires["date"] = pd.to_datetime(fires["date"], errors="coerce")
            fires = fires[
                (fires["date"].dt.year >= start_year) & (fires["date"].dt.year <= end_year)
            ]

        fire_cells = self.assign_fires_to_grid(fires, grid_df)

        # Step 2: Sample negatives
        negatives = self.sample_negatives(fire_cells, grid_df, features_dir)

        # Step 3: Combine positive + negative
        fire_cells["date"] = pd.to_datetime(fire_cells["date"])
        labels = pd.concat([fire_cells, negatives], ignore_index=True)
        labels = labels.sort_values(["date", "cell_id"]).reset_index(drop=True)
        logger.info(
            "Training labels: %d positive, %d negative",
            labels["fire"].sum(),
            (~labels["fire"]).sum(),
        )

        # Step 4: Join with features
        # Group labels by year-month to match feature files
        labels["year"] = labels["date"].dt.year
        labels["month"] = labels["date"].dt.month
        labels["date_str"] = labels["date"].dt.strftime("%Y-%m-%d")

        all_rows = []
        for (year, month), group in labels.groupby(["year", "month"]):
            # Find feature files: monthly or chunked weekly
            prefix = f"features_{int(year)}_{int(month):02d}"
            monthly_file = features_dir / f"{prefix}.parquet"
            chunk_files = sorted(features_dir.glob(f"{prefix}_w*.parquet"))

            parquet_files = [monthly_file] if monthly_file.exists() else chunk_files
            if not parquet_files:
                logger.warning("Feature files missing for %d-%02d", year, month)
                continue

            # Stream chunk files one at a time to avoid OOM on large grids
            needed_cells = set(group["cell_id"].values)
            needed_dates = set(group["date_str"].values)

            for pf in parquet_files:
                chunk = pd.read_parquet(pf)
                chunk["date_str"] = chunk["date"].astype(str)

                # Pre-filter to only needed rows before merge
                mask = chunk["cell_id"].isin(needed_cells) & chunk["date_str"].isin(needed_dates)
                chunk = chunk[mask]
                if chunk.empty:
                    del chunk
                    continue

                merged = group.merge(
                    chunk,
                    left_on=["cell_id", "date_str"],
                    right_on=["cell_id", "date_str"],
                    how="inner",
                    suffixes=("", "_feat"),
                )
                del chunk

                if len(merged) > 0:
                    all_rows.append(merged)
                del merged

        if not all_rows:
            logger.error("No feature data matched to labels")
            # Write empty dataset with correct schema
            empty_df = pd.DataFrame(
                columns=["cell_id", "date", "fire", "lat", "lon"] + FEATURE_NAMES
            )
            empty_df.to_parquet(output_path, index=False)
            return output_path

        result = pd.concat(all_rows, ignore_index=True)

        # Select final columns
        keep_cols = ["cell_id", "date_str", "lat", "lon", "fire"] + FEATURE_NAMES
        available_cols = [c for c in keep_cols if c in result.columns]
        result = result[available_cols].rename(columns={"date_str": "date"})
        result["fire"] = result["fire"].astype(int)

        result.to_parquet(output_path, index=False)
        logger.info(
            "Training dataset written to %s: %d samples, %d features",
            output_path,
            len(result),
            len(FEATURE_NAMES),
        )
        return output_path
