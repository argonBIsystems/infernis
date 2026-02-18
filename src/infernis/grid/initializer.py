"""Grid initialization - generates BC grid and populates static features."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from infernis.config import settings
from infernis.grid.generator import generate_bc_grid

logger = logging.getLogger(__name__)


def initialize_grid(resolution_km: float | None = None) -> pd.DataFrame:
    """Generate the BC grid and populate static topographic features.

    Returns a DataFrame with all grid cells and their static attributes.
    This is called once during system setup, not during daily pipeline runs.
    """
    resolution_km = resolution_km or settings.grid_resolution_km
    logger.info("Generating BC grid at %s km resolution...", resolution_km)
    grid = generate_bc_grid(resolution_km)
    n_cells = len(grid)
    logger.info("Generated %d grid cells", n_cells)

    # Fetch topographic features from GEE
    grid = _populate_topography(grid)

    # Assign BEC zones and fuel types based on centroid location
    grid = _populate_bec_zones(grid)
    grid = _populate_fuel_types(grid)

    return grid


def _populate_topography(grid) -> pd.DataFrame:
    """Fetch elevation, slope, aspect, hillshade from GEE CDEM."""
    try:
        from infernis.pipelines.gee_pipeline import GEEPipeline

        gee = GEEPipeline()
        lats = grid["lat"].values
        lons = grid["lon"].values

        logger.info("Fetching topography from GEE for %d cells...", len(grid))
        topo = gee.fetch_topography(lats, lons)

        grid["elevation_m"] = topo["elevation_m"]
        grid["slope_deg"] = topo["slope_deg"]
        grid["aspect_deg"] = topo["aspect_deg"]
        grid["hillshade"] = topo["hillshade"]
        logger.info("Topography populated successfully")
    except Exception as e:
        logger.warning("GEE topography fetch failed (%s) - using defaults", e)
        n = len(grid)
        grid["elevation_m"] = np.zeros(n)
        grid["slope_deg"] = np.zeros(n)
        grid["aspect_deg"] = np.zeros(n)
        grid["hillshade"] = np.full(n, 128.0)

    return grid


def _populate_bec_zones(grid) -> pd.DataFrame:
    """Assign BEC zones to grid cells.

    Uses vectorized numpy conditions (latitude/elevation heuristic)
    instead of row iteration. Handles ~2M cells in seconds.
    """
    lat = grid["lat"].values
    lon = grid["lon"].values
    elev = grid["elevation_m"].values if "elevation_m" in grid.columns else np.zeros(len(grid))
    elev = np.where(np.isnan(elev), 0, elev)

    # Apply conditions in priority order (first match wins)
    conditions = [
        lat > 57,
        (lat > 55) & (elev > 1200),
        (lat > 55),
        elev > 1800,
        elev > 1400,
        (elev > 1000) & (lon < -124),
        (elev > 1000),
        (lat < 49.5) & (lon > -120) & (elev < 600),
        (lat < 49.5) & (lon > -120),
        (lon < -125) & (elev < 800),
        (lon < -125),
        (lon < -122) & (lat < 49.5),
        (lon < -122),
        lat > 52,
        elev < 1200,
    ]
    choices = [
        "BWBS",
        "SWB",
        "BWBS",
        "AT",
        "ESSF",
        "MH",
        "MS",
        "PP",
        "IDF",
        "CWH",
        "MH",
        "CDF",
        "ICH",
        "SBS",
        "IDF",
    ]

    grid["bec_zone"] = np.select(conditions, choices, default="ESSF")
    logger.info("BEC zones assigned to %d cells", len(grid))
    return grid


def _populate_fuel_types(grid) -> pd.DataFrame:
    """Assign fuel types based on BEC zone heuristics.

    In production, this would use the CFFDRS FBP fuel type raster.
    """
    zone_to_fuel = {
        "AT": "NF",
        "BG": "O1A",
        "BWBS": "C2",
        "CDF": "M1",
        "CWH": "C5",
        "ESSF": "C3",
        "ICH": "M2",
        "IDF": "C3",
        "MH": "C7",
        "MS": "C3",
        "PP": "C7",
        "SBPS": "C3",
        "SBS": "C2",
        "SWB": "C1",
    }
    grid["fuel_type"] = grid["bec_zone"].map(zone_to_fuel).fillna("C3")
    logger.info("Fuel types assigned to %d cells", len(grid))
    return grid


def grid_to_db(grid_df: pd.DataFrame, batch_size: int = 5000):
    """Insert grid cells into the database using batched bulk inserts."""

    from infernis.db.engine import SessionLocal
    from infernis.db.tables import GridCellDB

    db = SessionLocal()
    try:
        existing = db.query(GridCellDB).count()
        if existing > 0:
            logger.info("Grid already populated (%d cells), skipping insert", existing)
            return existing

        n_cells = len(grid_df)
        logger.info("Inserting %d grid cells into database (batch_size=%d)...", n_cells, batch_size)

        for start in range(0, n_cells, batch_size):
            batch = grid_df.iloc[start : start + batch_size]
            records = []
            for _, row in batch.iterrows():
                centroid_wkt = f"POINT({row['lon']} {row['lat']})"
                records.append(
                    GridCellDB(
                        cell_id=row["cell_id"],
                        geom=f"SRID=3005;{row['geometry'].wkt}",
                        centroid=f"SRID=4326;{centroid_wkt}",
                        lat=row["lat"],
                        lon=row["lon"],
                        bec_zone=row.get("bec_zone"),
                        fuel_type=row.get("fuel_type"),
                        elevation_m=row.get("elevation_m"),
                        slope_deg=row.get("slope_deg"),
                        aspect_deg=row.get("aspect_deg"),
                        hillshade=row.get("hillshade"),
                    )
                )
            db.bulk_save_objects(records)
            db.flush()
            if (start // batch_size) % 20 == 0:
                logger.info(
                    "  inserted %d / %d cells...", min(start + batch_size, n_cells), n_cells
                )

        db.commit()
        logger.info("Grid cells inserted successfully")
        return n_cells
    except Exception as e:
        db.rollback()
        logger.error("Failed to insert grid cells: %s", e)
        raise
    finally:
        db.close()


def load_grid_from_db() -> pd.DataFrame | None:
    """Load grid cells from database as a DataFrame."""
    from infernis.db.engine import SessionLocal
    from infernis.db.tables import GridCellDB

    db = SessionLocal()
    try:
        cells = db.query(GridCellDB).all()
        if not cells:
            return None

        records = []
        for c in cells:
            records.append(
                {
                    "cell_id": c.cell_id,
                    "lat": c.lat,
                    "lon": c.lon,
                    "bec_zone": c.bec_zone,
                    "fuel_type": c.fuel_type,
                    "elevation_m": c.elevation_m,
                    "slope_deg": c.slope_deg,
                    "aspect_deg": c.aspect_deg,
                    "hillshade": c.hillshade,
                }
            )
        return pd.DataFrame(records)
    finally:
        db.close()


def load_grid_from_parquet(path: str) -> pd.DataFrame:
    """Load a grid from a parquet file.

    Parquet is faster than DB for large grids (~2M cells at 1km).
    Strips geometry column if present to return a plain DataFrame
    matching the interface of load_grid_from_db().
    """
    import geopandas as _gpd

    logger.info("Loading grid from parquet: %s", path)
    gdf = _gpd.read_parquet(path)
    # Drop geometry for consistency with load_grid_from_db()
    df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    logger.info("Loaded %d grid cells from parquet", len(df))
    return df


def save_grid_to_parquet(grid_gdf, path: str) -> str:
    """Save a grid GeoDataFrame to parquet."""
    import geopandas as _gpd

    if not isinstance(grid_gdf, _gpd.GeoDataFrame):
        raise TypeError("Expected a GeoDataFrame, got %s" % type(grid_gdf).__name__)

    grid_gdf.to_parquet(path, index=False)
    logger.info("Grid saved to parquet: %s (%d cells)", path, len(grid_gdf))
    return path
