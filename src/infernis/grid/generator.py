import logging

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box

logger = logging.getLogger(__name__)

# BC Albers projection (EPSG:3005) for equal-area grid cells
BC_CRS = "EPSG:3005"
WGS84 = "EPSG:4326"

# Simplified BC boundary polygon in WGS84
BC_BOUNDARY_SIMPLE = Polygon(
    [
        (-139.06, 48.30),
        (-139.06, 60.00),
        (-114.03, 60.00),
        (-114.03, 48.99),
        (-118.84, 49.00),
        (-120.00, 49.00),
        (-123.32, 48.30),
        (-139.06, 48.30),
    ]
)


def generate_bc_grid(resolution_km: float = 5.0) -> gpd.GeoDataFrame:
    """Generate a square grid covering British Columbia at given resolution.

    Grid is generated in BC Albers (EPSG:3005) for equal-area cells,
    then cell centroids are converted to WGS84 lat/lon.

    Uses vectorized spatial join for performance — handles 1km grids
    (~2M cells) in under 2 minutes.
    """
    bc_gdf = gpd.GeoDataFrame(geometry=[BC_BOUNDARY_SIMPLE], crs=WGS84)
    bc_proj = bc_gdf.to_crs(BC_CRS)
    bc_bounds = bc_proj.total_bounds  # [minx, miny, maxx, maxy]

    resolution_m = resolution_km * 1000
    x_coords = np.arange(bc_bounds[0], bc_bounds[2], resolution_m)
    y_coords = np.arange(bc_bounds[1], bc_bounds[3], resolution_m)

    logger.info(
        "Generating %dx%d candidate grid (%d cells) at %s km...",
        len(x_coords),
        len(y_coords),
        len(x_coords) * len(y_coords),
        resolution_km,
    )

    # Build all candidate cell boxes vectorized
    xx, yy = np.meshgrid(x_coords, y_coords)
    x_flat = xx.ravel()
    y_flat = yy.ravel()

    boxes = [box(x, y, x + resolution_m, y + resolution_m) for x, y in zip(x_flat, y_flat)]
    centroids_x = x_flat + resolution_m / 2
    centroids_y = y_flat + resolution_m / 2

    # Build centroid GeoDataFrame for spatial join
    centroid_points = gpd.points_from_xy(centroids_x, centroids_y)
    points_gdf = gpd.GeoDataFrame(
        {"box_idx": np.arange(len(boxes))},
        geometry=centroid_points,
        crs=BC_CRS,
    )

    # Spatial join: keep only centroids inside BC boundary
    inside = gpd.sjoin(points_gdf, bc_proj, predicate="within", how="inner")
    mask = inside["box_idx"].values

    logger.info(
        "Spatial filter: %d / %d centroids inside BC boundary",
        len(mask),
        len(boxes),
    )

    # Build final grid from filtered cells
    filtered_boxes = [boxes[i] for i in mask]
    filtered_cx = centroids_x[mask]
    filtered_cy = centroids_y[mask]

    # 7-digit cell IDs to support up to ~10M cells (1km grid has ~2M)
    res_tag = int(resolution_km)
    cell_ids = [f"BC-{res_tag}K-{i:07d}" for i in range(len(mask))]

    grid = gpd.GeoDataFrame(
        {"cell_id": cell_ids},
        geometry=filtered_boxes,
        crs=BC_CRS,
    )

    # Convert centroids to WGS84 for lat/lon
    centroids_proj = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(filtered_cx, filtered_cy),
        crs=BC_CRS,
    ).to_crs(WGS84)

    grid["lat"] = centroids_proj.geometry.y.round(6)
    grid["lon"] = centroids_proj.geometry.x.round(6)

    logger.info("Grid generated: %d cells at %s km resolution", len(grid), resolution_km)
    return grid
