"""Coarse weather grid with interpolation to prediction grid.

Open-Meteo's GEM model has ~25km native resolution. Fetching weather for
every 5km prediction cell (84,535 points) wastes API quota — the data is
identical for neighboring cells. Instead:

1. Generate a coarse weather grid (~25km spacing, ~1,400 points for BC)
2. Fetch weather for those ~1,400 points (5 API batches, <1 minute)
3. Interpolate to the 5km prediction grid using nearest-neighbor (KDTree)

This reduces Open-Meteo API calls from 282 batches to ~5, eliminating
the rate-limit bottleneck entirely.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# Weather grid resolution in degrees (~50km at BC latitudes).
# GEM model native resolution is ~25km, but interpolation from 50km
# is more than adequate for fire weather — temperature, humidity, and
# wind are smooth fields at this scale. This keeps API calls under 10
# batches (free tier friendly).
WEATHER_GRID_RESOLUTION_DEG = 0.45

# BC bounding box
BC_LAT_MIN, BC_LAT_MAX = 48.3, 60.0
BC_LON_MIN, BC_LON_MAX = -139.06, -114.03


def generate_weather_grid() -> tuple[np.ndarray, np.ndarray]:
    """Generate a coarse regular grid for weather fetching.

    Returns:
        Tuple of (lats, lons) arrays for the weather grid points.
    """
    lats = np.arange(BC_LAT_MIN, BC_LAT_MAX, WEATHER_GRID_RESOLUTION_DEG)
    lons = np.arange(BC_LON_MIN, BC_LON_MAX, WEATHER_GRID_RESOLUTION_DEG)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    weather_lats = lat_grid.ravel()
    weather_lons = lon_grid.ravel()

    logger.info(
        "Weather grid: %d points at %.3f° resolution (%d lat x %d lon)",
        len(weather_lats),
        WEATHER_GRID_RESOLUTION_DEG,
        len(lats),
        len(lons),
    )
    return weather_lats, weather_lons


def build_interpolation_tree(
    weather_lats: np.ndarray, weather_lons: np.ndarray
) -> cKDTree:
    """Build a KDTree from weather grid coordinates for fast nearest-neighbor lookup."""
    coords = np.column_stack([weather_lats, weather_lons])
    return cKDTree(coords)


def interpolate_to_prediction_grid(
    weather_data: dict[str, np.ndarray],
    weather_lats: np.ndarray,
    weather_lons: np.ndarray,
    pred_lats: np.ndarray,
    pred_lons: np.ndarray,
    tree: cKDTree | None = None,
) -> dict[str, np.ndarray]:
    """Interpolate coarse weather data to the fine prediction grid.

    Uses nearest-neighbor interpolation via KDTree. This is appropriate
    because weather fields are smooth at 5km scale — the nearest 25km
    grid point's value is a good approximation.

    Args:
        weather_data: Dict of variable_name → array[n_weather_points].
        weather_lats, weather_lons: Coarse weather grid coordinates.
        pred_lats, pred_lons: Fine prediction grid coordinates.
        tree: Pre-built KDTree (optional, built if not provided).

    Returns:
        Dict of variable_name → array[n_pred_points] interpolated values.
    """
    if tree is None:
        tree = build_interpolation_tree(weather_lats, weather_lons)

    pred_coords = np.column_stack([pred_lats, pred_lons])
    _, indices = tree.query(pred_coords)

    result = {}
    for key, values in weather_data.items():
        if isinstance(values, np.ndarray) and values.ndim == 1:
            result[key] = values[indices]
        else:
            result[key] = values  # pass through non-array values

    return result


def interpolate_forecast_to_prediction_grid(
    forecast_data: dict[int, dict[str, np.ndarray]],
    weather_lats: np.ndarray,
    weather_lons: np.ndarray,
    pred_lats: np.ndarray,
    pred_lons: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    """Interpolate multi-day forecast from weather grid to prediction grid.

    Args:
        forecast_data: Dict of day_index → weather dict (from OpenMeteoPipeline).
        weather_lats, weather_lons: Coarse weather grid coordinates.
        pred_lats, pred_lons: Fine prediction grid coordinates.

    Returns:
        Same structure but with arrays resized to n_pred_points.
    """
    tree = build_interpolation_tree(weather_lats, weather_lons)

    result = {}
    for day_idx, day_data in forecast_data.items():
        result[day_idx] = interpolate_to_prediction_grid(
            day_data, weather_lats, weather_lons, pred_lats, pred_lons, tree
        )

    logger.info(
        "Interpolated %d forecast days from %d weather points to %d prediction cells",
        len(forecast_data),
        len(weather_lats),
        len(pred_lats),
    )
    return result
