"""Map tile endpoints for slippy map overlays (Google Maps, Leaflet, Mapbox)."""

from __future__ import annotations

import io
import logging
import math

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from infernis.config import settings

logger = logging.getLogger(__name__)

tiles_router = APIRouter(prefix=settings.api_prefix, tags=["tiles"])

TILE_SIZE = 256
MIN_ZOOM = 4
MAX_ZOOM = 12


def _tile_to_bbox(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Convert tile coordinates to lat/lon bounding box (south, west, north, east)."""
    n = 2**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return south, west, north, east


def _score_to_rgba_pixel(score: float) -> tuple[int, int, int, int]:
    """Convert a risk score to an RGBA color tuple."""
    if score < 0.05:
        r, g, b = 34, 197, 94
    elif score < 0.15:
        r, g, b = 59, 130, 246
    elif score < 0.35:
        r, g, b = 234, 179, 8
    elif score < 0.60:
        r, g, b = 249, 115, 22
    elif score < 0.80:
        r, g, b = 239, 68, 68
    else:
        r, g, b = 180, 20, 20
    alpha = min(255, int(50 + score * 205))
    return r, g, b, alpha


@tiles_router.get(
    "/tiles/{z}/{x}/{y}.png",
    tags=["tiles"],
    responses={
        200: {"content": {"image/png": {}}, "description": "256x256 PNG tile with risk overlay"},
        422: {"description": "Invalid zoom level (supported: 4-12)"},
        503: {"description": "Predictions not yet available"},
    },
)
async def get_tile(z: int, x: int, y: int):
    """Render a 256x256 PNG map tile with fire risk overlay.

    Uses standard slippy map tile coordinates (z/x/y). Overlay directly on
    Google Maps, Leaflet, or Mapbox. Transparent where no data.

    **Use cases:**
    - Google Maps tile layer overlay with one URL
    - Leaflet `L.tileLayer` for wildfire risk visualization
    - Mapbox raster source for risk heatmaps

    **Example — Google Maps JavaScript API:**
    ```javascript
    const riskLayer = new google.maps.ImageMapType({
      getTileUrl: (coord, zoom) =>
        `https://api.infernis.ca/v1/tiles/${zoom}/${coord.x}/${coord.y}.png`,
      tileSize: new google.maps.Size(256, 256),
      opacity: 0.6,
    });
    map.overlayMapTypes.push(riskLayer);
    ```

    **Example — Leaflet:**
    ```javascript
    L.tileLayer('https://api.infernis.ca/v1/tiles/{z}/{x}/{y}.png', {
      opacity: 0.6,
      attribution: 'Risk data &copy; INFERNIS'
    }).addTo(map);
    ```

    Zoom levels 4-12 supported. Tiles are cached for 10 minutes.
    """
    if z < MIN_ZOOM or z > MAX_ZOOM:
        raise HTTPException(
            status_code=422,
            detail=f"Zoom level {z} not supported. Valid range: {MIN_ZOOM}-{MAX_ZOOM}.",
        )

    from infernis.api.routes import _grid_cells, _predictions_cache

    if not _predictions_cache:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")

    south, west, north, east = _tile_to_bbox(z, x, y)

    raster = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    lat_step = (north - south) / TILE_SIZE
    lon_step = (east - west) / TILE_SIZE
    cells_rendered = 0

    for cell_id, cell in _grid_cells.items():
        lat, lon = cell["lat"], cell["lon"]
        if not (south <= lat <= north and west <= lon <= east):
            continue

        pred = _predictions_cache.get(cell_id)
        if pred is None:
            continue

        score = pred.get("score", 0.0)
        row = min(int((north - lat) / lat_step), TILE_SIZE - 1)
        col = min(int((lon - west) / lon_step), TILE_SIZE - 1)
        rgba = _score_to_rgba_pixel(score)

        block = max(1, int(settings.grid_resolution_km * 0.02 * (2**z)))
        r_start = max(0, row - block // 2)
        r_end = min(TILE_SIZE, row + block // 2 + 1)
        c_start = max(0, col - block // 2)
        c_end = min(TILE_SIZE, col + block // 2 + 1)
        raster[r_start:r_end, c_start:c_end] = rgba
        cells_rendered += 1

    from PIL import Image

    img = Image.fromarray(raster, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=600",
            "X-Cells-Rendered": str(cells_rendered),
        },
    )
