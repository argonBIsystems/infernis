"""INFERNIS API routes - REST endpoints for fire risk data."""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from infernis.config import settings
from infernis.models.enums import DangerLevel
from infernis.models.schemas import (
    ForecastDay,
    ForecastResponse,
    FWIComponents,
    RiskResponse,
    RiskScore,
    StatusResponse,
    WeatherConditions,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.api_prefix)

# In-memory store for the latest predictions (populated by pipeline)
# In production this reads from Redis/PostGIS
_predictions_cache: dict = {}
_grid_cells: dict = {}
_last_pipeline_run: str | None = None
_kdtree = None
_cell_ids_ordered: list = []

# Forecast cache: cell_id → list of forecast day dicts
_forecast_cache: dict[str, list[dict]] = {}
_forecast_base_date: str | None = None


def set_predictions_cache(predictions: dict, grid_cells: dict, run_time: str):
    """Called by the pipeline to update the prediction cache."""
    global _predictions_cache, _grid_cells, _last_pipeline_run, _kdtree, _cell_ids_ordered
    _predictions_cache = predictions
    _grid_cells = grid_cells
    _last_pipeline_run = run_time

    # Build KD-tree for nearest-cell lookups
    if grid_cells:
        import numpy as np
        from scipy.spatial import KDTree

        _cell_ids_ordered = list(grid_cells.keys())
        coords = np.array(
            [[grid_cells[cid]["lat"], grid_cells[cid]["lon"]] for cid in _cell_ids_ordered]
        )
        _kdtree = KDTree(coords)


def set_forecast_cache(forecasts: dict[str, list[dict]], base_date: str):
    """Called by the forecast pipeline to update the forecast cache."""
    global _forecast_cache, _forecast_base_date
    _forecast_cache = forecasts
    _forecast_base_date = base_date


def _find_nearest_cell(lat: float, lon: float) -> str | None:
    """Find the nearest grid cell to the given coordinates."""
    if _kdtree is None or not _cell_ids_ordered:
        return None
    _, idx = _kdtree.query([lat, lon])
    return _cell_ids_ordered[idx]


def _validate_bc_coords(lat: float, lon: float):
    """Validate coordinates are within BC boundaries."""
    if not (settings.bc_bbox_south <= lat <= settings.bc_bbox_north):
        raise HTTPException(
            status_code=422,
            detail=f"Latitude {lat} is outside BC boundaries ({settings.bc_bbox_south} to {settings.bc_bbox_north}).",
        )
    if not (settings.bc_bbox_west <= lon <= settings.bc_bbox_east):
        raise HTTPException(
            status_code=422,
            detail=f"Longitude {lon} is outside BC boundaries ({settings.bc_bbox_west} to {settings.bc_bbox_east}).",
        )


@router.get("/risk/{lat}/{lon}")
async def get_risk(lat: float, lon: float):
    """Point risk query. Returns fire risk for the nearest grid cell."""
    _validate_bc_coords(lat, lon)

    cell_id = _find_nearest_cell(lat, lon)
    if cell_id is None or cell_id not in _predictions_cache:
        raise HTTPException(
            status_code=503, detail="Predictions not yet available. Pipeline may be initializing."
        )

    pred = _predictions_cache[cell_id]
    cell = _grid_cells.get(cell_id, {})

    score = pred.get("score", 0.0)
    level = DangerLevel.from_score(score)

    return RiskResponse(
        location={"lat": lat, "lon": lon},
        grid_cell_id=cell_id,
        timestamp=pred.get("timestamp", datetime.now(timezone.utc).isoformat()),
        risk=RiskScore(score=score, level=level),
        fwi=FWIComponents(
            ffmc=pred.get("ffmc", 0.0),
            dmc=pred.get("dmc", 0.0),
            dc=pred.get("dc", 0.0),
            isi=pred.get("isi", 0.0),
            bui=pred.get("bui", 0.0),
            fwi=pred.get("fwi", 0.0),
        ),
        conditions=WeatherConditions(
            temperature_c=pred.get("temperature_c", 0.0),
            rh_pct=pred.get("rh_pct", 0.0),
            wind_kmh=pred.get("wind_kmh", 0.0),
            precip_24h_mm=pred.get("precip_24h_mm", 0.0),
            soil_moisture=pred.get("soil_moisture", 0.0),
            ndvi=pred.get("ndvi", 0.0),
            snow_cover=pred.get("snow_cover", False),
        ),
        context={
            "bec_zone": cell.get("bec_zone", ""),
            "fuel_type": cell.get("fuel_type", ""),
            "elevation_m": cell.get("elevation_m", 0),
        },
        next_update=pred.get("next_update", ""),
    )


@router.get("/forecast/{lat}/{lon}")
async def get_forecast(
    lat: float,
    lon: float,
    days: int = Query(default=10, ge=1, le=10, description="Number of forecast days"),
):
    """Multi-day fire risk forecast for a location (up to 10 days)."""
    _validate_bc_coords(lat, lon)

    if not _forecast_cache:
        raise HTTPException(
            status_code=503, detail="Forecast not yet available. Pipeline may be initializing."
        )

    cell_id = _find_nearest_cell(lat, lon)
    if cell_id is None or cell_id not in _forecast_cache:
        raise HTTPException(status_code=404, detail="No forecast data for this location.")

    forecast_days = _forecast_cache[cell_id][:days]

    return ForecastResponse(
        latitude=lat,
        longitude=lon,
        cell_id=cell_id,
        base_date=_forecast_base_date or "",
        forecast=[
            ForecastDay(
                valid_date=d["valid_date"],
                lead_day=d["lead_day"],
                risk_score=d["risk_score"],
                danger_level=d["danger_level"],
                danger_label=d["danger_label"],
                confidence=d["confidence"],
                fwi=FWIComponents(**d["fwi"]),
                data_source=d.get("data_source", ""),
            )
            for d in forecast_days
        ],
        generated_at=_last_pipeline_run or "",
    )


@router.get("/risk/zones")
async def get_risk_zones():
    """Returns aggregate risk levels for all BEC zones."""
    if not _predictions_cache:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")

    zones: dict = {}
    for cell_id, pred in _predictions_cache.items():
        cell = _grid_cells.get(cell_id, {})
        zone = cell.get("bec_zone", "UNKNOWN")
        if zone not in zones:
            zones[zone] = {"scores": [], "cells": 0, "high_risk": 0}
        score = pred.get("score", 0.0)
        zones[zone]["scores"].append(score)
        zones[zone]["cells"] += 1
        if score >= 0.60:
            zones[zone]["high_risk"] += 1

    result = []
    for zone, data in sorted(zones.items()):
        scores = data["scores"]
        avg = sum(scores) / len(scores) if scores else 0.0
        mx = max(scores) if scores else 0.0
        result.append(
            {
                "bec_zone": zone,
                "avg_risk_score": round(avg, 3),
                "max_risk_score": round(mx, 3),
                "level": DangerLevel.from_score(avg).value,
                "cell_count": data["cells"],
                "high_risk_cells": data["high_risk"],
            }
        )

    return {"zones": result, "timestamp": _last_pipeline_run}


@router.get("/fwi/{lat}/{lon}")
async def get_fwi(lat: float, lon: float):
    """Raw FWI components for a location."""
    _validate_bc_coords(lat, lon)

    cell_id = _find_nearest_cell(lat, lon)
    if cell_id is None or cell_id not in _predictions_cache:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")

    pred = _predictions_cache[cell_id]
    return {
        "location": {"lat": lat, "lon": lon},
        "grid_cell_id": cell_id,
        "timestamp": pred.get("timestamp", ""),
        "fwi": {
            "ffmc": pred.get("ffmc", 0.0),
            "dmc": pred.get("dmc", 0.0),
            "dc": pred.get("dc", 0.0),
            "isi": pred.get("isi", 0.0),
            "bui": pred.get("bui", 0.0),
            "fwi": pred.get("fwi", 0.0),
        },
    }


@router.get("/conditions/{lat}/{lon}")
async def get_conditions(lat: float, lon: float):
    """Current weather and environmental conditions."""
    _validate_bc_coords(lat, lon)

    cell_id = _find_nearest_cell(lat, lon)
    if cell_id is None or cell_id not in _predictions_cache:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")

    pred = _predictions_cache[cell_id]
    return {
        "location": {"lat": lat, "lon": lon},
        "grid_cell_id": cell_id,
        "timestamp": pred.get("timestamp", ""),
        "conditions": {
            "temperature_c": pred.get("temperature_c", 0.0),
            "rh_pct": pred.get("rh_pct", 0.0),
            "wind_kmh": pred.get("wind_kmh", 0.0),
            "precip_24h_mm": pred.get("precip_24h_mm", 0.0),
            "soil_moisture": pred.get("soil_moisture", 0.0),
            "ndvi": pred.get("ndvi", 0.0),
            "snow_cover": pred.get("snow_cover", False),
        },
    }


@router.get("/status")
async def get_status():
    """Pipeline health and system status."""
    return StatusResponse(
        status="operational" if _predictions_cache else "initializing",
        version=settings.app_version,
        last_pipeline_run=_last_pipeline_run,
        model_version="fire_core_v1",
        grid_cells=len(_grid_cells),
        pipeline_healthy=bool(_predictions_cache),
    )


@router.get("/coverage")
async def get_coverage():
    """BC coverage boundary and grid metadata."""
    return {
        "province": "British Columbia",
        "crs": "EPSG:4326",
        "grid": {
            "resolution_km": settings.grid_resolution_km,
            "total_cells": len(_grid_cells),
            "lat_range": [settings.bc_bbox_south, settings.bc_bbox_north],
            "lon_range": [settings.bc_bbox_west, settings.bc_bbox_east],
        },
        "bec_zones_count": 14,
        "fuel_types_count": 16,
    }


@router.get("/risk/grid")
async def get_risk_grid(
    bbox: str = Query(
        ...,
        description="Bounding box: south,west,north,east",
        examples=["49.0,-123.5,50.0,-122.0"],
    ),
    level: Optional[str] = Query(None, description="Filter by danger level"),
):
    """Area risk query. Returns GeoJSON FeatureCollection for cells in bbox."""
    if not _predictions_cache:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")

    try:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        south, west, north, east = parts
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422, detail="bbox must be 4 comma-separated floats: south,west,north,east"
        )

    features = []
    for cell_id, cell in _grid_cells.items():
        lat, lon = cell["lat"], cell["lon"]
        if not (south <= lat <= north and west <= lon <= east):
            continue

        pred = _predictions_cache.get(cell_id)
        if pred is None:
            continue

        cell_level = pred.get("level", "")
        if level and cell_level != level.upper():
            continue

        # Build GeoJSON feature with approximate cell polygon
        half_lat = settings.grid_resolution_km * 0.0045  # ~0.0225 deg at 5km
        half_lon = settings.grid_resolution_km * 0.006  # ~0.03 deg (wider at BC latitudes)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [lon - half_lon, lat - half_lat],
                            [lon + half_lon, lat - half_lat],
                            [lon + half_lon, lat + half_lat],
                            [lon - half_lon, lat + half_lat],
                            [lon - half_lon, lat - half_lat],
                        ]
                    ],
                },
                "properties": {
                    "cell_id": cell_id,
                    "score": pred.get("score", 0.0),
                    "level": cell_level,
                    "bec_zone": cell.get("bec_zone", ""),
                    "fuel_type": cell.get("fuel_type", ""),
                    "fwi": pred.get("fwi", 0.0),
                    "temperature_c": pred.get("temperature_c", 0.0),
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "bbox": [south, west, north, east],
            "cell_count": len(features),
            "timestamp": _last_pipeline_run,
        },
    }


@router.get("/risk/heatmap")
async def get_risk_heatmap(
    bbox: str = Query(
        ...,
        description="Bounding box: south,west,north,east",
        examples=["49.0,-123.5,50.0,-122.0"],
    ),
    width: int = Query(256, ge=64, le=2048, description="Image width in pixels"),
    height: int = Query(256, ge=64, le=2048, description="Image height in pixels"),
    colormap: str = Query("risk", description="Color map: risk, grayscale"),
):
    """Returns a PNG heatmap image of fire risk scores for the given bounding box."""
    if not _predictions_cache:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")

    try:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        south, west, north, east = parts
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422, detail="bbox must be 4 comma-separated floats: south,west,north,east"
        )

    import numpy as np

    # Build raster from grid cell predictions
    raster = np.full((height, width), np.nan, dtype=np.float32)
    lat_step = (north - south) / height
    lon_step = (east - west) / width

    for cell_id, cell in _grid_cells.items():
        lat, lon = cell["lat"], cell["lon"]
        if not (south <= lat <= north and west <= lon <= east):
            continue

        pred = _predictions_cache.get(cell_id)
        if pred is None:
            continue

        row = min(int((north - lat) / lat_step), height - 1)
        col = min(int((lon - west) / lon_step), width - 1)
        raster[row, col] = pred.get("score", 0.0)

    # Interpolate sparse grid to fill pixels
    from scipy.ndimage import uniform_filter

    mask = ~np.isnan(raster)
    if mask.any():
        filled = np.where(mask, raster, 0.0)
        weights = mask.astype(np.float32)
        # Smooth with kernel roughly matching grid resolution
        kernel = max(3, int(min(width, height) / 20))
        smoothed = uniform_filter(filled, size=kernel)
        weight_smoothed = uniform_filter(weights, size=kernel)
        with np.errstate(invalid="ignore", divide="ignore"):
            raster = np.where(weight_smoothed > 0, smoothed / weight_smoothed, 0.0)
        raster = np.clip(raster, 0.0, 1.0)
    else:
        raster = np.zeros((height, width), dtype=np.float32)

    # Convert to RGBA PNG
    rgba = _score_to_rgba(raster, colormap)

    # Encode as PNG
    from PIL import Image

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Bbox": f"{south},{west},{north},{east}",
            "X-Timestamp": _last_pipeline_run or "",
        },
    )


def _score_to_rgba(scores, colormap: str = "risk"):
    """Convert a 2D array of fire risk scores [0,1] to RGBA uint8 array."""
    import numpy as np

    h, w = scores.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if colormap == "grayscale":
        gray = (scores * 255).astype(np.uint8)
        rgba[:, :, 0] = gray
        rgba[:, :, 1] = gray
        rgba[:, :, 2] = gray
        rgba[:, :, 3] = 255
        return rgba

    # Risk colormap matching DangerLevel colors
    # VERY_LOW (0-0.05): green, LOW (0.05-0.15): blue,
    # MODERATE (0.15-0.35): yellow, HIGH (0.35-0.60): orange,
    # VERY_HIGH (0.60-0.80): red, EXTREME (0.80-1.0): dark red
    thresholds = [0.05, 0.15, 0.35, 0.60, 0.80]
    colors = [
        (34, 197, 94),  # green
        (59, 130, 246),  # blue
        (234, 179, 8),  # yellow
        (249, 115, 22),  # orange
        (239, 68, 68),  # red
        (180, 20, 20),  # dark red
    ]

    for y in range(h):
        for x in range(w):
            s = scores[y, x]
            idx = 0
            for t in thresholds:
                if s > t:
                    idx += 1
            rgba[y, x, :3] = colors[idx]
            # Alpha: transparent for very low values, opaque for higher
            rgba[y, x, 3] = min(255, int(50 + s * 205))

    return rgba


@router.get("/history/{lat}/{lon}")
async def get_fire_history(
    lat: float,
    lon: float,
    years: int = Query(5, ge=1, le=50, description="Years of history to include"),
    radius_km: float = Query(25.0, ge=1, le=100, description="Search radius in km"),
):
    """Historical fire events near a location."""
    _validate_bc_coords(lat, lon)

    try:
        from sqlalchemy import func

        from infernis.db.engine import SessionLocal
        from infernis.db.fire_history import FireHistoryDB

        db = SessionLocal()
        try:
            current_year = datetime.now().year
            min_year = current_year - years

            # ST_DWithin uses meters for geography type, convert km
            radius_m = radius_km * 1000
            query_point = func.ST_SetSRID(func.ST_MakePoint(lon, lat), 4326)

            fires = (
                db.query(FireHistoryDB)
                .filter(
                    FireHistoryDB.year >= min_year,
                    func.ST_DWithin(
                        func.ST_Geography(FireHistoryDB.geom),
                        func.ST_Geography(query_point),
                        radius_m,
                    ),
                )
                .order_by(FireHistoryDB.year.desc(), FireHistoryDB.size_ha.desc())
                .limit(100)
                .all()
            )

            results = []
            for f in fires:
                distance = db.scalar(
                    func.ST_Distance(
                        func.ST_Geography(f.geom),
                        func.ST_Geography(query_point),
                    )
                )
                results.append(
                    {
                        "fire_id": f.fire_id,
                        "fire_name": f.fire_name,
                        "year": f.year,
                        "start_date": f.start_date.isoformat() if f.start_date else None,
                        "end_date": f.end_date.isoformat() if f.end_date else None,
                        "cause": f.cause,
                        "size_ha": f.size_ha,
                        "distance_km": round(distance / 1000, 1) if distance else None,
                        "lat": f.lat,
                        "lon": f.lon,
                        "source": f.source,
                    }
                )

            return {
                "location": {"lat": lat, "lon": lon},
                "search_radius_km": radius_km,
                "years_back": years,
                "fires": results,
                "total_fires": len(results),
            }
        finally:
            db.close()

    except ImportError:
        raise HTTPException(status_code=503, detail="Database not available")
    except Exception as e:
        logger.error("Fire history query failed: %s", e)
        # Return empty results if DB not set up yet
        return {
            "location": {"lat": lat, "lon": lon},
            "search_radius_km": radius_km,
            "years_back": years,
            "fires": [],
            "total_fires": 0,
        }
