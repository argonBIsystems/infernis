"""INFERNIS API routes - REST endpoints for fire risk data."""

from __future__ import annotations

import io
import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from infernis.config import settings
from infernis.models.enums import DangerLevel
from infernis.models.schemas import (
    ConfidenceInterval,
    FireBehaviour,
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


def _safe_float(val, default=0.0):
    """Sanitize a float value for JSON serialization — replace NaN/Inf with default."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


# Grid cells + KDTree are kept in memory (small: ~8 MB for 84K cells).
# Predictions and forecasts are read from Redis on-demand to save RAM.
_grid_cells: dict = {}
_last_pipeline_run: str | None = None
_kdtree = None
_cell_ids_ordered: list = []
_forecast_base_date: str | None = None

# Thin compatibility layer — endpoints that used _predictions_cache now
# call _get_prediction(cell_id) which reads from Redis.
_predictions_cache: dict = {}  # only populated during pipeline run, cleared after
_forecast_cache: dict[str, list[dict]] = {}  # same


def _get_prediction(cell_id: str) -> dict | None:
    """Fetch a single cell's prediction from Redis (on-demand, no preload)."""
    try:
        from infernis.services.cache import get_redis

        r = get_redis()
        if r is None:
            return _predictions_cache.get(cell_id)
        val = r.get(f"pred:latest:{cell_id}")
        if val:
            return json.loads(val)
        return _predictions_cache.get(cell_id)
    except Exception:
        return _predictions_cache.get(cell_id)


def _get_forecast(cell_id: str) -> list[dict] | None:
    """Fetch a single cell's forecast from Redis (on-demand)."""
    try:
        from infernis.services.cache import get_redis

        r = get_redis()
        if r is None:
            return _forecast_cache.get(cell_id)
        val = r.get(f"forecast:latest:{cell_id}")
        if val:
            return json.loads(val)
        return _forecast_cache.get(cell_id)
    except Exception:
        return _forecast_cache.get(cell_id)


def _has_predictions() -> bool:
    """Check if predictions are available (in cache or Redis)."""
    if _predictions_cache:
        return True
    try:
        from infernis.services.cache import get_redis

        r = get_redis()
        return r is not None and r.exists("pred:last_run")
    except Exception:
        return False


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


@router.get("/risk/{lat}/{lon}", tags=["risk"])
async def get_risk(lat: float, lon: float):
    """Point fire risk query for a BC location.

    Returns the current fire risk score, danger level, FWI components,
    weather conditions, and ecological context for the nearest grid cell.

    **Use cases:**
    - Mobile app showing risk for user's current location
    - Property-level wildfire assessment for insurance underwriting
    - Pre-trip safety check for hikers and campers
    - Integration into weather apps and outdoor recreation tools

    **Example:**
    ```
    GET /v1/risk/50.67/-120.33
    ```
    Returns risk score, 6 FWI components, weather conditions, BEC zone, and 24h change.
    """
    _validate_bc_coords(lat, lon)

    cell_id = _find_nearest_cell(lat, lon)
    if cell_id is None:
        raise HTTPException(
            status_code=503, detail="Predictions not yet available. Pipeline may be initializing."
        )

    pred = _get_prediction(cell_id)
    if pred is None:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")
    cell = _grid_cells.get(cell_id, {})

    score = _safe_float(pred.get("score"), 0.0)
    level = DangerLevel.from_score(score)

    # Compute 24h change from DB
    change_24h = None
    try:
        from datetime import date, timedelta

        from infernis.db.engine import SessionLocal
        from infernis.db.tables import PredictionDB

        yesterday = date.today() - timedelta(days=1)
        db = SessionLocal()
        try:
            prev = (
                db.query(PredictionDB.score)
                .filter(
                    PredictionDB.cell_id == cell_id,
                    PredictionDB.prediction_date == yesterday,
                )
                .first()
            )
            if prev:
                change_24h = round(score - prev.score, 4)
        finally:
            db.close()
    except Exception:
        pass

    # Build confidence interval if available in prediction cache
    ci_raw = pred.get("confidence_interval")
    confidence_interval = None
    if ci_raw is not None:
        try:
            confidence_interval = ConfidenceInterval(
                lower=ci_raw["lower"],
                upper=ci_raw["upper"],
                level=ci_raw.get("level", 0.90),
            )
        except Exception:
            confidence_interval = None

    # Build FireBehaviour from prediction cache if available
    fb_raw = pred.get("fire_behaviour")
    fire_behaviour = None
    if fb_raw is not None:
        try:
            fire_behaviour = FireBehaviour(**fb_raw)
        except Exception:
            fire_behaviour = None

    return RiskResponse(
        location={"lat": lat, "lon": lon},
        grid_cell_id=cell_id,
        timestamp=pred.get("timestamp", datetime.now(timezone.utc).isoformat()),
        risk=RiskScore(score=score, level=level),
        fwi=FWIComponents(
            ffmc=_safe_float(pred.get("ffmc")),
            dmc=_safe_float(pred.get("dmc")),
            dc=_safe_float(pred.get("dc")),
            isi=_safe_float(pred.get("isi")),
            bui=_safe_float(pred.get("bui")),
            fwi=_safe_float(pred.get("fwi")),
        ),
        conditions=WeatherConditions(
            temperature_c=_safe_float(pred.get("temperature_c")),
            rh_pct=_safe_float(pred.get("rh_pct")),
            wind_kmh=_safe_float(pred.get("wind_kmh")),
            precip_24h_mm=_safe_float(pred.get("precip_24h_mm")),
            soil_moisture=_safe_float(pred.get("soil_moisture")),
            ndvi=_safe_float(pred.get("ndvi")),
            snow_cover=pred.get("snow_cover", False),
            c_haines=_safe_float(pred.get("c_haines"), default=None),
        ),
        context={
            "bec_zone": cell.get("bec_zone", ""),
            "fuel_type": cell.get("fuel_type", ""),
            "elevation_m": cell.get("elevation_m", 0),
        },
        next_update=pred.get("next_update", ""),
        change_24h=change_24h,
        confidence_interval=confidence_interval,
        fire_behaviour=fire_behaviour,
    )


@router.get("/forecast/{lat}/{lon}", tags=["forecast"])
async def get_forecast(
    lat: float,
    lon: float,
    days: int = Query(default=10, ge=1, le=10, description="Number of forecast days"),
):
    """Multi-day fire risk forecast for a location (up to 10 days).

    Uses real NWP weather forecasts from ECCC's GEM model (HRDPS 2.5km for
    days 1-2, GDPS for days 3-10) with FWI roll-forward and confidence decay.

    **Use cases:**
    - "Is Saturday safe for a campfire?" — check the 5-day outlook
    - Fire crew pre-positioning based on predicted risk trajectory
    - Utility transmission line de-energization planning
    - Event planning and outdoor activity scheduling

    **Example:**
    ```
    GET /v1/forecast/50.67/-120.33?days=7
    ```
    Returns daily risk scores with weather data (temp, RH, wind, precip),
    FWI components, confidence level, and data source per day.
    """
    _validate_bc_coords(lat, lon)

    cell_id = _find_nearest_cell(lat, lon)
    if cell_id is None:
        raise HTTPException(status_code=404, detail="No forecast data for this location.")

    fc = _get_forecast(cell_id)
    if not fc:
        raise HTTPException(
            status_code=503, detail="Forecast not yet available. Pipeline may be initializing."
        )

    forecast_days = fc[:days]

    return ForecastResponse(
        latitude=lat,
        longitude=lon,
        cell_id=cell_id,
        base_date=_forecast_base_date or "",
        forecast=[
            ForecastDay(
                valid_date=d["valid_date"],
                lead_day=d["lead_day"],
                risk_score=_safe_float(d.get("risk_score")),
                danger_level=d["danger_level"],
                danger_label=d["danger_label"],
                confidence=_safe_float(d.get("confidence"), 0.0),
                fwi=FWIComponents(
                    ffmc=_safe_float(d.get("fwi", {}).get("ffmc")),
                    dmc=_safe_float(d.get("fwi", {}).get("dmc")),
                    dc=_safe_float(d.get("fwi", {}).get("dc")),
                    isi=_safe_float(d.get("fwi", {}).get("isi")),
                    bui=_safe_float(d.get("fwi", {}).get("bui")),
                    fwi=_safe_float(d.get("fwi", {}).get("fwi")),
                ),
                data_source=d.get("data_source", ""),
                temperature_c=_safe_float(d.get("temperature_c"), default=None),
                rh_pct=_safe_float(d.get("rh_pct"), default=None),
                wind_kmh=_safe_float(d.get("wind_kmh"), default=None),
                precip_24h_mm=_safe_float(d.get("precip_24h_mm"), default=None),
            )
            for d in forecast_days
        ],
        generated_at=_last_pipeline_run or "",
    )


@router.get("/risk/zones", tags=["risk"])
async def get_risk_zones():
    """Risk summary aggregated by BEC zone across all of BC.

    Returns average and maximum risk scores for each of BC's biogeoclimatic
    zones, with cell counts and number of high-risk cells.

    **Use cases:**
    - Province-wide risk dashboard for government briefings
    - News media fire risk summary
    - Identifying which ecological zones are most at risk today
    """
    if not _has_predictions():
        raise HTTPException(status_code=503, detail="Predictions not yet available.")

    # Aggregate by BEC zone — fetch scores from Redis in batches
    from infernis.services.cache import get_redis

    r = get_redis()
    zones: dict = {}

    # Batch fetch all predictions from Redis using pipeline
    cell_ids_list = list(_grid_cells.keys())
    all_preds = {}
    if r:
        BATCH = 5000
        for i in range(0, len(cell_ids_list), BATCH):
            batch_ids = cell_ids_list[i : i + BATCH]
            pipe = r.pipeline()
            for cid in batch_ids:
                pipe.get(f"pred:latest:{cid}")
            values = pipe.execute()
            for cid, val in zip(batch_ids, values):
                if val:
                    all_preds[cid] = json.loads(val)

    for cell_id, pred in all_preds.items():
        cell = _grid_cells.get(cell_id, {})
        zone = cell.get("bec_zone", "UNKNOWN")
        if zone not in zones:
            zones[zone] = {"scores": [], "cells": 0, "high_risk": 0}
        score = _safe_float(pred.get("score"), 0.0)
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


@router.get("/fwi/{lat}/{lon}", tags=["risk"])
async def get_fwi(lat: float, lon: float):
    """Raw FWI components for a location."""
    _validate_bc_coords(lat, lon)

    cell_id = _find_nearest_cell(lat, lon)
    if cell_id is None:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")

    pred = _get_prediction(cell_id)
    if pred is None:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")
    return {
        "location": {"lat": lat, "lon": lon},
        "grid_cell_id": cell_id,
        "timestamp": pred.get("timestamp", ""),
        "fwi": {
            "ffmc": _safe_float(pred.get("ffmc")),
            "dmc": _safe_float(pred.get("dmc")),
            "dc": _safe_float(pred.get("dc")),
            "isi": _safe_float(pred.get("isi")),
            "bui": _safe_float(pred.get("bui")),
            "fwi": _safe_float(pred.get("fwi")),
        },
    }


@router.get("/conditions/{lat}/{lon}", tags=["risk"])
async def get_conditions(lat: float, lon: float):
    """Current weather and environmental conditions."""
    _validate_bc_coords(lat, lon)

    cell_id = _find_nearest_cell(lat, lon)
    if cell_id is None:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")

    pred = _get_prediction(cell_id)
    if pred is None:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")
    return {
        "location": {"lat": lat, "lon": lon},
        "grid_cell_id": cell_id,
        "timestamp": pred.get("timestamp", ""),
        "conditions": {
            "temperature_c": _safe_float(pred.get("temperature_c")),
            "rh_pct": _safe_float(pred.get("rh_pct")),
            "wind_kmh": _safe_float(pred.get("wind_kmh")),
            "precip_24h_mm": _safe_float(pred.get("precip_24h_mm")),
            "soil_moisture": _safe_float(pred.get("soil_moisture")),
            "ndvi": _safe_float(pred.get("ndvi")),
            "snow_cover": pred.get("snow_cover", False),
            "c_haines": _safe_float(pred.get("c_haines"), default=None),
        },
    }


@router.get("/status", tags=["system"])
async def get_status():
    """Pipeline health and system status.

    Returns operational status, last pipeline run time, grid cell count,
    and model version. Use this for monitoring and uptime checks.

    **Use cases:**
    - Monitoring dashboard health check
    - Automated uptime monitoring (e.g., Uptime Robot, Pingdom)
    - Verifying the pipeline ran today before using predictions
    """
    # Check if predictions are available (in Redis, not in-memory cache)
    has_preds = _has_predictions()
    return StatusResponse(
        status="operational" if has_preds else "initializing",
        version=settings.app_version,
        last_pipeline_run=_last_pipeline_run,
        model_version="fire_core_v1",
        grid_cells=len(_grid_cells),
        pipeline_healthy=has_preds,
    )


@router.get("/coverage", tags=["system"])
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


@router.get("/risk/grid", tags=["risk"])
async def get_risk_grid(
    bbox: str = Query(
        ...,
        description="Bounding box: south,west,north,east",
        examples=["49.0,-123.5,50.0,-122.0"],
    ),
    level: Optional[str] = Query(None, description="Filter by danger level"),
):
    """Area risk query returning GeoJSON FeatureCollection.

    Returns risk data for all grid cells within a bounding box as GeoJSON,
    ready for direct rendering on a map. Each feature includes score, level,
    color, BEC zone, fuel type, FWI, and temperature.

    **Use cases:**
    - Colored risk overlay on a web map (Leaflet, Mapbox)
    - Municipality-wide risk assessment
    - Identifying the highest-risk cells in an area

    **Bbox format:** `south,west,north,east` (e.g., `49.0,-120.5,50.5,-119.0`)

    **Example:**
    ```
    GET /v1/risk/grid?bbox=49.0,-120.5,50.5,-119.0
    ```
    """
    if not _has_predictions():
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

        pred = _get_prediction(cell_id)
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
                    "score": _safe_float(pred.get("score")),
                    "level": cell_level,
                    "bec_zone": cell.get("bec_zone", ""),
                    "fuel_type": cell.get("fuel_type", ""),
                    "fwi": _safe_float(pred.get("fwi")),
                    "temperature_c": _safe_float(pred.get("temperature_c")),
                    "color": DangerLevel.from_score(_safe_float(pred.get("score"))).color,
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


@router.get("/risk/heatmap", tags=["risk"])
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
    if not _has_predictions():
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

        pred = _get_prediction(cell_id)
        if pred is None:
            continue

        row = min(int((north - lat) / lat_step), height - 1)
        col = min(int((lon - west) / lon_step), width - 1)
        raster[row, col] = _safe_float(pred.get("score"))

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


# ---------------------------------------------------------------------------
# Demo / test endpoints — return mock data so devs can explore response shapes
# without an API key.  No rate limit, no auth.
# ---------------------------------------------------------------------------

_DEMO_SAMPLES = [
    {
        "name": "very_low",
        "location": {"lat": 49.70, "lon": -123.16},
        "cell_id": "DEMO-VERY-LOW",
        "risk": {"score": 0.02, "level": "VERY_LOW"},
        "fwi": {"ffmc": 72.0, "dmc": 8.0, "dc": 15.0, "isi": 0.8, "bui": 8.0, "fwi": 0.5},
        "conditions": {
            "temperature_c": 4.0,
            "rh_pct": 92.0,
            "wind_kmh": 6.0,
            "precip_24h_mm": 12.0,
            "soil_moisture": 0.55,
            "ndvi": 0.7,
            "snow_cover": True,
        },
        "context": {"bec_zone": "CWH", "fuel_type": "C7", "elevation_m": 120},
        "description": "Wet coastal forest in winter — minimal fire risk.",
    },
    {
        "name": "low",
        "location": {"lat": 50.27, "lon": -119.27},
        "cell_id": "DEMO-LOW",
        "risk": {"score": 0.08, "level": "LOW"},
        "fwi": {"ffmc": 80.0, "dmc": 14.0, "dc": 40.0, "isi": 1.5, "bui": 14.0, "fwi": 2.0},
        "conditions": {
            "temperature_c": 11.0,
            "rh_pct": 65.0,
            "wind_kmh": 10.0,
            "precip_24h_mm": 2.0,
            "soil_moisture": 0.40,
            "ndvi": 0.6,
            "snow_cover": False,
        },
        "context": {"bec_zone": "IDF", "fuel_type": "C3", "elevation_m": 450},
        "description": "Spring in the Okanagan — drying but still low.",
    },
    {
        "name": "moderate",
        "location": {"lat": 50.67, "lon": -120.33},
        "cell_id": "DEMO-MODERATE",
        "risk": {"score": 0.25, "level": "MODERATE"},
        "fwi": {"ffmc": 86.0, "dmc": 35.0, "dc": 120.0, "isi": 4.5, "bui": 40.0, "fwi": 12.0},
        "conditions": {
            "temperature_c": 22.0,
            "rh_pct": 38.0,
            "wind_kmh": 15.0,
            "precip_24h_mm": 0.0,
            "soil_moisture": 0.25,
            "ndvi": 0.5,
            "snow_cover": False,
        },
        "context": {"bec_zone": "IDF", "fuel_type": "C4", "elevation_m": 350},
        "description": "Hot dry spell in Kamloops — watch closely.",
    },
    {
        "name": "high",
        "location": {"lat": 50.23, "lon": -121.58},
        "cell_id": "DEMO-HIGH",
        "risk": {"score": 0.48, "level": "HIGH"},
        "fwi": {"ffmc": 90.0, "dmc": 65.0, "dc": 280.0, "isi": 8.0, "bui": 80.0, "fwi": 24.0},
        "conditions": {
            "temperature_c": 34.0,
            "rh_pct": 18.0,
            "wind_kmh": 22.0,
            "precip_24h_mm": 0.0,
            "soil_moisture": 0.12,
            "ndvi": 0.35,
            "snow_cover": False,
        },
        "context": {"bec_zone": "PP", "fuel_type": "C4", "elevation_m": 230},
        "description": "Lytton-area heatwave — high danger, fire bans likely.",
    },
    {
        "name": "very_high",
        "location": {"lat": 52.13, "lon": -122.14},
        "cell_id": "DEMO-VERY-HIGH",
        "risk": {"score": 0.72, "level": "VERY_HIGH"},
        "fwi": {"ffmc": 93.0, "dmc": 90.0, "dc": 400.0, "isi": 14.0, "bui": 120.0, "fwi": 38.0},
        "conditions": {
            "temperature_c": 36.0,
            "rh_pct": 12.0,
            "wind_kmh": 35.0,
            "precip_24h_mm": 0.0,
            "soil_moisture": 0.08,
            "ndvi": 0.28,
            "snow_cover": False,
        },
        "context": {"bec_zone": "SBS", "fuel_type": "C3", "elevation_m": 680},
        "description": "Williams Lake area — extreme heat, strong wind, bone-dry fuels.",
    },
    {
        "name": "extreme",
        "location": {"lat": 54.02, "lon": -124.00},
        "cell_id": "DEMO-EXTREME",
        "risk": {"score": 0.91, "level": "EXTREME"},
        "fwi": {"ffmc": 96.0, "dmc": 120.0, "dc": 500.0, "isi": 22.0, "bui": 160.0, "fwi": 55.0},
        "conditions": {
            "temperature_c": 38.0,
            "rh_pct": 8.0,
            "wind_kmh": 45.0,
            "precip_24h_mm": 0.0,
            "soil_moisture": 0.05,
            "ndvi": 0.20,
            "snow_cover": False,
        },
        "context": {"bec_zone": "ICH", "fuel_type": "M2", "elevation_m": 700},
        "description": "Worst-case scenario — evacuate. Lightning-ignition imminent.",
    },
]


def _find_nearest_demo(lat: float, lon: float) -> dict:
    """Find the nearest demo test location by Euclidean distance."""
    best = None
    best_dist = float("inf")
    for sample in _DEMO_SAMPLES:
        slat = sample["location"]["lat"]
        slon = sample["location"]["lon"]
        dist = (lat - slat) ** 2 + (lon - slon) ** 2
        if dist < best_dist:
            best_dist = dist
            best = sample
    return best


def _demo_risk_response(sample: dict) -> dict:
    """Build a risk response from a demo sample."""
    s = sample
    return {
        "location": s["location"],
        "grid_cell_id": s["cell_id"],
        "timestamp": "2026-07-15T14:00:00-07:00",
        "risk": {
            "score": s["risk"]["score"],
            "level": s["risk"]["level"],
            "color": DangerLevel.from_score(s["risk"]["score"]).color,
        },
        "fwi": s["fwi"],
        "conditions": s["conditions"],
        "context": s["context"],
        "forecast_horizon": "24h",
        "next_update": "2026-07-16T14:00:00-07:00",
        "_demo": True,
        "_nearest_test_location": s["name"],
        "_description": s["description"],
    }


@router.get("/demo/risk", tags=["demo"])
async def get_demo_risk():
    """Sample risk responses at all six danger levels. No API key required.

    Returns mock data for developer testing and integration — not real predictions.
    """
    return {
        "description": "Sample INFERNIS risk data at all six danger levels. "
        "Use these to test your integration. This is mock data, not live predictions.",
        "danger_levels": ["VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH", "EXTREME"],
        "samples": [
            {
                "location": s["location"],
                "grid_cell_id": s["cell_id"],
                "timestamp": "2026-07-15T14:00:00-07:00",
                "risk": {
                    "score": s["risk"]["score"],
                    "level": s["risk"]["level"],
                    "color": DangerLevel.from_score(s["risk"]["score"]).color,
                },
                "fwi": s["fwi"],
                "conditions": s["conditions"],
                "context": s["context"],
                "forecast_horizon": "24h",
                "next_update": "2026-07-16T14:00:00-07:00",
                "_demo": True,
                "_description": s["description"],
            }
            for s in _DEMO_SAMPLES
        ],
    }


@router.get("/demo/risk/zones", tags=["demo"])
async def get_demo_risk_zones():
    """BEC zone risk summary using demo data. No API key required."""
    zones = {}
    for s in _DEMO_SAMPLES:
        bec = s["context"]["bec_zone"]
        zones[bec] = {
            "bec_zone": bec,
            "avg_risk_score": round(s["risk"]["score"], 3),
            "max_risk_score": round(s["risk"]["score"], 3),
            "level": s["risk"]["level"],
            "cell_count": 1000,
            "high_risk_cells": 50 if s["risk"]["score"] >= 0.6 else 0,
        }
    return {
        "zones": list(zones.values()),
        "timestamp": "2026-07-15T14:00:00-07:00",
        "_demo": True,
    }


@router.get("/demo/risk/{level}", tags=["demo"])
async def get_demo_risk_by_level(level: str):
    """Single sample risk response for a specific danger level. No API key required.

    Valid levels: very_low, low, moderate, high, very_high, extreme
    """
    sample = next((s for s in _DEMO_SAMPLES if s["name"] == level.lower()), None)
    if sample is None:
        valid = [s["name"] for s in _DEMO_SAMPLES]
        raise HTTPException(
            status_code=404,
            detail=f"Unknown level '{level}'. Valid levels: {', '.join(valid)}",
        )
    s = sample
    return {
        "location": s["location"],
        "grid_cell_id": s["cell_id"],
        "timestamp": "2026-07-15T14:00:00-07:00",
        "risk": {
            "score": s["risk"]["score"],
            "level": s["risk"]["level"],
            "color": DangerLevel.from_score(s["risk"]["score"]).color,
        },
        "fwi": s["fwi"],
        "conditions": s["conditions"],
        "context": s["context"],
        "forecast_horizon": "24h",
        "next_update": "2026-07-16T14:00:00-07:00",
        "_demo": True,
        "_description": s["description"],
    }


@router.get("/demo/risk/{lat}/{lon}", tags=["demo"])
async def get_demo_risk_by_coords(lat: float, lon: float):
    """Point risk query using demo data. Snaps to nearest test location. No API key required."""
    sample = _find_nearest_demo(lat, lon)
    return _demo_risk_response(sample)


@router.get("/demo/forecast", tags=["demo"])
async def get_demo_forecast():
    """Sample 10-day forecast showing risk escalation. No API key required."""
    from datetime import date, timedelta

    base = date(2026, 7, 15)
    # Simulate a drying trend with increasing risk
    day_profiles = [
        {
            "score": 0.08,
            "dl": 2,
            "label": "LOW",
            "conf": 0.95,
            "src": "HRDPS",
            "fwi": {"ffmc": 82.0, "dmc": 20.0, "dc": 80.0, "isi": 2.5, "bui": 22.0, "fwi": 4.0},
        },
        {
            "score": 0.14,
            "dl": 2,
            "label": "LOW",
            "conf": 0.90,
            "src": "HRDPS",
            "fwi": {"ffmc": 85.0, "dmc": 28.0, "dc": 95.0, "isi": 3.8, "bui": 30.0, "fwi": 8.0},
        },
        {
            "score": 0.22,
            "dl": 3,
            "label": "MODERATE",
            "conf": 0.86,
            "src": "GDPS",
            "fwi": {"ffmc": 87.0, "dmc": 38.0, "dc": 115.0, "isi": 5.0, "bui": 42.0, "fwi": 13.0},
        },
        {
            "score": 0.31,
            "dl": 3,
            "label": "MODERATE",
            "conf": 0.81,
            "src": "GDPS",
            "fwi": {"ffmc": 88.5, "dmc": 48.0, "dc": 140.0, "isi": 6.2, "bui": 55.0, "fwi": 17.0},
        },
        {
            "score": 0.42,
            "dl": 4,
            "label": "HIGH",
            "conf": 0.77,
            "src": "GDPS",
            "fwi": {"ffmc": 90.0, "dmc": 58.0, "dc": 170.0, "isi": 7.5, "bui": 68.0, "fwi": 22.0},
        },
        {
            "score": 0.55,
            "dl": 4,
            "label": "HIGH",
            "conf": 0.74,
            "src": "GDPS",
            "fwi": {"ffmc": 91.0, "dmc": 65.0, "dc": 200.0, "isi": 8.8, "bui": 78.0, "fwi": 26.0},
        },
        {
            "score": 0.62,
            "dl": 5,
            "label": "VERY_HIGH",
            "conf": 0.70,
            "src": "GDPS",
            "fwi": {"ffmc": 92.0, "dmc": 72.0, "dc": 235.0, "isi": 10.5, "bui": 88.0, "fwi": 31.0},
        },
        {
            "score": 0.58,
            "dl": 4,
            "label": "HIGH",
            "conf": 0.66,
            "src": "GDPS",
            "fwi": {"ffmc": 89.0, "dmc": 70.0, "dc": 250.0, "isi": 7.0, "bui": 85.0, "fwi": 24.0},
        },
        {
            "score": 0.45,
            "dl": 4,
            "label": "HIGH",
            "conf": 0.63,
            "src": "GDPS",
            "fwi": {"ffmc": 86.0, "dmc": 62.0, "dc": 240.0, "isi": 5.5, "bui": 72.0, "fwi": 18.0},
        },
        {
            "score": 0.30,
            "dl": 3,
            "label": "MODERATE",
            "conf": 0.60,
            "src": "GDPS",
            "fwi": {"ffmc": 82.0, "dmc": 55.0, "dc": 225.0, "isi": 3.5, "bui": 60.0, "fwi": 12.0},
        },
    ]
    return {
        "latitude": 50.67,
        "longitude": -120.33,
        "cell_id": "DEMO-FORECAST",
        "base_date": str(base),
        "forecast": [
            {
                "valid_date": str(base + timedelta(days=i + 1)),
                "lead_day": i + 1,
                "risk_score": dp["score"],
                "danger_level": dp["dl"],
                "danger_label": dp["label"],
                "confidence": dp["conf"],
                "fwi": dp["fwi"],
                "data_source": dp["src"],
            }
            for i, dp in enumerate(day_profiles)
        ],
        "generated_at": "2026-07-15T14:00:00-07:00",
        "_demo": True,
        "_description": "Simulated 10-day drying event near Kamloops. "
        "Risk ramps from LOW to VERY_HIGH (day 7), then eases as a front moves in.",
    }


@router.get("/demo/forecast/{lat}/{lon}", tags=["demo"])
async def get_demo_forecast_by_coords(lat: float, lon: float):
    """Multi-day forecast using demo data. Snaps to nearest test location. No API key required."""
    import math
    from datetime import date, timedelta

    sample = _find_nearest_demo(lat, lon)
    base = date(2026, 7, 15)
    base_score = sample["risk"]["score"]

    days = []
    for i in range(1, 11):
        variation = 0.05 * math.sin(i * 0.7)
        score = max(0.0, min(1.0, base_score + variation * (1 + i * 0.1)))
        level = DangerLevel.from_score(score)
        dl_num = list(DangerLevel).index(level) + 1
        conf = round(0.95**i, 4)
        days.append(
            {
                "valid_date": (base + timedelta(days=i)).isoformat(),
                "lead_day": i,
                "risk_score": round(score, 4),
                "danger_level": dl_num,
                "danger_label": level.value,
                "confidence": conf,
                "fwi": sample["fwi"],
                "data_source": "HRDPS" if i <= 2 else "GDPS",
            }
        )

    return {
        "latitude": sample["location"]["lat"],
        "longitude": sample["location"]["lon"],
        "cell_id": sample["cell_id"],
        "base_date": base.isoformat(),
        "forecast": days,
        "generated_at": "2026-07-15T14:00:00-07:00",
        "_demo": True,
        "_nearest_test_location": sample["name"],
    }


@router.get("/demo/fwi/{lat}/{lon}", tags=["demo"])
async def get_demo_fwi_by_coords(lat: float, lon: float):
    """Raw FWI components using demo data. Snaps to nearest test location. No API key required."""
    sample = _find_nearest_demo(lat, lon)
    return {
        "location": sample["location"],
        "grid_cell_id": sample["cell_id"],
        "timestamp": "2026-07-15T14:00:00-07:00",
        "fwi": sample["fwi"],
        "_demo": True,
        "_nearest_test_location": sample["name"],
    }


@router.get("/demo/conditions/{lat}/{lon}", tags=["demo"])
async def get_demo_conditions_by_coords(lat: float, lon: float):
    """Weather conditions using demo data. Snaps to nearest test location. No API key required."""
    sample = _find_nearest_demo(lat, lon)
    return {
        "location": sample["location"],
        "grid_cell_id": sample["cell_id"],
        "timestamp": "2026-07-15T14:00:00-07:00",
        "conditions": sample["conditions"],
        "_demo": True,
        "_nearest_test_location": sample["name"],
    }


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
