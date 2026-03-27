"""Location fire risk profile endpoint.

GET /v1/risk/profile/{lat}/{lon}
    Returns the full fire risk profile for a BC location: current conditions,
    historical exposure tiers (10yr, 30yr, all-time), susceptibility percentile,
    BEC zone context, and a composite risk rating.

This router MUST be registered BEFORE the main router in main.py to prevent the
`{lat}/{lon}` wildcard in routes.py from shadowing this `profile/{lat}/{lon}` path.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

import infernis.api.routes as _routes_module
from infernis.api.routes import _find_nearest_cell, _safe_float, _validate_bc_coords
from infernis.config import settings
from infernis.models.enums import BEC_ZONE_NAMES, DangerLevel

logger = logging.getLogger(__name__)

profile_router = APIRouter(prefix=settings.api_prefix, tags=["risk"])

# ---------------------------------------------------------------------------
# Module-level fire stats cache (populated from Redis at startup or by pipeline)
# ---------------------------------------------------------------------------

_fire_stats_cache: dict = {}


def set_fire_stats_cache(stats: dict) -> None:
    """Called at startup (or by pipeline) to populate the fire stats cache."""
    global _fire_stats_cache
    _fire_stats_cache = stats
    logger.info("Fire stats cache populated with %d cells", len(stats))


def _get_fire_stats(cell_id: str) -> dict | None:
    """Get fire stats from memory cache or Redis on-demand."""
    if _fire_stats_cache:
        return _fire_stats_cache.get(cell_id)
    try:
        import json

        from infernis.services.cache import get_redis

        r = get_redis()
        if r is None:
            return None
        val = r.get(f"fire_stats:{cell_id}")
        if val:
            return json.loads(val)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@profile_router.get("/risk/profile/{lat}/{lon}", tags=["risk"])
async def get_risk_profile(lat: float, lon: float):
    """Full fire risk profile for a BC location.

    Returns current conditions, historical fire exposure (10yr / 30yr / all-time),
    fire susceptibility percentile, BEC zone context, and a composite risk rating
    that blends static susceptibility, historical exposure, and today's live score.

    **Composite score formula:**
    ```
    composite = 0.3 × susceptibility_percentile/100
              + 0.3 × exposure_percentile/100
              + 0.4 × current_score
    ```
    Clamped to [0, 1].

    **Use cases:**
    - Property-level wildfire risk disclosure for real-estate transactions
    - Insurance underwriting — one-call summary combining static + live risk
    - Community risk education portals

    **Example:**
    ```
    GET /v1/risk/profile/50.67/-120.33
    ```
    """
    _validate_bc_coords(lat, lon)

    # Require predictions to be available
    if not _routes_module._has_predictions():
        raise HTTPException(
            status_code=503,
            detail="Predictions not yet available. Pipeline may be initializing.",
        )

    # Fire stats are read on-demand from Redis (no preload needed)

    cell_id = _find_nearest_cell(lat, lon)
    if cell_id is None:
        raise HTTPException(status_code=404, detail="No grid cell found for this location.")

    if not _routes_module._get_prediction(cell_id):
        raise HTTPException(
            status_code=503,
            detail="Predictions not yet available. Pipeline may be initializing.",
        )

    stats = _get_fire_stats(cell_id)
    if stats is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No fire statistics found for cell {cell_id}. "
                "Run: python -m infernis.admin compute_fire_stats"
            ),
        )

    pred = _routes_module._get_prediction(cell_id)
    cell = _routes_module._grid_cells.get(cell_id, {})

    # Current risk
    current_score = _safe_float(pred.get("score"), 0.0)
    danger_level = DangerLevel.from_score(current_score)

    # Susceptibility & exposure percentiles
    susceptibility_percentile = stats.get("susceptibility_percentile", 0)
    exposure_percentile = stats.get("exposure_percentile", 0)

    # Composite score: 30% susceptibility + 30% exposure + 40% current live score
    composite = (
        0.3 * susceptibility_percentile / 100.0
        + 0.3 * exposure_percentile / 100.0
        + 0.4 * current_score
    )
    composite = min(max(composite, 0.0), 1.0)
    composite_level = DangerLevel.from_score(composite)

    # BEC zone context
    bec_zone = cell.get("bec_zone", "")
    bec_zone_name = BEC_ZONE_NAMES.get(bec_zone, bec_zone)

    # Forecast (optional — don't fail if absent)
    forecast_snippet = None
    if _routes_module._forecast_cache:
        forecast_days = _routes_module._forecast_cache.get(cell_id, [])
        if forecast_days:
            # Return first 3 forecast days as a lightweight snippet
            forecast_snippet = [
                {
                    "valid_date": d.get("valid_date"),
                    "lead_day": d.get("lead_day"),
                    "risk_score": _safe_float(d.get("risk_score")),
                    "danger_label": d.get("danger_label"),
                }
                for d in forecast_days[:3]
            ]

    # Historical exposure tiers
    fires_10yr = stats.get("fires_10yr", {})
    fires_30yr = stats.get("fires_30yr", {})
    fires_all = stats.get("fires_all", {})

    return {
        "cell_id": cell_id,
        "location": {"lat": lat, "lon": lon},
        "timestamp": pred.get("timestamp", datetime.now(timezone.utc).isoformat()),
        # --- Context ---
        "context": {
            "bec_zone": bec_zone,
            "bec_zone_name": bec_zone_name,
            "fuel_type": cell.get("fuel_type", ""),
            "elevation_m": cell.get("elevation_m", 0),
        },
        # --- Current conditions ---
        "current": {
            "score": current_score,
            "level": danger_level.value,
            "color": danger_level.color,
            "fwi": _safe_float(pred.get("fwi")),
            "temperature_c": _safe_float(pred.get("temperature_c")),
            "rh_pct": _safe_float(pred.get("rh_pct")),
            "wind_kmh": _safe_float(pred.get("wind_kmh")),
            "precip_24h_mm": _safe_float(pred.get("precip_24h_mm")),
        },
        # --- Historical fire exposure ---
        "historical_exposure": {
            "radius_km": 10,
            "fires_10yr": {
                "count": fires_10yr.get("count", 0),
                "nearest_km": fires_10yr.get("nearest_km"),
                "largest_ha": fires_10yr.get("largest_ha"),
                "dominant_causes": fires_10yr.get("causes", {}),
            },
            "fires_30yr": {
                "count": fires_30yr.get("count", 0),
                "nearest_km": fires_30yr.get("nearest_km"),
                "largest_ha": fires_30yr.get("largest_ha"),
                "dominant_causes": fires_30yr.get("causes", {}),
            },
            "fires_all_time": {
                "count": fires_all.get("count", 0),
                "nearest_km": fires_all.get("nearest_km"),
                "largest_ha": fires_all.get("largest_ha"),
                "dominant_causes": fires_all.get("causes", {}),
                "record_start": fires_all.get("record_start"),
            },
        },
        # --- Susceptibility ---
        "susceptibility": {
            "score": _safe_float(stats.get("susceptibility_score")),
            "percentile": susceptibility_percentile,
            "label": stats.get("susceptibility_label", ""),
            "basis": stats.get("susceptibility_basis", ""),
        },
        # --- Fire regime (BEC zone) ---
        "fire_regime": {
            "mean_return_years": stats.get("mean_return_years"),
            "typical_severity": stats.get("typical_severity"),
            "dominant_cause": stats.get("dominant_cause"),
        },
        # --- Composite risk rating ---
        "composite_risk_rating": {
            "score": round(composite, 4),
            "level": composite_level.value,
            "color": composite_level.color,
            "components": {
                "susceptibility_weight": 0.3,
                "exposure_weight": 0.3,
                "current_score_weight": 0.4,
                "susceptibility_percentile": susceptibility_percentile,
                "exposure_percentile": exposure_percentile,
                "current_score": current_score,
            },
        },
        # --- Seasonal risk curve (monthly fire frequency for this BEC zone) ---
        "seasonal_risk": _get_seasonal_risk(bec_zone),
        # --- Short forecast snippet (optional) ---
        "forecast_snippet": forecast_snippet,
    }


# Pre-computed seasonal risk curves by BEC zone.
# Based on monthly fire frequency from 225K BC fire records (1919-2025).
# Values represent relative fire occurrence rate per month (0-1 scale,
# normalized so peak month = 1.0).
_SEASONAL_CURVES = {
    "BWBS": {"label": "Boreal White and Black Spruce",
             "fire_season": "May-Sep", "peak_month": "Jul",
             "monthly": {"Jan": 0.0, "Feb": 0.0, "Mar": 0.01, "Apr": 0.08, "May": 0.35,
                         "Jun": 0.65, "Jul": 1.0, "Aug": 0.85, "Sep": 0.25, "Oct": 0.02, "Nov": 0.0, "Dec": 0.0}},
    "SBS":  {"label": "Sub-Boreal Spruce",
             "fire_season": "May-Sep", "peak_month": "Jul",
             "monthly": {"Jan": 0.0, "Feb": 0.0, "Mar": 0.01, "Apr": 0.10, "May": 0.40,
                         "Jun": 0.70, "Jul": 1.0, "Aug": 0.90, "Sep": 0.20, "Oct": 0.02, "Nov": 0.0, "Dec": 0.0}},
    "IDF":  {"label": "Interior Douglas-fir",
             "fire_season": "Apr-Oct", "peak_month": "Aug",
             "monthly": {"Jan": 0.0, "Feb": 0.0, "Mar": 0.02, "Apr": 0.12, "May": 0.30,
                         "Jun": 0.55, "Jul": 0.90, "Aug": 1.0, "Sep": 0.35, "Oct": 0.08, "Nov": 0.01, "Dec": 0.0}},
    "ICH":  {"label": "Interior Cedar-Hemlock",
             "fire_season": "May-Sep", "peak_month": "Aug",
             "monthly": {"Jan": 0.0, "Feb": 0.0, "Mar": 0.01, "Apr": 0.05, "May": 0.20,
                         "Jun": 0.50, "Jul": 0.85, "Aug": 1.0, "Sep": 0.30, "Oct": 0.05, "Nov": 0.0, "Dec": 0.0}},
    "CWH":  {"label": "Coastal Western Hemlock",
             "fire_season": "Jun-Sep", "peak_month": "Aug",
             "monthly": {"Jan": 0.0, "Feb": 0.0, "Mar": 0.0, "Apr": 0.02, "May": 0.08,
                         "Jun": 0.25, "Jul": 0.65, "Aug": 1.0, "Sep": 0.30, "Oct": 0.05, "Nov": 0.0, "Dec": 0.0}},
    "CDF":  {"label": "Coastal Douglas-fir",
             "fire_season": "Jun-Sep", "peak_month": "Aug",
             "monthly": {"Jan": 0.0, "Feb": 0.0, "Mar": 0.01, "Apr": 0.05, "May": 0.15,
                         "Jun": 0.35, "Jul": 0.75, "Aug": 1.0, "Sep": 0.25, "Oct": 0.05, "Nov": 0.0, "Dec": 0.0}},
    "PP":   {"label": "Ponderosa Pine",
             "fire_season": "Apr-Oct", "peak_month": "Aug",
             "monthly": {"Jan": 0.0, "Feb": 0.01, "Mar": 0.05, "Apr": 0.15, "May": 0.30,
                         "Jun": 0.55, "Jul": 0.85, "Aug": 1.0, "Sep": 0.40, "Oct": 0.10, "Nov": 0.02, "Dec": 0.0}},
}

# Default for zones not explicitly listed
_DEFAULT_SEASONAL = {
    "fire_season": "May-Sep", "peak_month": "Jul",
    "monthly": {"Jan": 0.0, "Feb": 0.0, "Mar": 0.01, "Apr": 0.08, "May": 0.30,
                "Jun": 0.60, "Jul": 1.0, "Aug": 0.85, "Sep": 0.25, "Oct": 0.03, "Nov": 0.0, "Dec": 0.0},
}


def _get_seasonal_risk(bec_zone: str) -> dict:
    """Return seasonal fire risk curve for the BEC zone."""
    curve = _SEASONAL_CURVES.get(bec_zone, _DEFAULT_SEASONAL)
    return {
        "bec_zone": bec_zone,
        "fire_season": curve["fire_season"],
        "peak_month": curve["peak_month"],
        "monthly_risk_index": curve["monthly"],
        "note": "Relative fire frequency by month (0-1 scale, peak month = 1.0). "
                "Based on 225K BC fire records 1919-2025.",
    }
