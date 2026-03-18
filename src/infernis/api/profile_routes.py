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
    if not _routes_module._predictions_cache:
        raise HTTPException(
            status_code=503,
            detail="Predictions not yet available. Pipeline may be initializing.",
        )

    # Require fire statistics to be available
    if not _fire_stats_cache:
        raise HTTPException(
            status_code=503,
            detail=(
                "Fire statistics not yet computed. "
                "Run: python -m infernis.admin compute_fire_stats"
            ),
        )

    cell_id = _find_nearest_cell(lat, lon)
    if cell_id is None:
        raise HTTPException(status_code=404, detail="No grid cell found for this location.")

    if cell_id not in _routes_module._predictions_cache:
        raise HTTPException(
            status_code=503,
            detail="Predictions not yet available. Pipeline may be initializing.",
        )

    stats = _fire_stats_cache.get(cell_id)
    if stats is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No fire statistics found for cell {cell_id}. "
                "Run: python -m infernis.admin compute_fire_stats"
            ),
        )

    pred = _routes_module._predictions_cache[cell_id]
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
        # --- Short forecast snippet (optional) ---
        "forecast_snippet": forecast_snippet,
    }
