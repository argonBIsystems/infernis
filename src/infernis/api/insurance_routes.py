"""Insurance vertical endpoints — bulk property risk assessment.

POST /v1/insurance/portfolio
    Accepts up to 1,000 properties with coordinates and optional metadata.
    Returns per-property risk profiles and aggregate portfolio metrics.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from infernis.config import settings
from infernis.models.enums import BEC_ZONE_NAMES, DangerLevel

logger = logging.getLogger(__name__)

insurance_router = APIRouter(prefix=settings.api_prefix, tags=["insurance"])


def _safe(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


class PropertyInput(BaseModel):
    id: str = Field(..., description="Your property identifier")
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    value_cad: float | None = Field(None, description="Insured value in CAD (optional)")


class PortfolioRequest(BaseModel):
    properties: list[PropertyInput] = Field(
        ..., min_length=1, max_length=1000,
        description="List of properties to assess (max 1,000)",
    )


@insurance_router.post("/insurance/portfolio")
async def assess_portfolio(req: PortfolioRequest):
    """Bulk wildfire risk assessment for insurance underwriting.

    Accepts up to 1,000 properties and returns:
    - Per-property: risk score, danger level, susceptibility, historical fire
      exposure, BEC zone, fire regime, seasonal risk profile
    - Portfolio aggregate: risk distribution, value-at-risk by tier,
      mean/max scores, zone breakdown

    **Use cases:**
    - Underwriting: screen new policy applications against wildfire exposure
    - Portfolio management: identify concentration of risk by zone
    - Renewal pricing: adjust premiums based on current + historical risk
    - Regulatory reporting: aggregate exposure metrics

    **Example:**
    ```json
    POST /v1/insurance/portfolio
    {
      "properties": [
        {"id": "POL-001", "lat": 50.67, "lon": -120.33, "value_cad": 450000},
        {"id": "POL-002", "lat": 49.88, "lon": -119.49, "value_cad": 620000}
      ]
    }
    ```
    """
    from infernis.api.profile_routes import _get_fire_stats, _get_seasonal_risk
    from infernis.api.routes import (
        _find_nearest_cell,
        _get_forecast,
        _get_prediction,
        _grid_cells,
        _has_predictions,
        _safe_float,
    )

    if not _has_predictions():
        raise HTTPException(status_code=503, detail="Predictions not yet available.")

    results = []
    total_value = 0.0
    value_at_risk = {"VERY_LOW": 0, "LOW": 0, "MODERATE": 0, "HIGH": 0, "VERY_HIGH": 0, "EXTREME": 0}
    risk_distribution = {"VERY_LOW": 0, "LOW": 0, "MODERATE": 0, "HIGH": 0, "VERY_HIGH": 0, "EXTREME": 0}
    zone_counts: dict[str, int] = {}
    scores = []

    for prop in req.properties:
        cell_id = _find_nearest_cell(prop.lat, prop.lon)
        if cell_id is None:
            results.append({
                "id": prop.id,
                "lat": prop.lat,
                "lon": prop.lon,
                "error": "Outside BC coverage area",
            })
            continue

        pred = _get_prediction(cell_id)
        if pred is None:
            results.append({
                "id": prop.id,
                "lat": prop.lat,
                "lon": prop.lon,
                "error": "No prediction data available",
            })
            continue

        cell = _grid_cells.get(cell_id, {})
        stats = _get_fire_stats(cell_id) or {}
        bec_zone = cell.get("bec_zone", "")

        # Current risk
        score = _safe_float(pred.get("score"), 0.0)
        level = DangerLevel.from_score(score)

        # Susceptibility
        susc_pct = stats.get("susceptibility_percentile", 0)
        susc_label = stats.get("susceptibility_label", "")
        exp_pct = stats.get("exposure_percentile", 0)

        # Composite
        composite = 0.3 * susc_pct / 100.0 + 0.3 * exp_pct / 100.0 + 0.4 * score
        composite = min(max(composite, 0.0), 1.0)
        composite_level = DangerLevel.from_score(composite)

        # Historical exposure
        fires_10yr = stats.get("fires_10yr", {})
        fires_30yr = stats.get("fires_30yr", {})

        # Forecast max (3-day)
        fc = _get_forecast(cell_id)
        forecast_3day_max = None
        if fc:
            forecast_3day_max = max((_safe(d.get("risk_score")) for d in fc[:3]), default=None)

        # SHAP top drivers
        shap = pred.get("shap_values", {})
        drivers = []
        if shap:
            sorted_shap = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            drivers = [{"feature": f, "contribution": round(c, 4)} for f, c in sorted_shap]

        prop_result = {
            "id": prop.id,
            "lat": prop.lat,
            "lon": prop.lon,
            "cell_id": cell_id,
            "value_cad": prop.value_cad,
            "risk": {
                "current_score": round(score, 4),
                "current_level": level.value,
                "composite_score": round(composite, 4),
                "composite_level": composite_level.value,
                "forecast_3day_max": round(forecast_3day_max, 4) if forecast_3day_max else None,
            },
            "susceptibility": {
                "percentile": susc_pct,
                "label": susc_label,
            },
            "historical_exposure": {
                "fires_10yr": fires_10yr.get("count", 0),
                "fires_30yr": fires_30yr.get("count", 0),
                "nearest_fire_km": fires_10yr.get("nearest_km"),
                "largest_fire_ha": fires_30yr.get("largest_ha"),
            },
            "context": {
                "bec_zone": bec_zone,
                "bec_zone_name": BEC_ZONE_NAMES.get(bec_zone, bec_zone),
                "fuel_type": cell.get("fuel_type", ""),
                "elevation_m": cell.get("elevation_m", 0),
            },
            "seasonal_risk": _get_seasonal_risk(bec_zone),
            "drivers": drivers,
        }
        results.append(prop_result)

        # Aggregate stats
        scores.append(composite)
        risk_distribution[composite_level.value] += 1
        zone_counts[bec_zone] = zone_counts.get(bec_zone, 0) + 1

        if prop.value_cad:
            total_value += prop.value_cad
            value_at_risk[composite_level.value] += prop.value_cad

    # Build portfolio summary
    assessed = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    summary = {
        "total_properties": len(req.properties),
        "assessed": len(assessed),
        "failed": len(failed),
        "mean_composite_score": round(sum(scores) / len(scores), 4) if scores else None,
        "max_composite_score": round(max(scores), 4) if scores else None,
        "risk_distribution": risk_distribution,
        "total_insured_value": total_value if total_value > 0 else None,
        "value_at_risk": {k: v for k, v in value_at_risk.items() if v > 0} if total_value > 0 else None,
        "zone_breakdown": {
            zone: {"count": count, "name": BEC_ZONE_NAMES.get(zone, zone)}
            for zone, count in sorted(zone_counts.items(), key=lambda x: -x[1])
        },
    }

    return {
        "portfolio_summary": summary,
        "properties": results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
