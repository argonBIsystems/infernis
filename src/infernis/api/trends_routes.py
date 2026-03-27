"""Anomaly & trend detection endpoint.

GET /v1/trends/{lat}/{lon}
    Compare today's risk against historical baseline (susceptibility score)
    and recent trajectory. Answers: "Is this unusual for this location?"
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Query

from infernis.config import settings
from infernis.models.enums import DangerLevel

logger = logging.getLogger(__name__)

trends_router = APIRouter(prefix=settings.api_prefix, tags=["trends"])


def _safe(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _anomaly_status(departure_pct: float) -> str:
    """Classify departure from baseline."""
    if departure_pct < -20:
        return "WELL_BELOW_NORMAL"
    elif departure_pct < 0:
        return "BELOW_NORMAL"
    elif departure_pct < 50:
        return "NEAR_NORMAL"
    elif departure_pct < 150:
        return "ABOVE_NORMAL"
    elif departure_pct < 300:
        return "WELL_ABOVE_NORMAL"
    else:
        return "RECORD_HIGH"


@trends_router.get("/trends/{lat}/{lon}")
async def get_trends(
    lat: float,
    lon: float,
    days: int = Query(default=7, ge=1, le=30, description="Lookback days for velocity (max 30)"),
):
    """Compare current risk against historical baseline and recent trajectory.

    Returns how today's risk compares to the location's long-term susceptibility,
    the rate of change over the past N days, and anomaly classification.

    **Use cases:**
    - "Is this unusual for this location?" — departure from baseline
    - "Is risk accelerating?" — 3-day and 7-day velocity
    - Government: flag zones where risk is spiking in historically safe areas
    - Insurance: identify locations experiencing abnormal conditions

    **Example:**
    ```
    GET /v1/trends/50.67/-120.33?days=7
    ```
    """
    from infernis.api.routes import (
        _find_nearest_cell,
        _get_prediction,
        _grid_cells,
        _safe_float,
        _validate_bc_coords,
    )

    _validate_bc_coords(lat, lon)

    cell_id = _find_nearest_cell(lat, lon)
    if cell_id is None:
        raise HTTPException(status_code=404, detail="No grid cell found for this location.")

    pred = _get_prediction(cell_id)
    if pred is None:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")

    current_score = _safe_float(pred.get("score"), 0.0)
    cell = _grid_cells.get(cell_id, {})

    # Get historical baseline from fire statistics
    from infernis.api.profile_routes import _get_fire_stats

    stats = _get_fire_stats(cell_id) or {}
    susceptibility_score = _safe(stats.get("susceptibility_score", 0.0))
    susceptibility_percentile = stats.get("susceptibility_percentile", 50)
    susceptibility_label = stats.get("susceptibility_label", "AVERAGE")

    # Compute departure from baseline
    if susceptibility_score > 0:
        departure_pct = ((current_score - susceptibility_score) / susceptibility_score) * 100
    else:
        departure_pct = current_score * 10000 if current_score > 0 else 0.0

    status = _anomaly_status(departure_pct)

    # Get recent history for velocity calculation
    history_scores = []
    try:
        from infernis.db.engine import SessionLocal
        from infernis.db.tables import PredictionDB

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        db = SessionLocal()
        try:
            rows = (
                db.query(PredictionDB.prediction_date, PredictionDB.score)
                .filter(
                    PredictionDB.cell_id == cell_id,
                    PredictionDB.prediction_date >= start_date,
                    PredictionDB.prediction_date <= end_date,
                )
                .order_by(PredictionDB.prediction_date)
                .all()
            )
            history_scores = [
                {"date": r.prediction_date.isoformat(), "score": _safe(r.score)} for r in rows
            ]
        finally:
            db.close()
    except Exception as e:
        logger.warning("Trends history query failed: %s", e)

    # Compute velocity (rate of change)
    velocity_3day = None
    velocity_7day = None
    if len(history_scores) >= 2:
        scores = [h["score"] for h in history_scores]
        if len(scores) >= 4:
            recent_3 = scores[-1]
            past_3 = scores[-min(4, len(scores))]
            if past_3 > 0:
                velocity_3day = round(((recent_3 - past_3) / past_3) * 100, 1)
        if len(scores) >= 7:
            recent_7 = scores[-1]
            past_7 = scores[-min(8, len(scores))]
            if past_7 > 0:
                velocity_7day = round(((recent_7 - past_7) / past_7) * 100, 1)

    return {
        "cell_id": cell_id,
        "location": {"lat": lat, "lon": lon},
        "date": date.today().isoformat(),
        "current_score": round(current_score, 4),
        "level": DangerLevel.from_score(current_score).value,
        "baseline": {
            "susceptibility_score": round(susceptibility_score, 6),
            "susceptibility_percentile": susceptibility_percentile,
            "susceptibility_label": susceptibility_label,
        },
        "departure": {
            "pct": round(departure_pct, 1),
            "status": status,
        },
        "velocity": {
            "days_analyzed": len(history_scores),
            "velocity_3day_pct": velocity_3day,
            "velocity_7day_pct": velocity_7day,
        },
        "history": history_scores,
        "context": {
            "bec_zone": cell.get("bec_zone", ""),
            "fuel_type": cell.get("fuel_type", ""),
            "elevation_m": cell.get("elevation_m", 0),
        },
    }
