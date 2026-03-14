"""Historical risk endpoint — serves daily risk history from PostgreSQL."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Query

from infernis.config import settings
from infernis.models.enums import DangerLevel

logger = logging.getLogger(__name__)

history_router = APIRouter(prefix=settings.api_prefix, tags=["history"])


@history_router.get("/risk/history/{lat}/{lon}")
async def get_risk_history(
    lat: float,
    lon: float,
    days: int = Query(default=30, ge=1, le=90, description="Number of days of history (max 90)"),
):
    """Historical fire risk for a location over time.

    Returns daily risk scores, danger levels, and key weather data from
    the predictions database (up to 90 days of retention).

    **Use cases:**
    - Trend chart showing risk climbing over the past week
    - Seasonal risk comparison for a property
    - Insurance claim context ("risk was HIGH for 5 consecutive days")
    - Research: correlating risk scores with actual fire outcomes

    **Example request:**
    ```
    GET /v1/risk/history/50.67/-120.33?days=14
    X-API-Key: your_key
    ```

    **Example response:**
    ```json
    {
      "latitude": 50.67,
      "longitude": -120.33,
      "cell_id": "BC-5K-0015812",
      "days_requested": 14,
      "history": [
        {"date": "2026-03-01", "score": 0.05, "level": "LOW", "color": "#3B82F6", "fwi": 3.2, "temperature_c": 1.5, "rh_pct": 55},
        {"date": "2026-03-02", "score": 0.08, "level": "LOW", "color": "#3B82F6", "fwi": 4.1, "temperature_c": 3.0, "rh_pct": 48},
        ...
      ],
      "count": 14
    }
    ```
    """
    if not (settings.bc_bbox_south <= lat <= settings.bc_bbox_north):
        raise HTTPException(status_code=422, detail=f"Latitude {lat} outside BC boundaries.")
    if not (settings.bc_bbox_west <= lon <= settings.bc_bbox_east):
        raise HTTPException(status_code=422, detail=f"Longitude {lon} outside BC boundaries.")

    from infernis.api.routes import _find_nearest_cell, _predictions_cache

    if not _predictions_cache:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")

    cell_id = _find_nearest_cell(lat, lon)
    if cell_id is None:
        raise HTTPException(status_code=404, detail="No grid cell found for this location.")

    try:
        from infernis.db.engine import SessionLocal
        from infernis.db.tables import PredictionDB

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        db = SessionLocal()
        try:
            rows = (
                db.query(PredictionDB)
                .filter(
                    PredictionDB.cell_id == cell_id,
                    PredictionDB.prediction_date >= start_date,
                    PredictionDB.prediction_date <= end_date,
                )
                .order_by(PredictionDB.prediction_date)
                .all()
            )

            history = []
            for r in rows:
                level = DangerLevel.from_score(r.score)
                history.append(
                    {
                        "date": r.prediction_date.isoformat(),
                        "score": round(r.score, 4),
                        "level": level.value,
                        "color": level.color,
                        "fwi": r.fwi,
                        "temperature_c": r.temperature_c,
                        "rh_pct": r.rh_pct,
                    }
                )
        finally:
            db.close()
    except Exception as e:
        logger.error("History query failed: %s", e)
        history = []

    return {
        "latitude": lat,
        "longitude": lon,
        "cell_id": cell_id,
        "days_requested": days,
        "history": history,
        "count": len(history),
    }
