"""Webhook alert endpoints — register callbacks for risk threshold exceedances."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from infernis.config import settings

logger = logging.getLogger(__name__)

alerts_router = APIRouter(prefix=settings.api_prefix, tags=["alerts"])


class AlertCreate(BaseModel):
    latitude: float
    longitude: float
    threshold: float = Field(..., ge=0.0, le=1.0, description="Risk score threshold (0-1)")
    webhook_url: str = Field(..., max_length=500, description="URL to POST when threshold exceeded")


@alerts_router.post("/alerts", status_code=201)
async def create_alert(alert: AlertCreate, request: Request):
    """Register a webhook alert for risk threshold exceedance.

    When the daily pipeline runs and the risk score for the nearest grid cell
    exceeds your threshold, a POST is sent to your webhook URL with the risk payload.

    **Use cases:**
    - Push notification when risk at your cabin exceeds HIGH (0.35)
    - Slack webhook for fire operations team
    - Email trigger via Zapier/Make when risk changes

    **Example request:**
    ```
    POST /v1/alerts
    X-API-Key: your_key
    Content-Type: application/json

    {"latitude": 50.67, "longitude": -120.33, "threshold": 0.35,
     "webhook_url": "https://hooks.slack.com/services/xxx/yyy/zzz"}
    ```
    """
    from infernis.api.routes import _find_nearest_cell

    cell_id = _find_nearest_cell(alert.latitude, alert.longitude)
    if cell_id is None:
        raise HTTPException(status_code=404, detail="No grid cell found for this location.")

    api_key_id = getattr(request.state, "api_key_id", 0)

    try:
        from infernis.db.engine import SessionLocal
        from infernis.db.tables import AlertDB

        db = SessionLocal()
        try:
            record = AlertDB(
                api_key_id=api_key_id,
                latitude=alert.latitude,
                longitude=alert.longitude,
                cell_id=cell_id,
                threshold=alert.threshold,
                webhook_url=alert.webhook_url,
            )
            db.add(record)
            db.commit()
            db.refresh(record)

            return {
                "id": record.id,
                "cell_id": cell_id,
                "threshold": alert.threshold,
                "webhook_url": alert.webhook_url,
                "status": "active",
            }
        finally:
            db.close()
    except Exception as e:
        logger.error("Failed to create alert: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create alert")


@alerts_router.get("/alerts")
async def list_alerts(request: Request):
    """List all active alerts for your API key.

    **Example response:**
    ```json
    {"alerts": [{"id": 1, "cell_id": "BC-5K-0015812", "threshold": 0.35,
      "webhook_url": "https://...", "last_triggered": null}], "count": 1}
    ```
    """
    api_key_id = getattr(request.state, "api_key_id", 0)

    try:
        from infernis.db.engine import SessionLocal
        from infernis.db.tables import AlertDB

        db = SessionLocal()
        try:
            alerts = (
                db.query(AlertDB)
                .filter(AlertDB.api_key_id == api_key_id, AlertDB.is_active == True)  # noqa: E712
                .all()
            )
            return {
                "alerts": [
                    {
                        "id": a.id,
                        "cell_id": a.cell_id,
                        "latitude": a.latitude,
                        "longitude": a.longitude,
                        "threshold": a.threshold,
                        "webhook_url": a.webhook_url,
                        "last_triggered": a.last_triggered.isoformat()
                        if a.last_triggered
                        else None,
                    }
                    for a in alerts
                ],
                "count": len(alerts),
            }
        finally:
            db.close()
    except Exception as e:
        logger.error("Failed to list alerts: %s", e)
        return {"alerts": [], "count": 0}


@alerts_router.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: int, request: Request):
    """Deactivate an alert by ID.

    Only deactivates alerts belonging to your API key.
    """
    api_key_id = getattr(request.state, "api_key_id", 0)

    try:
        from infernis.db.engine import SessionLocal
        from infernis.db.tables import AlertDB

        db = SessionLocal()
        try:
            alert = (
                db.query(AlertDB)
                .filter(AlertDB.id == alert_id, AlertDB.api_key_id == api_key_id)
                .first()
            )
            if not alert:
                raise HTTPException(status_code=404, detail="Alert not found")
            alert.is_active = False
            db.commit()
            return {"id": alert_id, "status": "deactivated"}
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete alert: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete alert")
