"""Batch risk query endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import infernis.api.routes as _routes
from infernis.config import settings
from infernis.models.enums import DangerLevel

logger = logging.getLogger(__name__)

batch_router = APIRouter(prefix=settings.api_prefix, tags=["batch"])

MAX_BATCH_SIZE = 50


class LocationInput(BaseModel):
    lat: float
    lon: float


class BatchRequest(BaseModel):
    locations: list[LocationInput] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


@batch_router.post("/risk/batch")
async def batch_risk(req: BatchRequest):
    """Query fire risk for up to 50 locations in a single request.

    Returns an array of risk results in the same order as the input locations.
    Locations that fall outside BC or have no data return an error entry.

    **Use cases:**
    - Dashboard showing risk for 20 saved properties at once
    - Insurance portfolio risk assessment across multiple sites
    - Batch geocoded address risk lookups

    **Example request:**
    ```
    POST /v1/risk/batch
    X-API-Key: your_key
    Content-Type: application/json

    {
      "locations": [
        {"lat": 50.67, "lon": -120.33},
        {"lat": 49.25, "lon": -123.10},
        {"lat": 54.02, "lon": -124.00}
      ]
    }
    ```

    **Example response:**
    ```json
    {
      "results": [
        {"lat": 50.67, "lon": -120.33, "grid_cell_id": "BC-5K-0015812",
         "risk": {"score": 0.29, "level": "MODERATE", "color": "#EAB308"},
         "fwi": 7.0, "temperature_c": 3.2, "bec_zone": "IDF"},
        ...
      ],
      "count": 3
    }
    ```

    Maximum 50 locations per request. Counts as 1 API call regardless of batch size.
    """
    if not _routes._predictions_cache:
        raise HTTPException(status_code=503, detail="Predictions not yet available.")

    results = []
    for loc in req.locations:
        cell_id = _routes._find_nearest_cell(loc.lat, loc.lon)
        if cell_id is None or cell_id not in _routes._predictions_cache:
            results.append({"lat": loc.lat, "lon": loc.lon, "error": "No data for this location"})
            continue

        pred = _routes._predictions_cache[cell_id]
        cell = _routes._grid_cells.get(cell_id, {})
        score = pred.get("score", 0.0)
        level = DangerLevel.from_score(score)

        results.append(
            {
                "lat": loc.lat,
                "lon": loc.lon,
                "grid_cell_id": cell_id,
                "risk": {"score": score, "level": level.value, "color": level.color},
                "fwi": pred.get("fwi", 0.0),
                "temperature_c": pred.get("temperature_c", 0.0),
                "bec_zone": cell.get("bec_zone", ""),
            }
        )

    return {"results": results, "count": len(results)}
