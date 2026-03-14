"""Nearby active fires endpoint — real-time data from BC Wildfire Service."""

from __future__ import annotations

import logging
import math

from fastapi import APIRouter, HTTPException, Query

from infernis.config import settings

logger = logging.getLogger(__name__)

fires_router = APIRouter(prefix=settings.api_prefix, tags=["fires"])

BCWS_ACTIVE_FIRES_URL = (
    "https://services6.arcgis.com/ubm4tcTYICKBpist/arcgis/rest/services/"
    "BCWS_ActiveFires_PublicView/FeatureServer/0/query"
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@fires_router.get("/fires/near/{lat}/{lon}")
async def get_nearby_fires(
    lat: float,
    lon: float,
    radius_km: float = Query(default=50.0, ge=1.0, le=500.0, description="Search radius in km"),
):
    """Find active wildfires near a location.

    Queries the BC Wildfire Service (BCWS) ArcGIS REST API for active
    fire incidents, filters by distance, and returns sorted by proximity.

    **Use cases:**
    - "5 active fires within 50 km" alert card in a mobile app
    - Map marker layer showing nearby fire incidents
    - Evacuation planning: which fires are closest to my location?

    **Example request:**
    ```
    GET /v1/fires/near/50.67/-120.33?radius_km=100
    X-API-Key: your_key
    ```

    **Example response:**
    ```json
    {
      "latitude": 50.67,
      "longitude": -120.33,
      "radius_km": 100,
      "fires": [
        {
          "fire_number": "K62331",
          "status": "Under Control",
          "cause": "Lightning",
          "size_hectares": 45.2,
          "description": "Pocket Knife Creek",
          "latitude": 50.82,
          "longitude": -120.15,
          "distance_km": 18.3
        }
      ],
      "count": 1
    }
    ```

    Data is live from BC Wildfire Service. Results may be empty outside fire season.
    """
    if not (settings.bc_bbox_south <= lat <= settings.bc_bbox_north):
        raise HTTPException(status_code=422, detail=f"Latitude {lat} outside BC boundaries.")
    if not (settings.bc_bbox_west <= lon <= settings.bc_bbox_east):
        raise HTTPException(status_code=422, detail=f"Longitude {lon} outside BC boundaries.")

    fires = []
    try:
        import httpx

        params = {
            "where": "1=1",
            "outFields": "FIRE_NUMBER,FIRE_CAUSE,FIRE_STATUS,FIRE_SIZE_HECTARES,"
            "LATITUDE,LONGITUDE,GEOGRAPHIC_DESCRIPTION,DISCOVERED_DATE",
            "geometry": f"{lon - 2},{lat - 2},{lon + 2},{lat + 2}",
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "f": "json",
            "resultRecordCount": 100,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(BCWS_ACTIVE_FIRES_URL, params=params)
            if resp.status_code == 200:
                data = resp.json()
                for feature in data.get("features", []):
                    attrs = feature.get("attributes", {})
                    fire_lat = attrs.get("LATITUDE")
                    fire_lon = attrs.get("LONGITUDE")
                    if fire_lat and fire_lon:
                        dist = _haversine_km(lat, lon, fire_lat, fire_lon)
                        if dist <= radius_km:
                            fires.append(
                                {
                                    "fire_number": attrs.get("FIRE_NUMBER"),
                                    "status": attrs.get("FIRE_STATUS"),
                                    "cause": attrs.get("FIRE_CAUSE"),
                                    "size_hectares": attrs.get("FIRE_SIZE_HECTARES"),
                                    "description": attrs.get("GEOGRAPHIC_DESCRIPTION"),
                                    "latitude": fire_lat,
                                    "longitude": fire_lon,
                                    "distance_km": round(dist, 1),
                                }
                            )
    except Exception as e:
        logger.warning("BCWS fire query failed: %s", e)

    fires.sort(key=lambda f: f.get("distance_km", 999))

    return {
        "latitude": lat,
        "longitude": lon,
        "radius_km": radius_km,
        "fires": fires,
        "count": len(fires),
    }
