"""SHAP-based explainability endpoints for INFERNIS.

GET /v1/explain/{lat}/{lon}
    Top SHAP drivers for the nearest grid cell, with human-readable
    descriptions and a composed plain-English summary.

GET /v1/explain/zones
    Per-BEC-zone aggregated SHAP drivers (mean |SHAP| per feature per zone).

Both endpoints read from the in-memory predictions cache populated by the
daily pipeline. shap_values are stored in each prediction dict as
{feature_name: contribution_float}. When shap_values are absent (pipeline
ran without SHAP), the endpoints return empty driver lists gracefully.
"""

from __future__ import annotations

import logging
import math

import infernis.api.routes as _routes_module
from fastapi import APIRouter, HTTPException, Query
from infernis.api.routes import _find_nearest_cell, _validate_bc_coords
from infernis.config import settings
from infernis.services.explainability import FEATURE_DESCRIPTIONS, ExplainabilityService

logger = logging.getLogger(__name__)


def _safe(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


explain_router = APIRouter(prefix=settings.api_prefix)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drivers_from_shap(shap_values: dict | None, feature_values: dict, top_n: int) -> list[dict]:
    """Convert a {feature: contribution} dict into a sorted driver list.

    Does NOT call shap.TreeExplainer — the values are already precomputed and
    stored in the predictions cache by the daily pipeline.

    Parameters
    ----------
    shap_values:
        Dict mapping feature_name → SHAP contribution float.  None → empty list.
    feature_values:
        Dict mapping feature_name → observed value for display.
    top_n:
        Maximum number of drivers to return.

    Returns
    -------
    List of driver dicts sorted by descending |contribution|.
    """
    if not shap_values:
        return []

    # Build (feature, contribution, value) triples sorted by |contribution|
    triples = [
        (feat, contrib, feature_values.get(feat, 0.0)) for feat, contrib in shap_values.items()
    ]
    triples.sort(key=lambda t: abs(t[1]), reverse=True)
    triples = triples[:top_n]

    drivers = []
    for feat, contrib, val in triples:
        direction = "increasing" if contrib > 0 else "decreasing"
        tmpl = FEATURE_DESCRIPTIONS.get(feat, f"{feat} = {{value:.4f}}")
        try:
            description = tmpl.format(value=_safe(val))
        except Exception:
            description = tmpl

        drivers.append(
            {
                "feature": feat,
                "contribution": round(_safe(contrib), 6),
                "value": _safe(val),
                "direction": direction,
                "description": description,
            }
        )
    return drivers


def _build_feature_values(pred: dict) -> dict:
    """Extract a feature-value dict from a prediction cache entry.

    Maps prediction cache keys to the FEATURE_NAMES used by SHAP.
    Some features (e.g. soil_moisture_2/3/4) may not be in the cache
    and will default to 0.0.
    """
    return {
        "ffmc": pred.get("ffmc", 0.0),
        "dmc": pred.get("dmc", 0.0),
        "dc": pred.get("dc", 0.0),
        "isi": pred.get("isi", 0.0),
        "bui": pred.get("bui", 0.0),
        "fwi": pred.get("fwi", 0.0),
        "temperature_c": pred.get("temperature_c", 0.0),
        "rh_pct": pred.get("rh_pct", 0.0),
        "wind_kmh": pred.get("wind_kmh", 0.0),
        "wind_dir_deg": pred.get("wind_dir_deg", 0.0),
        "precip_24h_mm": pred.get("precip_24h_mm", 0.0),
        "soil_moisture_1": pred.get("soil_moisture", 0.0),
        "soil_moisture_2": pred.get("soil_moisture_2", 0.0),
        "soil_moisture_3": pred.get("soil_moisture_3", 0.0),
        "soil_moisture_4": pred.get("soil_moisture_4", 0.0),
        "evapotrans_mm": pred.get("evapotrans_mm", 0.0),
        "ndvi": pred.get("ndvi", 0.0),
        "snow_cover": float(pred.get("snow_cover", False)),
        "lai": pred.get("lai", 0.0),
        "elevation_m": 0.0,  # not stored in predictions dict
        "slope_deg": 0.0,
        "aspect_deg": 0.0,
        "hillshade": 0.0,
        "distance_to_road_km": 0.0,
        "doy_sin": pred.get("doy_sin", 0.0),
        "doy_cos": pred.get("doy_cos", 0.0),
        "lightning_24h": pred.get("lightning_24h", 0.0),
        "lightning_72h": pred.get("lightning_72h", 0.0),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@explain_router.get("/explain/{lat}/{lon}", tags=["explainability"])
async def explain_point(
    lat: float,
    lon: float,
    top_n: int = Query(default=5, ge=1, le=28, description="Number of top drivers to return"),
):
    """SHAP-based risk explanation for the nearest grid cell.

    Returns the top SHAP drivers explaining WHY fire risk is high or low at
    the given location, with human-readable descriptions and a composed
    plain-English summary.

    **Use cases:**
    - "Why is risk high here today?" — show users the specific conditions
    - Regulatory filings that require explainable AI justification
    - Training materials for fire officers

    **Example:**
    ```
    GET /v1/explain/50.67/-120.33
    ```
    Returns top 5 SHAP drivers, each with feature name, contribution value,
    direction (increasing/decreasing risk), and plain-English description.
    """
    _validate_bc_coords(lat, lon)

    if not _routes_module._has_predictions():
        raise HTTPException(
            status_code=503,
            detail="Predictions not yet available. Pipeline may be initializing.",
        )

    cell_id = _find_nearest_cell(lat, lon)
    if cell_id is None or _routes_module._get_prediction(cell_id) is None:
        raise HTTPException(
            status_code=503,
            detail="No prediction data for this location.",
        )

    pred = _routes_module._get_prediction(cell_id)
    shap_values = pred.get("shap_values")
    feature_values = _build_feature_values(pred)

    drivers = _drivers_from_shap(shap_values, feature_values, top_n=top_n)

    # Generate human-readable summary using ExplainabilityService
    level = pred.get("level", "MODERATE")
    try:
        # Use a model-less service just for generate_summary()
        from infernis.pipelines.data_processor import FEATURE_NAMES

        svc = ExplainabilityService(None, FEATURE_NAMES)
        # Monkey-patch: generate_summary doesn't need _shap_explainer
        summary = svc.generate_summary(drivers, level)
    except Exception as e:
        logger.warning("explain_point: summary generation failed: %s", e)
        summary = f"Risk is {level.lower()}."

    return {
        "location": {"lat": lat, "lon": lon},
        "cell_id": cell_id,
        "risk_score": _safe(pred.get("score")),
        "danger_level": level,
        "timestamp": pred.get("timestamp", ""),
        "drivers": drivers,
        "summary": summary,
        "shap_available": shap_values is not None,
    }


@explain_router.get("/explain/zones", tags=["explainability"])
async def explain_zones(
    top_n: int = Query(default=5, ge=1, le=28, description="Top drivers per zone"),
):
    """Per-BEC-zone aggregated SHAP drivers.

    Aggregates mean absolute SHAP value per feature per biogeoclimatic zone.
    Returns the top contributing features for each zone — useful for
    understanding which environmental factors are most important across
    different ecological regions of BC.

    **Use cases:**
    - Compare what drives risk in coastal (CWH) vs interior (IDF) zones
    - Identify province-wide patterns in fire weather drivers
    - Research and model validation

    **Example:**
    ```
    GET /v1/explain/zones
    ```
    Returns a list of zones, each with bec_zone, cell_count, and a ranked
    list of top_drivers with mean absolute SHAP contribution.
    """
    if not _routes_module._has_predictions():
        raise HTTPException(
            status_code=503,
            detail="Predictions not yet available.",
        )

    # Aggregate per-zone SHAP contributions
    # zone_data: { bec_zone: { feature: [abs_shap, ...], cell_count: int } }
    zone_data: dict[str, dict] = {}

    # Batch-fetch predictions from Redis for zone aggregation
    import json as _json

    from infernis.services.cache import get_redis as _get_redis

    _r = _get_redis()
    _all_preds = {}
    if _r:
        _cids = list(_routes_module._grid_cells.keys())
        for _i in range(0, len(_cids), 5000):
            _batch = _cids[_i : _i + 5000]
            _pipe = _r.pipeline()
            for _cid in _batch:
                _pipe.get(f"pred:latest:{_cid}")
            _vals = _pipe.execute()
            for _cid, _v in zip(_batch, _vals):
                if _v:
                    _all_preds[_cid] = _json.loads(_v)

    for cell_id, pred in _all_preds.items():
        cell = _routes_module._grid_cells.get(cell_id, {})
        zone = cell.get("bec_zone") or "UNKNOWN"
        shap_values = pred.get("shap_values")

        if zone not in zone_data:
            zone_data[zone] = {"feature_shap": {}, "cell_count": 0}

        zone_data[zone]["cell_count"] += 1

        if not shap_values:
            continue

        for feat, contrib in shap_values.items():
            if feat not in zone_data[zone]["feature_shap"]:
                zone_data[zone]["feature_shap"][feat] = []
            zone_data[zone]["feature_shap"][feat].append(abs(_safe(contrib)))

    # Build result list
    import numpy as np

    result = []
    for zone, data in sorted(zone_data.items()):
        feat_means = {feat: _safe(np.mean(vals)) for feat, vals in data["feature_shap"].items()}
        # Sort features by mean |SHAP| descending and take top_n
        sorted_feats = sorted(feat_means.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

        top_drivers = [
            {"feature": feat, "mean_abs_shap": round(mean_val, 6)}
            for feat, mean_val in sorted_feats
        ]

        result.append(
            {
                "bec_zone": zone,
                "cell_count": data["cell_count"],
                "top_drivers": top_drivers,
            }
        )

    return {
        "zones": result,
        "timestamp": _routes_module._last_pipeline_run,
    }
