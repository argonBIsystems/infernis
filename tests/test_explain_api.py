"""Tests for /v1/explain endpoints (SHAP-based risk explainability)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from infernis.api.explain_routes import explain_router
from infernis.api.routes import set_predictions_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FEATURE_NAMES = [
    "ffmc", "dmc", "dc", "isi", "bui", "fwi",
    "temperature_c", "rh_pct", "wind_kmh", "wind_dir_deg", "precip_24h_mm",
    "soil_moisture_1", "soil_moisture_2", "soil_moisture_3", "soil_moisture_4",
    "evapotrans_mm", "ndvi", "snow_cover", "lai",
    "elevation_m", "slope_deg", "aspect_deg", "hillshade", "distance_to_road_km",
    "doy_sin", "doy_cos", "lightning_24h", "lightning_72h",
]


def _make_shap_values(seed: int = 0) -> dict:
    """Build a synthetic {feature: contribution} dict."""
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal(28)
    return {name: round(float(v), 6) for name, v in zip(_FEATURE_NAMES, vals)}


def _make_prediction(cell_id: str, score: float = 0.5, seed: int = 0) -> dict:
    return {
        "score": score,
        "level": "HIGH",
        "timestamp": "2026-03-15T14:00:00+00:00",
        "ffmc": 88.0,
        "dmc": 45.0,
        "dc": 200.0,
        "isi": 7.0,
        "bui": 65.0,
        "fwi": 22.0,
        "temperature_c": 28.0,
        "rh_pct": 25.0,
        "wind_kmh": 20.0,
        "precip_24h_mm": 0.0,
        "soil_moisture": 0.18,
        "ndvi": 0.35,
        "snow_cover": False,
        "c_haines": None,
        "confidence_interval": None,
        "shap_values": _make_shap_values(seed),
        "next_update": "",
    }


def _make_grid_cells(*cell_ids):
    """Build a grid_cells dict for the given cell IDs."""
    cells = {}
    for i, cid in enumerate(cell_ids):
        cells[cid] = {
            "lat": 50.0 + i * 0.01,
            "lon": -120.0 + i * 0.01,
            "bec_zone": "IDF",
            "fuel_type": "C4",
            "elevation_m": 400.0,
        }
    return cells


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_data():
    """TestClient with predictions cache populated with 3 cells."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(explain_router)

    cell_ids = ["CELL_001", "CELL_002", "CELL_003"]
    predictions = {cid: _make_prediction(cid, score=0.4 + i * 0.1, seed=i)
                   for i, cid in enumerate(cell_ids)}
    grid_cells = _make_grid_cells(*cell_ids)
    set_predictions_cache(predictions, grid_cells, "2026-03-15T14:00:00+00:00")

    return TestClient(app)


@pytest.fixture
def client_no_data():
    """TestClient with empty predictions cache."""
    from fastapi import FastAPI

    from infernis.api import routes as _r

    app = FastAPI()
    app.include_router(explain_router)

    # Clear the cache
    _r._predictions_cache.clear()
    _r._grid_cells.clear()
    _r._kdtree = None
    _r._cell_ids_ordered.clear()

    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /v1/explain/{lat}/{lon}
# ---------------------------------------------------------------------------


class TestExplainPoint:
    def test_returns_200(self, client_with_data):
        """Returns 200 for a valid BC coordinate."""
        r = client_with_data.get("/v1/explain/50.0/-120.0")
        assert r.status_code == 200

    def test_response_has_drivers(self, client_with_data):
        """Response contains a non-empty 'drivers' list."""
        r = client_with_data.get("/v1/explain/50.0/-120.0")
        body = r.json()
        assert "drivers" in body
        assert isinstance(body["drivers"], list)
        assert len(body["drivers"]) > 0

    def test_drivers_have_required_keys(self, client_with_data):
        """Each driver has: feature, contribution, value, direction, description."""
        r = client_with_data.get("/v1/explain/50.0/-120.0")
        required = {"feature", "contribution", "value", "direction", "description"}
        for d in r.json()["drivers"]:
            assert required.issubset(d.keys()), f"Missing keys: {required - d.keys()}"

    def test_top_n_default_is_5(self, client_with_data):
        """Default response has 5 drivers."""
        r = client_with_data.get("/v1/explain/50.0/-120.0")
        assert len(r.json()["drivers"]) == 5

    def test_top_n_query_param(self, client_with_data):
        """top_n query param is respected (e.g. 3)."""
        r = client_with_data.get("/v1/explain/50.0/-120.0?top_n=3")
        assert len(r.json()["drivers"]) == 3

    def test_response_has_summary(self, client_with_data):
        """Response contains a non-empty 'summary' string."""
        r = client_with_data.get("/v1/explain/50.0/-120.0")
        body = r.json()
        assert "summary" in body
        assert isinstance(body["summary"], str)
        assert len(body["summary"]) > 0

    def test_response_has_cell_id(self, client_with_data):
        """Response contains 'cell_id'."""
        r = client_with_data.get("/v1/explain/50.0/-120.0")
        assert "cell_id" in r.json()

    def test_response_has_score_and_level(self, client_with_data):
        """Response contains 'risk_score' and 'danger_level'."""
        r = client_with_data.get("/v1/explain/50.0/-120.0")
        body = r.json()
        assert "risk_score" in body
        assert "danger_level" in body

    def test_503_when_no_predictions(self, client_no_data):
        """Returns 503 when predictions cache is empty."""
        r = client_no_data.get("/v1/explain/50.0/-120.0")
        assert r.status_code == 503

    def test_out_of_bc_lat_returns_422(self, client_with_data):
        """Latitude outside BC bounding box returns 422."""
        r = client_with_data.get("/v1/explain/20.0/-120.0")
        assert r.status_code == 422

    def test_out_of_bc_lon_returns_422(self, client_with_data):
        """Longitude outside BC bounding box returns 422."""
        r = client_with_data.get("/v1/explain/50.0/-50.0")
        assert r.status_code == 422

    def test_sorted_by_abs_contribution(self, client_with_data):
        """Drivers are sorted by descending absolute contribution."""
        r = client_with_data.get("/v1/explain/50.0/-120.0")
        contribs = [abs(d["contribution"]) for d in r.json()["drivers"]]
        assert contribs == sorted(contribs, reverse=True)

    def test_drivers_fallback_without_shap(self):
        """When shap_values is None in cache, endpoint returns graceful response."""
        from fastapi import FastAPI

        from infernis.api import routes as _r

        app = FastAPI()
        app.include_router(explain_router)

        pred_no_shap = _make_prediction("CELL_X")
        pred_no_shap["shap_values"] = None  # simulate no SHAP
        grid_cells = _make_grid_cells("CELL_X")
        set_predictions_cache({"CELL_X": pred_no_shap}, grid_cells, "2026-03-15T14:00:00+00:00")

        c = TestClient(app)
        r = c.get("/v1/explain/50.0/-120.0")
        # Should still succeed; drivers may be empty list
        assert r.status_code in (200, 503)


# ---------------------------------------------------------------------------
# GET /v1/explain/zones
# ---------------------------------------------------------------------------


class TestExplainZones:
    def test_returns_200(self, client_with_data):
        """Returns 200 for the zones endpoint."""
        r = client_with_data.get("/v1/explain/zones")
        assert r.status_code == 200

    def test_response_has_zones(self, client_with_data):
        """Response contains a 'zones' list."""
        r = client_with_data.get("/v1/explain/zones")
        body = r.json()
        assert "zones" in body
        assert isinstance(body["zones"], list)

    def test_zone_entry_structure(self, client_with_data):
        """Each zone entry has bec_zone, cell_count, top_drivers."""
        r = client_with_data.get("/v1/explain/zones")
        for zone in r.json()["zones"]:
            assert "bec_zone" in zone
            assert "cell_count" in zone
            assert "top_drivers" in zone

    def test_top_drivers_per_zone_is_list(self, client_with_data):
        """top_drivers for each zone is a list."""
        r = client_with_data.get("/v1/explain/zones")
        for zone in r.json()["zones"]:
            assert isinstance(zone["top_drivers"], list)

    def test_top_driver_keys(self, client_with_data):
        """Each top_driver has feature and mean_abs_shap keys."""
        r = client_with_data.get("/v1/explain/zones")
        for zone in r.json()["zones"]:
            for td in zone["top_drivers"]:
                assert "feature" in td
                assert "mean_abs_shap" in td

    def test_503_when_no_predictions(self, client_no_data):
        """Returns 503 when predictions cache is empty."""
        r = client_no_data.get("/v1/explain/zones")
        assert r.status_code == 503

    def test_zones_without_shap_still_returns_200(self):
        """When no cells have shap_values, zones endpoint returns gracefully."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(explain_router)

        cell_ids = ["CELL_A", "CELL_B"]
        predictions = {}
        for i, cid in enumerate(cell_ids):
            p = _make_prediction(cid, score=0.3 + i * 0.1)
            p["shap_values"] = None
            predictions[cid] = p
        grid_cells = _make_grid_cells(*cell_ids)
        set_predictions_cache(predictions, grid_cells, "2026-03-15T14:00:00+00:00")

        c = TestClient(app)
        r = c.get("/v1/explain/zones")
        assert r.status_code == 200
