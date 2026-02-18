"""Tests for REST API endpoints."""

import os

# Enable debug mode for tests (skips API key auth)
# Must be set before importing infernis modules
os.environ["INFERNIS_DEBUG"] = "true"

import pytest
from fastapi.testclient import TestClient

from infernis.api.routes import set_predictions_cache
from infernis.main import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def populate_cache():
    """Populate the prediction cache for testing."""
    predictions = {
        "BC-5K-000000": {
            "score": 0.45,
            "level": "HIGH",
            "timestamp": "2025-07-15T20:00:00+00:00",
            "ffmc": 88.5,
            "dmc": 45.2,
            "dc": 320.1,
            "isi": 6.3,
            "bui": 52.0,
            "fwi": 15.8,
            "temperature_c": 28.0,
            "rh_pct": 25.0,
            "wind_kmh": 15.0,
            "precip_24h_mm": 0.0,
            "soil_moisture": 0.18,
            "ndvi": 0.55,
            "snow_cover": False,
            "next_update": "2025-07-16T21:00:00Z",
        },
        "BC-5K-000001": {
            "score": 0.72,
            "level": "VERY_HIGH",
            "timestamp": "2025-07-15T20:00:00+00:00",
            "ffmc": 92.0,
            "dmc": 80.0,
            "dc": 450.0,
            "isi": 12.0,
            "bui": 90.0,
            "fwi": 28.0,
            "temperature_c": 35.0,
            "rh_pct": 12.0,
            "wind_kmh": 25.0,
            "precip_24h_mm": 0.0,
            "soil_moisture": 0.10,
            "ndvi": 0.40,
            "snow_cover": False,
            "next_update": "2025-07-16T21:00:00Z",
        },
    }
    grid_cells = {
        "BC-5K-000000": {
            "lat": 50.0,
            "lon": -122.0,
            "bec_zone": "IDF",
            "fuel_type": "C3",
            "elevation_m": 500,
        },
        "BC-5K-000001": {
            "lat": 51.5,
            "lon": -120.0,
            "bec_zone": "SBPS",
            "fuel_type": "C3",
            "elevation_m": 900,
        },
    }
    set_predictions_cache(predictions, grid_cells, "2025-07-15T20:00:00Z")


class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_has_redis_status(self, client):
        r = client.get("/health")
        assert "redis" in r.json()


class TestRiskEndpoint:
    def test_get_risk_valid(self, client):
        r = client.get("/v1/risk/50.0/-122.0")
        assert r.status_code == 200
        data = r.json()
        assert data["grid_cell_id"] == "BC-5K-000000"
        assert 0.0 <= data["risk"]["score"] <= 1.0
        assert data["fwi"]["ffmc"] == 88.5

    def test_get_risk_outside_bc(self, client):
        r = client.get("/v1/risk/40.0/-122.0")
        assert r.status_code == 422

    def test_get_risk_nearest_cell(self, client):
        """Nearby coordinates should map to the same cell."""
        r = client.get("/v1/risk/50.01/-121.99")
        assert r.status_code == 200
        assert r.json()["grid_cell_id"] == "BC-5K-000000"


class TestFWIEndpoint:
    def test_get_fwi(self, client):
        r = client.get("/v1/fwi/50.0/-122.0")
        assert r.status_code == 200
        data = r.json()
        assert data["fwi"]["ffmc"] == 88.5
        assert data["fwi"]["fwi"] == 15.8


class TestConditionsEndpoint:
    def test_get_conditions(self, client):
        r = client.get("/v1/conditions/50.0/-122.0")
        assert r.status_code == 200
        data = r.json()
        assert data["conditions"]["temperature_c"] == 28.0
        assert data["conditions"]["snow_cover"] is False


class TestStatusEndpoint:
    def test_status_operational(self, client):
        r = client.get("/v1/status")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "operational"
        assert data["pipeline_healthy"] is True


class TestCoverageEndpoint:
    def test_coverage(self, client):
        r = client.get("/v1/coverage")
        assert r.status_code == 200
        data = r.json()
        assert data["province"] == "British Columbia"
        assert data["grid"]["total_cells"] == 2


class TestZonesEndpoint:
    def test_zones(self, client):
        r = client.get("/v1/risk/zones")
        assert r.status_code == 200
        data = r.json()
        assert len(data["zones"]) == 2

    def test_zones_have_high_risk_count(self, client):
        r = client.get("/v1/risk/zones")
        data = r.json()
        sbps = [z for z in data["zones"] if z["bec_zone"] == "SBPS"][0]
        assert sbps["high_risk_cells"] == 1


class TestGridEndpoint:
    def test_grid_full_bbox(self, client):
        r = client.get("/v1/risk/grid?bbox=48.0,-125.0,53.0,-118.0")
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "FeatureCollection"
        assert data["metadata"]["cell_count"] == 2

    def test_grid_partial_bbox(self, client):
        r = client.get("/v1/risk/grid?bbox=49.5,-123.0,50.5,-121.0")
        assert r.status_code == 200
        data = r.json()
        assert data["metadata"]["cell_count"] == 1
        assert data["features"][0]["properties"]["cell_id"] == "BC-5K-000000"

    def test_grid_empty_bbox(self, client):
        r = client.get("/v1/risk/grid?bbox=55.0,-130.0,56.0,-129.0")
        assert r.status_code == 200
        assert r.json()["metadata"]["cell_count"] == 0

    def test_grid_level_filter(self, client):
        r = client.get("/v1/risk/grid?bbox=48.0,-125.0,53.0,-118.0&level=VERY_HIGH")
        assert r.status_code == 200
        data = r.json()
        assert data["metadata"]["cell_count"] == 1
        assert data["features"][0]["properties"]["level"] == "VERY_HIGH"

    def test_grid_bad_bbox(self, client):
        r = client.get("/v1/risk/grid?bbox=invalid")
        assert r.status_code == 422


class TestHeatmapEndpoint:
    def test_heatmap_returns_png(self, client):
        r = client.get("/v1/risk/heatmap?bbox=49.5,-123.0,52.0,-119.0")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        # PNG magic bytes
        assert r.content[:4] == b"\x89PNG"

    def test_heatmap_bad_bbox(self, client):
        r = client.get("/v1/risk/heatmap?bbox=invalid")
        assert r.status_code == 422

    def test_heatmap_custom_size(self, client):
        r = client.get("/v1/risk/heatmap?bbox=49.5,-123.0,52.0,-119.0&width=128&height=128")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"


class TestHistoryEndpoint:
    def test_history_returns_empty(self, client):
        r = client.get("/v1/history/50.0/-122.0")
        assert r.status_code == 200
        data = r.json()
        assert data["fires"] == []
        assert data["total_fires"] == 0

    def test_history_outside_bc(self, client):
        r = client.get("/v1/history/40.0/-122.0")
        assert r.status_code == 422
