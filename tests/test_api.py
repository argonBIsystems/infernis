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
        assert r.status_code in (200, 503)
        assert r.json()["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_arsite_spec(self, client):
        r = client.get("/health")
        data = r.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "checks" in data
        assert isinstance(data["checks"], dict)


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


class TestCHainesInAPI:
    def test_risk_response_has_c_haines_field(self, client):
        """Risk response conditions include c_haines (may be None when unavailable)."""
        r = client.get("/v1/risk/50.0/-122.0")
        assert r.status_code == 200
        data = r.json()
        assert "c_haines" in data["conditions"]

    def test_c_haines_none_when_not_in_prediction(self, client):
        """c_haines is None when prediction dict has no pressure-level data."""
        r = client.get("/v1/risk/50.0/-122.0")
        data = r.json()
        # Test fixture doesn't include c_haines, so it should be None
        assert data["conditions"]["c_haines"] is None

    def test_c_haines_float_when_present(self, client):
        """c_haines is a float in [0, 13] when present in prediction."""
        from infernis.api.routes import set_predictions_cache

        preds_with_chaines = {
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
                "c_haines": 9.4,
                "next_update": "",
            }
        }
        grid = {
            "BC-5K-000000": {
                "lat": 50.0,
                "lon": -122.0,
                "bec_zone": "IDF",
                "fuel_type": "C3",
                "elevation_m": 500,
            },
        }
        set_predictions_cache(preds_with_chaines, grid, "2025-07-15T20:00:00Z")
        r = client.get("/v1/risk/50.0/-122.0")
        assert r.status_code == 200
        data = r.json()
        ch = data["conditions"]["c_haines"]
        assert ch == 9.4
        assert 0.0 <= ch <= 13.0

    def test_conditions_endpoint_has_c_haines(self, client):
        """/conditions endpoint also surfaces c_haines."""
        r = client.get("/v1/conditions/50.0/-122.0")
        assert r.status_code == 200
        data = r.json()
        assert "c_haines" in data["conditions"]


class TestConfidenceIntervalInAPI:
    def test_risk_response_has_confidence_interval_field(self, client):
        """Risk response must include confidence_interval (may be None)."""
        r = client.get("/v1/risk/50.0/-122.0")
        assert r.status_code == 200
        data = r.json()
        assert "confidence_interval" in data

    def test_confidence_interval_none_when_not_in_prediction(self, client):
        """confidence_interval is None when prediction dict has no CI data."""
        r = client.get("/v1/risk/50.0/-122.0")
        data = r.json()
        # Test fixture doesn't include confidence_interval, so it should be None
        assert data["confidence_interval"] is None

    def test_confidence_interval_present_when_provided(self, client):
        """confidence_interval is populated when prediction includes CI dict."""
        from infernis.api.routes import set_predictions_cache

        preds_with_ci = {
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
                "confidence_interval": {"lower": 0.32, "upper": 0.58, "level": 0.90},
                "next_update": "",
            }
        }
        grid = {
            "BC-5K-000000": {
                "lat": 50.0,
                "lon": -122.0,
                "bec_zone": "IDF",
                "fuel_type": "C3",
                "elevation_m": 500,
            },
        }
        set_predictions_cache(preds_with_ci, grid, "2025-07-15T20:00:00Z")

        r = client.get("/v1/risk/50.0/-122.0")
        assert r.status_code == 200
        data = r.json()
        ci = data["confidence_interval"]
        assert ci is not None
        assert ci["lower"] == 0.32
        assert ci["upper"] == 0.58
        assert ci["level"] == 0.90

    def test_confidence_interval_lower_leq_upper(self, client):
        """Lower bound must be <= upper bound in API response."""
        from infernis.api.routes import set_predictions_cache

        preds_with_ci = {
            "BC-5K-000000": {
                "score": 0.60,
                "level": "VERY_HIGH",
                "timestamp": "2025-07-15T20:00:00+00:00",
                "ffmc": 90.0,
                "dmc": 60.0,
                "dc": 350.0,
                "isi": 9.0,
                "bui": 70.0,
                "fwi": 22.0,
                "temperature_c": 32.0,
                "rh_pct": 18.0,
                "wind_kmh": 20.0,
                "precip_24h_mm": 0.0,
                "soil_moisture": 0.12,
                "ndvi": 0.40,
                "snow_cover": False,
                "confidence_interval": {"lower": 0.45, "upper": 0.75, "level": 0.90},
                "next_update": "",
            }
        }
        grid = {
            "BC-5K-000000": {
                "lat": 50.0,
                "lon": -122.0,
                "bec_zone": "IDF",
                "fuel_type": "C3",
                "elevation_m": 500,
            },
        }
        set_predictions_cache(preds_with_ci, grid, "2025-07-15T20:00:00Z")

        r = client.get("/v1/risk/50.0/-122.0")
        ci = r.json()["confidence_interval"]
        assert ci["lower"] <= ci["upper"]


class TestFireBehaviourInAPI:
    """Tests for fire_behaviour field in the /risk endpoint response."""

    def test_risk_response_has_fire_behaviour_field(self, client):
        """Risk response must include fire_behaviour key (may be None)."""
        r = client.get("/v1/risk/50.0/-122.0")
        assert r.status_code == 200
        data = r.json()
        assert "fire_behaviour" in data

    def test_fire_behaviour_none_when_not_in_prediction(self, client):
        """fire_behaviour is None when prediction dict has no FBP data."""
        r = client.get("/v1/risk/50.0/-122.0")
        data = r.json()
        # Default test fixture has no fire_behaviour key → None
        assert data["fire_behaviour"] is None

    def test_fire_behaviour_populated_when_provided(self, client):
        """fire_behaviour is a full object when prediction includes FBP dict."""
        from infernis.api.routes import set_predictions_cache

        preds_with_fb = {
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
                "next_update": "",
                "fire_behaviour": {
                    "rate_of_spread_mpm": 12.5,
                    "head_fire_intensity_kwm": 8500.0,
                    "fire_type": "intermittent_crown",
                    "crown_fraction_burned": 0.45,
                    "flame_length_m": 18.2,
                },
            }
        }
        grid = {
            "BC-5K-000000": {
                "lat": 50.0,
                "lon": -122.0,
                "bec_zone": "IDF",
                "fuel_type": "C3",
                "elevation_m": 500,
            },
        }
        set_predictions_cache(preds_with_fb, grid, "2025-07-15T20:00:00Z")

        r = client.get("/v1/risk/50.0/-122.0")
        assert r.status_code == 200
        data = r.json()
        fb = data["fire_behaviour"]
        assert fb is not None
        assert fb["rate_of_spread_mpm"] == 12.5
        assert fb["head_fire_intensity_kwm"] == 8500.0
        assert fb["fire_type"] == "intermittent_crown"
        assert fb["crown_fraction_burned"] == 0.45
        assert fb["flame_length_m"] == 18.2

    def test_fire_behaviour_has_all_required_keys_when_present(self, client):
        """When fire_behaviour is present, all 5 fields must be included."""
        from infernis.api.routes import set_predictions_cache

        preds = {
            "BC-5K-000000": {
                "score": 0.5,
                "level": "HIGH",
                "timestamp": "2025-07-15T20:00:00+00:00",
                "ffmc": 90.0,
                "dmc": 50.0,
                "dc": 300.0,
                "isi": 8.0,
                "bui": 70.0,
                "fwi": 20.0,
                "temperature_c": 30.0,
                "rh_pct": 20.0,
                "wind_kmh": 20.0,
                "precip_24h_mm": 0.0,
                "soil_moisture": 0.15,
                "ndvi": 0.45,
                "snow_cover": False,
                "next_update": "",
                "fire_behaviour": {
                    "rate_of_spread_mpm": 5.0,
                    "head_fire_intensity_kwm": 2000.0,
                    "fire_type": "surface",
                    "crown_fraction_burned": 0.0,
                    "flame_length_m": 6.1,
                },
            }
        }
        grid = {
            "BC-5K-000000": {
                "lat": 50.0,
                "lon": -122.0,
                "bec_zone": "IDF",
                "fuel_type": "C3",
                "elevation_m": 500,
            },
        }
        set_predictions_cache(preds, grid, "2025-07-15T20:00:00Z")

        r = client.get("/v1/risk/50.0/-122.0")
        fb = r.json()["fire_behaviour"]
        assert fb is not None
        required_keys = {
            "rate_of_spread_mpm",
            "head_fire_intensity_kwm",
            "fire_type",
            "crown_fraction_burned",
            "flame_length_m",
        }
        assert required_keys.issubset(fb.keys())


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


class TestDemoEndpoints:
    def test_demo_risk_returns_all_levels(self, client):
        r = client.get("/v1/demo/risk")
        assert r.status_code == 200
        data = r.json()
        assert len(data["samples"]) == 6
        levels = [s["risk"]["level"] for s in data["samples"]]
        assert "VERY_LOW" in levels
        assert "EXTREME" in levels

    def test_demo_risk_by_level(self, client):
        r = client.get("/v1/demo/risk/high")
        assert r.status_code == 200
        data = r.json()
        assert data["risk"]["level"] == "HIGH"
        assert data["_demo"] is True

    def test_demo_risk_invalid_level(self, client):
        r = client.get("/v1/demo/risk/nonexistent")
        assert r.status_code == 404

    def test_demo_forecast(self, client):
        r = client.get("/v1/demo/forecast")
        assert r.status_code == 200
        data = r.json()
        assert len(data["forecast"]) == 10
        assert data["_demo"] is True

    def test_demo_risk_by_coords_snaps_to_nearest(self, client):
        r = client.get("/v1/demo/risk/50.5/-120.0")
        assert r.status_code == 200
        data = r.json()
        assert data["_demo"] is True
        assert data["risk"]["level"] in [
            "VERY_LOW",
            "LOW",
            "MODERATE",
            "HIGH",
            "VERY_HIGH",
            "EXTREME",
        ]
        assert "grid_cell_id" in data

    def test_demo_forecast_by_coords(self, client):
        r = client.get("/v1/demo/forecast/50.67/-120.33")
        assert r.status_code == 200
        data = r.json()
        assert data["_demo"] is True
        assert len(data["forecast"]) == 10

    def test_demo_fwi_by_coords(self, client):
        r = client.get("/v1/demo/fwi/50.67/-120.33")
        assert r.status_code == 200
        data = r.json()
        assert data["_demo"] is True
        assert "fwi" in data

    def test_demo_conditions_by_coords(self, client):
        r = client.get("/v1/demo/conditions/50.67/-120.33")
        assert r.status_code == 200
        data = r.json()
        assert data["_demo"] is True
        assert "conditions" in data

    def test_demo_risk_zones(self, client):
        r = client.get("/v1/demo/risk/zones")
        assert r.status_code == 200
        data = r.json()
        assert data["_demo"] is True
        assert len(data["zones"]) > 0
