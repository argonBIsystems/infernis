"""Tests for batch risk endpoint."""

import os

os.environ["INFERNIS_DEBUG"] = "true"

from fastapi.testclient import TestClient

from infernis.api.routes import set_predictions_cache
from infernis.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestBatchRisk:
    def setup_method(self):
        predictions = {
            "BC-5K-0000001": {
                "score": 0.3,
                "level": "MODERATE",
                "timestamp": "2026-03-14T00:00:00Z",
                "ffmc": 86.0,
                "dmc": 22.0,
                "dc": 38.0,
                "isi": 4.0,
                "bui": 22.0,
                "fwi": 7.0,
                "temperature_c": 3.2,
                "rh_pct": 40.0,
                "wind_kmh": 9.0,
                "precip_24h_mm": 0.0,
                "soil_moisture": 0.37,
                "ndvi": 0.31,
                "snow_cover": False,
            },
        }
        grid_cells = {
            "BC-5K-0000001": {
                "lat": 49.25,
                "lon": -123.1,
                "bec_zone": "CWH",
                "fuel_type": "C5",
                "elevation_m": 100,
            },
        }
        set_predictions_cache(predictions, grid_cells, "2026-03-14T00:00:00Z")

    def test_batch_returns_results(self):
        r = client.post("/v1/risk/batch", json={"locations": [{"lat": 49.25, "lon": -123.1}]})
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) == 1
        assert data["count"] == 1
        assert "risk" in data["results"][0]

    def test_batch_max_50_locations(self):
        locs = [{"lat": 49.0 + i * 0.01, "lon": -123.0} for i in range(51)]
        r = client.post("/v1/risk/batch", json={"locations": locs})
        assert r.status_code == 422

    def test_batch_empty_locations(self):
        r = client.post("/v1/risk/batch", json={"locations": []})
        assert r.status_code == 422

    def test_batch_multiple_locations(self):
        r = client.post(
            "/v1/risk/batch",
            json={
                "locations": [
                    {"lat": 49.25, "lon": -123.1},
                    {"lat": 49.25, "lon": -123.1},
                ]
            },
        )
        assert r.status_code == 200
        assert r.json()["count"] == 2
