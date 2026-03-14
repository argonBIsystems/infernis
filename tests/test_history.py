"""Tests for historical risk endpoint."""

import os

os.environ["INFERNIS_DEBUG"] = "true"

from fastapi.testclient import TestClient

from infernis.api.routes import set_predictions_cache
from infernis.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestHistoryEndpoint:
    def setup_method(self):
        predictions = {
            "BC-5K-0000001": {
                "score": 0.3,
                "level": "MODERATE",
                "timestamp": "2026-03-14T00:00:00Z",
            },
        }
        grid_cells = {
            "BC-5K-0000001": {"lat": 49.25, "lon": -123.1, "bec_zone": "CWH", "fuel_type": "C5"},
        }
        set_predictions_cache(predictions, grid_cells, "2026-03-14T00:00:00Z")

    def test_history_returns_structure(self):
        r = client.get("/v1/risk/history/49.25/-123.1")
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            data = r.json()
            assert "history" in data
            assert "count" in data
            assert "cell_id" in data

    def test_history_invalid_coords(self):
        r = client.get("/v1/risk/history/40.0/-123.0")
        assert r.status_code == 422

    def test_history_days_param(self):
        r = client.get("/v1/risk/history/49.25/-123.1?days=7")
        assert r.status_code in (200, 503)

    def test_history_max_90_days(self):
        r = client.get("/v1/risk/history/49.25/-123.1?days=91")
        assert r.status_code == 422
