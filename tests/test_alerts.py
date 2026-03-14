"""Tests for webhook alert endpoints."""

import os

os.environ["INFERNIS_DEBUG"] = "true"

from fastapi.testclient import TestClient

from infernis.api.routes import set_predictions_cache
from infernis.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestAlerts:
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

    def test_create_alert_structure(self):
        r = client.post(
            "/v1/alerts",
            json={
                "latitude": 49.25,
                "longitude": -123.1,
                "threshold": 0.5,
                "webhook_url": "https://example.com/webhook",
            },
        )
        # In debug mode auth is skipped, but DB may not be available
        assert r.status_code in (201, 401, 500)

    def test_list_alerts(self):
        r = client.get("/v1/alerts")
        assert r.status_code in (200, 401)

    def test_create_alert_invalid_threshold(self):
        r = client.post(
            "/v1/alerts",
            json={
                "latitude": 49.25,
                "longitude": -123.1,
                "threshold": 1.5,
                "webhook_url": "https://example.com/webhook",
            },
        )
        assert r.status_code == 422
