"""Tests for nearby fires endpoint."""

import os

os.environ["INFERNIS_DEBUG"] = "true"

from fastapi.testclient import TestClient

from infernis.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestNearbyFires:
    def test_fires_returns_structure(self):
        r = client.get("/v1/fires/near/50.67/-120.33")
        assert r.status_code == 200
        data = r.json()
        assert "fires" in data
        assert "radius_km" in data
        assert "count" in data

    def test_fires_custom_radius(self):
        r = client.get("/v1/fires/near/50.67/-120.33?radius_km=100")
        assert r.status_code == 200

    def test_fires_invalid_coords(self):
        r = client.get("/v1/fires/near/40.0/-120.0")
        assert r.status_code == 422

    def test_fires_max_radius(self):
        r = client.get("/v1/fires/near/50.67/-120.33?radius_km=501")
        assert r.status_code == 422
