"""Tests for map tile endpoints."""

import os

os.environ["INFERNIS_DEBUG"] = "true"

from fastapi.testclient import TestClient

from infernis.api.routes import set_predictions_cache
from infernis.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestTileEndpoint:
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

    def test_tile_returns_png(self):
        r = client.get("/v1/tiles/6/10/22.png")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"

    def test_tile_has_cache_header(self):
        r = client.get("/v1/tiles/6/10/22.png")
        assert "cache-control" in r.headers

    def test_tile_invalid_zoom(self):
        r = client.get("/v1/tiles/20/0/0.png")
        assert r.status_code == 422

    def test_tile_256x256(self):
        r = client.get("/v1/tiles/6/10/22.png")
        if r.status_code == 200:
            from io import BytesIO

            from PIL import Image

            img = Image.open(BytesIO(r.content))
            assert img.size == (256, 256)
