"""Tests for lightning pipeline."""

from datetime import date

import numpy as np

from infernis.pipelines.lightning_pipeline import LightningPipeline


class TestLightningPipeline:
    def test_aggregate_to_grid_empty(self):
        lp = LightningPipeline()
        grid_lats = np.array([50.0, 51.0, 52.0])
        grid_lons = np.array([-120.0, -121.0, -122.0])

        counts = lp._aggregate_to_grid([], grid_lats, grid_lons)
        assert counts.shape == (3,)
        assert np.all(counts == 0)

    def test_aggregate_to_grid_with_flashes(self):
        lp = LightningPipeline()
        grid_lats = np.array([50.0, 51.0, 52.0])
        grid_lons = np.array([-120.0, -121.0, -122.0])

        # Flashes very close to grid cells (within max_dist for any resolution)
        flashes = [
            {"lat": 50.001, "lon": -120.001, "strength_kA": 15.0},
            {"lat": 50.002, "lon": -119.999, "strength_kA": 20.0},
            {"lat": 51.001, "lon": -121.001, "strength_kA": 10.0},
        ]

        counts = lp._aggregate_to_grid(flashes, grid_lats, grid_lons)
        assert counts[0] == 2  # Two flashes near first cell
        assert counts[1] == 1  # One flash near second cell
        assert counts[2] == 0  # No flashes near third cell

    def test_aggregate_far_flash_excluded(self):
        lp = LightningPipeline()
        grid_lats = np.array([50.0])
        grid_lons = np.array([-120.0])

        flashes = [
            {"lat": 55.0, "lon": -115.0, "strength_kA": 30.0},
        ]

        counts = lp._aggregate_to_grid(flashes, grid_lats, grid_lons)
        assert counts[0] == 0  # Too far from any cell

    def test_parse_cldn_csv(self):
        lp = LightningPipeline()
        csv = "50.123,-120.456,15.3,1,1\n51.0,-121.5,20.0,2,1\n"
        result = lp._parse_cldn_csv(csv, date(2025, 7, 15), 12)
        assert len(result) == 2
        assert result[0]["lat"] == 50.123
        assert result[0]["lon"] == -120.456
        assert result[0]["strength_kA"] == 15.3

    def test_parse_cldn_csv_skips_headers(self):
        lp = LightningPipeline()
        csv = "# comment\nlatitude,longitude,strength\n50.0,-120.0,10.0\n"
        result = lp._parse_cldn_csv(csv, date(2025, 7, 15), 12)
        assert len(result) == 1

    def test_fetch_returns_zeros_on_error(self):
        from unittest.mock import patch

        import httpx

        lp = LightningPipeline()
        grid_lats = np.array([50.0, 51.0])
        grid_lons = np.array([-120.0, -121.0])

        # Mock HTTP client to fail immediately (no real network calls)
        with patch.object(lp._client, "get", side_effect=httpx.ConnectError("mocked")):
            result = lp.fetch_lightning_density(grid_lats, grid_lons, date(2000, 1, 1))
        assert result["lightning_24h"].shape == (2,)
        assert result["lightning_72h"].shape == (2,)
