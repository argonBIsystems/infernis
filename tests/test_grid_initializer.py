"""Tests for grid initializer (BEC zone and fuel type assignment)."""

import pandas as pd
import pytest

from infernis.grid.initializer import _populate_bec_zones, _populate_fuel_types


@pytest.fixture
def sample_grid():
    """Grid with diverse locations across BC."""
    return pd.DataFrame(
        {
            "cell_id": [f"BC-5K-{i:07d}" for i in range(6)],
            "lat": [49.2, 50.5, 53.0, 58.0, 51.0, 49.0],
            "lon": [-123.5, -122.0, -125.0, -130.0, -120.0, -116.0],
            "elevation_m": [200.0, 800.0, 1500.0, 500.0, 2000.0, 400.0],
        }
    )


class TestBECZoneAssignment:
    def test_assigns_zones(self, sample_grid):
        result = _populate_bec_zones(sample_grid)
        assert "bec_zone" in result.columns
        assert len(result) == 6
        assert all(z != "" for z in result["bec_zone"])

    def test_northern_bc_gets_bwbs(self, sample_grid):
        result = _populate_bec_zones(sample_grid)
        # lat=58.0 should be BWBS
        northern = result[result["lat"] == 58.0].iloc[0]
        assert northern["bec_zone"] == "BWBS"

    def test_high_elevation_gets_alpine(self, sample_grid):
        result = _populate_bec_zones(sample_grid)
        # elev=2000 should be AT (alpine tundra)
        alpine = result[result["elevation_m"] == 2000.0].iloc[0]
        assert alpine["bec_zone"] == "AT"


class TestFuelTypeAssignment:
    def test_assigns_fuel_types(self, sample_grid):
        grid = _populate_bec_zones(sample_grid)
        result = _populate_fuel_types(grid)
        assert "fuel_type" in result.columns
        assert all(ft != "" for ft in result["fuel_type"])

    def test_alpine_is_nonfuel(self, sample_grid):
        grid = _populate_bec_zones(sample_grid)
        result = _populate_fuel_types(grid)
        alpine = result[result["bec_zone"] == "AT"]
        if len(alpine) > 0:
            assert alpine.iloc[0]["fuel_type"] == "NF"
