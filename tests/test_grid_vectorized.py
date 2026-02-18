"""Tests for vectorized grid generation and parquet roundtrip."""

from __future__ import annotations

import os
import re
import tempfile

import numpy as np
import pandas as pd
import pytest


class TestVectorizedGridGenerator:
    """Test the vectorized grid generator at various resolutions."""

    def test_coarse_grid_cell_count(self):
        """100km grid should produce a small but nonzero set of cells."""
        from infernis.grid.generator import generate_bc_grid

        grid = generate_bc_grid(resolution_km=100.0)
        assert len(grid) > 5
        assert len(grid) < 250

    def test_5km_regression(self):
        """5km grid should produce roughly 84K cells (regression check)."""
        from infernis.grid.generator import generate_bc_grid

        grid = generate_bc_grid(resolution_km=5.0)
        assert 80_000 < len(grid) < 90_000

    def test_cell_ids_unique(self):
        from infernis.grid.generator import generate_bc_grid

        grid = generate_bc_grid(resolution_km=50.0)
        assert grid["cell_id"].is_unique

    def test_cell_id_format_7_digits(self):
        from infernis.grid.generator import generate_bc_grid

        grid = generate_bc_grid(resolution_km=50.0)
        pattern = re.compile(r"^BC-50K-\d{7}$")
        for cid in grid["cell_id"]:
            assert pattern.match(cid), f"Bad cell_id format: {cid}"

    def test_all_cells_within_bc_bounds(self):
        """All WGS84 centroids should be within reasonable BC extent."""
        from infernis.grid.generator import generate_bc_grid

        grid = generate_bc_grid(resolution_km=50.0)
        assert grid["lat"].min() > 48.0
        assert grid["lat"].max() < 61.0
        assert grid["lon"].min() > -140.0
        assert grid["lon"].max() < -113.0

    def test_crs_is_bc_albers(self):
        from infernis.grid.generator import generate_bc_grid

        grid = generate_bc_grid(resolution_km=50.0)
        assert grid.crs.to_epsg() == 3005

    def test_has_required_columns(self):
        from infernis.grid.generator import generate_bc_grid

        grid = generate_bc_grid(resolution_km=50.0)
        for col in ["cell_id", "lat", "lon", "geometry"]:
            assert col in grid.columns, f"Missing column: {col}"


class TestVectorizedBECZones:
    """Test vectorized BEC zone assignment."""

    def test_no_null_zones(self):
        """Every cell should get a BEC zone assigned."""
        from infernis.grid.generator import generate_bc_grid
        from infernis.grid.initializer import _populate_bec_zones

        grid = generate_bc_grid(resolution_km=50.0)
        grid["elevation_m"] = np.zeros(len(grid))
        result = _populate_bec_zones(grid)
        assert result["bec_zone"].notna().all()
        assert (result["bec_zone"] != "").all()

    def test_vectorized_matches_known_zones(self):
        """Spot-check a few known locations."""
        from infernis.grid.initializer import _populate_bec_zones

        df = pd.DataFrame(
            {
                "lat": [58.5, 49.2, 51.0],
                "lon": [-130.0, -116.0, -120.0],
                "elevation_m": [500.0, 400.0, 2000.0],
            }
        )
        result = _populate_bec_zones(df)
        assert result.iloc[0]["bec_zone"] == "BWBS"  # far north
        assert result.iloc[1]["bec_zone"] == "PP"  # dry south-east low elev
        assert result.iloc[2]["bec_zone"] == "AT"  # high elevation

    def test_fuel_types_all_assigned(self):
        from infernis.grid.generator import generate_bc_grid
        from infernis.grid.initializer import _populate_bec_zones, _populate_fuel_types

        grid = generate_bc_grid(resolution_km=50.0)
        grid["elevation_m"] = np.zeros(len(grid))
        grid = _populate_bec_zones(grid)
        grid = _populate_fuel_types(grid)
        assert grid["fuel_type"].notna().all()
        assert (grid["fuel_type"] != "").all()


class TestParquetRoundtrip:
    """Test saving and loading grids via parquet."""

    def test_roundtrip_preserves_data(self):
        from infernis.grid.generator import generate_bc_grid
        from infernis.grid.initializer import (
            _populate_bec_zones,
            _populate_fuel_types,
            load_grid_from_parquet,
            save_grid_to_parquet,
        )

        grid = generate_bc_grid(resolution_km=100.0)
        grid["elevation_m"] = np.zeros(len(grid))
        grid = _populate_bec_zones(grid)
        grid = _populate_fuel_types(grid)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        try:
            save_grid_to_parquet(grid, path)
            loaded = load_grid_from_parquet(path)

            assert len(loaded) == len(grid)
            assert set(loaded.columns) >= {"cell_id", "lat", "lon", "bec_zone", "fuel_type"}
            assert list(loaded["cell_id"]) == list(grid["cell_id"])
            np.testing.assert_array_almost_equal(loaded["lat"].values, grid["lat"].values)
            np.testing.assert_array_almost_equal(loaded["lon"].values, grid["lon"].values)
        finally:
            os.unlink(path)

    def test_save_rejects_non_geodataframe(self):
        from infernis.grid.initializer import save_grid_to_parquet

        with pytest.raises(TypeError):
            save_grid_to_parquet(pd.DataFrame({"x": [1]}), "/tmp/bad.parquet")
