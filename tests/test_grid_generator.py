"""Tests for BC grid generator."""


class TestGenerateBCGrid:
    def test_generates_cells(self):
        """Grid at coarse resolution should produce cells."""
        from infernis.grid.generator import generate_bc_grid

        # Use very coarse resolution for fast tests
        grid = generate_bc_grid(resolution_km=100.0)
        assert len(grid) > 0
        assert "cell_id" in grid.columns
        assert "lat" in grid.columns
        assert "lon" in grid.columns
        assert "geometry" in grid.columns

    def test_cell_id_format(self):
        from infernis.grid.generator import generate_bc_grid

        grid = generate_bc_grid(resolution_km=100.0)
        for cid in grid["cell_id"]:
            assert cid.startswith("BC-100K-")
            parts = cid.split("-")
            assert len(parts) == 3
            assert len(parts[2]) == 7  # zero-padded

    def test_cells_within_bc(self):
        """All cell centroids should be within reasonable BC bounds."""
        from infernis.grid.generator import generate_bc_grid

        grid = generate_bc_grid(resolution_km=100.0)
        assert grid["lat"].min() > 48.0
        assert grid["lat"].max() < 61.0
        assert grid["lon"].min() > -140.0
        assert grid["lon"].max() < -113.0

    def test_crs_is_bc_albers(self):
        from infernis.grid.generator import generate_bc_grid

        grid = generate_bc_grid(resolution_km=100.0)
        assert grid.crs.to_epsg() == 3005
