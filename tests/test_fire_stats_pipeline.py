"""Tests for offline fire statistics spatial computation pipeline.

Only the pure helper functions are tested here (no DB or Redis required).
The full run_fire_stats_pipeline() integration path is exercised by the
admin CLI and manual QA.
"""

import pytest

from infernis.services.fire_stats_pipeline import compute_distance_km, match_fires_to_cell


# ---------------------------------------------------------------------------
# compute_distance_km — Haversine
# ---------------------------------------------------------------------------


class TestComputeDistanceKm:
    def test_same_point_is_zero(self):
        dist = compute_distance_km(49.2827, -123.1207, 49.2827, -123.1207)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_vancouver_to_kamloops(self):
        # Vancouver: 49.2827°N, 123.1207°W
        # Kamloops:  50.6745°N, 120.3273°W
        # Great-circle (as-the-crow-flies) distance ~250km
        dist = compute_distance_km(49.2827, -123.1207, 50.6745, -120.3273)
        assert 240.0 < dist < 270.0

    def test_symmetry(self):
        d1 = compute_distance_km(49.0, -122.0, 50.0, -120.0)
        d2 = compute_distance_km(50.0, -120.0, 49.0, -122.0)
        assert d1 == pytest.approx(d2, rel=1e-9)

    def test_short_distance(self):
        # 0.01 degrees latitude ≈ ~1.1 km
        dist = compute_distance_km(50.0, -120.0, 50.01, -120.0)
        assert 1.0 < dist < 1.2

    def test_north_south_pole_distance(self):
        # Equator to pole should be ~10,000 km
        dist = compute_distance_km(0.0, 0.0, 90.0, 0.0)
        assert 9_900.0 < dist < 10_100.0


# ---------------------------------------------------------------------------
# match_fires_to_cell
# ---------------------------------------------------------------------------


class TestMatchFiresToCell:
    def _make_fire(self, lat, lon, year=2020, size_ha=100.0, cause="lightning"):
        return {"lat": lat, "lon": lon, "year": year, "size_ha": size_ha, "cause": cause}

    def test_fire_within_radius_included(self):
        fires = [self._make_fire(50.0, -120.0)]
        # Cell is at same location — distance ~0
        matched = match_fires_to_cell(50.0, -120.0, fires, radius_km=10.0)
        assert len(matched) == 1

    def test_fire_outside_radius_excluded(self):
        fires = [self._make_fire(51.0, -120.0)]  # ~110km north
        matched = match_fires_to_cell(50.0, -120.0, fires, radius_km=10.0)
        assert len(matched) == 0

    def test_fire_exactly_at_boundary(self):
        # Place a fire ~10km north of cell (1 degree latitude ≈ 111km, so 0.09 deg ≈ 10km)
        fires = [self._make_fire(50.09, -120.0)]
        dist = compute_distance_km(50.0, -120.0, 50.09, -120.0)
        radius = dist  # exactly on boundary
        matched = match_fires_to_cell(50.0, -120.0, fires, radius_km=radius)
        assert len(matched) == 1

    def test_distance_km_field_added(self):
        fires = [self._make_fire(50.05, -120.0)]
        matched = match_fires_to_cell(50.0, -120.0, fires, radius_km=20.0)
        assert len(matched) == 1
        assert "distance_km" in matched[0]
        expected = compute_distance_km(50.0, -120.0, 50.05, -120.0)
        assert matched[0]["distance_km"] == pytest.approx(expected, rel=1e-9)

    def test_original_fire_fields_preserved(self):
        fires = [self._make_fire(50.01, -120.0, year=2015, size_ha=250.0, cause="human")]
        matched = match_fires_to_cell(50.0, -120.0, fires, radius_km=5.0)
        assert len(matched) == 1
        f = matched[0]
        assert f["year"] == 2015
        assert f["size_ha"] == pytest.approx(250.0)
        assert f["cause"] == "human"

    def test_empty_fires_list(self):
        matched = match_fires_to_cell(50.0, -120.0, [], radius_km=10.0)
        assert matched == []

    def test_multiple_fires_mixed_radius(self):
        fires = [
            self._make_fire(50.01, -120.0),  # ~1.1km — inside 10km
            self._make_fire(50.05, -120.0),  # ~5.5km — inside 10km
            self._make_fire(50.2, -120.0),  # ~22km — outside 10km
            self._make_fire(50.15, -120.0),  # ~16km — outside 10km
        ]
        matched = match_fires_to_cell(50.0, -120.0, fires, radius_km=10.0)
        assert len(matched) == 2

    def test_default_radius_is_10km(self):
        import inspect

        sig = inspect.signature(match_fires_to_cell)
        assert sig.parameters["radius_km"].default == 10.0
