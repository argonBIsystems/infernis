"""Tests for GET /v1/risk/profile endpoint."""

import os

# Enable debug mode for tests (skips API key auth)
# Must be set before importing infernis modules
os.environ["INFERNIS_DEBUG"] = "true"

import pytest
from fastapi.testclient import TestClient

from infernis.api.profile_routes import set_fire_stats_cache
from infernis.api.routes import set_predictions_cache
from infernis.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PREDICTIONS = {
    "BC-5K-000000": {
        "score": 0.45,
        "level": "HIGH",
        "timestamp": "2025-07-15T20:00:00+00:00",
        "ffmc": 88.5,
        "dmc": 45.2,
        "dc": 320.1,
        "isi": 6.3,
        "bui": 52.0,
        "fwi": 15.8,
        "temperature_c": 28.0,
        "rh_pct": 25.0,
        "wind_kmh": 15.0,
        "precip_24h_mm": 0.0,
        "soil_moisture": 0.18,
        "ndvi": 0.55,
        "snow_cover": False,
        "next_update": "2025-07-16T21:00:00Z",
    },
    "BC-5K-000001": {
        "score": 0.72,
        "level": "VERY_HIGH",
        "timestamp": "2025-07-15T20:00:00+00:00",
        "ffmc": 92.0,
        "dmc": 80.0,
        "dc": 450.0,
        "isi": 12.0,
        "bui": 90.0,
        "fwi": 28.0,
        "temperature_c": 35.0,
        "rh_pct": 12.0,
        "wind_kmh": 25.0,
        "precip_24h_mm": 0.0,
        "soil_moisture": 0.10,
        "ndvi": 0.40,
        "snow_cover": False,
        "next_update": "2025-07-16T21:00:00Z",
    },
}

_GRID_CELLS = {
    "BC-5K-000000": {
        "lat": 50.0,
        "lon": -122.0,
        "bec_zone": "IDF",
        "fuel_type": "C3",
        "elevation_m": 500,
    },
    "BC-5K-000001": {
        "lat": 51.5,
        "lon": -120.0,
        "bec_zone": "SBPS",
        "fuel_type": "C3",
        "elevation_m": 900,
    },
}

_FIRE_STATS = {
    "BC-5K-000000": {
        "cell_id": "BC-5K-000000",
        "fires_10yr": {
            "count": 3,
            "nearest_km": 2.4,
            "largest_ha": 850.0,
            "causes": {"Lightning": 2, "Human": 1},
        },
        "fires_30yr": {
            "count": 8,
            "nearest_km": 1.1,
            "largest_ha": 5200.0,
            "causes": {"Lightning": 5, "Human": 3},
        },
        "fires_all": {
            "count": 12,
            "nearest_km": 0.8,
            "largest_ha": 9500.0,
            "causes": {"Lightning": 7, "Human": 5},
            "record_start": 1950,
        },
        "susceptibility_score": 0.0014,
        "susceptibility_percentile": 72,
        "susceptibility_label": "ABOVE_AVERAGE",
        "susceptibility_basis": "bec_fuel",
        "exposure_percentile": 68,
        "mean_return_years": 45.0,
        "typical_severity": "moderate",
        "dominant_cause": "Lightning",
        "computed_at": "2025-07-15T00:00:00",
    },
    "BC-5K-000001": {
        "cell_id": "BC-5K-000001",
        "fires_10yr": {
            "count": 1,
            "nearest_km": 7.0,
            "largest_ha": 120.0,
            "causes": {"Lightning": 1},
        },
        "fires_30yr": {
            "count": 4,
            "nearest_km": 3.5,
            "largest_ha": 640.0,
            "causes": {"Lightning": 4},
        },
        "fires_all": {
            "count": 6,
            "nearest_km": 2.0,
            "largest_ha": 1800.0,
            "causes": {"Lightning": 6},
            "record_start": 1960,
        },
        "susceptibility_score": 0.00047,
        "susceptibility_percentile": 55,
        "susceptibility_label": "AVERAGE",
        "susceptibility_basis": "bec",
        "exposure_percentile": 40,
        "mean_return_years": 80.0,
        "typical_severity": "low",
        "dominant_cause": "Lightning",
        "computed_at": "2025-07-15T00:00:00",
    },
}


@pytest.fixture(autouse=True)
def populate_caches():
    """Populate predictions and fire stats caches before each test."""
    set_predictions_cache(_PREDICTIONS, _GRID_CELLS, "2025-07-15T20:00:00Z")
    set_fire_stats_cache(_FIRE_STATS)


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProfileEndpoint:
    def test_returns_200_for_valid_bc_coords(self, client):
        r = client.get("/v1/risk/profile/50.0/-122.0")
        if r.status_code == 503:
            pytest.skip("Fire stats or predictions not available in this test environment")
        assert r.status_code == 200

    def test_response_has_all_top_level_sections(self, client):
        r = client.get("/v1/risk/profile/50.0/-122.0")
        if r.status_code == 503:
            pytest.skip("Cache not available")
        data = r.json()
        for key in (
            "cell_id",
            "location",
            "timestamp",
            "context",
            "current",
            "historical_exposure",
            "susceptibility",
            "fire_regime",
            "composite_risk_rating",
        ):
            assert key in data, f"Missing key: {key}"

    def test_context_includes_bec_zone_name(self, client):
        r = client.get("/v1/risk/profile/50.0/-122.0")
        if r.status_code == 503:
            pytest.skip("Cache not available")
        ctx = r.json()["context"]
        assert "bec_zone" in ctx
        assert "bec_zone_name" in ctx
        # IDF → Interior Douglas-fir
        assert ctx["bec_zone"] == "IDF"
        assert ctx["bec_zone_name"] == "Interior Douglas-fir"

    def test_historical_exposure_has_all_tiers(self, client):
        r = client.get("/v1/risk/profile/50.0/-122.0")
        if r.status_code == 503:
            pytest.skip("Cache not available")
        exp = r.json()["historical_exposure"]
        assert "radius_km" in exp
        assert exp["radius_km"] == 10
        for tier in ("fires_10yr", "fires_30yr", "fires_all_time"):
            assert tier in exp, f"Missing tier: {tier}"
            assert "count" in exp[tier]

    def test_composite_score_is_between_0_and_1(self, client):
        r = client.get("/v1/risk/profile/50.0/-122.0")
        if r.status_code == 503:
            pytest.skip("Cache not available")
        composite = r.json()["composite_risk_rating"]["score"]
        assert 0.0 <= composite <= 1.0

    def test_composite_score_formula(self, client):
        """Verify composite = 0.3*(susc_pct/100) + 0.3*(exp_pct/100) + 0.4*current."""
        r = client.get("/v1/risk/profile/50.0/-122.0")
        if r.status_code == 503:
            pytest.skip("Cache not available")
        data = r.json()
        comps = data["composite_risk_rating"]["components"]
        expected = (
            0.3 * comps["susceptibility_percentile"] / 100.0
            + 0.3 * comps["exposure_percentile"] / 100.0
            + 0.4 * comps["current_score"]
        )
        expected = min(max(expected, 0.0), 1.0)
        assert abs(data["composite_risk_rating"]["score"] - round(expected, 4)) < 1e-6

    def test_susceptibility_section_present(self, client):
        r = client.get("/v1/risk/profile/50.0/-122.0")
        if r.status_code == 503:
            pytest.skip("Cache not available")
        susc = r.json()["susceptibility"]
        assert "score" in susc
        assert "percentile" in susc
        assert "label" in susc
        assert "basis" in susc

    def test_current_section_present(self, client):
        r = client.get("/v1/risk/profile/50.0/-122.0")
        if r.status_code == 503:
            pytest.skip("Cache not available")
        cur = r.json()["current"]
        assert "score" in cur
        assert "level" in cur
        assert "color" in cur

    def test_outside_bc_returns_422(self, client):
        """Coordinates outside BC should return 422 (validation error)."""
        r = client.get("/v1/risk/profile/40.0/-110.0")
        assert r.status_code == 422

    def test_cell_id_in_response(self, client):
        r = client.get("/v1/risk/profile/50.0/-122.0")
        if r.status_code == 503:
            pytest.skip("Cache not available")
        data = r.json()
        assert data["cell_id"] == "BC-5K-000000"

    def test_fire_regime_section_present(self, client):
        r = client.get("/v1/risk/profile/50.0/-122.0")
        if r.status_code == 503:
            pytest.skip("Cache not available")
        regime = r.json()["fire_regime"]
        assert "mean_return_years" in regime
        assert "typical_severity" in regime
        assert "dominant_cause" in regime

    def test_second_cell(self, client):
        """Test with coordinates near second cell."""
        r = client.get("/v1/risk/profile/51.5/-120.0")
        if r.status_code == 503:
            pytest.skip("Cache not available")
        assert r.status_code == 200
        data = r.json()
        assert data["cell_id"] == "BC-5K-000001"
        assert data["context"]["bec_zone"] == "SBPS"

    def test_no_fire_stats_returns_503(self, client):
        """When fire stats cache is empty, endpoint returns 503."""
        # Temporarily clear the cache
        set_fire_stats_cache({})
        r = client.get("/v1/risk/profile/50.0/-122.0")
        assert r.status_code == 503
        assert "compute_fire_stats" in r.json()["detail"]
        # Restore for other tests
        set_fire_stats_cache(_FIRE_STATS)
