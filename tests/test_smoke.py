"""Smoke tests — verify the app boots and core endpoints respond."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with debug mode (skips API key auth)."""
    from infernis.main import app

    return TestClient(app)


class TestHealthAndBoot:
    """App startup and health check."""

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_openapi_schema_loads(self, client):
        resp = client.get("/v1/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "INFERNIS"

    @pytest.mark.skipif(
        not __import__("pathlib")
        .Path(__file__)
        .resolve()
        .parent.parent.joinpath("static/index.html")
        .exists(),
        reason="No frontend (open-source build)",
    )
    def test_static_login_page(self, client):
        resp = client.get("/static/index.html")
        assert resp.status_code == 200
        assert b"INFERNIS" in resp.content

    @pytest.mark.skipif(
        not __import__("pathlib")
        .Path(__file__)
        .resolve()
        .parent.parent.joinpath("static/index.html")
        .exists(),
        reason="No frontend (open-source build)",
    )
    def test_firebase_config_js(self, client):
        resp = client.get("/static/js/firebase-config.js")
        assert resp.status_code == 200
        assert "firebaseConfig" in resp.text


class TestAPIRoutes:
    """Verify data API routes exist and reject unauthenticated requests in non-debug."""

    def test_v1_status_debug(self, client):
        """In debug mode, /v1/status should work without API key."""
        resp = client.get("/v1/status")
        assert resp.status_code == 200

    def test_v1_coverage_debug(self, client):
        resp = client.get("/v1/coverage")
        assert resp.status_code == 200

    def test_v1_risk_grid_debug(self, client):
        resp = client.get("/v1/risk/grid?bbox=49.0,-123.5,50.0,-122.0")
        assert resp.status_code in (200, 503)  # 503 if no predictions cached yet


_has_dashboard = True
try:
    from infernis.api.dashboard_routes import dashboard_router  # noqa: F401
except ImportError:
    _has_dashboard = False


@pytest.mark.skipif(not _has_dashboard, reason="No dashboard routes (open-source build)")
class TestDashboardRoutes:
    """Dashboard endpoints require Firebase auth."""

    def test_profile_requires_auth(self, client):
        resp = client.get("/api/dashboard/profile")
        assert resp.status_code in (401, 503)  # 503 if Firebase not configured

    def test_usage_requires_auth(self, client):
        resp = client.get("/api/dashboard/usage")
        assert resp.status_code in (401, 503)

    def test_register_requires_auth(self, client):
        resp = client.post("/api/dashboard/register")
        assert resp.status_code in (401, 503)

    def test_regenerate_requires_auth(self, client):
        resp = client.post("/api/dashboard/key/regenerate")
        assert resp.status_code in (401, 503)


class TestConfigIntegrity:
    """Verify settings load correctly."""

    def test_settings_load(self):
        from infernis.config import settings

        assert settings.app_name == "INFERNIS"
        assert settings.grid_resolution_km in (1.0, 5.0)
        assert settings.bc_bbox_west < settings.bc_bbox_east
        assert settings.bc_bbox_south < settings.bc_bbox_north

    def test_model_path_defined(self):
        from infernis.config import settings

        assert settings.model_path.endswith(".json")

    def test_api_prefix(self):
        from infernis.config import settings

        assert settings.api_prefix == "/v1"
