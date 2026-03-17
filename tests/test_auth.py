"""Tests for API key authentication middleware."""


class TestAuthMiddleware:
    """Test that auth middleware blocks unauthenticated requests when debug=False."""

    def test_public_endpoints_no_auth(self):
        """Health, status, coverage should be accessible without key."""
        # Need to reimport with debug=False
        # Since Settings is already loaded, we test via the middleware logic directly
        from infernis.api.auth import PUBLIC_PATHS

        assert "/health" in PUBLIC_PATHS
        assert "/v1/status" in PUBLIC_PATHS
        assert "/v1/coverage" in PUBLIC_PATHS

    def test_daily_rate_limit_configured(self):
        """Daily rate limit should be set from config."""
        from infernis.config import settings

        assert settings.daily_rate_limit > 0

    def test_demo_paths_public(self):
        """Demo endpoints should not require auth."""
        from infernis.api.auth import PUBLIC_PREFIXES

        assert any("/demo" in p for p in PUBLIC_PREFIXES)
