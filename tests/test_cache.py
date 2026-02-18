"""Tests for Redis cache service (graceful degradation when Redis unavailable)."""

from unittest.mock import patch

import infernis.services.cache as cache_mod
from infernis.services.cache import (
    cache_fwi_state,
    cache_predictions,
    get_cached_prediction,
    load_fwi_state,
    redis_healthy,
)


class TestCacheGracefulDegradation:
    """When Redis is unavailable, all cache operations should return gracefully."""

    def setup_method(self):
        """Reset Redis state and point to an unreachable server."""
        cache_mod._redis_client = None
        cache_mod._redis_available = None

    def test_redis_healthy_false_when_unavailable(self):
        with patch.object(cache_mod.settings, "redis_url", "redis://localhost:59999/0"):
            assert redis_healthy() is False

    def test_cache_predictions_returns_zero(self):
        with patch.object(cache_mod.settings, "redis_url", "redis://localhost:59999/0"):
            result = cache_predictions({"cell1": {"score": 0.5}}, "2025-07-15")
            assert result == 0

    def test_get_cached_prediction_returns_none(self):
        with patch.object(cache_mod.settings, "redis_url", "redis://localhost:59999/0"):
            result = get_cached_prediction("cell1")
            assert result is None

    def test_load_fwi_state_returns_empty(self):
        with patch.object(cache_mod.settings, "redis_url", "redis://localhost:59999/0"):
            result = load_fwi_state()
            assert result == {}

    def test_cache_fwi_state_does_not_error(self):
        with patch.object(cache_mod.settings, "redis_url", "redis://localhost:59999/0"):
            cache_fwi_state({"cell1": {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0}})

    def test_redis_available_cached_after_first_failure(self):
        """After first failure, subsequent calls should not re-attempt connection."""
        with patch.object(cache_mod.settings, "redis_url", "redis://localhost:59999/0"):
            get_cached_prediction("test")
            assert cache_mod._redis_available is False
            # Second call should return immediately without retrying
            result = get_cached_prediction("test2")
            assert result is None


class TestCacheWithRedis:
    """When Redis IS available, test actual caching operations."""

    def setup_method(self):
        cache_mod._redis_client = None
        cache_mod._redis_available = None

    def test_redis_healthy_true(self):
        assert redis_healthy() is True

    def test_cache_and_retrieve_prediction(self):
        preds = {"cell_test_1": {"score": 0.75, "level": "HIGH"}}
        count = cache_predictions(preds, "2025-07-15")
        assert count == 1
        result = get_cached_prediction("cell_test_1")
        assert result is not None
        assert result["score"] == 0.75

    def test_fwi_state_roundtrip(self):
        state = {"cell_1": {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0}}
        cache_fwi_state(state)
        loaded = load_fwi_state()
        assert "cell_1" in loaded
        assert loaded["cell_1"]["ffmc"] == 85.0
