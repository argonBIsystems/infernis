"""Redis caching layer for predictions."""

import json
import logging
from typing import Optional

from infernis.config import settings

logger = logging.getLogger(__name__)

_redis_client = None
_redis_available = None  # None = untested, True/False = tested


def get_redis():
    """Lazy-init Redis connection. Returns None if Redis unavailable."""
    global _redis_client, _redis_available
    if _redis_available is False:
        return None
    if _redis_client is not None:
        return _redis_client
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=30)
        client.ping()
        _redis_client = client
        _redis_available = True
        logger.info("Connected to Redis at %s", settings.redis_url)
        return _redis_client
    except Exception as e:
        _redis_available = False
        logger.warning("Redis unavailable (%s) - using in-memory cache only", e)
        return None


def cache_predictions(predictions: dict, run_date: str, ttl_seconds: int = 172800):
    """Write all predictions to Redis with TTL (default 48h).

    Keys: pred:{run_date}:{cell_id}
    Also sets pred:latest:{cell_id} for current lookups.

    Batches commands in chunks of 10,000 to avoid pipeline buffer overflow.
    """
    r = get_redis()
    if r is None:
        return 0

    BATCH_SIZE = 10_000
    count = 0
    items = list(predictions.items())

    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start : batch_start + BATCH_SIZE]
        pipe = r.pipeline()
        for cell_id, pred in batch:
            value = json.dumps(pred)
            pipe.setex(f"pred:{run_date}:{cell_id}", ttl_seconds, value)
            pipe.setex(f"pred:latest:{cell_id}", ttl_seconds, value)
            count += 1
        pipe.execute()

    r.setex("pred:last_run", ttl_seconds, run_date)
    logger.info("Cached %d predictions to Redis (TTL=%ds)", count, ttl_seconds)
    return count


def get_cached_prediction(cell_id: str) -> Optional[dict]:
    """Read a single prediction from Redis cache."""
    r = get_redis()
    if r is None:
        return None
    raw = r.get(f"pred:latest:{cell_id}")
    if raw:
        return json.loads(raw)
    return None


def cache_fwi_state(fwi_state: dict[str, dict]):
    """Persist FWI moisture codes to Redis for recovery on restart.

    Batches HSET commands to avoid pipeline buffer overflow.
    """
    r = get_redis()
    if r is None:
        return

    BATCH_SIZE = 10_000
    items = list(fwi_state.items())

    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start : batch_start + BATCH_SIZE]
        pipe = r.pipeline()
        for cell_id, state in batch:
            pipe.hset("fwi:state", cell_id, json.dumps(state))
        pipe.execute()

    logger.info("Persisted FWI state for %d cells", len(fwi_state))


def load_fwi_state() -> dict[str, dict]:
    """Load persisted FWI state from Redis."""
    r = get_redis()
    if r is None:
        return {}
    raw = r.hgetall("fwi:state")
    return {cell_id: json.loads(state) for cell_id, state in raw.items()}


def redis_healthy() -> bool:
    """Check if Redis is reachable."""
    r = get_redis()
    if r is None:
        return False
    try:
        return r.ping()
    except Exception:
        return False
