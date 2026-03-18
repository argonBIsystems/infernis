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


def load_predictions_from_redis() -> tuple[dict, str | None]:
    """Load all predictions from Redis. Returns (predictions_dict, run_time)."""
    r = get_redis()
    if r is None:
        return {}, None

    run_time = r.get("pred:last_run")
    if not run_time:
        return {}, None

    predictions = {}
    BATCH_SIZE = 5000
    keys = []
    for key in r.scan_iter("pred:latest:*", count=BATCH_SIZE):
        keys.append(key)

    for batch_start in range(0, len(keys), BATCH_SIZE):
        batch_keys = keys[batch_start : batch_start + BATCH_SIZE]
        pipe = r.pipeline()
        for k in batch_keys:
            pipe.get(k)
        values = pipe.execute()
        for k, v in zip(batch_keys, values):
            if v:
                cell_id = k.removeprefix("pred:latest:")
                predictions[cell_id] = json.loads(v)

    logger.info("Loaded %d predictions from Redis (last run: %s)", len(predictions), run_time)
    return predictions, run_time


def cache_forecasts(
    forecasts: dict[str, list[dict]], base_date: str, ttl_seconds: int = 172800
) -> int:
    """Write forecasts to Redis with TTL (default 48h)."""
    r = get_redis()
    if r is None:
        return 0

    BATCH_SIZE = 10_000
    count = 0
    items = list(forecasts.items())

    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start : batch_start + BATCH_SIZE]
        pipe = r.pipeline()
        for cell_id, days in batch:
            pipe.setex(f"forecast:latest:{cell_id}", ttl_seconds, json.dumps(days))
            count += 1
        pipe.execute()

    r.setex("forecast:base_date", ttl_seconds, base_date)
    logger.info("Cached %d forecast cells to Redis (TTL=%ds)", count, ttl_seconds)
    return count


def load_forecasts_from_redis() -> tuple[dict[str, list[dict]], str | None]:
    """Load all forecasts from Redis. Returns (forecasts_dict, base_date)."""
    r = get_redis()
    if r is None:
        return {}, None

    base_date = r.get("forecast:base_date")
    if not base_date:
        return {}, None

    forecasts = {}
    BATCH_SIZE = 5000
    keys = []
    for key in r.scan_iter("forecast:latest:*", count=BATCH_SIZE):
        keys.append(key)

    for batch_start in range(0, len(keys), BATCH_SIZE):
        batch_keys = keys[batch_start : batch_start + BATCH_SIZE]
        pipe = r.pipeline()
        for k in batch_keys:
            pipe.get(k)
        values = pipe.execute()
        for k, v in zip(batch_keys, values):
            if v:
                cell_id = k.removeprefix("forecast:latest:")
                forecasts[cell_id] = json.loads(v)

    logger.info("Loaded %d forecast cells from Redis (base date: %s)", len(forecasts), base_date)
    return forecasts, base_date


def cache_grid_cells(grid_cells: dict, ttl_seconds: int = 172800):
    """Persist grid_cells dict to Redis so startup doesn't need to regenerate the grid."""
    r = get_redis()
    if r is None:
        return

    BATCH_SIZE = 10_000
    items = list(grid_cells.items())

    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start : batch_start + BATCH_SIZE]
        pipe = r.pipeline()
        for cell_id, cell in batch:
            pipe.hset("grid:cells", cell_id, json.dumps(cell))
        pipe.execute()

    logger.info("Cached %d grid cells to Redis", len(grid_cells))


def load_grid_cells_from_redis() -> dict:
    """Load grid_cells dict from Redis."""
    r = get_redis()
    if r is None:
        return {}
    raw = r.hgetall("grid:cells")
    if not raw:
        return {}
    grid_cells = {cell_id: json.loads(data) for cell_id, data in raw.items()}
    logger.info("Loaded %d grid cells from Redis", len(grid_cells))
    return grid_cells


def cache_fire_stats(fire_stats: dict, ttl_seconds: int = 604800):
    """Persist fire_stats dict to Redis so startup doesn't need to recompute.

    Keys: fire_stats:{cell_id}
    Default TTL: 7 days (604800s) — stats change infrequently.

    Batches HSET commands in chunks of 10,000 to avoid pipeline buffer overflow.
    """
    r = get_redis()
    if r is None:
        return 0

    BATCH_SIZE = 10_000
    count = 0
    items = list(fire_stats.items())

    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start : batch_start + BATCH_SIZE]
        pipe = r.pipeline()
        for cell_id, stats in batch:
            pipe.setex(f"fire_stats:{cell_id}", ttl_seconds, json.dumps(stats))
            count += 1
        pipe.execute()

    logger.info("Cached %d fire stat cells to Redis (TTL=%ds)", count, ttl_seconds)
    return count


def load_fire_stats_from_redis() -> dict:
    """Load fire statistics from Redis. Returns dict[cell_id → data].

    Returns empty dict if no keys found or Redis unavailable.
    """
    r = get_redis()
    if r is None:
        return {}

    BATCH_SIZE = 5000
    keys = []
    for key in r.scan_iter("fire_stats:*", count=BATCH_SIZE):
        keys.append(key)

    if not keys:
        return {}

    fire_stats = {}
    for batch_start in range(0, len(keys), BATCH_SIZE):
        batch_keys = keys[batch_start : batch_start + BATCH_SIZE]
        pipe = r.pipeline()
        for k in batch_keys:
            pipe.get(k)
        values = pipe.execute()
        for k, v in zip(batch_keys, values):
            if v:
                cell_id = k.removeprefix("fire_stats:")
                fire_stats[cell_id] = json.loads(v)

    logger.info("Loaded %d fire stat cells from Redis", len(fire_stats))
    return fire_stats


def redis_healthy() -> bool:
    """Check if Redis is reachable."""
    r = get_redis()
    if r is None:
        return False
    try:
        return r.ping()
    except Exception:
        return False
