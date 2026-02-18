"""API key authentication and rate limiting middleware."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from infernis.config import settings

logger = logging.getLogger(__name__)

# Pacific Time (UTC-8 standard, UTC-7 daylight)
_PST = timezone(timedelta(hours=-8))


def _today_pst():
    """Return today's date in Pacific Time (matches dashboard 'midnight PST' reset)."""
    return datetime.now(_PST).date()


# Endpoints that don't require authentication
PUBLIC_PATHS = {"/health", f"{settings.api_prefix}/status", f"{settings.api_prefix}/coverage"}

# Tier rate limits (requests/day)
TIER_LIMITS = {
    "free": 50,
    "pro": 10_000,
    "enterprise": 100_000,
}

# Endpoints restricted by tier (paths relative to api_prefix)
# Free tier gets point queries; pro adds grid; enterprise adds heatmap
TIER_RESTRICTED = {
    "/risk/grid": {"pro", "enterprise"},
    "/risk/heatmap": {"enterprise"},
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validates API keys and enforces rate limits."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Skip auth for public endpoints, docs, dashboard, landing, and static assets
        if (
            path in PUBLIC_PATHS
            or path == "/"
            or path.startswith("/docs")
            or path.startswith("/openapi")
            or path.startswith(f"{settings.api_prefix}/docs")
            or path.startswith(f"{settings.api_prefix}/redoc")
            or path.startswith(f"{settings.api_prefix}/openapi")
            or path.startswith("/api/dashboard")
            or path.startswith("/static")
            or path.startswith("/landing")
            or path == "/favicon.ico"
        ):
            return await call_next(request)

        # Skip auth entirely in debug mode
        if settings.debug:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return JSONResponse(status_code=401, content={"detail": "Missing X-API-Key header"})

        # Look up key in database
        key_record = self._lookup_key(api_key)
        if key_record is None:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

        if not key_record["is_active"]:
            return JSONResponse(status_code=403, content={"detail": "API key has been deactivated"})

        # Check rate limit
        tier = key_record["tier"]
        daily_limit = TIER_LIMITS.get(tier, 50)
        requests_today = key_record["requests_today"]

        # Reset counter if new day
        if key_record["last_reset"] != _today_pst():
            requests_today = 0

        if requests_today >= daily_limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Daily rate limit exceeded"},
                headers={
                    "X-RateLimit-Limit": str(daily_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": "midnight PST",
                },
            )

        # Check tier-based endpoint access
        api_path = path.removeprefix(settings.api_prefix)
        for restricted_path, allowed_tiers in TIER_RESTRICTED.items():
            if api_path.startswith(restricted_path) and tier not in allowed_tiers:
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": f"Endpoint requires {' or '.join(sorted(allowed_tiers))} tier. "
                        f"Your tier: {tier}. Upgrade at https://infernis.ca",
                    },
                )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = max(0, daily_limit - requests_today - 1)
        response.headers["X-RateLimit-Limit"] = str(daily_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = "midnight PST"

        # Increment counter (fire-and-forget)
        self._increment_usage(key_record["id"], requests_today, _today_pst())

        return response

    def _lookup_key(self, api_key: str) -> dict | None:
        """Look up API key by SHA-256 hash."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        try:
            from infernis.db.engine import SessionLocal
            from infernis.db.tables import APIKeyDB

            db = SessionLocal()
            try:
                record = db.query(APIKeyDB).filter(APIKeyDB.key_hash == key_hash).first()
                if record is None:
                    return None
                return {
                    "id": record.id,
                    "tier": record.tier,
                    "daily_limit": record.daily_limit,
                    "requests_today": record.requests_today,
                    "last_reset": record.last_reset,
                    "is_active": record.is_active,
                }
            finally:
                db.close()
        except Exception as e:
            logger.error("API key lookup failed: %s", e)
            # Fail open in case DB is unavailable - allow request
            return {
                "id": 0,
                "tier": "free",
                "daily_limit": 50,
                "requests_today": 0,
                "last_reset": _today_pst(),
                "is_active": True,
            }

    def _increment_usage(self, key_id: int, current_count: int, today):
        """Increment the daily request counter."""
        try:
            from infernis.db.engine import SessionLocal
            from infernis.db.tables import APIKeyDB

            db = SessionLocal()
            try:
                record = db.query(APIKeyDB).filter(APIKeyDB.id == key_id).first()
                if record:
                    if record.last_reset != today:
                        record.requests_today = 1
                        record.last_reset = today
                    else:
                        record.requests_today = current_count + 1
                    db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.warning("Failed to increment API usage: %s", e)
