"""INFERNIS application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from infernis.api.alerts_routes import alerts_router
from infernis.api.auth import APIKeyMiddleware
from infernis.api.batch_routes import batch_router
from infernis.api.explain_routes import explain_router
from infernis.api.fires_routes import fires_router
from infernis.api.history_routes import history_router
from infernis.api.profile_routes import profile_router
from infernis.api.routes import router
from infernis.api.tiles_routes import tiles_router
from infernis.api.trends_routes import trends_router
from infernis.config import settings

try:
    from infernis.api.dashboard_routes import dashboard_router
except ImportError:
    dashboard_router = None

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Sentry if DSN is configured
if settings.sentry_dsn:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            traces_sample_rate=0.1,
            environment="production" if not settings.debug else "development",
            release=f"infernis@{settings.app_version}",
            integrations=[FastApiIntegration(), SqlalchemyIntegration()],
        )
        logger.info("Sentry error tracking initialized")
    except ImportError:
        logger.warning("sentry-sdk not installed, error tracking disabled")


class DemoCORSMiddleware(BaseHTTPMiddleware):
    """Add permissive CORS for demo and tile endpoints so browser JS apps work."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        path = request.url.path
        if "/demo" in path or "/tiles/" in path:
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "X-API-Key, Content-Type"
        return response


_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global _scheduler
    logger.info("INFERNIS %s starting up", settings.app_version)

    # Initialize Firebase for dashboard auth
    try:
        from infernis.api.firebase_auth import init_firebase

        init_firebase()
    except Exception as e:
        logger.warning("Firebase initialization skipped: %s", e)

    # Load cached predictions from Redis (survive deploys without re-running pipeline)
    try:
        from infernis.services.cache import (
            load_forecasts_from_redis,
            load_grid_cells_from_redis,
            load_predictions_from_redis,
        )

        predictions, run_time = load_predictions_from_redis()
        grid_cells = load_grid_cells_from_redis()

        if predictions and grid_cells:
            from infernis.api.routes import set_predictions_cache

            set_predictions_cache(predictions, grid_cells, run_time)
            logger.info(
                "Loaded %d predictions + %d grid cells from Redis (last run: %s) — API ready",
                len(predictions),
                len(grid_cells),
                run_time,
            )
        elif predictions:
            logger.warning("Predictions in Redis but no grid cells — API will wait for pipeline")
        else:
            logger.info("Redis cache empty — API will be available after first pipeline run")

        forecasts, base_date = load_forecasts_from_redis()
        if forecasts:
            from infernis.api.routes import set_forecast_cache

            set_forecast_cache(forecasts, base_date)
            logger.info("Loaded %d forecast cells from Redis (base: %s)", len(forecasts), base_date)

        try:
            from infernis.services.cache import load_fire_stats_from_redis

            fire_stats = load_fire_stats_from_redis()
            if fire_stats:
                from infernis.api.profile_routes import set_fire_stats_cache

                set_fire_stats_cache(fire_stats)
                logger.info("Loaded %d fire stat cells from Redis", len(fire_stats))
        except Exception as e:
            logger.warning("Fire stats cache restore skipped: %s", e)
    except Exception as e:
        logger.warning("Redis cache restore failed: %s — API will initialize from pipeline", e)

    # Start the daily pipeline scheduler (can be disabled to prevent OOM in
    # memory-constrained containers like Railway Hobby 8 GB)
    if settings.pipeline_enabled:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            _scheduler = BackgroundScheduler()
            _scheduler.add_job(
                _run_scheduled_pipeline,
                CronTrigger(
                    hour=settings.pipeline_hour,
                    minute=settings.pipeline_minute,
                    timezone="America/Vancouver",  # Pacific Time
                ),
                id="daily_pipeline",
                name="Daily fire risk pipeline",
                replace_existing=True,
            )
            _scheduler.start()
            logger.info(
                "Scheduler started - pipeline runs daily at %02d:%02d PT",
                settings.pipeline_hour,
                settings.pipeline_minute,
            )
        except Exception as e:
            logger.error("Failed to start scheduler: %s", e)
    else:
        logger.info("Pipeline scheduler disabled (set INFERNIS_PIPELINE_ENABLED=true to enable)")

    # Run initial pipeline on startup (non-blocking) — disabled by default
    # to avoid OOM in memory-constrained containers
    if settings.pipeline_enabled and settings.pipeline_run_on_startup:
        try:
            import threading

            threading.Thread(target=_run_scheduled_pipeline, daemon=True).start()
            logger.info("Initial pipeline run triggered in background")
        except Exception as e:
            logger.warning("Initial pipeline run failed: %s", e)
    else:
        logger.info(
            "Startup pipeline run disabled (set INFERNIS_PIPELINE_RUN_ON_STARTUP=true to enable)"
        )

    yield

    # Shutdown
    if _scheduler:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down")


def _run_scheduled_pipeline():
    """Wrapper for the scheduler to call the pipeline runner."""
    try:
        from infernis.pipelines.runner import run_daily_pipeline

        run_daily_pipeline()
    except Exception as e:
        logger.error("Scheduled pipeline failed: %s", e, exc_info=True)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""**Wildfire risk prediction API for British Columbia.**

INFERNIS processes weather data, satellite imagery, and 21 open data sources
through an automated daily pipeline to produce fire risk scores for 84,535 grid
cells across BC. Updated daily at 2 PM Pacific.

**Getting started:**
1. Hit any `/v1/demo/` endpoint — no API key needed, same response format as live API
2. Sign up at [infernis.ca](https://infernis.ca) for a free API key
3. Use the same URL structure as demo — just drop `/demo` and add your `X-API-Key` header

**Base URL:** `https://api.infernis.ca/v1`

**Authentication:** `X-API-Key` header on all non-demo endpoints. All endpoints available to all keys.

**Rate limits:** Daily request limit per key. Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.
""",
    openapi_tags=[
        {"name": "risk", "description": "Point and area fire risk queries — the core API"},
        {
            "name": "forecast",
            "description": "Multi-day fire risk forecasts using ECCC GEM weather model",
        },
        {
            "name": "tiles",
            "description": "Map tile overlays for Google Maps, Leaflet, and Mapbox — no API key needed",
        },
        {"name": "batch", "description": "Query multiple locations in a single request"},
        {"name": "history", "description": "Historical risk data from the last 90 days"},
        {"name": "fires", "description": "Active wildfire data from BC Wildfire Service"},
        {"name": "alerts", "description": "Webhook notifications when risk exceeds a threshold"},
        {
            "name": "demo",
            "description": "Mock data endpoints for testing — no API key required, same response format as live API",
        },
        {"name": "system", "description": "Health checks, pipeline status, and grid metadata"},
    ],
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Authorization", "Content-Type"],
)

# API key auth (skipped in debug mode)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(DemoCORSMiddleware)

app.include_router(profile_router)  # BEFORE router — profile must match before {lat}/{lon}
app.include_router(router)
app.include_router(explain_router)
app.include_router(tiles_router)
app.include_router(batch_router)
app.include_router(history_router)
app.include_router(fires_router)
app.include_router(alerts_router)
app.include_router(trends_router)
if dashboard_router is not None:
    app.include_router(dashboard_router)

# MCP Server — exposes INFERNIS endpoints as tools for AI agents
try:
    from fastapi_mcp import FastApiMCP

    mcp = FastApiMCP(
        app,
        name="INFERNIS",
        description="BC wildfire risk prediction API — fire risk scores, forecasts, "
        "explainability, fire behavior, and location risk profiles for British Columbia.",
    )
    mcp.mount_http()
    logger.info("MCP server mounted at /mcp")
except ImportError:
    logger.info("fastapi-mcp not installed — MCP server disabled")
except Exception as e:
    logger.warning("MCP server failed to mount: %s", e)

# Static file directories and domain routing (only when frontend is present)
# In Docker (pip-installed), __file__ points to site-packages; fall back to CWD
_project_root = Path(__file__).resolve().parent.parent.parent
_static_dir = _project_root / "static"
if not _static_dir.is_dir():
    _project_root = Path.cwd()
    _static_dir = _project_root / "static"
_landing_dir = _static_dir / "landing"

if _landing_dir.is_dir():
    # Landing domains (root domain serves landing page)
    _landing_hosts = {
        settings.landing_domain,
        f"www.{settings.landing_domain}",
        "localhost",
    }

    @app.middleware("http")
    async def domain_router(request: Request, call_next):
        """Route requests based on Host header for multi-domain setup."""
        host = request.headers.get("host", "").split(":")[0]
        if host in _landing_hosts and request.url.path in ("/", ""):
            landing_file = _landing_dir / "index.html"
            if landing_file.is_file():
                return FileResponse(str(landing_file))
        return await call_next(request)

    @app.get("/", include_in_schema=False)
    async def root(request: Request):
        """Serve landing page at root."""
        landing_file = _landing_dir / "index.html"
        if landing_file.is_file():
            return FileResponse(str(landing_file))
        return {"name": settings.app_name, "version": settings.app_version}

    @app.get("/static/js/firebase-config.js", include_in_schema=False)
    async def firebase_config():
        """Serve Firebase client config from env vars."""
        js = f"""// Auto-generated Firebase config
const firebaseConfig = {{
  apiKey: "{settings.firebase_api_key}",
  authDomain: "{settings.firebase_project_id}.firebaseapp.com",
  projectId: "{settings.firebase_project_id}",
  storageBucket: "{settings.firebase_project_id}.firebasestorage.app",
}};
"""
        return Response(content=js, media_type="application/javascript")

    app.mount("/landing", StaticFiles(directory=str(_landing_dir)), name="landing")

if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
else:

    @app.get("/", include_in_schema=False)
    async def root():
        """API root when no frontend is present."""
        return {"name": settings.app_name, "version": settings.app_version}


@app.get("/health", include_in_schema=False)
async def health():
    """Health check endpoint for Railway / load balancers."""
    from infernis.services.cache import redis_healthy

    redis_ok = redis_healthy()
    return {
        "status": "ok",
        "version": settings.app_version,
        "redis": "connected" if redis_ok else "unavailable",
    }
