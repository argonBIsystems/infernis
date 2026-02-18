"""INFERNIS application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from infernis.api.auth import APIKeyMiddleware
from infernis.api.routes import router
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
    description="Intelligence forged in fire. BC Forest Fire Prediction Engine.",
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

app.include_router(router)
if dashboard_router is not None:
    app.include_router(dashboard_router)

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
