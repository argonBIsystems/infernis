"""Pipeline runner - executes daily pipeline with DB persistence and cache updates."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_daily_pipeline(target_date: date | None = None):
    """Full pipeline execution with database writes and cache updates.

    Called by the scheduler or manually. This is the production entry point.
    """
    target_date = target_date or date.today()
    started_at = datetime.now(timezone.utc)
    run_id = None

    try:
        # Log pipeline start
        run_id = _log_pipeline_start(target_date, started_at)

        # Load grid
        grid_df = _load_grid()
        if grid_df is None or len(grid_df) == 0:
            raise RuntimeError("No grid cells available. Run grid initialization first.")

        # Build grid_cells dict for cache (vectorized array access, not iterrows)
        _cell_ids = grid_df["cell_id"].values
        _lats = grid_df["lat"].values
        _lons = grid_df["lon"].values
        _bec = grid_df.get("bec_zone", pd.Series([""] * len(grid_df))).fillna("").values
        _fuel = grid_df.get("fuel_type", pd.Series([""] * len(grid_df))).fillna("").values
        _elev = grid_df.get("elevation_m", pd.Series(np.zeros(len(grid_df)))).fillna(0).values

        grid_cells = {}
        for i in range(len(_cell_ids)):
            grid_cells[_cell_ids[i]] = {
                "lat": float(_lats[i]),
                "lon": float(_lons[i]),
                "bec_zone": str(_bec[i]),
                "fuel_type": str(_fuel[i]),
                "elevation_m": float(_elev[i]),
            }

        # Initialize pipeline and restore FWI state
        from infernis.config import settings
        from infernis.pipelines.daily_pipeline import DailyPipeline

        pipeline = DailyPipeline()
        # Select model path based on grid resolution
        model_path = (
            settings.model_1km_path if settings.grid_resolution_km <= 1.0 else settings.model_path
        )
        pipeline.load_model(model_path)

        # Restore FWI state from Redis
        from infernis.services.cache import load_fwi_state

        prev_state = load_fwi_state()
        if prev_state:
            pipeline._prev_fwi_state = prev_state
            logger.info("Restored FWI state for %d cells", len(prev_state))

        # Run pipeline
        predictions = pipeline.run(target_date=target_date, grid_df=grid_df)

        if not predictions:
            raise RuntimeError("Pipeline returned no predictions")

        # Persist FWI state to Redis
        from infernis.services.cache import cache_fwi_state, cache_grid_cells, cache_predictions

        cache_fwi_state(pipeline._prev_fwi_state)

        # Write predictions to database
        _save_predictions_to_db(predictions, target_date)

        # Write predictions + grid cells to Redis cache
        run_time = datetime.now(timezone.utc).isoformat()
        cache_predictions(predictions, target_date.isoformat())
        cache_grid_cells(grid_cells)

        # Update in-memory API cache
        from infernis.api.routes import set_predictions_cache

        set_predictions_cache(predictions, grid_cells, run_time)

        # Log pipeline success/partial
        completed_at = datetime.now(timezone.utc)
        status = pipeline.pipeline_status  # 'success' or 'partial'
        _log_pipeline_complete(run_id, completed_at, len(predictions), status)

        logger.info(
            "Pipeline run complete: %d cells in %.1fs",
            len(predictions),
            (completed_at - started_at).total_seconds(),
        )

        # Run forecast pipeline if enabled
        if settings.forecast_enabled:
            _run_forecast_pipeline(
                pipeline,
                grid_df,
                grid_cells,
                target_date,
                run_time,
            )

        # Fire webhook alerts for any thresholds exceeded
        _check_alerts(predictions)

        # Clean up old data
        cleanup_old_data()

        return predictions

    except Exception as e:
        logger.error("Pipeline run failed: %s", e, exc_info=True)
        if run_id:
            _log_pipeline_failure(run_id, str(e))
        raise


def _run_forecast_pipeline(
    daily_pipeline,
    grid_df: pd.DataFrame,
    grid_cells: dict,
    target_date: date,
    run_time: str,
):
    """Run the multi-day forecast pipeline after the daily pipeline."""
    try:
        from infernis.api.routes import set_forecast_cache
        from infernis.pipelines.forecast_pipeline import ForecastPipeline

        forecast = ForecastPipeline()
        # Use same resolution-aware model as daily pipeline
        from infernis.config import settings as _settings

        _model_path = (
            _settings.model_1km_path
            if _settings.grid_resolution_km <= 1.0
            else _settings.model_path
        )
        forecast.load_model(_model_path)

        # Pass today's observed vegetation so forecast doesn't use hardcoded defaults
        satellite = getattr(daily_pipeline, "_last_satellite", None)
        if satellite:
            forecast._observed_ndvi = satellite.get("ndvi")
            forecast._observed_snow = satellite.get("snow", np.zeros(0)).astype(np.float64)
            forecast._observed_lai = satellite.get("lai")
            logger.info("Forecast: using today's observed NDVI/snow/LAI")

        # Pass today's soil moisture (from ERA5 merge in daily pipeline)
        weather = getattr(daily_pipeline, "_last_weather", None)
        if weather:
            forecast._observed_soil_moisture = {
                "soil_moisture_1": weather.get("soil_moisture_1"),
                "soil_moisture_2": weather.get("soil_moisture_2"),
                "soil_moisture_3": weather.get("soil_moisture_3"),
                "soil_moisture_4": weather.get("soil_moisture_4"),
            }
            logger.info("Forecast: using today's soil moisture")

        # Reuse Open-Meteo forecast weather from daily pipeline (avoids 2nd API fetch)
        shared_weather = getattr(daily_pipeline, "_openmeteo_forecast_weather", None)
        if shared_weather:
            forecast._prefetched_weather = shared_weather
            logger.info(
                "Forecast: reusing Open-Meteo weather from daily pipeline (%d days)",
                len(shared_weather),
            )

        forecasts = forecast.run(
            grid_df=grid_df,
            current_fwi_state=daily_pipeline._prev_fwi_state,
            target_date=target_date,
        )

        if forecasts:
            set_forecast_cache(forecasts, target_date.isoformat())
            _save_forecasts_to_db(forecasts, target_date)
            from infernis.services.cache import cache_forecasts as _cache_fc

            _cache_fc(forecasts, target_date.isoformat())
            logger.info("Forecast pipeline complete: %d cells", len(forecasts))
    except Exception as e:
        logger.error("Forecast pipeline failed: %s", e, exc_info=True)


def _save_forecasts_to_db(forecasts: dict[str, list[dict]], base_date: date):
    """Batch insert forecast predictions into the database using raw SQL for speed."""
    try:
        import json

        from infernis.db.engine import SessionLocal
        from infernis.db.tables import ForecastPredictionDB

        db = SessionLocal()
        try:
            # Delete existing forecasts for this base date
            db.query(ForecastPredictionDB).filter(
                ForecastPredictionDB.base_date == base_date
            ).delete()
            db.commit()

            # Use raw SQL with executemany in batches for 21M+ records
            BATCH_SIZE = 50_000
            table = ForecastPredictionDB.__table__
            batch = []
            total = 0

            for cell_id, days in forecasts.items():
                for day in days:
                    batch.append(
                        {
                            "cell_id": cell_id,
                            "base_date": base_date,
                            "lead_day": day["lead_day"],
                            "valid_date": date.fromisoformat(day["valid_date"]),
                            "risk_score": day["risk_score"],
                            "danger_level": day["danger_level"],
                            "confidence": day["confidence"],
                            "fwi_components": json.dumps(day["fwi"]),
                        }
                    )
                    if len(batch) >= BATCH_SIZE:
                        db.execute(table.insert(), batch)
                        db.commit()
                        total += len(batch)
                        batch = []

            if batch:
                db.execute(table.insert(), batch)
                db.commit()
                total += len(batch)

            logger.info("Saved %d forecast records for base_date %s", total, base_date)
        finally:
            db.close()
    except Exception as e:
        logger.error("Failed to save forecasts to DB: %s", e)


def _check_alerts(predictions: dict):
    """Check all active alerts and fire webhooks for threshold exceedances.

    Guards:
    - Skips alerts whose API key is inactive
    - Skips alerts that were already triggered within the cooldown period (default 24h)
    - Tracks consecutive delivery failures; auto-disables after 5
    """
    MAX_CONSECUTIVE_FAILURES = 5

    try:
        import httpx

        from infernis.config import settings
        from infernis.db.engine import SessionLocal
        from infernis.db.tables import AlertDB, APIKeyDB
        from infernis.models.enums import DangerLevel

        cooldown = timedelta(hours=settings.alert_cooldown_hours)

        db = SessionLocal()
        try:
            # Join with api_keys to skip alerts for inactive keys
            alerts = (
                db.query(AlertDB)
                .join(APIKeyDB, AlertDB.api_key_id == APIKeyDB.id)
                .filter(
                    AlertDB.is_active == True,  # noqa: E712
                    APIKeyDB.is_active == True,  # noqa: E712
                )
                .all()
            )
            if not alerts:
                return

            now = datetime.now(timezone.utc)
            triggered = 0
            failed = 0
            skipped_cooldown = 0
            auto_disabled = 0

            for alert in alerts:
                pred = predictions.get(alert.cell_id)
                if pred is None:
                    continue

                score = pred.get("score", 0.0)
                if score < alert.threshold:
                    continue

                # Cooldown: skip if already fired recently
                if alert.last_triggered:
                    last = alert.last_triggered
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=timezone.utc)
                    if now - last < cooldown:
                        skipped_cooldown += 1
                        continue

                # Threshold exceeded — fire the webhook
                level = DangerLevel.from_score(score)
                payload = {
                    "alert_id": alert.id,
                    "cell_id": alert.cell_id,
                    "latitude": alert.latitude,
                    "longitude": alert.longitude,
                    "threshold": alert.threshold,
                    "risk": {
                        "score": round(score, 4),
                        "level": level.value,
                        "color": level.color,
                    },
                    "fwi": pred.get("fwi", 0.0),
                    "temperature_c": pred.get("temperature_c", 0.0),
                    "triggered_at": now.isoformat(),
                }

                delivery_ok = False
                try:
                    resp = httpx.post(
                        alert.webhook_url,
                        json=payload,
                        timeout=10.0,
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "INFERNIS-Alerts/1.0",
                        },
                    )
                    if 200 <= resp.status_code < 300:
                        delivery_ok = True
                    else:
                        logger.warning(
                            "Alert %d webhook returned HTTP %d",
                            alert.id,
                            resp.status_code,
                        )
                    logger.info(
                        "Alert %d fired: cell=%s score=%.3f → %s (HTTP %d)",
                        alert.id,
                        alert.cell_id,
                        score,
                        alert.webhook_url[:50],
                        resp.status_code,
                    )
                except Exception as e:
                    logger.warning("Alert %d webhook failed: %s", alert.id, e)

                if delivery_ok:
                    alert.last_triggered = now
                    alert.consecutive_failures = 0
                    triggered += 1
                else:
                    alert.consecutive_failures = (alert.consecutive_failures or 0) + 1
                    failed += 1

                    if alert.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        alert.is_active = False
                        alert.disabled_reason = (
                            f"Auto-disabled after {MAX_CONSECUTIVE_FAILURES} "
                            f"consecutive webhook failures"
                        )
                        auto_disabled += 1
                        logger.warning(
                            "Alert %d auto-disabled after %d consecutive failures: %s",
                            alert.id,
                            MAX_CONSECUTIVE_FAILURES,
                            alert.webhook_url[:50],
                        )

            db.commit()

            if triggered or failed or skipped_cooldown:
                logger.info(
                    "Alerts: %d triggered, %d failed, %d cooldown-skipped, "
                    "%d auto-disabled out of %d active",
                    triggered,
                    failed,
                    skipped_cooldown,
                    auto_disabled,
                    len(alerts),
                )

        finally:
            db.close()
    except Exception as e:
        logger.error("Alert check failed: %s", e)


def _load_grid() -> pd.DataFrame | None:
    """Load grid from parquet, DB, or generate in memory."""
    from infernis.config import settings

    # Try parquet first (fastest for large grids)
    if settings.grid_parquet_path:
        try:
            from infernis.grid.initializer import load_grid_from_parquet

            grid_df = load_grid_from_parquet(settings.grid_parquet_path)
            if grid_df is not None and len(grid_df) > 0:
                logger.info("Loaded %d grid cells from parquet", len(grid_df))
                return grid_df
        except Exception as e:
            logger.warning("Could not load grid from parquet (%s), trying DB", e)

    # Try database
    try:
        from infernis.grid.initializer import load_grid_from_db

        grid_df = load_grid_from_db()
        if grid_df is not None and len(grid_df) > 0:
            logger.info("Loaded %d grid cells from database", len(grid_df))
            return grid_df
    except Exception as e:
        logger.warning("Could not load grid from DB (%s), generating in memory", e)

    # Fallback: generate in memory
    from infernis.grid.initializer import initialize_grid

    return initialize_grid()


def _save_predictions_to_db(predictions: dict, target_date: date):
    """Batch insert predictions into the database using raw SQL for speed."""
    try:
        import json

        from infernis.db.engine import SessionLocal
        from infernis.db.tables import PredictionDB

        db = SessionLocal()
        try:
            # Delete existing predictions for this date (idempotent re-runs)
            db.query(PredictionDB).filter(PredictionDB.prediction_date == target_date).delete()
            db.commit()

            def _py(v):
                """Convert numpy scalars to native Python types for SQLAlchemy."""
                if v is None:
                    return None
                return float(v) if hasattr(v, "item") else v

            BATCH_SIZE = 50_000
            table = PredictionDB.__table__
            batch = []
            total = 0

            for cell_id, pred in predictions.items():
                # Serialize shap_values dict as JSON string for JSONB column
                shap_raw = pred.get("shap_values")
                shap_serialized = json.dumps(shap_raw) if shap_raw is not None else None

                batch.append(
                    {
                        "cell_id": cell_id,
                        "prediction_date": target_date,
                        "score": _py(pred["score"]),
                        "level": pred["level"],
                        "ffmc": _py(pred.get("ffmc")),
                        "dmc": _py(pred.get("dmc")),
                        "dc": _py(pred.get("dc")),
                        "isi": _py(pred.get("isi")),
                        "bui": _py(pred.get("bui")),
                        "fwi": _py(pred.get("fwi")),
                        "temperature_c": _py(pred.get("temperature_c")),
                        "rh_pct": _py(pred.get("rh_pct")),
                        "wind_kmh": _py(pred.get("wind_kmh")),
                        "precip_24h_mm": _py(pred.get("precip_24h_mm")),
                        "soil_moisture": _py(pred.get("soil_moisture")),
                        "ndvi": _py(pred.get("ndvi")),
                        "snow_cover": _py(pred.get("snow_cover")),
                        "shap_values": shap_serialized,
                    }
                )
                if len(batch) >= BATCH_SIZE:
                    db.execute(table.insert(), batch)
                    db.commit()
                    total += len(batch)
                    batch = []

            if batch:
                db.execute(table.insert(), batch)
                db.commit()
                total += len(batch)

            logger.info("Saved %d predictions to database for %s", total, target_date)
        finally:
            db.close()
    except Exception as e:
        logger.error("Failed to save predictions to DB: %s", e)


def _log_pipeline_start(target_date: date, started_at: datetime) -> int | None:
    """Create a pipeline_runs record for this execution."""
    try:
        from infernis.db.engine import SessionLocal
        from infernis.db.tables import PipelineRunDB

        db = SessionLocal()
        try:
            run = PipelineRunDB(
                run_date=target_date,
                started_at=started_at,
                status="running",
                model_version="fire_core_v1",
            )
            db.add(run)
            db.commit()
            db.refresh(run)
            return run.id
        finally:
            db.close()
    except Exception as e:
        logger.warning("Could not log pipeline start: %s", e)
        return None


def _log_pipeline_complete(
    run_id: int, completed_at: datetime, cells: int, status: str = "success"
):
    """Update pipeline_runs record with success or partial."""
    try:
        from infernis.db.engine import SessionLocal
        from infernis.db.tables import PipelineRunDB

        db = SessionLocal()
        try:
            run = db.query(PipelineRunDB).get(run_id)
            if run:
                run.completed_at = completed_at
                run.status = status
                run.cells_processed = cells
                db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.warning("Could not log pipeline completion: %s", e)


def _log_pipeline_failure(run_id: int, error_msg: str):
    """Update pipeline_runs record with failure."""
    try:
        from infernis.db.engine import SessionLocal
        from infernis.db.tables import PipelineRunDB

        db = SessionLocal()
        try:
            run = db.query(PipelineRunDB).get(run_id)
            if run:
                run.completed_at = datetime.now(timezone.utc)
                run.status = "failed"
                run.error_message = error_msg[:1000]
                db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.warning("Could not log pipeline failure: %s", e)


def cleanup_old_data(
    prediction_days: int | None = None,
    pipeline_run_days: int | None = None,
):
    """Delete predictions and pipeline_runs older than retention period.

    Also disables stale alerts (never triggered within alert_stale_days).
    Safe to call at any time — failures are logged but never raised.
    """
    from infernis.config import settings

    pred_days = prediction_days or settings.prediction_retention_days
    run_days = pipeline_run_days or settings.pipeline_run_retention_days

    try:
        from infernis.db.engine import SessionLocal
        from infernis.db.tables import AlertDB, PipelineRunDB, PredictionDB

        db = SessionLocal()
        try:
            # Prune old predictions
            pred_cutoff = date.today() - timedelta(days=pred_days)
            pred_deleted = (
                db.query(PredictionDB).filter(PredictionDB.prediction_date < pred_cutoff).delete()
            )
            if pred_deleted:
                logger.info(
                    "Cleanup: deleted %d predictions older than %s (%d-day retention)",
                    pred_deleted,
                    pred_cutoff,
                    pred_days,
                )

            # Prune old pipeline run logs
            run_cutoff = date.today() - timedelta(days=run_days)
            runs_deleted = (
                db.query(PipelineRunDB).filter(PipelineRunDB.run_date < run_cutoff).delete()
            )
            if runs_deleted:
                logger.info(
                    "Cleanup: deleted %d pipeline_runs older than %s (%d-day retention)",
                    runs_deleted,
                    run_cutoff,
                    run_days,
                )

            # Disable stale alerts (active but never triggered within stale_days)
            stale_cutoff = datetime.now(timezone.utc) - timedelta(days=settings.alert_stale_days)
            stale_alerts = (
                db.query(AlertDB)
                .filter(
                    AlertDB.is_active == True,  # noqa: E712
                    (AlertDB.last_triggered == None) | (AlertDB.last_triggered < stale_cutoff),  # noqa: E711
                    AlertDB.created_at < stale_cutoff,
                )
                .all()
            )
            if stale_alerts:
                for alert in stale_alerts:
                    alert.is_active = False
                    alert.disabled_reason = (
                        f"Auto-disabled: no successful delivery in {settings.alert_stale_days} days"
                    )
                logger.info(
                    "Cleanup: disabled %d stale alerts (no activity in %d days)",
                    len(stale_alerts),
                    settings.alert_stale_days,
                )

            db.commit()

            if not pred_deleted and not runs_deleted and not stale_alerts:
                logger.info("Cleanup: nothing to prune")
        finally:
            db.close()
    except Exception as e:
        logger.warning("Data cleanup failed (non-fatal): %s", e)
