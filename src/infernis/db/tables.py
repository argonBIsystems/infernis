from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB

from infernis.db.engine import Base


class GridCellDB(Base):
    __tablename__ = "grid_cells"

    id = Column(Integer, primary_key=True)
    cell_id = Column(String(30), unique=True, nullable=False, index=True)
    geom = Column(Geometry("POLYGON", srid=3005), nullable=False)
    centroid = Column(Geometry("POINT", srid=4326), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    bec_zone = Column(String(10))
    fuel_type = Column(String(5))
    elevation_m = Column(Float)
    slope_deg = Column(Float)
    aspect_deg = Column(Float)
    hillshade = Column(Float)

    __table_args__ = (Index("ix_grid_cells_centroid", centroid, postgresql_using="gist"),)


class PredictionDB(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    cell_id = Column(String(30), nullable=False, index=True)
    prediction_date = Column(Date, nullable=False)

    score = Column(Float, nullable=False)
    level = Column(String(20), nullable=False)

    # FWI components
    ffmc = Column(Float)
    dmc = Column(Float)
    dc = Column(Float)
    isi = Column(Float)
    bui = Column(Float)
    fwi = Column(Float)

    # Weather conditions
    temperature_c = Column(Float)
    rh_pct = Column(Float)
    wind_kmh = Column(Float)
    precip_24h_mm = Column(Float)
    soil_moisture = Column(Float)
    ndvi = Column(Float)
    snow_cover = Column(Boolean)

    # Feature vector (for debugging/explainability)
    features = Column(JSONB)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_predictions_cell_date", cell_id, prediction_date, unique=True),)


class PipelineRunDB(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True)
    run_date = Column(Date, nullable=False)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    status = Column(String(20), nullable=False)  # running, success, failed
    cells_processed = Column(Integer, default=0)
    error_message = Column(Text)
    model_version = Column(String(50))

    created_at = Column(DateTime, default=datetime.utcnow)


class APIKeyDB(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    key_preview = Column(String(20))  # e.g. "a1b2****c3d4"
    name = Column(String(100), nullable=False)
    tier = Column(String(20), nullable=False, default="free")
    daily_limit = Column(Integer, nullable=False, default=50)
    requests_today = Column(Integer, default=0)
    last_reset = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class ForecastPredictionDB(Base):
    __tablename__ = "forecast_predictions"

    id = Column(Integer, primary_key=True)
    cell_id = Column(String(30), nullable=False)
    base_date = Column(Date, nullable=False)
    lead_day = Column(Integer, nullable=False)
    valid_date = Column(Date, nullable=False)
    risk_score = Column(Float, nullable=False)
    danger_level = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    fwi_components = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_forecast_cell_valid", cell_id, valid_date),
        Index("ix_forecast_base_lead", base_date, lead_day),
    )


class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    firebase_uid = Column(String(128), unique=True, nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    display_name = Column(String(200))
    api_key_id = Column(Integer, ForeignKey("api_keys.id", ondelete="SET NULL"))
    tier = Column(String(20), nullable=False, default="free")
    billing_cycle_start = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
