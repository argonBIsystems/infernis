from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "INFERNIS"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, alias="INFERNIS_DEBUG")

    # Database
    database_url: str = Field(
        default="postgresql://localhost:5432/infernis",
        alias="INFERNIS_DATABASE_URL",
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="INFERNIS_REDIS_URL",
    )

    # ERA5 / Copernicus CDS
    cds_url: str = Field(default="https://cds.climate.copernicus.eu/api", alias="CDS_API_URL")
    cds_key: str = Field(default="", alias="CDS_API_KEY")

    # Google Earth Engine
    gee_project: str = Field(default="", alias="GEE_PROJECT_ID")
    gee_service_account_key: str = Field(default="", alias="GEE_PRIVATE_KEY")
    gee_client_email: str = Field(default="", alias="GEE_CLIENT_EMAIL")

    # NASA FIRMS
    firms_map_key: str = Field(default="", alias="FIRMS_MAP_KEY")

    # NASA Earthdata
    nasa_earthdata_user: str = Field(default="", alias="NASA_EARTHDATA_USER")
    nasa_earthdata_pass: str = Field(default="", alias="NASA_EARTHDATA_PASS")

    # Firebase
    firebase_project_id: str = Field(default="", alias="INFERNIS_FIREBASE_PROJECT_ID")
    firebase_service_account_json: str = Field(default="", alias="INFERNIS_FIREBASE_SA_JSON")
    firebase_api_key: str = Field(default="", alias="INFERNIS_FIREBASE_API_KEY")

    # Sentry
    sentry_dsn: str = Field(default="", alias="INFERNIS_SENTRY_DSN")

    # Grid
    grid_resolution_km: float = Field(default=1.0, alias="INFERNIS_GRID_RESOLUTION_KM")
    grid_parquet_path: str = Field(default="", alias="INFERNIS_GRID_PARQUET_PATH")
    bc_bbox_west: float = -139.06
    bc_bbox_east: float = -114.03
    bc_bbox_south: float = 48.30
    bc_bbox_north: float = 60.00

    # Model
    model_path: str = "models/fire_core_v1.json"
    model_1km_path: str = Field(
        default="models/fire_core_1km_v1.json", alias="INFERNIS_MODEL_1KM_PATH"
    )
    heatmap_model_path: str = Field(
        default="models/heatmap_v1.pt", alias="INFERNIS_HEATMAP_MODEL_PATH"
    )

    # Scheduler
    pipeline_enabled: bool = Field(default=True, alias="INFERNIS_PIPELINE_ENABLED")
    pipeline_hour: int = 14  # 2 PM PT
    pipeline_minute: int = 0
    pipeline_run_on_startup: bool = Field(default=False, alias="INFERNIS_PIPELINE_RUN_ON_STARTUP")

    # Data retention
    prediction_retention_days: int = Field(default=90, alias="INFERNIS_PREDICTION_RETENTION_DAYS")
    pipeline_run_retention_days: int = Field(
        default=365, alias="INFERNIS_PIPELINE_RUN_RETENTION_DAYS"
    )

    # Forecast
    forecast_enabled: bool = Field(default=False, alias="INFERNIS_FORECAST_ENABLED")
    forecast_max_days: int = Field(default=10, alias="INFERNIS_FORECAST_MAX_DAYS")
    forecast_confidence_decay: float = Field(
        default=0.95, alias="INFERNIS_FORECAST_CONFIDENCE_DECAY"
    )
    hrdps_data_dir: str = Field(default="data/raw/hrdps", alias="INFERNIS_HRDPS_DATA_DIR")
    gdps_data_dir: str = Field(default="data/raw/gdps", alias="INFERNIS_GDPS_DATA_DIR")

    # API
    api_prefix: str = "/v1"

    # CORS
    cors_origins: List[str] = [
        "https://infernis.ca",
        "https://www.infernis.ca",
        "https://app.infernis.ca",
        "https://api.infernis.ca",
        "http://localhost:8000",
        "http://localhost:3000",
    ]

    # Domain routing
    landing_domain: str = Field(default="infernis.ca", alias="INFERNIS_LANDING_DOMAIN")
    app_domain: str = Field(default="app.infernis.ca", alias="INFERNIS_APP_DOMAIN")
    api_domain: str = Field(default="api.infernis.ca", alias="INFERNIS_API_DOMAIN")

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "populate_by_name": True,
    }


settings = Settings()
