from typing import Optional

from pydantic import BaseModel, Field

from infernis.models.enums import BECZone, DangerLevel, FuelType


class FWIComponents(BaseModel):
    ffmc: float = Field(description="Fine Fuel Moisture Code (0-101)")
    dmc: float = Field(description="Duff Moisture Code")
    dc: float = Field(description="Drought Code")
    isi: float = Field(description="Initial Spread Index")
    bui: float = Field(description="Buildup Index")
    fwi: float = Field(description="Fire Weather Index")


class WeatherConditions(BaseModel):
    temperature_c: float
    rh_pct: float
    wind_kmh: float
    precip_24h_mm: float
    soil_moisture: float
    ndvi: float
    snow_cover: bool


class RiskScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    level: DangerLevel
    color: str = ""

    def model_post_init(self, __context):
        if not self.color:
            self.color = self.level.color


class GridCell(BaseModel):
    cell_id: str
    lat: float
    lon: float
    bec_zone: Optional[BECZone] = None
    fuel_type: Optional[FuelType] = None
    elevation_m: Optional[float] = None
    slope_deg: Optional[float] = None
    aspect_deg: Optional[float] = None
    hillshade: Optional[float] = None


class RiskResponse(BaseModel):
    location: dict
    grid_cell_id: str
    timestamp: str
    risk: RiskScore
    fwi: FWIComponents
    conditions: WeatherConditions
    context: dict
    forecast_horizon: str = "24h"
    next_update: str


class ZoneRiskSummary(BaseModel):
    zone_name: str
    bec_zone: BECZone
    avg_risk_score: float
    max_risk_score: float
    dominant_level: DangerLevel
    cell_count: int
    high_risk_cells: int


class ForecastDay(BaseModel):
    valid_date: str
    lead_day: int = Field(ge=1, le=10)
    risk_score: float = Field(ge=0.0, le=1.0)
    danger_level: int = Field(ge=1, le=6)
    danger_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    fwi: FWIComponents
    data_source: str = ""


class ForecastResponse(BaseModel):
    latitude: float
    longitude: float
    cell_id: str
    base_date: str
    forecast: list[ForecastDay]
    generated_at: str


class StatusResponse(BaseModel):
    status: str
    version: str
    last_pipeline_run: Optional[str]
    model_version: str
    grid_cells: int
    pipeline_healthy: bool
