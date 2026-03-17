from typing import Optional

from pydantic import BaseModel, Field

from infernis.models.enums import BECZone, DangerLevel, FuelType


class ConfidenceInterval(BaseModel):
    """90% prediction confidence interval for a fire risk score.

    Produced by companion XGBoost quantile regression models trained at
    the 5th and 95th percentiles of the fire risk distribution.  When
    quantile models are not available (e.g. freshly deployed instance
    without pre-trained quantile weights), this field is ``None`` in the
    API response.
    """

    lower: float = Field(ge=0.0, le=1.0, description="Lower bound (5th percentile)")
    upper: float = Field(ge=0.0, le=1.0, description="Upper bound (95th percentile)")
    level: float = Field(default=0.90, description="Confidence level (0.90 = 90%)")


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
    c_haines: Optional[float] = Field(
        None,
        description=(
            "Continuous Haines Index (0-13). Measures atmospheric instability + dryness. "
            ">8 = high pyroconvection potential. None when pressure-level data unavailable."
        ),
    )


class FireBehaviour(BaseModel):
    """Canadian FBP fire behaviour metrics for a single grid cell.

    Computed via cffdrs_py's fire_behaviour_prediction() function.
    All fields are zero for non-fuel types (NF, WA).
    """

    rate_of_spread_mpm: float = Field(
        description="Head fire rate of spread (m/min)"
    )
    head_fire_intensity_kwm: float = Field(
        description="Head fire intensity (kW/m)"
    )
    fire_type: str = Field(
        description="Fire type: 'surface', 'intermittent_crown', or 'active_crown'"
    )
    crown_fraction_burned: float = Field(
        ge=0.0, le=1.0, description="Crown fraction burned [0, 1]"
    )
    flame_length_m: float = Field(
        description="Byram flame length (m)"
    )


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
    change_24h: Optional[float] = Field(None, description="Score change vs yesterday (-1 to +1)")
    confidence_interval: Optional[ConfidenceInterval] = Field(
        None,
        description=(
            "90% prediction confidence interval for the risk score. "
            "None when quantile models are not loaded."
        ),
    )
    fire_behaviour: Optional[FireBehaviour] = Field(
        None,
        description=(
            "Canadian FBP fire behaviour metrics (ROS, HFI, fire type, CFB, flame length). "
            "None when fuel type data is unavailable or FBP computation fails."
        ),
    )


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
    temperature_c: Optional[float] = Field(None, description="Forecast temperature (°C)")
    rh_pct: Optional[float] = Field(None, description="Forecast relative humidity (%)")
    wind_kmh: Optional[float] = Field(None, description="Forecast wind speed (km/h)")
    precip_24h_mm: Optional[float] = Field(None, description="Forecast precipitation (mm)")


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
