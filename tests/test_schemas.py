"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from infernis.models.enums import DangerLevel
from infernis.models.schemas import (
    FWIComponents,
    RiskScore,
    StatusResponse,
    WeatherConditions,
)


class TestFWIComponents:
    def test_valid(self):
        fwi = FWIComponents(ffmc=87.0, dmc=30.0, dc=200.0, isi=5.0, bui=40.0, fwi=12.0)
        assert fwi.ffmc == 87.0
        assert fwi.fwi == 12.0


class TestWeatherConditions:
    def test_valid(self):
        w = WeatherConditions(
            temperature_c=25.0,
            rh_pct=30.0,
            wind_kmh=15.0,
            precip_24h_mm=0.0,
            soil_moisture=0.2,
            ndvi=0.6,
            snow_cover=False,
        )
        assert w.temperature_c == 25.0
        assert w.snow_cover is False


class TestRiskScore:
    def test_auto_color(self):
        rs = RiskScore(score=0.5, level=DangerLevel.HIGH)
        assert rs.color == DangerLevel.HIGH.color

    def test_custom_color(self):
        rs = RiskScore(score=0.5, level=DangerLevel.HIGH, color="#FF0000")
        assert rs.color == "#FF0000"

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            RiskScore(score=-0.1, level=DangerLevel.LOW)
        with pytest.raises(ValidationError):
            RiskScore(score=1.1, level=DangerLevel.LOW)


class TestStatusResponse:
    def test_operational(self):
        s = StatusResponse(
            status="operational",
            version="0.1.0",
            last_pipeline_run="2026-01-01T00:00:00Z",
            model_version="fire_core_v1",
            grid_cells=12000,
            pipeline_healthy=True,
        )
        assert s.pipeline_healthy is True

    def test_initializing(self):
        s = StatusResponse(
            status="initializing",
            version="0.1.0",
            last_pipeline_run=None,
            model_version="fire_core_v1",
            grid_cells=0,
            pipeline_healthy=False,
        )
        assert s.last_pipeline_run is None
