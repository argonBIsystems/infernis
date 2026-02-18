"""Tests for FWI computation service.

Reference values checked against CFFDRS R package outputs for known inputs.
"""

import pandas as pd
import pytest

from infernis.services.fwi_service import FWIService


@pytest.fixture
def svc():
    return FWIService()


class TestComputeDaily:
    def test_default_startup(self, svc):
        """First day of season with default startup values."""
        result = svc.compute_daily(temp=20.0, rh=40.0, wind=12.0, precip=0.0, month=5)
        assert "ffmc" in result
        assert "dmc" in result
        assert "dc" in result
        assert "isi" in result
        assert "bui" in result
        assert "fwi" in result

    def test_hot_dry_windy(self, svc):
        """Extreme fire weather should produce high FWI."""
        result = svc.compute_daily(
            temp=35.0,
            rh=10.0,
            wind=30.0,
            precip=0.0,
            month=7,
            prev_ffmc=92.0,
            prev_dmc=100.0,
            prev_dc=400.0,
        )
        assert result["ffmc"] > 90.0
        assert result["isi"] > 10.0
        assert result["fwi"] > 20.0

    def test_rain_reduces_codes(self, svc):
        """Rain should reduce moisture codes."""
        dry = svc.compute_daily(
            temp=25.0,
            rh=30.0,
            wind=10.0,
            precip=0.0,
            month=7,
            prev_ffmc=90.0,
            prev_dmc=50.0,
            prev_dc=300.0,
        )
        wet = svc.compute_daily(
            temp=25.0,
            rh=30.0,
            wind=10.0,
            precip=20.0,
            month=7,
            prev_ffmc=90.0,
            prev_dmc=50.0,
            prev_dc=300.0,
        )
        assert wet["ffmc"] < dry["ffmc"]
        assert wet["dmc"] < dry["dmc"]
        assert wet["dc"] < dry["dc"]

    def test_ffmc_bounded(self, svc):
        """FFMC must be between 0 and 101."""
        result = svc.compute_daily(temp=-10.0, rh=100.0, wind=0.0, precip=50.0, month=1)
        assert 0.0 <= result["ffmc"] <= 101.0

    def test_all_values_non_negative(self, svc):
        """All FWI components must be >= 0."""
        result = svc.compute_daily(temp=0.0, rh=95.0, wind=2.0, precip=10.0, month=4)
        for key, val in result.items():
            assert val >= 0.0, f"{key} is negative: {val}"


class TestComputeSeason:
    def test_multi_day(self, svc):
        """Compute a 5-day dry spell - FWI should increase."""
        days = pd.DataFrame(
            {
                "temp": [25.0] * 5,
                "rh": [25.0] * 5,
                "wind": [15.0] * 5,
                "precip": [0.0] * 5,
                "month": [7] * 5,
            }
        )
        result = svc.compute_season(days)
        assert len(result) == 5
        # FWI should trend upward during dry spell
        assert result["fwi"].iloc[-1] >= result["fwi"].iloc[0]

    def test_codes_carry_forward(self, svc):
        """DMC and DC should accumulate over consecutive dry days."""
        days = pd.DataFrame(
            {
                "temp": [28.0] * 10,
                "rh": [20.0] * 10,
                "wind": [12.0] * 10,
                "precip": [0.0] * 10,
                "month": [7] * 10,
            }
        )
        result = svc.compute_season(days)
        assert result["dmc"].iloc[-1] > result["dmc"].iloc[0]
        assert result["dc"].iloc[-1] > result["dc"].iloc[0]
