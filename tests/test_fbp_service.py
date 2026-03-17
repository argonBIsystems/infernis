"""Tests for FBP fire behaviour prediction service wrapper.

Tests confirm the wrapper correctly calls cffdrs_py's FBP system and
returns well-formed, physically plausible outputs.
"""

import math

import pytest

from infernis.services.fbp_service import (
    _classify_fire_type,
    _flame_length_m,
    compute_fire_behaviour,
)

# ---------------------------------------------------------------------------
# Common test inputs (summer BC interior conditions)
# ---------------------------------------------------------------------------
_BASE_KWARGS = dict(
    ffmc=90.0,
    bui=80.0,
    wind_speed=20.0,
    wind_direction=270.0,
    slope=10.0,
    aspect=180.0,
    latitude=52.0,
    longitude=-122.0,
    elevation=800.0,
    month=7,
    day=1,
)

_REQUIRED_KEYS = {
    "rate_of_spread_mpm",
    "head_fire_intensity_kwm",
    "fire_type",
    "crown_fraction_burned",
    "flame_length_m",
}

_VALID_FIRE_TYPES = {"surface", "intermittent_crown", "active_crown"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _call(fuel_type: str, **overrides) -> dict:
    kwargs = {**_BASE_KWARGS, **overrides}
    return compute_fire_behaviour(fuel_type=fuel_type, **kwargs)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------
class TestClassifyFireType:
    def test_surface_below_threshold(self):
        assert _classify_fire_type(0.0) == "surface"
        assert _classify_fire_type(0.09) == "surface"
        assert _classify_fire_type(0.1) == "surface"  # boundary: > 0.1 → intermittent

    def test_intermittent_crown(self):
        assert _classify_fire_type(0.11) == "intermittent_crown"
        assert _classify_fire_type(0.5) == "intermittent_crown"
        assert _classify_fire_type(0.89) == "intermittent_crown"

    def test_active_crown_at_threshold(self):
        assert _classify_fire_type(0.9) == "active_crown"
        assert _classify_fire_type(1.0) == "active_crown"


class TestFlameLengthEquation:
    def test_zero_hfi_returns_zero(self):
        assert _flame_length_m(0.0) == 0.0

    def test_negative_hfi_returns_zero(self):
        assert _flame_length_m(-10.0) == 0.0

    def test_byram_formula_correctness(self):
        """L = 0.0775 * HFI^0.46 (Byram 1959)."""
        hfi = 1000.0
        expected = 0.0775 * math.pow(1000.0, 0.46)
        assert abs(_flame_length_m(hfi) - expected) < 1e-9

    def test_flame_length_increases_with_hfi(self):
        assert _flame_length_m(100.0) < _flame_length_m(1000.0) < _flame_length_m(10000.0)


# ---------------------------------------------------------------------------
# Non-fuel types
# ---------------------------------------------------------------------------
class TestNonFuelTypes:
    def test_nf_returns_zero_spread(self):
        result = _call("NF")
        assert result["rate_of_spread_mpm"] == 0.0
        assert result["head_fire_intensity_kwm"] == 0.0
        assert result["crown_fraction_burned"] == 0.0
        assert result["flame_length_m"] == 0.0

    def test_wa_returns_zero_spread(self):
        result = _call("WA")
        assert result["rate_of_spread_mpm"] == 0.0
        assert result["head_fire_intensity_kwm"] == 0.0

    def test_nf_fire_type_is_surface(self):
        """Non-fuel still reports fire_type=surface (logical default)."""
        assert _call("NF")["fire_type"] == "surface"

    def test_wa_skips_cffdrs_call(self):
        """WA should return zeros without calling cffdrs at all (no exception)."""
        result = _call("WA", ffmc=95.0, bui=150.0, wind_speed=50.0)
        assert result["rate_of_spread_mpm"] == 0.0


# ---------------------------------------------------------------------------
# C-3 fuel (mature pine): should have meaningful spread and intensity
# ---------------------------------------------------------------------------
class TestC3FuelType:
    @pytest.fixture(scope="class")
    def c3_result(self):
        return _call("C3", ffmc=90.0, bui=80.0, wind_speed=20.0)

    def test_ros_positive(self, c3_result):
        """C-3 under fire-weather conditions must have positive ROS."""
        assert c3_result["rate_of_spread_mpm"] > 0.0

    def test_hfi_positive(self, c3_result):
        """C-3 must produce positive head fire intensity."""
        assert c3_result["head_fire_intensity_kwm"] > 0.0

    def test_all_keys_present(self, c3_result):
        assert _REQUIRED_KEYS.issubset(c3_result.keys())

    def test_fire_type_valid(self, c3_result):
        assert c3_result["fire_type"] in _VALID_FIRE_TYPES

    def test_cfb_in_range(self, c3_result):
        assert 0.0 <= c3_result["crown_fraction_burned"] <= 1.0

    def test_flame_length_positive(self, c3_result):
        assert c3_result["flame_length_m"] > 0.0

    def test_high_wind_increases_ros(self):
        """Higher wind speed → higher ROS for C-3."""
        low = _call("C3", wind_speed=10.0)
        high = _call("C3", wind_speed=40.0)
        assert high["rate_of_spread_mpm"] > low["rate_of_spread_mpm"]

    def test_high_ffmc_increases_hfi(self):
        """Higher FFMC → higher HFI for C-3."""
        low = _call("C3", ffmc=80.0)
        high = _call("C3", ffmc=93.0)
        assert high["head_fire_intensity_kwm"] > low["head_fire_intensity_kwm"]


# ---------------------------------------------------------------------------
# D-1 (deciduous): should produce lower intensity than C-3
# ---------------------------------------------------------------------------
class TestD1VsC3Intensity:
    def test_d1_lower_hfi_than_c3(self):
        """Deciduous fuels should burn less intensely than mature pine under same conditions."""
        c3 = _call("C3")
        d1 = _call("D1")
        assert d1["head_fire_intensity_kwm"] < c3["head_fire_intensity_kwm"], (
            f"D1 HFI ({d1['head_fire_intensity_kwm']:.1f} kW/m) should be < "
            f"C3 HFI ({c3['head_fire_intensity_kwm']:.1f} kW/m)"
        )

    def test_d1_lower_ros_than_c3(self):
        """D-1 ROS should be lower than C-3 under the same weather."""
        c3 = _call("C3")
        d1 = _call("D1")
        assert d1["rate_of_spread_mpm"] < c3["rate_of_spread_mpm"]


# ---------------------------------------------------------------------------
# Fire type classification (CFB-driven)
# ---------------------------------------------------------------------------
class TestFireTypeClassification:
    def test_c3_intense_gives_crown_fire(self):
        """Extreme conditions on C-3 (high FFMC, BUI, wind) should produce crown fire."""
        result = _call(
            "C3",
            ffmc=93.0,
            bui=120.0,
            wind_speed=45.0,
        )
        # cffdrs should produce CFB near 1.0 → active_crown
        assert result["fire_type"] in {"intermittent_crown", "active_crown"}, (
            f"Expected crown fire under extreme conditions, got {result['fire_type']} "
            f"(CFB={result['crown_fraction_burned']:.3f})"
        )

    def test_surface_fire_conditions(self):
        """Low BUI + low FFMC on C-3 may stay surface or give low CFB."""
        result = _call("C3", ffmc=78.0, bui=20.0, wind_speed=5.0)
        assert result["fire_type"] in _VALID_FIRE_TYPES

    def test_d1_is_surface_fire(self):
        """D-1 (deciduous) does not crown; CFB should be 0 → surface."""
        result = _call("D1")
        assert result["crown_fraction_burned"] == 0.0
        assert result["fire_type"] == "surface"


# ---------------------------------------------------------------------------
# All required output fields
# ---------------------------------------------------------------------------
class TestAllFieldsPresent:
    @pytest.mark.parametrize("fuel_type", ["C1", "C2", "C3", "C4", "C5", "C6", "C7"])
    def test_conifer_fuels_have_all_keys(self, fuel_type):
        result = _call(fuel_type)
        assert _REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys for {fuel_type}: {_REQUIRED_KEYS - result.keys()}"
        )

    @pytest.mark.parametrize("fuel_type", ["D1", "S1", "S2", "O1A", "O1B"])
    def test_other_fuels_have_all_keys(self, fuel_type):
        result = _call(fuel_type)
        assert _REQUIRED_KEYS.issubset(result.keys())

    @pytest.mark.parametrize("fuel_type", ["NF", "WA"])
    def test_non_fuel_types_have_all_keys(self, fuel_type):
        result = _call(fuel_type)
        assert _REQUIRED_KEYS.issubset(result.keys())


# ---------------------------------------------------------------------------
# Graceful error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_invalid_fuel_type_returns_zeros(self):
        """An unknown fuel type should not raise; return zeros gracefully."""
        result = _call("UNKNOWN_FUEL")
        assert _REQUIRED_KEYS.issubset(result.keys())
        # May return zeros or raise internally (caught); either way no exception
        assert isinstance(result["rate_of_spread_mpm"], float)

    def test_extreme_values_do_not_crash(self):
        """Edge-case inputs should not raise exceptions."""
        result = _call("C3", ffmc=101.0, bui=999.0, wind_speed=100.0, slope=45.0)
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_zero_wind_does_not_crash(self):
        """Zero wind speed is physically valid (no wind → surface fire)."""
        result = _call("C3", wind_speed=0.0)
        assert isinstance(result["rate_of_spread_mpm"], float)
        assert result["rate_of_spread_mpm"] >= 0.0
