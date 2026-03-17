"""Tests for the diurnal FFMC adjustment module.

Based on CFFDRS Red Book 3rd edition (2018) diurnal FFMC curves.
Afternoon FFMC (14:00) is 15-20 points higher than daily under dry conditions.
"""

import numpy as np
import pytest

from infernis.services.diurnal_ffmc import adjust_ffmc_diurnal


class TestDiurnalFFMCDirectionality:
    """Validate that adjustments move in the correct direction."""

    def test_afternoon_higher_than_daily_in_dry_conditions(self):
        """At 14:00, FFMC should be higher than the daily value in dry conditions."""
        daily_ffmc = 85.0
        temp_c = 30.0   # hot
        rh_pct = 20.0   # very dry
        adjusted = adjust_ffmc_diurnal(daily_ffmc, temp_c, rh_pct, hour=14)
        assert adjusted > daily_ffmc, (
            f"Afternoon FFMC {adjusted:.2f} should exceed daily {daily_ffmc:.2f} in dry conditions"
        )

    def test_morning_lower_than_daily_due_to_overnight_recovery(self):
        """At 06:00, FFMC should be lower due to overnight humidity recovery."""
        daily_ffmc = 85.0
        temp_c = 15.0   # cooler morning
        rh_pct = 70.0   # higher overnight RH
        adjusted = adjust_ffmc_diurnal(daily_ffmc, temp_c, rh_pct, hour=6)
        assert adjusted < daily_ffmc, (
            f"Morning FFMC {adjusted:.2f} should be below daily {daily_ffmc:.2f}"
        )

    def test_noon_near_daily_average(self):
        """At 12:00, FFMC should be closer to daily than the afternoon peak value.

        Noon marks the transition from morning recovery to afternoon peak drying.
        The adjustment is noticeably less than the 14:00 peak but already positive.
        We verify noon is between the morning low and afternoon high — not that it
        equals the daily value (it shouldn't; peak drying has already begun).
        """
        daily_ffmc = 85.0
        temp_c = 22.0
        rh_pct = 40.0
        noon_adj = adjust_ffmc_diurnal(daily_ffmc, temp_c, rh_pct, hour=12)
        pm_adj = adjust_ffmc_diurnal(daily_ffmc, temp_c, rh_pct, hour=14)
        am_adj = adjust_ffmc_diurnal(daily_ffmc, temp_c, rh_pct, hour=6)
        # Noon is between morning minimum and afternoon peak
        assert am_adj < noon_adj < pm_adj, (
            f"Expected am ({am_adj:.2f}) < noon ({noon_adj:.2f}) < pm ({pm_adj:.2f})"
        )
        # And the noon adjustment is less than 10 points from daily
        assert abs(noon_adj - daily_ffmc) < 10.0, (
            f"Noon FFMC {noon_adj:.2f} should be within 10 of daily {daily_ffmc:.2f}"
        )

    def test_afternoon_increase_larger_with_hotter_drier_conditions(self):
        """Higher temp + lower RH should produce a larger afternoon FFMC increase."""
        daily_ffmc = 85.0
        hour = 14

        mild_adj = adjust_ffmc_diurnal(daily_ffmc, temp_c=20.0, rh_pct=50.0, hour=hour)
        extreme_adj = adjust_ffmc_diurnal(daily_ffmc, temp_c=35.0, rh_pct=15.0, hour=hour)

        mild_delta = mild_adj - daily_ffmc
        extreme_delta = extreme_adj - daily_ffmc

        assert extreme_delta > mild_delta, (
            f"Extreme conditions delta {extreme_delta:.2f} should exceed mild {mild_delta:.2f}"
        )


class TestDiurnalFFMCClamping:
    """Validate output is clamped to [0, 101]."""

    def test_output_never_exceeds_101(self):
        """FFMC output is clamped to a maximum of 101."""
        # Start near the ceiling with extreme drying conditions
        adjusted = adjust_ffmc_diurnal(daily_ffmc=100.0, temp_c=40.0, rh_pct=5.0, hour=14)
        assert adjusted <= 101.0, f"FFMC must not exceed 101, got {adjusted:.2f}"

    def test_output_never_below_zero(self):
        """FFMC output is clamped to a minimum of 0."""
        # Start near zero with extreme wetting
        adjusted = adjust_ffmc_diurnal(daily_ffmc=1.0, temp_c=5.0, rh_pct=99.0, hour=3)
        assert adjusted >= 0.0, f"FFMC must not go below 0, got {adjusted:.2f}"

    def test_clamp_upper_boundary(self):
        """Even extreme conditions cannot push FFMC above 101."""
        for hour in [13, 14, 15]:
            result = adjust_ffmc_diurnal(101.0, temp_c=45.0, rh_pct=2.0, hour=hour)
            assert result <= 101.0

    def test_clamp_lower_boundary(self):
        """Even extreme wetting cannot push FFMC below 0."""
        for hour in [2, 3, 4]:
            result = adjust_ffmc_diurnal(0.0, temp_c=0.0, rh_pct=100.0, hour=hour)
            assert result >= 0.0


class TestDiurnalFFMCVectorized:
    """Vectorized (numpy array) inputs must match scalar results."""

    def test_vectorized_matches_scalar(self):
        """Array inputs must produce results matching element-wise scalar calls."""
        daily_ffmc = np.array([80.0, 85.0, 90.0, 70.0])
        temp_c = np.array([25.0, 32.0, 18.0, 28.0])
        rh_pct = np.array([35.0, 20.0, 60.0, 30.0])
        hour = 14

        vec_result = adjust_ffmc_diurnal(daily_ffmc, temp_c, rh_pct, hour)

        scalar_results = np.array([
            adjust_ffmc_diurnal(float(daily_ffmc[i]), float(temp_c[i]), float(rh_pct[i]), hour)
            for i in range(len(daily_ffmc))
        ])

        np.testing.assert_allclose(
            vec_result, scalar_results, rtol=1e-6,
            err_msg="Vectorized result must match element-wise scalar results"
        )

    def test_vectorized_returns_array_for_array_input(self):
        """Passing numpy arrays should return a numpy array."""
        result = adjust_ffmc_diurnal(
            np.array([85.0, 90.0]),
            np.array([25.0, 30.0]),
            np.array([40.0, 20.0]),
            hour=14,
        )
        assert isinstance(result, np.ndarray), "Result should be a numpy array for array input"
        assert result.shape == (2,)

    def test_scalar_returns_scalar_or_float(self):
        """Passing scalar inputs should return a scalar (float or 0-d array)."""
        result = adjust_ffmc_diurnal(85.0, 25.0, 40.0, hour=14)
        # Accept Python float or numpy scalar
        assert np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0), (
            f"Expected scalar, got {type(result)}"
        )

    def test_vectorized_clamped_to_valid_range(self):
        """All array outputs should be in [0, 101]."""
        n = 50
        rng = np.random.default_rng(42)
        daily_ffmc = rng.uniform(0, 101, n)
        temp_c = rng.uniform(-5, 45, n)
        rh_pct = rng.uniform(5, 100, n)

        for hour in range(24):
            result = adjust_ffmc_diurnal(daily_ffmc, temp_c, rh_pct, hour)
            assert np.all(result >= 0.0), f"Hour {hour}: found value below 0"
            assert np.all(result <= 101.0), f"Hour {hour}: found value above 101"


class TestDiurnalFFMCHourCoverage:
    """Sanity checks across all 24 hours."""

    def test_all_hours_valid(self):
        """adjust_ffmc_diurnal should accept all hours 0-23 without error."""
        for hour in range(24):
            result = adjust_ffmc_diurnal(85.0, 25.0, 40.0, hour=hour)
            assert 0.0 <= result <= 101.0, f"Hour {hour} produced out-of-range: {result}"

    def test_peak_adjustment_near_midday(self):
        """The maximum adjustment should occur around midday (10:00-16:00)."""
        daily_ffmc = 85.0
        temp_c = 32.0
        rh_pct = 15.0

        adjustments = {
            hour: adjust_ffmc_diurnal(daily_ffmc, temp_c, rh_pct, hour) - daily_ffmc
            for hour in range(24)
        }

        peak_hour = max(adjustments, key=adjustments.get)
        assert 10 <= peak_hour <= 16, (
            f"Peak adjustment expected between 10:00-16:00, got hour {peak_hour}"
        )

    def test_minimum_adjustment_in_early_morning(self):
        """The minimum (most negative) adjustment should occur in early morning (00:00-08:00)."""
        daily_ffmc = 85.0
        temp_c = 25.0
        rh_pct = 40.0

        adjustments = {
            hour: adjust_ffmc_diurnal(daily_ffmc, temp_c, rh_pct, hour) - daily_ffmc
            for hour in range(24)
        }

        min_hour = min(adjustments, key=adjustments.get)
        assert 0 <= min_hour <= 8, (
            f"Minimum adjustment expected between 00:00-08:00, got hour {min_hour}"
        )
