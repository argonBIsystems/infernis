"""Tests for pure fire statistics computation functions."""

import pytest

from infernis.services.fire_statistics import (
    FIRE_SEASON_DAYS_10YR,
    aggregate_fires_by_tier,
    compute_fire_regime,
    compute_susceptibility,
    label_from_percentile,
)


# ---------------------------------------------------------------------------
# aggregate_fires_by_tier
# ---------------------------------------------------------------------------


class TestAggregateFiresByTier:
    def test_empty_fires(self):
        result = aggregate_fires_by_tier([], current_year=2026)
        assert result["fires_10yr"] == {
            "count": 0,
            "nearest_km": None,
            "largest_ha": None,
            "causes": {},
        }
        assert result["fires_30yr"] == {
            "count": 0,
            "nearest_km": None,
            "largest_ha": None,
            "causes": {},
        }
        assert result["fires_all"]["count"] == 0
        assert result["fires_all"]["record_start"] is None

    def test_single_recent_fire_appears_in_all_tiers(self):
        fires = [{"year": 2020, "distance_km": 5.0, "size_ha": 100.0, "cause": "lightning"}]
        result = aggregate_fires_by_tier(fires, current_year=2026)
        assert result["fires_10yr"]["count"] == 1
        assert result["fires_30yr"]["count"] == 1
        assert result["fires_all"]["count"] == 1

    def test_old_fire_only_in_all_time(self):
        fires = [{"year": 1980, "distance_km": 3.0, "size_ha": 50.0, "cause": "human"}]
        result = aggregate_fires_by_tier(fires, current_year=2026)
        assert result["fires_10yr"]["count"] == 0
        assert result["fires_30yr"]["count"] == 0
        assert result["fires_all"]["count"] == 1

    def test_fire_at_30yr_boundary_included(self):
        # current_year - 30 = 1996; fires from 1996 onward should appear in 30yr
        fires = [{"year": 1996, "distance_km": 2.0, "size_ha": 200.0, "cause": "lightning"}]
        result = aggregate_fires_by_tier(fires, current_year=2026)
        assert result["fires_30yr"]["count"] == 1
        assert result["fires_10yr"]["count"] == 0

    def test_fire_at_10yr_boundary_included(self):
        fires = [{"year": 2016, "distance_km": 1.5, "size_ha": 300.0, "cause": "human"}]
        result = aggregate_fires_by_tier(fires, current_year=2026)
        assert result["fires_10yr"]["count"] == 1

    def test_nearest_km_is_minimum(self):
        fires = [
            {"year": 2022, "distance_km": 8.0, "size_ha": 10.0, "cause": "lightning"},
            {"year": 2021, "distance_km": 2.0, "size_ha": 20.0, "cause": "lightning"},
        ]
        result = aggregate_fires_by_tier(fires, current_year=2026)
        assert result["fires_10yr"]["nearest_km"] == pytest.approx(2.0)

    def test_largest_ha_is_maximum(self):
        fires = [
            {"year": 2022, "distance_km": 5.0, "size_ha": 500.0, "cause": "lightning"},
            {"year": 2021, "distance_km": 3.0, "size_ha": 1200.0, "cause": "human"},
        ]
        result = aggregate_fires_by_tier(fires, current_year=2026)
        assert result["fires_10yr"]["largest_ha"] == pytest.approx(1200.0)

    def test_cause_breakdown(self):
        fires = [
            {"year": 2022, "distance_km": 5.0, "size_ha": 100.0, "cause": "lightning"},
            {"year": 2021, "distance_km": 3.0, "size_ha": 200.0, "cause": "lightning"},
            {"year": 2020, "distance_km": 4.0, "size_ha": 50.0, "cause": "human"},
        ]
        result = aggregate_fires_by_tier(fires, current_year=2026)
        causes = result["fires_10yr"]["causes"]
        assert causes["lightning"] == 2
        assert causes["human"] == 1

    def test_record_start_is_earliest_year(self):
        fires = [
            {"year": 2010, "distance_km": 5.0, "size_ha": 10.0, "cause": "lightning"},
            {"year": 1990, "distance_km": 8.0, "size_ha": 20.0, "cause": "human"},
            {"year": 2000, "distance_km": 6.0, "size_ha": 30.0, "cause": "lightning"},
        ]
        result = aggregate_fires_by_tier(fires, current_year=2026)
        assert result["fires_all"]["record_start"] == 1990

    def test_record_start_not_present_on_10yr_or_30yr(self):
        fires = [{"year": 2022, "distance_km": 5.0, "size_ha": 10.0, "cause": "lightning"}]
        result = aggregate_fires_by_tier(fires, current_year=2026)
        assert "record_start" not in result["fires_10yr"]
        assert "record_start" not in result["fires_30yr"]

    def test_mixed_tiers(self):
        fires = [
            {"year": 2024, "distance_km": 1.0, "size_ha": 10.0, "cause": "lightning"},  # 10yr
            {"year": 2000, "distance_km": 9.0, "size_ha": 500.0, "cause": "human"},  # 30yr only
            {"year": 1980, "distance_km": 7.0, "size_ha": 1000.0, "cause": "lightning"},  # all only
        ]
        result = aggregate_fires_by_tier(fires, current_year=2026)
        assert result["fires_10yr"]["count"] == 1
        assert result["fires_30yr"]["count"] == 2
        assert result["fires_all"]["count"] == 3


# ---------------------------------------------------------------------------
# compute_susceptibility
# ---------------------------------------------------------------------------


class TestComputeSusceptibility:
    def setup_method(self):
        self.group_rates = {"IDF_C3": 0.005, "SBS_M2": 0.003}
        self.zone_rates = {"IDF": 0.002, "SBS": 0.001}
        self.group_cell_counts = {"IDF_C3": 15, "SBS_M2": 5}

    def test_cell_level_with_2_fires(self):
        score, basis = compute_susceptibility(
            fires_10yr_count=2,
            bec_zone="IDF",
            fuel_type="C3",
            group_rates=self.group_rates,
            zone_rates=self.zone_rates,
            group_cell_counts=self.group_cell_counts,
        )
        assert basis == "cell"
        assert score == pytest.approx(2 / FIRE_SEASON_DAYS_10YR)

    def test_cell_level_with_many_fires(self):
        score, basis = compute_susceptibility(
            fires_10yr_count=10,
            bec_zone="IDF",
            fuel_type="C3",
            group_rates=self.group_rates,
            zone_rates=self.zone_rates,
            group_cell_counts=self.group_cell_counts,
        )
        assert basis == "cell"
        assert score == pytest.approx(10 / FIRE_SEASON_DAYS_10YR)

    def test_bec_fuel_group_fallback(self):
        # fires_10yr_count=1 (<2), group has 15 cells (>=10)
        score, basis = compute_susceptibility(
            fires_10yr_count=1,
            bec_zone="IDF",
            fuel_type="C3",
            group_rates=self.group_rates,
            zone_rates=self.zone_rates,
            group_cell_counts=self.group_cell_counts,
        )
        assert basis == "bec_fuel"
        assert score == pytest.approx(0.005)

    def test_bec_zone_fallback_when_group_too_small(self):
        # SBS_M2 has only 5 cells (<10), should fall back to zone
        score, basis = compute_susceptibility(
            fires_10yr_count=0,
            bec_zone="SBS",
            fuel_type="M2",
            group_rates=self.group_rates,
            zone_rates=self.zone_rates,
            group_cell_counts=self.group_cell_counts,
        )
        assert basis == "bec"
        assert score == pytest.approx(0.001)

    def test_bec_zone_fallback_when_group_missing(self):
        # Group key doesn't exist in group_rates at all
        score, basis = compute_susceptibility(
            fires_10yr_count=0,
            bec_zone="IDF",
            fuel_type="C5",
            group_rates=self.group_rates,
            zone_rates=self.zone_rates,
            group_cell_counts={"IDF_C5": 20},
        )
        assert basis == "bec"
        assert score == pytest.approx(0.002)

    def test_zero_fires_unknown_zone(self):
        # Zone not in zone_rates → score 0.0
        score, basis = compute_susceptibility(
            fires_10yr_count=0,
            bec_zone="UNKNOWN",
            fuel_type="X1",
            group_rates={},
            zone_rates={},
            group_cell_counts={},
        )
        assert basis == "bec"
        assert score == pytest.approx(0.0)

    def test_exactly_1_fire_uses_group_not_cell(self):
        # 1 fire is below the cell threshold of 2
        score, basis = compute_susceptibility(
            fires_10yr_count=1,
            bec_zone="IDF",
            fuel_type="C3",
            group_rates=self.group_rates,
            zone_rates=self.zone_rates,
            group_cell_counts=self.group_cell_counts,
        )
        assert basis == "bec_fuel"


# ---------------------------------------------------------------------------
# compute_fire_regime
# ---------------------------------------------------------------------------


class TestComputeFireRegime:
    def test_basic_computation(self):
        result = compute_fire_regime(
            zone_area_ha=1_000_000,
            total_burned_ha=50_000,
            years_of_record=30,
            fire_sizes=[100.0, 200.0, 500.0],
            causes=["lightning", "lightning", "human"],
        )
        # annual_burn = 50000/30 ≈ 1666.7 → return = 1000000/1666.7 ≈ 600
        assert result["mean_return_years"] == pytest.approx(600.0, rel=0.01)
        assert result["typical_severity"] == "moderate"
        assert result["dominant_cause"] == "lightning"

    def test_low_severity_small_fires(self):
        result = compute_fire_regime(
            zone_area_ha=500_000,
            total_burned_ha=10_000,
            years_of_record=20,
            fire_sizes=[10.0, 20.0, 50.0, 80.0],
            causes=["human"],
        )
        assert result["typical_severity"] == "low"

    def test_high_severity_large_fires(self):
        result = compute_fire_regime(
            zone_area_ha=2_000_000,
            total_burned_ha=500_000,
            years_of_record=50,
            fire_sizes=[1500.0, 2000.0, 5000.0],
            causes=["lightning"],
        )
        assert result["typical_severity"] == "high"

    def test_median_exactly_100_is_moderate(self):
        # median([50, 100, 150]) = 100 → "moderate"
        result = compute_fire_regime(
            zone_area_ha=1_000_000,
            total_burned_ha=10_000,
            years_of_record=10,
            fire_sizes=[50.0, 100.0, 150.0],
            causes=["lightning"],
        )
        assert result["typical_severity"] == "moderate"

    def test_median_above_1000_is_high(self):
        # median([1001, 2000, 3000]) > 1000 → "high"
        result = compute_fire_regime(
            zone_area_ha=1_000_000,
            total_burned_ha=10_000,
            years_of_record=10,
            fire_sizes=[1001.0, 2000.0, 3000.0],
            causes=["lightning"],
        )
        assert result["typical_severity"] == "high"

    def test_empty_fire_sizes_defaults_to_low(self):
        result = compute_fire_regime(
            zone_area_ha=1_000_000,
            total_burned_ha=0,
            years_of_record=10,
            fire_sizes=[],
            causes=[],
        )
        assert result["typical_severity"] == "low"
        assert result["dominant_cause"] == "unknown"
        assert result["mean_return_years"] is None

    def test_dominant_cause_human(self):
        result = compute_fire_regime(
            zone_area_ha=500_000,
            total_burned_ha=100_000,
            years_of_record=30,
            fire_sizes=[100.0],
            causes=["human", "human", "lightning"],
        )
        assert result["dominant_cause"] == "human"


# ---------------------------------------------------------------------------
# label_from_percentile
# ---------------------------------------------------------------------------


class TestLabelFromPercentile:
    def test_well_below_average(self):
        assert label_from_percentile(0) == "WELL_BELOW_AVERAGE"
        assert label_from_percentile(10) == "WELL_BELOW_AVERAGE"
        assert label_from_percentile(19.9) == "WELL_BELOW_AVERAGE"

    def test_below_average(self):
        assert label_from_percentile(20) == "BELOW_AVERAGE"
        assert label_from_percentile(30) == "BELOW_AVERAGE"
        assert label_from_percentile(39.9) == "BELOW_AVERAGE"

    def test_average(self):
        assert label_from_percentile(40) == "AVERAGE"
        assert label_from_percentile(50) == "AVERAGE"
        assert label_from_percentile(59.9) == "AVERAGE"

    def test_above_average(self):
        assert label_from_percentile(60) == "ABOVE_AVERAGE"
        assert label_from_percentile(70) == "ABOVE_AVERAGE"
        assert label_from_percentile(79.9) == "ABOVE_AVERAGE"

    def test_well_above_average(self):
        assert label_from_percentile(80) == "WELL_ABOVE_AVERAGE"
        assert label_from_percentile(90) == "WELL_ABOVE_AVERAGE"
        assert label_from_percentile(100) == "WELL_ABOVE_AVERAGE"
