"""Pure functions for fire statistics computation.

No DB dependencies — all functions operate on plain Python data structures,
making them easy to test and reuse.
"""

from __future__ import annotations

from collections import Counter
from statistics import median

# 214 fire-season days × 10 years
FIRE_SEASON_DAYS_10YR = 2140


def aggregate_fires_by_tier(
    fires: list[dict],
    current_year: int = 2026,
) -> dict:
    """Aggregate fire records into 10yr, 30yr, and all-time tiers.

    Args:
        fires: List of dicts with keys: year, distance_km, size_ha, cause.
        current_year: Reference year for computing look-back windows.

    Returns:
        Dict with keys fires_10yr, fires_30yr, fires_all, each containing:
            count, nearest_km, largest_ha, causes (dict).
        fires_all also contains record_start (earliest year in the matched set).
    """
    cutoff_10yr = current_year - 10
    cutoff_30yr = current_year - 30

    def _summarise(subset: list[dict], include_record_start: bool = False) -> dict:
        if not subset:
            result = {
                "count": 0,
                "nearest_km": None,
                "largest_ha": None,
                "causes": {},
            }
            if include_record_start:
                result["record_start"] = None
            return result

        result = {
            "count": len(subset),
            "nearest_km": min(f["distance_km"] for f in subset),
            "largest_ha": max(f["size_ha"] for f in subset if f.get("size_ha") is not None),
            "causes": dict(Counter(f["cause"] for f in subset if f.get("cause"))),
        }
        if include_record_start:
            result["record_start"] = min(f["year"] for f in subset)
        return result

    fires_10yr = [f for f in fires if f["year"] >= cutoff_10yr]
    fires_30yr = [f for f in fires if f["year"] >= cutoff_30yr]
    fires_all = fires

    return {
        "fires_10yr": _summarise(fires_10yr),
        "fires_30yr": _summarise(fires_30yr),
        "fires_all": _summarise(fires_all, include_record_start=True),
    }


def compute_susceptibility(
    fires_10yr_count: int,
    bec_zone: str,
    fuel_type: str,
    group_rates: dict[str, float],
    zone_rates: dict[str, float],
    group_cell_counts: dict[str, int],
) -> tuple[float, str]:
    """Compute fire susceptibility score with hierarchical fallback.

    Priority:
      1. Cell level: if fires_10yr_count >= 2, use fires_10yr_count / FIRE_SEASON_DAYS_10YR
      2. BEC+fuel group: if group has >= 10 cells, use group_rates["{bec_zone}_{fuel_type}"]
      3. BEC zone: fall back to zone_rates[bec_zone]

    Returns:
        (score, basis) where basis is one of "cell", "bec_fuel", "bec".
    """
    if fires_10yr_count >= 2:
        return fires_10yr_count / FIRE_SEASON_DAYS_10YR, "cell"

    group_key = f"{bec_zone}_{fuel_type}"
    if group_cell_counts.get(group_key, 0) >= 10 and group_key in group_rates:
        return group_rates[group_key], "bec_fuel"

    return zone_rates.get(bec_zone, 0.0), "bec"


def compute_fire_regime(
    zone_area_ha: float,
    total_burned_ha: float,
    years_of_record: int,
    fire_sizes: list[float],
    causes: list[str],
) -> dict:
    """Compute fire regime statistics for a BEC zone.

    Args:
        zone_area_ha: Total area of the BEC zone in hectares.
        total_burned_ha: Cumulative hectares burned in the zone over the record.
        years_of_record: Number of years in the fire record.
        fire_sizes: List of individual fire sizes (ha) within the zone.
        causes: List of cause strings for all fires in the zone.

    Returns:
        Dict with keys:
            mean_return_years: Average fire return interval in years.
            typical_severity: "low", "moderate", or "high".
            dominant_cause: Most common cause string.
    """
    # Mean return interval: area / annual_burn_rate
    annual_burn_rate = total_burned_ha / years_of_record if years_of_record > 0 else 0.0
    mean_return_years = zone_area_ha / annual_burn_rate if annual_burn_rate > 0 else None

    # Severity from median fire size
    if fire_sizes:
        med = median(fire_sizes)
        if med < 100:
            typical_severity = "low"
        elif med <= 1000:
            typical_severity = "moderate"
        else:
            typical_severity = "high"
    else:
        typical_severity = "low"

    # Dominant cause
    if causes:
        dominant_cause = Counter(causes).most_common(1)[0][0]
    else:
        dominant_cause = "unknown"

    return {
        "mean_return_years": mean_return_years,
        "typical_severity": typical_severity,
        "dominant_cause": dominant_cause,
    }


def label_from_percentile(percentile: float) -> str:
    """Convert a numeric percentile (0–100) to a descriptive label.

    Ranges:
        <20   → WELL_BELOW_AVERAGE
        20–40 → BELOW_AVERAGE
        40–60 → AVERAGE
        60–80 → ABOVE_AVERAGE
        >=80  → WELL_ABOVE_AVERAGE
    """
    if percentile < 20:
        return "WELL_BELOW_AVERAGE"
    elif percentile < 40:
        return "BELOW_AVERAGE"
    elif percentile < 60:
        return "AVERAGE"
    elif percentile < 80:
        return "ABOVE_AVERAGE"
    else:
        return "WELL_ABOVE_AVERAGE"
