"""Offline fire statistics computation pipeline.

Loads fire_history + grid_cells from the database, performs spatial matching
per cell (Haversine within 10km), aggregates statistics by tier, computes
susceptibility scores with hierarchical fallback, derives fire regime metrics
per BEC zone, computes percentiles, then writes results to the fire_statistics
table and caches per-cell JSON in Redis.

Intended to be run offline (nightly or on demand) via the admin CLI:
    python -m infernis.admin compute_fire_stats
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np

from infernis.services.cache import get_redis

logger = logging.getLogger(__name__)

# Earth radius used in Haversine
_EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Pure spatial helpers
# ---------------------------------------------------------------------------


def compute_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two WGS-84 points (Haversine)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_KM * c


def match_fires_to_cell(
    cell_lat: float,
    cell_lon: float,
    fires: list[dict],
    radius_km: float = 10.0,
) -> list[dict]:
    """Return fires within radius_km of the cell centroid, adding distance_km field.

    Args:
        cell_lat: Centroid latitude of the grid cell.
        cell_lon: Centroid longitude of the grid cell.
        fires: List of fire dicts; each must have 'lat' and 'lon' fields.
        radius_km: Search radius in kilometres (default 10.0).

    Returns:
        Filtered list of fire dicts, each augmented with a 'distance_km' key.
    """
    matched = []
    for fire in fires:
        dist = compute_distance_km(cell_lat, cell_lon, fire["lat"], fire["lon"])
        if dist <= radius_km:
            matched.append({**fire, "distance_km": dist})
    return matched


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _load_fire_history(db) -> list[dict]:
    """Load fire history from DB, falling back to raw files if DB is empty."""
    from infernis.db.fire_history import FireHistoryDB

    logger.info("Loading fire history …")
    fire_rows = db.query(FireHistoryDB).limit(1).all()

    if fire_rows:
        # DB has data — load from DB
        fire_rows = db.query(FireHistoryDB).all()
        fires = [
            {
                "lat": f.lat,
                "lon": f.lon,
                "year": f.year,
                "size_ha": f.size_ha or 0.0,
                "cause": f.cause or "unknown",
            }
            for f in fire_rows
        ]
        logger.info("Loaded %d fire records from database", len(fires))
        return fires

    # Fall back to raw files (same approach as training/feature_builder.py)
    logger.info("No fire records in DB — loading from raw files …")
    try:
        from infernis.training.feature_builder import FeatureBuilder

        builder = FeatureBuilder()
        df = builder.load_fire_history()
        if df.empty:
            logger.warning("No fire history data found in files either")
            return []
        fires = []
        for _, row in df.iterrows():
            year = row["date"].year if hasattr(row["date"], "year") else int(str(row["date"])[:4])
            fires.append({
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "year": year,
                "size_ha": float(row.get("size_ha", 0) or 0),
                "cause": str(row.get("cause", "unknown") or "unknown"),
            })
        logger.info("Loaded %d fire records from raw files", len(fires))
        return fires
    except Exception as e:
        logger.error("Failed to load fire history from files: %s", e)
        return []


def _load_grid_cells(db) -> list[dict]:
    """Load grid cells from DB, falling back to parquet/in-memory generation."""
    from infernis.db.tables import GridCellDB

    logger.info("Loading grid cells …")
    cell_count = db.query(GridCellDB).count()

    if cell_count > 0:
        cell_rows = db.query(GridCellDB).all()
        cells = [
            {
                "cell_id": c.cell_id,
                "lat": c.lat,
                "lon": c.lon,
                "bec_zone": c.bec_zone or "UNKNOWN",
                "fuel_type": c.fuel_type or "C3",
            }
            for c in cell_rows
        ]
        logger.info("Loaded %d grid cells from database", len(cells))
        return cells

    # Fall back to parquet or in-memory generation
    logger.info("No grid cells in DB — loading from parquet/generator …")
    try:
        from infernis.pipelines.runner import _load_grid

        grid_df = _load_grid()
        if grid_df is None or len(grid_df) == 0:
            logger.warning("No grid cells available")
            return []
        cells = []
        for _, row in grid_df.iterrows():
            cells.append({
                "cell_id": row["cell_id"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "bec_zone": str(row.get("bec_zone", "UNKNOWN") or "UNKNOWN"),
                "fuel_type": str(row.get("fuel_type", "C3") or "C3"),
            })
        logger.info("Loaded %d grid cells from parquet/generator", len(cells))
        return cells
    except Exception as e:
        logger.error("Failed to load grid cells: %s", e)
        return []


def run_fire_stats_pipeline() -> None:
    """Pre-compute fire statistics for every grid cell and persist results.

    Steps:
      1. Load fire_history + grid_cells from DB.
      2. Group cells by BEC zone.
      3. For each zone: bbox pre-filter fires, then Haversine-match per cell.
      4. Aggregate by tier (aggregate_fires_by_tier).
      5. Compute group/zone susceptibility rates.
      6. Assign susceptibility per cell (compute_susceptibility).
      7. Compute fire regime per BEC zone (compute_fire_regime).
      8. Compute percentiles with numpy.
      9. Write to fire_statistics table (delete + batch insert).
     10. Cache to Redis (fire_stats:{cell_id} with no TTL).
    """
    from infernis.db.engine import SessionLocal
    from infernis.db.tables import FireStatisticsDB
    from infernis.services.fire_statistics import (
        FIRE_SEASON_DAYS_10YR,
        aggregate_fires_by_tier,
        compute_fire_regime,
        compute_susceptibility,
        label_from_percentile,
    )

    db = SessionLocal()
    try:
        # Load fire history — try DB first, fall back to raw files
        fires_all = _load_fire_history(db)

        # Load grid cells — try DB first, fall back to parquet/in-memory
        cells = _load_grid_cells(db)

        # ------------------------------------------------------------------
        # Step 2-4: Match fires to cells and aggregate by tier
        # ------------------------------------------------------------------
        current_year = datetime.utcnow().year

        # Per-cell aggregated tier data: {cell_id: {fires_10yr, fires_30yr, fires_all}}
        cell_tiers: dict[str, dict] = {}

        # Group cells by BEC zone for bbox pre-filtering
        zones: dict[str, list[dict]] = defaultdict(list)
        for cell in cells:
            zones[cell["bec_zone"]].append(cell)

        for bec_zone, zone_cells in zones.items():
            # Rough bbox for the zone to pre-filter fires (±0.1° buffer)
            lats = [c["lat"] for c in zone_cells]
            lons = [c["lon"] for c in zone_cells]
            lat_min, lat_max = min(lats) - 0.1, max(lats) + 0.1
            lon_min, lon_max = min(lons) - 0.1, max(lons) + 0.1

            # Pre-filter fires to zone bbox
            zone_fires = [
                f
                for f in fires_all
                if lat_min <= f["lat"] <= lat_max and lon_min <= f["lon"] <= lon_max
            ]

            for cell in zone_cells:
                matched = match_fires_to_cell(cell["lat"], cell["lon"], zone_fires)
                tiers = aggregate_fires_by_tier(matched, current_year=current_year)
                cell_tiers[cell["cell_id"]] = tiers

        logger.info("Tier aggregation complete for %d cells", len(cell_tiers))

        # ------------------------------------------------------------------
        # Step 5: Compute group/zone susceptibility rates
        # ------------------------------------------------------------------
        # Accumulate (fires_10yr_count, cell_count) per group key
        group_fire_counts: dict[str, int] = defaultdict(int)
        group_cell_counts: dict[str, int] = defaultdict(int)
        zone_fire_counts: dict[str, int] = defaultdict(int)
        zone_cell_counts: dict[str, int] = defaultdict(int)

        cell_map = {c["cell_id"]: c for c in cells}

        for cell_id, tiers in cell_tiers.items():
            c = cell_map[cell_id]
            count_10yr = tiers["fires_10yr"]["count"]
            group_key = f"{c['bec_zone']}_{c['fuel_type']}"
            group_fire_counts[group_key] += count_10yr
            group_cell_counts[group_key] += 1
            zone_fire_counts[c["bec_zone"]] += count_10yr
            zone_cell_counts[c["bec_zone"]] += 1

        # Rate = total fires / (cell_count * FIRE_SEASON_DAYS_10YR)
        group_rates: dict[str, float] = {
            k: group_fire_counts[k] / (group_cell_counts[k] * FIRE_SEASON_DAYS_10YR)
            for k in group_cell_counts
            if group_cell_counts[k] > 0
        }
        zone_rates: dict[str, float] = {
            z: zone_fire_counts[z] / (zone_cell_counts[z] * FIRE_SEASON_DAYS_10YR)
            for z in zone_cell_counts
            if zone_cell_counts[z] > 0
        }

        # ------------------------------------------------------------------
        # Step 6: Assign susceptibility per cell
        # ------------------------------------------------------------------
        cell_susceptibility: dict[str, tuple[float, str]] = {}
        for cell_id, tiers in cell_tiers.items():
            c = cell_map[cell_id]
            score, basis = compute_susceptibility(
                fires_10yr_count=tiers["fires_10yr"]["count"],
                bec_zone=c["bec_zone"],
                fuel_type=c["fuel_type"],
                group_rates=group_rates,
                zone_rates=zone_rates,
                group_cell_counts=group_cell_counts,
            )
            cell_susceptibility[cell_id] = (score, basis)

        # ------------------------------------------------------------------
        # Step 7: Fire regime per BEC zone
        # ------------------------------------------------------------------
        # Collect zone-level fires for regime calculation
        zone_fire_sizes: dict[str, list[float]] = defaultdict(list)
        zone_fire_causes: dict[str, list[str]] = defaultdict(list)
        zone_burned_ha: dict[str, float] = defaultdict(float)
        zone_area_ha: dict[str, float] = defaultdict(float)  # approx from cell count × 1km²

        for cell_id, tiers in cell_tiers.items():
            c = cell_map[cell_id]
            bz = c["bec_zone"]
            # Approx 1 km² per cell = 100 ha
            zone_area_ha[bz] += 100.0
            all_fires = tiers["fires_all"]
            zone_burned_ha[bz] += all_fires.get("largest_ha", 0.0) or 0.0

        # Gather actual fire sizes/causes per zone from fire_history
        for fire in fires_all:
            # We don't have a direct zone→fire mapping; use a broad estimate.
            # A proper implementation would spatially join fires to zones.
            # For now, attribute fires to cells in the zone via nearest-cell lookup,
            # which would require a full spatial join. Instead we aggregate by
            # whether any cell in the zone matched the fire.
            pass

        # Re-gather by scanning cell_tiers for fires matched to any cell in each zone
        # This gives a per-zone view of fires actually matched to grid cells
        zone_sizes_agg: dict[str, list[float]] = defaultdict(list)
        zone_causes_agg: dict[str, list[str]] = defaultdict(list)
        zone_total_burned: dict[str, float] = defaultdict(float)

        # We need the raw matched fires per cell — re-derive from zone pass
        # Since we discarded matched fires above, we rebuild a fast lookup:
        # zone → set of (fire lat,lon) → avoid double-counting per zone
        seen_fires_per_zone: dict[str, set] = defaultdict(set)

        for bec_zone, zone_cells in zones.items():
            lats = [c["lat"] for c in zone_cells]
            lons = [c["lon"] for c in zone_cells]
            lat_min, lat_max = min(lats) - 0.1, max(lats) + 0.1
            lon_min, lon_max = min(lons) - 0.1, max(lons) + 0.1
            zone_fires_raw = [
                f
                for f in fires_all
                if lat_min <= f["lat"] <= lat_max and lon_min <= f["lon"] <= lon_max
            ]
            for f in zone_fires_raw:
                key = (f["lat"], f["lon"], f["year"])
                if key not in seen_fires_per_zone[bec_zone]:
                    seen_fires_per_zone[bec_zone].add(key)
                    zone_sizes_agg[bec_zone].append(f["size_ha"])
                    zone_causes_agg[bec_zone].append(f["cause"])
                    zone_total_burned[bec_zone] += f["size_ha"]

        # Earliest year in fire record for years_of_record
        if fires_all:
            earliest_year = min(f["year"] for f in fires_all)
            years_of_record = max(current_year - earliest_year, 1)
        else:
            years_of_record = 1

        zone_regimes: dict[str, dict] = {}
        for bec_zone in zones:
            regime = compute_fire_regime(
                zone_area_ha=zone_area_ha.get(bec_zone, 100.0),
                total_burned_ha=zone_total_burned.get(bec_zone, 0.0),
                years_of_record=years_of_record,
                fire_sizes=zone_sizes_agg.get(bec_zone, []),
                causes=zone_causes_agg.get(bec_zone, []),
            )
            zone_regimes[bec_zone] = regime

        # ------------------------------------------------------------------
        # Step 8: Percentiles
        # ------------------------------------------------------------------
        all_scores = np.array([cell_susceptibility[cid][0] for cid in cell_susceptibility])
        all_exposure = np.array([
            cell_tiers[cid]["fires_10yr"]["count"] for cid in cell_susceptibility
        ], dtype=float)

        def _percentile_rank(arr: np.ndarray, value: float) -> int:
            if arr.max() == arr.min():
                return 50
            return int(np.round(np.searchsorted(np.sort(arr), value) / len(arr) * 100))

        scores_sorted = np.sort(all_scores)
        exposure_sorted = np.sort(all_exposure)

        cell_susceptibility_pct: dict[str, int] = {}
        cell_exposure_pct: dict[str, int] = {}
        for cell_id, (score, _) in cell_susceptibility.items():
            pct_s = int(np.round(np.searchsorted(scores_sorted, score) / len(scores_sorted) * 100))
            pct_e = int(np.round(
                np.searchsorted(exposure_sorted, cell_tiers[cell_id]["fires_10yr"]["count"])
                / len(exposure_sorted) * 100
            ))
            cell_susceptibility_pct[cell_id] = min(pct_s, 100)
            cell_exposure_pct[cell_id] = min(pct_e, 100)

        # ------------------------------------------------------------------
        # Step 9: Write to DB (delete + batch insert)
        # ------------------------------------------------------------------
        logger.info("Writing fire statistics to DB …")
        db.query(FireStatisticsDB).delete()
        db.commit()

        BATCH = 500
        records = list(cell_susceptibility.keys())
        now = datetime.utcnow()

        for batch_start in range(0, len(records), BATCH):
            batch_ids = records[batch_start : batch_start + BATCH]
            rows = []
            for cell_id in batch_ids:
                tiers = cell_tiers[cell_id]
                score, basis = cell_susceptibility[cell_id]
                c = cell_map[cell_id]
                regime = zone_regimes.get(c["bec_zone"], {})
                pct_s = cell_susceptibility_pct[cell_id]
                pct_e = cell_exposure_pct[cell_id]

                t10 = tiers["fires_10yr"]
                t30 = tiers["fires_30yr"]
                ta = tiers["fires_all"]

                rows.append(
                    FireStatisticsDB(
                        cell_id=cell_id,
                        fires_10yr_count=t10["count"],
                        fires_10yr_nearest_km=t10["nearest_km"],
                        fires_10yr_largest_ha=t10["largest_ha"],
                        fires_10yr_causes=json.dumps(t10["causes"]),
                        fires_30yr_count=t30["count"],
                        fires_30yr_nearest_km=t30["nearest_km"],
                        fires_30yr_largest_ha=t30["largest_ha"],
                        fires_30yr_causes=json.dumps(t30["causes"]),
                        fires_all_count=ta["count"],
                        fires_all_nearest_km=ta["nearest_km"],
                        fires_all_largest_ha=ta["largest_ha"],
                        fires_all_causes=json.dumps(ta["causes"]),
                        fires_all_record_start=ta.get("record_start"),
                        susceptibility_score=score,
                        susceptibility_percentile=pct_s,
                        susceptibility_label=label_from_percentile(pct_s),
                        susceptibility_basis=basis,
                        exposure_percentile=pct_e,
                        mean_return_years=regime.get("mean_return_years"),
                        typical_severity=regime.get("typical_severity"),
                        dominant_cause=regime.get("dominant_cause"),
                        computed_at=now,
                    )
                )
            db.bulk_save_objects(rows)
            db.commit()
            logger.info("Inserted batch %d–%d", batch_start, batch_start + len(batch_ids))

        logger.info("DB write complete: %d rows", len(records))

        # ------------------------------------------------------------------
        # Step 10: Cache to Redis (no TTL)
        # ------------------------------------------------------------------
        r = get_redis()
        if r is not None:
            REDIS_BATCH = 10_000
            all_cids = list(cell_susceptibility.keys())
            for batch_start in range(0, len(all_cids), REDIS_BATCH):
                batch_ids = all_cids[batch_start : batch_start + REDIS_BATCH]
                pipe = r.pipeline()
                for cell_id in batch_ids:
                    tiers = cell_tiers[cell_id]
                    score, basis = cell_susceptibility[cell_id]
                    c = cell_map[cell_id]
                    regime = zone_regimes.get(c["bec_zone"], {})
                    pct_s = cell_susceptibility_pct[cell_id]
                    pct_e = cell_exposure_pct[cell_id]
                    payload = {
                        "cell_id": cell_id,
                        "fires_10yr": tiers["fires_10yr"],
                        "fires_30yr": tiers["fires_30yr"],
                        "fires_all": tiers["fires_all"],
                        "susceptibility_score": score,
                        "susceptibility_percentile": pct_s,
                        "susceptibility_label": label_from_percentile(pct_s),
                        "susceptibility_basis": basis,
                        "exposure_percentile": pct_e,
                        "mean_return_years": regime.get("mean_return_years"),
                        "typical_severity": regime.get("typical_severity"),
                        "dominant_cause": regime.get("dominant_cause"),
                        "computed_at": now.isoformat(),
                    }
                    pipe.set(f"fire_stats:{cell_id}", json.dumps(payload))
                pipe.execute()
            logger.info("Cached %d fire stats records to Redis", len(all_cids))
        else:
            logger.warning("Redis unavailable — skipping fire stats cache")

    finally:
        db.close()
