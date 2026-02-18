"""Download Copernicus historical FWI (ERA5-based) for BC.

Data source: Copernicus Emergency Management Service (CEMS)
Dataset: cems-fire-historical (dataset ID may change — verified against CDS catalogue)

NOTE: The cems-fire-historical-v1 dataset was removed from CDS circa 2025.
This script attempts the download but will fail gracefully. FWI components
are also available from CWFIS (07_cwfis.py) and computed from ERA5 weather
data via cffdrs_py in the pipeline itself.

Output: data/raw/copernicus_fwi/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BC_BBOX, RAW_DIR, base_argparser, ensure_dir, get_cds_client, get_logger, log_summary

LOGGER = get_logger("copernicus_fwi")
OUT_DIR = RAW_DIR / "copernicus_fwi"

AREA = [BC_BBOX["north"], BC_BBOX["west"], BC_BBOX["south"], BC_BBOX["east"]]

FWI_VARIABLES = [
    "fire_weather_index",
    "fine_fuel_moisture_code",
    "duff_moisture_code",
    "drought_code",
    "initial_spread_index",
    "buildup_index",
]


def download_copernicus_fwi(start_year: int, end_year: int, dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    client = None if dry_run else get_cds_client()
    downloaded, skipped = 0, 0

    for year in range(start_year, end_year + 1):
        fname = f"copernicus_fwi_bc_{year}.nc"
        dest = OUT_DIR / fname

        if not force and dest.exists() and dest.stat().st_size > 0:
            LOGGER.info("Skipping (exists): %s", fname)
            skipped += 1
            continue

        if dry_run:
            LOGGER.info("[DRY RUN] Would download: %s", fname)
            skipped += 1
            continue

        LOGGER.info("Requesting: %s", fname)
        request = {
            "product_type": "reanalysis",
            "variable": FWI_VARIABLES,
            "year": str(year),
            "month": [str(m).zfill(2) for m in range(4, 11)],
            "day": [str(d).zfill(2) for d in range(1, 32)],
            "area": AREA,
            "data_format": "netcdf",
        }
        try:
            client.retrieve("cems-fire-historical-v1", request, str(dest))
            LOGGER.info("Downloaded: %s", fname)
            downloaded += 1
        except Exception as exc:
            LOGGER.error("Failed %s: %s", fname, exc)
            LOGGER.info("Note: cems-fire-historical-v1 may no longer be available on CDS. "
                        "FWI is computed from ERA5 via cffdrs_py and also available from CWFIS (07_cwfis.py).")
            break  # Don't retry all years if dataset is unavailable

    log_summary("Copernicus FWI", downloaded, skipped, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download Copernicus historical FWI for BC")
    args = parser.parse_args()
    download_copernicus_fwi(args.start_year, args.end_year, args.dry_run, args.force)


if __name__ == "__main__":
    main()
