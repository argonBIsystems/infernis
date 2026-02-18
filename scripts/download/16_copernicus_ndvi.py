"""Download Copernicus Global 300m NDVI product.

Data source: Copernicus Global Land Service
URL: https://land.copernicus.eu/en/products/vegetation/normalised-difference-vegetation-index-v2-0-300m
Output: data/raw/copernicus_ndvi/

NOTE: This script uses the CDS API with satellite-lai-fapar dataset.
NDVI is also computed from Sentinel-2 via GEE (02_gee_satellite.py),
so this source is optional / supplementary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BC_BBOX, RAW_DIR, base_argparser, ensure_dir, get_cds_client, get_logger, log_summary

LOGGER = get_logger("copernicus_ndvi")
OUT_DIR = RAW_DIR / "copernicus_ndvi"


def download_copernicus_ndvi(start_year: int, end_year: int, dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    client = None if dry_run else get_cds_client()
    downloaded, skipped = 0, 0

    for year in range(max(start_year, 2020), end_year + 1):
        for dekad in ["01", "11", "21"]:
            for month in range(5, 10):
                fname = f"copernicus_ndvi_bc_{year}_{month:02d}_{dekad}.nc"
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
                    "variable": "ndvi",
                    "year": str(year),
                    "month": str(month).zfill(2),
                    "day": dekad,
                    "area": [BC_BBOX["north"], BC_BBOX["west"], BC_BBOX["south"], BC_BBOX["east"]],
                    "data_format": "netcdf",
                }
                try:
                    client.retrieve("satellite-lai-fapar", request, str(dest))
                    LOGGER.info("Downloaded: %s", fname)
                    downloaded += 1
                except Exception as exc:
                    LOGGER.error("Failed %s: %s", fname, exc)
                    if "not found" in str(exc).lower() or "no data" in str(exc).lower():
                        LOGGER.info("NDVI is also available from GEE (02_gee_satellite.py)")
                        break

    log_summary("Copernicus NDVI", downloaded, skipped, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download Copernicus Global 300m NDVI for BC")
    args = parser.parse_args()
    download_copernicus_ndvi(args.start_year, args.end_year, args.dry_run, args.force)


if __name__ == "__main__":
    main()
