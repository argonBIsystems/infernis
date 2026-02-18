"""Download NASA FIRMS active fire detections for Canada.

Data source: NASA FIRMS (VIIRS and MODIS)
API: https://firms.modaps.eosdis.nasa.gov/api/
Output: data/raw/firms/firms_viirs_canada_{year}.csv
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, ensure_dir, get_firms_map_key, get_logger, log_summary

LOGGER = get_logger("firms")
OUT_DIR = RAW_DIR / "firms"


def download_firms(start_year: int, end_year: int, dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    map_key = None if dry_run else get_firms_map_key()
    downloaded, skipped = 0, 0

    for year in range(start_year, end_year + 1):
        dest = OUT_DIR / f"firms_viirs_canada_{year}.csv"

        if not force and dest.exists() and dest.stat().st_size > 0:
            LOGGER.info("Skipping (exists): %s", dest.name)
            skipped += 1
            continue

        if dry_run:
            LOGGER.info("[DRY RUN] Would download FIRMS data for %d", year)
            skipped += 1
            continue

        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/VIIRS_SNPP_SP/-140,48,-114,60/{year}-01-01/{year}-12-31"
        LOGGER.info("Requesting FIRMS data for %d ...", year)
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            dest.write_text(resp.text, encoding="utf-8")
            LOGGER.info("Downloaded: %s (%d bytes)", dest.name, len(resp.content))
            downloaded += 1
        except Exception as exc:
            LOGGER.error("Failed to download FIRMS %d: %s", year, exc)

    log_summary("FIRMS", downloaded, skipped, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download NASA FIRMS active fire data for BC")
    args = parser.parse_args()
    download_firms(args.start_year, args.end_year, args.dry_run, args.force)


if __name__ == "__main__":
    main()
