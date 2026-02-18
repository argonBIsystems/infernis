"""Download Canada Land Cover 2020.

Data source: Open Canada
URL: https://open.canada.ca/data/en/dataset/ee1580ab-a23d-4f86-a09b-79763677eb47
Output: data/raw/land_cover/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, download_file, ensure_dir, get_logger, log_summary

LOGGER = get_logger("land_cover")
OUT_DIR = RAW_DIR / "land_cover"

LAND_COVER_URL = (
    "https://datacube-prod-data-public.s3.ca-central-1.amazonaws.com"
    "/store/land/landcover/landcover-2020-classification.tif"
)


def download_land_cover(dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    dest = OUT_DIR / "landcover-2020-classification.tif"

    if dry_run:
        LOGGER.info("[DRY RUN] Would download Canada Land Cover 2020 (~2 GB)")
        log_summary("Land Cover", 0, 1, logger=LOGGER)
        return

    skip = not force
    was_downloaded = download_file(LAND_COVER_URL, dest, skip_existing=skip, logger=LOGGER, timeout=1200)

    downloaded = 1 if was_downloaded else 0
    skipped = 0 if was_downloaded else 1
    log_summary("Land Cover", downloaded, skipped, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download Canada Land Cover 2020")
    args = parser.parse_args()
    download_land_cover(args.dry_run, args.force)


if __name__ == "__main__":
    main()
