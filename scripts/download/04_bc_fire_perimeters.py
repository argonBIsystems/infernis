"""Download BC historical fire perimeters from BC Data Catalogue.

Data source: BC Wildfire Service
URL: https://catalogue.data.gov.bc.ca/dataset/fire-perimeters-historical
Output: data/raw/bc_fire_perimeters/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, download_file, ensure_dir, get_logger, log_summary

LOGGER = get_logger("bc_fire_perimeters")
OUT_DIR = RAW_DIR / "bc_fire_perimeters"

WFS_URL = (
    "https://openmaps.gov.bc.ca/geo/pub/WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_FIRE_POLYS_SP/ows"
    "?service=WFS&version=2.0.0&request=GetFeature&typeName=pub:WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_FIRE_POLYS_SP"
    "&outputFormat=json&count=100000"
)


def download_bc_perimeters(dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    dest = OUT_DIR / "bc_fire_perimeters.geojson"

    if dry_run:
        LOGGER.info("[DRY RUN] Would download BC fire perimeters via WFS")
        log_summary("BC Fire Perimeters", 0, 1, logger=LOGGER)
        return

    skip = not force
    was_downloaded = download_file(WFS_URL, dest, skip_existing=skip, logger=LOGGER, timeout=300)
    downloaded = 1 if was_downloaded else 0
    skipped = 0 if was_downloaded else 1
    log_summary("BC Fire Perimeters", downloaded, skipped, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download BC historical fire perimeters")
    args = parser.parse_args()
    download_bc_perimeters(args.dry_run, args.force)


if __name__ == "__main__":
    main()
