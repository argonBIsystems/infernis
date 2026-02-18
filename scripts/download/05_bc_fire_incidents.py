"""Download BC fire incident locations from BC Data Catalogue.

Data source: BC Wildfire Service
URL: https://catalogue.data.gov.bc.ca/dataset/fire-incident-locations-historical
Output: data/raw/bc_fire_incidents/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, download_file, ensure_dir, get_logger, log_summary

LOGGER = get_logger("bc_fire_incidents")
OUT_DIR = RAW_DIR / "bc_fire_incidents"

WFS_URL = (
    "https://openmaps.gov.bc.ca/geo/pub/WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_INCIDENTS_SP/ows"
    "?service=WFS&version=2.0.0&request=GetFeature&typeName=pub:WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_INCIDENTS_SP"
    "&outputFormat=json&count=100000"
)


def download_bc_incidents(dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    dest = OUT_DIR / "bc_fire_incidents.geojson"

    if dry_run:
        LOGGER.info("[DRY RUN] Would download BC fire incidents via WFS")
        log_summary("BC Fire Incidents", 0, 1, logger=LOGGER)
        return

    skip = not force
    was_downloaded = download_file(WFS_URL, dest, skip_existing=skip, logger=LOGGER, timeout=300)
    downloaded = 1 if was_downloaded else 0
    skipped = 0 if was_downloaded else 1
    log_summary("BC Fire Incidents", downloaded, skipped, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download BC fire incident locations")
    args = parser.parse_args()
    download_bc_incidents(args.dry_run, args.force)


if __name__ == "__main__":
    main()
