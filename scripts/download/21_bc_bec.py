"""Download BC Biogeoclimatic Ecosystem Classification (BEC) zone map.

Data source: BC Data Catalogue / Open Canada
URL: https://catalogue.data.gov.bc.ca/dataset/bec-map
Output: data/raw/bc_bec/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, download_file, ensure_dir, get_logger, log_summary

LOGGER = get_logger("bc_bec")
OUT_DIR = RAW_DIR / "bc_bec"

BC_BEC_WFS = (
    "https://openmaps.gov.bc.ca/geo/pub/WHSE_FOREST_VEGETATION.BEC_BIOGEOCLIMATIC_POLY/ows"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=pub:WHSE_FOREST_VEGETATION.BEC_BIOGEOCLIMATIC_POLY"
    "&outputFormat=json&count=100000"
)


def download_bc_bec(dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    dest = OUT_DIR / "bc_bec_zones.geojson"

    if dry_run:
        LOGGER.info("[DRY RUN] Would download BC BEC zone map via WFS")
        log_summary("BC BEC", 0, 1, logger=LOGGER)
        return

    skip = not force
    was_downloaded = download_file(BC_BEC_WFS, dest, skip_existing=skip, logger=LOGGER, timeout=600)
    downloaded = 1 if was_downloaded else 0
    skipped = 0 if was_downloaded else 1
    log_summary("BC BEC", downloaded, skipped, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download BC BEC zone map")
    args = parser.parse_args()
    download_bc_bec(args.dry_run, args.force)


if __name__ == "__main__":
    main()
