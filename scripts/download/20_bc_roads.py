"""Download BC Digital Road Atlas.

Data source: BC Data Catalogue
URL: https://catalogue.data.gov.bc.ca/dataset/digital-road-atlas-dra-demographic-partially-attributed-roads
Output: data/raw/bc_roads/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, download_file, ensure_dir, get_logger, log_summary

LOGGER = get_logger("bc_roads")
OUT_DIR = RAW_DIR / "bc_roads"

BC_ROADS_WFS = (
    "https://openmaps.gov.bc.ca/geo/pub/WHSE_BASEMAPPING.DRA_DGTL_ROAD_ATLAS_MPAR_SP/ows"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=pub:WHSE_BASEMAPPING.DRA_DGTL_ROAD_ATLAS_MPAR_SP"
    "&outputFormat=json&count=100000"
)


def download_bc_roads(dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    dest = OUT_DIR / "bc_roads.geojson"

    if dry_run:
        LOGGER.info("[DRY RUN] Would download BC Digital Road Atlas via WFS")
        log_summary("BC Roads", 0, 1, logger=LOGGER)
        return

    skip = not force
    was_downloaded = download_file(BC_ROADS_WFS, dest, skip_existing=skip, logger=LOGGER, timeout=600)
    downloaded = 1 if was_downloaded else 0
    skipped = 0 if was_downloaded else 1
    log_summary("BC Roads", downloaded, skipped, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download BC Digital Road Atlas")
    args = parser.parse_args()
    download_bc_roads(args.dry_run, args.force)


if __name__ == "__main__":
    main()
