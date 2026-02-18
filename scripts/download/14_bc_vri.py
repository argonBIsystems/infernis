"""Download BC Vegetation Resources Inventory (VRI) data.

Data source: BC Data Catalogue
URL: https://catalogue.data.gov.bc.ca/dataset/vri-forest-vegetation-composite-polygons-and-rank-1-layer
Output: data/raw/bc_vri/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, download_file, ensure_dir, get_logger, log_summary

LOGGER = get_logger("bc_vri")
OUT_DIR = RAW_DIR / "bc_vri"

VRI_WFS = (
    "https://openmaps.gov.bc.ca/geo/pub/WHSE_FOREST_VEGETATION.VEG_COMP_LYR_R1_POLY/ows"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=pub:WHSE_FOREST_VEGETATION.VEG_COMP_LYR_R1_POLY"
    "&outputFormat=json&count=50000"
    "&propertyName=SPECIES_CD_1,SPECIES_PCT_1,PROJ_AGE_1,PROJ_HEIGHT_1,"
    "CROWN_CLOSURE,STAND_PERCENTAGE_DEAD,BCLCS_LEVEL_1,BEC_ZONE_CODE,GEOMETRY"
)


def download_bc_vri(dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    dest = OUT_DIR / "bc_vri_subset.geojson"

    if dry_run:
        LOGGER.info("[DRY RUN] Would download BC VRI data (subset of key attributes)")
        LOGGER.info("Note: Full VRI is very large (millions of polygons). This downloads a 50K polygon sample.")
        log_summary("BC VRI", 0, 1, logger=LOGGER)
        return

    skip = not force
    was_downloaded = download_file(VRI_WFS, dest, skip_existing=skip, logger=LOGGER, timeout=600)
    downloaded = 1 if was_downloaded else 0
    skipped = 0 if was_downloaded else 1
    log_summary("BC VRI", downloaded, skipped, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download BC Vegetation Resources Inventory")
    args = parser.parse_args()
    download_bc_vri(args.dry_run, args.force)


if __name__ == "__main__":
    main()
