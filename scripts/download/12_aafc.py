"""Download Agriculture and Agri-Food Canada soil moisture products.

Data source: AAFC
Note: AAFC coverage is limited for BC (agricultural regions only). RISMA stations
are only in SK, MB, ON - not BC.
Output: data/raw/aafc/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, ensure_dir, get_logger, log_summary

LOGGER = get_logger("aafc")
OUT_DIR = RAW_DIR / "aafc"


def download_aafc(dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    readme = OUT_DIR / "README.md"

    if dry_run:
        LOGGER.info("[DRY RUN] Would create AAFC data directory with instructions")
        log_summary("AAFC", 0, 1, logger=LOGGER)
        return

    if not force and readme.exists():
        LOGGER.info("Skipping (exists): AAFC README")
        log_summary("AAFC", 0, 1, logger=LOGGER)
        return

    readme.write_text(
        "# AAFC Soil Moisture Data\n\n"
        "AAFC satellite soil moisture products for agricultural regions of Canada.\n\n"
        "**Note:** RISMA in-situ stations are NOT in BC (only SK, MB, ON).\n"
        "Satellite product focuses on agricultural land with limited BC forest coverage.\n\n"
        "Manual download:\n"
        "- https://agriculture.canada.ca/en/agricultural-production/weather/satellite-soil-moisture\n",
        encoding="utf-8",
    )
    LOGGER.info("Created AAFC data directory with instructions")
    log_summary("AAFC", 1, 0, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download AAFC soil moisture data")
    args = parser.parse_args()
    download_aafc(args.dry_run, args.force)


if __name__ == "__main__":
    main()
