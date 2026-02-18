"""Download Digital Elevation Models for BC (CDEM tiles).

Data sources:
  - CDEM: https://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdem_mnec/
  - SRTM: via GEE (see 02_gee_satellite.py)
Output: data/raw/dem/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, download_file, ensure_dir, get_logger, log_summary

LOGGER = get_logger("dem")
OUT_DIR = RAW_DIR / "dem"

CDEM_TILE_BASE = "https://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdem_mnec/"
BC_NTS_SHEETS = [
    "082", "083", "092", "093", "094", "102", "103", "104", "114",
]


def download_cdem_tiles(dry_run: bool, force: bool) -> tuple[int, int]:
    out = ensure_dir(OUT_DIR / "cdem")
    downloaded, skipped = 0, 0

    for sheet in BC_NTS_SHEETS:
        fname = f"cdem_dem_{sheet}.tif"
        dest = out / fname
        url = f"{CDEM_TILE_BASE}{sheet}/{fname}"

        if not force and dest.exists() and dest.stat().st_size > 0:
            LOGGER.info("Skipping (exists): %s", fname)
            skipped += 1
            continue
        if dry_run:
            LOGGER.info("[DRY RUN] Would download: %s", fname)
            skipped += 1
            continue

        try:
            download_file(url, dest, skip_existing=False, logger=LOGGER, timeout=300)
            downloaded += 1
        except Exception as exc:
            LOGGER.error("Failed %s: %s", fname, exc)
            skipped += 1

    return downloaded, skipped


def main() -> None:
    parser = base_argparser("Download DEMs for BC (CDEM tiles)")
    args = parser.parse_args()
    ensure_dir(OUT_DIR)

    dl, sk = download_cdem_tiles(args.dry_run, args.force)
    LOGGER.info("Note: SRTM DEM for BC is downloaded via GEE in 02_gee_satellite.py")
    log_summary("DEM", dl, sk, logger=LOGGER)


if __name__ == "__main__":
    main()
