"""Download Canadian National Fire Database (CNFDB) point-of-origin records.

Data source: Canadian Forest Service (CFS)
URL: https://cwfis.cfs.nrcan.gc.ca/datamart/download/nfdbpnt
Output: data/raw/cnfdb/nfdbpnt.zip (+ extracted shapefiles)
"""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, download_file, ensure_dir, get_logger, log_summary

LOGGER = get_logger("cnfdb")
OUT_DIR = RAW_DIR / "cnfdb"
CNFDB_URL = "https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version/NFDB_point.zip"


def download_cnfdb(dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    dest = OUT_DIR / "NFDB_point.zip"

    if dry_run:
        LOGGER.info("[DRY RUN] Would download: %s", CNFDB_URL)
        log_summary("CNFDB", 0, 1, logger=LOGGER)
        return

    skip = not force
    was_downloaded = download_file(CNFDB_URL, dest, skip_existing=skip, logger=LOGGER)

    if was_downloaded or force:
        LOGGER.info("Extracting %s ...", dest.name)
        with zipfile.ZipFile(dest, "r") as zf:
            zf.extractall(OUT_DIR)

    downloaded = 1 if was_downloaded else 0
    skipped = 0 if was_downloaded else 1
    log_summary("CNFDB", downloaded, skipped, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download Canadian National Fire Database")
    args = parser.parse_args()
    download_cnfdb(args.dry_run, args.force)


if __name__ == "__main__":
    main()
