"""Download Pacific Climate Impacts Consortium data for BC.

Data source: Pacific Climate Impacts Consortium
URL: https://www.pacificclimate.org/data/bc-station-data
Output: data/raw/pcic/

NOTE: The PCIC met-data-portal API is intermittently unavailable (503).
BC climate station data is also available via ECCC (08_eccc.py).
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, ensure_dir, get_logger, log_summary

LOGGER = get_logger("pcic")
OUT_DIR = RAW_DIR / "pcic"

PCIC_CATALOG_URL = "https://services.pacificclimate.org/met-data-portal-pcds/api/"


def download_pcic(dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    dest = OUT_DIR / "pcic_station_catalog.json"

    if dry_run:
        LOGGER.info("[DRY RUN] Would download PCIC station catalog")
        log_summary("PCIC", 0, 1, logger=LOGGER)
        return

    if not force and dest.exists() and dest.stat().st_size > 0:
        LOGGER.info("Skipping (exists): %s", dest.name)
        log_summary("PCIC", 0, 1, logger=LOGGER)
        return

    try:
        resp = requests.get(PCIC_CATALOG_URL, timeout=60)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")
        LOGGER.info("Downloaded: %s", dest.name)
        log_summary("PCIC", 1, 0, logger=LOGGER)
    except Exception as exc:
        LOGGER.error("Failed to download PCIC catalog: %s", exc)
        log_summary("PCIC", 0, 0, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download PCIC climate data for BC")
    args = parser.parse_args()
    download_pcic(args.dry_run, args.force)


if __name__ == "__main__":
    main()
