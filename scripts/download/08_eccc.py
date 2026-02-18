"""Download ECCC weather station data for BC via bulk CSV download.

Data source: Environment and Climate Change Canada
URL: https://climate.weather.gc.ca/
Output: data/raw/eccc/
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, ensure_dir, get_logger, log_summary

LOGGER = get_logger("eccc")
OUT_DIR = RAW_DIR / "eccc"

BC_STATIONS = {
    "KAMLOOPS A": 1275,
    "PRINCE GEORGE A": 1096,
    "WILLIAMS LAKE A": 1341,
    "CRANBROOK A": 1152,
    "KELOWNA A": 1235,
    "PENTICTON A": 1282,
    "CASTLEGAR A": 1143,
    "QUESNEL A": 1293,
    "SMITHERS A": 1312,
    "TERRACE A": 1328,
}

BULK_URL_TEMPLATE = (
    "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    "?format=csv&stationID={station_id}&Year={year}&Month=1&timeframe=2"
)


def download_eccc(start_year: int, end_year: int, dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    downloaded, skipped = 0, 0

    for station_name, station_id in BC_STATIONS.items():
        station_dir = ensure_dir(OUT_DIR / f"station_{station_id}")
        for year in range(start_year, end_year + 1):
            dest = station_dir / f"eccc_{station_id}_{year}_daily.csv"

            if not force and dest.exists() and dest.stat().st_size > 0:
                LOGGER.info("Skipping (exists): %s", dest.name)
                skipped += 1
                continue

            if dry_run:
                LOGGER.info("[DRY RUN] Would download %s %d", station_name, year)
                skipped += 1
                continue

            url = BULK_URL_TEMPLATE.format(station_id=station_id, year=year)
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                dest.write_text(resp.text, encoding="utf-8")
                LOGGER.info("Downloaded: %s", dest.name)
                downloaded += 1
            except Exception as exc:
                LOGGER.error("Failed %s %d: %s", station_name, year, exc)

    log_summary("ECCC", downloaded, skipped, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download ECCC weather station data for BC")
    args = parser.parse_args()
    download_eccc(args.start_year, args.end_year, args.dry_run, args.force)


if __name__ == "__main__":
    main()
