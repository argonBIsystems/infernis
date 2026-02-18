"""Download MSC Datamart data (NWP forecasts, lightning, BC fire weather stations).

Data source: Meteorological Service of Canada
URL: https://dd.weather.gc.ca/
Output: data/raw/msc_datamart/

Supports downloading:
- Lightning index HTML
- HRDPS 2.5km GRIB2 forecasts (48h, BC-relevant variables)
- GDPS 15km GRIB2 forecasts (10-day, BC-relevant variables)
"""

import sys
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, ensure_dir, get_logger, log_summary

LOGGER = get_logger("msc_datamart")
OUT_DIR = RAW_DIR / "msc_datamart"

LIGHTNING_URL = "https://dd.weather.gc.ca/today/lightning/"

# HRDPS: High Resolution Deterministic Prediction System (2.5km, 48h)
HRDPS_BASE = "https://dd.weather.gc.ca/model_hrdps/continental/2.5km"
HRDPS_VARIABLES = ["TMP_TGL_2", "RH_TGL_2", "WIND_TGL_10", "WDIR_TGL_10", "APCP_SFC_0"]
HRDPS_HOURS = list(range(1, 49))

# GDPS: Global Deterministic Prediction System (15km, 10 days)
GDPS_BASE = "https://dd.weather.gc.ca/model_gdps/15km"
GDPS_VARIABLES = ["TMP_TGL_2", "RH_TGL_2", "WIND_TGL_10", "WDIR_TGL_10", "APCP_SFC_0"]
GDPS_HOURS = list(range(3, 241, 3))


def download_lightning(dry_run: bool, force: bool) -> tuple[int, int]:
    out = ensure_dir(OUT_DIR / "lightning")
    dest = out / "lightning_index.html"

    if dry_run:
        LOGGER.info("[DRY RUN] Would download lightning data index")
        return 0, 1

    if not force and dest.exists() and dest.stat().st_size > 0:
        LOGGER.info("Skipping (exists): %s", dest.name)
        return 0, 1

    try:
        resp = requests.get(LIGHTNING_URL, timeout=60)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")
        LOGGER.info("Downloaded lightning index: %s", dest.name)
        return 1, 0
    except Exception as exc:
        LOGGER.error("Failed to download lightning index: %s", exc)
        return 0, 1


def download_hrdps(
    run_hour: int, dry_run: bool, force: bool, target_date: date | None = None
) -> tuple[int, int]:
    """Download HRDPS GRIB2 files for a model run.

    Downloads BC-relevant weather variables for all 48 forecast hours.
    """
    target_date = target_date or date.today()
    date_str = target_date.strftime("%Y%m%d")
    out = ensure_dir(OUT_DIR / "hrdps" / date_str / f"{run_hour:02d}Z")

    if dry_run:
        total = len(HRDPS_VARIABLES) * len(HRDPS_HOURS)
        LOGGER.info("[DRY RUN] Would download %d HRDPS files for %s %02dZ", total, date_str, run_hour)
        return 0, total

    downloaded = 0
    skipped = 0
    for fh in HRDPS_HOURS:
        for var in HRDPS_VARIABLES:
            filename = f"{date_str}T{run_hour:02d}Z_MSC_HRDPS_{var}_RLatLon0.0225_PT{fh:03d}H.grib2"
            dest = out / filename

            if not force and dest.exists() and dest.stat().st_size > 0:
                skipped += 1
                continue

            url = f"{HRDPS_BASE}/{run_hour:02d}/{fh:03d}/{filename}"
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                downloaded += 1
            except Exception as exc:
                LOGGER.warning("Failed: %s (%s)", filename, exc)
                skipped += 1

    LOGGER.info("HRDPS %s %02dZ: %d downloaded, %d skipped", date_str, run_hour, downloaded, skipped)
    return downloaded, skipped


def download_gdps(
    run_hour: int, dry_run: bool, force: bool, target_date: date | None = None
) -> tuple[int, int]:
    """Download GDPS GRIB2 files for a model run.

    Downloads BC-relevant weather variables for 10-day forecast (3-hourly).
    """
    target_date = target_date or date.today()
    date_str = target_date.strftime("%Y%m%d")
    out = ensure_dir(OUT_DIR / "gdps" / date_str / f"{run_hour:02d}Z")

    if dry_run:
        total = len(GDPS_VARIABLES) * len(GDPS_HOURS)
        LOGGER.info("[DRY RUN] Would download %d GDPS files for %s %02dZ", total, date_str, run_hour)
        return 0, total

    downloaded = 0
    skipped = 0
    for fh in GDPS_HOURS:
        for var in GDPS_VARIABLES:
            filename = f"{date_str}T{run_hour:02d}Z_MSC_GDPS_{var}_LatLon0.15_PT{fh:03d}H.grib2"
            dest = out / filename

            if not force and dest.exists() and dest.stat().st_size > 0:
                skipped += 1
                continue

            url = f"{GDPS_BASE}/{run_hour:02d}/{fh:03d}/{filename}"
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                downloaded += 1
            except Exception as exc:
                LOGGER.warning("Failed: %s (%s)", filename, exc)
                skipped += 1

    LOGGER.info("GDPS %s %02dZ: %d downloaded, %d skipped", date_str, run_hour, downloaded, skipped)
    return downloaded, skipped


def main() -> None:
    parser = base_argparser("Download MSC Datamart data (lightning, HRDPS, GDPS)")
    parser.add_argument(
        "--hrdps", action="store_true",
        help="Download HRDPS 2.5km forecast data (48h)",
    )
    parser.add_argument(
        "--gdps", action="store_true",
        help="Download GDPS 15km forecast data (10-day)",
    )
    parser.add_argument(
        "--run-hour", type=int, default=12,
        help="Model run hour UTC (default: 12)",
    )
    args = parser.parse_args()
    ensure_dir(OUT_DIR)

    total_d, total_s = 0, 0

    # Lightning (always)
    d, s = download_lightning(args.dry_run, args.force)
    total_d += d
    total_s += s

    # HRDPS
    if args.hrdps:
        d, s = download_hrdps(args.run_hour, args.dry_run, args.force)
        total_d += d
        total_s += s

    # GDPS
    if args.gdps:
        d, s = download_gdps(args.run_hour, args.dry_run, args.force)
        total_d += d
        total_s += s

    log_summary("MSC Datamart", total_d, total_s, logger=LOGGER)


if __name__ == "__main__":
    main()
