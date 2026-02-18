"""Download CFFDRS fuel type maps (national via CWFIS WCS + BC provincial WFS).

Data sources:
  - National: CWFIS GeoServer WCS (100m CFFDRS FBP fuel types)
  - BC: BC Data Catalogue WFS
Output: data/raw/fuel_types/
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, download_file, ensure_dir, get_logger, log_summary

LOGGER = get_logger("fuel_types")
OUT_DIR = RAW_DIR / "fuel_types"

# CWFIS GeoServer WCS — CFFDRS FBP fuel types (national, ~30MB GeoTIFF)
# NRCAN FTP paths are dead; CWFIS hosts the same data via WCS.
NATIONAL_WCS = (
    "https://cwfis.cfs.nrcan.gc.ca/geoserver/ows"
    "?service=WCS&version=2.0.1&request=GetCoverage"
    "&coverageId=public__FBP_FuelLayer_wBurnScars"
    "&format=image/geotiff"
)

# BC provincial fuel types — WFS with limited count (paginate if needed)
BC_FUEL_WFS = (
    "https://openmaps.gov.bc.ca/geo/pub/WHSE_LAND_AND_NATURAL_RESOURCE.PROT_FUEL_TYPE_SP/ows"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=pub:WHSE_LAND_AND_NATURAL_RESOURCE.PROT_FUEL_TYPE_SP"
    "&outputFormat=json&count=50000"
)


def download_fuel_types(dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    downloaded, skipped = 0, 0

    # --- National fuel types via CWFIS WCS ---
    dest_national = OUT_DIR / "national_fuel_types.tif"
    if dry_run:
        LOGGER.info("[DRY RUN] Would download national fuel type map from CWFIS")
        skipped += 1
    elif not force and dest_national.exists() and dest_national.stat().st_size > 1000:
        LOGGER.info("Skipping (exists): %s", dest_national.name)
        skipped += 1
    else:
        LOGGER.info("Downloading national fuel types from CWFIS WCS...")
        try:
            resp = requests.get(NATIONAL_WCS, timeout=600, stream=True)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "xml" in ct or len(resp.content) < 5000:
                LOGGER.error("CWFIS returned error XML instead of GeoTIFF: %s", resp.text[:300])
            else:
                dest_national.write_bytes(resp.content)
                LOGGER.info("Downloaded: %s (%d MB)", dest_national.name, len(resp.content) // (1024 * 1024))
                downloaded += 1
        except Exception as exc:
            LOGGER.error("Failed national fuel types: %s", exc)

    # --- BC provincial fuel types via WFS ---
    dest_bc = OUT_DIR / "bc_fuel_types.geojson"
    if dry_run:
        LOGGER.info("[DRY RUN] Would download BC provincial fuel types")
        skipped += 1
    elif not force and dest_bc.exists() and dest_bc.stat().st_size > 1000:
        LOGGER.info("Skipping (exists): %s", dest_bc.name)
        skipped += 1
    else:
        LOGGER.info("Downloading BC fuel types from WFS...")
        try:
            resp = requests.get(BC_FUEL_WFS, timeout=600)
            resp.raise_for_status()
            dest_bc.write_text(resp.text, encoding="utf-8")
            LOGGER.info("Downloaded: %s (%d MB)", dest_bc.name, len(resp.content) // (1024 * 1024))
            downloaded += 1
        except Exception as exc:
            LOGGER.error("Failed BC fuel types: %s", exc)

    log_summary("Fuel Types", downloaded, skipped, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download CFFDRS fuel type maps")
    args = parser.parse_args()
    download_fuel_types(args.dry_run, args.force)


if __name__ == "__main__":
    main()
