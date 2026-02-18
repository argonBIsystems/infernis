"""Download CWFIS active fires and pre-computed FWI grids.

Data source: Canadian Wildland Fire Information System (CWFIS)
WFS: https://cwfis.cfs.nrcan.gc.ca/geoserver/wfs
WCS: https://cwfis.cfs.nrcan.gc.ca/geoserver/wcs
Output: data/raw/cwfis/
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BC_BBOX, RAW_DIR, base_argparser, ensure_dir, get_logger, log_summary

LOGGER = get_logger("cwfis")
OUT_DIR = RAW_DIR / "cwfis"

ACTIVE_FIRES_WFS = (
    "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/ows"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=public:activefires_current"
    "&outputFormat=application/json"
    f"&bbox={BC_BBOX['south']},{BC_BBOX['west']},{BC_BBOX['north']},{BC_BBOX['east']}"
)

# Coverage IDs use double-underscore and _current suffix for latest grids
FWI_COMPONENTS = ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]


def download_active_fires(dry_run: bool, force: bool) -> tuple[int, int]:
    dest = OUT_DIR / "cwfis_active_fires.geojson"
    if not force and dest.exists() and dest.stat().st_size > 0:
        LOGGER.info("Skipping (exists): %s", dest.name)
        return 0, 1
    if dry_run:
        LOGGER.info("[DRY RUN] Would download CWFIS active fires")
        return 0, 1
    try:
        resp = requests.get(ACTIVE_FIRES_WFS, timeout=120)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")
        LOGGER.info("Downloaded: %s", dest.name)
        return 1, 0
    except Exception as exc:
        LOGGER.error("Failed: %s", exc)
        return 0, 1


def download_fwi_grids(dry_run: bool, force: bool) -> tuple[int, int]:
    downloaded, skipped = 0, 0
    for component in FWI_COMPONENTS:
        dest = OUT_DIR / f"cwfis_{component}.tif"
        if not force and dest.exists() and dest.stat().st_size > 0:
            LOGGER.info("Skipping (exists): %s", dest.name)
            skipped += 1
            continue
        if dry_run:
            LOGGER.info("[DRY RUN] Would download CWFIS %s grid", component.upper())
            skipped += 1
            continue
        # Use _current suffix (archived grids have server-side geometry bugs)
        # Use /geoserver/ows (not /geoserver/public/ows) to avoid 404
        wcs_url = (
            f"https://cwfis.cfs.nrcan.gc.ca/geoserver/ows"
            f"?service=WCS&version=2.0.1&request=GetCoverage"
            f"&coverageId=public__{component}_current"
            f"&format=image/geotiff"
        )
        try:
            resp = requests.get(wcs_url, timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            LOGGER.info("Downloaded: %s", dest.name)
            downloaded += 1
        except Exception as exc:
            LOGGER.error("Failed to download CWFIS %s: %s", component, exc)
            skipped += 1
    return downloaded, skipped


def main() -> None:
    parser = base_argparser("Download CWFIS active fires and FWI grids")
    args = parser.parse_args()
    ensure_dir(OUT_DIR)

    d1, s1 = download_active_fires(args.dry_run, args.force)
    d2, s2 = download_fwi_grids(args.dry_run, args.force)
    log_summary("CWFIS", d1 + d2, s1 + s2, logger=LOGGER)


if __name__ == "__main__":
    main()
