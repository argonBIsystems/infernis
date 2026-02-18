"""Download ERA5 reanalysis weather and soil moisture data for BC.

Data source: Copernicus Climate Data Store (CDS)
Dataset: reanalysis-era5-single-levels
Output: data/raw/era5/era5_bc_{year}_{month:02d}.nc
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BC_BBOX,
    RAW_DIR,
    base_argparser,
    ensure_dir,
    get_cds_client,
    get_logger,
    log_summary,
)

LOGGER = get_logger("era5")
OUT_DIR = RAW_DIR / "era5"

# All ERA5 single-level variables needed by INFERNIS
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "potential_evaporation",
    "surface_pressure",
    "surface_net_solar_radiation",
]

# BC bounding box in CDS format: [N, W, S, E]
AREA = [BC_BBOX["north"], BC_BBOX["west"], BC_BBOX["south"], BC_BBOX["east"]]


def download_era5(start_year: int, end_year: int, dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)
    client = None if dry_run else get_cds_client()
    downloaded = 0
    skipped = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            fname = f"era5_bc_{year}_{month:02d}.nc"
            dest = OUT_DIR / fname

            if not force and dest.exists() and dest.stat().st_size > 0:
                LOGGER.info("Skipping (exists): %s", fname)
                skipped += 1
                continue

            if dry_run:
                LOGGER.info("[DRY RUN] Would download: %s", fname)
                skipped += 1
                continue

            LOGGER.info("Requesting: %s", fname)
            days = [str(d).zfill(2) for d in range(1, 32)]
            request = {
                "product_type": ["reanalysis"],
                "variable": VARIABLES,
                "year": str(year),
                "month": str(month).zfill(2),
                "day": days,
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": AREA,
                "data_format": "netcdf",
            }
            try:
                client.retrieve("reanalysis-era5-single-levels", request, str(dest))
                # CDS wraps NetCDF in a zip archive — extract the actual .nc file
                import zipfile
                if zipfile.is_zipfile(dest):
                    import shutil
                    with zipfile.ZipFile(dest, "r") as zf:
                        nc_members = [m for m in zf.namelist() if m.endswith(".nc")]
                        if nc_members:
                            zf.extract(nc_members[0], dest.parent / "_tmp_extract")
                            shutil.move(str(dest.parent / "_tmp_extract" / nc_members[0]), str(dest))
                            (dest.parent / "_tmp_extract").rmdir()
                LOGGER.info("Downloaded: %s", fname)
                downloaded += 1
            except Exception as exc:
                LOGGER.error("Failed to download %s: %s", fname, exc)

    log_summary("ERA5", downloaded, skipped, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download ERA5 reanalysis data for BC")
    args = parser.parse_args()
    download_era5(args.start_year, args.end_year, args.dry_run, args.force)


if __name__ == "__main__":
    main()
