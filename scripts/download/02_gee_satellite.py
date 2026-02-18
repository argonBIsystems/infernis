"""Download satellite data from Google Earth Engine for BC.

Datasets:
  - MODIS NDVI (MOD13A1) - 500m, 16-day composites
  - MODIS LAI/FPAR (MOD15A2H) - 500m, 8-day composites
  - MODIS Snow Cover (MOD10A1) - 500m, daily
  - Sentinel-2 NDVI - 10m (aggregated to 500m for consistency)
  - CDEM Elevation - 20m (aggregated to 500m)

Output: data/raw/gee/<dataset>/<dataset>_bc_{year}.tif

NOTE: BC is too large for a single getDownloadURL call at 500m.
This script splits BC into 4x4-degree tiles, downloads each, then
merges them into a single GeoTIFF per image using rasterio.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BC_BBOX,
    RAW_DIR,
    base_argparser,
    ensure_dir,
    get_gee_credentials,
    get_logger,
    log_summary,
)

LOGGER = get_logger("gee_satellite")
OUT_DIR = RAW_DIR / "gee"

EXPORT_SCALE = 500

# Tile size in degrees - 4x4 keeps each tile well under GEE download limits
TILE_DEG = 4


def init_gee():
    """Initialize Earth Engine with service account credentials."""
    import ee

    creds_dict = get_gee_credentials()
    credentials = ee.ServiceAccountCredentials(
        creds_dict["client_email"], key_data=creds_dict["private_key"]
    )
    ee.Initialize(credentials=credentials, project=creds_dict["project_id"])
    LOGGER.info("GEE initialized for project: %s", creds_dict["project_id"])


def _get_bc_geometry():
    import ee

    return ee.Geometry.Rectangle(
        [BC_BBOX["west"], BC_BBOX["south"], BC_BBOX["east"], BC_BBOX["north"]]
    )


def _generate_tiles():
    """Generate a list of (south, west, north, east) tile bboxes covering BC."""
    tiles = []
    lat = BC_BBOX["south"]
    while lat < BC_BBOX["north"]:
        lon = BC_BBOX["west"]
        while lon < BC_BBOX["east"]:
            tiles.append((
                lat,
                lon,
                min(lat + TILE_DEG, BC_BBOX["north"]),
                min(lon + TILE_DEG, BC_BBOX["east"]),
            ))
            lon += TILE_DEG
        lat += TILE_DEG
    return tiles


def _download_tile(image, tile_bbox, dest: Path, scale: int = EXPORT_SCALE) -> bool:
    """Download a single tile of an ee.Image to a GeoTIFF."""
    import ee
    import requests as req

    south, west, north, east = tile_bbox
    region = ee.Geometry.Rectangle([west, south, east, north])

    try:
        url = image.getDownloadURL({
            "scale": scale,
            "region": region,
            "format": "GEO_TIFF",
            "maxPixels": 1e10,
        })
        response = req.get(url, stream=True, timeout=600)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as exc:
        LOGGER.warning("Tile download failed (%s): %s", dest.name, exc)
        return False


def _merge_tiles(tile_paths: list, dest: Path) -> bool:
    """Merge multiple GeoTIFF tiles into a single output file."""
    import rasterio
    from rasterio.merge import merge

    try:
        datasets = [rasterio.open(p) for p in tile_paths]
        mosaic, out_transform = merge(datasets)
        out_meta = datasets[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "lzw",
        })
        for ds in datasets:
            ds.close()

        ensure_dir(dest.parent)
        with rasterio.open(dest, "w", **out_meta) as dst:
            dst.write(mosaic)
        LOGGER.info("Merged %d tiles -> %s", len(tile_paths), dest.name)
        return True
    except Exception as exc:
        LOGGER.error("Merge failed for %s: %s", dest.name, exc)
        return False


def export_image_tiled(image, description: str, dest: Path, scale: int = EXPORT_SCALE) -> bool:
    """Export an ee.Image by downloading tiles and merging them."""
    tiles = _generate_tiles()
    LOGGER.info("Exporting %s via %d tiles at %dm scale...", description, len(tiles), scale)

    with tempfile.TemporaryDirectory(prefix="gee_tiles_") as tmpdir:
        tile_paths = []
        for i, tile_bbox in enumerate(tiles):
            tile_path = Path(tmpdir) / f"tile_{i:03d}.tif"
            LOGGER.info(
                "  Tile %d/%d [%.1f,%.1f -> %.1f,%.1f]",
                i + 1, len(tiles),
                tile_bbox[0], tile_bbox[1], tile_bbox[2], tile_bbox[3],
            )
            if _download_tile(image, tile_bbox, tile_path, scale):
                # Only include tiles that have actual data (not empty)
                if tile_path.stat().st_size > 1000:
                    tile_paths.append(tile_path)
                else:
                    LOGGER.info("  Tile %d empty (no data in region), skipping", i + 1)
            else:
                LOGGER.warning("  Tile %d failed, continuing with remaining tiles", i + 1)

        if not tile_paths:
            LOGGER.error("No tiles downloaded for %s", description)
            return False

        return _merge_tiles(tile_paths, dest)


def download_modis_ndvi(start_year: int, end_year: int, dry_run: bool, force: bool) -> tuple:
    out = ensure_dir(OUT_DIR / "modis_ndvi")
    downloaded, skipped = 0, 0

    if dry_run:
        for year in range(start_year, end_year + 1):
            LOGGER.info("[DRY RUN] Would export: modis_ndvi_bc_%d.tif", year)
            skipped += 1
        return downloaded, skipped

    import ee

    bc = _get_bc_geometry()
    for year in range(start_year, end_year + 1):
        dest = out / f"modis_ndvi_bc_{year}.tif"
        if not force and dest.exists() and dest.stat().st_size > 0:
            LOGGER.info("Skipping (exists): %s", dest.name)
            skipped += 1
            continue
        collection = (
            ee.ImageCollection("MODIS/061/MOD13A1")
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filterBounds(bc)
            .select("NDVI")
        )
        image = collection.median().clip(bc)
        if export_image_tiled(image, f"MODIS NDVI {year}", dest):
            downloaded += 1
        else:
            skipped += 1
    return downloaded, skipped


def download_modis_lai(start_year: int, end_year: int, dry_run: bool, force: bool) -> tuple:
    out = ensure_dir(OUT_DIR / "modis_lai")
    downloaded, skipped = 0, 0

    if dry_run:
        for year in range(start_year, end_year + 1):
            LOGGER.info("[DRY RUN] Would export: modis_lai_bc_%d.tif", year)
            skipped += 1
        return downloaded, skipped

    import ee

    bc = _get_bc_geometry()
    for year in range(start_year, end_year + 1):
        dest = out / f"modis_lai_bc_{year}.tif"
        if not force and dest.exists() and dest.stat().st_size > 0:
            LOGGER.info("Skipping (exists): %s", dest.name)
            skipped += 1
            continue
        collection = (
            ee.ImageCollection("MODIS/061/MOD15A2H")
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filterBounds(bc)
            .select(["Lai_500m", "Fpar_500m"])
        )
        image = collection.median().clip(bc)
        if export_image_tiled(image, f"MODIS LAI {year}", dest):
            downloaded += 1
        else:
            skipped += 1
    return downloaded, skipped


def download_modis_snow(start_year: int, end_year: int, dry_run: bool, force: bool) -> tuple:
    out = ensure_dir(OUT_DIR / "modis_snow")
    downloaded, skipped = 0, 0

    if dry_run:
        for year in range(start_year, end_year + 1):
            LOGGER.info("[DRY RUN] Would export: modis_snow_bc_%d.tif", year)
            skipped += 1
        return downloaded, skipped

    import ee

    bc = _get_bc_geometry()
    for year in range(start_year, end_year + 1):
        dest = out / f"modis_snow_bc_{year}.tif"
        if not force and dest.exists() and dest.stat().st_size > 0:
            LOGGER.info("Skipping (exists): %s", dest.name)
            skipped += 1
            continue
        collection = (
            ee.ImageCollection("MODIS/061/MOD10A1")
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filterBounds(bc)
            .select("NDSI_Snow_Cover")
        )
        image = collection.max().clip(bc)
        if export_image_tiled(image, f"MODIS Snow {year}", dest):
            downloaded += 1
        else:
            skipped += 1
    return downloaded, skipped


def download_sentinel2_ndvi(start_year: int, end_year: int, dry_run: bool, force: bool) -> tuple:
    out = ensure_dir(OUT_DIR / "sentinel2_ndvi")
    downloaded, skipped = 0, 0

    if dry_run:
        for year in range(max(start_year, 2016), end_year + 1):
            LOGGER.info("[DRY RUN] Would export: sentinel2_ndvi_bc_%d.tif", year)
            skipped += 1
        return downloaded, skipped

    import ee

    bc = _get_bc_geometry()
    for year in range(max(start_year, 2016), end_year + 1):
        dest = out / f"sentinel2_ndvi_bc_{year}.tif"
        if not force and dest.exists() and dest.stat().st_size > 0:
            LOGGER.info("Skipping (exists): %s", dest.name)
            skipped += 1
            continue
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(f"{year}-05-01", f"{year}-09-30")
            .filterBounds(bc)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        )
        ndvi = (
            collection.median()
            .normalizedDifference(["B8", "B4"])
            .rename("NDVI")
            .clip(bc)
        )
        if export_image_tiled(ndvi, f"Sentinel-2 NDVI {year}", dest):
            downloaded += 1
        else:
            skipped += 1
    return downloaded, skipped


def download_cdem(dry_run: bool, force: bool) -> tuple:
    out = ensure_dir(OUT_DIR / "cdem")
    dest = out / "cdem_bc.tif"
    if not force and dest.exists() and dest.stat().st_size > 0:
        LOGGER.info("Skipping (exists): %s", dest.name)
        return 0, 1
    if dry_run:
        LOGGER.info("[DRY RUN] Would export: cdem_bc.tif")
        return 0, 1

    import ee

    bc = _get_bc_geometry()
    image = ee.Image("NRCan/CDEM").select("elevation").clip(bc)
    if export_image_tiled(image, "CDEM", dest, scale=100):
        return 1, 0
    return 0, 1


def main() -> None:
    parser = base_argparser("Download satellite data from Google Earth Engine for BC")
    args = parser.parse_args()

    if not args.dry_run:
        init_gee()

    total_dl, total_sk = 0, 0
    for name, func in [
        ("MODIS NDVI", lambda: download_modis_ndvi(args.start_year, args.end_year, args.dry_run, args.force)),
        ("MODIS LAI", lambda: download_modis_lai(args.start_year, args.end_year, args.dry_run, args.force)),
        ("MODIS Snow", lambda: download_modis_snow(args.start_year, args.end_year, args.dry_run, args.force)),
        ("Sentinel-2 NDVI", lambda: download_sentinel2_ndvi(args.start_year, args.end_year, args.dry_run, args.force)),
        ("CDEM", lambda: download_cdem(args.dry_run, args.force)),
    ]:
        LOGGER.info("--- %s ---", name)
        dl, sk = func()
        total_dl += dl
        total_sk += sk

    log_summary("GEE Satellite", total_dl, total_sk, logger=LOGGER)


if __name__ == "__main__":
    main()
