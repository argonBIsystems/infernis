"""Google Earth Engine satellite data pipeline."""

import logging
from datetime import date, timedelta

import numpy as np

from infernis.config import settings

logger = logging.getLogger(__name__)

# GEE sampleRegions limit per call
_BATCH_SIZE = 5000


class GEEPipeline:
    """Fetches satellite data from Google Earth Engine for BC grid cells.

    All fetch methods handle arbitrarily large grids by batching requests
    to stay within GEE's per-call limits (~5000 features).
    """

    def __init__(self):
        self._initialized = False

    def _ensure_init(self):
        """Lazy initialization of Earth Engine."""
        if self._initialized:
            return

        import json
        import os

        import ee

        gee_key = settings.gee_service_account_key
        if gee_key:
            # Detect key type: file path, full JSON, or raw private key
            if os.path.isfile(gee_key):
                with open(gee_key) as f:
                    key_data = json.load(f)
            elif gee_key.strip().startswith("{"):
                key_data = json.loads(gee_key)
            elif gee_key.strip().startswith("-----BEGIN"):
                # Raw private key — reconstruct service account dict from env vars
                client_email = settings.gee_client_email or os.environ.get("GEE_CLIENT_EMAIL", "")
                key_data = {
                    "type": "service_account",
                    "project_id": settings.gee_project,
                    "private_key_id": os.environ.get("GEE_PRIVATE_KEY_ID", ""),
                    "private_key": gee_key.replace("\\n", "\n"),
                    "client_email": client_email,
                    "client_id": os.environ.get("GEE_CLIENT_ID", ""),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email.replace('@', '%40')}",
                    "universe_domain": "googleapis.com",
                }
            else:
                raise ValueError("GEE_PRIVATE_KEY is not a file path, JSON, or PEM key")

            credentials = ee.ServiceAccountCredentials(
                key_data["client_email"],
                key_data=key_data["private_key"],
            )
            ee.Initialize(credentials=credentials, project=settings.gee_project)
        else:
            ee.Initialize(project=settings.gee_project)

        self._initialized = True
        logger.info("Google Earth Engine initialized (project=%s)", settings.gee_project)

    def _sample_image_batched(
        self,
        image,
        lats: np.ndarray,
        lons: np.ndarray,
        bands: list[str],
        scale: int,
        defaults: dict[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample an ee.Image at point locations in batches.

        Uses index properties to maintain point-value correspondence even when
        some points fall outside the image extent (e.g., ocean).

        Returns dict mapping band name to numpy arrays (same length as input).
        Missing values are filled with defaults.
        """
        import ee

        if defaults is None:
            defaults = {band: 0.0 for band in bands}

        n = len(lats)
        results = {band: np.full(n, defaults.get(band, 0.0)) for band in bands}

        for start in range(0, n, _BATCH_SIZE):
            end = min(start + _BATCH_SIZE, n)
            batch_lats = lats[start:end]
            batch_lons = lons[start:end]

            points = ee.FeatureCollection(
                [
                    ee.Feature(
                        ee.Geometry.Point(float(lon), float(lat)),
                        {"_idx": i},
                    )
                    for i, (lat, lon) in enumerate(zip(batch_lats, batch_lons))
                ]
            )

            sampled = image.sampleRegions(collection=points, scale=scale)
            indices = sampled.aggregate_array("_idx").getInfo()
            for band in bands:
                values = sampled.aggregate_array(band).getInfo()
                for idx, val in zip(indices, values):
                    results[band][start + idx] = val

            if n > _BATCH_SIZE and (start // _BATCH_SIZE) % 50 == 0:
                logger.info("  GEE sampling: %d / %d points...", end, n)

        return results

    def fetch_ndvi(
        self,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        target_date: date,
    ) -> np.ndarray:
        """Fetch MODIS NDVI for grid cells. Returns array of NDVI values."""
        self._ensure_init()
        import ee

        end_date = target_date
        start_date = target_date - timedelta(days=32)

        collection = (
            ee.ImageCollection("MODIS/061/MOD13A1")
            .filterDate(start_date.isoformat(), end_date.isoformat())
            .select("NDVI")
        )

        image = collection.sort("system:time_start", False).first()
        if image is None:
            logger.warning("No MODIS NDVI data available for %s", target_date)
            return np.full(len(grid_lats), 0.5)

        image = image.multiply(0.0001)

        logger.info("Fetching MODIS NDVI for %d cells...", len(grid_lats))
        sampled = self._sample_image_batched(
            image,
            grid_lats,
            grid_lons,
            ["NDVI"],
            scale=500,
            defaults={"NDVI": 0.5},
        )
        return sampled["NDVI"]

    def fetch_snow_cover(
        self,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        target_date: date,
    ) -> np.ndarray:
        """Fetch MODIS snow cover. Returns boolean array."""
        self._ensure_init()
        import ee

        end_date = target_date
        start_date = target_date - timedelta(days=8)

        collection = (
            ee.ImageCollection("MODIS/061/MOD10A1")
            .filterDate(start_date.isoformat(), end_date.isoformat())
            .select("NDSI_Snow_Cover")
        )

        image = collection.sort("system:time_start", False).first()
        if image is None:
            logger.warning("No MODIS snow data available for %s", target_date)
            return np.zeros(len(grid_lats), dtype=bool)

        snow_binary = image.gt(50)

        logger.info("Fetching MODIS snow cover for %d cells...", len(grid_lats))
        sampled = self._sample_image_batched(
            snow_binary,
            grid_lats,
            grid_lons,
            ["NDSI_Snow_Cover"],
            scale=500,
            defaults={"NDSI_Snow_Cover": 0.0},
        )
        return sampled["NDSI_Snow_Cover"].astype(bool)

    def fetch_topography(
        self,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Fetch CDEM-derived topographic features. Returns dict of arrays."""
        self._ensure_init()
        import ee

        dem = ee.ImageCollection("NRCan/CDEM").mosaic()
        elevation = dem.select("elevation")
        slope = ee.Terrain.slope(elevation)
        aspect = ee.Terrain.aspect(elevation)
        hillshade = ee.Terrain.hillshade(elevation)

        combined = elevation.addBands(slope).addBands(aspect).addBands(hillshade)

        logger.info("Fetching CDEM topography for %d cells...", len(grid_lats))
        sampled = self._sample_image_batched(
            combined,
            grid_lats,
            grid_lons,
            ["elevation", "slope", "aspect", "hillshade"],
            scale=100,
            defaults={"elevation": 0.0, "slope": 0.0, "aspect": 0.0, "hillshade": 128.0},
        )

        return {
            "elevation_m": sampled["elevation"],
            "slope_deg": sampled["slope"],
            "aspect_deg": sampled["aspect"],
            "hillshade": sampled["hillshade"],
        }
