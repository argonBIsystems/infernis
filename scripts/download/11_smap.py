"""Download NASA SMAP soil moisture data subsetted to BC.

Uses NASA Harmony API for server-side spatial subsetting so we download
only the BC region (~2-5 MB per file) instead of global L3 files (~650 MB).

Data source: NASA NSIDC (Soil Moisture Active Passive)
Products: SPL3SMP_E (enhanced L3 surface, 9km, daily)
Output: data/raw/smap/
"""

import calendar
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BC_BBOX,
    RAW_DIR,
    base_argparser,
    ensure_dir,
    get_logger,
    log_summary,
)

LOGGER = get_logger("smap")
OUT_DIR = RAW_DIR / "smap"

# NASA Harmony API for server-side spatial subsetting
HARMONY_BASE = "https://harmony.earthdata.nasa.gov"
COLLECTION_ID = "C2938664763-NSIDC_CPRD"  # SPL3SMP_E v006

# Sample 4 days per month during fire season to keep volume manageable.
# For fire prediction, this captures biweekly soil moisture drying trends.
SAMPLE_DAYS = [1, 8, 15, 22]


def _get_token() -> str:
    """Get Earthdata bearer token from env."""
    token = os.environ.get("NASA_EARTHDATA_TOKEN")
    if not token:
        raise EnvironmentError("Missing env var: NASA_EARTHDATA_TOKEN (required for Harmony API)")
    return token


def _harmony_url(year: int, month: int, day: int) -> str:
    """Build Harmony spatial subset request URL for a single day."""
    date_str = f"{year}-{month:02d}-{day:02d}"
    return (
        f"{HARMONY_BASE}/{COLLECTION_ID}/ogc-api-coverages/1.0.0/collections/"
        f"all/coverage/rangeset"
        f"?subset=lat({BC_BBOX['south']}:{BC_BBOX['north']})"
        f"&subset=lon({BC_BBOX['west']}:{BC_BBOX['east']})"
        f"&subset=time(\"{date_str}T00:00:00Z\":\"{date_str}T23:59:59Z\")"
        f"&skipPreview=true&maxResults=1"
    )


def download_smap(start_year: int, end_year: int, dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)

    if dry_run:
        LOGGER.info("[DRY RUN] Would download SMAP data for %d-%d", start_year, end_year)
        log_summary("SMAP", 0, 0, logger=LOGGER)
        return

    token = _get_token()
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    downloaded, skipped, errors = 0, 0, 0

    for year in range(max(start_year, 2015), end_year + 1):
        for month in range(4, 11):  # Fire season: April-October
            for day in SAMPLE_DAYS:
                max_day = calendar.monthrange(year, month)[1]
                if day > max_day:
                    continue

                fname = f"smap_bc_{year}_{month:02d}_{day:02d}.nc4"
                dest = OUT_DIR / fname
                if not force and dest.exists() and dest.stat().st_size > 1000:
                    skipped += 1
                    continue

                url = _harmony_url(year, month, day)
                try:
                    # Harmony returns 303 with CloudFront URL in Location header.
                    # Don't auto-follow — grab the Location and download separately
                    # so we can handle errors on each step.
                    resp = session.get(url, timeout=180, allow_redirects=False)

                    if resp.status_code == 303:
                        data_url = resp.headers.get("Location", "")
                        if not data_url:
                            LOGGER.error("No Location header in 303 for %s", fname)
                            errors += 1
                            continue
                        # Download the subsetted file (no auth needed, signed URL)
                        dl = requests.get(data_url, timeout=120)
                        dl.raise_for_status()
                        if len(dl.content) < 1000:
                            LOGGER.warning("Skipping %s: too small (%d bytes)", fname, len(dl.content))
                            errors += 1
                            continue
                        dest.write_bytes(dl.content)
                        LOGGER.info("Downloaded: %s (%d KB)", fname, len(dl.content) // 1024)
                        downloaded += 1

                    elif resp.status_code == 200:
                        # Synchronous response (small subset might return directly)
                        ct = resp.headers.get("content-type", "")
                        if "json" in ct:
                            # Async job response — poll
                            data = resp.json()
                            job_id = data.get("jobID", "")
                            if not job_id:
                                LOGGER.warning("Unexpected JSON for %s: %s", fname, str(data)[:200])
                                errors += 1
                                continue
                            data_url = _poll_job(session, job_id)
                            if not data_url:
                                errors += 1
                                continue
                            dl = requests.get(data_url, timeout=120)
                            dl.raise_for_status()
                            dest.write_bytes(dl.content)
                            LOGGER.info("Downloaded: %s (%d KB)", fname, len(dl.content) // 1024)
                            downloaded += 1
                        else:
                            # Direct binary data
                            if len(resp.content) < 1000:
                                LOGGER.warning("Skipping %s: too small (%d bytes)", fname, len(resp.content))
                                errors += 1
                                continue
                            dest.write_bytes(resp.content)
                            LOGGER.info("Downloaded: %s (%d KB)", fname, len(resp.content) // 1024)
                            downloaded += 1
                    else:
                        LOGGER.error("Harmony HTTP %d for %s", resp.status_code, fname)
                        errors += 1

                except Exception as exc:
                    LOGGER.error("Failed: %s: %s", fname, exc)
                    errors += 1

                # Be polite to Harmony
                time.sleep(0.5)

        LOGGER.info("Year %d done — downloaded: %d, skipped: %d, errors: %d", year, downloaded, skipped, errors)

    log_summary("SMAP", downloaded, skipped, logger=LOGGER)


def _poll_job(session: requests.Session, job_id: str, timeout: int = 300) -> str | None:
    """Poll Harmony async job until complete, return download URL."""
    job_url = f"{HARMONY_BASE}/jobs/{job_id}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = session.get(job_url, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        status = data.get("status", "")
        if status == "successful":
            for link in data.get("links", []):
                href = link.get("href", "")
                if any(href.endswith(ext) for ext in (".nc4", ".nc", ".h5")):
                    return href
            return None
        elif status in ("failed", "canceled"):
            LOGGER.warning("Job %s: %s", status, data.get("message", ""))
            return None
        time.sleep(5)
    return None


def main() -> None:
    parser = base_argparser("Download NASA SMAP soil moisture data for BC (subsetted)")
    args = parser.parse_args()
    download_smap(args.start_year, args.end_year, args.dry_run, args.force)


if __name__ == "__main__":
    main()
