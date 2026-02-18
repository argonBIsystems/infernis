"""Shared utilities for INFERNIS data download scripts."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
DATA_DIR = _PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# BC bounding box [North, West, South, East]
# ---------------------------------------------------------------------------
BC_BBOX = {"north": 60, "south": 48, "west": -140, "east": -114}
BC_BBOX_LIST = [BC_BBOX["north"], BC_BBOX["west"], BC_BBOX["south"], BC_BBOX["east"]]

# ---------------------------------------------------------------------------
# Default temporal scope
# ---------------------------------------------------------------------------
YEAR_START = 2015
YEAR_END = 2026

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# CLI argument helpers
# ---------------------------------------------------------------------------
def base_argparser(description: str) -> argparse.ArgumentParser:
    """Return an ArgumentParser with common flags used by all download scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--start-year", type=int, default=YEAR_START, help=f"Start year (default: {YEAR_START})")
    parser.add_argument("--end-year", type=int, default=YEAR_END, help=f"End year (default: {YEAR_END})")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    return parser


# ---------------------------------------------------------------------------
# HTTP download with retries and progress bar
# ---------------------------------------------------------------------------
def download_file(
    url: str,
    dest: Path,
    skip_existing: bool = True,
    retries: int = 3,
    timeout: int = 60,
    chunk_size: int = 8192,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Download a file from *url* to *dest* with retry logic and progress bar.

    Returns True if the file was downloaded, False if skipped.
    """
    log = logger or get_logger("download")
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        log.info("Skipping (exists): %s", dest.name)
        return False

    ensure_dir(dest.parent)
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(dest, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=dest.name, disable=total == 0
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))
            log.info("Downloaded: %s", dest.name)
            return True
        except (requests.RequestException, IOError) as exc:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                log.error("Failed to download after %d attempts: %s", retries, url)
                raise


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------
def get_gee_credentials() -> dict:
    """Reconstruct GEE service account JSON dict from env vars."""
    required = ["GEE_PROJECT_ID", "GEE_PRIVATE_KEY_ID", "GEE_PRIVATE_KEY", "GEE_CLIENT_EMAIL", "GEE_CLIENT_ID"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(f"Missing GEE env vars: {', '.join(missing)}")
    client_email = os.environ["GEE_CLIENT_EMAIL"]
    return {
        "type": "service_account",
        "project_id": os.environ["GEE_PROJECT_ID"],
        "private_key_id": os.environ["GEE_PRIVATE_KEY_ID"],
        "private_key": os.environ["GEE_PRIVATE_KEY"].replace("\\n", "\n"),
        "client_email": client_email,
        "client_id": os.environ["GEE_CLIENT_ID"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email.replace('@', '%40')}",
        "universe_domain": "googleapis.com",
    }


def get_cds_client():
    """Return a configured cdsapi.Client from env vars."""
    import cdsapi

    url = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
    key = os.environ.get("CDS_API_KEY")
    if not key:
        raise EnvironmentError("Missing env var: CDS_API_KEY")
    return cdsapi.Client(url=url, key=key)


def get_earthdata_session() -> requests.Session:
    """Return an authenticated requests.Session for NASA Earthdata.

    Prefers bearer token auth (NASA_EARTHDATA_TOKEN) which is required for
    NSIDC/SMAP OAuth redirects. Falls back to basic auth if no token is set.
    """
    token = os.environ.get("NASA_EARTHDATA_TOKEN")
    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
        return session
    user = os.environ.get("NASA_EARTHDATA_USER")
    password = os.environ.get("NASA_EARTHDATA_PASS")
    if not user or not password:
        raise EnvironmentError(
            "Missing env vars: set NASA_EARTHDATA_TOKEN (preferred) "
            "or both NASA_EARTHDATA_USER and NASA_EARTHDATA_PASS"
        )
    session.auth = (user, password)
    return session


def get_firms_map_key() -> str:
    """Return the NASA FIRMS MAP_KEY from env vars."""
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise EnvironmentError("Missing env var: FIRMS_MAP_KEY")
    return key


# ---------------------------------------------------------------------------
# Summary logging
# ---------------------------------------------------------------------------
def log_summary(
    source: str,
    files_downloaded: int,
    files_skipped: int,
    total_bytes: int = 0,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Print a consistent download summary for a source."""
    log = logger or get_logger("download")
    size_mb = total_bytes / (1024 * 1024) if total_bytes else 0
    log.info(
        "=== %s Summary: %d downloaded, %d skipped, %.1f MB total ===",
        source,
        files_downloaded,
        files_skipped,
        size_mb,
    )
