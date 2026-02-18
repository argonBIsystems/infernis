"""Download Canadian Lightning Detection Network gridded flash density data.

Data source: Open Canada / MSC Datamart
URL: https://dd.weather.gc.ca/today/lightning/
Output: data/raw/cldn/
"""

import sys
from html.parser import HTMLParser
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW_DIR, base_argparser, ensure_dir, get_logger, log_summary

LOGGER = get_logger("cldn")
OUT_DIR = RAW_DIR / "cldn"

CLDN_BASE_URL = "https://dd.weather.gc.ca/today/lightning/"


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and (value.endswith(".csv") or value.endswith(".xml") or value.endswith(".gif")):
                    self.links.append(value)


def download_cldn(dry_run: bool = False, force: bool = False) -> None:
    ensure_dir(OUT_DIR)

    if dry_run:
        LOGGER.info("[DRY RUN] Would download CLDN lightning data from MSC Datamart")
        log_summary("CLDN", 0, 1, logger=LOGGER)
        return

    dest = OUT_DIR / "cldn_current_index.html"
    try:
        resp = requests.get(CLDN_BASE_URL, timeout=60)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")
        LOGGER.info("Downloaded CLDN index: %s", dest.name)

        parser = _LinkParser()
        parser.feed(resp.text)

        downloaded = 0
        for link in parser.links:
            url = f"{CLDN_BASE_URL}{link}"
            file_dest = OUT_DIR / link
            if not force and file_dest.exists() and file_dest.stat().st_size > 0:
                continue
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                file_dest.write_bytes(r.content)
                downloaded += 1
            except Exception as exc:
                LOGGER.error("Failed %s: %s", link, exc)

        log_summary("CLDN", downloaded + 1, 0, logger=LOGGER)
    except Exception as exc:
        LOGGER.error("Failed to download CLDN index: %s", exc)
        log_summary("CLDN", 0, 0, logger=LOGGER)


def main() -> None:
    parser = base_argparser("Download CLDN lightning data")
    args = parser.parse_args()
    download_cldn(args.dry_run, args.force)


if __name__ == "__main__":
    main()
