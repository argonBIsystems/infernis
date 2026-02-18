"""Orchestrator that runs all INFERNIS data download scripts in sequence.

Usage:
    python scripts/download/download_all.py                    # Run all scripts
    python scripts/download/download_all.py --source era5,dem  # Run specific scripts
    python scripts/download/download_all.py --dry-run          # Show what would run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    ("era5", "01_era5.py"),
    ("gee", "02_gee_satellite.py"),
    ("cnfdb", "03_cnfdb.py"),
    ("bc_fire_perimeters", "04_bc_fire_perimeters.py"),
    ("bc_fire_incidents", "05_bc_fire_incidents.py"),
    ("firms", "06_firms.py"),
    ("cwfis", "07_cwfis.py"),
    ("eccc", "08_eccc.py"),
    ("msc_datamart", "09_msc_datamart.py"),
    ("pcic", "10_pcic.py"),
    ("smap", "11_smap.py"),
    ("aafc", "12_aafc.py"),
    ("fuel_types", "13_fuel_types.py"),
    ("bc_vri", "14_bc_vri.py"),
    ("land_cover", "15_land_cover.py"),
    ("copernicus_ndvi", "16_copernicus_ndvi.py"),
    ("dem", "17_dem.py"),
    ("cldn", "18_cldn.py"),
    ("copernicus_fwi", "19_copernicus_fwi.py"),
    ("bc_roads", "20_bc_roads.py"),
    ("bc_bec", "21_bc_bec.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all INFERNIS data download scripts")
    parser.add_argument("--source", type=str, default="", help="Comma-separated list of sources to run (e.g. era5,dem,cnfdb)")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to all scripts")
    parser.add_argument("--force", action="store_true", help="Pass --force to all scripts")
    parser.add_argument("--start-year", type=int, default=None, help="Override start year")
    parser.add_argument("--end-year", type=int, default=None, help="Override end year")
    args = parser.parse_args()

    if args.source:
        requested = {s.strip().lower() for s in args.source.split(",")}
        scripts_to_run = [(name, script) for name, script in SCRIPTS if name in requested]
        unknown = requested - {name for name, _ in scripts_to_run}
        if unknown:
            print(f"WARNING: Unknown sources: {', '.join(sorted(unknown))}")
            print(f"Available: {', '.join(name for name, _ in SCRIPTS)}")
    else:
        scripts_to_run = SCRIPTS

    print(f"\n{'='*60}")
    print(f"INFERNIS Data Download Orchestrator")
    print(f"Scripts to run: {len(scripts_to_run)}/{len(SCRIPTS)}")
    print(f"{'='*60}\n")

    results = []

    for name, script in scripts_to_run:
        script_path = SCRIPTS_DIR / script
        if not script_path.exists():
            print(f"[SKIP] {name}: script not found ({script})")
            results.append((name, "NOT FOUND", 0))
            continue

        cmd = [sys.executable, str(script_path)]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.force:
            cmd.append("--force")
        if args.start_year is not None:
            cmd.extend(["--start-year", str(args.start_year)])
        if args.end_year is not None:
            cmd.extend(["--end-year", str(args.end_year)])

        print(f"\n--- [{name.upper()}] Running {script} ---")
        start = time.time()
        try:
            result = subprocess.run(cmd, capture_output=False, text=True, timeout=3600)
            elapsed = time.time() - start
            status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
            results.append((name, status, elapsed))
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"[TIMEOUT] {name} exceeded 1 hour limit")
            results.append((name, "TIMEOUT", elapsed))
        except Exception as exc:
            elapsed = time.time() - start
            print(f"[ERROR] {name}: {exc}")
            results.append((name, f"ERROR: {exc}", elapsed))

    print(f"\n{'='*60}")
    print(f"{'Source':<25} {'Status':<20} {'Time':>10}")
    print(f"{'-'*25} {'-'*20} {'-'*10}")
    for name, status, elapsed in results:
        print(f"{name:<25} {status:<20} {elapsed:>8.1f}s")
    print(f"{'='*60}")

    ok_count = sum(1 for _, s, _ in results if s == "OK")
    fail_count = len(results) - ok_count
    print(f"\nTotal: {ok_count} OK, {fail_count} failed/skipped")


if __name__ == "__main__":
    main()
