#!/usr/bin/env python3
"""Generate the INFERNIS BC grid at a given resolution and save to parquet.

Usage:
    python scripts/generate_grid.py                     # 5km (default)
    python scripts/generate_grid.py --resolution 1      # 1km (~2M cells)
    python scripts/generate_grid.py --resolution 1 --to-db  # also insert into DB
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Generate INFERNIS BC grid")
    parser.add_argument(
        "--resolution", type=float, default=None,
        help="Grid resolution in km (default: from config or 5.0)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output parquet path (default: data/processed/grid_{res}km.parquet)",
    )
    parser.add_argument(
        "--to-db", action="store_true",
        help="Also insert grid cells into the database",
    )
    parser.add_argument(
        "--skip-topo", action="store_true",
        help="Skip GEE topography fetch (use zeros)",
    )
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from infernis.config import settings
    from infernis.grid.generator import generate_bc_grid
    from infernis.grid.initializer import (
        _populate_bec_zones,
        _populate_fuel_types,
        _populate_topography,
        save_grid_to_parquet,
    )

    resolution_km = args.resolution or settings.grid_resolution_km
    output_path = args.output or f"data/processed/grid_{int(resolution_km)}km.parquet"

    print(f"Generating BC grid at {resolution_km} km resolution...")
    t0 = time.time()

    # Step 1: Generate raw grid
    grid = generate_bc_grid(resolution_km)
    t_grid = time.time() - t0
    print(f"  Grid generated: {len(grid):,} cells in {t_grid:.1f}s")

    # Step 2: Topography
    if args.skip_topo:
        import numpy as np
        n = len(grid)
        grid["elevation_m"] = np.zeros(n)
        grid["slope_deg"] = np.zeros(n)
        grid["aspect_deg"] = np.zeros(n)
        grid["hillshade"] = np.full(n, 128.0)
        print("  Topography: skipped (using defaults)")
    else:
        grid = _populate_topography(grid)
        print(f"  Topography populated in {time.time() - t0 - t_grid:.1f}s")

    # Step 3: BEC zones
    grid = _populate_bec_zones(grid)

    # Step 4: Fuel types
    grid = _populate_fuel_types(grid)

    # Print summary
    elapsed = time.time() - t0
    print(f"\nGrid Summary:")
    print(f"  Resolution: {resolution_km} km")
    print(f"  Total cells: {len(grid):,}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"\n  BEC zone distribution:")
    for zone, count in grid["bec_zone"].value_counts().items():
        print(f"    {zone}: {count:,} ({100 * count / len(grid):.1f}%)")

    # Save to parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_grid_to_parquet(grid, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Saved to: {output_path} ({file_size_mb:.1f} MB)")

    # Optionally insert into DB
    if args.to_db:
        from infernis.grid.initializer import grid_to_db
        n_inserted = grid_to_db(grid)
        print(f"  Inserted {n_inserted:,} cells into database")


if __name__ == "__main__":
    main()
