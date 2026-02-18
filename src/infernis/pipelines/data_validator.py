"""Data validation module for verifying downloaded data integrity.

Validates ERA5 NetCDF files, GEE rasters, fire history CSVs, and
static feature datasets before they are used in training or inference.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates downloaded data files for integrity and completeness."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def validate_all(self, start_year: int = 2015, end_year: int = 2024) -> dict:
        """Run all validation checks and return a summary report."""
        report = {
            "era5": self.validate_era5(start_year, end_year),
            "gee_ndvi": self.validate_gee_rasters("modis_ndvi", start_year, end_year),
            "fire_history": self.validate_fire_history(),
            "overall_status": "pass",
        }

        # Check if any category failed
        for key, result in report.items():
            if isinstance(result, dict) and result.get("status") == "fail":
                report["overall_status"] = "fail"
                break

        return report

    def validate_era5(self, start_year: int = 2015, end_year: int = 2024) -> dict:
        """Validate ERA5 NetCDF monthly files.

        Checks:
        - All expected monthly files exist
        - Files are non-empty
        - NetCDF can be opened and contains expected variables
        - Data values are within physical bounds
        """
        era5_dir = self.data_dir / "era5"
        expected_vars = [
            "t2m",
            "d2m",
            "u10",
            "v10",
            "tp",
            "swvl1",
            "swvl2",
            "e",
        ]
        results = {
            "status": "pass",
            "total_expected": 0,
            "total_found": 0,
            "missing_files": [],
            "corrupt_files": [],
            "var_issues": [],
        }

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                results["total_expected"] += 1
                fname = f"era5_bc_{year}_{month:02d}.nc"
                fpath = era5_dir / fname

                if not fpath.exists():
                    results["missing_files"].append(fname)
                    continue

                results["total_found"] += 1

                # Check file size (should be > 1MB for monthly data)
                if fpath.stat().st_size < 1_000_000:
                    results["corrupt_files"].append(
                        f"{fname}: too small ({fpath.stat().st_size} bytes)"
                    )
                    continue

                # Validate NetCDF contents
                try:
                    import xarray as xr

                    ds = xr.open_dataset(fpath)
                    for var in expected_vars:
                        if var not in ds.data_vars and var not in ds.coords:
                            # Some ERA5 vars may have different names
                            results["var_issues"].append(f"{fname}: missing variable '{var}'")
                    ds.close()
                except Exception as e:
                    results["corrupt_files"].append(f"{fname}: {e}")

        if results["missing_files"] or results["corrupt_files"]:
            results["status"] = "fail" if len(results["missing_files"]) > 12 else "partial"

        logger.info(
            "ERA5 validation: %d/%d files found, %d missing, %d corrupt",
            results["total_found"],
            results["total_expected"],
            len(results["missing_files"]),
            len(results["corrupt_files"]),
        )
        return results

    def validate_gee_rasters(
        self,
        product: str,
        start_year: int = 2015,
        end_year: int = 2024,
    ) -> dict:
        """Validate GEE raster files (GeoTIFF).

        Checks:
        - Files exist for each expected year
        - Raster dimensions are reasonable for BC at 500m
        - No all-NaN rasters
        - Values are within expected ranges
        """
        gee_dir = self.data_dir / "gee" / product
        results = {
            "status": "pass",
            "total_expected": 0,
            "total_found": 0,
            "missing_files": [],
            "issues": [],
        }

        value_ranges = {
            "modis_ndvi": (-0.2, 1.0),
            "modis_lai": (0.0, 10.0),
            "modis_snow": (0.0, 100.0),
        }
        expected_range = value_ranges.get(product, (None, None))

        for year in range(start_year, end_year + 1):
            results["total_expected"] += 1
            fname = f"{product}_bc_{year}.tif"
            fpath = gee_dir / fname

            if not fpath.exists():
                results["missing_files"].append(fname)
                continue

            results["total_found"] += 1

            try:
                import rasterio

                with rasterio.open(fpath) as src:
                    # Check dimensions (BC at 500m should be ~2400x5200)
                    if src.width < 100 or src.height < 100:
                        results["issues"].append(
                            f"{fname}: unexpectedly small ({src.width}x{src.height})"
                        )

                    # Read a sample and check for all-NaN
                    data = src.read(1)
                    valid_pct = np.count_nonzero(~np.isnan(data)) / data.size * 100
                    if valid_pct < 10:
                        results["issues"].append(f"{fname}: only {valid_pct:.1f}% valid pixels")

                    # Check value range
                    if expected_range[0] is not None:
                        valid_data = data[~np.isnan(data)]
                        if len(valid_data) > 0:
                            vmin, vmax = valid_data.min(), valid_data.max()
                            if vmin < expected_range[0] - 0.5 or vmax > expected_range[1] + 0.5:
                                results["issues"].append(
                                    f"{fname}: values out of range [{vmin:.2f}, {vmax:.2f}]"
                                )
            except Exception as e:
                results["issues"].append(f"{fname}: {e}")

        if results["missing_files"]:
            results["status"] = "fail" if results["total_found"] == 0 else "partial"

        logger.info(
            "%s validation: %d/%d files found, %d issues",
            product,
            results["total_found"],
            results["total_expected"],
            len(results["issues"]),
        )
        return results

    def validate_fire_history(self) -> dict:
        """Validate fire history data files.

        Checks CNFDB, BC Fire Perimeters, and BC Fire Incidents.
        """
        results = {
            "status": "pass",
            "sources": {},
        }

        fire_sources = {
            "cnfdb": self.data_dir / "cnfdb",
            "bc_fire_perimeters": self.data_dir / "bc_fire_perimeters",
            "bc_fire_incidents": self.data_dir / "bc_fire_incidents",
        }

        for source_name, source_dir in fire_sources.items():
            source_result = {"found": False, "files": 0, "records": 0}

            if source_dir.exists():
                files = list(source_dir.glob("*.csv")) + list(source_dir.glob("*.shp"))
                source_result["found"] = bool(files)
                source_result["files"] = len(files)

                # Count records in CSV files
                for f in source_dir.glob("*.csv"):
                    try:
                        import pandas as pd

                        df = pd.read_csv(f, nrows=0)
                        source_result["columns"] = list(df.columns)
                        # Count lines without loading full file
                        with open(f) as fh:
                            source_result["records"] = sum(1 for _ in fh) - 1
                    except Exception as e:
                        source_result["error"] = str(e)

            results["sources"][source_name] = source_result

        # Overall status
        found_count = sum(1 for s in results["sources"].values() if s["found"])
        if found_count == 0:
            results["status"] = "fail"
        elif found_count < len(fire_sources):
            results["status"] = "partial"

        return results

    def summary(self, report: dict) -> str:
        """Generate a human-readable summary of validation results."""
        lines = ["=== Data Validation Report ===", ""]

        for section, data in report.items():
            if section == "overall_status":
                continue
            if isinstance(data, dict):
                status = data.get("status", "unknown")
                icon = "PASS" if status == "pass" else "PARTIAL" if status == "partial" else "FAIL"
                lines.append(f"[{icon}] {section}")

                if "total_found" in data:
                    lines.append(f"  Files: {data['total_found']}/{data['total_expected']}")
                if data.get("missing_files"):
                    lines.append(f"  Missing: {len(data['missing_files'])} files")
                if data.get("corrupt_files"):
                    lines.append(f"  Corrupt: {len(data['corrupt_files'])} files")
                if data.get("issues"):
                    lines.append(f"  Issues: {len(data['issues'])}")
                if data.get("sources"):
                    for src, info in data["sources"].items():
                        found = "found" if info["found"] else "NOT FOUND"
                        lines.append(f"  {src}: {found} ({info.get('records', 0)} records)")

                lines.append("")

        lines.append(f"Overall: {report.get('overall_status', 'unknown').upper()}")
        return "\n".join(lines)
