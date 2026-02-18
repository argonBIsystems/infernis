"""CNN training data preparation for the heatmap engine.

Converts grid-cell tabular features and fire history into spatial rasters
for U-Net training. Each training sample is one day's complete BC raster.

Output structure (data/processed/heatmap/):
  features/YYYY-MM-DD.npy  -- [12, 256, 512] float16
  labels/YYYY-MM-DD.npy    -- [1, 256, 512] uint8
  cell_mapping.npz         -- pixel positions for each cell
  channel_stats.json       -- per-channel mean/std for normalization
  land_mask.npy            -- [256, 512] binary mask

Usage:
  # 1. Prepare rasters (one-time)
  prepare_heatmap_data(Path("path/to/project"))

  # 2. Create dataloaders for training
  train_loader, val_loader, test_loader = get_dataloaders(
      Path("path/to/project/data/processed/heatmap")
  )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from infernis.training.heatmap_model import CHANNEL_NAMES, INPUT_CHANNELS

logger = logging.getLogger(__name__)

# Raster dimensions — divisible by 2^4 = 16 for U-Net with 4 pooling layers
RASTER_H = 256
RASTER_W = 512

# Bounding box (WGS84) covering the BC grid with small padding
LAT_MIN = 48.30
LAT_MAX = 60.60
LON_MIN = -139.10
LON_MAX = -114.00

LAT_STEP = (LAT_MAX - LAT_MIN) / RASTER_H  # ~0.048 deg
LON_STEP = (LON_MAX - LON_MIN) / RASTER_W  # ~0.049 deg

# Mapping from feature parquet column names to CNN channel indices.
# Indices match CHANNEL_NAMES in heatmap_model.py.
PARQUET_TO_CHANNEL: dict[str, int] = {
    "temperature_c": 0,
    "rh_pct": 1,
    "wind_kmh": 2,
    "soil_moisture_1": 3,
    "fwi": 4,
    "ndvi": 5,
    "snow_cover": 6,
    "elevation_m": 7,
    "slope_deg": 8,
    # Channel 9  (fuel_type_encoded) — Tier 2 feature, zeros until available
    # Channel 10 (bec_zone_encoded)  — Tier 2 feature, zeros until available
    "doy_sin": 11,
}

# Default year splits
TRAIN_YEARS = list(range(2015, 2023))  # 2015-2022
VAL_YEARS = [2023]
TEST_YEARS = [2024]


# ---------------------------------------------------------------------------
# Cell-to-pixel mapping
# ---------------------------------------------------------------------------


def build_cell_mapping(grid_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map grid cells to raster pixel positions.

    Returns:
        cell_ids: [N] string array of cell IDs
        rows: [N] int array of pixel row indices (0 = north)
        cols: [N] int array of pixel col indices (0 = west)
    """
    grid = pd.read_parquet(grid_path)
    cell_ids = grid["cell_id"].values
    lats = grid["lat"].values
    lons = grid["lon"].values

    rows = np.clip(((LAT_MAX - lats) / LAT_STEP).astype(int), 0, RASTER_H - 1)
    cols = np.clip(((lons - LON_MIN) / LON_STEP).astype(int), 0, RASTER_W - 1)

    return cell_ids, rows, cols


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


def rasterize_features(
    day_df: pd.DataFrame,
    cell_id_to_idx: dict[str, int],
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Convert one day's tabular features to a [C, H, W] raster.

    Args:
        day_df: feature rows for a single date
        cell_id_to_idx: maps cell_id string to index in rows/cols arrays
        rows: precomputed row indices for each cell
        cols: precomputed col indices for each cell

    Returns:
        [INPUT_CHANNELS, RASTER_H, RASTER_W] float32 array
    """
    raster = np.zeros((INPUT_CHANNELS, RASTER_H, RASTER_W), dtype=np.float32)

    # Map cell_ids to their precomputed pixel indices (vectorised)
    idx_series = day_df["cell_id"].map(cell_id_to_idx)
    valid = idx_series.notna()
    indices = idx_series[valid].astype(int).values
    r = rows[indices]
    c = cols[indices]

    # Track counts per pixel for averaging when multiple cells map to same pixel
    count = np.zeros((RASTER_H, RASTER_W), dtype=np.float32)
    np.add.at(count, (r, c), 1.0)
    needs_avg = count.max() > 1  # Only average if there are overlapping cells

    for feat_name, ch in PARQUET_TO_CHANNEL.items():
        if feat_name not in day_df.columns:
            continue
        values = day_df.loc[valid, feat_name].values.astype(np.float32)
        # Filter NaN/Inf and DEM nodata sentinel (-32767)
        usable = np.isfinite(values) & (values > -9999)
        if needs_avg:
            # Accumulate and average for multi-cell-per-pixel (1km grids)
            np.add.at(raster[ch], (r[usable], c[usable]), values[usable])
        else:
            raster[ch, r[usable], c[usable]] = values[usable]

    if needs_avg:
        # Average where multiple cells contributed
        multi = count > 1
        for ch in range(INPUT_CHANNELS):
            raster[ch][multi] /= count[multi]

    return raster


def rasterize_fire_mask(
    fire_cell_ids: np.ndarray,
    cell_id_to_idx: dict[str, int],
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Create binary [1, H, W] fire mask from cell IDs that had fires."""
    mask = np.zeros((1, RASTER_H, RASTER_W), dtype=np.float32)
    if len(fire_cell_ids) == 0:
        return mask

    indices = [cell_id_to_idx[cid] for cid in fire_cell_ids if cid in cell_id_to_idx]
    if indices:
        idx_arr = np.array(indices)
        mask[0, rows[idx_arr], cols[idx_arr]] = 1.0

    return mask


# ---------------------------------------------------------------------------
# Main preparation pipeline
# ---------------------------------------------------------------------------


def prepare_heatmap_data(
    base_dir: Path,
    start_year: int = 2015,
    end_year: int = 2024,
) -> Path:
    """Rasterize all features and fire labels into per-day .npy files.

    Args:
        base_dir: project root containing data/
        start_year: first year to process (inclusive)
        end_year: last year to process (inclusive)

    Returns:
        Path to output directory (data/processed/heatmap)
    """
    from infernis.training.feature_builder import FeatureBuilder

    grid_path = base_dir / "data" / "processed" / "bc_grid.parquet"
    features_dir = base_dir / "data" / "processed" / "features"
    output_dir = base_dir / "data" / "processed" / "heatmap"

    (output_dir / "features").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    # --- Cell-to-pixel mapping ---
    logger.info("Building cell-to-pixel mapping...")
    cell_ids, rows, cols = build_cell_mapping(grid_path)
    cell_id_to_idx: dict[str, int] = {cid: i for i, cid in enumerate(cell_ids)}

    np.savez(
        output_dir / "cell_mapping.npz",
        cell_ids=cell_ids,
        rows=rows,
        cols=cols,
    )

    land_mask = np.zeros((RASTER_H, RASTER_W), dtype=np.float32)
    land_mask[rows, cols] = 1.0
    np.save(output_dir / "land_mask.npy", land_mask)
    logger.info(
        "Land mask: %d active pixels / %d total (%.1f%%)",
        int(land_mask.sum()),
        RASTER_H * RASTER_W,
        100.0 * land_mask.sum() / (RASTER_H * RASTER_W),
    )

    # --- Fire history ---
    logger.info("Loading fire history...")
    fb = FeatureBuilder(
        processed_dir=base_dir / "data" / "processed",
        raw_dir=base_dir / "data" / "raw",
    )
    fires = fb.load_fire_history()
    if not fires.empty:
        fires["date"] = pd.to_datetime(fires["date"], errors="coerce")
        fires = fires.dropna(subset=["date"])
        fires = fires[(fires["date"].dt.year >= start_year) & (fires["date"].dt.year <= end_year)]

    grid_df = pd.read_parquet(grid_path)
    fire_cells = fb.assign_fires_to_grid(fires, grid_df)

    # Build date -> fire cell_ids lookup
    fire_by_date: dict[str, np.ndarray] = {}
    if not fire_cells.empty:
        fire_cells["date"] = pd.to_datetime(fire_cells["date"])
        fire_cells["date_str"] = fire_cells["date"].dt.strftime("%Y-%m-%d")
        for date_str, grp in fire_cells.groupby("date_str"):
            fire_by_date[date_str] = grp["cell_id"].values

    logger.info("Fire data: %d unique days with fires", len(fire_by_date))

    # --- Rasterize each monthly feature parquet ---
    total_days = 0
    total_fire_days = 0
    total_fire_pixels = 0

    parquet_files = sorted(features_dir.glob("features_*.parquet"))
    for pf_idx, pf in enumerate(parquet_files):
        parts = pf.stem.split("_")
        year = int(parts[1])
        if year < start_year or year > end_year:
            continue

        logger.info("[%d/%d] Processing %s...", pf_idx + 1, len(parquet_files), pf.name)
        feat_df = pd.read_parquet(pf)
        # Normalise date column to YYYY-MM-DD strings
        feat_df["date_str"] = pd.to_datetime(feat_df["date"]).dt.strftime("%Y-%m-%d")

        # Aggregate sub-daily timesteps to daily means per cell
        agg_cols = [c for c in PARQUET_TO_CHANNEL if c in feat_df.columns]
        feat_df = feat_df.groupby(["cell_id", "date_str"], as_index=False)[agg_cols].mean()

        for date_str, day_df in feat_df.groupby("date_str"):
            # Feature raster
            feature_raster = rasterize_features(day_df, cell_id_to_idx, rows, cols)
            np.save(
                output_dir / "features" / f"{date_str}.npy",
                feature_raster.astype(np.float16),
            )

            # Fire label raster
            fire_cids = fire_by_date.get(date_str, np.array([], dtype=str))
            label_raster = rasterize_fire_mask(fire_cids, cell_id_to_idx, rows, cols)
            np.save(
                output_dir / "labels" / f"{date_str}.npy",
                label_raster.astype(np.uint8),
            )

            total_days += 1
            n_fire_px = int(label_raster.sum())
            if n_fire_px > 0:
                total_fire_days += 1
                total_fire_pixels += n_fire_px

        del feat_df

    logger.info(
        "Rasterized %d days (%d with fires, %d total fire pixels)",
        total_days,
        total_fire_days,
        total_fire_pixels,
    )

    # --- Channel normalisation stats (training years only) ---
    stats = compute_channel_stats(output_dir, train_end_year=2022)
    with open(output_dir / "channel_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Heatmap data preparation complete: %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------


def compute_channel_stats(
    heatmap_dir: Path,
    train_end_year: int = 2022,
) -> dict:
    """Compute per-channel mean and std from training data only.

    Uses a two-pass approach over land pixels to compute accurate statistics
    without loading all data into memory.
    """
    logger.info("Computing channel statistics (years <= %d)...", train_end_year)
    features_dir = heatmap_dir / "features"

    land_mask = np.load(heatmap_dir / "land_mask.npy")
    land_pixels = land_mask > 0
    n_land = int(land_pixels.sum())

    ch_sum = np.zeros(INPUT_CHANNELS, dtype=np.float64)
    ch_sq_sum = np.zeros(INPUT_CHANNELS, dtype=np.float64)
    n_files = 0

    npy_files = sorted(features_dir.glob("*.npy"))
    for npy_file in npy_files:
        year = int(npy_file.stem[:4])
        if year > train_end_year:
            continue

        raster = np.load(npy_file).astype(np.float32)  # [C, H, W]
        for ch in range(INPUT_CHANNELS):
            values = raster[ch][land_pixels].astype(np.float64)
            ch_sum[ch] += values.sum()
            ch_sq_sum[ch] += (values**2).sum()

        n_files += 1

    total_n = n_files * n_land
    if total_n > 0:
        mean = ch_sum / total_n
        variance = ch_sq_sum / total_n - mean**2
        std = np.sqrt(np.maximum(variance, 0))
    else:
        mean = np.zeros(INPUT_CHANNELS, dtype=np.float64)
        std = np.ones(INPUT_CHANNELS, dtype=np.float64)

    # Floor std to avoid division by zero (unused channels)
    std[std < 1e-6] = 1.0

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "channel_names": CHANNEL_NAMES,
        "n_files": n_files,
        "n_land_pixels": n_land,
    }
    logger.info("Channel means: %s", [f"{m:.3f}" for m in mean])
    logger.info("Channel stds:  %s", [f"{s:.3f}" for s in std])
    return stats


# ---------------------------------------------------------------------------
# PyTorch Dataset & DataLoaders
# ---------------------------------------------------------------------------


class FireRasterDataset(Dataset):
    """PyTorch Dataset yielding daily rasters for U-Net training.

    Each sample:
      features: [12, 256, 512] float32 tensor (normalised)
      labels:   [1, 256, 512]  float32 tensor (binary fire mask)
    """

    def __init__(
        self,
        heatmap_dir: Path,
        years: list[int],
        normalize: bool = True,
    ):
        self.heatmap_dir = Path(heatmap_dir)
        self.features_dir = self.heatmap_dir / "features"
        self.labels_dir = self.heatmap_dir / "labels"

        # Collect all date stems for the requested years
        self.file_stems = sorted(
            [f.stem for f in self.features_dir.glob("*.npy") if int(f.stem[:4]) in years]
        )

        # Channel normalisation
        self.normalize = normalize
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        if normalize:
            stats_path = self.heatmap_dir / "channel_stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    stats = json.load(f)
                self.mean = np.array(stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
                self.std = np.array(stats["std"], dtype=np.float32).reshape(-1, 1, 1)
            else:
                logger.warning("No channel_stats.json found — skipping normalisation")
                self.normalize = False

        logger.info("FireRasterDataset: %d samples from years %s", len(self), years)

    def __len__(self) -> int:
        return len(self.file_stems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        stem = self.file_stems[idx]

        features = np.load(self.features_dir / f"{stem}.npy").astype(np.float32)
        labels = np.load(self.labels_dir / f"{stem}.npy").astype(np.float32)

        if self.normalize and self.mean is not None and self.std is not None:
            features = (features - self.mean) / self.std

        return {
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
        }


def get_dataloaders(
    heatmap_dir: Path,
    batch_size: int = 4,
    num_workers: int = 2,
    train_years: list[int] | None = None,
    val_years: list[int] | None = None,
    test_years: list[int] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / validation / test DataLoaders.

    Default split: 2015-2022 train, 2023 val, 2024 test.
    """
    train_years = train_years or TRAIN_YEARS
    val_years = val_years or VAL_YEARS
    test_years = test_years or TEST_YEARS

    heatmap_dir = Path(heatmap_dir)

    train_ds = FireRasterDataset(heatmap_dir, train_years)
    val_ds = FireRasterDataset(heatmap_dir, val_years)
    test_ds = FireRasterDataset(heatmap_dir, test_years)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
