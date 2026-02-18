"""Heatmap Engine - U-Net CNN for spatial fire risk prediction.

Phase 2 model that captures spatial autocorrelation and topographic
fire corridors that the point-based XGBoost model cannot.

Architecture:
  - Encoder: 4 downsampling blocks (conv → BN → ReLU → maxpool)
  - Bottleneck: 2 conv layers at lowest resolution
  - Decoder: 4 upsampling blocks (upsample → concat → conv → BN → ReLU)
  - Output: 1-channel sigmoid (fire probability map)

Input: [batch, 12, H, W] raster covering BC study area
  Channels: temp, rh, wind, soil_moisture, fwi, ndvi, snow,
            elevation, slope, fuel_type, bec_zone, doy_sin

Output: [batch, 1, H, W] fire probability heatmap
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Input channels for the CNN
INPUT_CHANNELS = 12
CHANNEL_NAMES = [
    "temperature_c",
    "rh_pct",
    "wind_kmh",
    "soil_moisture_1",
    "fwi",
    "ndvi",
    "snow_cover",
    "elevation_m",
    "slope_deg",
    "fuel_type_encoded",
    "bec_zone_encoded",
    "doy_sin",
]


class ConvBlock(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FireUNet(nn.Module):
    """U-Net architecture for spatial fire risk prediction.

    Takes a multi-channel raster (12 features) and outputs a
    single-channel fire probability heatmap.
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        base_filters: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout)

        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_filters * 16, base_filters * 8)

        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters)

        # Output
        self.output_conv = nn.Conv2d(base_filters, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.dropout(self.pool(e2)))
        e4 = self.enc4(self.dropout(self.pool(e3)))

        # Bottleneck
        b = self.bottleneck(self.dropout(self.pool(e4)))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self._pad_and_cat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._pad_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_and_cat(d1, e1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.output_conv(d1))

    @staticmethod
    def _pad_and_cat(upsampled, skip):
        """Pad upsampled tensor to match skip connection dimensions, then concat."""
        diff_h = skip.size(2) - upsampled.size(2)
        diff_w = skip.size(3) - upsampled.size(3)
        upsampled = F.pad(
            upsampled,
            [
                diff_w // 2,
                diff_w - diff_w // 2,
                diff_h // 2,
                diff_h - diff_h // 2,
            ],
        )
        return torch.cat([upsampled, skip], dim=1)


class HeatmapTrainer:
    """Training pipeline for the U-Net heatmap model."""

    def __init__(
        self,
        model: FireUNet | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str | None = None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = (model or FireUNet()).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

    def train_epoch(self, dataloader) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            inputs = batch["features"].to(self.device)
            targets = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Focal loss for class imbalance
            loss = self._focal_loss(outputs, targets, alpha=0.75, gamma=2.0)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def validate(self, dataloader) -> dict:
        """Validate model. Returns dict with loss and metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["features"].to(self.device)
                targets = batch["labels"].to(self.device)

                outputs = self.model(inputs)
                loss = self._focal_loss(outputs, targets)

                total_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                n_batches += 1

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        avg_loss = total_loss / max(n_batches, 1)
        self.scheduler.step(avg_loss)

        # Pixel-level metrics
        pred_binary = (preds > 0.5).astype(np.float32)
        precision = np.sum(pred_binary * targets) / max(np.sum(pred_binary), 1)
        recall = np.sum(pred_binary * targets) / max(np.sum(targets), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "loss": avg_loss,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def train(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 100,
        output_path: Path | None = None,
    ) -> dict:
        """Full training loop with early stopping."""
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_f1"].append(val_metrics["f1"])

            logger.info(
                "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, val_f1=%.4f",
                epoch + 1,
                n_epochs,
                train_loss,
                val_metrics["loss"],
                val_metrics["f1"],
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                if output_path:
                    self.save(output_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        return history

    def predict(self, input_raster: np.ndarray) -> np.ndarray:
        """Run inference on a single raster.

        Args:
            input_raster: array of shape [C, H, W] with C=12 channels

        Returns:
            array of shape [H, W] with fire probabilities
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(input_raster).float().unsqueeze(0).to(self.device)
            output = self.model(x)
            return output.squeeze().cpu().numpy()

    def save(self, path: Path):
        """Save model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved to %s", path)

    def load(self, path: Path):
        """Load model weights."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        logger.info("Model loaded from %s", path)

    @staticmethod
    def _focal_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Focal loss for handling class imbalance in pixel classification."""
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, alpha, 1 - alpha)
        focal_weight = alpha_t * (1 - pt) ** gamma
        return (focal_weight * bce).mean()


def build_raster_from_grid(
    predictions: dict,
    grid_cells: dict,
    feature_key: str,
    h: int = 180,
    w: int = 260,
    bbox: tuple = (48.3, -139.06, 60.0, -114.03),
) -> np.ndarray:
    """Convert grid-cell predictions to a 2D raster.

    Args:
        predictions: dict[cell_id -> prediction_dict]
        grid_cells: dict[cell_id -> {lat, lon, ...}]
        feature_key: key to extract from predictions
        h, w: output raster dimensions
        bbox: (south, west, north, east)

    Returns:
        2D array of shape [h, w]
    """
    south, west, north, east = bbox
    raster = np.full((h, w), np.nan, dtype=np.float32)

    lat_step = (north - south) / h
    lon_step = (east - west) / w

    for cell_id, pred in predictions.items():
        cell = grid_cells.get(cell_id, {})
        lat = cell.get("lat")
        lon = cell.get("lon")
        if lat is None or lon is None:
            continue

        row = int((north - lat) / lat_step)
        col = int((lon - west) / lon_step)

        if 0 <= row < h and 0 <= col < w:
            val = pred.get(feature_key) if isinstance(pred, dict) else pred
            if val is not None:
                raster[row, col] = float(val)

    return raster
