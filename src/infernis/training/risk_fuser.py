"""Risk Fuser - combines XGBoost point predictions with CNN spatial predictions.

Implements per-BEC-zone calibration and weighted ensemble fusion to produce
the final fire risk score for each grid cell.

Fusion strategy:
  - XGBoost: strong on local weather/fuel conditions (per-cell)
  - U-Net CNN: captures spatial patterns (fire corridors, terrain)
  - Weights are learned per BEC zone via logistic regression on held-out data
  - Final output: calibrated probability [0, 1] per cell
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default fusion weights (prior to calibration training)
DEFAULT_XGBOOST_WEIGHT = 0.65
DEFAULT_CNN_WEIGHT = 0.35

# All 14 BC BEC zones
BEC_ZONES = [
    "AT",
    "BG",
    "BWBS",
    "CDF",
    "CWH",
    "ESSF",
    "ICH",
    "IDF",
    "MH",
    "MS",
    "PP",
    "SBPS",
    "SBS",
    "SWB",
]


class RiskFuser:
    """Fuses XGBoost and CNN predictions with per-zone calibration."""

    def __init__(self, weights_path: Path | None = None):
        # Per-zone weights: zone -> {"xgb_weight": float, "cnn_weight": float, "bias": float}
        # Defaults are identity-transform for XGB-only mode:
        #   sigmoid(1.0 * logit(score) + 0) == score
        self.zone_params: dict[str, dict[str, float]] = {}

        if weights_path and Path(weights_path).exists():
            self.load_weights(weights_path)
        else:
            for zone in BEC_ZONES:
                self.zone_params[zone] = {
                    "xgb_weight": 1.0,
                    "cnn_weight": 0.0,
                    "bias": 0.0,
                }

    def fuse(
        self,
        xgb_scores: np.ndarray,
        cnn_scores: np.ndarray,
        bec_zones: np.ndarray,
    ) -> np.ndarray:
        """Fuse XGBoost and CNN scores using per-zone calibrated weights.

        Calibration weights are learned in logit space, so scores are
        transformed to logits before applying the linear model, then
        sigmoid maps back to probabilities.

        Args:
            xgb_scores: array [n_cells] of XGBoost fire probabilities
            cnn_scores: array [n_cells] of CNN fire probabilities
            bec_zones: array [n_cells] of BEC zone strings

        Returns:
            array [n_cells] of fused fire probabilities
        """
        logit_xgb = self._logit(xgb_scores)
        logit_cnn = self._logit(cnn_scores)

        # Vectorized per-zone fusion: build weight/bias arrays by zone
        n = len(xgb_scores)
        w_xgb = np.full(n, DEFAULT_XGBOOST_WEIGHT)
        w_cnn = np.full(n, DEFAULT_CNN_WEIGHT)
        bias = np.zeros(n)

        for zone, params in self.zone_params.items():
            mask = bec_zones == zone
            if mask.any():
                w_xgb[mask] = params.get("xgb_weight", DEFAULT_XGBOOST_WEIGHT)
                w_cnn[mask] = params.get("cnn_weight", DEFAULT_CNN_WEIGHT)
                bias[mask] = params.get("bias", 0.0)

        raw = w_xgb * logit_xgb + w_cnn * logit_cnn + bias
        fused = 1.0 / (1.0 + np.exp(-raw))
        return np.clip(fused, 0.0, 1.0)

    def fuse_xgb_only(
        self,
        xgb_scores: np.ndarray,
        bec_zones: np.ndarray,
    ) -> np.ndarray:
        """Fuse with XGBoost only (Phase 1, before CNN is trained).

        Applies per-zone calibration in logit space without CNN scores.
        """
        logit_xgb = self._logit(xgb_scores)

        # Vectorized per-zone calibration
        n = len(xgb_scores)
        w_xgb = np.ones(n)
        bias = np.zeros(n)

        for zone, params in self.zone_params.items():
            mask = bec_zones == zone
            if mask.any():
                w_xgb[mask] = params.get("xgb_weight", 1.0)
                bias[mask] = params.get("bias", 0.0)

        raw = w_xgb * logit_xgb + bias
        fused = 1.0 / (1.0 + np.exp(-raw))
        return np.clip(fused, 0.0, 1.0)

    def calibrate(
        self,
        xgb_scores: np.ndarray,
        cnn_scores: np.ndarray | None,
        true_labels: np.ndarray,
        bec_zones: np.ndarray,
    ):
        """Learn per-zone fusion weights from labeled calibration data.

        Uses logistic regression per zone to learn optimal weights.
        """
        from sklearn.linear_model import LogisticRegression

        for zone in BEC_ZONES:
            mask = bec_zones == zone
            if mask.sum() < 10:
                logger.warning("Zone %s has only %d samples, using defaults", zone, mask.sum())
                continue

            zone_xgb = xgb_scores[mask]
            zone_y = true_labels[mask]

            if cnn_scores is not None:
                zone_cnn = cnn_scores[mask]
                X = np.column_stack(
                    [
                        self._logit(zone_xgb),
                        self._logit(zone_cnn),
                    ]
                )
            else:
                X = self._logit(zone_xgb).reshape(-1, 1)

            # Skip zones with no positive samples
            if zone_y.sum() == 0 or zone_y.sum() == len(zone_y):
                logger.warning("Zone %s has no class variation, using defaults", zone)
                continue

            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(X, zone_y)

            if cnn_scores is not None:
                self.zone_params[zone] = {
                    "xgb_weight": float(lr.coef_[0][0]),
                    "cnn_weight": float(lr.coef_[0][1]),
                    "bias": float(lr.intercept_[0]),
                }
            else:
                self.zone_params[zone] = {
                    "xgb_weight": float(lr.coef_[0][0]),
                    "cnn_weight": 0.0,
                    "bias": float(lr.intercept_[0]),
                }

            logger.info(
                "Zone %s calibrated: xgb=%.3f, cnn=%.3f, bias=%.3f",
                zone,
                self.zone_params[zone]["xgb_weight"],
                self.zone_params[zone]["cnn_weight"],
                self.zone_params[zone]["bias"],
            )

    def save_weights(self, path: Path):
        """Save calibrated weights to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.zone_params, f, indent=2)
        logger.info("Fuser weights saved to %s", path)

    def load_weights(self, path: Path):
        """Load calibrated weights from JSON."""
        with open(path) as f:
            self.zone_params = json.load(f)
        logger.info("Fuser weights loaded: %d zones", len(self.zone_params))

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Safe logit transform: log(p / (1 - p))."""
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))
