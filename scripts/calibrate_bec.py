#!/usr/bin/env python3
"""Compute per-BEC-zone calibration coefficients for the Risk Fuser.

Uses 2023 validation data to learn bias corrections for XGBoost predictions
in each biogeoclimatic zone.

Output: models/bec_calibration.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# Ensure project root on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from infernis.pipelines.data_processor import FEATURE_NAMES
from infernis.training.risk_fuser import RiskFuser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("calibrate_bec")


def assign_bec_zone(lat: float, lon: float, elev: float) -> str:
    """Assign BEC zone using the heuristic from initializer.py."""
    if lat > 57:
        return "BWBS"
    if lat > 55:
        return "SWB" if elev > 1200 else "BWBS"
    if elev > 1800:
        return "AT"
    if elev > 1400:
        return "ESSF"
    if elev > 1000:
        return "MH" if lon < -124 else "MS"
    if lat < 49.5 and lon > -120:
        return "PP" if elev < 600 else "IDF"
    if lon < -125:
        return "CWH" if elev < 800 else "MH"
    if lon < -122:
        return "CDF" if lat < 49.5 else "ICH"
    if lat > 52:
        return "SBS"
    return "IDF" if elev < 1200 else "ESSF"


def main():
    data_path = project_root / "data" / "processed" / "training_data.parquet"
    model_path = project_root / "models" / "xgboost_fire.json"
    output_path = project_root / "models" / "bec_calibration.json"

    # --- Load data ---
    logger.info("Loading training data from %s", data_path)
    df = pd.read_parquet(data_path)
    df["year"] = pd.to_datetime(df["date"]).dt.year

    # Assign BEC zones
    logger.info("Assigning BEC zones to %d samples...", len(df))
    # Replace DEM nodata with 0 before BEC assignment
    elev = df["elevation_m"].values.copy()
    elev[elev < -9999] = 0
    df["bec_zone"] = [
        assign_bec_zone(lat, lon, e)
        for lat, lon, e in zip(df["lat"].values, df["lon"].values, elev)
    ]
    logger.info("BEC zone distribution:")
    for zone, count in df["bec_zone"].value_counts().sort_index().items():
        logger.info("  %s: %d samples", zone, count)

    # --- Load XGBoost model ---
    logger.info("Loading XGBoost model from %s", model_path)
    model = xgb.Booster()
    model.load_model(str(model_path))

    # --- Prepare features ---
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    X_all = df[available_features].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=1e6, neginf=-1e6)
    y_all = df["fire"].values.astype(np.int32)
    bec_all = df["bec_zone"].values

    # --- Generate predictions ---
    logger.info("Generating XGBoost predictions...")
    dmat = xgb.DMatrix(X_all, feature_names=available_features)
    preds = model.predict(dmat)
    logger.info(
        "Predictions: mean=%.4f, min=%.4f, max=%.4f",
        preds.mean(), preds.min(), preds.max(),
    )

    # --- Calibrate using 2023 validation data ---
    val_mask = df["year"].values == 2023
    logger.info(
        "Calibrating on %d validation samples (2023): %d fires, %d non-fires",
        val_mask.sum(),
        y_all[val_mask].sum(),
        val_mask.sum() - y_all[val_mask].sum(),
    )

    fuser = RiskFuser()
    fuser.calibrate(
        xgb_scores=preds[val_mask],
        cnn_scores=None,
        true_labels=y_all[val_mask],
        bec_zones=bec_all[val_mask],
    )

    # --- Evaluate calibration on 2024 test data ---
    test_mask = df["year"].values == 2024
    logger.info(
        "Evaluating on %d test samples (2024): %d fires, %d non-fires",
        test_mask.sum(),
        y_all[test_mask].sum(),
        test_mask.sum() - y_all[test_mask].sum(),
    )

    # Compare uncalibrated vs calibrated predictions
    raw_preds_test = preds[test_mask]
    cal_preds_test = fuser.fuse_xgb_only(raw_preds_test, bec_all[test_mask])
    y_test = y_all[test_mask]

    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    logger.info("--- 2024 Test Results ---")
    logger.info(
        "Uncalibrated: AUC=%.4f, Brier=%.4f, LogLoss=%.4f",
        roc_auc_score(y_test, raw_preds_test),
        brier_score_loss(y_test, raw_preds_test),
        log_loss(y_test, np.clip(raw_preds_test, 1e-7, 1 - 1e-7)),
    )
    logger.info(
        "Calibrated:   AUC=%.4f, Brier=%.4f, LogLoss=%.4f",
        roc_auc_score(y_test, cal_preds_test),
        brier_score_loss(y_test, cal_preds_test),
        log_loss(y_test, np.clip(cal_preds_test, 1e-7, 1 - 1e-7)),
    )

    # Per-zone breakdown
    logger.info("--- Per-Zone Breakdown (2024 test) ---")
    for zone in sorted(set(bec_all[test_mask])):
        zm = (bec_all[test_mask] == zone)
        if zm.sum() < 10:
            continue
        zone_y = y_test[zm]
        zone_raw = raw_preds_test[zm]
        zone_cal = cal_preds_test[zm]
        fire_rate = zone_y.mean()
        try:
            auc = roc_auc_score(zone_y, zone_cal) if zone_y.sum() > 0 and zone_y.sum() < len(zone_y) else float("nan")
        except ValueError:
            auc = float("nan")
        logger.info(
            "  %s: n=%d, fire_rate=%.4f, pred_mean_raw=%.4f, pred_mean_cal=%.4f, AUC=%.4f",
            zone, zm.sum(), fire_rate, zone_raw.mean(), zone_cal.mean(), auc,
        )

    # --- Save calibration ---
    fuser.save_weights(output_path)
    logger.info("Calibration saved to %s", output_path)

    # Also save a summary
    summary = {
        "calibration_year": 2023,
        "test_year": 2024,
        "n_val_samples": int(val_mask.sum()),
        "n_test_samples": int(test_mask.sum()),
        "zone_params": fuser.zone_params,
    }
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
