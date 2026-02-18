#!/usr/bin/env python3
"""Historical backtesting CLI for INFERNIS fire prediction models.

Subcommands:
    backtest     Walk-forward temporal CV on a single model.
    compare      Compare two models (e.g. 5 km vs 1 km).
    zone-report  Per-BEC-zone accuracy breakdown.

Usage examples:
    python scripts/backtest.py backtest \
        --data data/processed/training_data.parquet \
        --model models/fire_core_v1.json \
        --output reports/backtest_5km.json

    python scripts/backtest.py compare \
        --model-a models/fire_core_v1.json \
        --model-b models/fire_core_1km_v1.json \
        --data data/processed/training_data.parquet \
        --output reports/comparison_5km_vs_1km.json

    python scripts/backtest.py zone-report \
        --data data/processed/training_data.parquet \
        --model models/fire_core_v1.json \
        --output reports/zone_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infernis.pipelines.data_processor import FEATURE_NAMES
from infernis.training.backtester import HistoricalBacktester
from infernis.training.evaluator import ModelEvaluator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# BEC zone heuristic (same as calibrate_bec.py)
# ---------------------------------------------------------------------------
def assign_bec_zone(lat: float, lon: float, elev: float) -> str:
    """Assign a BEC zone using the elevation / lat-lon heuristic."""
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(data_path: Path) -> pd.DataFrame:
    """Load a training parquet and return the full DataFrame."""
    logger.info("Loading data from %s", data_path)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def _extract_bec_zones(df: pd.DataFrame) -> np.ndarray:
    """Return a BEC zone array, computing from lat/lon/elev if needed."""
    if "bec_zone" in df.columns:
        logger.info("Using existing bec_zone column")
        return df["bec_zone"].values

    logger.info("Assigning BEC zones from lat/lon/elevation_m ...")
    elev = df["elevation_m"].values.copy() if "elevation_m" in df.columns else np.zeros(len(df))
    elev = np.where(elev < -9999, 0.0, elev)
    zones = np.array([
        assign_bec_zone(lat, lon, e)
        for lat, lon, e in zip(df["lat"].values, df["lon"].values, elev)
    ])
    return zones


def _extract_features(df: pd.DataFrame) -> tuple:
    """Return (X, available_features) from a DataFrame."""
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    logger.info("Using %d / %d features", len(available_features), len(FEATURE_NAMES))
    X = df[available_features].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return X, available_features


def _load_model(model_path: Path) -> xgb.Booster:
    """Load an XGBoost model from disk."""
    logger.info("Loading model from %s", model_path)
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        sys.exit(1)
    model = xgb.Booster()
    model.load_model(str(model_path))
    return model


def _predict(model: xgb.Booster, X: np.ndarray, feature_names: list) -> np.ndarray:
    """Run inference and return predicted probabilities."""
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    preds = model.predict(dmat)
    logger.info(
        "Predictions: n=%d, mean=%.4f, min=%.4f, max=%.4f",
        len(preds), preds.mean(), preds.min(), preds.max(),
    )
    return preds


def _save_report(report: dict, output_path: Path) -> None:
    """Write a JSON report to disk, creating parent dirs as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", output_path)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_backtest(args: argparse.Namespace) -> None:
    """Walk-forward temporal cross-validation on a single model."""
    data_path = Path(args.data)
    model_path = Path(args.model)
    output_path = Path(args.output)

    # Load data
    df = _load_data(data_path)
    X, available_features = _extract_features(df)
    y = df["fire"].values.astype(np.int32)
    years = pd.to_datetime(df["date"]).dt.year.values
    bec_zones = _extract_bec_zones(df)

    unique_years = sorted(set(years))
    logger.info("Year range: %d - %d (%d unique years)", unique_years[0], unique_years[-1], len(unique_years))
    logger.info("Samples: %d total, %d fires (%.4f prevalence)", len(y), y.sum(), y.mean())

    # Run temporal CV
    backtester = HistoricalBacktester()
    cv_results = backtester.temporal_cv(X, y, years, bec_zones)

    # Generate evaluation summary
    evaluator = ModelEvaluator()
    report = {
        "command": "backtest",
        "data_path": str(data_path),
        "model_path": str(model_path),
        "n_samples": int(len(y)),
        "n_fires": int(y.sum()),
        "prevalence": round(float(y.mean()), 6),
        "year_range": [int(unique_years[0]), int(unique_years[-1])],
        "cv_results": cv_results,
    }

    # Log key metrics
    if isinstance(cv_results, dict):
        for key in ("auc_roc_mean", "auc_roc_std", "pr_auc_mean", "brier_mean"):
            if key in cv_results:
                logger.info("  %s: %.4f", key, cv_results[key])
        if "fold_metrics" in cv_results:
            for i, fold in enumerate(cv_results["fold_metrics"]):
                auc = fold.get("auc_roc", float("nan"))
                logger.info("  Fold %d: AUC=%.4f", i, auc)

    _save_report(report, output_path)
    logger.info("Backtest complete.")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two models on the same (or separate) datasets."""
    model_a_path = Path(args.model_a)
    model_b_path = Path(args.model_b)
    data_path = Path(args.data)
    data_a_path = Path(args.data_a) if args.data_a else data_path
    data_b_path = Path(args.data_b) if args.data_b else data_path
    output_path = Path(args.output)

    # Load data for model A
    df_a = _load_data(data_a_path)
    X_a, features_a = _extract_features(df_a)
    y_a = df_a["fire"].values.astype(np.int32)
    years_a = pd.to_datetime(df_a["date"]).dt.year.values

    # Load data for model B (may be the same DataFrame)
    if str(data_a_path) == str(data_b_path):
        df_b, X_b, features_b, y_b, years_b = df_a, X_a, features_a, y_a, years_a
    else:
        df_b = _load_data(data_b_path)
        X_b, features_b = _extract_features(df_b)
        y_b = df_b["fire"].values.astype(np.int32)
        years_b = pd.to_datetime(df_b["date"]).dt.year.values

    # Load models
    model_a = _load_model(model_a_path)
    model_b = _load_model(model_b_path)

    # Generate predictions on full datasets
    logger.info("Generating predictions for model A ...")
    preds_a = _predict(model_a, X_a, features_a)
    logger.info("Generating predictions for model B ...")
    preds_b = _predict(model_b, X_b, features_b)

    # Use the latest year as the test set
    latest_year_a = int(years_a.max())
    latest_year_b = int(years_b.max())
    test_year = min(latest_year_a, latest_year_b)
    logger.info("Using year %d as test set", test_year)

    test_mask_a = years_a == test_year
    test_mask_b = years_b == test_year

    test_preds_a = preds_a[test_mask_a]
    test_y_a = y_a[test_mask_a]
    test_preds_b = preds_b[test_mask_b]
    test_y_b = y_b[test_mask_b]

    logger.info(
        "Test set A: %d samples, %d fires | Test set B: %d samples, %d fires",
        len(test_y_a), test_y_a.sum(), len(test_y_b), test_y_b.sum(),
    )

    # Evaluate each model on its test split
    evaluator = ModelEvaluator()
    metrics_a = evaluator.evaluate(test_y_a, test_preds_a)
    metrics_b = evaluator.evaluate(test_y_b, test_preds_b)

    logger.info("--- Model A (%s) ---", model_a_path.name)
    for key in ("auc_roc", "pr_auc", "brier_score", "ece", "best_f1"):
        logger.info("  %s: %.4f", key, metrics_a.get(key, float("nan")))

    logger.info("--- Model B (%s) ---", model_b_path.name)
    for key in ("auc_roc", "pr_auc", "brier_score", "ece", "best_f1"):
        logger.info("  %s: %.4f", key, metrics_b.get(key, float("nan")))

    # Run backtester comparison (requires same y_true for both models)
    backtester = HistoricalBacktester()
    if str(data_a_path) == str(data_b_path):
        bec_zones = _extract_bec_zones(df_a)
        test_bec = bec_zones[test_mask_a]
        comparison = backtester.compare_models(
            y_true=test_y_a,
            preds_a=test_preds_a,
            preds_b=test_preds_b,
            label_a=model_a_path.stem,
            label_b=model_b_path.stem,
            bec_zones=test_bec,
        )
    else:
        # Different datasets — side-by-side evaluation only
        comparison = {
            model_a_path.stem: metrics_a,
            model_b_path.stem: metrics_b,
        }

    # Build report
    report = {
        "command": "compare",
        "test_year": test_year,
        "model_a": {
            "path": str(model_a_path),
            "data_path": str(data_a_path),
            "n_test_samples": int(len(test_y_a)),
            "n_test_fires": int(test_y_a.sum()),
            "metrics": metrics_a,
        },
        "model_b": {
            "path": str(model_b_path),
            "data_path": str(data_b_path),
            "n_test_samples": int(len(test_y_b)),
            "n_test_fires": int(test_y_b.sum()),
            "metrics": metrics_b,
        },
        "comparison": comparison,
    }

    # Log comparison summary
    auc_diff = metrics_a.get("auc_roc", 0) - metrics_b.get("auc_roc", 0)
    logger.info(
        "AUC-ROC delta (A - B): %+.4f (%s is better)",
        auc_diff, "A" if auc_diff > 0 else "B",
    )

    _save_report(report, output_path)
    logger.info("Comparison complete.")


def cmd_zone_report(args: argparse.Namespace) -> None:
    """Per-BEC-zone accuracy breakdown for a single model."""
    data_path = Path(args.data)
    model_path = Path(args.model)
    output_path = Path(args.output)

    # Load data and model
    df = _load_data(data_path)
    X, available_features = _extract_features(df)
    y = df["fire"].values.astype(np.int32)
    bec_zones = _extract_bec_zones(df)

    model = _load_model(model_path)
    preds = _predict(model, X, available_features)

    # Per-zone breakdown via backtester
    backtester = HistoricalBacktester()
    zone_breakdown = backtester.per_zone_breakdown(
        y_true=y,
        y_pred=preds,
        bec_zones=bec_zones,
    )

    # Build report
    report = {
        "command": "zone-report",
        "data_path": str(data_path),
        "model_path": str(model_path),
        "n_samples": int(len(y)),
        "n_fires": int(y.sum()),
        "n_zones": len(set(bec_zones)),
        "zones": zone_breakdown,
    }

    # Log per-zone summary
    logger.info("--- Per-BEC-Zone Breakdown ---")
    if isinstance(zone_breakdown, dict):
        for zone, info in sorted(zone_breakdown.items()):
            n = info.get("n_samples", 0)
            fires = info.get("n_fires", 0)
            auc = info.get("auc_roc", float("nan"))
            logger.info(
                "  %s: n=%d, fires=%d, AUC=%.4f",
                zone, n, fires, auc,
            )
    elif isinstance(zone_breakdown, list):
        for entry in zone_breakdown:
            zone = entry.get("zone", "?")
            n = entry.get("n_samples", 0)
            fires = entry.get("n_fires", 0)
            auc = entry.get("auc_roc", float("nan"))
            logger.info(
                "  %s: n=%d, fires=%d, AUC=%.4f",
                zone, n, fires, auc,
            )

    _save_report(report, output_path)
    logger.info("Zone report complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="INFERNIS Historical Backtesting CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # --- backtest -----------------------------------------------------------
    p_bt = subparsers.add_parser(
        "backtest",
        help="Walk-forward temporal CV on a single model",
    )
    p_bt.add_argument("--data", required=True, help="Path to training parquet")
    p_bt.add_argument("--model", required=True, help="Path to XGBoost model JSON")
    p_bt.add_argument("--output", required=True, help="Path for JSON report output")
    p_bt.set_defaults(func=cmd_backtest)

    # --- compare ------------------------------------------------------------
    p_cmp = subparsers.add_parser(
        "compare",
        help="Compare two models (e.g. 5 km vs 1 km)",
    )
    p_cmp.add_argument("--model-a", required=True, help="Path to first model")
    p_cmp.add_argument("--model-b", required=True, help="Path to second model")
    p_cmp.add_argument("--data", required=True, help="Shared data path (used for both unless --data-a / --data-b set)")
    p_cmp.add_argument("--data-a", default=None, help="Data path for model A (overrides --data)")
    p_cmp.add_argument("--data-b", default=None, help="Data path for model B (overrides --data)")
    p_cmp.add_argument("--output", required=True, help="Path for JSON comparison report")
    p_cmp.set_defaults(func=cmd_compare)

    # --- zone-report --------------------------------------------------------
    p_zr = subparsers.add_parser(
        "zone-report",
        help="Per-BEC-zone accuracy breakdown",
    )
    p_zr.add_argument("--data", required=True, help="Path to training parquet")
    p_zr.add_argument("--model", required=True, help="Path to XGBoost model JSON")
    p_zr.add_argument("--output", required=True, help="Path for JSON zone report")
    p_zr.set_defaults(func=cmd_zone_report)

    # --- Parse & dispatch ---------------------------------------------------
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
