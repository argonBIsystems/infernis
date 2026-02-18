"""INFERNIS model training CLI.

Usage:
    # Full pipeline: process data, build training set, train model
    python scripts/train.py all

    # Individual steps:
    python scripts/train.py process    # Process raw data into features
    python scripts/train.py build      # Build labeled training dataset
    python scripts/train.py train      # Train XGBoost model
    python scripts/train.py evaluate   # Evaluate trained model
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train")

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODEL_PATH = PROJECT_ROOT / "models" / "fire_core_v1.json"


def load_grid(grid_path: str | None = None):
    """Load the BC grid from parquet or generate if needed.

    Args:
        grid_path: Explicit path to grid parquet. If None, uses default cache.
    """
    import pandas as pd
    from infernis.grid.generator import generate_bc_grid

    if grid_path:
        path = Path(grid_path)
        if not path.exists():
            logger.error("Grid file not found: %s", path)
            sys.exit(1)
        logger.info("Loading grid from %s", path)
        return pd.read_parquet(path)

    grid_cache = PROCESSED_DIR / "bc_grid.parquet"
    if grid_cache.exists():
        logger.info("Loading cached grid from %s", grid_cache)
        return pd.read_parquet(grid_cache)

    logger.info("Generating BC grid...")
    grid = generate_bc_grid()
    grid_cache.parent.mkdir(parents=True, exist_ok=True)

    # Save without geometry for parquet compatibility
    grid_flat = grid.drop(columns=["geometry"]).copy()
    grid_flat.to_parquet(grid_cache, index=False)
    logger.info("Grid cached: %d cells", len(grid_flat))
    return grid_flat


def cmd_process(args):
    """Process raw data into grid-aligned feature matrices."""
    from infernis.pipelines.data_processor import DataProcessor

    grid_df = load_grid(getattr(args, "grid_path", None))
    processor = DataProcessor(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)

    output_dir = processor.process_training_period(
        grid_df=grid_df,
        start_year=args.start_year,
        end_year=args.end_year,
        fire_season_only=not args.full_year,
        chunk_days=getattr(args, "chunk_days", 0),
    )
    logger.info("Feature processing complete. Output: %s", output_dir)


def cmd_build(args):
    """Build labeled training dataset from features + fire history."""
    from infernis.training.feature_builder import FeatureBuilder

    grid_df = load_grid(getattr(args, "grid_path", None))
    builder = FeatureBuilder(processed_dir=PROCESSED_DIR, raw_dir=RAW_DIR)

    output_path = builder.build_training_dataset(
        grid_df=grid_df,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    logger.info("Training dataset built: %s", output_path)


def cmd_train(args):
    """Train XGBoost model."""
    from infernis.training.trainer import FireModelTrainer

    data_path = PROCESSED_DIR / "training_data.parquet"
    if not data_path.exists():
        logger.error("Training data not found at %s. Run 'build' first.", data_path)
        sys.exit(1)

    trainer = FireModelTrainer(
        n_folds=args.folds,
        n_rounds=args.rounds,
    )

    X, y = trainer.load_data(data_path)

    if len(y) == 0 or y.sum() == 0:
        logger.error("Training data has no positive samples. Cannot train.")
        sys.exit(1)

    # Split: 80% train, 10% calibration, 10% test
    from sklearn.model_selection import train_test_split

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.11, random_state=42, stratify=y_trainval
    )

    logger.info("Split: %d train, %d calibration, %d test", len(y_train), len(y_cal), len(y_test))

    # Train
    output_path = Path(args.output) if args.output else MODEL_PATH
    cv_metrics = trainer.train(X_train, y_train, output_path=output_path)

    # Calibrate
    logger.info("Calibrating probabilities...")
    trainer.calibrate(X_cal, y_cal)

    # Evaluate
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(X_test, y_test)

    # SHAP analysis
    if not args.no_shap:
        logger.info("Computing SHAP values...")
        shap_importance = trainer.compute_shap(X_test)

    # Save metrics
    import json
    metrics_path = output_path.parent / "training_metrics.json"
    all_metrics = {
        "cv": cv_metrics,
        "test": {k: v for k, v in test_metrics.items() if k != "classification_report"},
        "data": {
            "total_samples": len(y),
            "positive_samples": int(y.sum()),
            "negative_samples": int(len(y) - y.sum()),
            "n_features": X.shape[1],
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


def cmd_evaluate(args):
    """Evaluate an existing model on the test set."""
    from infernis.training.trainer import FireModelTrainer

    data_path = PROCESSED_DIR / "training_data.parquet"
    if not data_path.exists():
        logger.error("Training data not found. Run 'build' first.")
        sys.exit(1)

    model_path = Path(args.model) if args.model else MODEL_PATH
    if not model_path.exists():
        logger.error("Model not found at %s. Run 'train' first.", model_path)
        sys.exit(1)

    trainer = FireModelTrainer()
    X, y = trainer.load_data(data_path)

    # Use same split as training
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    import xgboost as xgb
    trainer.model = xgb.Booster()
    trainer.model.load_model(str(model_path))

    metrics = trainer.evaluate(X_test, y_test)

    logger.info("Evaluation complete:")
    for key, value in metrics.items():
        if key != "classification_report":
            logger.info("  %s: %s", key, value)


def cmd_all(args):
    """Run the full pipeline: process -> build -> train."""
    logger.info("=== INFERNIS Full Training Pipeline ===")
    cmd_process(args)
    cmd_build(args)
    cmd_train(args)
    logger.info("=== Training pipeline complete ===")


def main():
    parser = argparse.ArgumentParser(description="INFERNIS Model Training")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--start-year", type=int, default=2015)
    common.add_argument("--end-year", type=int, default=2026)

    # Process
    p_process = subparsers.add_parser("process", parents=[common], help="Process raw data into features")
    p_process.add_argument("--full-year", action="store_true", help="Process all months (not just fire season)")
    p_process.add_argument("--grid-path", type=str, default=None, help="Path to grid parquet (default: data/processed/bc_grid.parquet)")
    p_process.add_argument("--chunk-days", type=int, default=0, help="Split monthly output into chunks of N days (0 = no chunking)")
    p_process.set_defaults(func=cmd_process)

    # Build
    p_build = subparsers.add_parser("build", parents=[common], help="Build labeled training dataset")
    p_build.add_argument("--grid-path", type=str, default=None, help="Path to grid parquet")
    p_build.set_defaults(func=cmd_build)

    # Train
    p_train = subparsers.add_parser("train", parents=[common], help="Train XGBoost model")
    p_train.add_argument("--folds", type=int, default=5, help="CV folds")
    p_train.add_argument("--rounds", type=int, default=1000, help="Max boosting rounds")
    p_train.add_argument("--output", type=str, default=None, help="Model output path")
    p_train.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis")
    p_train.set_defaults(func=cmd_train)

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", parents=[common], help="Evaluate trained model")
    p_eval.add_argument("--model", type=str, default=None, help="Model path")
    p_eval.set_defaults(func=cmd_evaluate)

    # All
    p_all = subparsers.add_parser("all", parents=[common], help="Run full pipeline")
    p_all.add_argument("--folds", type=int, default=5)
    p_all.add_argument("--rounds", type=int, default=1000)
    p_all.add_argument("--output", type=str, default=None)
    p_all.add_argument("--no-shap", action="store_true")
    p_all.add_argument("--full-year", action="store_true")
    p_all.add_argument("--grid-path", type=str, default=None, help="Path to grid parquet")
    p_all.add_argument("--chunk-days", type=int, default=0, help="Split output into chunks of N days")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
