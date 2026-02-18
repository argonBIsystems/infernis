"""XGBoost fire risk model trainer.

Trains a binary classifier to predict fire ignition probability (0-1)
for each grid cell on a given day.

Features:
- Stratified K-fold cross-validation
- Class imbalance handling via scale_pos_weight
- Platt scaling for probability calibration
- SHAP-based feature importance analysis
- Model serialization to JSON format
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from infernis.pipelines.data_processor import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Default XGBoost hyperparameters tuned for fire prediction
DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "max_depth": 8,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}


class FireModelTrainer:
    """Trains and evaluates the XGBoost fire risk model."""

    def __init__(
        self,
        params: dict | None = None,
        n_folds: int = 5,
        n_rounds: int = 1000,
        early_stopping_rounds: int = 50,
    ):
        self.params = params or DEFAULT_PARAMS.copy()
        self.n_folds = n_folds
        self.n_rounds = n_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.calibrated_model = None
        self.feature_importance = None

    def load_data(self, data_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load training data from parquet file.

        Returns (X, y) where X is the feature matrix and y is the label array.
        """
        df = pd.read_parquet(data_path)

        available_features = [f for f in FEATURE_NAMES if f in df.columns]
        if len(available_features) < len(FEATURE_NAMES):
            missing = set(FEATURE_NAMES) - set(available_features)
            logger.warning("Missing features: %s", missing)

        X = df[available_features].values.astype(np.float32)
        y = df["fire"].values.astype(np.int32)

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        n_pos = y.sum()
        n_neg = len(y) - n_pos
        logger.info(
            "Loaded data: %d samples (%d positive, %d negative, ratio %.1f:1)",
            len(y),
            n_pos,
            n_neg,
            n_neg / max(n_pos, 1),
        )
        return X, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_path: Path | None = None,
    ) -> dict:
        """Train the XGBoost model with cross-validation.

        Returns dict with training metrics.
        """
        import xgboost as xgb

        # Set scale_pos_weight for class imbalance
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        self.params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        logger.info("scale_pos_weight: %.2f", self.params["scale_pos_weight"])

        # Cross-validation for evaluation
        cv_metrics = self._cross_validate(X, y)

        # Train final model on all data
        logger.info("Training final model on all %d samples...", len(y))
        dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES[: X.shape[1]])

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_rounds,
            evals=[(dtrain, "train")],
            verbose_eval=100,
        )

        # Feature importance
        self.feature_importance = self.model.get_score(importance_type="gain")
        self._log_feature_importance()

        # Save model
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_model(str(output_path))
            logger.info("Model saved to %s", output_path)

            # Save feature importance alongside
            importance_path = output_path.parent / "feature_importance.json"
            with open(importance_path, "w") as f:
                json.dump(self.feature_importance, f, indent=2)

        return cv_metrics

    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Run stratified K-fold cross-validation."""
        import xgboost as xgb

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        fold_metrics = {
            "auc_roc": [],
            "avg_precision": [],
            "brier_score": [],
            "log_loss": [],
        }

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES[: X.shape[1]])
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES[: X.shape[1]])

            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.n_rounds,
                evals=[(dval, "val")],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )

            y_pred = model.predict(dval)

            fold_metrics["auc_roc"].append(roc_auc_score(y_val, y_pred))
            fold_metrics["avg_precision"].append(average_precision_score(y_val, y_pred))
            fold_metrics["brier_score"].append(brier_score_loss(y_val, y_pred))
            fold_metrics["log_loss"].append(log_loss(y_val, y_pred))

            logger.info(
                "Fold %d/%d: AUC=%.4f, AP=%.4f, Brier=%.4f",
                fold + 1,
                self.n_folds,
                fold_metrics["auc_roc"][-1],
                fold_metrics["avg_precision"][-1],
                fold_metrics["brier_score"][-1],
            )

        # Compute mean and std
        cv_results = {}
        for metric, values in fold_metrics.items():
            cv_results[f"{metric}_mean"] = np.mean(values)
            cv_results[f"{metric}_std"] = np.std(values)

        logger.info(
            "CV Results: AUC=%.4f +/- %.4f, AP=%.4f +/- %.4f",
            cv_results["auc_roc_mean"],
            cv_results["auc_roc_std"],
            cv_results["avg_precision_mean"],
            cv_results["avg_precision_std"],
        )

        return cv_results

    def calibrate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Apply Platt scaling for probability calibration.

        Uses a held-out set for calibration to avoid overfitting.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before calibration")

        import xgboost as xgb
        from sklearn.base import BaseEstimator, ClassifierMixin

        # Wrap XGBoost model for sklearn calibration
        class _XGBWrapper(BaseEstimator, ClassifierMixin):
            _estimator_type = "classifier"

            def __init__(self, booster):
                self.booster = booster
                self.classes_ = np.array([0, 1])

            def fit(self, X, y):
                return self

            def predict_proba(self, X):
                dmatrix = xgb.DMatrix(X, feature_names=FEATURE_NAMES[: X.shape[1]])
                pos_proba = self.booster.predict(dmatrix)
                return np.column_stack([1 - pos_proba, pos_proba])

            def predict(self, X):
                proba = self.predict_proba(X)
                return (proba[:, 1] >= 0.5).astype(int)

        wrapper = _XGBWrapper(self.model)

        # Manual sigmoid (Platt scaling) calibration to avoid sklearn version issues
        from sklearn.linear_model import LogisticRegression

        y_pred_raw = wrapper.predict_proba(X)[:, 1]
        self._platt_model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._platt_model.fit(y_pred_raw.reshape(-1, 1), y)

        y_pred_cal = self._platt_model.predict_proba(y_pred_raw.reshape(-1, 1))[:, 1]

        brier_raw = brier_score_loss(y, y_pred_raw)
        brier_cal = brier_score_loss(y, y_pred_cal)

        logger.info(
            "Calibration: Brier score raw=%.4f, calibrated=%.4f",
            brier_raw,
            brier_cal,
        )

    def compute_shap(self, X: np.ndarray, max_samples: int = 5000) -> dict:
        """Compute SHAP feature importance values.

        Returns dict with mean absolute SHAP values per feature.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before SHAP analysis")

        try:
            import shap
        except ImportError:
            logger.warning("shap not installed, skipping SHAP analysis")
            return {}

        # Subsample for speed
        if len(X) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), size=max_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        # Mean absolute SHAP per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_names = FEATURE_NAMES[: X.shape[1]]

        shap_importance = dict(zip(feature_names, mean_abs_shap.tolist()))
        shap_importance = dict(sorted(shap_importance.items(), key=lambda x: -x[1]))

        logger.info("Top 10 SHAP features:")
        for name, val in list(shap_importance.items())[:10]:
            logger.info("  %s: %.4f", name, val)

        return shap_importance

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the model on a test set."""
        import xgboost as xgb

        if self.model is None:
            raise RuntimeError("Model must be trained before evaluation")

        dmatrix = xgb.DMatrix(X, feature_names=FEATURE_NAMES[: X.shape[1]])

        y_pred_raw = self.model.predict(dmatrix)
        # Use Platt calibration if available
        if hasattr(self, "_platt_model") and self._platt_model is not None:
            y_pred = self._platt_model.predict_proba(y_pred_raw.reshape(-1, 1))[:, 1]
        else:
            y_pred = y_pred_raw

        metrics = {
            "auc_roc": roc_auc_score(y, y_pred),
            "avg_precision": average_precision_score(y, y_pred),
            "brier_score": brier_score_loss(y, y_pred),
            "log_loss": log_loss(y, y_pred),
        }

        # Classification report at 0.5 threshold
        y_binary = (y_pred >= 0.5).astype(int)
        report = classification_report(y, y_binary, output_dict=True)
        metrics["classification_report"] = report

        logger.info(
            "Test metrics: AUC=%.4f, AP=%.4f, Brier=%.4f",
            metrics["auc_roc"],
            metrics["avg_precision"],
            metrics["brier_score"],
        )

        return metrics

    def _log_feature_importance(self):
        """Log top features by gain."""
        if not self.feature_importance:
            return

        sorted_feats = sorted(self.feature_importance.items(), key=lambda x: -x[1])
        logger.info("Top 10 features by gain:")
        for name, gain in sorted_feats[:10]:
            logger.info("  %s: %.2f", name, gain)
