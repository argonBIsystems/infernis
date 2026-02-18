"""Tests for the XGBoost model trainer."""

import numpy as np
import pandas as pd

from infernis.pipelines.data_processor import FEATURE_NAMES
from infernis.training.trainer import DEFAULT_PARAMS, FireModelTrainer


class TestDefaultParams:
    def test_objective(self):
        assert DEFAULT_PARAMS["objective"] == "binary:logistic"

    def test_max_depth(self):
        assert DEFAULT_PARAMS["max_depth"] == 8

    def test_learning_rate(self):
        assert DEFAULT_PARAMS["learning_rate"] == 0.05


class TestLoadData:
    def test_loads_parquet(self, tmp_path):
        # Create a small training dataset
        n = 100
        data = {"fire": np.concatenate([np.ones(20), np.zeros(80)]).astype(int)}
        for feat in FEATURE_NAMES:
            data[feat] = np.random.randn(n).astype(np.float32)

        df = pd.DataFrame(data)
        path = tmp_path / "test_data.parquet"
        df.to_parquet(path, index=False)

        trainer = FireModelTrainer()
        X, y = trainer.load_data(path)

        assert X.shape == (n, len(FEATURE_NAMES))
        assert y.shape == (n,)
        assert y.sum() == 20

    def test_handles_nan(self, tmp_path):
        n = 50
        data = {"fire": np.zeros(n, dtype=int)}
        data["fire"][:5] = 1
        for feat in FEATURE_NAMES:
            vals = np.random.randn(n).astype(np.float32)
            vals[0] = np.nan
            data[feat] = vals

        df = pd.DataFrame(data)
        path = tmp_path / "nan_data.parquet"
        df.to_parquet(path, index=False)

        trainer = FireModelTrainer()
        X, y = trainer.load_data(path)
        assert not np.any(np.isnan(X))


class TestTraining:
    def test_train_small_dataset(self, tmp_path):
        """Train on a small synthetic dataset."""
        rng = np.random.default_rng(42)
        n = 500
        n_pos = 50

        # Create separable data: positive samples have higher feature values
        X_neg = rng.standard_normal((n - n_pos, len(FEATURE_NAMES))).astype(np.float32)
        X_pos = rng.standard_normal((n_pos, len(FEATURE_NAMES))).astype(np.float32) + 1.0
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(n_pos), np.zeros(n - n_pos)]).astype(np.int32)

        trainer = FireModelTrainer(n_folds=3, n_rounds=50, early_stopping_rounds=10)
        output_path = tmp_path / "test_model.json"

        metrics = trainer.train(X, y, output_path=output_path)

        assert output_path.exists()
        assert "auc_roc_mean" in metrics
        assert metrics["auc_roc_mean"] > 0.5  # Better than random
        assert trainer.model is not None

    def test_feature_importance_computed(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, len(FEATURE_NAMES))).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)  # Feature 0 is perfectly predictive

        trainer = FireModelTrainer(n_folds=2, n_rounds=20, early_stopping_rounds=5)
        trainer.train(X, y, output_path=tmp_path / "model.json")

        assert trainer.feature_importance is not None
        assert len(trainer.feature_importance) > 0

    def test_evaluate_returns_metrics(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, len(FEATURE_NAMES))).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)

        trainer = FireModelTrainer(n_folds=2, n_rounds=20, early_stopping_rounds=5)
        trainer.train(X, y, output_path=tmp_path / "model.json")

        metrics = trainer.evaluate(X, y)
        assert "auc_roc" in metrics
        assert "brier_score" in metrics
        assert metrics["auc_roc"] > 0.5

    def test_large_dataset_training(self, tmp_path):
        """Train on 5000 samples to verify hist tree method works at scale."""
        rng = np.random.default_rng(42)
        n = 5000
        n_pos = 500
        X_pos = rng.standard_normal((n_pos, len(FEATURE_NAMES))).astype(np.float32) + 1.0
        X_neg = rng.standard_normal((n - n_pos, len(FEATURE_NAMES))).astype(np.float32)
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(n_pos), np.zeros(n - n_pos)]).astype(np.int32)

        trainer = FireModelTrainer(n_folds=3, n_rounds=50, early_stopping_rounds=10)
        output_path = tmp_path / "large_model.json"
        metrics = trainer.train(X, y, output_path=output_path)

        assert output_path.exists()
        assert metrics["auc_roc_mean"] > 0.7

    def test_model_serialization_roundtrip(self, tmp_path):
        """Save and reload model, verify predictions match."""
        import xgboost as xgb

        rng = np.random.default_rng(42)
        n = 500
        X = rng.standard_normal((n, len(FEATURE_NAMES))).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)

        trainer = FireModelTrainer(n_folds=2, n_rounds=30, early_stopping_rounds=10)
        model_path = tmp_path / "roundtrip_model.json"
        trainer.train(X, y, output_path=model_path)

        # Get predictions from trained model
        dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
        preds_before = trainer.model.predict(dmat)

        # Reload model from disk
        reloaded = xgb.Booster()
        reloaded.load_model(str(model_path))
        preds_after = reloaded.predict(dmat)

        np.testing.assert_array_almost_equal(preds_before, preds_after, decimal=6)
