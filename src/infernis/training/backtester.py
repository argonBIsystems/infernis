"""Walk-forward historical backtesting for fire prediction models.

Trains on [start_year, test_year-1], tests on test_year for each test year.
Produces per-year and per-BEC-zone evaluation breakdowns.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from infernis.training.evaluator import ModelEvaluator
from infernis.training.trainer import DEFAULT_PARAMS

logger = logging.getLogger(__name__)


class HistoricalBacktester:
    """Walk-forward backtesting across fire seasons."""

    def __init__(self, evaluator: ModelEvaluator | None = None):
        self.evaluator = evaluator or ModelEvaluator()

    def temporal_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        years: np.ndarray,
        bec_zones: np.ndarray | None = None,
        train_start: int = 2015,
        test_years: list[int] | None = None,
        xgb_params: dict | None = None,
        n_rounds: int = 1000,
    ) -> list[dict]:
        """Walk-forward temporal cross-validation.

        For each test year, trains on [train_start, test_year-1],
        evaluates on test_year.

        Args:
            X: Feature matrix [n_samples, n_features]
            y: Binary labels [n_samples]
            years: Year label per sample [n_samples]
            bec_zones: Optional BEC zone label per sample [n_samples]
            train_start: First year to include in training
            test_years: Years to test on (default: [2019..2024])
            xgb_params: XGBoost parameters (default: from trainer.DEFAULT_PARAMS)
            n_rounds: Max boosting rounds

        Returns:
            List of per-year evaluation dicts with keys:
            test_year, train_years, n_train, n_test, metrics, per_zone (if bec_zones provided)
        """
        if test_years is None:
            test_years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

        params = dict(xgb_params or DEFAULT_PARAMS)
        # ensure eval_metric for early stopping
        params.setdefault("eval_metric", "logloss")

        results = []

        for test_year in test_years:
            train_mask = (years >= train_start) & (years < test_year)
            test_mask = years == test_year

            n_train = int(train_mask.sum())
            n_test = int(test_mask.sum())

            # Skip if insufficient data
            if n_train < 10 or n_test < 10:
                logger.warning(
                    "Skipping test_year=%d: insufficient data (n_train=%d, n_test=%d)",
                    test_year,
                    n_train,
                    n_test,
                )
                continue

            X_train_full = X[train_mask]
            y_train_full = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]

            # Split training data into train/eval for early stopping
            X_train, X_eval, y_train, y_eval = train_test_split(
                X_train_full,
                y_train_full,
                test_size=0.1,
                stratify=y_train_full,
                random_state=42,
            )

            dtrain = xgb.DMatrix(X_train, label=y_train)
            deval = xgb.DMatrix(X_eval, label=y_eval)
            dtest = xgb.DMatrix(X_test)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=n_rounds,
                evals=[(deval, "eval")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            preds = model.predict(dtest)

            # Evaluate
            metrics = self.evaluator.evaluate(y_test, preds)

            train_year_range = list(range(train_start, test_year))
            result = {
                "test_year": test_year,
                "train_years": train_year_range,
                "n_train": n_train,
                "n_test": n_test,
                "metrics": metrics,
            }

            # Per-zone breakdown if zones provided
            if bec_zones is not None:
                zone_breakdown = self.per_zone_breakdown(
                    y_test,
                    preds,
                    bec_zones[test_mask],
                )
                result["per_zone"] = zone_breakdown

            results.append(result)

            logger.info(
                "Year %d: train=%d, test=%d, AUC=%.4f, PR-AUC=%.4f, Brier=%.4f",
                test_year,
                n_train,
                n_test,
                metrics["auc_roc"],
                metrics["pr_auc"],
                metrics["brier_score"],
            )

        return results

    def per_zone_breakdown(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bec_zones: np.ndarray,
        min_samples: int = 50,
    ) -> dict[str, dict]:
        """Compute evaluation metrics per BEC zone.

        Args:
            y_true: Binary ground truth
            y_pred: Predicted probabilities
            bec_zones: Zone label per sample
            min_samples: Minimum samples to include a zone

        Returns:
            Dict mapping zone name to evaluation metrics dict
        """
        unique_zones = np.unique(bec_zones)
        zone_results = {}

        for zone in unique_zones:
            zone_mask = bec_zones == zone
            n_samples = int(zone_mask.sum())

            if n_samples < min_samples:
                logger.debug(
                    "Skipping zone %s: only %d samples (min=%d)",
                    zone,
                    n_samples,
                    min_samples,
                )
                continue

            zone_y = y_true[zone_mask]
            zone_pred = y_pred[zone_mask]

            # Need both positive and negative examples for meaningful metrics
            n_fires = int(zone_y.sum())
            if n_fires == 0 or n_fires == n_samples:
                logger.debug(
                    "Skipping zone %s: no class variation (n_samples=%d, n_fires=%d)",
                    zone,
                    n_samples,
                    n_fires,
                )
                continue

            zone_metrics = self.evaluator.evaluate(zone_y, zone_pred)
            zone_metrics["n_samples"] = n_samples
            zone_metrics["n_fires"] = n_fires

            zone_name = str(zone)
            zone_results[zone_name] = zone_metrics

        return zone_results

    def compare_models(
        self,
        y_true: np.ndarray,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
        label_a: str = "model_a",
        label_b: str = "model_b",
        bec_zones: np.ndarray | None = None,
    ) -> dict:
        """Compare two model predictions on the same test set.

        Returns:
            Dict with per-metric comparison, deltas, and per-zone if zones provided.
        """
        metrics_a = self.evaluator.evaluate(y_true, preds_a)
        metrics_b = self.evaluator.evaluate(y_true, preds_b)

        # Key metrics to compare
        key_metrics = ["auc_roc", "pr_auc", "brier_score", "ece"]
        deltas = {}
        for metric in key_metrics:
            val_a = metrics_a.get(metric, 0.0)
            val_b = metrics_b.get(metric, 0.0)
            deltas[metric] = round(val_b - val_a, 4)

        comparison = {
            label_a: metrics_a,
            label_b: metrics_b,
            "deltas": deltas,
        }

        # Per-zone comparison if zones provided
        if bec_zones is not None:
            zones_a = self.per_zone_breakdown(y_true, preds_a, bec_zones)
            zones_b = self.per_zone_breakdown(y_true, preds_b, bec_zones)

            per_zone_deltas = {}
            # Compute deltas for zones present in both breakdowns
            common_zones = set(zones_a.keys()) & set(zones_b.keys())
            for zone in sorted(common_zones):
                zone_delta = {}
                for metric in key_metrics:
                    val_a = zones_a[zone].get(metric, 0.0)
                    val_b = zones_b[zone].get(metric, 0.0)
                    zone_delta[metric] = round(val_b - val_a, 4)
                per_zone_deltas[zone] = zone_delta

            comparison["per_zone"] = {
                label_a: zones_a,
                label_b: zones_b,
                "deltas": per_zone_deltas,
            }

        return comparison

    def generate_report(
        self,
        temporal_results: list[dict],
        model_label: str = "fire_core_v1",
        grid_resolution_km: float = 5.0,
    ) -> dict:
        """Generate comprehensive backtesting report.

        Args:
            temporal_results: Output from temporal_cv()
            model_label: Model identifier
            grid_resolution_km: Grid resolution

        Returns:
            Full report dict with overall, per-year, per-zone sections
        """
        if not temporal_results:
            return {
                "metadata": {
                    "model_label": model_label,
                    "grid_resolution_km": grid_resolution_km,
                    "generated_at": datetime.now(tz=None).isoformat(),
                },
                "overall": {},
                "per_year": [],
                "per_zone": {},
            }

        # Key metrics to aggregate
        key_metrics = [
            "auc_roc",
            "pr_auc",
            "brier_score",
            "ece",
            "best_f1",
        ]

        # Collect per-year metric values
        metric_values = {m: [] for m in key_metrics}
        for result in temporal_results:
            metrics = result.get("metrics", {})
            for m in key_metrics:
                if m in metrics:
                    metric_values[m].append(metrics[m])

        # Compute overall mean and std
        overall = {}
        for m in key_metrics:
            values = metric_values[m]
            if values:
                overall[f"{m}_mean"] = round(float(np.mean(values)), 4)
                overall[f"{m}_std"] = round(float(np.std(values)), 4)

        overall["n_years"] = len(temporal_results)
        overall["total_train_samples"] = sum(r.get("n_train", 0) for r in temporal_results)
        overall["total_test_samples"] = sum(r.get("n_test", 0) for r in temporal_results)

        # Per-year summary (compact view)
        per_year = []
        for result in temporal_results:
            metrics = result.get("metrics", {})
            year_summary = {
                "test_year": result["test_year"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
            }
            for m in key_metrics:
                if m in metrics:
                    year_summary[m] = metrics[m]
            per_year.append(year_summary)

        # Aggregate per-zone results across all years
        zone_aggregated: dict[str, dict[str, list[float]]] = {}
        for result in temporal_results:
            per_zone = result.get("per_zone", {})
            for zone, zone_metrics in per_zone.items():
                if zone not in zone_aggregated:
                    zone_aggregated[zone] = {m: [] for m in key_metrics}
                    zone_aggregated[zone]["n_samples_list"] = []
                    zone_aggregated[zone]["n_fires_list"] = []
                for m in key_metrics:
                    if m in zone_metrics:
                        zone_aggregated[zone][m].append(zone_metrics[m])
                if "n_samples" in zone_metrics:
                    zone_aggregated[zone]["n_samples_list"].append(zone_metrics["n_samples"])
                if "n_fires" in zone_metrics:
                    zone_aggregated[zone]["n_fires_list"].append(zone_metrics["n_fires"])

        # Compute zone-level summaries
        per_zone_summary = {}
        for zone, values in zone_aggregated.items():
            zone_summary = {}
            for m in key_metrics:
                if values.get(m):
                    zone_summary[f"{m}_mean"] = round(
                        float(np.mean(values[m])),
                        4,
                    )
                    zone_summary[f"{m}_std"] = round(
                        float(np.std(values[m])),
                        4,
                    )
            zone_summary["n_years_present"] = len(values.get("n_samples_list", []))
            zone_summary["total_samples"] = sum(values.get("n_samples_list", []))
            zone_summary["total_fires"] = sum(values.get("n_fires_list", []))
            per_zone_summary[zone] = zone_summary

        report = {
            "metadata": {
                "model_label": model_label,
                "grid_resolution_km": grid_resolution_km,
                "generated_at": datetime.now(tz=None).isoformat(),
                "test_years": [r["test_year"] for r in temporal_results],
            },
            "overall": overall,
            "per_year": per_year,
            "per_zone": per_zone_summary,
        }

        return report
