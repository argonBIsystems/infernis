"""Model evaluation and drift detection module.

Evaluates model performance against held-out test data and monitors
for prediction drift by comparing predicted probabilities against
observed fire occurrences during each fire season.

Drift detection triggers:
- AUC-ROC drops below 0.85
- Calibration error exceeds 0.05
- Annual retraining recommended regardless
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Drift detection thresholds
AUC_ROC_THRESHOLD = 0.85
CALIBRATION_ERROR_THRESHOLD = 0.05


class ModelEvaluator:
    """Evaluates fire prediction model performance and detects drift."""

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        """Compute comprehensive evaluation metrics.

        Args:
            y_true: Binary ground truth (0/1) array
            y_prob: Predicted probability array [0, 1]

        Returns:
            Dict with AUC-ROC, PR-AUC, F1, Brier score, calibration error,
            and per-threshold metrics.
        """
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        # Core metrics
        auc_roc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)

        # Calibration error (Expected Calibration Error)
        ece = self._expected_calibration_error(y_true, y_prob)

        # Threshold-specific metrics at different operating points
        thresholds = [0.1, 0.2, 0.3, 0.5]
        threshold_metrics = {}
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            threshold_metrics[str(t)] = {
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }

        # Best F1 threshold search
        best_f1 = 0.0
        best_threshold = 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            y_pred = (y_prob >= t).astype(int)
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        return {
            "auc_roc": round(auc_roc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
            "best_f1": round(best_f1, 4),
            "best_threshold": round(best_threshold, 2),
            "n_samples": int(len(y_true)),
            "n_positive": int(y_true.sum()),
            "prevalence": round(float(y_true.mean()), 6),
            "threshold_metrics": threshold_metrics,
        }

    def check_drift(self, current_metrics: dict, baseline_metrics: dict = None) -> dict:
        """Check for model drift by comparing current to baseline metrics.

        Args:
            current_metrics: Output from evaluate()
            baseline_metrics: Previous evaluation metrics to compare against

        Returns:
            Drift report with status and recommendations
        """
        report = {
            "drift_detected": False,
            "alerts": [],
            "recommendations": [],
        }

        auc = current_metrics.get("auc_roc", 0.0)
        ece = current_metrics.get("ece", 1.0)

        # Check absolute thresholds
        if auc < AUC_ROC_THRESHOLD:
            report["drift_detected"] = True
            report["alerts"].append(f"AUC-ROC ({auc:.3f}) below threshold ({AUC_ROC_THRESHOLD})")
            report["recommendations"].append("Trigger model retraining with latest data")

        if ece > CALIBRATION_ERROR_THRESHOLD:
            report["drift_detected"] = True
            report["alerts"].append(
                f"Calibration error ({ece:.3f}) above threshold ({CALIBRATION_ERROR_THRESHOLD})"
            )
            report["recommendations"].append("Recalibrate model with Platt scaling")

        # Check relative drift against baseline
        if baseline_metrics:
            baseline_auc = baseline_metrics.get("auc_roc", 0.0)
            if baseline_auc > 0 and auc < baseline_auc * 0.95:
                report["drift_detected"] = True
                report["alerts"].append(
                    f"AUC-ROC dropped {((baseline_auc - auc) / baseline_auc * 100):.1f}% "
                    f"from baseline ({baseline_auc:.3f} -> {auc:.3f})"
                )

        if not report["drift_detected"]:
            report["recommendations"].append("Model performance is within acceptable bounds")

        return report

    def compute_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """Compute calibration curve (reliability diagram) data.

        Returns bin edges, mean predicted probability per bin,
        and fraction of positives per bin.
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_fractions = []
        bin_counts = []

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append(float(y_prob[mask].mean()))
                bin_fractions.append(float(y_true[mask].mean()))
                bin_counts.append(int(mask.sum()))
            else:
                bin_centers.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))
                bin_fractions.append(0.0)
                bin_counts.append(0)

        return {
            "bin_edges": [round(e, 2) for e in bin_edges.tolist()],
            "predicted_mean": [round(c, 4) for c in bin_centers],
            "actual_fraction": [round(f, 4) for f in bin_fractions],
            "bin_counts": bin_counts,
        }

    def _expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE)."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_total = len(y_true)

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            n_bin = mask.sum()
            if n_bin > 0:
                avg_pred = y_prob[mask].mean()
                avg_true = y_true[mask].mean()
                ece += (n_bin / n_total) * abs(avg_pred - avg_true)

        return ece

    def evaluate_per_zone(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        bec_zones: np.ndarray,
        min_samples: int = 50,
    ) -> dict[str, dict]:
        """Run evaluate() per BEC zone, filtering zones with too few samples.

        Args:
            y_true: Binary ground truth
            y_prob: Predicted probabilities
            bec_zones: Zone label per sample
            min_samples: Minimum samples to include a zone

        Returns:
            Dict mapping zone name to evaluation metrics
        """
        results = {}
        for zone in sorted(set(bec_zones)):
            mask = bec_zones == zone
            n = int(mask.sum())
            if n < min_samples:
                continue
            zone_y = y_true[mask]
            zone_p = y_prob[mask]
            # Need both classes present for AUC
            if zone_y.sum() == 0 or zone_y.sum() == len(zone_y):
                continue
            try:
                metrics = self.evaluate(zone_y, zone_p)
                metrics["n_samples"] = n
                metrics["n_fires"] = int(zone_y.sum())
                results[str(zone)] = metrics
            except Exception as e:
                logger.warning("Could not evaluate zone %s: %s", zone, e)
        return results

    def compare_models(
        self,
        y_true: np.ndarray,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
        label_a: str = "model_a",
        label_b: str = "model_b",
    ) -> dict:
        """Compare two model predictions on the same test set.

        Returns per-metric comparison with deltas.
        """
        metrics_a = self.evaluate(y_true, preds_a)
        metrics_b = self.evaluate(y_true, preds_b)

        delta_keys = ["auc_roc", "pr_auc", "brier_score", "ece", "best_f1"]
        deltas = {}
        for k in delta_keys:
            va = metrics_a.get(k, 0.0)
            vb = metrics_b.get(k, 0.0)
            deltas[k] = round(vb - va, 4)

        return {
            label_a: metrics_a,
            label_b: metrics_b,
            "deltas": deltas,
            "summary": {
                "auc_roc_improved": deltas.get("auc_roc", 0) > 0,
                "brier_improved": deltas.get("brier_score", 0) < 0,
                "ece_improved": deltas.get("ece", 0) < 0,
            },
        }

    def save_report(self, metrics: dict, filepath: str):
        """Save evaluation metrics to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Evaluation report saved to %s", filepath)

    def load_baseline(self, filepath: str) -> dict | None:
        """Load baseline metrics from a previous evaluation."""
        path = Path(filepath)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)


class SeasonalDriftMonitor:
    """Monitors model drift across fire seasons by comparing predictions
    against actual fire observations.

    Architecture spec: During fire season (May-October), actual fire reports
    from CWFIS are ingested weekly and compared against predictions.
    """

    def __init__(self):
        self.evaluator = ModelEvaluator()

    def evaluate_season(
        self,
        predictions: dict,
        actual_fires: list,
        grid_cells: dict,
    ) -> dict:
        """Evaluate model predictions against observed fires for a season.

        Args:
            predictions: Dict of cell_id -> list of {date, score} predictions
            actual_fires: List of {lat, lon, date} fire observations
            grid_cells: Dict of cell_id -> {lat, lon}

        Returns:
            Seasonal evaluation metrics
        """
        from scipy.spatial import KDTree

        if not grid_cells:
            return {"error": "No grid cells available"}

        # Build KD-tree for fire-to-cell assignment
        cell_ids = list(grid_cells.keys())
        coords = np.array([[grid_cells[c]["lat"], grid_cells[c]["lon"]] for c in cell_ids])
        tree = KDTree(coords)

        # Assign fires to cells
        fire_cells = set()
        for fire in actual_fires:
            _, idx = tree.query([fire["lat"], fire["lon"]])
            cell_id = cell_ids[idx]
            fire_date = fire["date"] if isinstance(fire["date"], str) else fire["date"].isoformat()
            fire_cells.add((cell_id, fire_date))

        # Build prediction/observation arrays
        y_true_list = []
        y_prob_list = []

        for cell_id, pred_list in predictions.items():
            for pred in pred_list:
                pred_date = (
                    pred["date"] if isinstance(pred["date"], str) else pred["date"].isoformat()
                )
                y_true_list.append(1.0 if (cell_id, pred_date) in fire_cells else 0.0)
                y_prob_list.append(pred["score"])

        if not y_true_list:
            return {"error": "No prediction-observation pairs"}

        y_true = np.array(y_true_list)
        y_prob = np.array(y_prob_list)

        metrics = self.evaluator.evaluate(y_true, y_prob)
        drift = self.evaluator.check_drift(metrics)

        return {
            "season_metrics": metrics,
            "drift_report": drift,
            "total_predictions": len(y_true),
            "total_fires_observed": len(fire_cells),
            "hit_rate": round(
                sum(
                    1
                    for c, d in fire_cells
                    if any(p["date"] == d and p["score"] > 0.3 for p in predictions.get(c, []))
                )
                / max(len(fire_cells), 1),
                3,
            ),
        }
