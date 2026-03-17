"""XGBoost quantile regression for confidence interval estimation.

Trains two companion models — a lower-quantile model (q=0.05 by default)
and an upper-quantile model (q=0.95 by default) — that bracket the main
fire risk score with a 90% confidence interval.

Requires XGBoost >= 2.0 for ``reg:quantileerror`` objective support.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# XGBoost hyperparameters for quantile regression.
# Lighter than the main classifier — quantile models are auxiliary.
_QUANTILE_PARAMS_BASE = {
    "objective": "reg:quantileerror",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": 42,
    "n_estimators": 500,
}


def train_quantile_models(
    X: np.ndarray,
    y: np.ndarray,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> tuple:
    """Train lower and upper quantile regression models.

    Parameters
    ----------
    X:
        Feature matrix, shape ``(n_samples, n_features)``.
    y:
        Continuous target values in ``[0, 1]`` (fire risk scores).
    lower_q:
        Lower quantile (default 0.05 → 5th percentile).
    upper_q:
        Upper quantile (default 0.95 → 95th percentile).

    Returns
    -------
    Tuple of ``(lower_model, upper_model)``, both ``xgboost.XGBRegressor``
    instances trained on the respective quantiles.

    Raises
    ------
    ImportError:
        If XGBoost < 2.0 is installed (``reg:quantileerror`` unavailable).
    ValueError:
        If ``lower_q >= upper_q``, or if X / y are empty / mismatched.
    """
    import xgboost as xgb

    _check_xgb_version(xgb)

    if lower_q >= upper_q:
        raise ValueError(f"lower_q ({lower_q}) must be less than upper_q ({upper_q})")
    if len(X) == 0 or len(y) == 0:
        raise ValueError("X and y must not be empty")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    logger.info(
        "Training quantile models: lower_q=%.2f, upper_q=%.2f, n_samples=%d",
        lower_q,
        upper_q,
        len(y),
    )

    lower_model = _train_single_quantile(X, y, quantile_alpha=lower_q)
    upper_model = _train_single_quantile(X, y, quantile_alpha=upper_q)

    return lower_model, upper_model


def _train_single_quantile(X: np.ndarray, y: np.ndarray, quantile_alpha: float):
    """Train a single quantile regression model."""
    import xgboost as xgb

    params = dict(_QUANTILE_PARAMS_BASE)
    params["quantile_alpha"] = quantile_alpha

    n_estimators = params.pop("n_estimators")

    model = xgb.XGBRegressor(n_estimators=n_estimators, **params)
    model.fit(X, y)

    logger.info(
        "Quantile model (alpha=%.2f) trained — %d estimators",
        quantile_alpha,
        model.n_estimators,
    )
    return model


def predict_quantiles(
    X: np.ndarray,
    lower_model,
    upper_model,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict lower and upper quantile bounds.

    Bounds are clipped to ``[0, 1]`` and ordering is enforced: wherever
    the lower model predicts above the upper model the two values are
    swapped so that ``lower[i] <= upper[i]`` always holds.

    Parameters
    ----------
    X:
        Feature matrix, shape ``(n_samples, n_features)``.
    lower_model:
        Trained lower-quantile ``XGBRegressor``.
    upper_model:
        Trained upper-quantile ``XGBRegressor``.

    Returns
    -------
    ``(lower_bounds, upper_bounds)`` — both float64 arrays clipped to
    ``[0, 1]`` with ``lower_bounds[i] <= upper_bounds[i]`` guaranteed.
    """
    X = np.asarray(X, dtype=np.float32)

    lower = np.asarray(lower_model.predict(X), dtype=np.float64)
    upper = np.asarray(upper_model.predict(X), dtype=np.float64)

    # Clip to valid probability range
    lower = np.clip(lower, 0.0, 1.0)
    upper = np.clip(upper, 0.0, 1.0)

    # Enforce ordering: swap wherever lower > upper
    crossed = lower > upper
    if crossed.any():
        logger.debug(
            "Quantile crossing detected in %d/%d cells — swapping to enforce ordering",
            int(crossed.sum()),
            len(crossed),
        )
        lower[crossed], upper[crossed] = upper[crossed].copy(), lower[crossed].copy()

    return lower, upper


def save_quantile_models(
    lower_model,
    upper_model,
    lower_path: str | Path,
    upper_path: str | Path,
) -> None:
    """Save lower and upper quantile models to disk in XGBoost JSON format.

    Parameters
    ----------
    lower_model:
        Trained lower-quantile ``XGBRegressor``.
    upper_model:
        Trained upper-quantile ``XGBRegressor``.
    lower_path:
        Destination path for the lower-quantile model file.
    upper_path:
        Destination path for the upper-quantile model file.
    """
    lower_path = Path(lower_path)
    upper_path = Path(upper_path)

    lower_path.parent.mkdir(parents=True, exist_ok=True)
    upper_path.parent.mkdir(parents=True, exist_ok=True)

    lower_model.save_model(str(lower_path))
    upper_model.save_model(str(upper_path))

    logger.info(
        "Saved quantile models — lower: %s, upper: %s", lower_path, upper_path
    )


def load_quantile_models(
    lower_path: str | Path,
    upper_path: str | Path,
) -> tuple:
    """Load lower and upper quantile models from disk.

    Returns ``(None, None)`` if either file is missing, so the caller can
    treat absent quantile models as gracefully unavailable.

    Parameters
    ----------
    lower_path:
        Path to the saved lower-quantile model file.
    upper_path:
        Path to the saved upper-quantile model file.

    Returns
    -------
    ``(lower_model, upper_model)`` if both files exist, else ``(None, None)``.
    """
    lower_path = Path(lower_path)
    upper_path = Path(upper_path)

    if not lower_path.exists():
        logger.info("Lower quantile model not found at %s — CI unavailable", lower_path)
        return None, None
    if not upper_path.exists():
        logger.info("Upper quantile model not found at %s — CI unavailable", upper_path)
        return None, None

    try:
        import xgboost as xgb

        lower_model = xgb.XGBRegressor()
        lower_model.load_model(str(lower_path))

        upper_model = xgb.XGBRegressor()
        upper_model.load_model(str(upper_path))

        logger.info(
            "Loaded quantile models — lower: %s, upper: %s", lower_path, upper_path
        )
        return lower_model, upper_model

    except Exception as exc:
        logger.warning("Failed to load quantile models: %s — CI unavailable", exc)
        return None, None


def _check_xgb_version(xgb) -> None:
    """Raise ImportError if XGBoost version does not support quantile objective."""
    version_str = getattr(xgb, "__version__", "0.0.0")
    parts = version_str.split(".")
    try:
        major = int(parts[0])
    except (ValueError, IndexError):
        major = 0
    if major < 2:
        raise ImportError(
            f"XGBoost >= 2.0 required for reg:quantileerror objective "
            f"(installed: {version_str})"
        )
