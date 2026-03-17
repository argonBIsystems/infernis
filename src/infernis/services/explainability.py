"""SHAP-based explainability service for INFERNIS fire risk predictions.

Uses TreeSHAP (shap.TreeExplainer) for fast, exact per-feature contributions
from the XGBoost model. Every risk prediction can include a ranked list of the
top drivers explaining WHY risk is high or low.

Usage:
    svc = ExplainabilityService(model, FEATURE_NAMES)
    shap_vals = svc.compute_shap_values(X)          # [n_cells, n_features]
    drivers = svc.get_drivers(feature_values, shap_values=shap_vals[i])
    summary = svc.generate_summary(drivers, level="HIGH")
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Human-readable feature descriptions (templates)
# Each value is a string template.  {value:.Xf} placeholders are supported.
# Direction ("increasing" or "decreasing") is resolved at runtime.
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS: dict[str, str] = {
    # FWI components
    "ffmc": "Fine Fuel Moisture Code {value:.1f} — indicates how dry surface fuels are",
    "dmc": "Duff Moisture Code {value:.1f} — measures moisture in moderate-depth organic layers",
    "dc": "Drought Code {value:.1f} — long-term soil moisture deficit",
    "isi": "Initial Spread Index {value:.1f} — fire spread rate driven by wind and FFMC",
    "bui": "Build-Up Index {value:.1f} — total fuel available for combustion",
    "fwi": "Fire Weather Index {value:.1f} — composite fire danger rating",
    # Weather
    "temperature_c": "Temperature {value:.1f} °C — high heat accelerates fuel drying",
    "rh_pct": "Relative humidity {value:.0f}% — low RH dries fuels rapidly",
    "wind_kmh": "Wind speed {value:.1f} km/h — drives fire spread and intensifies ignition",
    "wind_dir_deg": "Wind direction {value:.0f}° — affects spread direction",
    "precip_24h_mm": "Precipitation {value:.1f} mm in last 24 h — wets fuels and suppresses fire",
    "soil_moisture_1": "Surface soil moisture {value:.3f} m³/m³ — shallow root zone wetness",
    "soil_moisture_2": "Soil moisture layer 2 ({value:.3f} m³/m³) — mid-depth soil water content",
    "soil_moisture_3": "Soil moisture layer 3 ({value:.3f} m³/m³) — deep soil water content",
    "soil_moisture_4": "Soil moisture layer 4 ({value:.3f} m³/m³) — subsoil water reservoir",
    "evapotrans_mm": "Evapotranspiration {value:.2f} mm — atmospheric drying demand on vegetation",
    # Vegetation / satellite
    "ndvi": "NDVI {value:.3f} — vegetation greenness; low values mean dry cured fuels",
    "snow_cover": "Snow cover {value:.0f} — presence of snow suppresses ignition",
    "lai": "Leaf Area Index {value:.2f} — canopy density and live fuel load",
    # Topography / infrastructure
    "elevation_m": "Elevation {value:.0f} m — affects temperature, humidity, and fuel type",
    "slope_deg": "Slope {value:.1f}° — steeper slopes accelerate uphill fire spread",
    "aspect_deg": "Aspect {value:.0f}° — south-facing slopes receive more sun and dry faster",
    "hillshade": "Hillshade {value:.0f} — solar exposure index; lower means more shade",
    "distance_to_road_km": (
        "Distance to road {value:.1f} km — access for suppression and ignition source proximity"
    ),
    # Temporal
    "doy_sin": "Day-of-year (sine component) — seasonal position in fire year",
    "doy_cos": "Day-of-year (cosine component) — seasonal position in fire year",
    # Lightning
    "lightning_24h": "Lightning strikes in last 24 h ({value:.1f}) — direct ignition risk",
    "lightning_72h": "Lightning strikes in last 72 h ({value:.1f}) — recent ignition potential",
}

# Level-specific preamble phrases for generate_summary()
_LEVEL_PREAMBLES = {
    "VERY_LOW": "Risk is very low. Conditions are generally safe.",
    "LOW": "Risk is low. Conditions are relatively benign.",
    "MODERATE": "Risk is moderate. Conditions warrant attention.",
    "HIGH": "Risk is high. Dangerous fire weather conditions are present.",
    "VERY_HIGH": "Risk is very high. Extreme caution is advised.",
    "EXTREME": "Risk is extreme. Life-threatening fire conditions are possible.",
}

# Direction-qualified contribution phrases
_CONTRIBUTION_PHRASES = {
    "increasing": "is pushing risk higher",
    "decreasing": "is pulling risk lower",
}


class ExplainabilityService:
    """Wraps shap.TreeExplainer to provide per-cell SHAP-based explainability.

    Parameters
    ----------
    model:
        An XGBoost Booster instance.  Pass None to create a no-op service
        that returns empty/None results gracefully.
    feature_names:
        Ordered list of feature names matching the columns of X.
    """

    def __init__(self, model: Any, feature_names: list[str]):
        self.feature_names = list(feature_names)
        self._model = model
        self._shap_explainer = None

        if model is None:
            return

        try:
            import shap

            self._shap_explainer = shap.TreeExplainer(model)
            logger.info("ExplainabilityService: TreeExplainer initialized")
        except Exception as e:
            logger.warning("ExplainabilityService: TreeExplainer init failed: %s", e)
            self._shap_explainer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray | None:
        """Compute SHAP values for all cells in X.

        Parameters
        ----------
        X:
            Feature matrix of shape [n_cells, n_features].

        Returns
        -------
        np.ndarray of shape [n_cells, n_features], or None if SHAP is
        unavailable (no model, init failure, or computation error).
        """
        if self._shap_explainer is None:
            return None

        try:
            shap_matrix = self._shap_explainer.shap_values(X)
            # For regression models shap_values returns [n, p] directly.
            # For binary classifiers it may return a list; take index 1.
            if isinstance(shap_matrix, list):
                shap_matrix = shap_matrix[1]
            return np.asarray(shap_matrix, dtype=np.float64)
        except Exception as e:
            logger.warning("compute_shap_values failed: %s", e)
            return None

    def get_drivers(
        self,
        feature_values: dict[str, float],
        shap_values: np.ndarray | None = None,
        top_n: int = 5,
    ) -> list[dict]:
        """Return the top-N SHAP-based risk drivers for a single cell.

        Parameters
        ----------
        feature_values:
            Mapping of feature_name → observed value for this cell.
        shap_values:
            1-D array of SHAP contributions [n_features] for this cell.
            If None and the explainer is available, a single-row prediction
            is attempted; otherwise returns [].
        top_n:
            Number of top drivers to return (ranked by |SHAP|).

        Returns
        -------
        List of driver dicts, each with keys:
            feature, contribution, value, direction, description
        """
        if self._shap_explainer is None:
            return []

        if shap_values is None:
            # Attempt to compute from feature_values
            try:
                X_row = np.array(
                    [feature_values.get(f, 0.0) for f in self.feature_names],
                    dtype=np.float32,
                ).reshape(1, -1)
                computed = self.compute_shap_values(X_row)
                if computed is None:
                    return []
                shap_values = computed[0]
            except Exception as e:
                logger.warning("get_drivers: SHAP computation failed: %s", e)
                return []

        shap_values = np.asarray(shap_values, dtype=np.float64)
        if shap_values.ndim != 1 or len(shap_values) != len(self.feature_names):
            logger.warning(
                "get_drivers: shap_values shape mismatch: got %s, expected (%d,)",
                shap_values.shape,
                len(self.feature_names),
            )
            return []

        # Sort by absolute contribution (descending)
        order = np.argsort(np.abs(shap_values))[::-1]
        top_indices = order[:top_n]

        drivers = []
        for idx in top_indices:
            feat_name = self.feature_names[idx]
            contrib = float(shap_values[idx])
            val = feature_values.get(feat_name, 0.0)
            direction = "increasing" if contrib > 0 else "decreasing"

            # Format description template
            tmpl = FEATURE_DESCRIPTIONS.get(feat_name, f"{feat_name} = {{value:.4f}}")
            try:
                description = tmpl.format(value=float(val))
            except Exception:
                description = tmpl

            drivers.append(
                {
                    "feature": feat_name,
                    "contribution": round(contrib, 6),
                    "value": float(val),
                    "direction": direction,
                    "description": description,
                }
            )

        return drivers

    def generate_summary(self, drivers: list[dict], level: str) -> str:
        """Compose a human-readable risk explanation from the top drivers.

        Parameters
        ----------
        drivers:
            List of driver dicts as returned by get_drivers().
        level:
            Danger level string (e.g. "HIGH", "EXTREME").

        Returns
        -------
        A plain-English paragraph summarising the risk conditions.
        """
        preamble = _LEVEL_PREAMBLES.get(level.upper(), "Risk conditions are present.")

        if not drivers:
            return preamble + " Insufficient data to identify specific drivers."

        # Pick up to 3 most significant drivers for the narrative
        top = drivers[:3]

        lines = [preamble, "Key contributing factors:"]
        for i, d in enumerate(top, start=1):
            direction_phrase = _CONTRIBUTION_PHRASES.get(d["direction"], d["direction"])
            lines.append(f"  {i}. {d['description']} — {direction_phrase}.")

        # Append a note about remaining drivers if there are more
        if len(drivers) > 3:
            rest = [d["feature"].replace("_", " ") for d in drivers[3:]]
            lines.append(f"Additional factors: {', '.join(rest)}.")

        return " ".join(lines)
