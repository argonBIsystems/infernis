"""Fire Behaviour Prediction (FBP) service wrapper.

Wraps cffdrs_py's fire_behaviour_prediction() to compute per-cell FBP outputs:
rate of spread, head fire intensity, crown fraction burned, fire type, and
flame length.

Reference: Van Wagner (1993), Byram (1959), CFFDRS Technical Report.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# Non-fuel types that produce zero spread by definition
_NON_FUEL_TYPES = {"NF", "WA"}

# Fire type thresholds based on Crown Fraction Burned (CFB)
_CFB_ACTIVE_CROWN = 0.9
_CFB_INTERMITTENT_CROWN = 0.1

# Byram's flame length equation: L = 0.0775 * HFI^0.46
# HFI in kW/m, L in metres
_BYRAM_A = 0.0775
_BYRAM_B = 0.46

# Zeros dict returned for non-fuel or error cases
_ZERO_RESULT: dict = {
    "rate_of_spread_mpm": 0.0,
    "head_fire_intensity_kwm": 0.0,
    "fire_type": "surface",
    "crown_fraction_burned": 0.0,
    "flame_length_m": 0.0,
}


def _classify_fire_type(cfb: float) -> str:
    """Classify fire type from Crown Fraction Burned.

    Parameters
    ----------
    cfb:
        Crown fraction burned in [0, 1].

    Returns
    -------
    One of "active_crown", "intermittent_crown", or "surface".
    """
    if cfb >= _CFB_ACTIVE_CROWN:
        return "active_crown"
    elif cfb > _CFB_INTERMITTENT_CROWN:
        return "intermittent_crown"
    return "surface"


def _flame_length_m(hfi_kwm: float) -> float:
    """Compute flame length from head fire intensity via Byram's equation.

    L = 0.0775 * HFI^0.46 (Byram 1959, as used in CFFDRS).

    Parameters
    ----------
    hfi_kwm:
        Head fire intensity in kW/m.

    Returns
    -------
    Flame length in metres. Returns 0.0 for non-positive HFI.
    """
    if hfi_kwm <= 0.0:
        return 0.0
    return _BYRAM_A * math.pow(hfi_kwm, _BYRAM_B)


def compute_fire_behaviour(
    fuel_type: str,
    ffmc: float,
    bui: float,
    wind_speed: float,
    wind_direction: float,
    slope: float,
    aspect: float,
    latitude: float,
    longitude: float,
    elevation: float,
    month: int,
    day: int,
) -> dict:
    """Compute fire behaviour for a single grid cell using cffdrs FBP.

    Parameters
    ----------
    fuel_type:
        CFFDRS fuel type string (e.g. "C3", "D1", "NF", "WA").
    ffmc:
        Fine Fuel Moisture Code.
    bui:
        Buildup Index.
    wind_speed:
        Wind speed in km/h.
    wind_direction:
        Wind direction in degrees (meteorological convention, 0=N, 90=E).
    slope:
        Terrain slope in degrees (0 = flat).
    aspect:
        Terrain aspect in degrees (0=N, 90=E).
    latitude:
        Cell latitude in decimal degrees.
    longitude:
        Cell longitude in decimal degrees.
    elevation:
        Cell elevation in metres above sea level.
    month:
        Calendar month (1-12); used to compute Julian day for foliar moisture.
    day:
        Calendar day of month (1-31); used to compute Julian day.

    Returns
    -------
    dict with keys:
        rate_of_spread_mpm: float — head fire rate of spread (m/min)
        head_fire_intensity_kwm: float — head fire intensity (kW/m)
        fire_type: str — "surface", "intermittent_crown", or "active_crown"
        crown_fraction_burned: float — fraction of crown burned [0, 1]
        flame_length_m: float — Byram flame length (m)
    """
    # Non-fuel: no computation needed
    fuel_upper = fuel_type.upper() if fuel_type else ""
    if fuel_upper in _NON_FUEL_TYPES:
        return dict(_ZERO_RESULT)

    try:
        from cffdrs.fire_behaviour_prediction import fire_behaviour_prediction
        from cffdrs.models import FBPInput

        # Julian day of year (approximate: days in each month up to current)
        _DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        dj = float(sum(_DAYS_IN_MONTH[: month - 1]) + day)

        fbp_input = FBPInput(
            fuel_type=fuel_upper,
            ffmc=float(ffmc),
            bui=float(bui),
            ws=float(wind_speed),
            wd=float(wind_direction),
            gs=float(slope),
            aspect=float(aspect),
            lat=float(latitude),
            lon=float(longitude),
            elv=float(elevation),
            dj=dj,
        )

        result = fire_behaviour_prediction(fbp_input, output="Primary")

        ros = float(result.ros)
        hfi = float(result.hfi)
        cfb = float(result.cfb)

        return {
            "rate_of_spread_mpm": round(ros, 4),
            "head_fire_intensity_kwm": round(hfi, 2),
            "fire_type": _classify_fire_type(cfb),
            "crown_fraction_burned": round(cfb, 4),
            "flame_length_m": round(_flame_length_m(hfi), 2),
        }

    except Exception as exc:
        logger.warning(
            "FBP computation failed for fuel_type=%s: %s — returning zeros",
            fuel_type,
            exc,
        )
        return dict(_ZERO_RESULT)
