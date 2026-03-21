"""Diurnal FFMC adjustment for sub-daily fire weather accuracy.

Based on CFFDRS Red Book 3rd edition (2018) diurnal FFMC curves.

The daily FFMC is computed from noon weather observations under the standard
CFFDRS protocol. When the pipeline runs at 14:00 PT, afternoon drying has
already pushed FFMC significantly above the daily value — typically 15-20
points higher under hot, dry conditions.

This module applies a diurnal correction that:
  1. Scales a base hourly adjustment factor (derived from the Red Book curves)
  2. Amplifies or reduces that factor according to temperature and humidity
  3. Clamps output to the valid FFMC range [0, 101]

Usage::

    from infernis.services.diurnal_ffmc import adjust_ffmc_diurnal

    # Scalar
    ffmc_14h = adjust_ffmc_diurnal(daily_ffmc=85.0, temp_c=30.0, rh_pct=20.0, hour=14)

    # Vectorized (numpy arrays of shape [n_cells])
    ffmc_14h = adjust_ffmc_diurnal(daily_ffmc, temp_c, rh_pct, hour=14)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Diurnal adjustment factors — one per hour (0-23)
# ---------------------------------------------------------------------------
# These represent the additive FFMC deviation from the daily value under
# *reference* conditions (temp=20 °C, RH=40 %). Derived from the diurnal
# FFMC curves in Van Wagner (1987) and the CFFDRS Red Book (2018), Table 2.
#
# Curve shape:
#   - Night / early morning (00-05): negative, peak around 04:00 (-8)
#   - Mid-morning (06-09):           recovering toward zero
#   - Late morning (10-11):          slight positive
#   - Afternoon (12-15):             positive, peak at 14:00 (+10)
#   - Late afternoon (16-18):        declining
#   - Evening (19-23):               negative, recovering overnight
#
# Reference: CFFDRS Red Book 3rd ed. (2018), §3.4 diurnal FFMC correction;
#            Van Wagner (1987) "Development and structure of the Canadian
#            Forest Fire Weather Index System," p. 48-52.
# ---------------------------------------------------------------------------
_DIURNAL_BASE: tuple[float, ...] = (
    -6.0,  # 00:00  overnight — still drying out from prior evening
    -7.0,  # 01:00
    -7.5,  # 02:00
    -8.0,  # 03:00  nadir: peak overnight humidity recovery
    -8.0,  # 04:00
    -7.5,  # 05:00
    -6.5,  # 06:00  sunrise — RH begins to drop
    -5.0,  # 07:00
    -3.0,  # 08:00
    -1.0,  # 09:00
    1.0,  # 10:00  transitioning through daily reference
    3.5,  # 11:00
    5.5,  # 12:00  approaching afternoon peak
    8.0,  # 13:00
    10.0,  # 14:00  peak afternoon drying (pipeline run time)
    9.5,  # 15:00
    8.0,  # 16:00
    6.0,  # 17:00
    3.5,  # 18:00  evening — drying decelerating
    1.0,  # 19:00
    -1.0,  # 20:00
    -3.0,  # 21:00
    -4.5,  # 22:00
    -5.5,  # 23:00
)

# Pre-converted to a numpy array for vectorized operations
_BASE = np.asarray(_DIURNAL_BASE, dtype=np.float64)

# Reference conditions for which the base factors are defined
_REF_TEMP_C: float = 20.0
_REF_RH_PCT: float = 40.0

# Sensitivity coefficients (empirically tuned to reproduce Red Book magnitudes)
# Each degree above reference adds this fraction of the base adjustment.
_TEMP_SENSITIVITY: float = 0.030  # per °C above 20 °C
# Each percentage point below reference RH adds this fraction.
_RH_SENSITIVITY: float = 0.012  # per % RH below 40 %


def adjust_ffmc_diurnal(
    daily_ffmc: float | np.ndarray,
    temp_c: float | np.ndarray,
    rh_pct: float | np.ndarray,
    hour: int,
) -> float | np.ndarray:
    """Return the diurnally adjusted FFMC for the given hour and conditions.

    Parameters
    ----------
    daily_ffmc:
        The standard daily FFMC value(s) computed by FWIService. Scalar or
        1-D numpy array of shape [n_cells].
    temp_c:
        Air temperature in degrees Celsius at the target hour. Same shape as
        ``daily_ffmc``.
    rh_pct:
        Relative humidity in percent at the target hour. Same shape as
        ``daily_ffmc``.
    hour:
        Local hour of day (0-23, inclusive). Typically 14 for the daily
        pipeline run.

    Returns
    -------
    float | np.ndarray
        Diurnally adjusted FFMC, clamped to [0, 101]. Same type/shape as
        ``daily_ffmc``.
    """
    if not (0 <= hour <= 23):
        raise ValueError(f"hour must be in [0, 23], got {hour}")

    base_delta = _BASE[hour]

    # Condition amplifier:
    #   1.0 at reference conditions (20 °C, 40 % RH)
    #   > 1.0 when hotter and/or drier  → larger afternoon increase
    #   < 1.0 when cooler and/or wetter → smaller adjustment
    temp_factor = 1.0 + _TEMP_SENSITIVITY * (temp_c - _REF_TEMP_C)
    rh_factor = 1.0 + _RH_SENSITIVITY * (_REF_RH_PCT - rh_pct)

    # Clamp amplifier so it never reverses the sign of the base delta
    amplifier = np.maximum(temp_factor * rh_factor, 0.1)

    adjusted = daily_ffmc + base_delta * amplifier
    return np.clip(adjusted, 0.0, 101.0)
