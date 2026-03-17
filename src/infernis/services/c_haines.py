"""Continuous Haines Index (C-Haines) computation.

The Continuous Haines Index measures atmospheric instability and dryness
as a leading indicator for extreme fire behavior and pyroconvection potential.

Reference:
    Mills & McCaw (2010). The Continuous Haines Index: development and
    description. Australasian Journal of Disaster and Trauma Studies.

Formula:
    C-Haines = CA + CB

    Stability term (CA):
        ΔT = T850 - T500   (°C, positive = lapse rate)
        CA = (ΔT - 22.0) / 8.0  (scaled), clipped to [0, 6]

    Moisture term (CB):
        DD850 = T850 - Td850   (dewpoint depression at 850 hPa, °C)
        CB = (DD850 - 1.0) / 2.5  (scaled), clipped to [0, 7]

    C-Haines = clip(CA + CB, 0, 13)

    Interpretation:
        < 4   : low potential for extreme fire behavior
        4–7   : moderate potential
        > 8   : high potential — pyroconvection / plume-driven fire possible
        > 10  : very high potential (AFAC threshold for elevated risk briefings)
"""

from __future__ import annotations

import numpy as np


def compute_c_haines(
    t850: float | np.ndarray,
    t500: float | np.ndarray,
    td850: float | np.ndarray,
) -> float | np.ndarray:
    """Compute the Continuous Haines Index.

    Parameters
    ----------
    t850 : float or np.ndarray
        Temperature at 850 hPa pressure level (°C).
    t500 : float or np.ndarray
        Temperature at 500 hPa pressure level (°C).
    td850 : float or np.ndarray
        Dewpoint temperature at 850 hPa pressure level (°C).

    Returns
    -------
    float or np.ndarray
        C-Haines value(s) in the range [0, 13].
        Scalar input → scalar output.
        Array input → array output with same shape.

    Notes
    -----
    The stability term CA is based on the 850–500 hPa lapse rate.
    The moisture term CB is based on the dewpoint depression at 850 hPa.
    Both terms are scaled so that C-Haines naturally ranges 0–13.
    """
    scalar_input = np.isscalar(t850) and np.isscalar(t500) and np.isscalar(td850)

    t850 = np.asarray(t850, dtype=float)
    t500 = np.asarray(t500, dtype=float)
    td850 = np.asarray(td850, dtype=float)

    # Stability term: lapse rate from 850 to 500 hPa
    # Larger ΔT → more unstable → higher CA
    lapse = t850 - t500
    ca = np.clip((lapse - 22.0) / 8.0, 0.0, 6.0)

    # Moisture term: dewpoint depression at 850 hPa
    # Larger depression → drier mid-level air → higher CB
    dd850 = t850 - td850  # always >= 0 by definition (T >= Td)
    cb = np.clip((dd850 - 1.0) / 2.5, 0.0, 7.0)

    c_haines = np.clip(ca + cb, 0.0, 13.0)

    if scalar_input:
        return float(c_haines)
    return c_haines
