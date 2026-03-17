"""FWI validation tests comparing INFERNIS against cffdrs_py canonical implementation.

cffdrs_py is the official NRCan CFFDRS Python package:
    https://github.com/cffdrs/cffdrs_py

All tests use a tolerance of 0.5 per FWI component (rounding differences acceptable).
cffdrs functions require a latitude for DMC and DC day-length adjustment; we use 54.0
(central BC) with lat_adjust=True, which matches the Northern-hemisphere (≥30N) table
that INFERNIS hardcodes internally.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cffdrs", reason="cffdrs package not installed")

import cffdrs  # noqa: E402 — conditional import above guards this

from infernis.services.fwi_service import FWIService  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOL = 0.5  # maximum acceptable per-component absolute difference
LAT = 54.0  # central BC latitude — selects the ell01/fl01 tables in cffdrs (≥30N)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cffdrs_daily(
    temp: float,
    rh: float,
    wind: float,
    precip: float,
    month: int,
    prev_ffmc: float,
    prev_dmc: float,
    prev_dc: float,
) -> dict[str, float]:
    """Run a single-day cffdrs calculation and return a dict matching INFERNIS output."""
    ffmc = cffdrs.fine_fuel_moisture_code(prev_ffmc, temp, rh, wind, precip)
    dmc = cffdrs.duff_moisture_code(prev_dmc, temp, rh, precip, LAT, month)
    dc = cffdrs.drought_code(prev_dc, temp, rh, precip, LAT, month)
    isi = cffdrs.initial_spread_index(ffmc, wind)
    bui = cffdrs.buildup_index(dmc, dc)
    fwi = cffdrs.fire_weather_index(isi, bui)
    return {"ffmc": ffmc, "dmc": dmc, "dc": dc, "isi": isi, "bui": bui, "fwi": fwi}


def assert_within_tolerance(inf: dict[str, float], cf: dict[str, float], label: str) -> None:
    """Assert all six components are within TOL of each other."""
    for key in ("ffmc", "dmc", "dc", "isi", "bui", "fwi"):
        diff = abs(inf[key] - cf[key])
        assert diff <= TOL, (
            f"{label} — {key}: INFERNIS={inf[key]:.3f}, cffdrs={cf[key]:.3f}, "
            f"diff={diff:.3f} exceeds tolerance {TOL}"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def svc() -> FWIService:
    return FWIService()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestFWIValidationAgainstCffdrs:
    """Systematic validation of INFERNIS FWI against the cffdrs canonical implementation."""

    def test_normal_summer_conditions(self, svc):
        """Normal summer fire-weather in BC interior — the baseline case."""
        params = dict(temp=20.0, rh=40.0, wind=12.0, precip=0.0, month=5)
        prev = dict(prev_ffmc=85.0, prev_dmc=6.0, prev_dc=15.0)

        inf = svc.compute_daily(**params, **prev)
        cf = cffdrs_daily(**params, **prev)

        assert_within_tolerance(inf, cf, "normal summer")

    def test_hot_dry_windy_extremes(self, svc):
        """Hot/dry/windy extreme conditions — tests upper-range behaviour."""
        params = dict(temp=35.0, rh=10.0, wind=30.0, precip=0.0, month=7)
        prev = dict(prev_ffmc=92.0, prev_dmc=100.0, prev_dc=400.0)

        inf = svc.compute_daily(**params, **prev)
        cf = cffdrs_daily(**params, **prev)

        assert_within_tolerance(inf, cf, "hot/dry/windy extreme")
        # Sanity: these conditions must produce dangerous FWI
        assert inf["fwi"] > 20.0

    def test_cool_wet_conditions(self, svc):
        """Cool/wet spring conditions — tests lower-range behaviour."""
        params = dict(temp=10.0, rh=80.0, wind=5.0, precip=0.0, month=6)
        prev = dict(prev_ffmc=85.0, prev_dmc=6.0, prev_dc=15.0)

        inf = svc.compute_daily(**params, **prev)
        cf = cffdrs_daily(**params, **prev)

        assert_within_tolerance(inf, cf, "cool/wet conditions")

    def test_light_rain(self, svc):
        """Light rain (2 mm) — tests DMC rain reduction path."""
        params = dict(temp=18.0, rh=60.0, wind=8.0, precip=2.0, month=6)
        prev = dict(prev_ffmc=88.0, prev_dmc=20.0, prev_dc=100.0)

        inf = svc.compute_daily(**params, **prev)
        cf = cffdrs_daily(**params, **prev)

        assert_within_tolerance(inf, cf, "light rain (2 mm)")

    def test_heavy_rain(self, svc):
        """Heavy rain (25 mm) — tests all three rain-reduction paths."""
        params = dict(temp=15.0, rh=90.0, wind=5.0, precip=25.0, month=7)
        prev = dict(prev_ffmc=90.0, prev_dmc=50.0, prev_dc=300.0)

        inf = svc.compute_daily(**params, **prev)
        cf = cffdrs_daily(**params, **prev)

        assert_within_tolerance(inf, cf, "heavy rain (25 mm)")

    def test_cold_start_of_season(self, svc):
        """Cold early-season day — tests temperature floor clamping."""
        params = dict(temp=5.0, rh=60.0, wind=10.0, precip=0.0, month=4)
        prev = dict(prev_ffmc=85.0, prev_dmc=6.0, prev_dc=15.0)

        inf = svc.compute_daily(**params, **prev)
        cf = cffdrs_daily(**params, **prev)

        assert_within_tolerance(inf, cf, "cold start of season")

    def test_near_freezing_temperature(self, svc):
        """Near-freezing temp — exercises the DMC/DC temperature clamp boundary."""
        params = dict(temp=0.0, rh=70.0, wind=5.0, precip=0.0, month=5)
        prev = dict(prev_ffmc=85.0, prev_dmc=6.0, prev_dc=15.0)

        inf = svc.compute_daily(**params, **prev)
        cf = cffdrs_daily(**params, **prev)

        assert_within_tolerance(inf, cf, "near-freezing temperature")

    def test_july_peak_drought(self, svc):
        """July peak drought conditions — high DMC and DC carry-forward."""
        params = dict(temp=30.0, rh=20.0, wind=20.0, precip=0.0, month=7)
        prev = dict(prev_ffmc=90.0, prev_dmc=80.0, prev_dc=350.0)

        inf = svc.compute_daily(**params, **prev)
        cf = cffdrs_daily(**params, **prev)

        assert_within_tolerance(inf, cf, "july peak drought")

    def test_august_high_fire(self, svc):
        """August peak fire weather — tests ISI/BUI/FWI at high severity."""
        params = dict(temp=32.0, rh=15.0, wind=25.0, precip=0.0, month=8)
        prev = dict(prev_ffmc=93.0, prev_dmc=120.0, prev_dc=500.0)

        inf = svc.compute_daily(**params, **prev)
        cf = cffdrs_daily(**params, **prev)

        assert_within_tolerance(inf, cf, "august high fire")

    def test_multiday_carry_forward_5_days(self, svc):
        """Five sequential days with state carry-forward — matches cffdrs day-by-day."""
        weather_sequence = [
            dict(temp=22.0, rh=35.0, wind=10.0, precip=0.0, month=7),
            dict(temp=25.0, rh=30.0, wind=12.0, precip=0.0, month=7),
            dict(temp=28.0, rh=25.0, wind=15.0, precip=0.0, month=7),
            dict(temp=30.0, rh=20.0, wind=18.0, precip=0.0, month=7),
            dict(temp=32.0, rh=18.0, wind=20.0, precip=0.0, month=7),
        ]

        # Starting state
        inf_ffmc, inf_dmc, inf_dc = 85.0, 6.0, 15.0
        cf_ffmc, cf_dmc, cf_dc = 85.0, 6.0, 15.0

        for i, day in enumerate(weather_sequence):
            inf = svc.compute_daily(**day, prev_ffmc=inf_ffmc, prev_dmc=inf_dmc, prev_dc=inf_dc)
            cf = cffdrs_daily(**day, prev_ffmc=cf_ffmc, prev_dmc=cf_dmc, prev_dc=cf_dc)

            assert_within_tolerance(inf, cf, f"multiday carry-forward day {i + 1}")

            # Carry INFERNIS state forward for INFERNIS, cffdrs state for cffdrs
            inf_ffmc, inf_dmc, inf_dc = inf["ffmc"], inf["dmc"], inf["dc"]
            cf_ffmc, cf_dmc, cf_dc = cf["ffmc"], cf["dmc"], cf["dc"]

    def test_multiday_with_rain_event(self, svc):
        """Five sequential days including a rain event on day 3."""
        weather_sequence = [
            dict(temp=26.0, rh=30.0, wind=12.0, precip=0.0, month=7),
            dict(temp=28.0, rh=25.0, wind=15.0, precip=0.0, month=7),
            dict(temp=14.0, rh=85.0, wind=5.0, precip=15.0, month=7),  # rain day
            dict(temp=20.0, rh=50.0, wind=8.0, precip=0.0, month=7),
            dict(temp=24.0, rh=35.0, wind=12.0, precip=0.0, month=7),
        ]

        inf_ffmc, inf_dmc, inf_dc = 88.0, 30.0, 150.0
        cf_ffmc, cf_dmc, cf_dc = 88.0, 30.0, 150.0

        for i, day in enumerate(weather_sequence):
            inf = svc.compute_daily(**day, prev_ffmc=inf_ffmc, prev_dmc=inf_dmc, prev_dc=inf_dc)
            cf = cffdrs_daily(**day, prev_ffmc=cf_ffmc, prev_dmc=cf_dmc, prev_dc=cf_dc)

            assert_within_tolerance(inf, cf, f"multiday-with-rain day {i + 1}")

            inf_ffmc, inf_dmc, inf_dc = inf["ffmc"], inf["dmc"], inf["dc"]
            cf_ffmc, cf_dmc, cf_dc = cf["ffmc"], cf["dmc"], cf["dc"]

    def test_vectorized_matches_scalar(self, svc):
        """Vectorized compute_daily_vec must produce values consistent with scalar compute_daily."""
        import numpy as np

        test_cases = [
            dict(temp=20.0, rh=40.0, wind=12.0, precip=0.0, month=7),
            dict(temp=32.0, rh=15.0, wind=25.0, precip=0.0, month=7),
            dict(temp=15.0, rh=75.0, wind=5.0, precip=8.0, month=7),
        ]

        temps = np.array([c["temp"] for c in test_cases])
        rhs = np.array([c["rh"] for c in test_cases])
        winds = np.array([c["wind"] for c in test_cases])
        precips = np.array([c["precip"] for c in test_cases])

        prev_ffmc_arr = np.full(3, 88.0)
        prev_dmc_arr = np.full(3, 25.0)
        prev_dc_arr = np.full(3, 120.0)

        vec_results = svc.compute_daily_vec(
            temps, rhs, winds, precips, 7, prev_ffmc_arr, prev_dmc_arr, prev_dc_arr
        )
        vec_ffmc, vec_dmc, vec_dc, vec_isi, vec_bui, vec_fwi = vec_results

        for i, case in enumerate(test_cases):
            scalar = svc.compute_daily(
                **case,
                prev_ffmc=88.0,
                prev_dmc=25.0,
                prev_dc=120.0,
            )
            for key, arr in zip(
                ("ffmc", "dmc", "dc", "isi", "bui", "fwi"),
                (vec_ffmc, vec_dmc, vec_dc, vec_isi, vec_bui, vec_fwi),
            ):
                diff = abs(scalar[key] - float(arr[i]))
                assert diff <= TOL, (
                    f"vec vs scalar mismatch case {i} {key}: "
                    f"scalar={scalar[key]:.3f}, vec={float(arr[i]):.3f}, diff={diff:.3f}"
                )
