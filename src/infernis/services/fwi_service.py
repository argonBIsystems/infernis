from __future__ import annotations

import numpy as np
import pandas as pd


class FWIService:
    """Computes Canadian Forest Fire Weather Index System components.

    Uses the standard CFFDRS equations. FWI codes are cumulative -
    FFMC, DMC, DC carry forward daily.

    Standard startup values (beginning of fire season):
        FFMC=85.0, DMC=6.0, DC=15.0
    """

    DEFAULT_FFMC = 85.0
    DEFAULT_DMC = 6.0
    DEFAULT_DC = 15.0

    # Day-length adjustment factors for DMC (by month, 1-indexed)
    _EL = [0, 6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
    # Day-length factors for DC
    _FL = [0, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]

    def compute_daily(
        self,
        temp: float,
        rh: float,
        wind: float,
        precip: float,
        month: int,
        prev_ffmc: float | None = None,
        prev_dmc: float | None = None,
        prev_dc: float | None = None,
    ) -> dict[str, float]:
        if prev_ffmc is None:
            prev_ffmc = self.DEFAULT_FFMC
        if prev_dmc is None:
            prev_dmc = self.DEFAULT_DMC
        if prev_dc is None:
            prev_dc = self.DEFAULT_DC

        ffmc = self._calc_ffmc(temp, rh, wind, precip, prev_ffmc)
        dmc = self._calc_dmc(temp, rh, precip, prev_dmc, month)
        dc = self._calc_dc(temp, precip, prev_dc, month)
        isi = self._calc_isi(wind, ffmc)
        bui = self._calc_bui(dmc, dc)
        fwi = self._calc_fwi(isi, bui)

        return {
            "ffmc": round(ffmc, 1),
            "dmc": round(dmc, 1),
            "dc": round(dc, 1),
            "isi": round(isi, 1),
            "bui": round(bui, 1),
            "fwi": round(fwi, 1),
        }

    def compute_season(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Compute FWI for a sequence of days. Columns: temp, rh, wind, precip, month."""
        results = []
        prev_ffmc = self.DEFAULT_FFMC
        prev_dmc = self.DEFAULT_DMC
        prev_dc = self.DEFAULT_DC

        for _, row in weather_df.iterrows():
            result = self.compute_daily(
                temp=row["temp"],
                rh=row["rh"],
                wind=row["wind"],
                precip=row["precip"],
                month=int(row["month"]),
                prev_ffmc=prev_ffmc,
                prev_dmc=prev_dmc,
                prev_dc=prev_dc,
            )
            results.append(result)
            prev_ffmc = result["ffmc"]
            prev_dmc = result["dmc"]
            prev_dc = result["dc"]

        return pd.DataFrame(results)

    def _calc_ffmc(self, temp, rh, wind, precip, prev_ffmc):
        mo = 147.2 * (101.0 - prev_ffmc) / (59.5 + prev_ffmc)

        if precip > 0.5:
            rf = precip - 0.5
            if mo <= 150.0:
                mr = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
            else:
                mr = (
                    mo
                    + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
                    + 0.0015 * (mo - 150.0) ** 2 * rf**0.5
                )
            mo = min(mr, 250.0)

        ed = (
            0.942 * rh**0.679
            + 11.0 * np.exp((rh - 100.0) / 10.0)
            + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
        )
        ew = (
            0.618 * rh**0.753
            + 10.0 * np.exp((rh - 100.0) / 10.0)
            + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
        )

        if mo > ed:
            ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * wind**0.5 * (
                1.0 - (rh / 100.0) ** 8
            )
            kd = ko * 0.581 * np.exp(0.0365 * temp)
            m = ed + (mo - ed) * 10.0 ** (-kd)
        elif mo < ew:
            k1 = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * wind**0.5 * (
                1.0 - ((100.0 - rh) / 100.0) ** 8
            )
            kw = k1 * 0.581 * np.exp(0.0365 * temp)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo

        ffmc = 59.5 * (250.0 - m) / (147.2 + m)
        return max(0.0, min(101.0, ffmc))

    def _calc_dmc(self, temp, rh, precip, prev_dmc, month):
        if temp < -1.1:
            temp = -1.1

        el = self._EL[month]

        if precip > 1.5:
            rk = 1.894 * (precip - 1.5)
            mo = 20.0 + np.exp(5.6348 - prev_dmc / 43.43)
            if prev_dmc <= 33.0:
                b = 100.0 / (0.5 + 0.3 * prev_dmc)
            elif prev_dmc <= 65.0:
                b = 14.0 - 1.3 * np.log(prev_dmc)
            else:
                b = 6.2 * np.log(prev_dmc) - 17.2
            mr = mo + 1000.0 * rk / (48.77 + b * rk)
            pr = 244.72 - 43.43 * np.log(mr - 20.0)
            prev_dmc = max(0.0, pr)

        dl = el
        k = 1.894 * (temp + 1.1) * (100.0 - rh) * dl * 1e-6
        dmc = prev_dmc + 100.0 * k
        return max(0.0, dmc)

    def _calc_dc(self, temp, precip, prev_dc, month):
        if temp < -2.8:
            temp = -2.8

        fl = self._FL[month]

        if precip > 2.8:
            rd = 0.83 * precip - 1.27
            qo = 800.0 * np.exp(-prev_dc / 400.0)
            qr = qo + 3.937 * rd
            dr = 400.0 * np.log(800.0 / qr)
            prev_dc = max(0.0, dr)

        v = 0.36 * (temp + 2.8) + fl
        v = max(0.0, v)
        dc = prev_dc + 0.5 * v
        return max(0.0, dc)

    def _calc_isi(self, wind, ffmc):
        m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        fw = np.exp(0.05039 * wind)
        ff = 91.9 * np.exp(-0.1386 * m) * (1.0 + m**5.31 / (4.93e7))
        isi = 0.208 * fw * ff
        return isi

    def _calc_bui(self, dmc, dc):
        if dmc <= 0.4 * dc:
            bui = 0.8 * dmc * dc / (dmc + 0.4 * dc) if (dmc + 0.4 * dc) > 0 else 0.0
        else:
            bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
        return max(0.0, bui)

    def _calc_fwi(self, isi, bui):
        if bui <= 80.0:
            bb = 0.1 * isi * (0.626 * bui**0.809 + 2.0)
        else:
            bb = 0.1 * isi * (1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)))

        if bb <= 1.0:
            fwi = bb
        else:
            fwi = np.exp(2.72 * (0.434 * np.log(bb)) ** 0.647)
        return fwi

    # ------------------------------------------------------------------
    # Vectorized methods — process all cells at once using numpy arrays
    # ------------------------------------------------------------------

    def compute_daily_vec(
        self,
        temp: np.ndarray,
        rh: np.ndarray,
        wind: np.ndarray,
        precip: np.ndarray,
        month: int,
        prev_ffmc: np.ndarray,
        prev_dmc: np.ndarray,
        prev_dc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized FWI computation for all cells in one day.

        All inputs are 1-D arrays of shape [n_cells].
        Returns (ffmc, dmc, dc, isi, bui, fwi) as 1-D arrays.
        """
        ffmc = self._vec_ffmc(temp, rh, wind, precip, prev_ffmc)
        dmc = self._vec_dmc(temp, rh, precip, prev_dmc, month)
        dc = self._vec_dc(temp, precip, prev_dc, month)
        isi = self._vec_isi(wind, ffmc)
        bui = self._vec_bui(dmc, dc)
        fwi = self._vec_fwi(isi, bui)
        return ffmc, dmc, dc, isi, bui, fwi

    def _vec_ffmc(self, temp, rh, wind, precip, prev_ffmc):
        mo = 147.2 * (101.0 - prev_ffmc) / (59.5 + prev_ffmc)

        # Rain effect
        wet = precip > 0.5
        if np.any(wet):
            rf = np.where(wet, precip - 0.5, 0.0)
            rf_safe = np.maximum(rf, 1e-10)
            base_mr = mo + 42.5 * rf * np.exp(-100.0 / np.maximum(251.0 - mo, 1.0)) * (
                1.0 - np.exp(-6.93 / rf_safe)
            )
            extra = 0.0015 * np.maximum(mo - 150.0, 0.0) ** 2 * np.sqrt(rf_safe)
            mr = np.where(mo > 150.0, base_mr + extra, base_mr)
            mo = np.where(wet, np.minimum(mr, 250.0), mo)

        ed = (
            0.942 * np.power(rh, 0.679)
            + 11.0 * np.exp((rh - 100.0) / 10.0)
            + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
        )
        ew = (
            0.618 * np.power(rh, 0.753)
            + 10.0 * np.exp((rh - 100.0) / 10.0)
            + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
        )

        rh100 = rh / 100.0
        rh100_inv = (100.0 - rh) / 100.0

        ko = 0.424 * (1.0 - np.power(rh100, 1.7)) + 0.0694 * np.sqrt(wind) * (
            1.0 - np.power(rh100, 8)
        )
        kd = ko * 0.581 * np.exp(0.0365 * temp)
        m_dry = ed + (mo - ed) * np.power(10.0, -kd)

        k1 = 0.424 * (1.0 - np.power(rh100_inv, 1.7)) + 0.0694 * np.sqrt(wind) * (
            1.0 - np.power(rh100_inv, 8)
        )
        kw = k1 * 0.581 * np.exp(0.0365 * temp)
        m_wet = ew - (ew - mo) * np.power(10.0, -kw)

        m = np.where(mo > ed, m_dry, np.where(mo < ew, m_wet, mo))

        ffmc = 59.5 * (250.0 - m) / (147.2 + m)
        return np.clip(ffmc, 0.0, 101.0)

    def _vec_dmc(self, temp, rh, precip, prev_dmc, month):
        temp = np.maximum(temp, -1.1)
        el = self._EL[month]
        dmc = prev_dmc.copy()

        # Rain effect
        wet = precip > 1.5
        if np.any(wet):
            rk = 1.894 * (precip - 1.5)
            mo = 20.0 + np.exp(5.6348 - dmc / 43.43)
            b = np.where(
                dmc <= 33.0,
                100.0 / (0.5 + 0.3 * dmc),
                np.where(
                    dmc <= 65.0,
                    14.0 - 1.3 * np.log(np.maximum(dmc, 1e-10)),
                    6.2 * np.log(np.maximum(dmc, 1e-10)) - 17.2,
                ),
            )
            mr = mo + 1000.0 * rk / (48.77 + b * rk)
            pr = 244.72 - 43.43 * np.log(np.maximum(mr - 20.0, 1e-10))
            dmc = np.where(wet, np.maximum(pr, 0.0), dmc)

        k = 1.894 * (temp + 1.1) * (100.0 - rh) * el * 1e-6
        dmc = dmc + 100.0 * k
        return np.maximum(dmc, 0.0)

    def _vec_dc(self, temp, precip, prev_dc, month):
        temp = np.maximum(temp, -2.8)
        fl = self._FL[month]
        dc = prev_dc.copy()

        wet = precip > 2.8
        if np.any(wet):
            rd = 0.83 * precip - 1.27
            qo = 800.0 * np.exp(-dc / 400.0)
            qr = qo + 3.937 * rd
            dr = 400.0 * np.log(800.0 / np.maximum(qr, 1e-10))
            dc = np.where(wet, np.maximum(dr, 0.0), dc)

        v = np.maximum(0.36 * (temp + 2.8) + fl, 0.0)
        dc = dc + 0.5 * v
        return np.maximum(dc, 0.0)

    def _vec_isi(self, wind, ffmc):
        m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        fw = np.exp(0.05039 * wind)
        ff = 91.9 * np.exp(-0.1386 * m) * (1.0 + np.power(m, 5.31) / 4.93e7)
        return 0.208 * fw * ff

    def _vec_bui(self, dmc, dc):
        denom = dmc + 0.4 * dc
        safe_denom = np.maximum(denom, 1e-10)
        bui_low = 0.8 * dmc * dc / safe_denom
        bui_high = dmc - (1.0 - 0.8 * dc / safe_denom) * (0.92 + np.power(0.0114 * dmc, 1.7))
        bui = np.where(dmc <= 0.4 * dc, bui_low, bui_high)
        return np.maximum(bui, 0.0)

    def _vec_fwi(self, isi, bui):
        bb_low = 0.1 * isi * (0.626 * np.power(bui, 0.809) + 2.0)
        bb_high = 0.1 * isi * (1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)))
        bb = np.where(bui <= 80.0, bb_low, bb_high)
        fwi = np.where(
            bb <= 1.0, bb, np.exp(2.72 * np.power(0.434 * np.log(np.maximum(bb, 1e-10)), 0.647))
        )
        return fwi
