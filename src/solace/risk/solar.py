"""Solar-cycle analog forecast of F10.7 and Ap, using only data available at the issue date.

The 11-year cycle is the dominant, and fairly repeatable, driver of thermospheric density over
months to years. At an issue date we locate the current cycle's minimum, take the phase (days
since that minimum), and read every earlier cycle (since 1957) forward from the same phase. Each
earlier cycle is rescaled so its 81-day flux at that phase matches today's; the rescaling fades
over `fade_days`, because cycle amplitude is only partly predictable from the present level.
The members form an ensemble; their median is the point forecast. See docs/PHYSICS.md section 8.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

FLOOR = 65.0  # sfu, quiet-Sun F10.7 background that does not scale with the cycle


class SolarHistory:
    """Daily observed F10.7 and Ap from the space-weather dict (observed rows only)."""

    def __init__(self, sw: dict, last_obs: date):
        days = sorted(d for d in sw if d <= last_obs)
        self.start = days[0]
        n = (days[-1] - self.start).days + 1
        f = np.full(n, np.nan)
        ap = np.full(n, np.nan)
        for d in days:
            r = sw[d]
            f[(d - self.start).days] = r.get("f107", np.nan)
            ap[(d - self.start).days] = r.get("ap", np.nan)
        self.f107 = _ffill(f, 150.0)
        self.ap = _ffill(ap, 10.0)
        self.f81 = _trailing_mean(self.f107, 81)  # what a forecaster sees on each day
        self.ap81 = _trailing_mean(self.ap, 81)
        self.f81c = _centered_mean(self.f107, 81)  # for reading past cycles forward
        self.ap81c = _centered_mean(self.ap, 81)

    def idx(self, d: date) -> int:
        return (d - self.start).days

    def minima_before(self, d: date) -> list[int]:
        """Cycle minima (indices) known on day d: minima of the 13-month centred mean that are
        complete at least 7 months before d, and at least 8 years apart."""
        cut = self.idx(d) - 213
        s = _centered_mean(self.f107[: max(cut + 200, 0)], 395)[:cut]
        mins: list[int] = []
        for i in np.argsort(s):
            if not np.isfinite(s[i]):
                continue
            if all(abs(i - m) > 8 * 365 for m in mins):
                mins.append(int(i))
        mins = sorted(m for m in mins if 200 < m < cut - 30)
        return mins

    def analog(self, d: date, days_ahead: int, fade_days: float = 365.0):
        """Ensemble of (f107a, ap) trajectories for days 0..days_ahead after d.
        Returns (members_f (k, n), members_ap (k, n), phase_days)."""
        mins = self.minima_before(d)
        i_now = self.idx(d) - 1
        cur_min = mins[-1]
        phase = i_now - cur_min
        level = self.f81[i_now]
        ap_level = self.ap81[i_now]
        ahead = np.arange(days_ahead + 1)
        w = np.exp(-ahead / fade_days)
        mf, ma = [], []
        for m in mins[:-1]:
            j = m + phase + ahead
            if j[-1] >= i_now - 40:  # analog must be fully in the past
                continue
            ref = self.f81[m + phase]
            s = np.clip((level - FLOOR) / max(ref - FLOOR, 5.0), 0.3, 3.0)
            mf.append(FLOOR + (self.f81c[j] - FLOOR) * (1 + (s - 1) * w))
            sa = np.clip(ap_level / max(self.ap81[m + phase], 2.0), 0.3, 3.0)
            ma.append(self.ap81c[j] * (1 + (sa - 1) * w))
        return np.array(mf), np.array(ma), phase


def _ffill(x: np.ndarray, first: float) -> np.ndarray:
    out = x.copy()
    last = first
    for i, v in enumerate(out):
        if np.isfinite(v):
            last = v
        else:
            out[i] = last
    return out


def _trailing_mean(x: np.ndarray, n: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    lo = np.maximum(i - n + 1, 0)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def _centered_mean(x: np.ndarray, n: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    lo = np.maximum(i - n // 2, 0)
    hi = np.minimum(i + n // 2 + 1, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def issue_dates(first: date, last: date, step: int = 28) -> list[date]:
    out, d = [], first
    while d <= last:
        out.append(d)
        d += timedelta(step)
    return out


class MSFCForecasts:
    """NASA MSFC's archived monthly solar-cycle forecasts (scripts/fetch_msfc.py).

    `forecast(d, n)` gives daily F10.7 and Ap for days 0..n after d from the latest forecast
    issued in a month before d's month, so it was public on d. With `anchor` (today's trailing
    81-day F10.7) the forecast is rescaled to match today's level, the rescaling fading over
    `fade_days` as in the analog forecast.
    """

    def __init__(self, path=None):
        from pathlib import Path

        from ..ingest.cache import DATA_DIR

        path = Path(path or Path(DATA_DIR) / "solar" / "msfc_forecasts.npz")
        with np.load(path) as z:
            self.z = {k: z[k] for k in z.files}
        self.issues = np.unique(self.z["issue"])

    def issued_before(self, d: date) -> int:
        cut = date(d.year, d.month, 1).toordinal()
        ok = self.issues[self.issues < cut]
        if not len(ok):
            raise ValueError(f"no MSFC forecast before {d}")
        return int(ok[-1])

    def forecast(self, d: date, days_ahead: int, pct: str = "50", anchor: float | None = None,
                 fade_days: float = 365.0):
        m = self.z["issue"] == self.issued_before(d)
        t = self.z["t"][m]
        # Tabulated times are month starts in decimal years; values are monthly means, so sit mid-month.
        x = np.array([(date(int(y), 1, 1).toordinal() + (y - int(y)) * 365.25 + 15) for y in t])
        days = d.toordinal() + np.arange(days_ahead + 1)
        f = np.interp(days, x, self.z["f" + pct][m])
        ap = np.interp(days, x, self.z["ap" + pct][m])
        if anchor is not None:
            s = np.clip((anchor - FLOOR) / max(f[0] - FLOOR, 5.0), 0.3, 3.0)
            f = FLOOR + (f - FLOOR) * (1 + (s - 1) * np.exp(-np.arange(days_ahead + 1) / fade_days))
        return f, ap
