"""Online adaptive conformal windows for re-entry dates (docs/PHYSICS.md §8.4).

At each issue date the window around a forecast is set by empirical quantiles of
log(actual / forecast) among earlier forecasts whose re-entry had already happened by that date
(no look-ahead). Quantiles are taken per group: predicted lead time x perigee band. The
miscoverage level alpha moves by gamma x (target miss - observed miss) as outcomes arrive
(adaptive conformal inference; Gibbs and Candes 2021), widening the windows after a run of misses.
"""

from __future__ import annotations

import numpy as np

LEAD_EDGES = np.array([0, 30, 90, 180, 365, np.inf])


def groups(pred: np.ndarray, hp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lead = np.clip(np.searchsorted(LEAD_EDGES, np.minimum(pred, 1e6), side="right") - 1, 0, len(LEAD_EDGES) - 2)
    return lead * 3 + (hp >= 300) + (hp >= 450), lead


def online_windows(day: np.ndarray, act: np.ndarray, pred: np.ndarray, hp: np.ndarray, target: float = 0.8,
                   gamma: float = 0.05, min_cal: int = 100):
    """Windows (lo, hi) for every row, plus the final alpha. `day` is the issue date as an ordinal;
    `act` the actual days to re-entry (inf if none)."""
    res = np.log(act / np.maximum(pred, 0.5))
    ok = np.isfinite(res) & (act > 0) & (act <= 730)
    grp, lead = groups(pred, hp)
    lo = np.full(len(act), np.nan)
    hi = lo.copy()
    alpha = 1 - target
    seen = np.zeros(len(act), bool)
    for d in np.unique(day):
        new = ok & ~seen & (day + act <= d) & np.isfinite(lo)
        if new.any():
            miss = 1 - np.mean((act[new] >= lo[new]) & (act[new] <= hi[new]))
            alpha = float(np.clip(alpha + gamma * ((1 - target) - miss), 0.01, 0.5))
        seen |= new
        cal = ok & (day + act <= d)
        m = day == d
        for g in np.unique(grp[m]):
            c = cal & (grp == g)
            if c.sum() < min_cal:
                c = cal & (lead == g // 3)
            if c.sum() < min_cal:
                continue
            ql, qh = np.quantile(res[c], (alpha / 2, 1 - alpha / 2))
            k = m & (grp == g) & np.isfinite(pred)
            lo[k] = pred[k] * np.exp(ql)
            hi[k] = pred[k] * np.exp(qh)
    return lo, hi, alpha


def quantile_table(act: np.ndarray, pred: np.ndarray, hp: np.ndarray, alpha: float, min_cal: int = 100) -> dict:
    """Group -> (lower, upper) log-ratio quantiles from all resolved forecasts, for issuing new windows."""
    res = np.log(act / np.maximum(pred, 0.5))
    ok = np.isfinite(res) & (act > 0) & (act <= 730)
    grp, lead = groups(pred, hp)
    out = {}
    for g in range(3 * (len(LEAD_EDGES) - 1)):
        c = ok & (grp == g)
        if c.sum() < min_cal:
            c = ok & (lead == g // 3)
        if c.sum() >= min_cal:
            out[g] = tuple(float(q) for q in np.quantile(res[c], (alpha / 2, 1 - alpha / 2)))
    return out


def event_probability(day: np.ndarray, act: np.ndarray, pred: np.ndarray, hp: np.ndarray, horizon: float,
                      resolve_days: int = 730, min_cal: int = 100) -> np.ndarray:
    """P(re-entry within `horizon` days) for every forecast, with no look-ahead.

    Calibration uses earlier forecasts that are fully resolved (issued at least `resolve_days`
    before). A finite forecast gets the empirical distribution of log(actual / forecast) among
    resolved finite forecasts in its lead x perigee group (objects still up count as +inf). A
    forecast of "no re-entry within the model horizon" (inf) gets the observed event rate among
    resolved forecasts of the same kind in its perigee band. NaN while calibration data are short."""
    fin = np.isfinite(pred)
    scale = np.where(fin, np.maximum(pred, 0.5), 1.0)
    res = np.where(np.isfinite(act) & (act > 0), np.log(np.maximum(act, 0.5) / scale), np.inf)
    grp, lead = groups(np.where(fin, pred, 1e6), hp)
    band = (hp >= 300).astype(int) + (hp >= 450)
    out = np.full(len(act), np.nan)
    thr = np.log(horizon / np.where(fin, np.maximum(pred, 0.5), 1.0))
    event = act <= horizon
    for d in np.unique(day):
        cal = day + resolve_days <= d
        m = day == d
        if cal.sum() < min_cal:
            continue
        for g in np.unique(grp[m & fin]):
            c = cal & fin & (grp == g)
            if c.sum() < min_cal:
                c = cal & fin & (lead == g // 3)
            if c.sum() < min_cal:
                continue
            r = np.sort(res[c])
            k = m & fin & (grp == g)
            out[k] = np.searchsorted(r, thr[k], side="right") / len(r)
        for bnd in np.unique(band[m & ~fin]):
            c = cal & ~fin & (band == bnd)
            if c.sum() < min_cal:
                c = cal & ~fin
            if c.sum() >= min_cal:
                out[m & ~fin & (band == bnd)] = event[c].mean()
    return out
