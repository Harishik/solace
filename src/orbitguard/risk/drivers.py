"""Space-weather history and thermospheric density: the exogenous drivers of the forecast."""

from __future__ import annotations

import csv
import io
from datetime import date

import numpy as np

from ..ingest.cache import cached_get
from .shells import ALT_EDGES

SW_ALL_URL = "https://celestrak.org/SpaceData/SW-All.csv"


def load_space_weather() -> dict[date, dict]:
    """Daily F10.7 (observed), its 81-day centred mean, and daily Ap since 1957.

    CelesTrak SW-All also carries predicted values (F10.7_DATA_TYPE 'PRD'/'PRM') beyond today,
    which lets the forecaster run on predicted drivers rather than peeking at observed future ones.
    """
    text = cached_get(SW_ALL_URL, "SW-All.csv", max_age_s=24 * 3600, timeout=120)
    out = {}
    for r in csv.DictReader(io.StringIO(text)):
        try:
            d = date.fromisoformat(r["DATE"])
            f = float(r["F10.7_OBS"])
            f81 = float(r["F10.7_OBS_CENTER81"] or r["F10.7_OBS_LAST81"] or f)
        except (ValueError, KeyError):
            continue
        ap = float(r["AP_AVG"]) if r.get("AP_AVG") else np.nan
        out[d] = {"f107": f, "f107a": f81, "ap": ap, "type": r.get("F10.7_DATA_TYPE", "")}
    return out


def drivers_for(days: list[date], sw: dict[date, dict] | None = None) -> np.ndarray:
    """(len(days), 3) array of F10.7, F10.7 81-day mean, Ap, forward-filling gaps."""
    sw = sw or load_space_weather()
    last = {"f107": 150.0, "f107a": 150.0, "ap": 10.0}
    rows = []
    for d in days:
        rec = sw.get(d)
        if rec:
            last = {k: (rec[k] if np.isfinite(rec[k]) else last[k]) for k in last}
        rows.append([last["f107"], last["f107a"], last["ap"]])
    return np.asarray(rows)


def shell_density(f107: float, f107a: float, ap: float, when: np.datetime64 | None = None) -> np.ndarray:
    """NRLMSIS 2.0 mass density (kg/m^3) at each altitude-band midpoint, zonally averaged.

    Averaging over a coarse lat/lon grid removes the day/night bulge, leaving the global level
    that controls orbit-averaged decay.
    """
    import pymsis

    when = when or np.datetime64("2025-01-01T12:00")
    alts = 0.5 * (ALT_EDGES[:-1] + ALT_EDGES[1:])
    lats = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    lons = np.array([0.0, 90.0, 180.0, 270.0])
    out = pymsis.calculate(when, lons, lats, alts, f107s=[f107], f107as=[f107a], aps=[[ap] * 7])
    rho = out[..., pymsis.Variable.MASS_DENSITY]  # (1, lon, lat, alt)
    return np.nanmean(rho.reshape(-1, len(alts)), axis=0)
