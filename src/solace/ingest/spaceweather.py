"""Space-weather drivers: F10.7 and Kp (NOAA SWPC) and solar flares (NASA DONKI).

F10.7 and Kp are what actually move LEO debris: they heat the thermosphere, raising density and
drag, which changes decay rates and therefore shell populations. DONKI flare lists are kept as an
extra event feature.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta

import numpy as np

from .cache import cached_get

F107_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
DONKI_FLR_URL = "https://api.nasa.gov/DONKI/FLR?startDate={start}&endDate={end}&api_key={key}"


def fetch_f107() -> list[tuple[datetime, float]]:
    rows = json.loads(cached_get(F107_URL, "f107.json"))
    out = [(datetime.fromisoformat(r["time_tag"]), float(r["flux"])) for r in rows if r.get("flux")]
    return sorted(out)


def fetch_kp() -> list[tuple[datetime, float]]:
    rows = json.loads(cached_get(KP_URL, "kp.json"))
    out = []
    for r in rows:
        # SWPC has shipped this both as [header, rows...] lists and as a list of dicts.
        if isinstance(r, dict):
            t, kp = r.get("time_tag"), r.get("Kp")
        elif r and r[0] != "time_tag":
            t, kp = r[0], r[1]
        else:
            continue
        try:
            out.append((datetime.fromisoformat(str(t).replace(" ", "T")), float(kp)))
        except (TypeError, ValueError):
            pass
    return sorted(out)


_FLARE_CLASS = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


def flare_peak_flux(class_type: str) -> float:
    """'M2.4' -> 2.4e-5 W/m^2 (GOES 1-8 A peak flux)."""
    try:
        return _FLARE_CLASS[class_type[0].upper()] * float(class_type[1:] or 1)
    except (KeyError, ValueError, IndexError):
        return 0.0


def fetch_flares(days: int = 60, api_key: str | None = None) -> list[tuple[datetime, float]]:
    """(peak time, peak X-ray flux) for the last ``days``. Returns [] if DONKI is unavailable."""
    key = api_key or os.environ.get("NASA_API_KEY", "DEMO_KEY")
    end = date.today()
    start = end - timedelta(days=days)
    url = DONKI_FLR_URL.format(start=start, end=end, key=key)
    try:
        rows = json.loads(cached_get(url, f"donki_flr_{start}_{end}.json") or "[]")
    except Exception as exc:
        print(f"[solace] DONKI unavailable ({exc}); continuing without flare features")
        return []
    out = []
    for r in rows:
        t = r.get("peakTime") or r.get("beginTime")
        if t:
            out.append((datetime.fromisoformat(t.replace("Z", "")), flare_peak_flux(r.get("classType", ""))))
    return sorted(out)


def daily_series(samples: list[tuple[datetime, float]], days: list[date], fill: float) -> np.ndarray:
    """Daily mean on ``days``, forward-filled; ``fill`` before the first sample."""
    by_day: dict[date, list[float]] = {}
    for t, v in samples:
        by_day.setdefault(t.date(), []).append(v)
    out, last = [], fill
    for d in days:
        if d in by_day:
            last = float(np.mean(by_day[d]))
        out.append(last)
    return np.asarray(out)
