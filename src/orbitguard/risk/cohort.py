"""Debris-cohort decay forecasting: the drag-dominated version of the shell forecast.

For each snapshot k, the cohort is every debris object and rocket body on orbit at k. The target
is where that same cohort (matched by NORAD id) sits h weeks later.

Launches and new fragments never enter the target, so what's left is decay: atmospheric drag,
which space weather drives. This is the setting where space-weather inputs should earn their
place, unlike total shell population, which launches dominate.

Objects can vanish from a snapshot without re-entering (tracking gaps: on 2016-2026 history about
half of the objects missing after 4 weeks sit above 600 km, and many reappear later). So for each
horizon the cohort keeps only objects whose fate is known -- seen again at k + h, or given a decay
date in (k, k + h] by the satellite catalog. That selection is label cleaning and uses the future;
every method (model, physics, persistence) sees exactly the same set, so comparisons stay fair.
Re-entry counts use the catalog decay dates as truth. Inputs are otherwise only what was known at
k: TLEs at k and space weather observed up to k.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime, timezone

import numpy as np

from ..ingest.celestrak import TLE
from ..ingest.spacebook import satcat
from .baseline import sgp4_forecast
from .dataset import GraphSeries, node_features, targets
from .drivers import drivers_for, load_space_weather, shell_density
from .shells import build_grid, kind_index


def is_cohort(t: TLE) -> bool:
    return kind_index(t.name) in (1, 2)


def _decayed_between(cat: dict, norad_id: int, d0: date, d1: date) -> bool:
    dd = (cat.get(norad_id) or {}).get("decayDate")
    return bool(dd) and d0 < date.fromisoformat(dd[:10]) <= d1


def weekly_max_ap(days: list[date], sw: dict) -> np.ndarray:
    """Max daily Ap over the 7 days ending on each snapshot day, so short storms between weekly
    snapshots still register."""
    from datetime import timedelta

    out = []
    for d in days:
        vals = [sw[d - timedelta(days=i)]["ap"] for i in range(7) if (d - timedelta(days=i)) in sw]
        vals = [v for v in vals if np.isfinite(v)]
        out.append(max(vals) if vals else np.nan)
    return np.asarray(out)


def build_cohort_series(load, days: list[date], horizons: tuple[int, ...] = (4, 13, 26),
                        progress: bool = False) -> dict[int, GraphSeries]:
    """One GraphSeries per horizon (in snapshot steps). Each snapshot is parsed once; a small
    window of recent snapshots is kept so k + h lookups don't re-read files."""
    days = sorted(days)
    step = (days[1] - days[0]).days
    sw = load_space_weather()
    drv = drivers_for(days, sw)
    ap_week = weekly_max_ap(days, sw)
    H = max(horizons)
    cat = satcat()
    cache: OrderedDict[int, dict[int, TLE]] = OrderedDict()

    def cohort(i: int) -> dict[int, TLE]:
        if i not in cache:
            cache[i] = {t.norad_id: t for t in load(days[i]) if is_cohort(t)}
            while len(cache) > H + 2:
                cache.popitem(last=False)
        return cache[i]

    parts = {h: {"X": [], "Y": [], "prior": [], "pers": []} for h in horizons}
    reentries = {h: {"actual": [], "physics": []} for h in horizons}
    for k, d in enumerate(days):
        noon = datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc)
        now = cohort(k)
        rho = shell_density(*drv[k], when=np.datetime64(d.isoformat() + "T12:00"))
        for h in horizons:
            P = parts[h]
            if k + h < len(days):
                later = cohort(k + h)
                dk = days[k + h]
                decayed = {i for i in now if i not in later and _decayed_between(cat, i, d, dk)}
                known = [t for i, t in now.items() if i in later or i in decayed]
                survivors = [later[i] for i in now if i in later]
            else:
                known, survivors, decayed = list(now.values()), None, set()
            g = build_grid(known, when=noon)
            pr = sgp4_forecast(known, noon, h * step)
            P["X"].append(node_features(g, drv[k], rho, pr))
            P["prior"].append(np.log1p(pr.total).ravel())
            P["pers"].append(np.log1p(g.total).ravel())
            if survivors is not None:
                actual = build_grid(survivors, when=datetime(dk.year, dk.month, dk.day, 12, tzinfo=timezone.utc))
                P["Y"].append(targets(actual))
                reentries[h]["actual"].append(len(decayed))
                reentries[h]["physics"].append(float(g.total.sum() - pr.total.sum()))
            else:
                P["Y"].append(np.full((g.total.size, 2), np.nan))
                reentries[h]["actual"].append(np.nan)
                reentries[h]["physics"].append(np.nan)
        if progress and (k % 25 == 0 or k == len(days) - 1):
            print(f"[{k + 1}/{len(days)}] {d}: cohort {len(now)}", flush=True)

    out = {}
    for h in horizons:
        P = parts[h]
        gs = GraphSeries(days=days, X=np.stack(P["X"]), Y=np.stack(P["Y"]), prior=np.stack(P["prior"]),
                         persistence=np.stack(P["pers"]), ap=ap_week, horizon=h)
        gs.reentries_actual = np.asarray(reentries[h]["actual"], dtype=float)
        gs.reentries_physics = np.asarray(reentries[h]["physics"], dtype=float)
        out[h] = gs
    return out
