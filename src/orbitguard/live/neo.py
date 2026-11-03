"""Near-Earth asteroids: JPL data, heliocentric Kepler propagation and Earth MOID.

MOID (minimum orbit intersection distance) is the closest the two orbit *curves* ever get,
regardless of where the bodies are on them. It is the standard first screen for asteroid
impact hazard: JPL flags an object as potentially hazardous when MOID <= 0.05 au and H <= 22.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..ingest.cache import cached_get

CAD_URL = ("https://ssd-api.jpl.nasa.gov/cad.api?date-min=now&date-max=%2B{days}"
           "&dist-max={dist}&sort=date&fullname=true&diameter=true")
SENTRY_URL = "https://ssd-api.jpl.nasa.gov/sentry.api"
SBDB_QUERY_URL = ("https://ssd-api.jpl.nasa.gov/sbdb_query.api?"
                  "fields=full_name,pdes,a,e,i,om,w,ma,epoch,H,moid,diameter&sb-group={group}&limit={limit}")

GAUSS_K = 0.01720209895  # au^1.5 / day
AU_KM = 149_597_870.7
LD_AU = 384_400 / AU_KM  # lunar distance in au

# Earth-Moon barycentre mean elements, J2000 ecliptic (Standish, JPL "Keplerian elements for
# approximate positions of the major planets"). Good to ~1e-4 au near the present, ample for MOID.
EARTH = dict(a=1.00000261, e=0.01671123, i=-0.00001531, om=0.0, w=102.93768193, ma=-2.47311027,
             epoch=2451545.0)


@dataclass(frozen=True)
class Orbit:
    name: str
    a: float
    e: float
    i: float  # deg
    om: float  # deg, ascending node
    w: float  # deg, argument of perihelion
    ma: float  # deg, mean anomaly at epoch
    epoch: float  # JD (TDB)
    H: float = float("nan")
    jpl_moid: float = float("nan")
    diameter_km: float = float("nan")

    @property
    def period_days(self) -> float:
        return 2 * np.pi / (GAUSS_K / self.a**1.5)


EARTH_ORBIT = Orbit("Earth", **EARTH)


def _rotation(o: Orbit) -> np.ndarray:
    """Columns map perifocal (P, Q) unit vectors to ecliptic xyz."""
    node, w, i = np.deg2rad([o.om, o.w, o.i])
    cO, sO, cw, sw, ci, si = np.cos(node), np.sin(node), np.cos(w), np.sin(w), np.cos(i), np.sin(i)
    P = np.array([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si])
    Q = np.array([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si])
    return np.column_stack([P, Q])


def position_at_true_anomaly(o: Orbit, nu: np.ndarray) -> np.ndarray:
    nu = np.asarray(nu, dtype=float)
    r = o.a * (1 - o.e**2) / (1 + o.e * np.cos(nu))
    pq = np.stack([r * np.cos(nu), r * np.sin(nu)], axis=-1)
    return pq @ _rotation(o).T


def solve_kepler(M: np.ndarray, e: float | np.ndarray) -> np.ndarray:
    E = np.where(np.asarray(e) > 0.8, np.pi * np.ones_like(M), M)
    for _ in range(50):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - dE
        if np.max(np.abs(dE)) < 1e-12:
            break
    return E


def position(o: Orbit, jd: float | np.ndarray) -> np.ndarray:
    """Heliocentric ecliptic J2000 position (au) at Julian date(s)."""
    n = GAUSS_K / o.a**1.5
    M = np.deg2rad(o.ma) + n * (np.asarray(jd, dtype=float) - o.epoch)
    M = np.mod(M, 2 * np.pi)
    E = solve_kepler(M, o.e)
    nu = 2 * np.arctan2(np.sqrt(1 + o.e) * np.sin(E / 2), np.sqrt(1 - o.e) * np.cos(E / 2))
    return position_at_true_anomaly(o, nu)


def orbit_path(o: Orbit, n: int = 180) -> np.ndarray:
    return position_at_true_anomaly(o, np.linspace(0, 2 * np.pi, n))


def moid(o1: Orbit, o2: Orbit = EARTH_ORBIT, grid: int = 240) -> float:
    """MOID in au: global grid over both true anomalies, then local refinement of the best minima."""
    nu = np.linspace(0, 2 * np.pi, grid, endpoint=False)
    p1 = position_at_true_anomaly(o1, nu)
    p2 = position_at_true_anomaly(o2, nu)
    d = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1)
    # Up to two distinct minima can exist; refine the few best grid cells.
    flat = np.argsort(d, axis=None)[:6]
    best = np.inf

    def f(x):
        return np.linalg.norm(position_at_true_anomaly(o1, x[0]) - position_at_true_anomaly(o2, x[1]))

    for k in flat:
        i, j = np.unravel_index(k, d.shape)
        res = minimize(f, [nu[i], nu[j]], method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 2000})
        best = min(best, res.fun)
    return float(best)


def _f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def fetch_population(group: str = "pha", limit: int = 3000) -> list[Orbit]:
    """Orbital elements for a JPL small-body group (``pha`` = potentially hazardous asteroids)."""
    raw = json.loads(cached_get(SBDB_QUERY_URL.format(group=group, limit=limit), f"sbdb_{group}.json",
                                max_age_s=24 * 3600, timeout=60))
    idx = {k: n for n, k in enumerate(raw["fields"])}
    out = []
    for row in raw.get("data", []):
        try:
            out.append(Orbit(
                name=(row[idx["full_name"]] or row[idx["pdes"]]).strip(),
                a=float(row[idx["a"]]), e=float(row[idx["e"]]), i=float(row[idx["i"]]),
                om=float(row[idx["om"]]), w=float(row[idx["w"]]), ma=float(row[idx["ma"]]),
                epoch=float(row[idx["epoch"]]), H=_f(row[idx["H"]]), jpl_moid=_f(row[idx["moid"]]),
                diameter_km=_f(row[idx["diameter"]]),
            ))
        except (TypeError, ValueError):
            continue
    return [o for o in out if 0 <= o.e < 1]


def fetch_close_approaches(days: int = 60, dist_au: float = 0.05) -> list[dict]:
    raw = json.loads(cached_get(CAD_URL.format(days=days, dist=dist_au), f"cad_{days}_{dist_au}.json",
                                max_age_s=3600))
    fields = raw.get("fields", [])
    out = []
    for row in raw.get("data", []):
        r = dict(zip(fields, row))
        out.append({
            "name": (r.get("fullname") or r["des"]).strip(),
            "des": r["des"],
            "jd": float(r["jd"]),
            "date": r["cd"],
            "dist_au": float(r["dist"]),
            "dist_ld": float(r["dist"]) / LD_AU,
            "v_rel_kms": float(r["v_rel"]),
            "H": _f(r.get("h")),
            "diameter_km": _f(r.get("diameter")),
        })
    return out


def fetch_sentry(top: int = 25) -> list[dict]:
    raw = json.loads(cached_get(SENTRY_URL, "sentry.json", max_age_s=6 * 3600))
    rows = raw.get("data", [])
    rows.sort(key=lambda r: _f(r.get("ps_cum"), -99), reverse=True)
    return [{
        "name": r.get("fullname", r.get("des", "")).strip(),
        "des": r.get("des"),
        "impact_prob": _f(r.get("ip")),
        "palermo_cum": _f(r.get("ps_cum")),
        "palermo_max": _f(r.get("ps_max")),
        "torino_max": int(_f(r.get("ts_max"), 0)),
        "years": r.get("range"),
        "diameter_km": _f(r.get("diameter")),
        "v_inf_kms": _f(r.get("v_inf")),
    } for r in rows[:top]]
