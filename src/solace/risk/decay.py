"""Orbit-averaged drag decay with NRLMSIS density, and ballistic-coefficient fitting from TLEs.

Physics and assumptions are written up in docs/PHYSICS.md. In short: drag acts only along track
with atmospheric co-rotation; the Gauss equations for a and e are averaged over one orbit on
eccentric-anomaly nodes; density is NRLMSIS 2.0 averaged over latitude and local time and
tabulated in log rho against altitude and day. Every object carries its own clock, so one
vectorised loop handles slow and fast decayers together.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

MU = 3.986004418e14  # m^3/s^2
RE = 6378.137  # km, equatorial; TLE perigee altitudes use this convention
OMEGA_E = 7.2921159e-5  # rad/s
REENTRY_KM = 120.0  # perigee altitude at which a near-circular orbit has < 1 day left
DAY = 86400.0
BSTAR_TO_BC = 12.741621  # B* (1/ER) -> C_D A/m (m^2/kg): 2 / rho0, rho0 = 0.15696615 kg/m^2/ER

ALTS = np.arange(80.0, 1505.0, 10.0)
_LATS = np.array([-70.0, -45.0, -20.0, 0.0, 20.0, 45.0, 70.0])
_LONS = np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0])  # at 12 UT, longitude = local time


def msis_log_density(times: np.ndarray, f107: np.ndarray, f107a: np.ndarray, ap: np.ndarray) -> np.ndarray:
    """(len(times), len(ALTS)) natural log of mass density (kg/m^3), averaged over latitude and
    local time. `f107` should be the previous day's flux, as MSIS expects."""
    import pymsis

    aps = np.repeat(np.asarray(ap, float)[:, None], 7, axis=1)
    out = pymsis.calculate(times, _LONS, _LATS, ALTS, f107s=f107, f107as=f107a, aps=aps)
    rho = out[..., pymsis.Variable.MASS_DENSITY].reshape(len(times), -1, len(ALTS))
    # Weight latitudes by cos(lat): equal area, which is what an inclined orbit samples on average.
    w = np.tile(np.cos(np.radians(_LATS)), len(_LONS))
    return np.log(np.einsum("tla,l->ta", rho, w) / w.sum())


@dataclass
class DensityTable:
    """log rho on a (day, altitude) grid starting at `start` with `step` days between rows."""

    start: date
    step: float
    logrho: np.ndarray
    f107a: np.ndarray | None = None  # drivers behind each row, so a density-model correction can be applied
    ap: np.ndarray | None = None

    def __call__(self, t_days: np.ndarray, alt_km: np.ndarray) -> np.ndarray:
        """rho (kg/m^3) for each object's time (days since `start`) and altitudes (..., nodes).
        Linear in time, linear in log rho with altitude; clamped at the table's ends."""
        x = np.clip(t_days / self.step, 0, len(self.logrho) - 1.000001)
        i = x.astype(int)
        f = (x - i)[:, None]
        prof = self.logrho[i] * (1 - f) + self.logrho[i + 1] * f  # (n, n_alt)
        y = np.clip((alt_km - ALTS[0]) / (ALTS[1] - ALTS[0]), 0, len(ALTS) - 1.000001)
        j = y.astype(int)
        g = y - j
        rows = np.arange(len(prof))[:, None]
        return np.exp(prof[rows, j] * (1 - g) + prof[rows, j + 1] * g)

    def offset(self, d: date) -> float:
        return float((d - self.start).days)

    def corrected(self, correction) -> "DensityTable":
        """Same table with log rho shifted by correction(f107a, ap, ALTS) -> (rows, len(ALTS)), the
        modelled log(true / MSIS) density ratio (see density_bias.py)."""
        if correction is None:
            return self
        shift = correction(self.f107a, self.ap, ALTS)
        return DensityTable(self.start, self.step, self.logrho + shift, self.f107a, self.ap)


def observed_table(sw: dict, start: date, end: date) -> DensityTable:
    """Density from the space weather that actually happened (daily)."""
    days = [start + timedelta(i) for i in range((end - start).days + 1)]
    f, fa, ap = _drivers(days, sw)
    times = np.array([np.datetime64(d.isoformat() + "T12:00") for d in days])
    return DensityTable(start, 1.0, msis_log_density(times, f, fa, ap), fa, ap)


def persistence_table(sw: dict, issue: date, days_ahead: int, step: int = 5, trend: bool = False) -> DensityTable:
    """Density a forecaster could have used on `issue`, from data up to that day only.

    Persistence holds F10.7 and Ap at their trailing 81-day means. With `trend`, F10.7 follows
    the straight line fitted to the last 180 days of its 81-day mean (the solar cycle's local
    slope), clipped to the observed range of cycles 24-25. Day-of-year (seasonal) variation
    stays either way, because MSIS knows the calendar."""
    past = [issue - timedelta(i) for i in range(180, 0, -1)]
    f, fa, ap = _drivers(past, sw)
    fbar, apbar = float(fa[-1]), float(np.mean(ap[-81:]))
    n = days_ahead // step + 2
    ahead = step * np.arange(n, dtype=float)
    if trend:
        slope = np.polyfit(np.arange(-180, 0, dtype=float), fa, 1)[0]
        fseries = np.clip(fbar + slope * ahead, 65.0, 220.0)
    else:
        fseries = np.full(n, fbar)
    times = np.array([np.datetime64(issue.isoformat() + "T12:00") + np.timedelta64(int(x), "D") for x in ahead])
    lr = msis_log_density(times, fseries, fseries, np.full(n, apbar))
    return DensityTable(issue, float(step), lr, fseries, np.full(n, apbar))


def analog_table(hist, issue: date, days_ahead: int, step: int = 5, fade_days: float = 730.0) -> DensityTable:
    """Density from the solar-cycle analog forecast (median of earlier cycles read forward from
    today's phase; see solar.py). Uses only data available on `issue`."""
    mf, ma, _ = hist.analog(issue, days_ahead + 2 * step, fade_days)
    f, ap = np.median(mf, 0), np.median(ma, 0)
    n = days_ahead // step + 2
    ahead = step * np.arange(n)
    times = np.array([np.datetime64(issue.isoformat() + "T12:00") + np.timedelta64(int(x), "D") for x in ahead])
    return DensityTable(issue, float(step), msis_log_density(times, f[ahead], f[ahead], ap[ahead]), f[ahead], ap[ahead])


def _drivers(days: list[date], sw: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Previous-day F10.7, trailing 81-day mean (no future data) and daily Ap, gap-filled."""
    f, fa, ap = [], [], []
    hist: list[float] = []
    last = {"f107": 150.0, "ap": 10.0}
    d0 = min(days) - timedelta(82)
    for k in range((max(days) - d0).days + 1):
        d = d0 + timedelta(k)
        r = sw.get(d - timedelta(1))
        if r and np.isfinite(r["f107"]):
            last["f107"] = r["f107"]
        r2 = sw.get(d)
        if r2 and np.isfinite(r2["ap"]):
            last["ap"] = r2["ap"]
        hist.append(last["f107"])
        if d >= min(days):
            f.append(last["f107"])
            fa.append(float(np.mean(hist[-81:])))
            ap.append(last["ap"])
    idx = [(d - min(days)).days for d in days]
    return np.asarray(f)[idx], np.asarray(fa)[idx], np.asarray(ap)[idx]


_NODES = 32
_E = (np.arange(_NODES) + 0.5) * 2 * np.pi / _NODES
_COSE = np.cos(_E)


def rates(a_km: np.ndarray, e: np.ndarray, inc: np.ndarray, bc: np.ndarray, t: np.ndarray,
          table: DensityTable) -> tuple[np.ndarray, np.ndarray]:
    """Orbit-averaged da/dt (km/day) and de/dt (1/day) from drag. `inc` in radians."""
    a = a_km[:, None] * 1e3
    ee = e[:, None]
    r = a * (1 - ee * _COSE)
    v2 = MU * (2 / r - 1 / a)
    v = np.sqrt(v2)
    cosnu = (_COSE - ee) / (1 - ee * _COSE)
    # Co-rotation: relative speed reduced by the atmosphere's along-track motion.
    vrel = v * (1 - (r * OMEGA_E * np.cos(inc)[:, None]) / v)
    rho = table(t, r / 1e3 - RE)
    k = rho * bc[:, None] * vrel**2 / v2  # rho B (v_rel/v)^2
    w = (1 - ee * _COSE) / _NODES  # dM/dE weight -> time average
    dadt = -np.sum(k * a**2 * v**3 / MU * w, axis=1)
    dedt = -np.sum(k * v * (ee + cosnu) * w, axis=1)
    return dadt * DAY / 1e3, dedt * DAY


J2 = 1.08262668e-3
GM_SUN = 1.32712440018e20  # m^3/s^2
GM_MOON = 4.9028e12


@dataclass
class ThirdBodies:
    """Geocentric Sun and Moon positions (m, GCRS, close enough to TEME for secular rates) every
    `step` days from `start`, linearly interpolated."""

    start: date
    step: float
    sun: np.ndarray  # (n, 3)
    moon: np.ndarray

    def __call__(self, t_days: np.ndarray):
        x = np.clip(t_days / self.step, 0, len(self.sun) - 1.000001)
        i = x.astype(int)
        f = (x - i)[:, None]
        return self.sun[i] * (1 - f) + self.sun[i + 1] * f, self.moon[i] * (1 - f) + self.moon[i + 1] * f


def third_bodies(start: date, end: date, step: float = 0.25) -> ThirdBodies:
    """Sun and Moon from the JPL DE440s ephemeris cached under data/skyfield."""
    from skyfield.api import Loader

    from ..ingest.cache import DATA_DIR

    load = Loader(str(Path(DATA_DIR) / "skyfield"))
    eph = load("de440s.bsp")
    ts = load.timescale()
    n = int((end - start).days / step) + 2
    t = ts.utc(start.year, start.month, start.day + np.arange(n) * step)
    earth = eph["earth"]
    sun = earth.at(t).observe(eph["sun"]).position.m.T
    moon = earth.at(t).observe(eph["moon"]).position.m.T
    return ThirdBodies(start, step, sun, moon)


def _to_vectors(e, inc, raan, argp):
    """Eccentricity vector and unit angular-momentum vector from classical angles (rad)."""
    si, ci = np.sin(inc), np.cos(inc)
    h = np.column_stack([si * np.sin(raan), -si * np.cos(raan), ci])
    node = np.column_stack([np.cos(raan), np.sin(raan), np.zeros_like(raan)])
    q = np.cross(h, node)
    peri = np.cos(argp)[:, None] * node + np.sin(argp)[:, None] * q
    return e[:, None] * peri, h


def _to_angles(ev, h):
    inc = np.arccos(np.clip(h[:, 2], -1, 1))
    raan = np.arctan2(h[:, 0], -h[:, 1])
    node = np.column_stack([np.cos(raan), np.sin(raan), np.zeros_like(raan)])
    q = np.cross(h, node)
    argp = np.arctan2(np.sum(ev * q, 1), np.sum(ev * node, 1))
    return inc, raan, argp


def secular_rates(a_km, ev, jv, bodies_pos=()):
    """Orbit-averaged rates (1/day) of the eccentricity vector e and the angular-momentum vector
    j = sqrt(1-e^2) h_hat from J2 and from point-mass third bodies at quadrupole order
    (Milankovitch form, Tremaine, Touma & Kazandjian 2009). `bodies_pos`: (GM, positions (n, 3) m)."""
    a = a_km * 1e3
    n = np.sqrt(MU / a**3)
    jn = np.linalg.norm(jv, axis=1)
    z = np.array([0.0, 0.0, 1.0])
    jz = jv[:, 2]
    # J2: averaged potential C (j^2 - 3 j_z^2) / j^5, C = mu J2 Re^2 / (4 a^3).
    k = 1.5 * n * J2 * (RE * 1e3 / a) ** 2 / 6  # C / L
    jxz = np.cross(jv, z)
    exj, exz = np.cross(ev, jv), np.cross(ev, z)
    dj = (6 * k * jz / jn**5)[:, None] * jxz
    c1 = 2 / jn**5 - 5 * (jn**2 - 3 * jz**2) / jn**7
    de = -k[:, None] * (c1[:, None] * exj - (6 * jz / jn**5)[:, None] * exz)
    for gm, pos in bodies_pos:
        r3 = np.linalg.norm(pos, axis=1)
        d = pos / r3[:, None]
        amp = (1.5 * gm / (r3**3 * n))[:, None]
        ed, jd = np.sum(ev * d, 1)[:, None], np.sum(jv * d, 1)[:, None]
        jxd, exd = np.cross(jv, d), np.cross(ev, d)
        de = de + amp * (5 * ed * jxd - 2 * np.cross(jv, ev) - jd * exd)
        dj = dj + amp * (5 * ed * exd - jd * jxd)
    return de * DAY, dj * DAY


def propagate(a_km, e, inc, bc, t0, t_end, table: DensityTable, max_step: float = 1.0,
              raan=None, argp=None, bodies: ThirdBodies | None = None):
    """Integrate each object from its own start time t0 (days, table time) to t_end, or until
    re-entry. Drag changes a and |e|; J2 always, and Sun and Moon when `bodies` is given, turn
    and stretch the eccentricity and angular-momentum vectors (this is what moves a GTO perigee).
    Angles in degrees. Returns (a, e, t_stop, reentered, angles (n, 3): inc, raan, argp in deg).
    Midpoint (RK2) steps sized so perigee falls by at most ~1/4 of a density scale height."""
    a = np.asarray(a_km, float).copy()
    t = np.asarray(t0, float).copy()
    nobj = len(a)
    e0 = np.clip(np.asarray(e, float), 0, 0.95)
    zeros = np.zeros(nobj)
    ev, h = _to_vectors(e0, np.radians(np.asarray(inc, float)), np.radians(zeros if raan is None else raan),
                        np.radians(zeros if argp is None else argp))
    jv = np.sqrt(1 - e0**2)[:, None] * h
    bc = np.asarray(bc, float)
    t_end = np.broadcast_to(np.asarray(t_end, float), t.shape).copy()
    off = 0.0 if bodies is None else float((table.start - bodies.start).days)

    def deriv(ai, evi, jvi, bi, ti):
        ei = np.linalg.norm(evi, axis=1)
        inc_i = np.arccos(np.clip(jvi[:, 2] / np.linalg.norm(jvi, axis=1), -1, 1))
        da, de_drag = rates(ai, ei, inc_i, bi, ti, table)
        ehat = evi / np.maximum(ei, 1e-12)[:, None]
        hhat = jvi / np.linalg.norm(jvi, axis=1)[:, None]
        pos = ()
        if bodies is not None:
            sun, moon = bodies(ti + off)
            pos = ((GM_SUN, sun), (GM_MOON, moon))
        dev, djv = secular_rates(ai, evi, jvi, pos)
        dev = dev + de_drag[:, None] * ehat
        djv = djv + (-ei * de_drag / np.sqrt(np.maximum(1 - ei**2, 1e-9)))[:, None] * hhat
        return da, dev, djv

    def fix(evi, jvi):
        ei = np.minimum(np.linalg.norm(evi, axis=1), 0.95)
        evi = evi / np.maximum(np.linalg.norm(evi, axis=1), 1e-12)[:, None] * ei[:, None]
        jvi = jvi / np.linalg.norm(jvi, axis=1)[:, None] * np.sqrt(1 - ei**2)[:, None]
        return evi, jvi

    hp_of = lambda a_, ev_: a_ * (1 - np.linalg.norm(ev_, axis=1)) - RE  # noqa: E731
    reentered = hp_of(a, ev) < REENTRY_KM
    done = reentered | (t >= t_end)
    for _ in range(200_000):
        act = ~done
        if not act.any():
            break
        ai, evi, jvi, bi, ti = a[act], ev[act], jv[act], bc[act], t[act]
        ei = np.linalg.norm(evi, axis=1)
        da1, dev1, djv1 = deriv(ai, evi, jvi, bi, ti)
        de_mag = np.sum(dev1 * evi, 1) / np.maximum(ei, 1e-12)
        dhp = np.abs(da1 * (1 - ei) - ai * de_mag) + 1e-9
        scale_h = np.clip(0.11 * (ai * (1 - ei) - RE) - 4, 8, 80)
        dt = np.minimum.reduce([np.full_like(ti, max_step), 0.25 * scale_h / dhp, t_end[act] - ti])
        dt = np.maximum(dt, 1e-4)
        am, (evm, jvm) = ai + 0.5 * dt * da1, fix(evi + 0.5 * dt[:, None] * dev1, jvi + 0.5 * dt[:, None] * djv1)
        da2, dev2, djv2 = deriv(am, evm, jvm, bi, ti + 0.5 * dt)
        a[act] = ai + dt * da2
        ev[act], jv[act] = fix(evi + dt[:, None] * dev2, jvi + dt[:, None] * djv2)
        t[act] = ti + dt
        down = hp_of(a, ev) < REENTRY_KM
        reentered |= down & act
        done |= down | (t >= t_end - 1e-9)
    h = jv / np.linalg.norm(jv, axis=1)[:, None]
    ang = np.degrees(np.column_stack(_to_angles(ev, h)))
    return a, np.linalg.norm(ev, axis=1), t, reentered, ang


def fit_bc(a0, e0, inc, t0, a1, t1, table: DensityTable, bc_guess, iters: int = 3,
           raan=None, argp=None, bodies: ThirdBodies | None = None):
    """Saunders-style ballistic coefficient: the B that makes the modelled drag decay of mean
    semi-major axis between two TLE epochs equal the observed one. Secant iteration from the
    B* guess. Returns B (m^2/kg), NaN where the observed decay is not usable."""
    da_obs = np.asarray(a1, float) - np.asarray(a0, float)
    b = np.where(np.isfinite(bc_guess) & (bc_guess > 0), bc_guess, 0.01).astype(float)
    b_prev, f_prev = None, None
    for _ in range(iters):
        a_end = propagate(a0, e0, inc, b, t0, t1, table, raan=raan, argp=argp, bodies=bodies)[0]
        f = (a_end - a0) - da_obs
        if b_prev is None:
            da_mod = a_end - a0
            nb = np.where(da_mod < 0, b * da_obs / np.minimum(da_mod, -1e-9), b)
        else:
            denom = f - f_prev
            nb = np.where(np.abs(denom) > 1e-9, b - f * (b - b_prev) / np.where(denom == 0, 1, denom), b)
        b_prev, f_prev = b, f
        b = np.clip(nb, 1e-4, 2.0)
    b[~(da_obs < 0)] = np.nan
    return b
