"""Per-object re-entry forecasting for debris and rocket bodies (Phase 3c).

Pipeline: weekly Spacebook snapshots -> a table of distinct TLEs per low-perigee object ->
at each issue date, fit a ballistic coefficient from that object's own mean-motion decay
(observed density) -> propagate under a density scenario -> predicted re-entry date, scored
against satcat decay dates. See docs/PHYSICS.md for the reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np

from ..ingest.spacebook import satcat
from . import decay as D

JD_UNIX = 2440587.5
COHORT_TYPES = ("DEBRIS", "ROCKET BODY")


def _epoch_jd(field: str) -> float:
    yy = int(field[:2])
    year = 2000 + yy if yy < 57 else 1900 + yy
    doy = float(field[2:])
    return datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() / 86400 + JD_UNIX + doy - 1


def _bstar(field: str) -> float:
    s = field.strip()
    if not s:
        return 0.0
    mant, exp = s[:-2], s[-2:]
    sign = -1.0 if mant.startswith("-") else 1.0
    mant = mant.lstrip("+-")
    return sign * float("0." + mant.strip()) * 10 ** int(exp)


@dataclass
class TLETable:
    """Distinct TLEs (one row per object and epoch) for low-perigee debris and rocket bodies."""

    norad: np.ndarray
    jd: np.ndarray
    a: np.ndarray  # Brouwer mean semi-major axis, km
    n: np.ndarray  # Kozai mean motion as published, rev/day (for the SGP4 baseline)
    e: np.ndarray
    inc: np.ndarray  # deg
    raan: np.ndarray  # deg
    argp: np.ndarray  # deg
    bstar: np.ndarray
    rb: np.ndarray  # True for rocket bodies

    def save(self, path):
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(**{k: z[k] for k in cls.__dataclass_fields__})


def build_tle_table(files, max_perigee_km: float = 800.0, progress: bool = False) -> TLETable:
    """Parse weekly snapshot files. Mean motion is converted to Brouwer mean a with SGP4's own
    initialisation (the Kozai/Brouwer difference is ~0.1 %, i.e. several km of perigee)."""
    from sgp4.api import Satrec

    cat = satcat()
    keep = {i for i, c in cat.items() if (c.get("objectType") or "") in COHORT_TYPES}
    rb = {i for i in keep if cat[i].get("objectType") == "ROCKET BODY"}
    seen: set[tuple[int, str]] = set()
    rows = []
    mu_km = 398600.4418
    for n, f in enumerate(files):
        lines = open(f).read().splitlines()
        for l1, l2 in zip(lines[0::2], lines[1::2]):
            try:
                sid = int(l1[2:7]) if l1[2:7].strip().isdigit() else None
            except ValueError:
                sid = None
            if sid is None or sid not in keep:
                continue
            key = (sid, l1[18:32])
            if key in seen:
                continue
            mm = float(l2[52:63])
            ecc = float("0." + l2[26:33])
            a_rough = (mu_km / (mm * 2 * np.pi / 86400) ** 2) ** (1 / 3)
            if a_rough * (1 - ecc) - D.RE > max_perigee_km + 20:
                continue
            seen.add(key)
            s = Satrec.twoline2rv(l1, l2)
            rows.append((sid, s.jdsatepoch + s.jdsatepochF, s.a * 6378.135, mm, s.ecco, np.degrees(s.inclo),
                         np.degrees(s.nodeo), np.degrees(s.argpo), _bstar(l1[53:61]), sid in rb))
        if progress and (n % 50 == 0 or n == len(files) - 1):
            print(f"[{n + 1}/{len(files)}] {len(rows)} TLEs", flush=True)
    r = np.array(rows, dtype=object)
    order = np.lexsort((r[:, 1].astype(float), r[:, 0].astype(int)))
    r = r[order]
    return TLETable(norad=r[:, 0].astype(int), jd=r[:, 1].astype(float), a=r[:, 2].astype(float),
                    n=r[:, 3].astype(float), e=r[:, 4].astype(float), inc=r[:, 5].astype(float),
                    raan=r[:, 6].astype(float), argp=r[:, 7].astype(float), bstar=r[:, 8].astype(float),
                    rb=r[:, 9].astype(bool))


def decay_dates() -> dict[int, date]:
    return {i: date.fromisoformat(c["decayDate"][:10]) for i, c in satcat().items() if c.get("decayDate")}


def _jd(d: date) -> float:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() / 86400 + JD_UNIX


@dataclass
class IssueSet:
    """Objects in play at one issue date, with the two TLEs used for the BC fit."""

    issue: date
    norad: np.ndarray
    jd0: np.ndarray
    a0: np.ndarray
    e0: np.ndarray
    ang0: np.ndarray  # (n, 3) inc, raan, argp at jd0, deg
    jd1: np.ndarray
    a1: np.ndarray
    e1: np.ndarray
    inc: np.ndarray
    ang1: np.ndarray  # (n, 3) inc, raan, argp at jd1, deg
    n1: np.ndarray
    bstar: np.ndarray
    rb: np.ndarray
    hp: np.ndarray  # perigee altitude at the latest TLE, km


def issue_set(tt: TLETable, issue: date, max_age_days: float = 10, baseline=(21, 70),
              max_perigee_km: float = 600.0) -> IssueSet:
    """For each object: latest TLE no older than `max_age_days` before `issue` (and not after
    it), plus the earliest TLE within `baseline` days before that one."""
    jd_issue = _jd(issue)
    m = tt.jd <= jd_issue
    ids = tt.norad[m]
    idx = np.nonzero(m)[0]
    # rows are sorted by (norad, jd): last row per id among those <= issue
    last = idx[np.r_[ids[1:] != ids[:-1], True]]
    last = last[jd_issue - tt.jd[last] <= max_age_days]
    hp = tt.a[last] * (1 - tt.e[last]) - D.RE
    last = last[(hp < max_perigee_km) & (hp > D.REENTRY_KM)]
    first = np.full(len(last), -1)
    for k, j in enumerate(last):
        i = j
        while i - 1 >= 0 and tt.norad[i - 1] == tt.norad[j] and tt.jd[j] - tt.jd[i - 1] <= baseline[1]:
            i -= 1
        if tt.jd[j] - tt.jd[i] >= baseline[0]:
            first[k] = i
    ok = first >= 0
    last, first = last[ok], first[ok]
    def ang(i):
        return np.column_stack([tt.inc[i], tt.raan[i], tt.argp[i]])

    return IssueSet(issue, tt.norad[last], tt.jd[first], tt.a[first], tt.e[first], ang(first), tt.jd[last],
                    tt.a[last], tt.e[last], tt.inc[last], ang(last), tt.n[last], tt.bstar[last], tt.rb[last],
                    tt.a[last] * (1 - tt.e[last]) - D.RE)


def physics_forecast(s: IssueSet, observed: D.DensityTable, future: D.DensityTable, horizon: float,
                     bc_override: np.ndarray | None = None, bodies: D.ThirdBodies | None = None) -> dict:
    """Fit B on the observed past, then propagate from the latest TLE under `future` density.
    With `bodies`, Sun and Moon move the perigee (essential for GTO). Returns days from the issue
    date to predicted re-entry (inf if beyond horizon) and the B used."""
    t0 = s.jd0 - _jd(observed.start)
    t1 = s.jd1 - _jd(observed.start)
    if bc_override is None:
        bc = D.fit_bc(s.a0, s.e0, s.ang0[:, 0], t0, s.a1, t1, observed, D.BSTAR_TO_BC * s.bstar,
                      raan=s.ang0[:, 1], argp=s.ang0[:, 2], bodies=bodies)
    else:
        bc = bc_override
    fallback = ~np.isfinite(bc)
    bc = np.where(fallback, np.clip(D.BSTAR_TO_BC * s.bstar, 1e-4, 2.0), bc)
    # The latest TLE can be up to a few days before the issue date: bridge that gap with
    # observed density (it is the past), then switch to the scenario.
    a, e, t, down, ang = D.propagate(s.a1, s.e1, s.ang1[:, 0], bc, t1, np.full(len(t1), observed.offset(s.issue)),
                                     observed, raan=s.ang1[:, 1], argp=s.ang1[:, 2], bodies=bodies)
    t_rel = np.where(down, t - observed.offset(s.issue), 0.0)
    start_f = np.full(len(a), future.offset(s.issue))
    _, _, t2, down2, _ = D.propagate(a, e, ang[:, 0], bc, start_f, start_f + horizon, future,
                                     raan=ang[:, 1], argp=ang[:, 2], bodies=bodies)
    days = np.where(down, t_rel, np.where(down2, t2 - start_f, np.inf))
    return {"days": days, "bc": bc, "bc_fallback": fallback}


def sgp4_forecast(s: IssueSet, horizon: float, step: float = 1.0) -> np.ndarray:
    """Baseline: propagate each latest TLE with SGP4 (B* frozen, static atmosphere) and report
    the first day the orbit drops below the re-entry altitude or SGP4 flags decay."""
    from sgp4.api import Satrec, SatrecArray

    sats = []
    for k in range(len(s.norad)):
        sat = Satrec()
        n_kozai = s.n1[k] * 2 * np.pi / 1440  # rad/min
        sat.sgp4init(0, "i", int(s.norad[k]), s.jd1[k] - 2433281.5, s.bstar[k], 0.0, 0.0, s.e1[k],
                     0.0, np.radians(s.inc[k]), 0.0, n_kozai, 0.0)
        sats.append(sat)
    arr = SatrecArray(sats)
    jd_issue = _jd(s.issue)
    ts = jd_issue + np.arange(0, horizon + step, step)
    out = np.full(len(sats), np.inf)
    for c in range(0, len(ts), 60):
        chunk = ts[c:c + 60]
        err, r, _ = arr.sgp4(np.floor(chunk), chunk - np.floor(chunk))
        alt = np.linalg.norm(r, axis=2) - D.RE
        bad = (err != 0) | (alt < D.REENTRY_KM) | ~np.isfinite(alt)
        hit = bad.any(axis=1)
        first = np.argmax(bad, axis=1)
        new = hit & ~np.isfinite(out)
        out[new] = chunk[first[new]] - jd_issue
    return out
