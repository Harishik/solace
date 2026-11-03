"""Empirical correction to NRLMSIS 2.0 from observed thermospheric densities.

Satellite densities (CHAMP, GRACE, GRACE-FO accelerometers; Swarm B/C precise orbits;
scripts/fetch_density.py) give delta = log(rho_observed / rho_MSIS) at ~300-530 km, 2000-2026.
A constant delta does not matter for re-entry (the fitted ballistic coefficient absorbs it); what
matters is how delta changes with solar and geomagnetic activity between the fit window and the
forecast window. The bias is not monotonic in solar flux (about -0.55 at solar minimum, ~0 at
moderate activity, ~-0.1 at high activity), so we model it as

    delta = g(F) + s(F) z + b ln(1 + Ap) + satellite offset,   z = (h - 450)/100

with g and s piecewise linear in the 81-day F10.7 (knots KNOTS, lightly smoothed), F clamped to
the range seen in training so the model never extrapolates. Fitted on daily medians only from
observations at least `latency_days` before the issue date. Below the lowest observed altitude
the correction is held (default) or tapered to zero at 120 km (`taper`). See docs/PHYSICS.md §9.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from ..ingest.cache import DATA_DIR

KNOTS = np.array([65.0, 75.0, 90.0, 110.0, 130.0, 150.0, 175.0, 200.0, 240.0])
H_LO, H_HI = 300.0, 540.0
SATS = ("gracefo1", "swarmC", "swarmB", "champ", "grace1")


@dataclass
class DailyBias:
    day: np.ndarray  # ordinal
    sat: np.ndarray
    delta: np.ndarray
    alt: np.ndarray
    f107a: np.ndarray
    ap: np.ndarray


def load_daily(sats=SATS) -> DailyBias:
    rows = []
    for k, sat in enumerate(sats):
        path = Path(DATA_DIR) / "density" / f"{sat}.npz"
        if not path.exists():
            continue
        with np.load(path) as f:
            z = {name: f[name] for name in f.files}  # NpzFile decompresses on every access
        day = np.floor(z["t"] / 86400).astype(int)
        order = np.argsort(day, kind="stable")
        day, cut = np.unique(day[order], return_index=True)
        for d, part in zip(day, np.split(order, cut[1:])):
            lr = z["logratio"][part]
            lr = lr[np.isfinite(lr)]
            if len(lr) < 100:
                continue
            ordinal = datetime.fromtimestamp(int(d) * 86400, timezone.utc).date().toordinal()
            rows.append((ordinal, k, np.median(lr), np.median(z["alt_km"][part]), z["f107a"][part[0]],
                         z["ap"][part[0]]))
    r = np.array(rows, float)
    return DailyBias(r[:, 0].astype(int), r[:, 1].astype(int), r[:, 2], r[:, 3], r[:, 4], r[:, 5])


def _hats(f):
    """Linear-spline (hat) basis on KNOTS, (n, len(KNOTS)); rows sum to 1."""
    f = np.clip(np.atleast_1d(np.asarray(f, float)), KNOTS[0], KNOTS[-1])
    j = np.clip(np.searchsorted(KNOTS, f, side="right") - 1, 0, len(KNOTS) - 2)
    w = (f - KNOTS[j]) / (KNOTS[j + 1] - KNOTS[j])
    out = np.zeros((len(f), len(KNOTS)))
    rows = np.arange(len(f))
    out[rows, j] = 1 - w
    out[rows, j + 1] += w
    return out


@dataclass
class BiasModel:
    g: np.ndarray  # value at each knot
    s: np.ndarray  # altitude slope (per 100 km) at each knot
    b: float  # ln(1 + Ap) coefficient
    f_range: tuple[float, float]
    h_min: float
    taper: bool = False
    n: int = 0

    def __call__(self, f107a, ap, alts):
        """(rows, len(alts)) correction to add to log rho."""
        f = np.clip(np.asarray(f107a, float), *self.f_range)
        H = _hats(f)
        h = np.clip(np.asarray(alts, float), self.h_min, H_HI)[None, :]
        d = (H @ self.g)[:, None] + (H @ self.s)[:, None] * (h - 450.0) / 100.0
        d = d + self.b * np.log1p(np.asarray(ap, float))[:, None]
        if self.taper:
            d = d * np.clip((np.asarray(alts, float)[None, :] - 120.0) / (self.h_min - 120.0), 0.0, 1.0)
        return d


def fit(db: DailyBias, issue: date, latency_days: int = 90, taper: bool = False, min_days: int = 200,
        smooth: float = 2.0) -> BiasModel | None:
    m = db.day <= issue.toordinal() - latency_days
    if m.sum() < min_days:
        return None
    f, z, la = db.f107a[m], (np.clip(db.alt[m], H_LO, H_HI) - 450.0) / 100.0, np.log1p(db.ap[m])
    H = _hats(f)
    sats = np.unique(db.sat[m])
    S = (db.sat[m][:, None] == sats[None, 1:]).astype(float)  # first satellite is the reference
    K = len(KNOTS)
    A = np.column_stack([H, H * z[:, None], la, S])
    # Smoothness: penalise second differences of g and s; weak ridge keeps unobserved knots defined.
    D2 = np.diff(np.eye(K), 2, axis=0)
    P = np.zeros((2 * (K - 2) + A.shape[1], A.shape[1]))
    P[: K - 2, :K] = smooth * D2
    P[K - 2: 2 * (K - 2), K:2 * K] = smooth * D2
    P[2 * (K - 2):, :] = 1e-3 * np.eye(A.shape[1])
    sol, *_ = np.linalg.lstsq(np.vstack([A, P]), np.r_[db.delta[m], np.zeros(len(P))], rcond=None)
    lo, hi = np.percentile(f, [0.5, 99.5])
    return BiasModel(sol[:K], sol[K:2 * K], float(sol[2 * K]), (float(lo), float(hi)),
                     float(max(H_LO, np.percentile(db.alt[m], 1))), taper, int(m.sum()))
