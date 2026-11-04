"""Orbital-shell population grid: the nodes of the forecasting graph.

Each node is an (altitude band, inclination band) cell. An object contributes to every altitude
band it passes through, weighted by the fraction of its orbit spent there (residence time), so
eccentric orbits are spread correctly instead of being pinned to their mean altitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sgp4.api import WGS72, Satrec

from ..ingest.celestrak import TLE
from ..orbit.propagate import MU, R_EARTH, Propagator, jday_array

ALT_EDGES = np.arange(200.0, 2050.0, 50.0)  # km, 37 bands
INC_EDGES = np.array([0, 30, 45, 55, 65, 75, 85, 95, 105, 180.0])  # deg; dense near SSO/polar
KINDS = ("payload", "rocket_body", "debris")
# Station-kept constellations: members are phased within their own shells, so at collision-relevant
# distances they meet each other far less often than random motion predicts.
CONSTELLATIONS = ("STARLINK", "ONEWEB", "KUIPER", "QIANFAN", "GUOWANG", "IRIDIUM", "ORBCOMM", "GLOBALSTAR")
# Measured weight of same-constellation pairs in the kinetic-gas rate (scripts/validate_kinetic.py:
# 40 Starlink satellites screened for one day, approaches < 1-2 km): 0.05-0.12 of the random rate.
# Pairs with every other object follow kinetic gas theory to within 0.7-1.4x at 1-10 km.
INTRA_WEIGHT = 0.08


def constellation_index(name: str) -> int:
    """Index into CONSTELLATIONS for an operational constellation member, else -1 (debris excluded)."""
    n = name.upper()
    if " DEB" in n or " R/B" in n:
        return -1
    return next((k for k, c in enumerate(CONSTELLATIONS) if n.startswith(c) or c in n.split()), -1)


def kind_index(name: str) -> int:
    n = name.upper()
    if " DEB" in n or n.endswith("DEB"):
        return 2
    if " R/B" in n:
        return 1
    return 0


@dataclass
class ShellGrid:
    counts: np.ndarray  # (n_alt, n_inc, n_kind), residence-weighted object counts
    bstar: np.ndarray  # (n_alt, n_inc), mean B* of residents (drag proxy)
    groups: np.ndarray | None = None  # (n_alt, n_inc, len(CONSTELLATIONS)) constellation members

    @property
    def shape(self):
        return self.counts.shape[:2]

    @property
    def total(self) -> np.ndarray:
        return self.counts.sum(axis=-1)

    def spatial_density(self) -> np.ndarray:
        """Objects per km^3 in each altitude band, split across inclination cells by count."""
        return self.total / band_volume()[:, None]


def band_volume() -> np.ndarray:
    r0, r1 = R_EARTH + ALT_EDGES[:-1], R_EARTH + ALT_EDGES[1:]
    return 4.0 / 3.0 * np.pi * (r1**3 - r0**3)


def elements(tle: TLE) -> tuple[float, float, float, float]:
    """(a km, e, i deg, B*) from mean elements."""
    s = Satrec.twoline2rv(tle.line1, tle.line2, WGS72)
    a = (MU / (s.no_kozai / 60.0) ** 2) ** (1 / 3)
    return a, s.ecco, np.rad2deg(s.inclo), s.bstar


def residence_weights(a: float, e: float, n: int = 72) -> np.ndarray:
    """Fraction of time in each altitude band. Uniform in mean anomaly = uniform in time."""
    M = np.linspace(0, 2 * np.pi, n, endpoint=False)
    E = M.copy()
    for _ in range(20):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    alt = a * (1 - e * np.cos(E)) - R_EARTH
    w, _ = np.histogram(alt, ALT_EDGES)
    return w / n


def build_grid(tles: list[TLE], when: datetime | None = None, n_samples: int = 24,
               window_min: float = 128.0) -> ShellGrid:
    """Grid at ``when`` from SGP4 positions sampled over one orbit (``window_min`` covers a full
    revolution everywhere below ~2000 km). Building every grid -- today's and the physics
    forecast's -- this same way keeps them directly comparable. Re-entered objects drop out.
    """
    when = when or datetime.now(timezone.utc)
    n_alt, n_inc = len(ALT_EDGES) - 1, len(INC_EDGES) - 1
    counts = np.zeros((n_alt, n_inc, len(KINDS)))
    groups = np.zeros((n_alt, n_inc, len(CONSTELLATIONS)))
    bsum = np.zeros((n_alt, n_inc))
    if not tles:
        return ShellGrid(counts=counts, bstar=bsum, groups=groups)
    prop = Propagator(tles)
    jd, fr = jday_array(when, np.linspace(0, window_min * 60, n_samples, endpoint=False))
    r, v, err = prop.states(jd, fr)
    alt = np.linalg.norm(r, axis=-1) - R_EARTH  # (n_obj, n_samples)
    h = np.cross(r[:, 0], v[:, 0])
    inc = np.degrees(np.arccos(np.clip(h[:, 2] / np.linalg.norm(h, axis=-1), -1, 1)))
    ok = (err == 0).all(axis=1) & np.isfinite(alt).all(axis=1) & (alt.min(axis=1) > 100)
    j = np.clip(np.searchsorted(INC_EDGES, inc, side="right") - 1, 0, n_inc - 1)
    a_idx = np.searchsorted(ALT_EDGES, alt, side="right") - 1  # -1 / n_alt mean out of range
    kinds = np.array([kind_index(t.name) for t in tles])
    grp = np.array([constellation_index(t.name) for t in tles])
    bstar = np.array([s.bstar for s in prop.sats])
    for k in np.nonzero(ok)[0]:
        bins = a_idx[k][(a_idx[k] >= 0) & (a_idx[k] < n_alt)]
        if bins.size == 0:
            continue
        w = np.bincount(bins, minlength=n_alt) / n_samples
        counts[:, j[k], kinds[k]] += w
        if grp[k] >= 0:
            groups[:, j[k], grp[k]] += w
        bsum[:, j[k]] += w * bstar[k]
    tot = counts.sum(axis=-1)
    return ShellGrid(counts=counts, bstar=np.divide(bsum, tot, out=np.zeros_like(bsum), where=tot > 0), groups=groups)


def kinetic_conjunction_rate(grid: ShellGrid, threshold_km: float = 1.0, v_rel_kms: float = 10.0,
                             intra_weight: float = 1.0) -> np.ndarray:
    """Expected close approaches (< threshold) per day in each cell from kinetic gas theory:
    rate = 1/2 * N * n * sigma * v_rel, with sigma = pi * threshold^2.

    Inclination cells share the altitude band's volume, so cross-inclination encounters (which
    dominate LEO conjunctions) are counted through the band density. Pairs inside one station-kept
    constellation count with `intra_weight` (1 = random motion, the default for reproducibility of
    the Phase 3 features; INTRA_WEIGHT is the measured value): rate = 1/2 sigma v (N n - (1 - w)
    sum_g N_g n_g).
    """
    vol = band_volume()[:, None]
    n_band = grid.spatial_density().sum(axis=1, keepdims=True)  # objects / km^3 in the band
    pairs = grid.total * n_band
    if intra_weight != 1.0 and grid.groups is not None:
        n_g = grid.groups.sum(axis=1, keepdims=True) / vol[..., None]
        pairs = pairs - (1.0 - intra_weight) * (grid.groups * n_g).sum(axis=-1)
    per_s = 0.5 * np.maximum(pairs, 0.0) * np.pi * threshold_km**2 * v_rel_kms
    return per_s * 86400.0


def object_conjunction_rate(grid: ShellGrid, a: float, e: float, group: int = -1, threshold_km: float = 1.0,
                            v_rel_kms: float = 10.0, intra_weight: float = INTRA_WEIGHT) -> float:
    """Expected close approaches per day for one object on orbit (a, e), kinetic gas theory. A
    constellation member (`group` >= 0) counts its own constellation with `intra_weight`."""
    n = grid.total.sum(axis=1)
    if group >= 0 and grid.groups is not None:
        n = n - (1.0 - intra_weight) * grid.groups[..., group].sum(axis=1)
    w = residence_weights(a, e)
    return float((w * n / band_volume()).sum() * np.pi * threshold_km**2 * v_rel_kms * 86400.0)


def adjacency(n_alt: int | None = None, n_inc: int | None = None) -> np.ndarray:
    """Symmetric-normalised adjacency with self-loops over the (alt, inc) grid.

    Cells are linked to altitude neighbours (decay moves objects downward) and inclination
    neighbours (fragments spread in inclination after breakups).
    """
    n_alt = n_alt or len(ALT_EDGES) - 1
    n_inc = n_inc or len(INC_EDGES) - 1
    N = n_alt * n_inc
    A = np.eye(N)
    idx = lambda i, j: i * n_inc + j  # noqa: E731
    for i in range(n_alt):
        for j in range(n_inc):
            for di, dj in ((1, 0), (0, 1)):
                ii, jj = i + di, j + dj
                if ii < n_alt and jj < n_inc:
                    A[idx(i, j), idx(ii, jj)] = A[idx(ii, jj), idx(i, j)] = 1.0
    d = 1.0 / np.sqrt(A.sum(axis=1))
    return A * d[:, None] * d[None, :]
