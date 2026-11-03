"""Conjunction screening and probability of collision.

Pipeline (the same shape operational screening uses, at TLE fidelity):

1. Apogee/perigee filter: drop objects whose altitude band can't overlap the asset's.
2. Coarse sampling: propagate everything on a fixed grid, keep local minima of range.
3. Refinement: Brent minimisation on exact SGP4 states to get time of closest approach (TCA).
4. Probability of collision: 2-D Foster integral in the encounter plane.

MOID is an asteroid-vs-planet concept for heliocentric orbits that don't share a time base; for
Earth orbiters with known epochs the right quantities are TCA, miss distance and Pc.

TLEs carry no covariance. Pc here uses an assumed per-object position uncertainty (radial,
in-track, cross-track sigmas) that grows with propagation time from epoch, which is the standard
workaround when only GP data is available. Treat Pc as a ranking signal, not an operational number.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import erf
from sgp4.api import WGS72, Satrec

from ..ingest.celestrak import TLE
from .propagate import Propagator, jday_array, orbit_shell


@dataclass
class Conjunction:
    primary: TLE
    secondary: TLE
    tca: datetime
    miss_km: float
    rel_speed_kms: float
    r_primary: np.ndarray
    r_secondary: np.ndarray
    v_rel: np.ndarray
    pc: float = 0.0
    impact_angle_deg: float = field(default=0.0)
    cov_source: str = "assumed"
    v_primary: np.ndarray | None = None
    v_secondary: np.ndarray | None = None
    hbr_km: float = 0.0055
    jd_tca: float = 0.0

    def as_dict(self) -> dict:
        return {
            "primary": self.primary.name,
            "primary_id": self.primary.norad_id,
            "secondary": self.secondary.name,
            "secondary_id": self.secondary.norad_id,
            "secondary_is_debris": self.secondary.is_debris,
            "tca_utc": self.tca.isoformat(),
            "miss_km": round(self.miss_km, 4),
            "rel_speed_kms": round(self.rel_speed_kms, 3),
            "pc": self.pc,
            "cov_source": self.cov_source,
        }


# 1-sigma position uncertainty (km) at epoch and growth per day, RIC frame.
# Rough magnitudes from published TLE accuracy studies (in-track error dominates and grows fastest).
SIGMA_EPOCH = {"payload": (0.1, 0.5, 0.15), "debris": (0.2, 1.0, 0.3)}
SIGMA_GROWTH_PER_DAY = {"payload": (0.05, 1.0, 0.05), "debris": (0.1, 2.0, 0.1)}


def _ric_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rows are radial, in-track, cross-track unit vectors."""
    rh = r / np.linalg.norm(r)
    ch = np.cross(r, v)
    ch /= np.linalg.norm(ch)
    ih = np.cross(ch, rh)
    return np.vstack([rh, ih, ch])


def position_covariance(tle: TLE, r: np.ndarray, v: np.ndarray, days_from_epoch: float) -> np.ndarray:
    kind = "debris" if tle.is_debris else "payload"
    s0 = np.asarray(SIGMA_EPOCH[kind])
    g = np.asarray(SIGMA_GROWTH_PER_DAY[kind])
    sig = s0 + g * abs(days_from_epoch)
    basis = _ric_basis(r, v)
    return basis.T @ np.diag(sig**2) @ basis


def collision_probability(miss_vec: np.ndarray, v_rel: np.ndarray, cov: np.ndarray, hbr_km: float) -> float:
    """Foster-style 2-D Pc: integrate the Gaussian over the hard-body disk in the encounter plane.

    ``miss_vec`` is secondary minus primary position at TCA, ``cov`` the combined 3x3 position
    covariance, ``hbr_km`` the combined hard-body radius.
    """
    z = v_rel / np.linalg.norm(v_rel)
    x = miss_vec - np.dot(miss_vec, z) * z
    if np.linalg.norm(x) < 1e-12:
        x = np.cross(z, [1.0, 0.0, 0.0])
        if np.linalg.norm(x) < 1e-9:
            x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    P = np.vstack([x, y])
    m = P @ miss_vec
    C = P @ cov @ P.T
    w, U = np.linalg.eigh(C)
    m = U.T @ m
    sx, sy = np.sqrt(np.maximum(w, 1e-18))
    mx, my = m
    R = hbr_km

    def integrand(u: float) -> float:
        h = np.sqrt(max(R * R - u * u, 0.0))
        gx = np.exp(-0.5 * ((u - mx) / sx) ** 2) / (np.sqrt(2 * np.pi) * sx)
        return gx * 0.5 * (erf((h - my) / (np.sqrt(2) * sy)) + erf((h + my) / (np.sqrt(2) * sy)))

    pc, _ = quad(integrand, -R, R, limit=200, points=[np.clip(mx, -R, R)])
    return float(min(max(pc, 0.0), 1.0))


def _epoch(sat) -> float:
    return sat.jdsatepoch + sat.jdsatepochF


def screen(
    asset: TLE,
    catalog: list[TLE],
    start: datetime | None = None,
    days: float = 3.0,
    step_s: float = 60.0,
    threshold_km: float = 10.0,
    shell_pad_km: float = 25.0,
    asset_radius_m: float = 5.0,
    object_radius_m: float = 0.5,
    chunk: int = 360,  # time steps per propagation batch
    block: int = 1000,  # candidates per coarse-pass block
    docked_km: float = 5.0,
    covariance_provider=None,
) -> list[Conjunction]:
    """``covariance_provider(tle, r, v, when) -> 3x3 | None`` supplies a realistic position
    covariance (e.g. ``orbit.covariance.spacebook_provider``); objects it can't cover fall back to
    the assumed RIC sigmas, and each event records which source it used."""
    """All approaches of ``catalog`` objects to ``asset`` closer than ``threshold_km`` in the window."""
    start = start or datetime.now(timezone.utc).replace(microsecond=0)
    p0, a0 = orbit_shell(asset)
    candidates = []
    for t in catalog:
        if t.norad_id == asset.norad_id:
            continue
        try:
            p, a = orbit_shell(t)
        except Exception:
            continue
        if p - shell_pad_km <= a0 and a + shell_pad_km >= p0:
            candidates.append(t)
    if not candidates:
        return []

    prim = Propagator([asset])
    sec = Propagator(candidates)
    secs = np.arange(0.0, days * 86400.0 + step_s, step_s)
    jd, fr = jday_array(start, secs)
    # A local minimum on the grid brackets the true minimum within +-step. Between samples the
    # range can drop by at most v_rel*step/2, so anything sampled under this gate may hide a close pass.
    gate = threshold_km + 16.0 * step_s / 2

    # The asset's track is shared by every block.
    rp_all = np.concatenate([prim.states(jd[i:i + chunk], fr[i:i + chunk])[0]
                             for i in range(0, len(secs), chunk)], axis=1)
    idx_obj_l, idx_t_l = [], []
    # Coarse pass in blocks of candidates, and of time within a block, so memory stays at
    # ~block * n_steps * 4 B however crowded the asset's shell is (Starlink shells exceed 10k objects).
    for b0 in range(0, len(candidates), block):
        blk = Propagator(candidates[b0:b0 + block])
        rng = np.empty((len(blk), len(secs)), dtype=np.float32)
        for i in range(0, len(secs), chunk):
            sl = slice(i, i + chunk)
            rs, _, _ = blk.states(jd[sl], fr[sl])
            rng[:, sl] = np.linalg.norm(rs - rp_all[:, sl], axis=-1)
        rng[~np.isfinite(rng)] = np.inf

        # Docked vehicles and station modules share the asset's orbit; they are part of it, not threats.
        rng[np.median(rng[:, ::10], axis=1) < docked_km] = np.inf

        # Interior samples only (views, no copies); strict on the left so a flat stretch yields one minimum.
        mid = rng[:, 1:-1]
        o, t = np.nonzero((mid < rng[:, :-2]) & (mid <= rng[:, 2:]) & (mid < gate))
        idx_obj_l.append(o + b0)
        idx_t_l.append(t + 1)
    idx_obj = np.concatenate(idx_obj_l) if idx_obj_l else np.array([], dtype=int)
    idx_t = np.concatenate(idx_t_l) if idx_t_l else np.array([], dtype=int)

    psat = prim.sats[0]
    out: list[Conjunction] = []
    hbr = (asset_radius_m + object_radius_m) / 1000.0
    for k, ti in zip(idx_obj, idx_t):
        ssat = sec.sats[k]
        t0 = secs[ti]

        def dist(ts: float, ssat=ssat) -> float:
            f = fr[0] + ts / 86400.0
            r1, _ = Propagator.state_one(psat, jd[0], f)
            r2, _ = Propagator.state_one(ssat, jd[0], f)
            d = np.linalg.norm(r2 - r1)
            return d if np.isfinite(d) else 1e9

        res = minimize_scalar(dist, bounds=(max(t0 - step_s, 0), t0 + step_s), method="bounded",
                              options={"xatol": 1e-3})
        if res.fun > threshold_km:
            continue
        f = fr[0] + res.x / 86400.0
        r1, v1 = Propagator.state_one(psat, jd[0], f)
        r2, v2 = Propagator.state_one(ssat, jd[0], f)
        vrel = v2 - v1
        jd_tca = jd[0] + f
        # Angle between relative velocity and the asset's ram direction, used as impact obliquity.
        cosang = abs(np.dot(-vrel, v1)) / (np.linalg.norm(vrel) * np.linalg.norm(v1))
        c = Conjunction(
            primary=asset,
            secondary=candidates[k],
            tca=start + timedelta(seconds=float(res.x)),
            miss_km=float(res.fun),
            rel_speed_kms=float(np.linalg.norm(vrel)),
            r_primary=r1,
            r_secondary=r2,
            v_rel=vrel,
            impact_angle_deg=float(np.rad2deg(np.arccos(np.clip(cosang, -1, 1)))),
            v_primary=v1,
            v_secondary=v2,
            hbr_km=hbr,
            jd_tca=float(jd_tca),
        )
        c.pc, c.cov_source = _score(c, covariance_provider)
        out.append(c)
    out.sort(key=lambda c: (-c.pc, c.miss_km))
    return out


def _score(c: Conjunction, provider=None) -> tuple[float, str]:
    covs, sources = [], []
    for tle, r, v in ((c.primary, c.r_primary, c.v_primary), (c.secondary, c.r_secondary, c.v_secondary)):
        cov = provider(tle, r, v, c.tca) if provider else None
        sources.append("spacebook" if cov is not None else "assumed")
        if cov is None:
            sat = Satrec.twoline2rv(tle.line1, tle.line2, WGS72)
            cov = position_covariance(tle, r, v, c.jd_tca - _epoch(sat))
        covs.append(cov)
    pc = collision_probability(c.r_secondary - c.r_primary, c.v_rel, covs[0] + covs[1], c.hbr_km)
    return pc, ("+".join(sources) if len(set(sources)) > 1 else sources[0])


def rescore(conjunctions: list[Conjunction], provider=None) -> list[Conjunction]:
    """Copies of ``conjunctions`` with Pc recomputed under another covariance source.

    Geometry (TCA, miss vector, relative velocity) is unchanged, so differences are purely due to
    the covariance; this is how the assumed-vs-realistic covariance comparison is made."""
    out = []
    for c in conjunctions:
        n = replace(c)
        n.pc, n.cov_source = _score(n, provider)
        out.append(n)
    return out
