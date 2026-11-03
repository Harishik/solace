"""Vectorised SGP4 propagation (TEME frame) and frame helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sgp4.api import WGS72, Satrec, SatrecArray, jday

from ..ingest.celestrak import TLE

MU = 398600.4418  # km^3/s^2
R_EARTH = 6378.137  # km (WGS84 equatorial)
F_WGS84 = 1 / 298.257223563


def jday_array(start: datetime, seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Julian day (integer-ish part, fraction) arrays for ``start + seconds``."""
    start = start.astimezone(timezone.utc) if start.tzinfo else start
    jd0, fr0 = jday(start.year, start.month, start.day, start.hour, start.minute,
                    start.second + start.microsecond * 1e-6)
    fr = fr0 + np.asarray(seconds, dtype=float) / 86400.0
    return np.full_like(fr, jd0), fr


def orbit_shell(tle: TLE) -> tuple[float, float]:
    """(perigee, apogee) altitude in km from the mean elements."""
    sat = Satrec.twoline2rv(tle.line1, tle.line2, WGS72)
    n = sat.no_kozai / 60.0  # rad/s
    a = (MU / n**2) ** (1 / 3)
    return a * (1 - sat.ecco) - R_EARTH, a * (1 + sat.ecco) - R_EARTH


class Propagator:
    """Holds a batch of satellites and propagates them together."""

    def __init__(self, tles: list[TLE]):
        self.tles = list(tles)
        self.sats = [Satrec.twoline2rv(t.line1, t.line2, WGS72) for t in self.tles]
        self._array = SatrecArray(self.sats) if self.sats else None

    def __len__(self) -> int:
        return len(self.sats)

    def states(self, jd: np.ndarray, fr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position/velocity (km, km/s) with shape (n_sats, n_times, 3), plus error codes (n_sats, n_times).

        Non-zero error codes (decayed, eccentricity out of range, ...) come back as NaN states.
        """
        err, r, v = self._array.sgp4(np.atleast_1d(jd), np.atleast_1d(fr))
        bad = err != 0
        r[bad] = np.nan
        v[bad] = np.nan
        return r, v, err

    @staticmethod
    def state_one(sat: Satrec, jd: float, fr: float) -> tuple[np.ndarray, np.ndarray]:
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            return np.full(3, np.nan), np.full(3, np.nan)
        return np.asarray(r), np.asarray(v)


def gmst(jd: np.ndarray, fr: np.ndarray) -> np.ndarray:
    """Greenwich mean sidereal time (rad), IAU-82, adequate for TEME->ECEF plotting."""
    t = ((jd - 2451545.0) + fr) / 36525.0
    sec = 67310.54841 + (876600.0 * 3600 + 8640184.812866) * t + 0.093104 * t**2 - 6.2e-6 * t**3
    return np.mod(np.deg2rad(sec / 240.0), 2 * np.pi)


def teme_to_geodetic(r: np.ndarray, jd: np.ndarray, fr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TEME positions (..., n_times, 3) -> geodetic lat, lon (deg) and altitude (km), WGS84.

    Polar motion is ignored (a few metres), which is invisible on a globe.
    """
    th = gmst(jd, fr)
    c, s = np.cos(th), np.sin(th)
    x = c * r[..., 0] + s * r[..., 1]
    y = -s * r[..., 0] + c * r[..., 1]
    z = r[..., 2]
    lon = np.rad2deg(np.arctan2(y, x))
    p = np.hypot(x, y)
    e2 = F_WGS84 * (2 - F_WGS84)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(4):
        n = R_EARTH / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - e2 * n / (n + alt)))
    n = R_EARTH / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.rad2deg(lat), lon, alt
