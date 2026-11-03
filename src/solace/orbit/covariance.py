"""Realistic position covariance from COMSPOC Spacebook (free, no login).

Spacebook publishes, for every public catalog object, a 12-day STK ephemeris in TEME with a 6x6
"synthetic" covariance every few minutes. The covariance comes from statistics of how that
object's past TLE predictions actually missed, binned by propagation time and orbit position, so
it reflects each object's real TLE error rather than a one-size guess.

Reference: Spacebook synthetic covariance, ESA Space Debris Conference 2025 (SDC9).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np

from ..ingest.cache import cached_get
from ..ingest.spacebook import satcat

COV_URL = "https://spacebook.com/api/entity/synthetic-covariance/{guid}"

# Lower-triangular storage order -> (row, col)
_TRI = [(i, j) for i in range(6) for j in range(i + 1)]


@dataclass
class CovarianceEphemeris:
    norad_id: int
    epoch: datetime
    t: np.ndarray  # seconds from epoch, ephemeris samples
    rv: np.ndarray  # (n, 6) TEME km, km/s
    t_cov: np.ndarray  # seconds from epoch, covariance samples
    cov: np.ndarray  # (m, 6, 6) km, km/s

    def covariance_at(self, when: datetime) -> np.ndarray | None:
        """6x6 covariance at ``when`` (linear interpolation), or None outside the span."""
        s = (when - self.epoch).total_seconds()
        if s < self.t_cov[0] or s > self.t_cov[-1]:
            return None
        k = int(np.clip(np.searchsorted(self.t_cov, s) - 1, 0, len(self.t_cov) - 2))
        f = (s - self.t_cov[k]) / (self.t_cov[k + 1] - self.t_cov[k])
        return (1 - f) * self.cov[k] + f * self.cov[k + 1]

    def position_at(self, when: datetime) -> np.ndarray | None:
        s = (when - self.epoch).total_seconds()
        if s < self.t[0] or s > self.t[-1]:
            return None
        return np.array([np.interp(s, self.t, self.rv[:, i]) for i in range(3)])


_MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                                        "Oct", "Nov", "Dec"], 1)}


def _parse_stk_time(s: str) -> datetime:
    day, mon, year, hms = s.split()
    h, m, sec = hms.split(":")
    return datetime(int(year), _MONTHS[mon], int(day), int(h), int(m), tzinfo=timezone.utc) + timedelta(
        seconds=float(sec))


def parse_stk_ephemeris(text: str, norad_id: int = 0) -> CovarianceEphemeris:
    epoch, block = None, None
    t, rv, tc, cv = [], [], [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("ScenarioEpoch"):
            epoch = _parse_stk_time(line[len("ScenarioEpoch"):].strip())
        elif line in ("EphemerisTimePosVel", "CovarianceTimePosVel"):
            block = line
        elif line.startswith("END") or line.startswith("End"):
            block = None
        elif block:
            vals = line.split()
            if block == "EphemerisTimePosVel" and len(vals) == 7:
                t.append(float(vals[0]))
                rv.append([float(x) for x in vals[1:]])
            elif block == "CovarianceTimePosVel" and len(vals) == 22:
                tc.append(float(vals[0]))
                m = np.zeros((6, 6))
                for (i, j), x in zip(_TRI, vals[1:]):
                    m[i, j] = m[j, i] = float(x)
                cv.append(m)
    if epoch is None or not cv:
        raise ValueError("not an STK ephemeris with covariance")
    return CovarianceEphemeris(norad_id, epoch, np.array(t), np.array(rv), np.array(tc), np.array(cv))


def _guid_map() -> dict[int, str]:
    return {k: v["id"] for k, v in satcat().items()}


@lru_cache(maxsize=64)  # each parsed file is ~2 MB; callers clear it after a batch
def fetch_covariance(norad_id: int) -> CovarianceEphemeris | None:
    guid = _guid_map().get(norad_id)
    if not guid:
        return None
    try:
        text = cached_get(COV_URL.format(guid=guid), f"spacebook_cov_{norad_id}.e", max_age_s=6 * 3600,
                          timeout=90)
        return parse_stk_ephemeris(text, norad_id)
    except Exception as exc:
        print(f"[solace] no Spacebook covariance for {norad_id}: {exc}")
        return None


def spacebook_provider(tle, r, v, when: datetime) -> np.ndarray | None:
    """Covariance provider for ``orbit.conjunction.screen``: 3x3 position block or None."""
    eph = fetch_covariance(tle.norad_id)
    if eph is None:
        return None
    c = eph.covariance_at(when)
    return None if c is None else c[:3, :3]
