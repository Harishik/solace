"""Historical catalog snapshots and public CDMs from Space-Track.org (free account required).

Credentials are read from the environment (``SPACETRACK_USER`` / ``SPACETRACK_PASS``) and are
never written to disk. Space-Track limits API use to 30 requests/minute and 300/hour and asks
that history be pulled sparingly, so snapshots are cached permanently and requests are throttled.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests

from ..ingest.cache import DATA_DIR
from ..ingest.celestrak import TLE, parse_tle_text

BASE = "https://www.space-track.org"
LOGIN = BASE + "/ajaxauth/login"
GP_HISTORY = (BASE + "/basicspacedata/query/class/gp_history/EPOCH/{d0}--{d1}"
              "/orderby/NORAD_CAT_ID,EPOCH/format/3le")
CDM_PUBLIC = BASE + "/basicspacedata/query/class/cdm_public/TCA/{d0}--{d1}/orderby/TCA/format/json"


class SpaceTrack:
    def __init__(self, user: str | None = None, password: str | None = None, min_interval_s: float = 3.0):
        self.user = user or os.environ.get("SPACETRACK_USER")
        self.password = password or os.environ.get("SPACETRACK_PASS")
        if not (self.user and self.password):
            raise RuntimeError("Set SPACETRACK_USER and SPACETRACK_PASS (free account at space-track.org)")
        self.s = requests.Session()
        self.min_interval = min_interval_s
        self._last = 0.0
        self._logged_in = False

    def _get(self, url: str) -> requests.Response:
        if not self._logged_in:
            r = self.s.post(LOGIN, data={"identity": self.user, "password": self.password}, timeout=60)
            r.raise_for_status()
            if "Failed" in r.text:
                raise RuntimeError("Space-Track login failed")
            self._logged_in = True
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = self.s.get(url, timeout=300)
        self._last = time.time()
        r.raise_for_status()
        return r

    def snapshot(self, day: date) -> list[TLE]:
        """Every element set with an epoch on ``day`` (one per object, the latest that day)."""
        path = DATA_DIR / "history" / f"gp_{day.isoformat()}.3le"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            text = self._get(GP_HISTORY.format(d0=day, d1=day + timedelta(days=1))).text
            path.write_text(text)
        latest: dict[int, TLE] = {}
        for t in parse_tle_text(path.read_text()):
            latest[t.norad_id] = t  # ordered by epoch, so the last one wins
        return list(latest.values())

    def cdms(self, d0: date, d1: date) -> list[dict]:
        import json

        path = DATA_DIR / "history" / f"cdm_{d0}_{d1}.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._get(CDM_PUBLIC.format(d0=d0, d1=d1)).text)
        return json.loads(path.read_text())


def snapshot_days(start: date, end: date, step_days: int = 7) -> list[date]:
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(days=step_days)
    return out


def cached_snapshots() -> list[date]:
    p = DATA_DIR / "history"
    if not p.exists():
        return []
    return sorted(date.fromisoformat(f.stem[3:]) for f in Path(p).glob("gp_*.3le"))


class SpacebookHistory:
    """Historical public-catalog snapshots from COMSPOC Spacebook (no account needed).

    ``/api/entity/tle/{date}`` returns the catalog's element sets as of that date, back to at least
    2005. Names (and so the payload / rocket body / debris split) come from Spacebook's satellite
    catalog; objects that had already decayed by the snapshot date are dropped.
    """

    URL = "https://spacebook.com/api/entity/tle/{day}"

    def __init__(self, min_interval_s: float = 2.0):
        self.min_interval = min_interval_s
        self._last = 0.0

    def snapshot(self, day: date) -> list[TLE]:
        from ..ingest.spacebook import satcat

        path = DATA_DIR / "history" / f"sb_{day.isoformat()}.tle"
        if not path.exists():
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            r = requests.get(self.URL.format(day=day), timeout=300)
            self._last = time.time()
            r.raise_for_status()
            if r.text.count("\n1 ") < 1000:
                raise RuntimeError(f"Spacebook snapshot for {day} looks empty")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text)
        cat = satcat()
        out = []
        for t in parse_tle_text(path.read_text()):
            meta = cat.get(t.norad_id, {})
            decay = meta.get("decayDate")
            if decay and date.fromisoformat(decay[:10]) <= day:
                continue
            out.append(TLE(meta.get("name") or str(t.norad_id), t.line1, t.line2))
        return out


def cached_spacebook_snapshots() -> list[date]:
    p = DATA_DIR / "history"
    if not p.exists():
        return []
    return sorted(date.fromisoformat(f.stem[3:]) for f in Path(p).glob("sb_*.tle"))
