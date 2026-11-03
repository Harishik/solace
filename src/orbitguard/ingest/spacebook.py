"""COMSPOC Spacebook: free public catalog mirror (no login).

Used for two things: per-object synthetic covariance (``orbit.covariance``) and as a fallback TLE
source when CelesTrak is unavailable. Spacebook's TLE file has no name lines, so names come from
its satellite catalog.
"""

from __future__ import annotations

import json
from functools import lru_cache

from .cache import cached_get
from .celestrak import TLE, parse_tle_text

SATCAT_URL = "https://spacebook.com/api/entity/satcat"
TLE_URL = "https://spacebook.com/api/entity/tle"


@lru_cache(maxsize=1)
def satcat() -> dict[int, dict]:
    """NORAD id -> {id (Spacebook GUID), name, objectType, decayDate}. Other fields are dropped:
    the full records for ~66k objects cost ~100 MB of RAM, which matters on a small host."""
    rows = json.loads(cached_get(SATCAT_URL, "spacebook_satcat.json", max_age_s=24 * 3600, timeout=120))
    keep = ("id", "name", "objectType", "decayDate")
    return {int(r["noradId"]): {k: r.get(k) for k in keep} for r in rows if r.get("noradId") is not None}


def fetch_spacebook_catalog() -> list[TLE]:
    """On-orbit objects with names from the satcat (debris keep their 'DEB' / 'R/B' suffixes)."""
    cat = satcat()
    out = []
    for t in parse_tle_text(cached_get(TLE_URL, "spacebook_tle.txt", max_age_s=6 * 3600, timeout=120)):
        meta = cat.get(t.norad_id)
        if meta is None or meta.get("decayDate"):
            continue
        out.append(TLE(meta.get("name") or str(t.norad_id), t.line1, t.line2))
    return out
