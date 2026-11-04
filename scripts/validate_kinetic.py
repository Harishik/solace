"""Check the kinetic-gas conjunction rate against SGP4 screening, with and without the
constellation correction (solace.risk.shells: same-constellation pairs are not random encounters).

For a random sample of LEO objects (40 Starlink, 40 others: payloads, rocket bodies and debris
between 400 and 1,200 km) we count every approach closer than 10 km over one day with the SGP4
screener, and compare with the kinetic-gas expectation for that object's orbit.

Run from the repo root:  .venv/bin/python scripts/validate_kinetic.py
Writes reports/kinetic_validation.json.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from solace.ingest.celestrak import fetch_catalog
from solace.orbit.conjunction import screen
from solace.risk.shells import build_grid, constellation_index, elements, object_conjunction_rate

OUT = Path("reports/kinetic_validation.json")
THRESH = 10.0
START = datetime(2026, 11, 4, tzinfo=timezone.utc)
_CAT = None


def _events(i):
    """(miss km, same constellation) for every approach under THRESH in one day."""
    global _CAT
    if _CAT is None:
        _CAT = fetch_catalog()
    g = constellation_index(_CAT[i].name)
    return [(c.miss_km, g >= 0 and constellation_index(c.secondary.name) == g)
            for c in screen(_CAT[i], _CAT, start=START, days=1.0, threshold_km=THRESH)]


def main():
    cat = fetch_catalog()
    grid = build_grid(cat, when=START)
    rng = np.random.default_rng(7)
    rows = []
    for i, t in enumerate(cat):
        a, e, _, _ = elements(t)
        hp = a * (1 - e) - 6378.137
        if 400 < hp < 1200 and e < 0.02:
            rows.append((i, constellation_index(t.name), a, e, hp))
    star = [r for r in rows if r[1] == 0]
    other = [r for r in rows if r[1] < 0]
    sample = [star[k] for k in rng.choice(len(star), 40, replace=False)] + \
             [other[k] for k in rng.choice(len(other), 40, replace=False)]
    with ProcessPoolExecutor(4) as ex:
        events = list(ex.map(_events, [r[0] for r in sample]))
    out = []
    for (i, g, a, e, hp), ev in zip(sample, events):
        out.append({"norad": cat[i].norad_id, "name": cat[i].name, "perigee_km": hp, "constellation": g,
                    "events": ev, "raw_10km": object_conjunction_rate(grid, a, e, -1, THRESH),
                    "excl_10km": object_conjunction_rate(grid, a, e, g, THRESH, intra_weight=0.0)})
    summary = {}
    for label, sel in (("starlink", lambda r: r["constellation"] == 0), ("others", lambda r: r["constellation"] < 0)):
        rs = [r for r in out if sel(r)]
        raw, excl = sum(r["raw_10km"] for r in rs), sum(r["excl_10km"] for r in rs)
        summary[label] = {"n": len(rs)}
        for t in (1.0, 2.0, 5.0, 10.0):  # kinetic-gas rates scale with the threshold squared
            ev = [x for r in rs for x in r["events"] if x[0] < t]
            same = sum(1 for x in ev if x[1])
            f = (t / THRESH) ** 2
            summary[label][f"{t:g}km"] = {"actual": len(ev), "same_constellation": same, "raw": raw * f,
                                           "excluding_own": excl * f}
            print(f"{label:8s} <{t:>4g} km  actual {len(ev):5d} ({same} same-constellation)  "
                  f"kinetic raw {raw * f:8.1f} (x{raw * f / max(len(ev), 1):.2f})  "
                  f"excluding own {excl * f:8.1f} (x{excl * f / max(len(ev), 1):.2f})")
    OUT.write_text(json.dumps({"summary": summary, "objects": out}, indent=1))


if __name__ == "__main__":
    main()
