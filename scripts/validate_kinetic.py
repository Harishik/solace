"""Check the kinetic-gas conjunction rate against SGP4 screening, with and without the
constellation correction (solace.risk.shells: same-constellation pairs are not random encounters).

For a random sample of LEO objects (40 Starlink, 20 OneWeb, 40 others: payloads, rocket bodies and
debris between 400 and 1,300 km) we count every approach closer than 10 km over `--days` days with
the SGP4 screener, and compare with the kinetic-gas expectation for that object's orbit. For each
constellation the same-constellation weight is estimated as (same-constellation approaches) /
(the same-constellation part of the kinetic rate), with a Poisson 95 % interval.

One long screening from a single set of TLEs is not valid for manoeuvring constellations: after a
few days the propagated Starlink positions drift by kilometres and spurious approaches pile up. So
`--weeks N` instead screens one day from each of the last N weekly Spacebook snapshots
(data/history), each with its own fresh TLEs, objects and sample.

Run from the repo root:  .venv/bin/python scripts/validate_kinetic.py [--days 7 | --weeks 7]
Writes reports/kinetic_validation[_<days>d | _<N>wk].json.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from solace.ingest.cache import DATA_DIR
from solace.ingest.celestrak import fetch_catalog, parse_tle_text
from solace.orbit.conjunction import screen
from solace.risk.shells import build_grid, constellation_index, elements, object_conjunction_rate

THRESH = 10.0
START = datetime(2026, 11, 4, tzinfo=timezone.utc)
_CAT = None
DAYS = 1.0


_SNAP = None


def snapshot_catalog(path: Path):
    """A weekly Spacebook snapshot (two-line elements) with names from the CelesTrak SATCAT."""
    import csv

    names = {}
    with open(Path(DATA_DIR) / "raw" / "celestrak_satcat.csv") as f:
        for row in csv.DictReader(f):
            names[int(row["NORAD_CAT_ID"])] = row["OBJECT_NAME"]
    lines = open(path).read().splitlines()
    text = []
    for l1, l2 in zip(lines[0::2], lines[1::2]):
        try:
            n = int(l1[2:7])
        except ValueError:
            continue
        text += [names.get(n, f"OBJECT {n}"), l1, l2]
    return parse_tle_text("\n".join(text))


def _catalog(snap):
    global _CAT, _SNAP
    if _CAT is None or _SNAP != snap:
        _CAT, _SNAP = (fetch_catalog() if snap is None else snapshot_catalog(Path(snap))), snap
    return _CAT


def _events(args):
    """(miss km, same constellation) for every approach under THRESH."""
    i, days, snap, start = args
    cat = _catalog(snap)
    g = constellation_index(cat[i].name)
    return [(c.miss_km, g >= 0 and constellation_index(c.secondary.name) == g)
            for c in screen(cat[i], cat, start=start, days=days, threshold_km=THRESH)]


def _poisson_ci(k):
    from scipy.stats import chi2
    lo = 0.5 * chi2.ppf(0.025, 2 * k) if k > 0 else 0.0
    return lo, 0.5 * chi2.ppf(0.975, 2 * k + 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=1.0)
    ap.add_argument("--weeks", type=int, default=0, help="screen 1 day from each of the last N weekly snapshots")
    args = ap.parse_args()
    if args.weeks:
        out_path = Path(f"reports/kinetic_validation_{args.weeks}wk.json")
        snaps = sorted((Path(DATA_DIR) / "history").glob("sb_*.tle"))[-args.weeks:]
        runs = [(str(f), datetime.fromisoformat(f.stem[3:]).replace(tzinfo=timezone.utc), 1.0) for f in snaps]
    else:
        out_path = Path("reports/kinetic_validation.json" if args.days == 1 else f"reports/kinetic_validation_{args.days:g}d.json")
        runs = [(None, START, args.days)]
    rng = np.random.default_rng(7)
    out = []
    for snap, start, days in runs:
        cat = _catalog(snap)
        grid = build_grid(cat, when=start)
        rows = []
        for i, t in enumerate(cat):
            a, e, _, _ = elements(t)
            hp = a * (1 - e) - 6378.137
            if 400 < hp < 1300 and e < 0.02:
                rows.append((i, constellation_index(t.name), a, e, hp))
        star = [r for r in rows if r[1] == 0]
        oneweb = [r for r in rows if r[1] == 1]
        other = [r for r in rows if r[1] < 0]
        sample = [star[k] for k in rng.choice(len(star), 40, replace=False)] + \
                 [oneweb[k] for k in rng.choice(len(oneweb), min(20, len(oneweb)), replace=False)] + \
                 [other[k] for k in rng.choice(len(other), 40, replace=False)]
        with ProcessPoolExecutor(8) as ex:
            events = list(ex.map(_events, [(r[0], days, snap, start) for r in sample]))
        for (i, g, a_, e, hp), ev in zip(sample, events):
            out.append({"snapshot": snap, "norad": cat[i].norad_id, "name": cat[i].name, "perigee_km": hp,
                        "constellation": g, "events": ev,
                        "raw_10km": days * object_conjunction_rate(grid, a_, e, -1, THRESH),
                        "excl_10km": days * object_conjunction_rate(grid, a_, e, g, THRESH, intra_weight=0.0)})
        print(f"{snap or 'live catalogue'}: {sum(len(r['events']) for r in out)} approaches so far", flush=True)
    summary = {}
    for label, sel in (("starlink", lambda r: r["constellation"] == 0), ("oneweb", lambda r: r["constellation"] == 1),
                       ("others", lambda r: r["constellation"] < 0)):
        rs = [r for r in out if sel(r)]
        if not rs:
            continue
        raw, excl = sum(r["raw_10km"] for r in rs), sum(r["excl_10km"] for r in rs)
        summary[label] = {"n": len(rs), "object_days": len(rs) * (1 if args.weeks else args.days)}
        for t in (1.0, 2.0, 5.0, 10.0):  # kinetic-gas rates scale with the threshold squared
            ev = [x for r in rs for x in r["events"] if x[0] < t]
            same = sum(1 for x in ev if x[1])
            f = (t / THRESH) ** 2
            intra = (raw - excl) * f
            lo, hi = _poisson_ci(same)
            summary[label][f"{t:g}km"] = {"actual": len(ev), "same_constellation": same, "raw": raw * f,
                                           "excluding_own": excl * f,
                                           "intra_weight": same / intra if intra > 0 else None,
                                           "intra_weight_ci95": [lo / intra, hi / intra] if intra > 0 else None}
            print(f"{label:8s} <{t:>4g} km  actual {len(ev):5d} ({same} same-constellation)  "
                  f"kinetic raw {raw * f:8.1f} (x{raw * f / max(len(ev), 1):.2f})  "
                  f"excluding own {excl * f:8.1f} (x{excl * f / max(len(ev), 1):.2f})")
    out_path.write_text(json.dumps({"summary": summary, "objects": out}, indent=1))


if __name__ == "__main__":
    main()
