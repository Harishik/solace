"""Sensitivity of the MSFC-driven re-entry forecasts to the anchoring choice (docs/PHYSICS.md §10).

The anchored MSFC forecast rescales the median flux to today's 81-day F10.7, fading over one year.
That choice was made after seeing 2024-25, so here we rerun with no anchoring and with fades of six
months and two years. Same objects and ballistic coefficients as Phase 3c; columns msfc_raw,
msfc_f182, msfc_f730, matched by (issue, norad).

Run from the repo root:  .venv/bin/python scripts/run_msfc_sensitivity.py --shard 0 --of 4
Writes reports/msfcsens_rows_<shard>.npz and resumes if interrupted.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from solace.ingest.cache import DATA_DIR
from solace.risk import decay as D
from solace.risk import reentry as R
from solace.risk import solar as S
from solace.risk.drivers import load_space_weather

HORIZON = 730
STEP = 5
VARIANTS = {"msfc_raw": None, "msfc_f182": 182.0, "msfc_f730": 730.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    out = Path(f"reports/msfcsens_rows_{a.shard}.npz")
    tt = R.TLETable.load(Path(DATA_DIR) / "reentry_tles.npz")
    sw = load_space_weather()
    last_day = max(d for d, r in sw.items() if r.get("type", "").startswith("OBS"))
    obs = D.observed_table(sw, date(2015, 6, 1), last_day)
    bodies = D.third_bodies(date(2015, 6, 1), last_day + timedelta(HORIZON + 30))
    hist = S.SolarHistory(sw, last_day)
    msfc = S.MSFCForecasts()
    with np.load("reports/phase3c_rows.npz", allow_pickle=True) as z:
        bc_of = {(i, n): b for i, n, b in zip(z["issue"], z["norad"], z["bc"])}
    issues = [date(2016, 6, 6) + timedelta(28 * k) for k in range(200)]
    issues = [d for d in issues if d <= last_day - timedelta(28)][a.shard::a.of]
    cols = {}
    if out.exists():
        z = np.load(out, allow_pickle=True)
        cols = {k: list(z[k]) for k in z.files}
    have = set(cols.get("issue", []))
    for d in issues:
        if d.isoformat() in have:
            continue
        t = time.time()
        s = R.issue_set(tt, d)
        bc = np.array([bc_of.get((d.isoformat(), n), np.nan) for n in s.norad], float)
        level = hist.f81[hist.idx(d) - 1]
        row = {"issue": [d.isoformat()] * len(s.norad), "norad": s.norad}
        for name, fade in VARIANTS.items():
            anchor = None if fade is None else level
            f, apv = msfc.forecast(d, HORIZON + 2 * STEP, anchor=anchor, fade_days=fade or 365.0)
            fc = R.physics_forecast(s, obs, D.forecast_table(d, f, apv, HORIZON, STEP), HORIZON,
                                    bc_override=bc, bodies=bodies)
            row[name] = fc["days"]
        for k, v in row.items():
            cols.setdefault(k, []).extend(list(v))
        np.savez_compressed(out, **{k: np.asarray(v) for k, v in cols.items()})
        print(f"{d} n={len(s.norad)} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
