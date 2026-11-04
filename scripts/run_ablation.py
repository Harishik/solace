"""Ablations of the re-entry hindcast: density model and drag-fit window.

Same 135 issue dates and cohort as Phase 3c. For each issue date and variant, fit the ballistic
coefficient and forecast with (a) the true weather ("oracle") and (b) the anchored NASA MSFC flux
forecast ("msfc", uncorrected atmosphere), so each variant is compared with the matching baseline
columns of reports/phase3c_rows.npz and phase3e_rows_*.npz.

Variants:
  --msis 0          NRLMSISE-00 instead of NRLMSIS 2.1           -> columns oracle_msis00, msfc_msis00
  --baseline 14 40  drag-fit baseline of 14-40 days (default 21-70) -> columns oracle_b14_40, msfc_b14_40
  --geometry        density at each orbit node's own latitude and local time (decay.GEOMETRY)
                    instead of the latitude/local-time mean -> columns oracle_geom, msfc_geom

Run from the repo root:  .venv/bin/python scripts/run_ablation.py --msis 0 --shard 0 --of 4
Writes reports/ablation_<tag>_rows_<shard>.npz and resumes if interrupted.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msis", type=float, default=2.1)
    ap.add_argument("--baseline", type=int, nargs=2, default=(21, 70))
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    D.MSIS_VERSION = 0 if a.msis == 0 else a.msis  # pymsis wants the integer 0 for NRLMSISE-00
    D.GEOMETRY = a.geometry
    tag = "geom" if a.geometry else "msis00" if a.msis == 0 else f"b{a.baseline[0]}_{a.baseline[1]}"
    out = Path(f"reports/ablation_{tag}_rows_{a.shard}.npz")
    tt = R.TLETable.load(Path(DATA_DIR) / "reentry_tles.npz")
    sw = load_space_weather()
    last_day = max(d for d, r in sw.items() if r.get("type", "").startswith("OBS"))
    obs = D.observed_table(sw, date(2015, 6, 1), last_day)
    bodies = D.third_bodies(date(2015, 6, 1), last_day + timedelta(HORIZON + 30))
    hist = S.SolarHistory(sw, last_day)
    msfc = S.MSFCForecasts()
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
        s = R.issue_set(tt, d, baseline=tuple(a.baseline))
        h_known = min(HORIZON, (last_day - d).days)
        oracle = R.physics_forecast(s, obs, obs, h_known, bodies=bodies)
        f, apv = msfc.forecast(d, HORIZON + 2 * STEP, anchor=hist.f81[hist.idx(d) - 1])
        fc = R.physics_forecast(s, obs, D.forecast_table(d, f, apv, HORIZON, STEP), HORIZON,
                                bc_override=oracle["bc"], bodies=bodies)
        row = {"issue": [d.isoformat()] * len(s.norad), "norad": s.norad,
               f"oracle_{tag}": oracle["days"], f"msfc_{tag}": fc["days"], f"bc_{tag}": oracle["bc"]}
        for k, v in row.items():
            cols.setdefault(k, []).extend(list(v))
        np.savez_compressed(out, **{k: np.asarray(v) for k, v in cols.items()})
        print(f"{d} n={len(s.norad)} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
