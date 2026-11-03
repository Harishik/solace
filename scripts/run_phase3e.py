"""Phase 3e: re-entry forecasts driven by NASA MSFC's archived solar-cycle forecasts.

Same 135 issue dates, objects and fitted ballistic coefficients as Phase 3c (reports/phase3c_rows.npz);
only the future space weather changes. At each issue date we take the latest MSFC forecast issued in
an earlier month (scripts/fetch_msfc.py), so it was public at the time:
- "msfc": MSFC median F10.7 and Ap, rescaled to today's 81-day flux, fading over a year;
- "blend": geometric mean of that flux and the solar-cycle analog's, with the analog's Ap.
With --bias, instead runs "msfc_hold": the MSFC forecast on the NRLMSIS atmosphere corrected with
observed densities (Phase 3d, src/solace/risk/density_bias.py), with Phase 3d's ballistic
coefficients; writes reports/phase3f_rows_<shard>.npz.
Rows match phase3c_rows by (issue, norad).

Run from the repo root:  .venv/bin/python scripts/run_phase3e.py --shard 0 --of 4
Writes reports/phase3e_rows_<shard>.npz and resumes if interrupted.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from solace.ingest.cache import DATA_DIR
from solace.risk import decay as D
from solace.risk import density_bias as B
from solace.risk import reentry as R
from solace.risk import solar as S
from solace.risk.drivers import load_space_weather

HORIZON = 730
STEP = 5


def _rel(p, act):
    m = act <= 365
    return np.median(np.abs(p[m] - act[m]) / np.maximum(act[m], 1)) if m.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--bias", action="store_true")
    a = ap.parse_args()
    out = Path(f"reports/{'phase3f' if a.bias else 'phase3e'}_rows_{a.shard}.npz")
    tt = R.TLETable.load(Path(DATA_DIR) / "reentry_tles.npz")
    sw = load_space_weather()
    last_day = max(d for d, r in sw.items() if r.get("type", "").startswith("OBS"))
    obs = D.observed_table(sw, date(2015, 6, 1), last_day)
    bodies = D.third_bodies(date(2015, 6, 1), last_day + timedelta(HORIZON + 30))
    hist = S.SolarHistory(sw, last_day)
    msfc = S.MSFCForecasts()
    bc_of = {}
    for path in sorted(Path("reports").glob("phase3d_rows_*.npz")) if a.bias else [Path("reports/phase3c_rows.npz")]:
        with np.load(path, allow_pickle=True) as z:
            bc_of.update(zip(zip(z["issue"].astype("U10"), z["norad"]), z["bc_hold" if a.bias else "bc"]))
    db = B.load_daily() if a.bias else None
    dd = R.decay_dates()
    issues = [date(2016, 6, 6) + timedelta(28 * k) for k in range(200)]
    issues = [d for d in issues if d <= last_day - timedelta(28)][a.shard::a.of]
    cols = {}
    if out.exists():
        z = np.load(out, allow_pickle=True)
        cols = {k: list(z[k]) for k in z.files}
    have = set(cols.get("issue", []))
    n_days = HORIZON + 2 * STEP
    for d in issues:
        if d.isoformat() in have:
            continue
        t = time.time()
        s = R.issue_set(tt, d)
        bc = np.array([bc_of.get((d.isoformat(), n), np.nan) for n in s.norad], float)
        act = np.array([(dd[i] - d).days if i in dd else np.inf for i in s.norad], float)
        level = hist.f81[hist.idx(d) - 1]
        f_m, ap_m = msfc.forecast(d, n_days, anchor=level)
        mf, ma, _ = hist.analog(d, n_days, 730.0)
        f_b, ap_b = np.sqrt(f_m * np.median(mf, 0)), np.median(ma, 0)
        row = {"issue": [d.isoformat()] * len(s.norad), "norad": s.norad}
        msg = []
        runs = (("msfc_hold", f_m, ap_m),) if a.bias else (("msfc", f_m, ap_m), ("blend", f_b, ap_b))
        model = B.fit(db, d) if a.bias else None
        for name, f, apv in runs:
            future = D.forecast_table(d, f, apv, HORIZON, STEP).corrected(model)
            fc = R.physics_forecast(s, obs.corrected(model), future, HORIZON, bc_override=bc, bodies=bodies)
            row[name] = fc["days"]
            msg.append(f"{name} {_rel(fc['days'], act):.2f}")
        for k, v in row.items():
            cols.setdefault(k, []).extend(list(v))
        np.savez_compressed(out, **{k: np.asarray(v) for k, v in cols.items()})
        print(f"{d} n={len(s.norad)} ({time.time() - t:.0f}s) " + " | ".join(msg), flush=True)


if __name__ == "__main__":
    main()
