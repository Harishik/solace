"""Phase 3d: re-entry forecasts on an NRLMSIS atmosphere corrected with observed densities.

Same 135 issue dates and objects as Phase 3c. At each issue date a density-bias model is fitted to
GRACE-FO and Swarm B/C densities older than 90 days (src/solace/risk/density_bias.py), then
applied to both the fit window (observed weather) and the forecast (true weather, or the
solar-cycle analog). Two variants below 400 km: correction held ("hold") or tapered to zero at
120 km ("taper"). Rows match reports/phase3c_rows.npz by (issue, norad).

Run from the repo root:  .venv/bin/python scripts/run_phase3d.py --shard 0 --of 4
Writes reports/phase3d_rows_<shard>.npz and resumes if interrupted.
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


def _rel(p, act):
    m = act <= 365
    return np.median(np.abs(p[m] - act[m]) / np.maximum(act[m], 1)) if m.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    out = Path(f"reports/phase3d_rows_{a.shard}.npz")
    tt = R.TLETable.load(Path(DATA_DIR) / "reentry_tles.npz")
    sw = load_space_weather()
    last_day = max(d for d, r in sw.items() if r.get("type", "").startswith("OBS"))
    obs = D.observed_table(sw, date(2015, 6, 1), last_day)
    bodies = D.third_bodies(date(2015, 6, 1), last_day + timedelta(HORIZON + 30))
    hist = S.SolarHistory(sw, last_day)
    db = B.load_daily()
    dd = R.decay_dates()
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
        h_known = min(HORIZON, (last_day - d).days)
        act = np.array([(dd[i] - d).days if i in dd else np.inf for i in s.norad], float)
        row = {"issue": [d.isoformat()] * len(s.norad), "norad": s.norad}
        msg = []
        for name, taper in (("hold", False), ("taper", True)):
            model = B.fit(db, d, taper=taper)
            obs_c = obs.corrected(model)
            oracle = R.physics_forecast(s, obs_c, obs_c, h_known, bodies=bodies)
            analog = R.physics_forecast(s, obs_c, D.analog_table(hist, d, HORIZON).corrected(model), HORIZON,
                                        bc_override=oracle["bc"], bodies=bodies)
            row[f"oracle_{name}"] = oracle["days"]
            row[f"analog_{name}"] = analog["days"]
            row[f"bc_{name}"] = oracle["bc"]
            msg.append(f"{name}: oracle {_rel(oracle['days'], act):.2f} analog {_rel(analog['days'], act):.2f}"
                       f" (model n={model.n if model else 0})")
        for k, v in row.items():
            cols.setdefault(k, []).extend(list(v))
        np.savez_compressed(out, **{k: np.asarray(v) for k, v in cols.items()})
        print(f"{d} n={len(s.norad)} ({time.time() - t:.0f}s) " + " | ".join(msg), flush=True)


if __name__ == "__main__":
    main()
