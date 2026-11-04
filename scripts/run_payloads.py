"""Re-entry hindcast for payloads (satellites), with manoeuvre screening.

Payloads were left out of Phase 3 because they can manoeuvre. Here, at each of the same 135 issue
dates, a payload is forecast only if its TLE history shows no manoeuvre in the 120 days before the
issue date (solace.risk.reentry.manoeuvre_epochs: a rise in mean semi-major axis > 0.1 km between
TLEs), which uses only information available at the time. Its ballistic coefficient is fitted and it
is forecast exactly like debris: true weather (oracle), persistence, solar-cycle analog, anchored MSFC,
and MSFC on the density-corrected atmosphere (msfc_hold, the recommended method). We also record
whether it manoeuvred later, so results can be split by that (not knowable in advance).

Run from the repo root:  .venv/bin/python scripts/run_payloads.py --shard 0 --of 4
Writes reports/payload_rows_<shard>.npz and resumes if interrupted.
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
LOOKBACK = 120  # days without a manoeuvre before the issue date


def _rel(p, act):
    m = act <= 365
    return np.median(np.abs(p[m] - act[m]) / np.maximum(act[m], 1)) if m.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    out = Path(f"reports/payload_rows_{a.shard}.npz")
    tt = R.TLETable.load(Path(DATA_DIR) / "reentry_tles_payload.npz")
    epochs = R.manoeuvre_epochs(tt)
    sw = load_space_weather()
    last_day = max(d for d, r in sw.items() if r.get("type", "").startswith("OBS"))
    obs = D.observed_table(sw, date(2015, 6, 1), last_day)
    bodies = D.third_bodies(date(2015, 6, 1), last_day + timedelta(HORIZON + 30))
    hist = S.SolarHistory(sw, last_day)
    msfc = S.MSFCForecasts()
    db = B.load_daily()
    dd = R.decay_dates()
    last_seen = {}
    for i, j in zip(tt.norad, tt.jd):
        last_seen[i] = j
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
        s_all = R.issue_set(tt, d)
        jd = R._jd(d)
        before = R.manoeuvred_between(epochs, s_all.norad, jd - LOOKBACK, jd)
        keep = ~before
        s = R.IssueSet(**{k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in vars(s_all).items()})
        act = np.array([(dd[i] - d).days if i in dd else np.inf for i in s.norad], float)
        end = jd + np.minimum(np.where(np.isfinite(act), act, HORIZON), HORIZON)
        later = R.manoeuvred_between(epochs, s.norad, jd, end)
        h_known = min(HORIZON, (last_day - d).days)
        oracle = R.physics_forecast(s, obs, obs, h_known, bodies=bodies)
        bc = oracle["bc"]
        pers = R.physics_forecast(s, obs, D.persistence_table(sw, d, HORIZON), HORIZON, bc_override=bc, bodies=bodies)
        analog = R.physics_forecast(s, obs, D.analog_table(hist, d, HORIZON), HORIZON, bc_override=bc, bodies=bodies)
        f_m, ap_m = msfc.forecast(d, HORIZON + 2 * STEP, anchor=hist.f81[hist.idx(d) - 1])
        fut = D.forecast_table(d, f_m, ap_m, HORIZON, STEP)
        mf = R.physics_forecast(s, obs, fut, HORIZON, bc_override=bc, bodies=bodies)
        model = B.fit(db, d)
        obs_c = obs.corrected(model)
        oracle_hold = R.physics_forecast(s, obs_c, obs_c, h_known, bodies=bodies)
        mh = R.physics_forecast(s, obs_c, fut.corrected(model), HORIZON, bc_override=oracle_hold["bc"], bodies=bodies)
        row = {
            "issue": [d.isoformat()] * len(s.norad), "norad": s.norad, "hp": s.hp,
            "ha": s.a1 * (1 + s.e1) - D.RE, "e": s.e1, "inc": s.inc, "bc": bc, "bc_hold": oracle_hold["bc"],
            "bc_fallback": oracle["bc_fallback"], "manoeuvred_later": later,
            "oracle": oracle["days"], "persist": pers["days"], "analog": analog["days"], "msfc": mf["days"],
            "oracle_hold": oracle_hold["days"], "msfc_hold": mh["days"], "actual": act,
            "observed_until": np.array([last_seen.get(i, 0) - jd for i in s.norad]),
            "known_until": np.full(len(s.norad), (last_day - d).days),
            "n_screened_out": np.full(len(s.norad), int(before.sum())),
        }
        for k, v in row.items():
            cols.setdefault(k, []).extend(list(v))
        np.savez_compressed(out, **{k: np.asarray(v) for k, v in cols.items()})
        print(f"{d} coasting={len(s.norad)} manoeuvred-before={before.sum()} later={later.sum()} "
              f"re<1y={(act <= 365).sum()} ({time.time() - t:.0f}s) analog {_rel(analog['days'], act):.2f} "
              f"msfc {_rel(mf['days'], act):.2f} msfc_hold {_rel(mh['days'], act):.2f} oracle {_rel(oracle['days'], act):.2f}",
              flush=True)


if __name__ == "__main__":
    main()
