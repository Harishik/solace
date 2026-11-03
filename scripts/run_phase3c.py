"""Phase 3c: per-object re-entry forecasts for debris and rocket bodies, 2016-2026.

At each issue date (every 4 weeks), objects with perigee < 600 km get a ballistic coefficient fitted
from their own TLE decay over the previous 3-10 weeks (observed density), then a predicted
re-entry date under four density scenarios: SGP4 with frozen B* (old baseline), persistence of
the 81-day solar flux, a solar-cycle analog forecast (earlier cycles read forward from today's
phase, src/solace/risk/solar.py), and the true future space weather (oracle, upper bound).
Version 2 adds Sun and Moon perturbations (J2 + lunisolar secular rates) to every physics run,
plus an oracle without them as an ablation. Version 1 results stay in reports/phase3c_v1_rows.npz.

Run from the repo root:  .venv/bin/python scripts/run_phase3c.py
Writes reports/phase3c_rows.npz (one row per object and issue date) and resumes if interrupted.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from solace.ingest.cache import DATA_DIR
from solace.risk import decay as D
from solace.risk import reentry as R
from solace.risk import solar as S
from solace.risk.drivers import load_space_weather

OUT = Path("reports/phase3c_rows.npz")  # v2
HORIZON = 730


def main():
    tt = R.TLETable.load(Path(DATA_DIR) / "reentry_tles.npz")
    sw = load_space_weather()
    last_day = max(d for d, r in sw.items() if r.get("type", "").startswith("OBS")) if any(
        r.get("type") for r in sw.values()) else date(2026, 10, 31)
    obs = D.observed_table(sw, date(2015, 6, 1), last_day)
    bodies = D.third_bodies(date(2015, 6, 1), last_day + timedelta(HORIZON + 30))
    hist = S.SolarHistory(sw, last_day)
    dd = R.decay_dates()
    last_seen = {}
    for i, j in zip(tt.norad, tt.jd):
        last_seen[i] = j
    issues = [date(2016, 6, 6) + timedelta(28 * k) for k in range(200)]
    issues = [d for d in issues if d <= last_day - timedelta(28)]
    done = {}
    if OUT.exists():
        z = np.load(OUT, allow_pickle=True)
        done = {k: z[k] for k in z.files}
    have = set(done.get("issue", np.array([], dtype="U10")).tolist())
    cols = {k: list(v) for k, v in done.items()}
    for d in issues:
        if d.isoformat() in have:
            continue
        t = time.time()
        s = R.issue_set(tt, d)
        h_known = min(HORIZON, (last_day - d).days)
        oracle = R.physics_forecast(s, obs, obs, h_known, bodies=bodies)
        bc = oracle["bc"]
        oracle_nols = R.physics_forecast(s, obs, obs, h_known)
        pers = R.physics_forecast(s, obs, D.persistence_table(sw, d, HORIZON), HORIZON, bc_override=bc, bodies=bodies)
        analog = R.physics_forecast(s, obs, D.analog_table(hist, d, HORIZON), HORIZON, bc_override=bc, bodies=bodies)
        bstar_bc = np.clip(D.BSTAR_TO_BC * s.bstar, 1e-4, 2.0)
        oracle_bstar = R.physics_forecast(s, obs, obs, h_known, bc_override=bstar_bc, bodies=bodies)
        sg = R.sgp4_forecast(s, 365)
        jd_issue = R._jd(d)
        act = np.array([(dd[i] - d).days if i in dd else np.inf for i in s.norad], float)
        # Fate known within the evaluation window: decayed (date known) or still tracked later.
        observed_until = np.array([last_seen.get(i, 0) - jd_issue for i in s.norad])
        past = [d - timedelta(i) for i in range(180, 0, -1)]
        f, fa, ap = D._drivers(past, sw)
        slope = np.polyfit(np.arange(-180, 0, dtype=float), fa, 1)[0]
        row = {
            "issue": [d.isoformat()] * len(s.norad), "norad": s.norad, "hp": s.hp,
            "ha": s.a1 * (1 + s.e1) - D.RE, "e": s.e1, "inc": s.inc, "bstar": s.bstar, "rb": s.rb,
            "bc": bc, "bc_fallback": oracle["bc_fallback"], "baseline_days": s.jd1 - s.jd0,
            "tle_age": jd_issue - s.jd1, "da_obs": s.a1 - s.a0,
            "f107a": np.full(len(s.norad), fa[-1]), "f107_slope": np.full(len(s.norad), slope),
            "ap81": np.full(len(s.norad), np.mean(ap[-81:])),
            "oracle": oracle["days"], "persist": pers["days"], "analog": analog["days"],
            "oracle_nols": oracle_nols["days"], "bc_nols": oracle_nols["bc"],
            "oracle_bstar": oracle_bstar["days"], "sgp4": sg, "actual": act,
            "observed_until": observed_until, "known_until": np.full(len(s.norad), (last_day - d).days),
        }
        for k, v in row.items():
            cols.setdefault(k, []).extend(list(v))
        np.savez_compressed(OUT, **{k: np.asarray(v) for k, v in cols.items()})
        m = act <= 365
        def rel(p, m=m, act=act):
            return np.median(np.abs(p[m] - act[m]) / np.maximum(act[m], 1)) if m.any() else np.nan
        print(f"{d} n={len(s.norad)} re<1y={m.sum()} ({time.time() - t:.0f}s) median rel err: "
              f"sgp4 {rel(sg):.2f} persist {rel(pers['days']):.2f} analog {rel(analog['days']):.2f} "
              f"oracle {rel(oracle['days']):.2f} no-lunisolar {rel(oracle_nols['days']):.2f}", flush=True)


if __name__ == "__main__":
    main()
