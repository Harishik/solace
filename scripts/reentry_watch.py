"""Re-entry watch: the paper's forecast, issued today for every low-perigee debris object and rocket body.

Method (docs/PHYSICS.md §10, "recommended configuration"):
- ballistic coefficient fitted to each object's own decay between a TLE 21-70 days ago (weekly
  Spacebook snapshots in data/history) and today's catalogue (Spacebook mirror, refreshed here);
- NRLMSIS 2.0 corrected with CHAMP/GRACE/GRACE-FO/Swarm densities older than 90 days (§9);
- NASA MSFC's latest solar-cycle forecast, anchored to today's 81-day flux (§10); Sun, Moon and J2;
- 80 % windows from the online conformal calibration of the 2016-2026 hindcast of exactly this
  method (reports/phase3f_rows_*.npz), using its final adaptive level.

Writes web/data/reentry_watch.json (served by the Live Sky app at /api/reentry).

Run from the repo root:  .venv/bin/python scripts/reentry_watch.py
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from solace.ingest.cache import DATA_DIR, cached_get
from solace.ingest.spacebook import TLE_URL, satcat
from solace.risk import conformal as C
from solace.risk import decay as D
from solace.risk import density_bias as B
from solace.risk import reentry as R
from solace.risk import solar as S
from solace.risk.drivers import load_space_weather

OUT = Path("web/data/reentry_watch.json")
HORIZON = 730
SHOW_DAYS = 400


def calibration():
    """Final adaptive level and group quantiles from the hindcast of this exact method."""
    spec = importlib.util.spec_from_file_location("a3c", Path(__file__).with_name("analyze_phase3c.py"))
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    r = a.load()
    keep = ~np.isnan(r["msfc_hold"])
    r = {k: v[keep] for k, v in r.items()}
    day = np.array([date.fromisoformat(s).toordinal() for s in r["issue"]])
    lo, hi, alpha = C.online_windows(day, r["actual"], r["msfc_hold"], r["hp"])
    known = (r["actual"] <= 365) & np.isfinite(lo)
    cover = float(np.mean((r["actual"][known] >= lo[known]) & (r["actual"][known] <= hi[known])))
    return C.quantile_table(r["actual"], r["msfc_hold"], r["hp"], alpha), alpha, cover


def main():
    q, alpha, cover = calibration()
    cached_get(TLE_URL, "spacebook_tle.txt", max_age_s=6 * 3600, timeout=120)  # today's catalogue
    sw = load_space_weather()
    last_obs = max(d for d, r in sw.items() if r.get("type", "").startswith("OBS"))
    today = datetime.now(timezone.utc).date()
    hist = sorted(p for p in (Path(DATA_DIR) / "history").glob("sb_*.tle")
                  if date.fromisoformat(p.stem[3:]) >= today - timedelta(80))
    tt = R.build_tle_table(hist + [Path(DATA_DIR) / "raw" / "spacebook_tle.txt"], max_perigee_km=600)
    issue = min(today, last_obs + timedelta(1))
    s = R.issue_set(tt, issue, max_age_days=10, max_perigee_km=600)
    print(f"issue {issue}: {len(s.norad)} objects with a 21-70 day decay baseline")

    model = B.fit(B.load_daily(), issue)
    obs = D.observed_table(sw, issue - timedelta(120), issue).corrected(model)
    bodies = D.third_bodies(issue - timedelta(120), issue + timedelta(HORIZON + 30))
    hist_f = S.SolarHistory(sw, last_obs)
    f, ap = S.MSFCForecasts().forecast(issue, HORIZON + 10, anchor=hist_f.f81[hist_f.idx(issue) - 1])
    fc = R.physics_forecast(s, obs, D.forecast_table(issue, f, ap, HORIZON).corrected(model), HORIZON, bodies=bodies)

    days = fc["days"]
    grp, _ = C.groups(days, s.hp)
    cat = satcat()
    rows, overdue = [], []
    t0 = datetime(issue.year, issue.month, issue.day, tzinfo=timezone.utc)
    for k in np.argsort(days):
        if not np.isfinite(days[k]) or days[k] > SHOW_DAYS:
            continue
        nid = int(s.norad[k])
        if days[k] <= 0.5:  # predicted down already: likely re-entered, not yet removed from the catalogue
            overdue.append({"id": nid, "name": (satcat().get(nid) or {}).get("name") or str(nid)})
            continue
        ql, qh = q.get(int(grp[k]), (np.nan, np.nan))
        d, lo, hi = float(days[k]), float(days[k] * np.exp(ql)), float(days[k] * np.exp(qh))
        rows.append({"id": nid, "name": (cat.get(nid) or {}).get("name") or str(nid), "rb": bool(s.rb[k]),
                     "perigee_km": float(s.hp[k]), "days": d, "lo_days": lo, "hi_days": hi,
                     "median_utc": (t0 + timedelta(d)).isoformat(), "lo_utc": (t0 + timedelta(lo)).isoformat(),
                     "hi_utc": (t0 + timedelta(hi)).isoformat(), "bc": float(fc["bc"][k])})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "ok", "issued": issue.isoformat(), "objects": rows, "overdue": overdue,
        "n_screened": int(len(s.norad)),
        "alpha": alpha, "hindcast_coverage_1y": cover,
        "method": (f"Fitted drag, density-corrected NRLMSIS 2.0, NASA MSFC flux forecast; windows calibrated on "
                   f"the 2016-2026 hindcast ({100 * cover:.0f} % of 1-year outcomes inside)."),
    }, indent=0))
    print(f"{len(rows)} objects re-entering within {SHOW_DAYS} days -> {OUT}")


if __name__ == "__main__":
    main()
