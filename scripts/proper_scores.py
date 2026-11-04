"""Brier scores for "re-enters within H days", over every forecast (not only objects that re-entered).

Each forecast becomes a probability through solace.risk.conformal.event_probability: the empirical
distribution of log(actual / forecast) among earlier forecasts that are fully resolved, by lead time and
perigee band, with no look-ahead. Whether an object re-entered within H is known for every object once
H days have passed, so the score needs no censoring correction. Skill is relative to climatology (the
calibration-set base rate). Intervals: circular block bootstrap over issue dates, as in significance.py.

Run from the repo root:  PYTHONPATH=scripts .venv/bin/python scripts/proper_scores.py
Writes reports/proper_scores.json.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from significance import BLOCK, draws, load  # same data and bootstrap (run with PYTHONPATH=scripts)

from solace.risk.conformal import event_probability

METHODS = ("sgp4", "persist", "analog", "msfc", "blend", "msfc_hold", "oracle")
HORIZONS = (90, 180, 365)
OUT = Path("reports/proper_scores.json")


def main():
    r = load()
    day = np.array([date.fromisoformat(s).toordinal() for s in r["issue"]])
    rng = np.random.default_rng(20261105)
    out = {"block": BLOCK}
    for H in HORIZONS:
        known = r["known_until"] >= H
        y = (r["actual"] <= H).astype(float)
        probs = {m: event_probability(day, r["actual"], r[m], r["hp"], H) for m in METHODS}
        ok = known & np.all([np.isfinite(p) for p in probs.values()], axis=0)
        # Climatology: base rate of the event among resolved forecasts known at the issue date.
        clim = np.full(len(y), np.nan)
        for d in np.unique(day[ok]):
            c = (day + 730 <= d)
            clim[day == d] = np.mean(r["actual"][c] <= H)
        for split, cut in (("from_2018_06", "2018-06-01"), ("test_2023plus", "2023-01-01")):
            sel = ok & (r["issue"] >= cut)
            issues = np.array(sorted(set(r["issue"][sel])))
            idx = draws(len(issues), rng)
            by = {d: np.nonzero(sel & (r["issue"] == d))[0] for d in issues}
            res = {"n": int(sel.sum()), "n_issues": int(len(issues)), "base_rate": float(y[sel].mean())}
            bs_c = np.array([np.mean((clim[by[d]] - y[by[d]]) ** 2) for d in issues])
            n_d = np.array([len(by[d]) for d in issues])
            for m in METHODS:
                bs_d = np.array([np.mean((probs[m][by[d]] - y[by[d]]) ** 2) for d in issues])
                bs = float(np.sum(bs_d * n_d) / n_d.sum())
                bss = 1 - bs / float(np.sum(bs_c * n_d) / n_d.sum())
                boot = [1 - np.sum(bs_d[s] * n_d[s]) / np.sum(bs_c[s] * n_d[s]) for s in idx]
                res[m] = {"brier": bs, "skill_vs_climatology": bss,
                          "skill_ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                          "bs_by_issue": bs_d.tolist()}
            for a, b in (("msfc", "analog"), ("msfc_hold", "analog"), ("analog", "persist"), ("msfc", "persist"),
                         ("persist", "sgp4")):
                da, db = np.array(res[a]["bs_by_issue"]), np.array(res[b]["bs_by_issue"])
                diff = [np.sum((da[s] - db[s]) * n_d[s]) / np.sum(n_d[s]) for s in idx]
                res[f"{a}-{b}"] = {"brier_diff": float(np.sum((da - db) * n_d) / n_d.sum()),
                                   "ci95": [float(x) for x in np.percentile(diff, [2.5, 97.5])],
                                   "p_not_better": float(np.mean(np.array(diff) >= 0))}
            for m in METHODS:
                res[m].pop("bs_by_issue")
            out[f"H{H}_{split}"] = res
            print(f"\n== H={H} d, {split}: {res['n']} forecasts, {res['n_issues']} dates, "
                  f"base rate {res['base_rate']:.3f}")
            for m in METHODS:
                s = res[m]
                print(f"  {m:10s} Brier {s['brier']:.4f}  skill vs climatology {s['skill_vs_climatology']:+.3f} "
                      f"[{s['skill_ci95'][0]:+.3f}, {s['skill_ci95'][1]:+.3f}]")
            for k in [k for k in res if "-" in k]:
                s = res[k]
                print(f"  {k:18s} dBrier {s['brier_diff']:+.5f} [{s['ci95'][0]:+.5f}, {s['ci95'][1]:+.5f}] "
                      f"P(not better) {s['p_not_better']:.3f}")
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
