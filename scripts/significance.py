"""Uncertainty of the headline re-entry results: block bootstrap over issue dates.

Forecasts issued on the same date share the same solar and atmospheric errors, and neighbouring
dates share the same phase of the cycle, so objects are not independent. We resample issue dates
in circular blocks of 6 (~6 months; Politis and Romano 1994 stationary-bootstrap style) and pool
every forecast of the drawn dates. For each method: median relative timing error at the 1-year
horizon, share within 20 %, and re-entry count error per date; for the 80 % windows: coverage and
the interval score (Gneiting and Raftery 2007) in log-time. Paired differences use the same draws.

Run from the repo root after analyze_phase3c.py:  .venv/bin/python scripts/significance.py
Writes reports/significance.json.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np

from solace.risk.conformal import online_windows

H = 365
B = 2000
BLOCK = 6
METHODS = ("persist", "analog", "msfc", "blend", "msfc_hold", "oracle", "oracle_hold", "sgp4")
PAIRS = (("msfc", "analog"), ("msfc_hold", "analog"), ("blend", "analog"), ("analog", "persist"),
         ("msfc", "persist"), ("oracle_hold", "oracle"))
WINDOWS = ("analog", "msfc", "msfc_hold")
OUT = Path("reports/significance.json")


def load():
    spec = importlib.util.spec_from_file_location("a3c", Path(__file__).with_name("analyze_phase3c.py"))
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    r = a.load()
    keep = np.ones(len(r["issue"]), bool)
    for m in ("oracle_hold", "msfc", "msfc_hold"):
        keep &= ~np.isnan(r[m])
    return {k: v[keep] for k, v in r.items()}


def per_issue_stats(r, pred, issues):
    """Per issue date: relative errors of re-entries within H (list) and count error."""
    act = r["actual"]
    errs, counts = [], []
    for d in issues:
        m = (r["issue"] == d) & (r["known_until"] >= H)
        hit = m & (act <= H)
        e = np.abs(pred[hit] - act[hit]) / np.maximum(act[hit], 1)
        errs.append(e)
        counts.append(abs(int(np.sum(m & (pred <= H))) - int(np.sum(hit))))
    return errs, np.array(counts, float)


def interval_stats(r, lo, hi, issues, alpha=0.2):
    """Per issue date: coverage indicators and interval scores in log-time, re-entries within H."""
    act = r["actual"]
    cov, score = [], []
    for d in issues:
        k = (r["issue"] == d) & (act <= H) & (act > 0) & np.isfinite(lo) & (r["known_until"] >= H)
        y, lo_, hi_ = np.log(act[k]), np.log(np.maximum(lo[k], 0.5)), np.log(np.maximum(hi[k], 0.5))
        cov.append((y >= lo_) & (y <= hi_))
        score.append((hi_ - lo_) + 2 / alpha * (lo_ - y) * (y < lo_) + 2 / alpha * (y - hi_) * (y > hi_))
    return cov, score


def draws(n, rng):
    """Circular block bootstrap indices over n ordered issue dates."""
    starts = rng.integers(0, n, size=(B, -(-n // BLOCK)))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n
    return idx.reshape(B, -1)[:, :n]


def summarise(name, point, boot):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)]}


def main():
    r = load()
    rng = np.random.default_rng(20261104)
    day = np.array([date.fromisoformat(s).toordinal() for s in r["issue"]])
    out = {}
    for split, cut in (("test_2023plus", "2023-01-01"), ("all_years", "2000-01-01")):
        issues = np.array(sorted({s for s in r["issue"] if s >= cut}))
        idx = draws(len(issues), rng)
        res = {"n_issues": int(len(issues)), "methods": {}, "pairs": {}, "windows": {}}
        stats = {m: per_issue_stats(r, r[m], issues) for m in METHODS}

        def med(errs, sel):
            return np.median(np.concatenate([errs[i] for i in sel]))

        def w20(errs, sel):
            return np.mean(np.concatenate([errs[i] for i in sel]) <= 0.2)

        boot, everything = {}, range(len(issues))
        for m, (errs, cnt) in stats.items():
            boot[m] = np.array([med(errs, s) for s in idx])
            res["methods"][m] = {
                "median_rel_err": summarise(m, med(errs, range(len(issues))), boot[m]),
                "within_20pct": summarise(m, w20(errs, range(len(issues))), [w20(errs, s) for s in idx]),
                "count_mae": summarise(m, cnt.mean(), [cnt[s].mean() for s in idx]),
            }
        for a, b in PAIRS:
            d = boot[a] - boot[b]
            dc = np.array([stats[a][1][s].mean() - stats[b][1][s].mean() for s in idx])
            res["pairs"][f"{a}-{b}"] = {
                "median_rel_err_diff": summarise("", med(stats[a][0], everything) - med(stats[b][0], everything), d),
                "p_not_better": float(np.mean(d >= 0)),
                "count_mae_diff": summarise("", stats[a][1].mean() - stats[b][1].mean(), dc),
            }
        for m in WINDOWS:
            lo, hi, _ = online_windows(day, r["actual"], r[m], r["hp"])
            cov, score = interval_stats(r, lo, hi, issues)
            cov_b = [np.mean(np.concatenate([cov[i] for i in s])) for s in idx]
            sc_b = [np.mean(np.concatenate([score[i] for i in s])) for s in idx]
            yearly = {}
            for y in sorted({d[:4] for d in issues}):
                sel = [i for i, d in enumerate(issues) if d.startswith(y)]
                c = np.concatenate([cov[i] for i in sel])
                if len(c):
                    yearly[y] = float(c.mean())
            res["windows"][m] = {
                "coverage": summarise("", np.mean(np.concatenate(cov)), cov_b),
                "interval_score": summarise("", np.mean(np.concatenate(score)), sc_b),
                "worst_year": min(yearly.items(), key=lambda kv: kv[1]),
            }
        out[split] = res
    OUT.write_text(json.dumps(out, indent=1))
    for split, res in out.items():
        print(f"\n== {split} ({res['n_issues']} issue dates), 1-year horizon, 95 % block-bootstrap CIs")
        for m, s in res["methods"].items():
            e, c = s["median_rel_err"], s["count_mae"]
            print(f"  {m:12s} median err {100 * e['point']:5.1f} % [{100 * e['ci95'][0]:.1f}, {100 * e['ci95'][1]:.1f}]"
                  f"   count err {c['point']:5.1f} [{c['ci95'][0]:.1f}, {c['ci95'][1]:.1f}]")
        for k, s in res["pairs"].items():
            e, c = s["median_rel_err_diff"], s["count_mae_diff"]
            print(f"  {k:22s} d(median) {100 * e['point']:+5.1f} pts "
                  f"[{100 * e['ci95'][0]:+.1f}, {100 * e['ci95'][1]:+.1f}] P(not better) {s['p_not_better']:.3f}"
                  f"   d(count) {c['point']:+.1f} [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}]")
        for m, s in res["windows"].items():
            c, i = s["coverage"], s["interval_score"]
            print(f"  window around {m:10s} coverage {c['point']:.2f} [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]"
                  f"  interval score {i['point']:.3f} [{i['ci95'][0]:.3f}, {i['ci95'][1]:.3f}]"
                  f"  worst year {s['worst_year']}")


if __name__ == "__main__":
    main()
