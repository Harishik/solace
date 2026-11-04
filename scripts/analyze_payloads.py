"""Payload re-entry hindcast results (scripts/run_payloads.py).

Median relative 1-year timing error per method with circular block-bootstrap 95 % intervals over issue
dates (as scripts/significance.py), for all screened payloads and split by whether they manoeuvred
after the issue date (not knowable in advance), plus coverage of online conformal 80 % windows
calibrated on payloads themselves.

Run from the repo root:  .venv/bin/python scripts/analyze_payloads.py
Writes reports/payloads.json.
"""

from __future__ import annotations

import glob
import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np

from solace.risk.conformal import online_windows

OUT = Path("reports/payloads.json")
METHODS = ("persist", "analog", "msfc", "msfc_hold", "oracle", "oracle_hold")


def _sig():
    spec = importlib.util.spec_from_file_location("sig", Path(__file__).with_name("significance.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load():
    cols = {}
    for f in sorted(glob.glob("reports/payload_rows_*.npz")):
        with np.load(f, allow_pickle=True) as z:
            for k in z.files:
                cols.setdefault(k, []).append(z[k])
    r = {k: np.concatenate(v) for k, v in cols.items()}
    r["issue"] = r["issue"].astype("U10")
    order = np.argsort(r["issue"], kind="stable")
    return {k: v[order] for k, v in r.items()}


def main():
    sig = _sig()
    r = load()
    rng = np.random.default_rng(20261106)
    day = np.array([date.fromisoformat(s).toordinal() for s in r["issue"]])
    out = {"n_rows": int(len(r["issue"])), "n_issues": int(len(set(r["issue"]))),
           "n_objects": int(len(set(r["norad"].tolist()))),
           "screened_out_per_issue_mean": float(np.mean([r["n_screened_out"][r["issue"] == d][0]
                                                         for d in sorted(set(r["issue"]))]))}
    subsets = {"all": np.ones(len(r["issue"]), bool), "no_later_manoeuvre": ~r["manoeuvred_later"],
               "later_manoeuvre": r["manoeuvred_later"]}
    for split, cut in (("all_years", "2000-01-01"), ("test_2023plus", "2023-01-01")):
        out[split] = {}
        for name, sel in subsets.items():
            sub = {k: v[sel] for k, v in r.items()}
            issues = np.array(sorted({s for s in sub["issue"] if s >= cut}))
            draws = sig.draws(len(issues), rng)
            res = {}
            for m in METHODS:
                errs, _ = sig.per_issue_stats(sub, sub[m], issues)
                n = int(sum(len(e) for e in errs))
                if n == 0:
                    continue
                every = range(len(issues))
                med = lambda sel_: float(np.median(np.concatenate([errs[i] for i in sel_])))  # noqa: E731
                allerr = np.concatenate(errs)
                res[m] = {"median_rel_err": med(every),
                          "share_forecast_no_reentry": float(np.mean(~np.isfinite(allerr))),
                          "ci95": [float(x) for x in np.percentile([med(s) for s in draws], [2.5, 97.5])],
                          "n_reentries": n}
            out[split][name] = res
    lo, hi, _ = online_windows(day, r["actual"], r["msfc_hold"], r["hp"])
    k = (r["actual"] <= 365) & (r["known_until"] >= 365) & np.isfinite(lo)
    cov = (r["actual"] >= lo) & (r["actual"] <= hi)
    out["windows_msfc_hold"] = {"coverage": float(cov[k].mean()), "n": int(k.sum()),
                                "coverage_no_later_manoeuvre": float(cov[k & ~r["manoeuvred_later"]].mean()),
                                "by_year": {
        y: float(cov[k & np.char.startswith(r["issue"], y)].mean())
        for y in sorted({s[:4] for s in r["issue"]}) if (k & np.char.startswith(r["issue"], y)).sum() >= 20}}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"{out['n_rows']} forecasts of {out['n_objects']} payloads on {out['n_issues']} dates; "
          f"{out['screened_out_per_issue_mean']:.0f} per date screened out for a recent manoeuvre")
    for split in ("all_years", "test_2023plus"):
        for name, res in out[split].items():
            print(f"\n{split} / {name}")
            for m, s in res.items():
                print(f"  {m:12s} {100 * s['median_rel_err']:5.1f} % [{100 * s['ci95'][0]:.1f}, {100 * s['ci95'][1]:.1f}]"
                      f"  n={s['n_reentries']}  no re-entry forecast {100 * s['share_forecast_no_reentry']:.0f} %")
    w = out["windows_msfc_hold"]
    print(f"\n80 % windows (MSFC + correction): coverage {w['coverage']:.2f} (n={w['n']}), {w['coverage_no_later_manoeuvre']:.2f} without later manoeuvres; by year {w['by_year']}")


if __name__ == "__main__":
    main()
