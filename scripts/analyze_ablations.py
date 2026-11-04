"""Ablations and MSFC-anchoring sensitivity of the re-entry hindcast against the baseline (paper §4.6).

Each ablation (scripts/run_ablation.py) re-fits the ballistic coefficient and re-forecasts with
the true weather (oracle) and the anchored MSFC flux (msfc). We match rows by (issue, norad) with
the baseline columns, keep forecasts both versions made, and compare the median relative timing
error at the 1-year horizon with the same circular block bootstrap over issue dates as
scripts/significance.py (blocks of 6, B = 2000).

Run from the repo root:  .venv/bin/python scripts/analyze_ablations.py
Writes reports/ablations.json.
"""

from __future__ import annotations

import glob
import importlib.util
import json
from pathlib import Path

import numpy as np

OUT = Path("reports/ablations.json")
TAGS = {"msis00": "NRLMSISE-00 instead of NRLMSIS 2.1",
        "b14_40": "drag-fit baseline 14-40 days (default 21-70)",
        "b40_100": "drag-fit baseline 40-100 days (default 21-70)"}


def _sig():
    spec = importlib.util.spec_from_file_location("sig", Path(__file__).with_name("significance.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SENS = {"msfc_raw": "MSFC median flux as issued, no anchoring",
        "msfc_f182": "anchoring fades over 6 months (default 1 year)",
        "msfc_f730": "anchoring fades over 2 years"}


def jobs():
    """(tag, label, files, {baseline column: ablation column})."""
    for tag, label in TAGS.items():
        yield tag, label, f"reports/ablation_{tag}_rows_*.npz", {"oracle": f"oracle_{tag}", "msfc": f"msfc_{tag}"}
    for col, label in SENS.items():
        yield col, label, "reports/msfcsens_rows_*.npz", {"msfc": col}


def main():
    sig = _sig()
    r = sig.load()
    key = {(i, int(n)): k for k, (i, n) in enumerate(zip(r["issue"], r["norad"]))}
    rng = np.random.default_rng(20261105)
    out = {}
    for tag, label, pattern, pairs in jobs():
        files = sorted(glob.glob(pattern))
        if not files:
            continue
        cols = {}
        for f in files:
            with np.load(f, allow_pickle=True) as z:
                for k in z.files:
                    cols.setdefault(k, []).append(z[k])
        cols = {k: np.concatenate(v) for k, v in cols.items()}
        idx = np.array([key.get((i, int(n)), -1) for i, n in zip(cols["issue"], cols["norad"])])
        ok = idx >= 0
        sub = {k: v[idx[ok]] for k, v in r.items()}
        keep = np.ones(len(sub["issue"]), bool)
        for m, c in pairs.items():
            sub[f"{m}_abl"] = cols[c][ok].astype(float)
            keep &= ~np.isnan(sub[f"{m}_abl"])
        sub = {k: v[keep] for k, v in sub.items()}
        res = {"label": label, "n_issues_done": int(len(set(cols["issue"]))), "n_rows_matched": int(keep.sum()),
               "n_rows_ablation": int(len(cols["issue"]))}
        for split, cut in (("test_2023plus", "2023-01-01"), ("all_years", "2000-01-01")):
            issues = np.array(sorted({s for s in sub["issue"] if s >= cut}))
            if len(issues) == 0:
                continue
            draws = sig.draws(len(issues), rng)
            res[split] = {}
            for m in pairs:
                base, _ = sig.per_issue_stats(sub, sub[m], issues)
                abl, _ = sig.per_issue_stats(sub, sub[f"{m}_abl"], issues)

                def med(errs, sel):
                    return float(np.median(np.concatenate([errs[i] for i in sel])))

                every = range(len(issues))
                db = np.array([med(abl, s) - med(base, s) for s in draws])
                res[split][m] = {
                    "baseline": med(base, every), "ablation": med(abl, every),
                    "diff": med(abl, every) - med(base, every),
                    "diff_ci95": [float(x) for x in np.percentile(db, [2.5, 97.5])],
                    "p_ablation_not_worse": float(np.mean(db <= 0)),
                    "n_reentries": int(sum(len(e) for e in base)),
                }
        out[tag] = res
    OUT.write_text(json.dumps(out, indent=1))
    for tag, res in out.items():
        print(f"\n== {tag}: {res['label']}  ({res['n_issues_done']} issue dates, {res['n_rows_matched']} rows matched)")
        for split in ("test_2023plus", "all_years"):
            for m, s in res.get(split, {}).items():
                print(f"  {split:14s} {m:7s} baseline {100 * s['baseline']:5.1f} %  ablation {100 * s['ablation']:5.1f} %"
                      f"  diff {100 * s['diff']:+5.1f} pts [{100 * s['diff_ci95'][0]:+.1f}, {100 * s['diff_ci95'][1]:+.1f}]"
                      f"  P(not worse) {s['p_ablation_not_worse']:.3f}  n={s['n_reentries']}")


if __name__ == "__main__":
    main()
