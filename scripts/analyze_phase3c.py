"""Score Phase 3c re-entry forecasts and train a learned residual on top of the physics.

Reads reports/phase3c_rows.npz (from run_phase3c.py) and writes reports/phase3c_results.json.

Metrics, per horizon H (days):
- timing: among objects that actually re-entered within H, |predicted - actual| / actual
  (actual = days from issue to re-entry), as median and share within 20 % (the usual target);
- counts: per issue date, predicted vs actual number of re-entries within H (MAE);
- detection: F1 for "re-enters within H" among objects whose fate over H is known.

Learned parts, all trained on issues before 2023 and tested on 2023 onwards, features known at
the issue date only:
- learned_sw: residual on log(actual / persistence) with space-weather features. Version 1 showed
  this overfits the solar phase it was trained on; kept as the negative control.
- obj: residual on log(actual / oracle), i.e. only the error that is NOT space weather (ballistic
  coefficient drift, fit noise), with object features only. It should be stationary across the
  cycle, so it is applied on top of the analog forecast ("analog_obj").
- 80 % windows around the analog forecast. Version 2 first fitted 10th/90th percentile regressions
  on pre-2023 data; they covered only 40 % on the test years (the error spread moves with the solar
  cycle). They are now online conformal windows: at each issue date, quantiles of log(actual /
  analog) from forecasts whose outcome was already known, per lead-time and perigee group, with
  the level adapted to the recent miss rate (adaptive conformal inference, Gibbs & Candes 2021).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

ROWS = Path("reports/phase3c_rows.npz")
OUT = Path("reports/phase3c_results.json")
HORIZONS = (30, 90, 180, 365)
METHODS = ("sgp4", "persist", "learned_sw", "analog", "analog_obj", "oracle_bstar", "oracle_nols", "oracle")
CORRECTED = ("analog_hold", "analog_taper", "oracle_hold", "oracle_taper")  # Phase 3d, NRLMSIS bias-corrected
MSFC = ("msfc", "blend", "msfc_hold")  # Phase 3e/3f, NASA MSFC solar-cycle forecasts
EXTRA = {"phase3d": CORRECTED, "phase3e": MSFC[:2], "phase3f": MSFC[2:]}
FEATURES = ("hp", "ha", "e", "inc", "log_bc", "bc_fallback", "rb", "f107a", "f107_slope", "ap81",
            "log_persist", "tle_age", "baseline_days", "log_decay_rate", "doy")
OBJ_FEATURES = ("hp", "ha", "e", "inc", "log_bc", "bc_fallback", "rb", "tle_age", "baseline_days", "log_decay_rate",
                "log_bc_ratio")
SPLIT = "2023-01-01"


def load():
    z = np.load(ROWS, allow_pickle=True)
    r = {k: z[k] for k in z.files}
    r["issue"] = r["issue"].astype("U10")
    key = {(i, n): k for k, (i, n) in enumerate(zip(r["issue"], r["norad"]))}
    for phase, names in EXTRA.items():
        shards = sorted(Path("reports").glob(f"{phase}_rows_*.npz"))
        if not shards:
            continue
        for name in names:
            r[name] = np.full(len(r["issue"]), np.nan)
        for sh in shards:
            d = np.load(sh, allow_pickle=True)
            idx = np.array([key[(i, n)] for i, n in zip(d["issue"].astype("U10"), d["norad"])])
            for name in names:
                r[name][idx] = d[name]
    return r


D_BSTAR = 12.741621


def features(r, names=FEATURES):
    persist = np.minimum(r["persist"], 2000.0)
    doy = np.array([int(s[5:7]) * 30 + int(s[8:10]) for s in r["issue"]], float)
    cols = {
        "hp": r["hp"], "ha": r["ha"], "e": r["e"], "inc": r["inc"], "log_bc": np.log(r["bc"]),
        "bc_fallback": r["bc_fallback"].astype(float), "rb": r["rb"].astype(float), "f107a": r["f107a"],
        "f107_slope": r["f107_slope"], "ap81": r["ap81"], "log_persist": np.log1p(persist),
        "tle_age": r["tle_age"], "baseline_days": r["baseline_days"],
        "log_decay_rate": np.log(np.maximum(-r["da_obs"], 1e-4) / r["baseline_days"]), "doy": doy,
        "log_bc_ratio": np.log(r["bc"] / np.clip(D_BSTAR * np.abs(r["bstar"]), 1e-5, None)),
    }
    return np.column_stack([cols[k] for k in names])


def _gbm(**kw):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=0, **kw)


def _apply(base, model, X):
    out = base.copy()
    fin = np.isfinite(base)
    out[fin] = np.maximum(base[fin], 0.5) * np.exp(model.predict(X[fin]))
    return out


def learned(r):
    """Returns corrected forecasts and the 80 % window bounds. Objects the physics says survive
    the 2-year run stay "no re-entry": a residual can't see them."""
    act = r["actual"]
    usable = np.isfinite(act) & (act > 0) & (act <= 730)
    train = usable & (r["issue"] < SPLIT)
    out = {}
    Xs = features(r)
    t = train & np.isfinite(r["persist"])
    m = _gbm().fit(Xs[t], np.log(act[t]) - np.log(np.maximum(r["persist"][t], 0.5)))
    out["learned_sw"] = _apply(r["persist"], m, Xs)
    Xo = features(r, OBJ_FEATURES)
    t = train & np.isfinite(r["oracle"])
    m = _gbm().fit(Xo[t], np.log(act[t]) - np.log(np.maximum(r["oracle"][t], 0.5)))
    out["analog_obj"] = _apply(r["analog"], m, Xo)
    out["lo"], out["hi"] = conformal_windows(r, r["analog"])
    return out


LEAD_EDGES = np.array([0, 30, 90, 180, 365, np.inf])


def conformal_windows(r, pred, target=0.8, gamma=0.05, min_cal=100):
    """Online 80 % windows around `pred`. At each issue date only forecasts whose re-entry had
    already happened are used for calibration (no look-ahead). Groups: predicted lead time x
    perigee band. The miscoverage level alpha moves by gamma x (target miss - observed miss) as
    outcomes arrive, which widens the windows after a run of misses."""
    act = r["actual"]
    day = np.array([date.fromisoformat(s).toordinal() for s in r["issue"]])
    res = np.log(act / np.maximum(pred, 0.5))
    ok = np.isfinite(res) & (act > 0) & (act <= 730)
    lead = np.clip(np.searchsorted(LEAD_EDGES, np.minimum(pred, 1e6), side="right") - 1, 0, len(LEAD_EDGES) - 2)
    grp = lead * 3 + (r["hp"] >= 300) + (r["hp"] >= 450)
    lo = np.full(len(act), np.nan)
    hi = lo.copy()
    alpha = 1 - target
    seen = np.zeros(len(act), bool)
    for d in np.unique(day):
        new = ok & ~seen & (day + act <= d) & np.isfinite(lo)
        if new.any():
            miss = 1 - np.mean((act[new] >= lo[new]) & (act[new] <= hi[new]))
            alpha = float(np.clip(alpha + gamma * ((1 - target) - miss), 0.01, 0.5))
        seen |= new
        cal = ok & (day + act <= d)
        m = day == d
        for g in np.unique(grp[m]):
            c = cal & (grp == g)
            if c.sum() < min_cal:
                c = cal & (lead == g // 3)
            if c.sum() < min_cal:
                continue
            ql, qh = np.quantile(res[c], (alpha / 2, 1 - alpha / 2))
            k = m & (grp == g) & np.isfinite(pred)
            lo[k] = pred[k] * np.exp(ql)
            hi[k] = pred[k] * np.exp(qh)
    return lo, hi


def windows(r, mask, lo, hi, H):
    """Coverage of the 80 % window among objects that re-entered within H with a finite forecast."""
    act = r["actual"]
    k = mask & (act <= H) & (act > 0) & np.isfinite(lo) & (r["known_until"] >= H)
    inside = (act[k] >= lo[k]) & (act[k] <= hi[k])
    width = np.log(np.maximum(hi[k], 0.5) / np.maximum(lo[k], 0.5))
    return {"n": int(k.sum()), "coverage_80": float(inside.mean()) if k.any() else None,
            "median_width_factor": float(np.exp(np.median(width))) if k.any() else None}


def score(r, mask, pred, H):
    act = r["actual"]
    known = mask & ((act <= H) | (r["observed_until"] >= H)) & (r["known_until"] >= H)
    hit = known & (act <= H)
    rel = np.abs(pred[hit] - act[hit]) / np.maximum(act[hit], 1)
    p, a = pred[known] <= H, act[known] <= H
    tp = (p & a).sum()
    f1 = 2 * tp / max(p.sum() + a.sum(), 1)
    counts = []
    for d in np.unique(r["issue"][known]):
        k = known & (r["issue"] == d)
        counts.append(abs(int((pred[k] <= H).sum()) - int((act[k] <= H).sum())))
    return {"n_reentries": int(hit.sum()), "median_rel_err": float(np.median(rel)) if len(rel) else None,
            "within_20pct": float(np.mean(rel < 0.2)) if len(rel) else None, "f1": float(f1),
            "count_mae": float(np.mean(counts)) if counts else None,
            "actual_per_issue": float(hit.sum() / max(len(counts), 1))}


def main():
    r = load()
    global METHODS
    # Score only the issue dates every extra run has finished, so all methods see the same objects.
    for names in EXTRA.values():
        if names[0] in r:
            keep = ~np.isnan(r[names[0]])
            r = {k: v[keep] for k, v in r.items()}
            METHODS = METHODS + names
    preds = {k: r[k] for k in METHODS if k in r}
    lw = learned(r)
    preds.update({k: lw[k] for k in ("learned_sw", "analog_obj")})
    other_windows = {m: conformal_windows(r, r[m]) for m in MSFC if m in r}
    test = r["issue"] >= SPLIT
    subsets = {
        "all_test": test,
        "all_years": np.ones_like(test),
        "test_rocket_bodies": test & r["rb"],
        "test_debris": test & ~r["rb"],
        "test_perigee_lt_300": test & (r["hp"] < 300),
        "test_perigee_300_450": test & (r["hp"] >= 300) & (r["hp"] < 450),
        "test_perigee_ge_450": test & (r["hp"] >= 450),
    }
    for y in sorted({s[:4] for s in r["issue"]}):
        subsets[f"year_{y}"] = np.array([s.startswith(y) for s in r["issue"]])
    subsets["test_e_lt_0.05"] = test & (r["e"] < 0.05)
    subsets["test_e_0.05_0.2"] = test & (r["e"] >= 0.05) & (r["e"] < 0.2)
    subsets["test_e_ge_0.2"] = test & (r["e"] >= 0.2)
    res = {name: {m: {H: score(r, mask, preds[m], H) for H in HORIZONS} for m in METHODS}
           for name, mask in subsets.items()}
    for name, mask in subsets.items():
        res[name]["window_80"] = {H: windows(r, mask, lw["lo"], lw["hi"], H) for H in HORIZONS}
        for m, (lo, hi) in other_windows.items():
            res[name][f"window_80_{m}"] = {H: windows(r, mask, lo, hi, H) for H in HORIZONS}
    OUT.write_text(json.dumps(res, indent=1))

    print(f"rows {len(r['norad'])}, issues {len(np.unique(r['issue']))}, test issues >= {SPLIT}: "
          f"{len(np.unique(r['issue'][test]))}")
    for name in ("all_test", "all_years", "test_rocket_bodies", "test_debris", "test_perigee_lt_300",
                 "test_perigee_300_450", "test_perigee_ge_450", "test_e_lt_0.05", "test_e_0.05_0.2", "test_e_ge_0.2"):
        print(f"\n== {name}: median relative timing error | share within 20 % | F1 | count MAE per issue")
        for H in HORIZONS:
            n = res[name]["oracle"][H]["n_reentries"]
            line = f"H={H:3d} (n={n:5d}) "
            for m in METHODS:
                s = res[name][m][H]
                if s["median_rel_err"] is None:
                    continue
                line += f"| {m} {s['median_rel_err']:.2f} {s['within_20pct']:.2f} {s['f1']:.2f} {s['count_mae']:.0f} "
            w = res[name]["window_80"][H]
            if w["coverage_80"] is not None:
                line += f"| 80% window covers {w['coverage_80']:.2f}, width x{w['median_width_factor']:.2f}"
            print(line)
    print("\n== by year, H=365: 80 % window coverage")
    print(" ".join(f"{k[5:]} {res[k]['window_80'][365]['coverage_80']:.2f}" for k in res
                   if k.startswith("year_") and res[k]["window_80"][365]["coverage_80"] is not None))
    for m in other_windows:
        cov = {k[5:]: res[k]["window_80_" + m][365]["coverage_80"] for k in res if k.startswith("year_")}
        print(f"  around {m}: " + " ".join(f"{y} {c:.2f}" for y, c in cov.items() if c is not None))
        for name in ("all_test", "all_years"):
            w = res[name]["window_80_" + m]
            print(f"    {name}: " + " | ".join(f"H{H} {w[H]['coverage_80']:.2f} x{w[H]['median_width_factor']:.2f}"
                                             for H in HORIZONS if w[H]["coverage_80"] is not None))
    print("\n== by year, H=365: median relative timing error")
    for name in [k for k in res if k.startswith("year_")]:
        s = res[name]
        if s["oracle"][365]["median_rel_err"] is None:
            continue
        print(name, " ".join(f"{m} {s[m][365]['median_rel_err']:.2f}" for m in METHODS),
              f"(n={s['oracle'][365]['n_reentries']})")


if __name__ == "__main__":
    main()
