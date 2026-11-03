"""Phase 3b: debris-cohort decay forecast at 1, 3 and 6 months, with ablations and re-entry scores.

Run from the repo root:  .venv/bin/python scripts/run_phase3b.py [--rebuild]
Writes reports/phase3b_h{H}.npz (cached datasets) and reports/phase3b_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_phase3 import SPACE_WEATHER_FEATURES, weekly  # noqa: E402

from orbitguard.risk import adjacency  # noqa: E402
from orbitguard.risk.cohort import build_cohort_series  # noqa: E402
from orbitguard.risk.dataset import GraphSeries  # noqa: E402
from orbitguard.risk.history import SpacebookHistory, cached_spacebook_snapshots  # noqa: E402
from orbitguard.risk.model import _windows, train  # noqa: E402

REPORTS = Path("reports")
HORIZONS = (4, 13, 26)  # weeks: ~1, 3, 6 months


def save(gs: GraphSeries, path: Path):
    np.savez_compressed(path, days=np.array([d.isoformat() for d in gs.days]), X=gs.X, Y=gs.Y, prior=gs.prior,
                        persistence=gs.persistence, ap=gs.ap, horizon=gs.horizon,
                        re_act=gs.reentries_actual, re_phys=gs.reentries_physics)


def load(path: Path) -> GraphSeries:
    z = np.load(path)
    gs = GraphSeries(days=[date.fromisoformat(d) for d in z["days"]], X=z["X"], Y=z["Y"], prior=z["prior"],
                     persistence=z["persistence"], ap=z["ap"], horizon=int(z["horizon"]))
    gs.reentries_actual, gs.reentries_physics = z["re_act"], z["re_phys"]
    return gs


def reentry_scores(model, gs: GraphSeries, idx, A, lookback):
    """Mean absolute error of the predicted number of cohort re-entries over the horizon."""
    w = _windows(gs, idx, lookback)
    keep = [k for k in idx if k - lookback + 1 >= 0 and not np.isnan(gs.Y[k]).any()]
    x, _, prior = w
    model.eval()
    with torch.no_grad():
        pred = (prior + model(x, torch.tensor(A, dtype=torch.float32))[..., 0]).numpy()
    now = np.expm1(gs.persistence[keep]).sum(1)
    gnn = now - np.expm1(np.clip(pred, 0, None)).sum(1)
    act = gs.reentries_actual[keep]
    phys = gs.reentries_physics[keep]
    return {"actual_mean": float(np.mean(act)), "gnn_mae": float(np.mean(np.abs(gnn - act))),
            "physics_mae": float(np.mean(np.abs(phys - act))), "zero_mae": float(np.mean(np.abs(act)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--lookback", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)
    paths = {h: REPORTS / f"phase3b_h{h}.npz" for h in HORIZONS}
    if a.rebuild or not all(p.exists() for p in paths.values()):
        days = weekly(cached_spacebook_snapshots())
        print(f"building cohort series from {len(days)} snapshots", flush=True)
        t = time.time()
        series = build_cohort_series(SpacebookHistory().snapshot, days, HORIZONS, progress=True)
        for h, gs in series.items():
            save(gs, paths[h])
        print(f"built in {(time.time() - t) / 60:.1f} min", flush=True)
    series = {h: load(p) for h, p in paths.items()}

    A = adjacency()
    plan = [("full", 2), ("no_space_weather", 2), ("no_graph", 1)]
    results = {}
    for h, gs in series.items():
        no_sw = replace(gs, X=gs.X.copy())
        no_sw.X[..., SPACE_WEATHER_FEATURES] = 0.0
        no_sw.reentries_actual, no_sw.reentries_physics = gs.reentries_actual, gs.reentries_physics
        variants = {"full": (gs, A), "no_space_weather": (no_sw, A), "no_graph": (gs, np.eye(A.shape[0]))}
        T = len(gs.days)
        test_idx = np.arange(T - int(T * 0.2), T)
        results[h] = {}
        for name, seeds in plan:
            s, adj = variants[name]
            runs = []
            for seed in range(seeds):
                t = time.time()
                r = train(s, adj, lookback=a.lookback, epochs=a.epochs, seed=seed)
                m = r.metrics["test"]
                m["reentries"] = reentry_scores(r.model, s, test_idx, adj, a.lookback)
                runs.append(m)
                print(f"h={h} {name} seed {seed} ({time.time() - t:.0f}s): gnn {m['gnn_rmse']:.4f} "
                      f"physics {m['physics_rmse']:.4f} persistence {m['persistence_rmse']:.4f} | storm gnn "
                      f"{(m['storm'] or {}).get('gnn_rmse', float('nan')):.4f} physics "
                      f"{(m['storm'] or {}).get('physics_rmse', float('nan')):.4f} | reentry MAE gnn "
                      f"{m['reentries']['gnn_mae']:.1f} physics {m['reentries']['physics_mae']:.1f}", flush=True)
            results[h][name] = runs
        (REPORTS / "phase3b_results.json").write_text(json.dumps(results, indent=1, default=float))

    print("\nmean test RMSE (all | storm) and re-entry MAE")
    for h, rr in results.items():
        for name, runs in rr.items():
            g = np.mean([r["gnn_rmse"] for r in runs])
            storm_runs = [r["storm"]["gnn_rmse"] for r in runs if r["storm"]]
            gs_ = np.mean(storm_runs) if storm_runs else np.nan
            ph = runs[0]["physics_rmse"]
            phs = runs[0]["storm"]["physics_rmse"] if runs[0]["storm"] else np.nan
            re = np.mean([r["reentries"]["gnn_mae"] for r in runs])
            print(f"h={h:2d} {name:17s} gnn {g:.4f} | {gs_:.4f}   physics {ph:.4f} | {phs:.4f}   "
                  f"persistence {runs[0]['persistence_rmse']:.4f}   reentry MAE gnn {re:.1f} "
                  f"physics {runs[0]['reentries']['physics_mae']:.1f} (actual mean "
                  f"{runs[0]['reentries']['actual_mean']:.1f})")


if __name__ == "__main__":
    main()
