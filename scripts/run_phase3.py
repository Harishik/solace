"""Phase 3 experiment: build the shell-graph series from cached history, train the ST-GNN, ablate.

Run from the repo root:  .venv/bin/python scripts/run_phase3.py [--rebuild] [--seeds 3]
Writes reports/phase3_series.npz (cached dataset) and reports/phase3_results.json.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from solace.risk import adjacency
from solace.risk.dataset import build_series, load_series, save_series
from solace.risk.history import SpacebookHistory, cached_spacebook_snapshots
from solace.risk.model import train

REPORTS = Path("reports")
SERIES = REPORTS / "phase3_series.npz"
SPACE_WEATHER_FEATURES = [5, 6, 7, 8]  # MSIS density, F10.7, F10.7a, Ap


def weekly(days):
    """Longest run of snapshots exactly 7 days apart (gaps would break the fixed horizon)."""
    best, cur = [], [days[0]]
    for a, b in zip(days, days[1:]):
        cur = cur + [b] if (b - a).days == 7 else [b]
        if len(cur) > len(best):
            best = cur
    return best if len(best) > len(cur) else cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--horizon", type=int, default=4, help="weeks ahead")
    ap.add_argument("--lookback", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)

    if SERIES.exists() and not a.rebuild:
        gs = load_series(SERIES)
    else:
        days = weekly(cached_spacebook_snapshots())
        print(f"building series from {len(days)} weekly snapshots, {days[0]} .. {days[-1]}", flush=True)
        t = time.time()
        gs = build_series(SpacebookHistory().snapshot, a.horizon, days=days, progress=True)
        save_series(gs, SERIES)
        print(f"built in {(time.time() - t) / 60:.1f} min", flush=True)

    A = adjacency()
    no_sw = replace(gs, X=gs.X.copy())
    no_sw.X[..., SPACE_WEATHER_FEATURES] = 0.0
    variants = {"full": (gs, A), "no_space_weather": (no_sw, A), "no_graph": (gs, np.eye(A.shape[0]))}

    results = {"days": [str(gs.days[0]), str(gs.days[-1])], "n_snapshots": len(gs.days),
               "horizon_weeks": gs.horizon, "lookback": a.lookback, "runs": {}}
    for name, (series, adj) in variants.items():
        runs = []
        for seed in range(a.seeds):
            t = time.time()
            r = train(series, adj, lookback=a.lookback, epochs=a.epochs, seed=seed)
            runs.append(r.metrics)
            print(f"{name} seed {seed}: test {json.dumps(r.metrics['test'])} ({time.time() - t:.0f}s)", flush=True)
        results["runs"][name] = runs

    def mean(name, key, split="test", sub=None):
        vals = [(r[split][sub] if sub else r[split]).get(key) for r in results["runs"][name]
                if (r[split][sub] if sub else r[split])]
        return float(np.mean(vals)) if vals else None

    summary = {}
    for name in variants:
        summary[name] = {k: mean(name, k) for k in
                         ("gnn_rmse", "physics_rmse", "persistence_rmse", "physics_plus_residual_rmse")}
        summary[name]["storm_gnn_rmse"] = mean(name, "gnn_rmse", sub="storm")
        summary[name]["storm_physics_plus_residual_rmse"] = mean(name, "physics_plus_residual_rmse", sub="storm")
    results["summary_test_mean"] = summary
    (REPORTS / "phase3_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
