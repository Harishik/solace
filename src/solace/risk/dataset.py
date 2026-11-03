"""Turn a sequence of catalog snapshots into graph time-series tensors.

Node features at snapshot k (per (alt, inc) cell):
    0-2  log1p residence-weighted counts: payload, rocket body, debris
    3    mean B* (scaled)
    4    log10 kinetic conjunction rate
    5    log10 NRLMSIS density at the band altitude under that day's drivers
    6-8  F10.7, 81-day F10.7, Ap (normalised, broadcast to all nodes)
    9-10 band altitude and inclination midpoints (normalised, static)
    11   log1p hybrid physics forecast of the total count at k + horizon (payloads persist,
         debris and rocket bodies decay under SGP4)

Targets at snapshot k:
    log1p total count at k + horizon, and log10 kinetic conjunction rate at k + horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np

from .baseline import hybrid_forecast
from .drivers import drivers_for, load_space_weather, shell_density
from .shells import ALT_EDGES, INC_EDGES, ShellGrid, build_grid, kinetic_conjunction_rate

N_FEATURES = 12
EPS_RATE = 1e-9


def node_features(grid: ShellGrid, drv: np.ndarray, rho: np.ndarray, prior: ShellGrid) -> np.ndarray:
    n_alt, n_inc = grid.shape
    alt_mid = 0.5 * (ALT_EDGES[:-1] + ALT_EDGES[1:])
    inc_mid = 0.5 * (INC_EDGES[:-1] + INC_EDGES[1:])
    f = np.zeros((n_alt, n_inc, N_FEATURES))
    f[..., 0:3] = np.log1p(grid.counts)
    f[..., 3] = np.clip(grid.bstar * 1e3, -5, 5)
    f[..., 4] = np.log10(kinetic_conjunction_rate(grid) + EPS_RATE)
    f[..., 5] = np.log10(rho)[:, None]
    f[..., 6] = (drv[0] - 150) / 50
    f[..., 7] = (drv[1] - 150) / 50
    f[..., 8] = np.log1p(drv[2]) - 2
    f[..., 9] = ((alt_mid - 1000) / 600)[:, None]
    f[..., 10] = ((inc_mid - 70) / 40)[None, :]
    f[..., 11] = np.log1p(prior.total)
    return f.reshape(n_alt * n_inc, N_FEATURES)


def targets(future: ShellGrid) -> np.ndarray:
    return np.stack([np.log1p(future.total), np.log10(kinetic_conjunction_rate(future) + EPS_RATE)],
                    axis=-1).reshape(-1, 2)


@dataclass
class GraphSeries:
    days: list[date]
    X: np.ndarray  # (T, N, F)
    Y: np.ndarray  # (T, N, 2), NaN where k + horizon is beyond the data
    prior: np.ndarray  # (T, N) physics-baseline log1p total
    persistence: np.ndarray  # (T, N) log1p total at k
    ap: np.ndarray  # (T,) daily Ap, for storm-period evaluation
    horizon: int = 4  # forecast horizon in snapshot steps


def build_series(snapshots, horizon_steps: int, days: list[date] | None = None, progress: bool = False) -> GraphSeries:
    """Build the graph time series.

    ``snapshots`` is either a dict day -> TLE list, or a callable day -> TLE list used together with
    ``days``; the callable form loads one snapshot at a time, so years of weekly history (tens of
    millions of element sets) never sit in memory together. Snapshots must be evenly spaced, and
    the horizon is in snapshot steps.
    """
    if callable(snapshots):
        load = snapshots
        days = sorted(days)
    else:
        days = sorted(snapshots)
        load = snapshots.__getitem__
    step = (days[1] - days[0]).days if len(days) > 1 else 7
    horizon_days = horizon_steps * step
    sw = load_space_weather()
    drv = drivers_for(days, sw)
    X, grids, prior, pers = [], [], [], []
    for k, d in enumerate(days):
        tles = load(d)
        noon = datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc)
        g = build_grid(tles, when=noon)
        pr = hybrid_forecast(tles, noon, horizon_days, today=g)
        rho = shell_density(*drv[k], when=np.datetime64(d.isoformat() + "T12:00"))
        X.append(node_features(g, drv[k], rho, pr))
        grids.append(targets(g))
        prior.append(np.log1p(pr.total).ravel())
        pers.append(np.log1p(g.total).ravel())
        if progress and (k % 25 == 0 or k == len(days) - 1):
            print(f"[{k + 1}/{len(days)}] {d}: {len(tles)} objects", flush=True)
    nan = np.full_like(grids[0], np.nan)
    Y = [grids[k + horizon_steps] if k + horizon_steps < len(days) else nan for k in range(len(days))]
    return GraphSeries(days=days, X=np.stack(X), Y=np.stack(Y), prior=np.stack(prior),
                       persistence=np.stack(pers), ap=drv[:, 2], horizon=horizon_steps)


def save_series(gs: GraphSeries, path) -> None:
    np.savez_compressed(path, days=np.array([d.isoformat() for d in gs.days]), X=gs.X, Y=gs.Y,
                        prior=gs.prior, persistence=gs.persistence, ap=gs.ap, horizon=gs.horizon)


def load_series(path) -> GraphSeries:
    z = np.load(path)
    return GraphSeries(days=[date.fromisoformat(d) for d in z["days"]], X=z["X"], Y=z["Y"], prior=z["prior"],
                       persistence=z["persistence"], ap=z["ap"], horizon=int(z["horizon"]))
