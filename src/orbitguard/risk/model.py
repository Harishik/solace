"""Spatio-temporal graph neural network (ST-GNN) shell forecaster.

Architecture: for each snapshot in a look-back window, two graph-convolution layers mix every
cell with its altitude/inclination neighbours (A_hat X W); a GRU then runs along time for each
node; a linear head predicts the *residual* over the SGP4 physics prior for the population
target, and the conjunction-rate target directly.

Predicting the residual means the network only has to learn what physics misses (launches,
breakups, storm-driven decay errors), and it falls back to the physics answer when unsure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .dataset import N_FEATURES, GraphSeries


class GraphConv(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # x: (..., N, d_in); a_hat: (N, N)
        return a_hat @ self.lin(x)


class STGNN(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.g1 = GraphConv(n_features, hidden)
        self.g2 = GraphConv(hidden, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + n_features, 2)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        """x: (B, L, N, F) window -> (B, N, 2): [residual log1p count, log10 conj rate]."""
        B, L, N, F = x.shape
        h = torch.relu(self.g1(x, a_hat))
        h = torch.relu(self.g2(self.drop(h), a_hat))
        h = h.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        _, last = self.gru(h)
        last = last[-1].reshape(B, N, -1)
        return self.head(torch.cat([self.drop(last), x[:, -1]], dim=-1))


@dataclass
class TrainResult:
    model: STGNN
    history: list[dict]
    metrics: dict


def _windows(gs: GraphSeries, idx: np.ndarray, lookback: int):
    xs, ys, pr = [], [], []
    for k in idx:
        if k - lookback + 1 < 0 or np.isnan(gs.Y[k]).any():
            continue
        xs.append(gs.X[k - lookback + 1:k + 1])
        ys.append(gs.Y[k])
        pr.append(gs.prior[k])
    if not xs:
        return None
    t = lambda a: torch.tensor(np.stack(a), dtype=torch.float32)  # noqa: E731
    return t(xs), t(ys), t(pr)


def evaluate(model: STGNN, gs: GraphSeries, idx: np.ndarray, a_hat: torch.Tensor, lookback: int) -> dict:
    """RMSE on log1p total counts (populated cells only) for the GNN against three baselines:

    * persistence -- the grid h steps ahead equals today's;
    * physics -- SGP4 decay forecast;
    * physics + last residual -- the physics forecast plus whatever physics missed over the previous
      horizon (mostly launches). This is the strongest simple competitor to a residual-learning
      model, so it is the bar the GNN has to clear.

    ``storm`` repeats the scores on windows whose horizon contains a day with Ap >= 50.
    """
    w = _windows(gs, idx, lookback)
    if w is None:
        return {}
    x, y, prior = w
    model.eval()
    with torch.no_grad():
        out = model(x, a_hat)
    pred = (prior + out[..., 0]).numpy()
    yt = y[..., 0].numpy()
    keep = [k for k in idx if k - lookback + 1 >= 0 and not np.isnan(gs.Y[k]).any()]
    h = gs.horizon
    pers = gs.persistence[keep]
    phys = gs.prior[keep]
    resid = np.stack([gs.persistence[k] - gs.prior[k - h] if k - h >= 0 else np.zeros_like(gs.prior[k])
                      for k in keep])
    phys_resid = phys + resid
    rate = out[..., 1].numpy()
    preds = {"gnn": pred, "persistence": pers, "physics": phys, "physics_plus_residual": phys_resid}

    def score(sel):
        if not sel.any():
            return None
        m = ((yt > 0) | (pers > 0)) & sel[:, None]
        res = {f"{k}_rmse": float(np.sqrt(np.mean((v - yt)[m] ** 2))) for k, v in preds.items()}
        res["gnn_rate_rmse_log10"] = float(np.sqrt(np.mean((rate - y[..., 1].numpy())[m] ** 2)))
        res["n_windows"] = int(sel.sum())
        return res

    storm = np.array([np.nanmax(gs.ap[k:k + h + 1]) >= 50 for k in keep])
    out_all = score(np.ones(len(keep), bool))
    out_all["storm"] = score(storm)
    return out_all


def train(gs: GraphSeries, a_hat: np.ndarray, lookback: int = 6, epochs: int = 200, lr: float = 3e-3,
          val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 0) -> TrainResult:
    """Chronological split (train | val | test): no future data ever leaks into training."""
    torch.manual_seed(seed)
    T = len(gs.days)
    n_test, n_val = int(T * test_frac), int(T * val_frac)
    idx = np.arange(T)
    tr, va, te = idx[: T - n_val - n_test], idx[T - n_val - n_test: T - n_test], idx[T - n_test:]
    A = torch.tensor(a_hat, dtype=torch.float32)
    model = STGNN()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    w = _windows(gs, tr, lookback)
    if w is None:
        raise ValueError("not enough snapshots for the look-back window")
    x, y, prior = w
    target = torch.stack([y[..., 0] - prior, y[..., 1]], dim=-1)
    best, best_state, hist = np.inf, None, []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x, A), target)
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            m = evaluate(model, gs, va, A, lookback)
            score = m.get("gnn_rmse", float(loss))
            hist.append({"epoch": ep, "train_loss": float(loss), **m})
            if score < best:
                best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return TrainResult(model=model, history=hist, metrics={"val": evaluate(model, gs, va, A, lookback),
                                                           "test": evaluate(model, gs, te, A, lookback)})
