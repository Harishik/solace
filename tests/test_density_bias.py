from datetime import date

import numpy as np

from solace.risk import decay as D
from solace.risk import density_bias as B


def _synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    day = date(2015, 1, 1).toordinal() + np.arange(n)
    f = rng.uniform(66, 230, n)
    ap = rng.uniform(2, 40, n)
    alt = rng.uniform(320, 520, n)
    sat = rng.integers(0, 2, n)
    g = lambda x: -0.6 + 0.6 * np.clip((x - 70) / 60, 0, 1) - 0.12 * np.clip((x - 160) / 60, 0, 1)  # noqa: E731
    delta = g(f) + 0.1 * (alt - 450) / 100 + 0.04 * np.log1p(ap) + np.where(sat == 1, 0.07, 0.0)
    return B.DailyBias(day, sat, delta + rng.normal(0, 0.01, n), alt, f, ap), g


def test_fit_recovers_shape_and_respects_latency():
    db, g = _synthetic()
    m = B.fit(db, date(2019, 6, 1), latency_days=90)
    f = np.array([75.0, 110.0, 150.0, 200.0])
    got = m(f, np.zeros(4), np.array([450.0]))[:, 0]
    assert np.allclose(got - got[1], g(f) - g(f[1]), atol=0.03)  # shape, up to the constant
    assert abs(m.b - 0.04) < 0.01
    assert m.n == int((db.day <= date(2019, 6, 1).toordinal() - 90).sum())
    assert B.fit(db, date(2015, 3, 1)) is None  # too little history before the cut
    # No extrapolation: beyond the training flux range the correction is flat.
    hi = m(np.array([m.f_range[1], m.f_range[1] + 80]), np.zeros(2), np.array([450.0]))
    assert np.isclose(hi[0, 0], hi[1, 0])


def test_taper_and_table_correction():
    db, _ = _synthetic()
    m = B.fit(db, date(2019, 6, 1), taper=True)
    d = m(np.array([120.0]), np.array([5.0]), D.ALTS)
    assert d[0, D.ALTS == 120][0] == 0.0
    tab = D.DensityTable(date(2020, 1, 1), 1.0, np.zeros((3, len(D.ALTS))), np.full(3, 120.0), np.full(3, 5.0))
    assert np.allclose(tab.corrected(m).logrho, np.repeat(d, 3, axis=0))
    assert tab.corrected(None) is tab
