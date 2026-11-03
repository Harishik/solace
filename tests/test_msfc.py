import importlib.util
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from solace.risk import solar as S

_spec = importlib.util.spec_from_file_location("fetch_msfc", Path(__file__).parents[1] / "scripts" / "fetch_msfc.py")
fetch_msfc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_msfc)

TABLE = """  TABLE 3 ESTIMATES OF 13-MONTH SMOOTH SOLAR ACTIVITY
    TIME         10.7 CM SOLAR FLUX   (F10.7)      GEOMAGNETIC INDEX   (Ap)
 2023.5003   JUL   166.2     160.0     156.4      11.3      10.8      10.5
 2023.5837   AUG   170.8     160.2     153.1      12.0      10.9      10.2
"""


def test_parse_table():
    r = fetch_msfc.parse(TABLE)
    assert r.shape == (2, 7)
    assert np.allclose(r[1], [2023.5837, 170.8, 160.2, 153.1, 12.0, 10.9, 10.2])


def _archive(tmp_path):
    rows = {k: [] for k in ("issue", "t", "f05", "f50", "f95", "ap05", "ap50", "ap95")}
    for month, level in ((date(2024, 1, 1), 100.0), (date(2024, 2, 1), 200.0)):
        t = 2024 + np.arange(36) / 12
        rows["issue"].append(np.full(36, month.toordinal()))
        rows["t"].append(t)
        for k in ("f05", "f50", "f95"):
            rows[k].append(np.full(36, level))
        for k in ("ap05", "ap50", "ap95"):
            rows[k].append(np.full(36, 12.0))
    path = tmp_path / "m.npz"
    np.savez(path, **{k: np.concatenate(v) for k, v in rows.items()})
    return S.MSFCForecasts(path)


def test_uses_only_forecasts_from_earlier_months(tmp_path):
    m = _archive(tmp_path)
    f, ap = m.forecast(date(2024, 2, 20), 30)  # February's forecast is not yet trusted to be public
    assert np.allclose(f, 100.0) and np.allclose(ap, 12.0)
    f, _ = m.forecast(date(2024, 3, 2), 30)
    assert np.allclose(f, 200.0)
    with pytest.raises(ValueError):
        m.forecast(date(2023, 12, 1), 30)


def test_anchor_matches_today_and_fades(tmp_path):
    m = _archive(tmp_path)
    f, _ = m.forecast(date(2024, 3, 2), 2000, anchor=140.0, fade_days=365.0)
    assert np.isclose(f[0], 140.0)
    assert abs(f[-1] - 200.0) < 1.0
