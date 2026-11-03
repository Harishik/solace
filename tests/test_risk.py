from datetime import datetime, timezone

import numpy as np
import pytest

from solace.ingest.celestrak import parse_tle_text
from solace.risk import ALT_EDGES, adjacency, asset_risk, build_grid
from solace.risk.baseline import sgp4_forecast
from solace.risk.index import size_multiplier
from solace.risk.shells import residence_weights

ISS = """ISS (ZARYA)
1 25544U 98067A   26294.50000000  .00016717  00000+0  30164-3 0  9993
2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49815361 12345
"""
DEB = """FENGYUN 1C DEB
1 29740U 99025BXK 26294.50000000  .00000400  00000+0  18000-3 0  9991
2 29740  98.8000 120.0000 0050000  90.0000 270.0000 14.30000000 12345
"""
WHEN = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)


def test_residence_weights_sum_to_one_and_spread_with_eccentricity():
    circ = residence_weights(6378.137 + 800, 0.0)
    ecc = residence_weights(6378.137 + 1000, 0.05)
    assert circ.sum() == pytest.approx(1.0) and ecc.sum() == pytest.approx(1.0)
    assert (circ > 0).sum() == 1 and (ecc > 0).sum() > 5


def test_grid_places_objects_in_the_right_cells():
    g = build_grid(parse_tle_text(ISS + DEB), when=WHEN)
    assert g.total.sum() == pytest.approx(2.0)
    iss_band = np.searchsorted(ALT_EDGES, 420, side="right") - 1
    assert g.counts[iss_band - 1:iss_band + 1, :, 0].sum() == pytest.approx(1.0)  # payload
    assert g.counts[..., 2].sum() == pytest.approx(1.0)  # debris kind


def test_physics_forecast_at_zero_horizon_equals_today():
    tles = parse_tle_text(ISS + DEB)
    assert np.allclose(sgp4_forecast(tles, WHEN, 0.0).counts, build_grid(tles, when=WHEN).counts)


def test_adjacency_is_symmetric_normalised():
    A = adjacency(4, 3)
    assert np.allclose(A, A.T)
    assert np.all(np.linalg.eigvalsh(A) <= 1 + 1e-9)


def test_size_multiplier_continuous_at_break():
    assert size_multiplier(1.0 + 1e-9) == pytest.approx(size_multiplier(1.0 - 1e-9), rel=1e-6)
    assert size_multiplier(10.0) == pytest.approx(1.0)


def test_asset_risk_grows_with_area_and_time():
    tles = parse_tle_text(ISS + DEB)
    g = build_grid(tles, when=WHEN)
    a = asset_risk(tles[0], g, days=30, area_m2=10).expected_penetrations
    b = asset_risk(tles[0], g, days=60, area_m2=20).expected_penetrations
    assert b == pytest.approx(4 * a)
