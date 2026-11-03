import numpy as np
import pytest

from orbitguard.ingest.celestrak import parse_tle_text
from orbitguard.orbit.conjunction import collision_probability
from orbitguard.orbit.propagate import Propagator, jday_array, orbit_shell, teme_to_geodetic

ISS = """ISS (ZARYA)
1 25544U 98067A   26294.50000000  .00016717  00000+0  30164-3 0  9993
2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49815361 12345
"""


def test_parse_tle_with_and_without_names():
    tles = parse_tle_text(ISS + "\n" + "\n".join(ISS.splitlines()[1:]))
    assert len(tles) == 2
    assert tles[0].name == "ISS (ZARYA)"
    assert all(t.norad_id == 25544 for t in tles)


def test_orbit_shell_iss_altitude():
    p, a = orbit_shell(parse_tle_text(ISS)[0])
    assert 380 < p < 440 and 380 < a < 440


def test_propagation_radius_and_geodetic():
    from datetime import datetime, timezone

    prop = Propagator(parse_tle_text(ISS))
    jd, fr = jday_array(datetime(2026, 10, 21, 12, tzinfo=timezone.utc), np.arange(0, 5400, 60))
    r, v, err = prop.states(jd, fr)
    assert (err == 0).all()
    radius = np.linalg.norm(r[0], axis=-1)
    assert np.all((6700 < radius) & (radius < 6850))
    lat, lon, alt = teme_to_geodetic(r[0], jd, fr)
    assert np.abs(lat).max() == pytest.approx(51.6, abs=0.5)
    assert np.all((370 < alt) & (alt < 450))


def test_pc_small_disk_matches_analytic():
    # Head-on miss of 0 with isotropic sigma: Pc ~ R^2 / (2 sigma^2) for R << sigma.
    sigma, R = 1.0, 0.01
    pc = collision_probability(np.zeros(3), np.array([0, 0, 10.0]), np.eye(3) * sigma**2, R)
    assert pc == pytest.approx(R**2 / (2 * sigma**2), rel=1e-3)


def test_pc_offset_miss_matches_rice():
    # Isotropic case has a closed form via the noncentral chi-square (Rice) distribution.
    from scipy.stats import ncx2

    sigma, R, d = 0.3, 0.2, 0.5
    pc = collision_probability(np.array([d, 0, 0]), np.array([0, 0, 7.0]), np.eye(3) * sigma**2, R)
    expected = ncx2.cdf((R / sigma) ** 2, df=2, nc=(d / sigma) ** 2)
    assert pc == pytest.approx(expected, rel=1e-4)


def test_pc_large_disk_is_one():
    pc = collision_probability(np.array([0.01, 0, 0]), np.array([1.0, 1, 0]), np.eye(3) * 0.01, 5.0)
    assert pc == pytest.approx(1.0, abs=1e-9)


def test_pc_ignores_miss_component_along_velocity():
    cov = np.diag([0.2, 0.5, 0.1]) ** 2
    v = np.array([0, 0, 7.5])
    a = collision_probability(np.array([0.1, 0.05, 0.0]), v, cov, 0.02)
    b = collision_probability(np.array([0.1, 0.05, 3.0]), v, cov, 0.02)
    assert a == pytest.approx(b, rel=1e-9)


def test_alpha5_catalog_numbers():
    from orbitguard.ingest.celestrak import decode_catnr

    assert decode_catnr("25544") == 25544
    assert decode_catnr("A0000") == 100000
    assert decode_catnr("E8493") == 148493
    assert decode_catnr("Z9999") == 339999
