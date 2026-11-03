import numpy as np
import pytest

from orbitguard.impact import SingleWall, SPHSolver, WhippleShield
from orbitguard.impact.sph import Body, lattice_box


def test_whipple_ble_continuous_at_regime_boundaries():
    s = WhippleShield(bumper_cm=0.127, rear_wall_cm=0.48, standoff_cm=11.4)
    for v in (3.0, 7.0):
        lo, hi = s.critical_diameter(v - 1e-6), s.critical_diameter(v + 1e-6)
        assert lo == pytest.approx(hi, rel=1e-4)


def test_whipple_ble_iss_like_shield_stops_about_a_centimetre():
    s = WhippleShield(bumper_cm=0.127, rear_wall_cm=0.48, standoff_cm=11.4)
    assert 0.6 < s.critical_diameter(7.0) < 1.1


def test_obliquity_and_thin_walls_behave():
    s = WhippleShield()
    # Hypervelocity regime: oblique hits have lower normal velocity, so the limit diameter grows.
    assert s.critical_diameter(10.0, 30.0) > s.critical_diameter(10.0, 0.0)
    # An unshielded 2 mm wall fails to sub-mm particles.
    assert SingleWall(thickness_cm=0.2).critical_diameter(7.0) < 0.1


@pytest.mark.parametrize("u", [1.0, 3.0])
def test_sph_symmetric_plate_impact_matches_hugoniot(u):
    # Equal Al plates at relative speed u: interface particle velocity u/2, P = rho0 Us up.
    dx = 0.05
    target = lattice_box(-3, 3, 0, 1.5, dx)
    flyer = lattice_box(-3, 3, 1.5, 3.0, dx)
    sol = SPHSolver([Body("t", target, np.zeros(2)), Body("f", flyer, np.array([0.0, -u]))], dx)
    sol.run(0.08)
    s = sol.state
    p = sol.snapshot()["p"]
    m = (np.abs(s.pos[:, 0]) < 1.0) & (np.abs(s.pos[:, 1] - 1.5) < 0.2)
    up = u / 2
    expected = 2.70e-3 * (5.35 + 1.34 * up) * up
    assert np.median(p[m]) == pytest.approx(expected, rel=0.05)
    assert np.median(s.vel[m, 1]) == pytest.approx(-up, rel=0.05)


def test_sph_conserves_momentum():
    dx = 0.1
    a = lattice_box(-1, 1, 0, 0.5, dx)
    b = lattice_box(-0.5, 0.5, 0.6, 1.1, dx)
    sol = SPHSolver([Body("a", a, np.zeros(2)), Body("b", b, np.array([0.3, -4.0]))], dx)
    p0 = sol.momentum()
    sol.run(0.3)
    assert np.allclose(sol.momentum(), p0, rtol=1e-9, atol=1e-12)
