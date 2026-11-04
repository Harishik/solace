import numpy as np
import pytest

from solace.impact import SingleWall, SPHSolver, WhippleShield
from solace.impact.sph import Body, lattice_box


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


def test_sph_strength_sits_on_yield_surface_in_uniaxial_strain():
    # Plate impact well above the Hugoniot elastic limit (~0.6 GPa): shocked material is in uniaxial
    # strain on the von Mises surface, S_yy = -2Y/3 and S_xx = S_zz = +Y/3. Post-shock ringing (no
    # artificial stress term) leaves part of it slightly inside the surface, hence the tolerance.
    dx = 0.05
    target = lattice_box(-3, 3, 0, 1.5, dx)
    flyer = lattice_box(-3, 3, 1.5, 3.0, dx)
    sol = SPHSolver([Body("t", target, np.zeros(2)), Body("f", flyer, np.array([0.0, -0.4]))], dx)
    sol.run(0.08)
    s, Y = sol.state, sol.m.yield_stress
    m = (np.abs(s.pos[:, 0]) < 1.0) & (np.abs(s.vel[:, 1] + 0.2) < 0.04)  # shocked to u_p = 0.2
    syy, sxx = np.median(s.sdev[m, 1]), np.median(s.sdev[m, 0])
    assert -0.75 * Y < syy < -0.4 * Y
    assert sxx == pytest.approx(-syy / 2, rel=0.05)
    vm = np.sqrt(1.5 * (s.sdev[:, :3] ** 2).sum(1) + 3 * s.sdev[:, 3] ** 2)
    assert vm.max() <= Y * (1 + 1e-9)
    # Hydrodynamic run of the same impact carries no deviatoric stress.
    hyd = SPHSolver([Body("t", target, np.zeros(2)), Body("f", flyer, np.array([0.0, -0.4]))], dx,
                    strength=False)
    hyd.run(0.02)
    assert not hyd.state.sdev.any()


def test_axisymmetric_plate_impact_matches_hugoniot_on_and_off_axis():
    from solace.impact.sph import AxisymmetricSPHSolver

    # Same set-up as the planar test, as round plates: the shocked state must not depend on radius.
    dx, u = 0.05, 3.0
    target = lattice_box(0, 3, 0, 1.5, dx)
    flyer = lattice_box(0, 3, 1.5, 3.0, dx)
    sol = AxisymmetricSPHSolver([Body("t", target, np.zeros(2)), Body("f", flyer, np.array([0.0, -u]))], dx)
    sol.run(0.08)
    s, p = sol.state, sol.snapshot()["p"]
    up = u / 2
    expected = 2.70e-3 * (5.35 + 1.34 * up) * up
    for lo, hi in ((0.0, 0.3), (0.5, 1.5)):  # next to the axis and away from it
        m = (s.pos[:, 0] >= lo) & (s.pos[:, 0] < hi) & (np.abs(s.pos[:, 1] - 1.5) < 0.2)
        assert np.median(p[m]) == pytest.approx(expected, rel=0.08)
        assert np.median(s.vel[m, 1]) == pytest.approx(-up, rel=0.05)
    assert (s.pos[:, 0] >= 0).all()


def test_tillotson_reference_state_and_compression():
    from solace.impact.sph import TILLOTSON_AL as T, tillotson

    assert abs(tillotson(np.array([T.rho0]), np.array([0.0]))[0]) < 1e-12
    p = tillotson(np.array([1.1 * T.rho0]), np.array([0.0]))[0]
    assert p == pytest.approx(T.A * 0.1 + T.B * 0.01, rel=1e-6)  # cold compressed branch
    # Hot, expanded material approaches the ideal-gas-like a rho E.
    rho, e = np.array([0.01 * T.rho0]), np.array([20.0])
    assert tillotson(rho, e)[0] == pytest.approx(T.a * rho[0] * e[0], rel=0.05)


def _axi(**kw):
    from solace.impact import WhippleImpact

    return WhippleImpact(projectile_diameter=1.0, resolution=0.2, width=6.0, standoff=3.0, axisymmetric=True,
                         **kw).build()


def test_flaws_cover_every_particle_and_grow_damage_only_in_tension():
    sol = _axi(fracture=True, seed=3)
    assert sol.n_flaws.min() >= 1
    assert np.all(np.diff(sol.flaw_eps[sol.flaw_owner == sol.flaw_owner[0]]) >= 0)
    n = len(sol.damage)
    sol._eps_local = np.zeros(n)
    sol._grow_damage(0.1)
    assert sol.damage.max() == 0.0  # no tension, no damage
    sol._eps_local = np.full(n, 1.0)  # far beyond every flaw
    for _ in range(50):
        sol._grow_damage(0.1)
    assert sol.damage.min() > 0.99


def test_artificial_stress_and_fracture_conserve_axial_momentum():
    sol = _axi(eos="tillotson", artificial_stress=0.3, fracture=True)
    p0 = sol.momentum()[1]
    for _ in range(150):
        sol.step()
    assert abs(sol.momentum()[1] - p0) < 1e-6 * abs(p0)
