from datetime import date

import numpy as np

from orbitguard.risk import decay as D


def _exp_table(rho0=1e-11, h0=300.0, scale=50.0, days=400):
    lr = np.log(rho0) - (D.ALTS - h0) / scale
    return D.DensityTable(date(2020, 1, 1), 1.0, np.tile(lr, (days, 1)))


def test_circular_rate_matches_closed_form():
    tab = _exp_table()
    a = D.RE + 300.0
    bc = 0.02
    da, de = D.rates(np.array([a]), np.array([0.0]), np.radians([90.0]), np.array([bc]), np.array([0.0]), tab)
    # Polar orbit: no co-rotation correction. da/dt = -rho B sqrt(mu a).
    expect = -1e-11 * bc * np.sqrt(D.MU * a * 1e3) * D.DAY / 1e3
    assert abs(da[0] / expect - 1) < 1e-3
    assert abs(de[0]) < 1e-12


def test_eccentric_orbit_circularises():
    tab = _exp_table()
    a0, e0 = D.RE + 900.0, 0.05  # perigee ~ 570 km, apogee ~ 1230 km
    a, e, _, _, _ = D.propagate([a0], [e0], [60.0], [0.05], [0.0], 300.0, tab)
    assert e[0] < e0 and a[0] < a0
    # Drag at perigee lowers apogee much more than perigee.
    assert (a0 * (1 + e0) - a[0] * (1 + e[0])) > 5 * (a0 * (1 - e0) - a[0] * (1 - e[0]))


def test_lifetime_scales_inversely_with_ballistic_coefficient():
    tab = _exp_table(days=5000)
    _, _, t, down, _ = D.propagate([D.RE + 350] * 2, [0.0] * 2, [51.6] * 2, [0.01, 0.02], [0.0] * 2, 4999.0, tab)
    assert down.all()
    assert abs(t[0] / t[1] - 2.0) < 0.05


def test_fit_bc_recovers_truth():
    tab = _exp_table()
    truth = np.array([0.005, 0.03, 0.1])
    a0 = D.RE + np.array([450.0, 400.0, 380.0])
    e0 = np.array([0.001, 0.01, 0.002])
    inc = np.array([98.0, 51.6, 74.0])
    a1 = D.propagate(a0, e0, inc, truth, np.zeros(3), 42.0, tab)[0]
    est = D.fit_bc(a0, e0, inc, np.zeros(3), a1, np.full(3, 42.0), tab, truth * 3)
    assert np.allclose(est, truth, rtol=0.02)


def _vec(e, inc, raan=0.0, argp=0.0):
    ev, h = D._to_vectors(np.array([e]), np.radians([inc]), np.radians([raan]), np.radians([argp]))
    return ev, np.sqrt(1 - e**2) * h


def test_j2_apsidal_and_nodal_rates():
    a = D.RE + 700.0
    n = np.sqrt(D.MU / (a * 1e3) ** 3)
    base = 0.75 * n * D.J2 * (D.RE / a) ** 2 * D.DAY  # rad/day
    for inc in (20.0, 63.4, 98.0):
        ev, jv = _vec(0.01, inc, 30.0, 40.0)
        de, dj = D.secular_rates(np.array([a]), ev, jv)
        ci = np.cos(np.radians(inc))
        h = jv / np.linalg.norm(jv)
        # node regression: dh/dt = raan_dot * z x h
        raan_dot = np.dot(dj[0], np.cross([0, 0, 1], h[0])) / np.linalg.norm(np.cross([0, 0, 1], h[0])) ** 2
        assert abs(raan_dot / (-2 * base * ci / (1 - 1e-4) ** 2) - 1) < 1e-3
        # |e| unchanged by J2
        assert abs(np.dot(de[0], ev[0])) < 1e-12


def test_third_body_ring_regresses_node_and_conserves_projection():
    a = D.RE + 20000.0
    n = np.sqrt(D.MU / (a * 1e3) ** 3)
    r3 = 3.844e8
    ev, jv = _vec(1e-6, 10.0)
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    dj = 0
    for th in angles:  # perturber averaged over a circular equatorial ring
        pos = np.array([[np.cos(th), np.sin(th), 0.0]]) * r3
        _, djb = D.secular_rates(np.array([a]), ev, jv, ((D.GM_MOON, pos),))
        _, dj0 = D.secular_rates(np.array([a]), ev, jv)
        dj = dj + (djb - dj0) / len(angles)
        assert abs(np.dot(djb - dj0, pos[0] / r3)) <= 1e-6 * np.abs(djb - dj0).max() + 1e-20  # j . d conserved
    h = jv[0] / np.linalg.norm(jv[0])
    zxh = np.cross([0, 0, 1], h)
    rate = np.dot(dj[0], zxh) / np.dot(zxh, zxh)
    expect = -0.75 * D.GM_MOON / (r3**3 * n) * np.cos(np.radians(10.0)) * D.DAY
    assert abs(rate / expect - 1) < 1e-3


def test_vector_roundtrip():
    ev, jv = _vec(0.3, 27.0, 120.0, 250.0)
    inc, raan, argp = D._to_angles(ev, jv / np.linalg.norm(jv))
    assert np.allclose(np.degrees([inc[0], raan[0] % (2 * np.pi), argp[0] % (2 * np.pi)]), [27.0, 120.0, 250.0])
