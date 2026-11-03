"""Ballistic limit equations (BLE) for spacecraft walls.

A BLE gives the critical projectile diameter ``d_c``: particles larger than this perforate (or
detach spall from) the rear wall. These are the same empirical fits that NASA's BUMPER code uses,
calibrated against thousands of light-gas-gun tests.

Units throughout follow the source papers: cm, g/cm^3, km/s, ksi; obliquity theta is measured from
the surface normal.

References
----------
Christiansen, E. L. (1993/2003), "Meteoroid/Debris Shielding", NASA TP-2003-210788, eqs. for the
Whipple shield (three velocity regimes) and the single-wall Cour-Palais equation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AL_DENSITY = 2.70  # g/cm^3, aluminium projectile (dominant debris material)


@dataclass(frozen=True)
class WhippleShield:
    bumper_cm: float = 0.16  # t_b
    rear_wall_cm: float = 0.32  # t_w
    standoff_cm: float = 10.0  # S
    bumper_density: float = 2.70  # rho_b, g/cm^3
    rear_wall_yield_ksi: float = 47.0  # sigma, Al 6061-T6 ~ 40 ksi, 2219-T87 ~ 57 ksi

    def critical_diameter(self, v_kms, theta_deg=0.0, rho_p: float = AL_DENSITY):
        return whipple_critical_diameter(self, v_kms, theta_deg, rho_p)


def _ballistic(s: WhippleShield, v, cos_t, rho_p):
    num = s.rear_wall_cm * np.sqrt(s.rear_wall_yield_ksi / 40.0) + s.bumper_cm
    return (num / (0.6 * cos_t ** (5 / 3) * np.sqrt(rho_p) * v ** (2 / 3))) ** (18 / 19)


def _hyper(s: WhippleShield, vn, rho_p):
    return (3.918 * s.rear_wall_cm ** (2 / 3) * rho_p ** (-1 / 3) * s.bumper_density ** (-1 / 9)
            * vn ** (-2 / 3) * s.standoff_cm ** (1 / 3) * (s.rear_wall_yield_ksi / 70.0) ** (1 / 3))


def whipple_critical_diameter(s: WhippleShield, v_kms, theta_deg=0.0, rho_p: float = AL_DENSITY):
    """Critical projectile diameter (cm) for a Whipple shield. Vectorised over v and theta."""
    v = np.asarray(v_kms, dtype=float)
    cos_t = np.cos(np.deg2rad(np.asarray(theta_deg, dtype=float)))
    cos_t = np.maximum(cos_t, 1e-3)
    vn = v * cos_t
    d_ball = _ballistic(s, v, cos_t, rho_p)
    d_hyper = _hyper(s, vn, rho_p)
    # Intermediate regime: linear blend between the 3 km/s and 7 km/s end points (in normal velocity).
    d3 = _ballistic(s, 3.0 / cos_t, cos_t, rho_p)
    d7 = _hyper(s, 7.0, rho_p)
    d_mid = d3 * (1.75 - vn / 4.0) + d7 * (vn / 4.0 - 0.75)
    return np.where(vn <= 3.0, d_ball, np.where(vn >= 7.0, d_hyper, d_mid))


@dataclass(frozen=True)
class SingleWall:
    """Unshielded wall (solar arrays, radiators, unprotected bus panels)."""

    thickness_cm: float = 0.2
    brinell_hardness: float = 95.0  # Al 6061-T6
    density: float = 2.70
    sound_speed_kms: float = 5.1
    # Wall thickness must exceed k * crater depth to avoid failure: 1.8 perforation, 2.2 detached spall.
    k: float = 1.8

    def critical_diameter(self, v_kms, theta_deg=0.0, rho_p: float = AL_DENSITY):
        return single_wall_critical_diameter(self, v_kms, theta_deg, rho_p)


def single_wall_critical_diameter(w: SingleWall, v_kms, theta_deg=0.0, rho_p: float = AL_DENSITY):
    """Invert the Cour-Palais crater-depth equation
    p = 5.24 d^(19/18) BH^-0.25 (rho_p/rho_t)^0.5 (V_n / C)^(2/3)
    for the diameter at which k * p equals the wall thickness."""
    vn = np.asarray(v_kms, dtype=float) * np.maximum(np.cos(np.deg2rad(theta_deg)), 1e-3)
    p_needed = w.thickness_cm / w.k
    coeff = (5.24 * w.brinell_hardness ** -0.25 * np.sqrt(rho_p / w.density)
             * (vn / w.sound_speed_kms) ** (2 / 3))
    return (p_needed / coeff) ** (18 / 19)


def penetration_probability_given_impact(d_crit_cm, d_min_cm: float = 0.1, slope: float = 2.6):
    """P(d > d_crit | impact by a particle larger than d_min), for a cumulative power-law size
    distribution N(>d) ~ d^-slope. ORDEM/MASTER give slopes around 2.5-2.7 in the mm-cm range."""
    d = np.maximum(np.asarray(d_crit_cm, dtype=float), d_min_cm)
    return (d / d_min_cm) ** (-slope)
