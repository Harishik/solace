"""Asset penetration-risk index: debris flux through the asset's shell x P(penetration) from the BLE.

The catalog only contains objects larger than ~10 cm, but shields fail to millimetre particles.
The flux of smaller debris is extrapolated with a two-segment cumulative power law
N(>d) ~ d^-s anchored at the catalog size: a shallow segment from 10 cm down to 1 cm and a steeper
one below 1 cm, the broad shape ORDEM and MASTER show in LEO. The slopes are the dominant
uncertainty. ORDEM 3.1 publishes its size distributions only as figures (Matney et al. 2019), so a
calibration needs ORDEM or MASTER output; until then every report carries the range spanned by
SLOPE_RANGES: fragment generation in the NASA Standard Breakup Model has N(>L) ~ L^-1.6 to L^-1.71
(Johnson et al. 2001), drag removes small debris faster (flatter), and in-situ data at mm sizes
are steeper.

Only debris and rocket bodies feed the flux: active payloads are few, tracked and manoeuvre around
each other, whereas fragments and spent stages are the population that impacts unannounced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..impact.ble import WhippleShield
from ..ingest.celestrak import TLE
from ..orbit.propagate import R_EARTH
from .shells import ALT_EDGES, ShellGrid, elements, residence_weights


@dataclass
class RiskReport:
    asset: str
    days: float
    area_m2: float
    flux_catalog_per_m2_yr: float
    d_crit_cm: float
    small_debris_multiplier: float
    expected_penetrations: float
    p_penetration: float
    p_penetration_low: float = float("nan")  # over SLOPE_RANGES
    p_penetration_high: float = float("nan")

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def size_multiplier(d_cm: float, catalog_cm: float = 10.0, break_cm: float = 1.0,
                    slope_large: float = 1.3, slope_small: float = 2.1) -> float:
    """N(>d) / N(>catalog size) for the two-segment power law."""
    if d_cm >= break_cm:
        return (d_cm / catalog_cm) ** (-slope_large)
    return (break_cm / catalog_cm) ** (-slope_large) * (d_cm / break_cm) ** (-slope_small)


SLOPE_RANGES = {"slope_large": (1.0, 1.71), "slope_small": (1.6, 2.6)}


def asset_risk(asset: TLE, grid: ShellGrid, days: float = 30.0, area_m2: float = 10.0,
               shield: WhippleShield | None = None, v_rel_kms: float = 10.0, impact_angle_deg: float = 45.0,
               **size_kw) -> RiskReport:
    shield = shield or WhippleShield()
    a, e, _, _ = elements(asset)
    w = residence_weights(a, e)
    r0, r1 = R_EARTH + ALT_EDGES[:-1], R_EARTH + ALT_EDGES[1:]
    vol = 4.0 / 3.0 * np.pi * (r1**3 - r0**3)
    n = grid.counts[..., 1:].sum(axis=(1, 2)) / vol  # rocket bodies + debris, objects / km^3
    flux = float((w * n).sum()) * v_rel_kms  # objects / km^2 / s
    d_crit = float(shield.critical_diameter(v_rel_kms, impact_angle_deg))
    mult = size_multiplier(d_crit, **size_kw)
    base = flux * 1e-6 * area_m2 * days * 86400
    expected = base * mult
    corners = [size_multiplier(d_crit, slope_large=sl, slope_small=ss)
               for sl in SLOPE_RANGES["slope_large"] for ss in SLOPE_RANGES["slope_small"]]
    return RiskReport(asset=asset.name, days=days, area_m2=area_m2,
                      flux_catalog_per_m2_yr=flux * 1e-6 * 3.156e7, d_crit_cm=d_crit,
                      small_debris_multiplier=mult, expected_penetrations=expected,
                      p_penetration=float(1 - np.exp(-expected)),
                      p_penetration_low=float(1 - np.exp(-base * min(corners))),
                      p_penetration_high=float(1 - np.exp(-base * max(corners))))
