"""2-D planar smoothed particle hydrodynamics (SPH) hydrocode for hypervelocity impact.

Physics:
* Mie-Grueneisen equation of state with a linear Us-up Hugoniot (aluminium defaults).
* Monaghan artificial viscosity to capture shocks.
* Cubic-spline kernel, KD-tree neighbour search, symplectic leap-frog time integration.
* Hydrodynamic only (no deviatoric strength). At 5-15 km/s impact pressures are 50-300 GPa against a
  yield of ~0.3 GPa, so strength barely affects the bumper hole or the debris cloud; it does matter
  for rear-wall bending, which this solver doesn't predict. A tensile cut-off stands in for spall.

Units are mm, microseconds, grams. Velocities in mm/us equal km/s, and pressure comes out in
g/(mm us^2) = 1 TPa, so 1 GPa = 1e-3.

Being 2-D planar, a "sphere" is really an infinite cylinder. Use this for cloud shape, timing and
qualitative damage; use the BLE for quantitative perforation limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Material:
    rho0: float = 2.70e-3  # g/mm^3
    c0: float = 5.35  # mm/us, bulk sound speed
    s: float = 1.34  # Us = c0 + s up
    gamma0: float = 2.0  # Grueneisen parameter
    p_min: float = -1.0e-3  # tensile cut-off (~ -1 GPa spall strength)


ALUMINIUM = Material()


def mie_gruneisen(rho, e, m: Material):
    mu = rho / m.rho0 - 1.0
    denom = np.maximum(1.0 - (m.s - 1.0) * mu, 0.1) ** 2
    p_comp = m.rho0 * m.c0**2 * mu * (1.0 + (1.0 - 0.5 * m.gamma0) * mu) / denom
    p_tens = m.rho0 * m.c0**2 * mu
    p = np.where(mu > 0, p_comp, p_tens) + m.gamma0 * m.rho0 * e
    return np.maximum(p, m.p_min)


def sound_speed(rho, e, p, m: Material):
    d = 1e-6 * m.rho0
    dpdrho = (mie_gruneisen(rho + d, e, m) - mie_gruneisen(rho - d, e, m)) / (2 * d)
    c2 = dpdrho + p / rho**2 * m.gamma0 * m.rho0
    return np.sqrt(np.clip(c2, (0.5 * m.c0) ** 2, None))


def kernel_grad(dx, r, h):
    """Gradient of the 2-D cubic spline kernel; dx (n,2), r (n,)."""
    sigma = 10.0 / (7.0 * np.pi * h * h)
    q = r / h
    dw = np.where(q < 1.0, sigma * (-3.0 * q + 2.25 * q * q),
                  np.where(q < 2.0, -0.75 * sigma * (2.0 - q) ** 2, 0.0)) / h
    return (dw / np.maximum(r, 1e-12))[:, None] * dx


@dataclass
class Body:
    name: str
    pos: np.ndarray
    vel: np.ndarray


def lattice_disk(center, radius, dx):
    g = np.arange(-radius, radius + dx / 2, dx)
    X, Y = np.meshgrid(g, g)
    keep = X**2 + Y**2 <= radius**2
    return np.column_stack([X[keep], Y[keep]]) + np.asarray(center)


def lattice_box(x0, x1, y0, y1, dx):
    xs = np.arange(x0 + dx / 2, x1, dx)
    ys = np.arange(y0 + dx / 2, y1, dx)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])


@dataclass
class SPHState:
    pos: np.ndarray
    vel: np.ndarray
    rho: np.ndarray
    e: np.ndarray
    mass: np.ndarray
    body: np.ndarray  # integer body id per particle
    pos0: np.ndarray
    h: float
    t: float = 0.0
    names: list[str] = field(default_factory=list)


class SPHSolver:
    def __init__(self, bodies: list[Body], dx: float, material: Material = ALUMINIUM,
                 h_factor: float = 1.3, alpha: float = 1.0, beta: float = 2.0, cfl: float = 0.3):
        pos = np.vstack([b.pos for b in bodies])
        vel = np.vstack([np.broadcast_to(b.vel, b.pos.shape) for b in bodies]).astype(float)
        body = np.concatenate([np.full(len(b.pos), i) for i, b in enumerate(bodies)])
        n = len(pos)
        self.m = material
        self.alpha, self.beta, self.cfl = alpha, beta, cfl
        self._cached = None
        self.state = SPHState(
            pos=pos.astype(float), vel=vel, rho=np.full(n, material.rho0), e=np.zeros(n),
            mass=np.full(n, material.rho0 * dx * dx), body=body, pos0=pos.copy(),
            h=h_factor * dx, names=[b.name for b in bodies],
        )

    def _rates(self, s: SPHState):
        h = s.h
        pairs = cKDTree(s.pos).query_pairs(2.0 * h, output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        dx = s.pos[i] - s.pos[j]
        r = np.linalg.norm(dx, axis=1)
        gw = kernel_grad(dx, r, h)
        dv = s.vel[i] - s.vel[j]
        p = mie_gruneisen(s.rho, s.e, self.m)
        c = sound_speed(s.rho, s.e, p, self.m)

        vr = np.einsum("ij,ij->i", dv, dx)
        mu = h * vr / (r * r + 0.01 * h * h)
        mu = np.where(vr < 0, mu, 0.0)
        cbar = 0.5 * (c[i] + c[j])
        rbar = 0.5 * (s.rho[i] + s.rho[j])
        pi_ij = (-self.alpha * cbar * mu + self.beta * mu * mu) / rbar

        term = p[i] / s.rho[i] ** 2 + p[j] / s.rho[j] ** 2 + pi_ij
        n = len(s.pos)
        mj, mi = s.mass[j], s.mass[i]
        dvgw = np.einsum("ij,ij->i", dv, gw)

        drho = np.bincount(i, mj * dvgw, n) + np.bincount(j, mi * dvgw, n)
        f = term[:, None] * gw
        acc = np.empty_like(s.pos)
        for k in range(2):
            acc[:, k] = -np.bincount(i, mj * f[:, k], n) + np.bincount(j, mi * f[:, k], n)
        de = 0.5 * (np.bincount(i, mj * term * dvgw, n) + np.bincount(j, mi * term * dvgw, n))
        vmax = np.max(c + np.linalg.norm(s.vel, axis=1))
        return drho, acc, de, vmax, p

    def step(self) -> float:
        s = self.state
        # Rates at the end of one step are the rates at the start of the next, so reuse them.
        drho, acc, de, vmax, _ = self._cached if self._cached else self._rates(s)
        dt = self.cfl * s.h / vmax
        # Kick-drift-kick leap-frog.
        s.vel += 0.5 * dt * acc
        s.rho += 0.5 * dt * drho
        s.e += 0.5 * dt * de
        s.pos += dt * s.vel
        self._cached = rates = self._rates(s)
        drho, acc, de, _, _ = rates
        s.vel += 0.5 * dt * acc
        s.rho += 0.5 * dt * drho
        s.e = np.maximum(s.e + 0.5 * dt * de, 0.0)
        s.rho = np.maximum(s.rho, 0.05 * self.m.rho0)
        s.t += dt
        return dt

    def run(self, t_end: float, snapshots: int = 0, callback=None):
        """Advance to ``t_end``; optionally record ``snapshots`` evenly spaced frames."""
        frames = []
        marks = list(np.linspace(0, t_end, snapshots)) if snapshots else []
        while self.state.t < t_end:
            while marks and self.state.t >= marks[0]:
                frames.append(self.snapshot())
                marks.pop(0)
            self.step()
            if callback:
                callback(self.state)
        frames.append(self.snapshot())
        return frames

    def snapshot(self) -> dict:
        s = self.state
        return {"t": s.t, "pos": s.pos.copy(), "vel": s.vel.copy(), "rho": s.rho.copy(),
                "p": mie_gruneisen(s.rho, s.e, self.m), "body": s.body.copy()}

    def momentum(self) -> np.ndarray:
        s = self.state
        return (s.mass[:, None] * s.vel).sum(axis=0)

    def total_energy(self) -> float:
        s = self.state
        return float((s.mass * (0.5 * (s.vel**2).sum(axis=1) + s.e)).sum())


@dataclass(frozen=True)
class WhippleImpact:
    """Planar Whipple-shield impact scenario (all lengths mm, speed km/s)."""

    projectile_diameter: float = 2.0
    speed: float = 7.0
    bumper_thickness: float = 0.5
    standoff: float = 15.0
    rear_wall_thickness: float = 1.5
    width: float = 30.0
    resolution: float = 0.1  # particle spacing

    def build(self) -> SPHSolver:
        dx = self.resolution
        rp = self.projectile_diameter / 2
        proj = lattice_disk((0.0, rp + dx), rp, dx)
        bumper = lattice_box(-self.width / 2, self.width / 2, -self.bumper_thickness, 0.0, dx)
        y_rear = -self.bumper_thickness - self.standoff
        rear = lattice_box(-self.width, self.width, y_rear - self.rear_wall_thickness, y_rear, dx)
        return SPHSolver([
            Body("projectile", proj, np.array([0.0, -self.speed])),
            Body("bumper", bumper, np.zeros(2)),
            Body("rear_wall", rear, np.zeros(2)),
        ], dx)

    def time_to_rear_wall(self) -> float:
        return (self.standoff + self.bumper_thickness + self.projectile_diameter) / self.speed


def damage_metrics(sc: WhippleImpact, snap: dict, pos0: np.ndarray) -> dict:
    """Bumper hole width, debris-cloud half-angle and rear-wall state from a snapshot."""
    dx = sc.resolution
    pos, body = snap["pos"], snap["body"]
    bm = body == 1
    # Hole: x-bins where no bumper particle remains inside the original slab (+-1 thickness).
    bx = pos0[bm, 0]
    in_slab = np.abs(pos[bm, 1] + sc.bumper_thickness / 2) < sc.bumper_thickness
    bins = np.arange(bx.min(), bx.max() + dx, dx)
    occupied = np.histogram(bx[in_slab], bins)[0] > 0
    centre = np.abs(0.5 * (bins[:-1] + bins[1:])) < sc.width / 2 - 2 * dx
    empty = (~occupied) & centre
    hole = float(empty.sum() * dx)

    # Cloud: projectile + bumper material well clear of the bumper (excludes spray along the plate).
    below = (body != 2) & (pos[:, 1] < -sc.bumper_thickness - 0.2 * sc.standoff)
    below &= pos[:, 1] > -sc.bumper_thickness - sc.standoff
    if below.sum() > 10:
        depth = -sc.bumper_thickness - pos[below, 1]
        ang = np.degrees(np.arctan2(np.abs(pos[below, 0]), np.maximum(depth, 1e-6)))
        half_angle = float(np.percentile(ang, 90))
    else:
        half_angle = 0.0

    rm = body == 2
    y_back = -sc.bumper_thickness - sc.standoff - sc.rear_wall_thickness
    deflection = float(np.max(pos0[rm, 1] - pos[rm, 1]))
    through = (body != 2) & (pos[:, 1] < y_back - dx)
    return {
        "time_us": snap["t"],
        "bumper_hole_mm": hole,
        "hole_to_projectile": hole / sc.projectile_diameter,
        "cloud_half_angle_deg": half_angle,
        "rear_wall_max_displacement_mm": deflection,
        "rear_wall_perforated": bool(through.sum() > 3 or deflection > sc.rear_wall_thickness),
        # High percentile, not max: single-particle spikes at contact interfaces are numerical.
        "p99_pressure_GPa": float(np.percentile(snap["p"], 99.5) * 1e3),
    }
