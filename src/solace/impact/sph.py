"""2-D planar smoothed particle hydrodynamics (SPH) hydrocode for hypervelocity impact.

Physics:
* Mie-Grueneisen equation of state with a linear Us-up Hugoniot (aluminium defaults).
* Monaghan artificial viscosity to capture shocks.
* Cubic-spline kernel, KD-tree neighbour search, symplectic leap-frog time integration.
* Material strength (Libersky and Petschek 1991): deviatoric stress evolved with the Jaumann rate
  under plane strain, elastic-perfectly-plastic von Mises yield with radial return, and linear
  thermal softening to zero at melt. At 5-15 km/s impact pressures (50-300 GPa) dwarf the yield
  (~0.3 GPa), so strength hardly changes the bumper hole or the debris cloud; it does decide whether
  the rear wall bends or tears. A tensile cut-off stands in for spall. `strength=False` gives the
  purely hydrodynamic solver.

Units are mm, microseconds, grams. Velocities in mm/us equal km/s, and pressure comes out in
g/(mm us^2) = 1 TPa, so 1 GPa = 1e-3.

Being 2-D planar, a "sphere" is really an infinite cylinder. Use this for cloud shape, timing and
qualitative damage; use the BLE for quantitative perforation limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Material:
    rho0: float = 2.70e-3  # g/mm^3
    c0: float = 5.35  # mm/us, bulk sound speed
    s: float = 1.34  # Us = c0 + s up
    gamma0: float = 2.0  # Grueneisen parameter
    p_min: float = -1.0e-3  # tensile cut-off (~ -1 GPa spall strength)
    shear_modulus: float = 26.0e-3  # 26 GPa (Al 6061-T6)
    yield_stress: float = 0.29e-3  # 0.29 GPa
    e_melt: float = 0.97  # specific energy to melt from room temperature, (km/s)^2 = MJ/kg
    # Fracture (Benz and Asphaug 1995, Grady-Kipp): Weibull flaw distribution n(eps) = k eps^m.
    # m is not well known for metals; k is calibrated so that the weakest flaw in a reference
    # volume of (0.1 mm)^3 activates at the dynamic spall strength of 6061-T6 (~1 GPa).
    weibull_m: float = 9.0
    spall_stress: float = 1.0e-3  # 1 GPa
    weibull_v_ref: float = 1.0e-3  # mm^3

    @property
    def bulk_modulus(self) -> float:
        return self.rho0 * self.c0**2

    @property
    def youngs_modulus(self) -> float:
        k, g = self.bulk_modulus, self.shear_modulus
        return 9.0 * k * g / (3.0 * k + g)

    @property
    def weibull_k(self) -> float:
        eps = self.spall_stress / self.youngs_modulus
        return 1.0 / (self.weibull_v_ref * eps**self.weibull_m)


ALUMINIUM = Material()


@dataclass(frozen=True)
class TillotsonParams:
    """Tillotson (1962) equation of state; aluminium values from Melosh (1989), Table AII.3.
    Units as the solver: g/mm^3, TPa (1 GPa = 1e-3), MJ/kg = (mm/us)^2."""

    rho0: float = 2.70e-3
    A: float = 75.2e-3
    B: float = 65.0e-3
    E0: float = 5.0
    a: float = 0.5
    b: float = 1.63
    alpha: float = 5.0
    beta: float = 5.0
    E_iv: float = 3.0  # incipient vaporisation
    E_cv: float = 13.9  # complete vaporisation


TILLOTSON_AL = TillotsonParams()


def tillotson(rho, e, t: TillotsonParams = TILLOTSON_AL):
    """Tillotson pressure: compressed/cold branch, expanded-hot branch, linear blend in between."""
    eta = rho / t.rho0
    mu = eta - 1.0
    omega = e / (t.E0 * eta * eta) + 1.0
    pc = (t.a + t.b / omega) * rho * e + t.A * mu + t.B * mu * mu
    x = t.rho0 / rho - 1.0
    pe = t.a * rho * e + (t.b * rho * e / omega + t.A * mu * np.exp(-t.beta * x)) * np.exp(-t.alpha * x * x)
    w = np.clip((e - t.E_iv) / (t.E_cv - t.E_iv), 0.0, 1.0)
    return np.where(mu >= 0, pc, w * pe + (1.0 - w) * pc)


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


def kernel(r, h):
    """2-D cubic spline kernel W(r, h)."""
    sigma = 10.0 / (7.0 * np.pi * h * h)
    q = r / h
    return sigma * np.where(q < 1.0, 1.0 - 1.5 * q * q + 0.75 * q**3,
                            np.where(q < 2.0, 0.25 * (2.0 - q) ** 3, 0.0))


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
    sdev: np.ndarray | None = None  # deviatoric stress (n, 4): xx, yy, zz, xy (plane strain)
    t: float = 0.0
    names: list[str] = field(default_factory=list)


class SPHSolver:
    def __init__(self, bodies: list[Body], dx: float, material: Material = ALUMINIUM,
                 h_factor: float = 1.3, alpha: float = 1.0, beta: float = 2.0, cfl: float = 0.3,
                 strength: bool = True, eos: str = "mie", artificial_stress: float = 0.0,
                 fracture: bool = False, seed: int = 0):
        if eos not in ("mie", "tillotson"):
            raise ValueError(f"unknown equation of state {eos!r}")
        self.eos, self.eps_as, self.fracture, self.seed = eos, artificial_stress, fracture, seed
        if (artificial_stress or fracture) and type(self) is SPHSolver:
            raise NotImplementedError("artificial stress and fracture are implemented for the axisymmetric solver")
        pos = np.vstack([b.pos for b in bodies])
        vel = np.vstack([np.broadcast_to(b.vel, b.pos.shape) for b in bodies]).astype(float)
        body = np.concatenate([np.full(len(b.pos), i) for i, b in enumerate(bodies)])
        n = len(pos)
        self.m = material
        self.alpha, self.beta, self.cfl, self.h_factor = alpha, beta, cfl, h_factor
        self.strength = strength
        self._cached = None
        self.state = SPHState(
            pos=pos.astype(float), vel=vel, rho=np.full(n, material.rho0), e=np.zeros(n),
            mass=np.full(n, material.rho0 * dx * dx), body=body, pos0=pos.copy(),
            h=h_factor * dx, names=[b.name for b in bodies], sdev=np.zeros((n, 4)),
        )

    def _rates(self, s: SPHState):
        h = s.h
        pairs = cKDTree(s.pos).query_pairs(2.0 * h, output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        dx = s.pos[i] - s.pos[j]
        r = np.linalg.norm(dx, axis=1)
        gw = kernel_grad(dx, r, h)
        dv = s.vel[i] - s.vel[j]
        p, c = self._eos(s.rho, s.e)

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
        ds = np.zeros((n, 4))
        if self.strength:
            # Deviatoric stress adds (S_i/rho_i^2 + S_j/rho_j^2) . grad W to the acceleration.
            S, r2 = s.sdev, s.rho**2
            sx = S[i, 0] / r2[i] + S[j, 0] / r2[j]
            sy = S[i, 1] / r2[i] + S[j, 1] / r2[j]
            sxy = S[i, 3] / r2[i] + S[j, 3] / r2[j]
            fx, fy = sx * gw[:, 0] + sxy * gw[:, 1], sxy * gw[:, 0] + sy * gw[:, 1]
            acc[:, 0] += np.bincount(i, mj * fx, n) - np.bincount(j, mi * fx, n)
            acc[:, 1] += np.bincount(i, mj * fy, n) - np.bincount(j, mi * fy, n)
            # Velocity gradient L_ab = dv_a/dx_b = sum_j (m_j/rho_j)(v_j - v_i)_a (grad_i W)_b.
            L = np.zeros((n, 2, 2))
            wi, wj = mj / s.rho[j], mi / s.rho[i]
            for a in range(2):
                for b in range(2):
                    g = dv[:, a] * gw[:, b]
                    L[:, a, b] = -np.bincount(i, wi * g, n) - np.bincount(j, wj * g, n)
            exx, eyy, exy = L[:, 0, 0], L[:, 1, 1], 0.5 * (L[:, 0, 1] + L[:, 1, 0])
            w = 0.5 * (L[:, 0, 1] - L[:, 1, 0])
            tr3 = (exx + eyy) / 3.0
            g2 = 2.0 * self.m.shear_modulus * self._soft(s.e)
            # Jaumann rate: S' = 2G dev(D) + W S - S W (plane strain, so D_zz = 0).
            ds[:, 0] = g2 * (exx - tr3) + 2.0 * w * S[:, 3]
            ds[:, 1] = g2 * (eyy - tr3) - 2.0 * w * S[:, 3]
            ds[:, 2] = g2 * (-tr3)
            ds[:, 3] = g2 * exy + w * (S[:, 1] - S[:, 0])
            de += (S[:, 0] * exx + S[:, 1] * eyy + 2.0 * S[:, 3] * exy) / s.rho
        vmax = np.max(c + np.linalg.norm(s.vel, axis=1))
        return drho, acc, de, vmax, p, ds

    def _pressure(self, rho, e):
        if self.eos == "tillotson":
            p = tillotson(rho, e)
        else:
            p = mie_gruneisen(rho, e, replace(self.m, p_min=-np.inf))
        # With fracture, damage limits tension; otherwise a fixed spall cut-off does.
        return p if self.fracture else np.maximum(p, self.m.p_min)

    def _eos(self, rho, e):
        """Pressure and sound speed, c^2 = dp/drho|_e + p/rho^2 dp/de|_rho (finite differences)."""
        p = self._pressure(rho, e)
        d, de = 1e-6 * self.m.rho0, 1e-6
        dpdrho = (self._pressure(rho + d, e) - self._pressure(rho - d, e)) / (2 * d)
        dpde = (self._pressure(rho, e + de) - self._pressure(rho, np.maximum(e - de, 0.0))) / (
            e + de - np.maximum(e - de, 0.0))
        c2 = dpdrho + p / rho**2 * dpde
        floor = 0.5 if self.eos == "mie" else 0.25  # Mie value as validated against the Hugoniot
        return p, np.sqrt(np.clip(c2, (floor * self.m.c0) ** 2, None))

    def _soft(self, e):
        """Linear thermal softening: full strength cold, none at melt."""
        return np.clip(1.0 - e / self.m.e_melt, 0.0, 1.0)

    def _yield(self, s: SPHState):
        """Radial return onto the von Mises surface."""
        S = s.sdev
        vm = np.sqrt(1.5 * (S[:, 0] ** 2 + S[:, 1] ** 2 + S[:, 2] ** 2 + 2.0 * S[:, 3] ** 2))
        y = self.m.yield_stress * self._soft(s.e)
        f = np.where(vm > y, y / np.maximum(vm, 1e-30), 1.0)
        S *= f[:, None]

    def step(self) -> float:
        s = self.state
        # Rates at the end of one step are the rates at the start of the next, so reuse them.
        drho, acc, de, vmax, _, ds = self._cached if self._cached else self._rates(s)
        dt = self.cfl * s.h / vmax
        # Kick-drift-kick leap-frog.
        s.vel += 0.5 * dt * acc
        s.rho += 0.5 * dt * drho
        s.e += 0.5 * dt * de
        s.sdev += 0.5 * dt * ds
        s.pos += dt * s.vel
        self._cached = rates = self._rates(s)
        drho, acc, de, _, _, ds = rates
        s.vel += 0.5 * dt * acc
        s.rho += 0.5 * dt * drho
        s.e = np.maximum(s.e + 0.5 * dt * de, 0.0)
        s.sdev += 0.5 * dt * ds
        if self.strength:
            self._yield(s)
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
                "p": self._eos(s.rho, s.e)[0], "body": s.body.copy()}

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
    strength: bool = True
    axisymmetric: bool = False  # True: a real sphere on round plates (r, z); False: planar cylinder
    eos: str = "mie"  # "mie" (Mie-Grueneisen) or "tillotson"
    artificial_stress: float = 0.0  # Monaghan epsilon (0.3 typical); axisymmetric only
    fracture: bool = False  # Benz-Asphaug / Grady-Kipp tensile damage; axisymmetric only
    seed: int = 0  # flaw distribution
    weibull_m: float = ALUMINIUM.weibull_m
    spall_stress_gpa: float = ALUMINIUM.spall_stress * 1e3

    def build(self) -> SPHSolver:
        dx = self.resolution
        rp = self.projectile_diameter / 2
        y_rear = -self.bumper_thickness - self.standoff
        if self.axisymmetric:
            g = np.arange(dx / 2, rp, dx)
            R, Z = np.meshgrid(g, np.arange(-rp, rp + dx / 2, dx))
            keep = R**2 + Z**2 <= rp**2
            proj = np.column_stack([R[keep], Z[keep] + rp + dx])
            bumper = lattice_box(0.0, self.width / 2, -self.bumper_thickness, 0.0, dx)
            rear = lattice_box(0.0, self.width, y_rear - self.rear_wall_thickness, y_rear, dx)
            return AxisymmetricSPHSolver([
                Body("projectile", proj, np.array([0.0, -self.speed])),
                Body("bumper", bumper, np.zeros(2)),
                Body("rear_wall", rear, np.zeros(2)),
            ], dx, material=replace(ALUMINIUM, weibull_m=self.weibull_m, spall_stress=self.spall_stress_gpa * 1e-3),
                strength=self.strength, eos=self.eos, artificial_stress=self.artificial_stress,
                fracture=self.fracture, seed=int(self.seed))
        proj = lattice_disk((0.0, rp + dx), rp, dx)
        bumper = lattice_box(-self.width / 2, self.width / 2, -self.bumper_thickness, 0.0, dx)
        rear = lattice_box(-self.width, self.width, y_rear - self.rear_wall_thickness, y_rear, dx)
        return SPHSolver([
            Body("projectile", proj, np.array([0.0, -self.speed])),
            Body("bumper", bumper, np.zeros(2)),
            Body("rear_wall", rear, np.zeros(2)),
        ], dx, strength=self.strength, eos=self.eos)

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
    hole = float(empty.sum() * dx) * (2.0 if sc.axisymmetric else 1.0)  # axisymmetric bins span the radius

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
    ok = ~snap.get("eroded", np.zeros(len(pos), bool))
    disp = pos0[rm, 1] - pos[rm, 1]
    deflection = float(np.max(disp[ok[rm]]))
    # Like material through the wall, a deflection needs more than 3 particles to count: a single
    # spalled particle is not a failed wall.
    bulged = int(np.sum(disp[ok[rm]] > sc.rear_wall_thickness))
    through = (body != 2) & (pos[:, 1] < y_back - dx) & ok
    # NASA ballistic-limit failure (Christiansen): perforation, or detached spall from the back of
    # the wall. Spall needs the fracture model: wall rings behind the back face, fully damaged.
    dmg = snap.get("damage")
    behind = (rm & ok & (pos[:, 1] < y_back - 2 * dx) & (dmg > 0.99)) if dmg is not None else np.zeros(len(pos), bool)
    # Detached spall: wall material behind the back face that is no longer connected to the wall,
    # i.e. outside the largest cluster of wall rings linked within 1.5 particle spacings. A damaged
    # but attached bulge is not spall.
    spall = np.zeros(len(pos), bool)
    if behind.any():
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        wall = np.nonzero(rm & ok)[0]
        pr = cKDTree(pos[wall]).query_pairs(1.5 * dx, output_type="ndarray")
        g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(wall), len(wall)))
        _, lab = connected_components(g, directed=False)
        main = np.bincount(lab).argmax()
        spall[wall[lab != main]] = True
        spall &= behind
    # Through-thickness crack: some radial bin of the wall where every ring is fully damaged.
    crack = False
    if dmg is not None:
        r0 = pos0[rm, 0]
        bins = np.floor(r0 / dx).astype(int)
        intact = np.bincount(bins, (dmg[rm] < 0.99) & ok[rm], bins.max() + 1)
        total = np.bincount(bins, minlength=bins.max() + 1)
        crack = bool(np.any((total > 0) & (intact == 0)))
    return {
        "time_us": snap["t"],
        "bumper_hole_mm": hole,
        "hole_to_projectile": hole / sc.projectile_diameter,
        "cloud_half_angle_deg": half_angle,
        "rear_wall_max_displacement_mm": deflection,
        "rear_wall_particles_beyond_thickness": bulged,
        "material_through_wall": int(through.sum()),
        "eroded_particles": int((~ok).sum()),
        "rear_wall_perforated_single_particle": bool(through.sum() > 3 or deflection > sc.rear_wall_thickness),
        "rear_wall_perforated": bool(through.sum() > 3 or bulged > 3),
        "detached_spall_particles": int(spall.sum()),
        "damaged_behind_back_face": int(behind.sum()),
        "through_crack": crack,
        # Christiansen failure: material through the wall, or detached spall. A fully damaged column
        # ("through_crack") is reported but is not failure: damaged material still carries compression.
        "rear_wall_failed_ble": bool(through.sum() > 3 or spall.sum() > 3) if dmg is not None else None,
        # High percentile, not max: single-particle spikes at contact interfaces are numerical.
        "p99_pressure_GPa": float(np.percentile(snap["p"], 99.5) * 1e3),
    }


class AxisymmetricSPHSolver(SPHSolver):
    """Axisymmetric (r, z) SPH: x is the radius r >= 0, y is the axis z. Each particle is a ring.

    With ring masses m and the 2-D kernel, the summed quantity is eta = 2 pi r rho. Varying the
    Lagrangian sum m (v^2/2 - u(rho)) with rho = eta / (2 pi r) gives the pair force
    sum_j m_j (sigma_i / (rho_i eta_i) + sigma_j / (rho_j eta_j)) . grad_i W_ij plus a hoop term
    -sigma_thetatheta / (rho r) along r; energy and the deviatoric stress (rr, zz, thetatheta, rz)
    pick up the hoop strain rate v_r / r. Particles within 2h of the axis are mirrored across it.
    A projectile disk in (r, z) is a sphere; a slab is a plate.
    """

    FLOOR = 0.05  # density floor, fraction of solid density

    def snapshot(self) -> dict:
        snap = super().snapshot()
        snap["eroded"] = self.eroded.copy()
        if self.fracture:
            snap["damage"] = self.damage.copy()
        return snap

    def __init__(self, bodies: list[Body], dx: float, material: Material = ALUMINIUM, **kw):
        bodies = [Body(b.name, b.pos[b.pos[:, 0] > 0], b.vel) for b in bodies]  # keep r > 0 only
        super().__init__(bodies, dx, material, **kw)
        s = self.state
        s.mass = 2 * np.pi * s.pos[:, 0] * material.rho0 * dx * dx  # ring masses
        self.eta = 2 * np.pi * s.pos[:, 0] * s.rho
        # Erosion limits: nothing can move at twice the impact speed or hold twice the impact's
        # specific kinetic energy (units mm/us, so v^2 is MJ/kg); such rings are numerical debris.
        v0 = float(np.max(np.linalg.norm(s.vel, axis=1)))
        self.v_erode, self.e_erode = 2.0 * v0, v0 * v0
        self.eroded = np.zeros(len(s.pos), bool)
        self.damage = np.zeros(len(s.pos))
        self._eps_local = np.zeros(len(s.pos))
        if self.fracture:
            self._seed_flaws(np.random.default_rng(self.seed))

    def _seed_flaws(self, rng):
        """Benz and Asphaug (1995): per body, flaws with activation strains eps_j = (j / (k V))^(1/m),
        j = 1..N (N ~ n ln n), each placed in a particle drawn with probability proportional to its
        volume; particles left without a flaw get the next ones, so every particle has at least one."""
        s, m = self.state, self.m
        vol = s.mass / m.rho0
        owners, eps = [], []
        for b in np.unique(s.body):
            idx = np.nonzero(s.body == b)[0]
            n = len(idx)
            nf = int(np.ceil(n * max(np.log(n), 1.0)))
            pick = rng.choice(idx, size=nf, p=vol[idx] / vol[idx].sum())
            missing = np.setdiff1d(idx, pick)
            pick = np.concatenate([pick, missing])
            j = np.arange(1, len(pick) + 1)
            owners.append(pick)
            eps.append((j / (m.weibull_k * vol[idx].sum())) ** (1.0 / m.weibull_m))
        self.flaw_owner = np.concatenate(owners)
        self.flaw_eps = np.concatenate(eps)
        self.n_flaws = np.bincount(self.flaw_owner, minlength=len(s.pos))
        k, g = m.bulk_modulus, m.shear_modulus
        self.crack_speed = 0.4 * np.sqrt((k + 4.0 * g / 3.0) / m.rho0)  # 0.4 x longitudinal wave speed

    def _grow_damage(self, dt):
        """Grady-Kipp crack growth: d(D^1/3)/dt = n_act c_g / R_s, capped at (n_act / n_flaws)^1/3."""
        act = self.flaw_eps <= self._eps_local[self.flaw_owner]
        n_act = np.bincount(self.flaw_owner[act], minlength=len(self.damage))
        d3 = np.cbrt(self.damage) + dt * n_act * self.crack_speed / self.state.h
        cap = np.cbrt(n_act / np.maximum(self.n_flaws, 1))
        self.damage = np.maximum(self.damage, np.minimum(d3, cap) ** 3)

    def _rates(self, s: SPHState):
        h, n = s.h, len(s.pos)
        r_axis = np.maximum(s.pos[:, 0], 0.25 * h)
        rho = self.eta / (2 * np.pi * r_axis)
        s.rho = rho
        ghost = np.nonzero(s.pos[:, 0] < 2 * h)[0]
        pos = np.vstack([s.pos, s.pos[ghost] * [-1, 1]])
        vel = np.vstack([s.vel, s.vel[ghost] * [-1, 1]])
        src = np.concatenate([np.arange(n), ghost])  # real particle each entry stands for
        S = np.vstack([s.sdev, s.sdev[ghost] * [1, 1, 1, -1]])
        pairs = cKDTree(pos).query_pairs(2.0 * h, output_type="ndarray")
        pairs = pairs[(pairs[:, 0] < n) | (pairs[:, 1] < n)]
        i, j = pairs[:, 0], pairs[:, 1]
        si, sj = src[i], src[j]
        dx = pos[i] - pos[j]
        r = np.linalg.norm(dx, axis=1)
        gw = kernel_grad(dx, r, h)
        dv = vel[i] - vel[j]
        p, c = self._eos(rho, s.e)
        # Erosion: a ring with no neighbour (other than its own mirror image), or expanded to the
        # density floor (vapour at <= 5 % solid density), is a fragment in free flight and carries no
        # stress. Without this, terms divided by its tiny density throw it to unphysical speeds and the
        # CFL time step collapses.
        other = si != sj
        free = (np.bincount(si[other], minlength=n) + np.bincount(sj[other], minlength=n)) == 0
        free |= rho <= 1.01 * self.FLOOR * self.m.rho0
        free |= self.eroded
        p = np.where(free, 0.0, p)
        c = np.where(free, 0.0, c)
        keep = 1.0 - self.damage  # damaged material carries no tension and no shear
        if self.fracture:
            p = np.where(p < 0, keep * p, p)
        if self.strength:
            s.sdev[free] = 0.0
        Seff = s.sdev * keep[:, None] if self.strength else np.zeros((n, 4))
        S = np.vstack([Seff, Seff[ghost] * [1, 1, 1, -1]])
        eta, m = self.eta, s.mass
        vr = np.einsum("ij,ij->i", dv, dx)
        mu = np.where(vr < 0, h * vr / (r * r + 0.01 * h * h), 0.0)
        ebar = 0.5 * (eta[si] + eta[sj])
        pi_ij = (-self.alpha * 0.5 * (c[si] + c[sj]) * mu + self.beta * mu * mu) / ebar  # = Pi_planar * rho/eta
        q = p / (rho * eta)
        live = ~(free[si] | free[sj])  # eroded rings no longer interact
        term = (q[si] + q[sj] + pi_ij) * live
        nt = len(pos)
        mi, mj = m[si], m[sj]
        dvgw = np.einsum("ij,ij->i", dv, gw)
        real_i, real_j = i < n, j < n
        deta = (np.bincount(i, mj * dvgw * real_i, nt) + np.bincount(j, mi * dvgw * real_j, nt))[:n]
        acc = np.zeros((nt, 2))
        for k in range(2):
            acc[:, k] = -np.bincount(i, mj * term * gw[:, k], nt) + np.bincount(j, mi * term * gw[:, k], nt)
        de = 0.5 * (np.bincount(i, mj * term * dvgw, nt) + np.bincount(j, mi * term * dvgw, nt))
        acc, de = acc[:n], de[:n]
        ur = s.vel[:, 0] / r_axis  # hoop strain rate
        acc[:, 0] += p / (rho * r_axis)  # hoop term, pressure part: -sigma_tt/(rho r) with sigma_tt = -p
        de -= p * ur / rho
        ds = np.zeros((n, 4))
        if self.strength:
            sw = S / (np.concatenate([rho, rho[ghost]]) * np.concatenate([eta, eta[ghost]]))[:, None]
            fx = (sw[i, 0] + sw[j, 0]) * gw[:, 0] + (sw[i, 3] + sw[j, 3]) * gw[:, 1]
            fy = (sw[i, 3] + sw[j, 3]) * gw[:, 0] + (sw[i, 1] + sw[j, 1]) * gw[:, 1]
            fx, fy = fx * live, fy * live
            acc[:, 0] += (np.bincount(i, mj * fx, nt) - np.bincount(j, mi * fx, nt))[:n]
            acc[:, 1] += (np.bincount(i, mj * fy, nt) - np.bincount(j, mi * fy, nt))[:n]
            acc[:, 0] -= Seff[:, 2] / (rho * r_axis)  # hoop term, deviatoric part
            L = np.zeros((n, 2, 2))
            wi, wj = mj / eta[sj], mi / eta[si]
            for a in range(2):
                for b in range(2):
                    g = dv[:, a] * gw[:, b]
                    L[:, a, b] = (-np.bincount(i, wi * g * real_i, nt) - np.bincount(j, wj * g * real_j, nt))[:n]
            drr, dzz, drz = L[:, 0, 0], L[:, 1, 1], 0.5 * (L[:, 0, 1] + L[:, 1, 0])
            w = 0.5 * (L[:, 0, 1] - L[:, 1, 0])
            tr3 = (drr + dzz + ur) / 3.0
            g2 = 2.0 * self.m.shear_modulus * self._soft(s.e)
            Sd = s.sdev
            ds[:, 0] = g2 * (drr - tr3) + 2.0 * w * Sd[:, 3]
            ds[:, 1] = g2 * (dzz - tr3) - 2.0 * w * Sd[:, 3]
            ds[:, 2] = g2 * (ur - tr3)
            ds[:, 3] = g2 * drz + w * (Sd[:, 1] - Sd[:, 0])
            de += (Seff[:, 0] * drr + Seff[:, 1] * dzz + Seff[:, 2] * ur + 2.0 * Seff[:, 3] * drz) / rho
        if self.eps_as or self.fracture:
            # In-plane principal stresses of sigma = -p I + S (damaged), and the hoop stress.
            srr, szz, srz = -p + Seff[:, 0], -p + Seff[:, 1], Seff[:, 3]
            mean, half = 0.5 * (srr + szz), 0.5 * (srr - szz)
            rad = np.sqrt(half * half + srz * srz)
            s1, s2 = mean + rad, mean - rad
            if self.fracture:
                stt = -p + Seff[:, 2]
                tension = np.maximum(np.maximum(s1, stt), 0.0)
                self._eps_local = np.where(free, 0.0, tension / (np.maximum(keep, 1e-3) * self.m.youngs_modulus))
            if self.eps_as:
                # Monaghan (2000) / Gray, Monaghan and Swift (2001): a repulsive stress
                # R' = -eps sigma' / (rho eta) along each tensile principal direction, applied with the
                # weight (W(r)/W(dx))^4, suppresses the tensile instability (artificial clumping).
                th = 0.5 * np.arctan2(srz, half)
                cs, sn = np.cos(th), np.sin(th)
                w_ = -self.eps_as / (rho * eta)
                r1, r2 = w_ * np.maximum(s1, 0.0), w_ * np.maximum(s2, 0.0)
                R = np.column_stack([r1 * cs * cs + r2 * sn * sn, r1 * sn * sn + r2 * cs * cs,
                                     (r1 - r2) * sn * cs])
                R[free] = 0.0
                Rg = np.vstack([R, R[ghost] * [1, 1, -1]])
                fn = (kernel(r, h) / kernel(np.full(1, h / self.h_factor), h)) ** 4 * live
                ax = ((Rg[i, 0] + Rg[j, 0]) * gw[:, 0] + (Rg[i, 2] + Rg[j, 2]) * gw[:, 1]) * fn
                az = ((Rg[i, 2] + Rg[j, 2]) * gw[:, 0] + (Rg[i, 1] + Rg[j, 1]) * gw[:, 1]) * fn
                acc[:, 0] += (np.bincount(i, mj * ax, nt) - np.bincount(j, mi * ax, nt))[:n]
                acc[:, 1] += (np.bincount(i, mj * az, nt) - np.bincount(j, mi * az, nt))[:n]
        speed = c + np.linalg.norm(s.vel, axis=1)
        vmax = np.max(speed[~self.eroded]) if (~self.eroded).any() else np.max(speed)
        return deta, acc, de, vmax, p, ds

    def step(self) -> float:
        s = self.state
        deta, acc, de, vmax, _, ds = self._cached if self._cached else self._rates(s)
        dt = self.cfl * s.h / vmax
        s.vel += 0.5 * dt * acc
        self.eta += 0.5 * dt * deta
        s.e += 0.5 * dt * de
        s.sdev += 0.5 * dt * ds
        s.pos += dt * s.vel
        crossed = s.pos[:, 0] < 0  # a ring cannot cross the axis: reflect position and radial velocity
        s.pos[crossed, 0] *= -1
        s.vel[crossed, 0] *= -1
        self._cached = rates = self._rates(s)
        deta, acc, de, _, _, ds = rates
        s.vel += 0.5 * dt * acc
        self.eta += 0.5 * dt * deta
        s.e = np.maximum(s.e + 0.5 * dt * de, 0.0)
        s.sdev += 0.5 * dt * ds
        if self.strength:
            self._yield(s)
        if self.fracture:
            self._grow_damage(dt)
        self.eroded |= (np.linalg.norm(s.vel, axis=1) > self.v_erode) | (s.e > self.e_erode)
        floor = self.FLOOR * self.m.rho0 * 2 * np.pi * np.maximum(s.pos[:, 0], 0.25 * s.h)
        self.eta = np.maximum(self.eta, floor)
        s.t += dt
        return dt
