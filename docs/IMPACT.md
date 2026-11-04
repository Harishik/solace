# Hypervelocity impact: the SPH model and how well it does

`solace.impact.sph` is a small smoothed-particle hydrodynamics (SPH) code for a sphere hitting a
two-wall aluminium Whipple shield. It exists to show *how* a shield fails (cloud shape, timing, which
wall gives way) and to check the empirical ballistic limit equations (BLE) the risk index relies on. For
pass/fail numbers use the BLE (`solace.impact.ble`).

## Physics

| Piece | Model | Why |
|---|---|---|
| Geometry | Planar (2-D slab) or axisymmetric rings (mass 2πrρ per particle, hoop-stress terms, mirror ghosts at the axis) | A normal impact is axisymmetric; the planar version over-predicts damage because a 2-D "sphere" is a cylinder |
| Equation of state | Mie–Grüneisen (shock Hugoniot, Γ = 2) or Tillotson (Melosh 1989 Al: A 75.2 GPa, B 65 GPa, a 0.5, b 1.63, α = β = 5, E_iv 3, E_cv 13.9 MJ/kg) | Tillotson stays valid in expanded, partly vaporised states behind the bumper |
| Strength | Elastic–perfectly-plastic, von Mises yield with thermal softening to the melt energy, Jaumann stress rate | Without strength the rear wall fails far too easily |
| Tensile instability | Gray–Monaghan–Swift artificial stress (ε = 0.3, n = 4) | SPH particles clump in tension; this repulsion keeps spall physical |
| Fracture | Benz–Asphaug / Grady–Kipp: Weibull flaws (m = 9, k set so the weakest flaw in 0.1 mm³ activates at 1 GPa), cracks grow at 0.4 c_l once local strain exceeds a flaw's threshold, damage D scales tensile pressure and the deviatoric stress by (1 − D) | Spall and fragmentation need a damage law; the Weibull flaws give the right strain-rate dependence |
| Erosion | A particle that is isolated, at the density floor (0.05 ρ₀), faster than 2× the impact speed or above v₀² specific energy is frozen out of the interactions | Without it a single runaway particle collapsed the time step and stalled a run for 5.5 h |

## Failure criterion

Christiansen's: the rear wall fails if material passes through it or a detached spall leaves its back
face. Through-wall material is more than 3 non-eroded wall or projectile particles behind the back face.
Detached spall is more than 3 rear-wall particles behind the back face that are not connected (within
1.5 particle spacings) to the main wall. An earlier criterion counted any damaged bulge as spall and
called the wall failed at 1.2 mm at every speed; the connectivity test fixes that.

## Validation

- **Shock Hugoniot** (Al on Al, 1–4 km/s): pressure within 1.3 %, particle velocity exact.
- **Debris cloud half-angle:** 38–48°, around the 30–45° seen experimentally.
- **Ballistic limit** (`scripts/sph_ballistic_limit.py --axisymmetric`, 0.5/1.5 mm shield, 15 mm
  standoff, 0.05 mm particles, all physics on):

| Speed | Holds at | Fails at | BLE critical diameter |
|---|---|---|---|
| 3 km/s | 1.2 mm | 1.5 mm (spall); 1.8 mm material through | 1.17 mm |
| 5 km/s | 1.2 mm | 1.5 mm (spall); 1.8 mm material through | 1.56 mm |
| 7 km/s | 1.2 mm | 1.5 mm (material through and spall) | 1.95 mm |

At 3 and 5 km/s the simulation agrees with the BLE to within one diameter step (0.3 mm). At 7 km/s it
is ~1.4× conservative. The likely cause is that the projectile melts and partly vaporises at 7 km/s,
spreading its momentum over a wider cloud, and the code has no latent heat or phase change.

**Convergence.** Without fracture the threshold is the same at 0.1, 0.07 and 0.05 mm particle spacing
(0.9–1.2 mm at 7 km/s). With fracture it rises one step from 0.1 to 0.05 mm (fails at 1.2 → 1.5 mm), so
the full-physics answer is not fully converged; finer runs need a GPU or a tree code. Doubling the
spall strength to 2 GPa changes nothing.

## Limits

Normal impacts only (axisymmetric); one material (aluminium) for projectile and walls; no phase change;
no multi-layer insulation or stuffed shields. Final particle states for every run are saved in
`reports/sph_snaps/` so the failure criterion can be changed without re-running.
