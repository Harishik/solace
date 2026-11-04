"""Ballistic-limit sweep with the SPH hydrocode, with and without material strength.

Whipple shield: 0.5 mm Al bumper, 15 mm standoff, 1.5 mm Al rear wall, normal impact at 7 km/s.
For each projectile diameter we run to twice the debris-cloud arrival time at the rear wall and
record whether it perforates (material through the wall, or deflection beyond its thickness),
then compare the threshold with the Christiansen ballistic limit equation (1.95 mm here).

The solver is 2-D planar: the "sphere" is an infinite cylinder (areal mass pi d/4 against 2d/3
for a sphere, ~1.18x), and, more importantly, its debris cloud spreads in two dimensions, so the
load reaching the rear wall falls off as ~1/r rather than ~1/r^2. Planar runs therefore fail at
smaller diameters than the (3-D, spherical) BLE; the sweep measures what strength changes.

With --axisymmetric the solver works in (r, z): the projectile is a real sphere and the cloud
spreads in three dimensions, which is the like-for-like comparison with the BLE.

Run from the repo root:  .venv/bin/python scripts/sph_ballistic_limit.py [--axisymmetric]
Writes reports/sph_ballistic_limit[_axisymmetric].json.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from solace.impact import WhippleImpact, WhippleShield, damage_metrics

DIAMETERS = (0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0)


def run(args):
    d, strength, axi, res, speed, extra, tag = args
    t = time.time()
    sc = WhippleImpact(projectile_diameter=d, resolution=res or (0.1 if axi else 0.15), width=24.0,
                       strength=strength, axisymmetric=axi, speed=speed, **extra)
    sol = sc.build()
    pos0 = sol.state.pos0.copy()
    snap = sol.run(2.0 * sc.time_to_rear_wall())[-1]
    m = damage_metrics(sc, snap, pos0)
    if tag:  # keep the final state so failure criteria can be re-evaluated without rerunning
        Path("reports/sph_snaps").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"reports/sph_snaps/{tag}_d{d}_v{speed}.npz", pos0=pos0,
                            **{k: v for k, v in snap.items() if isinstance(v, np.ndarray)})
    m.update(diameter_mm=d, strength=strength, resolution_mm=sc.resolution, speed_kms=speed,
             wall_s=time.time() - t, **extra)
    print(f"d={d} v={speed} dx={sc.resolution} strength={strength} {extra}: perforated={m['rear_wall_perforated']} "
          f"failed(BLE def)={m.get('rear_wall_failed_ble')} through={m['material_through_wall']} "
          f"spall={m.get('detached_spall_particles')} crack={m.get('through_crack')} "
          f"deflection={m['rear_wall_max_displacement_mm']:.2f} mm ({m['wall_s']:.0f}s)", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axisymmetric", action="store_true")
    ap.add_argument("--resolution", type=float, default=None, help="particle spacing, mm")
    ap.add_argument("--diameters", type=float, nargs="+", default=DIAMETERS)
    ap.add_argument("--speed", type=float, default=7.0, help="impact speed, km/s")
    ap.add_argument("--strength-only", action="store_true")
    ap.add_argument("--option", action="append", default=[], help="extra WhippleImpact field, key=value")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    extra = {}
    for kv in a.option:
        k, v = kv.split("=")
        if v.lower() in ("true", "false"):
            extra[k] = v.lower() == "true"
        else:
            try:
                extra[k] = float(v)
            except ValueError:
                extra[k] = v
    tag = ("_axisymmetric" if a.axisymmetric else "") + (f"_{a.tag}" if a.tag else "")
    out_path = Path(f"reports/sph_ballistic_limit{tag}.json")
    jobs = [(d, s, a.axisymmetric, a.resolution, a.speed, extra, a.tag)
            for s in ((True,) if a.strength_only else (True, False)) for d in a.diameters]
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(run, jobs))
    ble = float(WhippleShield(bumper_cm=0.05, rear_wall_cm=0.15, standoff_cm=1.5).critical_diameter(a.speed) * 10)
    out = {"ble_critical_diameter_mm": ble, "runs": rows}
    out.update(speed_kms=a.speed, resolution_mm=a.resolution, options=extra)
    if extra.get("fracture"):
        ok = [r["diameter_mm"] for r in rows if not r["rear_wall_failed_ble"]]
        bad = [r["diameter_mm"] for r in rows if r["rear_wall_failed_ble"]]
        out["threshold_ble_criterion_mm"] = [max([x for x in ok if x < min(bad, default=9)], default=None),
                                             min(bad, default=None)]
    for s in (True, False):
        failed = [r["diameter_mm"] for r in rows if r["strength"] == s and r["rear_wall_perforated"]]
        held = [r["diameter_mm"] for r in rows if r["strength"] == s and not r["rear_wall_perforated"]]
        out[f"threshold_{'strength' if s else 'hydro'}_mm"] = [max([x for x in held if x < min(failed, default=9)],
                                                                    default=None), min(failed, default=None)]
    out_path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
