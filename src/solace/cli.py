"""Command line: ``solace screen | impact | risk | validate | serve | history | train``."""

from __future__ import annotations

import argparse
import json
import sys


def _find(catalog, key: str):
    if key.isdigit():
        hit = [t for t in catalog if t.norad_id == int(key)]
    else:
        hit = [t for t in catalog if t.name.strip().upper() == key.upper()]
    if not hit:
        sys.exit(f"object {key!r} not in catalog")
    return hit[0]


def cmd_screen(a):
    from .ingest.celestrak import fetch_catalog
    from .orbit import screen

    cat = fetch_catalog()
    events = screen(_find(cat, a.asset), cat, days=a.days, threshold_km=a.threshold)
    for c in events[: a.top]:
        print(json.dumps(c.as_dict()))


def cmd_impact(a):
    from .impact import WhippleImpact, WhippleShield, damage_metrics

    s = WhippleShield(bumper_cm=a.bumper / 10, rear_wall_cm=a.rear / 10, standoff_cm=a.standoff / 10)
    d_c = float(s.critical_diameter(a.speed, a.angle))
    print(f"BLE critical diameter at {a.speed} km/s, {a.angle} deg: {d_c:.3f} cm")
    if a.sph:
        sc = WhippleImpact(projectile_diameter=a.diameter, speed=a.speed, bumper_thickness=a.bumper,
                           standoff=a.standoff, rear_wall_thickness=a.rear, resolution=a.resolution)
        sol = sc.build()
        pos0 = sol.state.pos0.copy()
        frames = sol.run(sc.time_to_rear_wall() * 1.3, snapshots=a.frames)
        print(json.dumps(damage_metrics(sc, frames[-1], pos0), default=float))
        if a.html:
            from .viz import impact_animation

            impact_animation(frames, sc, a.html)
            print(f"wrote {a.html}")


def cmd_risk(a):
    from .ingest.celestrak import fetch_catalog
    from .risk import asset_risk, build_grid

    cat = fetch_catalog()
    g = build_grid(cat)
    print(json.dumps(asset_risk(_find(cat, a.asset), g, days=a.days, area_m2=a.area).as_dict()))


def cmd_validate(_):
    from .validate import compare, summarize

    print(summarize(compare()))


def cmd_serve(a):
    import uvicorn

    uvicorn.run("solace.live.server:app", host=a.host, port=a.port)


def cmd_history(a):
    from datetime import date

    from .risk.history import SpacebookHistory, SpaceTrack, snapshot_days

    st = SpacebookHistory() if a.source == "spacebook" else SpaceTrack()
    days = snapshot_days(date.fromisoformat(a.start), date.fromisoformat(a.end), a.step)
    for i, d in enumerate(days, 1):
        n = len(st.snapshot(d))
        print(f"[{i}/{len(days)}] {d}: {n} objects")


def cmd_train(a):
    from .risk import adjacency
    from .risk.dataset import build_series
    from .risk.history import SpacebookHistory, SpaceTrack, cached_snapshots, cached_spacebook_snapshots
    from .risk.model import train

    if a.source == "spacebook":
        days, src = cached_spacebook_snapshots(), SpacebookHistory()
    else:
        days, src = cached_snapshots(), SpaceTrack.__new__(SpaceTrack)  # on-disk cache only, no login
    if len(days) < a.lookback + a.horizon + 8:
        sys.exit(f"only {len(days)} cached snapshots; run `solace history` first")
    snaps = {d: type(src).snapshot(src, d) for d in days}
    gs = build_series(snaps, horizon_steps=a.horizon)
    res = train(gs, adjacency(), lookback=a.lookback, epochs=a.epochs)
    print(json.dumps(res.metrics, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="solace")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("screen", help="conjunctions for one asset")
    s.add_argument("asset", help="NORAD id or exact name, e.g. 25544 or NISAR")
    s.add_argument("--days", type=float, default=3)
    s.add_argument("--threshold", type=float, default=10, help="km")
    s.add_argument("--top", type=int, default=20)
    s.set_defaults(fn=cmd_screen)

    s = sub.add_parser("impact", help="Whipple shield ballistic limit (+ optional SPH run); lengths in mm")
    s.add_argument("--speed", type=float, default=7.0)
    s.add_argument("--angle", type=float, default=0.0)
    s.add_argument("--diameter", type=float, default=2.0)
    s.add_argument("--bumper", type=float, default=0.5)
    s.add_argument("--standoff", type=float, default=15.0)
    s.add_argument("--rear", type=float, default=1.5)
    s.add_argument("--resolution", type=float, default=0.1)
    s.add_argument("--sph", action="store_true")
    s.add_argument("--frames", type=int, default=30)
    s.add_argument("--html", help="write an animated HTML of the SPH run")
    s.set_defaults(fn=cmd_impact)

    s = sub.add_parser("risk", help="penetration-risk index for one asset")
    s.add_argument("asset")
    s.add_argument("--days", type=float, default=30)
    s.add_argument("--area", type=float, default=10, help="exposed area, m^2")
    s.set_defaults(fn=cmd_risk)

    s = sub.add_parser("validate", help="compare screening with CelesTrak SOCRATES")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("serve", help="run the Live Sky web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("history", help="download historical catalog snapshots")
    s.add_argument("--source", choices=["spacebook", "spacetrack"], default="spacebook",
                   help="spacebook needs no account; spacetrack needs SPACETRACK_USER/PASS")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--step", type=int, default=7, help="days between snapshots")
    s.set_defaults(fn=cmd_history)

    s = sub.add_parser("train", help="train the ST-GNN on cached snapshots")
    s.add_argument("--source", choices=["spacebook", "spacetrack"], default="spacebook")
    s.add_argument("--horizon", type=int, default=4, help="snapshot steps ahead (4 weekly = 28 days)")
    s.add_argument("--lookback", type=int, default=6)
    s.add_argument("--epochs", type=int, default=300)
    s.set_defaults(fn=cmd_train)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
