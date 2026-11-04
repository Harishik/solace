"""Live Sky server: satellites/debris (WebSocket), near-Earth asteroids and the local sky.

Run with ``solace serve`` (or ``uvicorn solace.live.server:app``) and open http://localhost:8000.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..impact.ble import WhippleShield, penetration_probability_given_impact
from ..ingest.celestrak import TLE, fetch_catalog
from ..ingest.spaceweather import fetch_f107, fetch_kp
from ..orbit.conjunction import rescore, screen
from ..orbit.covariance import spacebook_provider
from ..orbit.propagate import Propagator, gmst, jday_array, orbit_shell
from . import neo, sky


def _web_dir() -> Path:
    """Front-end folder: $SOLACE_WEB, else ./web (Docker, repo root), else the source checkout."""
    if os.environ.get("SOLACE_WEB"):
        return Path(os.environ["SOLACE_WEB"])
    for p in (Path.cwd() / "web", Path(__file__).resolve().parents[3] / "web"):
        if (p / "index.html").exists():
            return p
    raise RuntimeError("web/ not found; set SOLACE_WEB")


WEB_DIR = _web_dir()

# Protected assets by NORAD id (catalog names differ between CelesTrak and Spacebook).
ASSETS = {25544: "ISS", 20580: "Hubble", 65053: "NISAR", 39634: "Sentinel-1A", 62261: "Sentinel-1C",
          66315: "Sentinel-1D", 48274: "Tiangong (Tianhe)"}
DEFAULT_SHIELD = WhippleShield()


def kind_of(t: TLE) -> int:
    n = t.name.upper()
    if " DEB" in n or n.endswith("DEB"):
        return 2
    if " R/B" in n:
        return 1
    return 0  # payload


def teme_to_ecef_m(r_km: np.ndarray, jd: float, fr: float) -> np.ndarray:
    th = float(gmst(np.array([jd]), np.array([fr]))[0])
    c, s = np.cos(th), np.sin(th)
    x = c * r_km[..., 0] + s * r_km[..., 1]
    y = -s * r_km[..., 0] + c * r_km[..., 1]
    return np.stack([x, y, r_km[..., 2]], axis=-1) * 1000.0


class State:
    def __init__(self):
        self.ready = threading.Event()
        self.tles: list[TLE] = []
        self.prop: Propagator | None = None
        self.kinds = np.zeros(0, dtype=np.uint8)
        self.by_id: dict[int, int] = {}
        self.assets: dict[int, dict] = {}
        self.neo: dict = {"status": "loading"}
        self.neo_orbits: dict[str, neo.Orbit] = {}

    def load_catalog(self):
        tles = [t for t in fetch_catalog()]
        prop = Propagator(tles)
        self.tles, self.prop = tles, prop
        self.kinds = np.array([kind_of(t) for t in tles], dtype=np.uint8)
        self.by_id = {t.norad_id: i for i, t in enumerate(tles)}
        self.ready.set()

    def screen_assets(self):
        """Screen every asset with assumed covariance first (fast, so the panel fills quickly),
        then re-score the same events with Spacebook synthetic covariance once it downloads."""
        found = {}
        for norad_id, name in ASSETS.items():
            i = self.by_id.get(norad_id)
            if i is None:
                continue
            t = self.tles[i]
            self.assets[t.norad_id] = {"id": t.norad_id, "name": name, "status": "screening", "events": []}
            try:
                found[t.norad_id] = screen(t, self.tles, days=3.0, threshold_km=10.0)[:25]
            except Exception as exc:
                self.assets[t.norad_id].update(status=f"error: {exc}")
                continue
            self._publish(t.norad_id, found[t.norad_id], "assumed")
        for norad_id, events in found.items():
            try:
                self._publish(norad_id, rescore(events, spacebook_provider), "spacebook")
            except Exception as exc:
                print(f"[solace] Spacebook re-score failed for {norad_id}: {exc}")
        # Parsed covariance files are ~2 MB each; free them so a small host stays under its limit.
        from ..orbit.covariance import fetch_covariance

        fetch_covariance.cache_clear()

    def _publish(self, norad_id: int, events, covariance: str):
        rows = []
        for c in sorted(events, key=lambda c: (-c.pc, c.miss_km)):
            d_crit = float(DEFAULT_SHIELD.critical_diameter(c.rel_speed_kms, min(c.impact_angle_deg, 85)))
            row = c.as_dict()
            row.update(
                d_crit_cm=round(d_crit, 3),
                p_penetrate_given_hit=float(penetration_probability_given_impact(d_crit)),
                r_secondary_ecef_m=teme_to_ecef_m(
                    c.r_secondary, *[float(x[0]) for x in jday_array(c.tca, np.array([0.0]))]).tolist(),
            )
            rows.append(row)
        self.assets[norad_id].update(status="ok", events=rows, covariance=covariance,
                                     max_pc=max((r["pc"] for r in rows), default=0.0))

    def load_neo(self):
        try:
            cad = neo.fetch_close_approaches(days=60, dist_au=0.05)
            sentry = neo.fetch_sentry(25)
            pha = neo.fetch_population("pha")
            for o in pha:
                self.neo_orbits[o.name] = o
            # Elements for the close-approach list come from SBDB one object at a time.
            for ca in cad[:40]:
                try:
                    o = _sbdb_orbit(ca["des"])
                    self.neo_orbits[ca["name"]] = o
                    ca["moid_au"] = neo.moid(o)
                    ca["jpl_moid_au"] = o.jpl_moid
                except Exception:
                    ca["moid_au"] = None
            self.neo = {"status": "ok", "close_approaches": cad, "sentry": sentry, "pha_count": len(pha)}
        except Exception as exc:
            self.neo = {"status": f"error: {exc}"}

    def positions(self, when: datetime) -> np.ndarray:
        jd, fr = jday_array(when, np.array([0.0]))
        r, _, _ = self.prop.states(jd, fr)
        return teme_to_ecef_m(r[:, 0, :], float(jd[0]), float(fr[0])).astype(np.float32)

    def frame(self, when: datetime) -> bytes:
        """One WebSocket frame: float64 sim time (unix s) and GMST (rad), then float32 inertial (TEME)
        positions (m) and velocities (m/s) for every object, NaN-free (0 = not propagated)."""
        jd, fr = jday_array(when, np.array([0.0]))
        r, v, _ = self.prop.states(jd, fr)
        th = float(gmst(jd, fr)[0])
        pos = np.nan_to_num(r[:, 0, :] * 1000.0).astype(np.float32)
        vel = np.nan_to_num(v[:, 0, :] * 1000.0).astype(np.float32)
        return np.array([when.timestamp(), th]).tobytes() + pos.tobytes() + vel.tobytes()


def _sbdb_orbit(des: str) -> neo.Orbit:
    from urllib.parse import quote

    from ..ingest.cache import cached_get

    raw = json.loads(cached_get(f"https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={quote(des)}&full-prec=true",
                                f"sbdb_{des.replace(' ', '_')}.json", max_age_s=24 * 3600))
    orb = raw["orbit"]
    el = {e["name"]: float(e["value"]) for e in orb["elements"] if e.get("value") is not None}
    return neo.Orbit(name=raw["object"]["fullname"], a=el["a"], e=el["e"], i=el["i"], om=el["om"],
                     w=el["w"], ma=el["ma"], epoch=float(orb["epoch"]),
                     jpl_moid=float(orb.get("moid") or "nan"))


def _clean(obj):
    """NaN/inf are common in JPL data (unknown diameters, H); JSON has no such values."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(_clean(content))


STATE = State()
app = FastAPI(title="solace Live Sky", default_response_class=SafeJSONResponse)


@app.on_event("startup")
def _startup():
    def work():
        STATE.load_catalog()
        STATE.screen_assets()

    threading.Thread(target=work, daemon=True).start()
    threading.Thread(target=STATE.load_neo, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/api/status")
def status():
    return {"catalog_ready": STATE.ready.is_set(), "objects": len(STATE.tles),
            "assets": {k: v["status"] for k, v in STATE.assets.items()}, "neo": STATE.neo.get("status")}


@app.get("/api/catalog")
def catalog():
    if not STATE.ready.wait(timeout=60):
        raise HTTPException(503, "catalog loading")
    return {"ids": [t.norad_id for t in STATE.tles], "names": [t.name for t in STATE.tles],
            "kinds": STATE.kinds.tolist(), "assets": list(STATE.assets.keys())}


@app.get("/api/object/{norad_id}")
def object_info(norad_id: int, t: float | None = None):
    i = STATE.by_id.get(norad_id)
    if i is None:
        raise HTTPException(404, "unknown object")
    tle = STATE.tles[i]
    sat = STATE.prop.sats[i]
    period_s = 2 * np.pi / (sat.no_kozai / 60.0)
    start = datetime.fromtimestamp(t, timezone.utc) if t else datetime.now(timezone.utc)
    secs = np.linspace(0, period_s, 121)
    jd, fr = jday_array(start, secs)
    r, _, _ = Propagator([tle]).states(jd, fr)
    track = np.stack([teme_to_ecef_m(r[0, k], float(jd[k]), float(fr[k])) for k in range(len(secs))])
    # Ground track: three orbits, Earth-fixed, so the client can draw it on the rotating globe.
    gsecs = np.linspace(0, 3 * period_s, 361)
    gjd, gfr = jday_array(start, gsecs)
    gr, _, _ = Propagator([tle]).states(gjd, gfr)
    ground = np.stack([teme_to_ecef_m(gr[0, k], float(gjd[k]), float(gfr[k])) for k in range(len(gsecs))])
    perigee, apogee = orbit_shell(tle)
    return _clean({
        "id": norad_id, "name": tle.name, "kind": ["payload", "rocket body", "debris"][STATE.kinds[i]],
        "inclination_deg": float(np.rad2deg(sat.inclo)), "period_min": period_s / 60,
        "perigee_km": perigee, "apogee_km": apogee, "tle": [tle.line1, tle.line2],
        "track_ecef_m": np.nan_to_num(track).tolist(),
        "track_teme_m": np.nan_to_num(r[0] * 1000.0).tolist(),
        "ground_ecef_m": np.nan_to_num(ground).tolist(),
        "asset": STATE.assets.get(norad_id),
        "reentry": getattr(_reentry_watch, "by_id", {}).get(norad_id) if _reentry_watch() else None,
    })


def _reentry_watch() -> dict:
    """Today's re-entry forecasts (scripts/reentry_watch.py writes web/data/reentry_watch.json)."""
    path = WEB_DIR / "data" / "reentry_watch.json"
    if not path.exists():
        return {"status": "No forecast yet: run scripts/reentry_watch.py"}
    if getattr(_reentry_watch, "mtime", None) != path.stat().st_mtime:
        _reentry_watch.data = json.loads(path.read_text())
        _reentry_watch.by_id = {r["id"]: r for r in _reentry_watch.data.get("objects", [])}
        _reentry_watch.mtime = path.stat().st_mtime
    return _reentry_watch.data


@app.get("/api/reentry")
def reentry():
    return _reentry_watch()


@app.get("/api/assets")
def assets():
    return _clean(list(STATE.assets.values()))


@app.get("/api/spaceweather")
def spaceweather():
    out = {}
    try:
        f = fetch_f107()
        out["f107"] = {"time": f[-1][0].isoformat(), "value": f[-1][1]}
    except Exception as exc:
        out["f107"] = {"error": str(exc)}
    try:
        k = fetch_kp()
        out["kp"] = {"time": k[-1][0].isoformat(), "value": k[-1][1],
                     "series": [[t.isoformat(), v] for t, v in k[-24:]]}
    except Exception as exc:
        out["kp"] = {"error": str(exc)}
    return out


def _jd_now() -> float:
    return datetime.now(timezone.utc).timestamp() / 86400.0 + 2440587.5


@app.get("/api/neo")
def neo_overview():
    data = dict(STATE.neo)
    if data.get("status") != "ok":
        return data
    jd = _jd_now()
    ts, eph = sky.ephemeris()
    t = ts.tt_jd(jd)
    planets = {}
    for name, key in [("Mercury", "mercury"), ("Venus", "venus"), ("Earth", "earth"), ("Mars", "mars barycenter"),
                      ("Jupiter", "jupiter barycenter")]:
        planets[name] = (eph[key] - eph["sun"]).at(t).ecliptic_xyz().au.tolist()
    pha = [o for o in STATE.neo_orbits.values()]
    pos = np.array([neo.position(o, jd) for o in pha]) if pha else np.zeros((0, 3))
    data.update(
        jd=jd,
        planets=planets,
        planet_paths=_planet_paths(),
        objects={"names": [o.name for o in pha], "xyz": np.round(pos, 5).tolist(),
                 "jpl_moid": [None if np.isnan(o.jpl_moid) else o.jpl_moid for o in pha]},
    )
    return _clean(data)


_PATHS: dict = {}


def _planet_paths():
    if not _PATHS:
        ts, eph = sky.ephemeris()
        for name, key, days in [("Mercury", "mercury", 88), ("Venus", "venus", 225), ("Earth", "earth", 365.25),
                                ("Mars", "mars barycenter", 687), ("Jupiter", "jupiter barycenter", 4333)]:
            t = ts.tt_jd(_jd_now() + np.linspace(0, days, 200))
            _PATHS[name] = np.round((eph[key] - eph["sun"]).at(t).ecliptic_xyz().au.T, 4).tolist()
    return _PATHS


@app.get("/api/neo/orbit")
def neo_orbit(name: str):
    o = STATE.neo_orbits.get(name)
    if o is None:
        raise HTTPException(404, "unknown object")
    return _clean({"name": o.name, "a": o.a, "e": o.e, "i": o.i, "H": o.H, "diameter_km": o.diameter_km,
            "period_days": o.period_days, "path": np.round(neo.orbit_path(o), 5).tolist(),
            "moid_au": neo.moid(o), "jpl_moid_au": None if np.isnan(o.jpl_moid) else o.jpl_moid,
            "position": neo.position(o, _jd_now()).tolist()})


@app.get("/api/sky")
def sky_view(lat: float, lon: float, mag: float = 4.5):
    snap = sky.sky_snapshot(lat, lon, mag_limit=mag)
    passes = []
    i = STATE.by_id.get(25544)
    if i is not None:
        for p in sky.visible_passes(STATE.tles[i], lat, lon, hours=48):
            p["track"] = np.round(sky.track_altaz(STATE.tles[i], lat, lon, p["rise_utc"], p["set_utc"], 40), 2).tolist()
            passes.append(p)
    snap["iss_passes"] = passes
    return _clean(snap)


@app.websocket("/ws/positions")
async def ws_positions(ws: WebSocket):
    """Binary frames once a second (see State.frame): sim time, GMST, inertial positions and velocities.

    The client may send {"rate": 60} to speed up the simulation clock or {"reset": true}.
    """
    await ws.accept()
    await asyncio.get_event_loop().run_in_executor(None, STATE.ready.wait)
    sim_t, wall_t, rate = time.time(), time.time(), 1.0

    async def receiver():
        nonlocal sim_t, wall_t, rate
        while True:
            msg = json.loads(await ws.receive_text())
            now = time.time()
            sim_t, wall_t = sim_t + (now - wall_t) * rate, now
            if msg.get("reset"):
                sim_t = now
            if "rate" in msg:
                rate = float(msg["rate"])

    task = asyncio.create_task(receiver())
    try:
        while True:
            now = time.time()
            t = sim_t + (now - wall_t) * rate
            frame = await asyncio.get_event_loop().run_in_executor(
                None, STATE.frame, datetime.fromtimestamp(t, timezone.utc))
            await ws.send_bytes(frame)
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        task.cancel()
