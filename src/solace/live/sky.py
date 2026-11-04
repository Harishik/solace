"""Planets, Moon, Sun and bright stars for an observer, plus visible satellite passes (Skyfield)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
from skyfield.api import EarthSatellite, Loader, Star, wgs84
from skyfield.data import hipparcos

from ..ingest.cache import DATA_DIR
from ..ingest.celestrak import TLE

BODIES = {
    "Sun": "sun", "Moon": "moon", "Mercury": "mercury", "Venus": "venus", "Mars": "mars barycenter",
    "Jupiter": "jupiter barycenter", "Saturn": "saturn barycenter",
    "Uranus": "uranus barycenter", "Neptune": "neptune barycenter",
}

# Common names for the brightest Hipparcos stars, so the sky view can label them.
STAR_NAMES = {
    32349: "Sirius", 30438: "Canopus", 69673: "Arcturus", 71683: "Rigil Kentaurus", 91262: "Vega",
    24608: "Capella", 24436: "Rigel", 37279: "Procyon", 7588: "Achernar", 27989: "Betelgeuse",
    68702: "Hadar", 97649: "Altair", 60718: "Acrux", 21421: "Aldebaran", 65474: "Spica",
    80763: "Antares", 37826: "Pollux", 113368: "Fomalhaut", 102098: "Deneb", 62434: "Mimosa",
    49669: "Regulus", 33579: "Adhara", 36850: "Castor", 11767: "Polaris", 25336: "Bellatrix",
    26311: "Alnilam", 26727: "Alnitak", 25930: "Mintaka", 54061: "Dubhe", 67301: "Alkaid",
}


@lru_cache(maxsize=1)
def _loader():
    path = DATA_DIR / "skyfield"
    path.mkdir(parents=True, exist_ok=True)
    return Loader(str(path), verbose=False)


@lru_cache(maxsize=1)
def ephemeris():
    load = _loader()
    return load.timescale(), load("de440s.bsp")


@lru_cache(maxsize=1)
def bright_stars(mag_limit: float = 4.5):
    load = _loader()
    with load.open(hipparcos.URL) as f:
        df = hipparcos.load_dataframe(f)
    df = df[(df["magnitude"] <= mag_limit) & df["ra_degrees"].notna()]
    return df, Star.from_dataframe(df)


def sky_snapshot(lat: float, lon: float, when: datetime | None = None, mag_limit: float = 4.5) -> dict:
    ts, eph = ephemeris()
    t = ts.from_datetime(when or datetime.now(timezone.utc))
    observer = eph["earth"] + wgs84.latlon(lat, lon)
    bodies = []
    for name, key in BODIES.items():
        app = observer.at(t).observe(eph[key]).apparent()
        alt, az, dist = app.altaz()
        ra, dec, _ = app.radec(epoch="date")
        bodies.append({"name": name, "alt": alt.degrees, "az": az.degrees, "dist_au": dist.au,
                       "ra": ra._degrees, "dec": dec.degrees})

    df, stars = bright_stars(mag_limit)
    app = observer.at(t).observe(stars).apparent()
    alt, az, _ = app.altaz()
    ra, dec, _ = app.radec(epoch="date")
    # The whole sky (not only what is up now), so the client can turn it with the simulation clock.
    star_rows = [
        {"hip": int(h), "name": STAR_NAMES.get(int(h), ""), "alt": float(a), "az": float(z), "mag": float(m),
         "ra": float(r), "dec": float(d)}
        for h, a, z, m, r, d in zip(df.index.to_numpy(), alt.degrees, az.degrees, df["magnitude"].to_numpy(),
                                    ra._degrees, dec.degrees)
    ]
    sun_alt = next(b["alt"] for b in bodies if b["name"] == "Sun")
    return {
        "time_utc": t.utc_iso(),
        "lat": lat,
        "lon": lon,
        "sun_alt": sun_alt,
        "is_night": sun_alt < -6,
        "bodies": bodies,
        "stars": star_rows,
    }


def visible_passes(tle: TLE, lat: float, lon: float, hours: float = 24, min_alt: float = 10.0) -> list[dict]:
    """Passes above ``min_alt`` where the satellite is sunlit and the observer is in darkness."""
    ts, eph = ephemeris()
    sat = EarthSatellite(tle.line1, tle.line2, tle.name, ts)
    site = wgs84.latlon(lat, lon)
    now = datetime.now(timezone.utc)
    t0, t1 = ts.from_datetime(now), ts.from_datetime(now + timedelta(hours=hours))
    times, events = sat.find_events(site, t0, t1, altitude_degrees=min_alt)
    out, cur = [], {}
    for t, ev in zip(times, events):
        if ev == 0:
            cur = {"rise_utc": t.utc_iso()}
        elif ev == 1 and cur:
            alt, az, _ = (sat - site).at(t).altaz()
            sun_alt = (eph["earth"] + site).at(t).observe(eph["sun"]).apparent().altaz()[0].degrees
            cur.update(culminate_utc=t.utc_iso(), max_alt=float(alt.degrees), az=float(az.degrees),
                       sunlit=bool(sat.at(t).is_sunlit(eph)), observer_dark=bool(sun_alt < -6))
        elif ev == 2 and "culminate_utc" in cur:
            cur["set_utc"] = t.utc_iso()
            cur["visible"] = cur["sunlit"] and cur["observer_dark"]
            out.append(cur)
            cur = {}
    return out


def track_altaz(tle: TLE, lat: float, lon: float, start: str, end: str, n: int = 60) -> np.ndarray:
    ts, _ = ephemeris()
    sat = EarthSatellite(tle.line1, tle.line2, tle.name, ts)
    a = datetime.fromisoformat(start.replace("Z", "+00:00"))
    b = datetime.fromisoformat(end.replace("Z", "+00:00"))
    t = ts.from_datetimes([a + (b - a) * k / (n - 1) for k in range(n)])
    alt, az, _ = (sat - wgs84.latlon(lat, lon)).at(t).altaz()
    return np.column_stack([alt.degrees, az.degrees])
