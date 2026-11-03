"""Download open thermospheric density observations and compare them with NRLMSIS 2.0.

Sources (ESA Swarm dissemination server, no account needed):
- GRACE-FO 1, accelerometer-derived (TU Delft TOLEOS, GF_OPER_DNS1ACC_2), 2018-, ~490 km;
- Swarm B and C, precise-orbit-derived (SW_OPER_DNSxPOD_2), 2014-, ~450-520 km;
- CHAMP (2000-2010, ~300-450 km) and GRACE 1 (2002-2017, ~400-500 km), accelerometer-derived
  (TOLEOS). These cover the cycle 23 maximum, where F10.7 was as high as in 2024.

Each daily file is reduced to 5-minute samples of log(rho_observed / rho_MSIS), with MSIS run at
the satellite's own position and time using the same F10.7/Ap drivers as the re-entry model
(previous-day F10.7, trailing 81-day mean, daily Ap). Raw files are deleted after reduction.
Writes data/density/<sat>.npz. Resumes if interrupted.

Run from the repo root:  .venv/bin/python scripts/fetch_density.py [--step 2]
"""

from __future__ import annotations

import argparse
import io
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np

from solace.ingest.cache import DATA_DIR
from solace.risk import decay as D
from solace.risk.drivers import load_space_weather

BASE = "https://swarm-diss.eo.esa.int/?do=download&file="
OUT = Path(DATA_DIR) / "density"
SATS = {
    "gracefo1": (date(2018, 6, 1), lambda d: f"swarm/Multimission/GRACE-FO/DNS/Sat_1/{d.year}/"
                 f"GF_OPER_DNS1ACC_2__{d:%Y%m%d}T000000_{d:%Y%m%d}T235959_0003.cdf", 300),
    "swarmC": (date(2014, 1, 1), lambda d: f"swarm/Level2daily/Latest_baselines/DNS/POD/Sat_C/"
               f"SW_OPER_DNSCPOD_2__{d:%Y%m%d}T000000_{d:%Y%m%d}T235930_0301.ZIP", 300),
    "swarmB": (date(2014, 1, 1), lambda d: f"swarm/Level2daily/Latest_baselines/DNS/POD/Sat_B/"
               f"SW_OPER_DNSBPOD_2__{d:%Y%m%d}T000000_{d:%Y%m%d}T235930_0301.ZIP", 300),
    "champ": (date(2000, 7, 29), lambda d: f"swarm/Multimission/CHAMP/DNS/"
              f"CH_OPER_DNS_ACC_2__{d:%Y%m%d}T000000_{d:%Y%m%d}T235959_0001.cdf", 300, date(2010, 9, 4)),
    "grace1": (date(2002, 4, 4), lambda d: f"swarm/Multimission/GRACE/DNS/Sat_1/{d.year}/"
               f"GR_OPER_DNS1ACC_2__{d:%Y%m%d}T000000_{d:%Y%m%d}T235959_0001.cdf", 300, date(2017, 10, 27)),
}
FIELDS = ("t", "alt_km", "lat", "lst", "logratio", "f107", "f107a", "ap")


def _read(raw: bytes, name: str):
    import cdflib

    with tempfile.TemporaryDirectory() as tmp:
        if name.endswith(".ZIP"):
            z = zipfile.ZipFile(io.BytesIO(raw))
            cdf = [n for n in z.namelist() if n.endswith(".cdf")][0]
            path = Path(tmp) / "x.cdf"
            path.write_bytes(z.read(cdf))
        else:
            path = Path(tmp) / "x.cdf"
            path.write_bytes(raw)
        c = cdflib.CDF(str(path))
        g = {v: np.asarray(c.varget(v)) for v in ("time", "altitude", "latitude", "longitude", "local_solar_time",
                                                   "density", "validity_flag")}
    return g


def reduce_day(sat: str, d: date, drivers, cadence_s: int):
    import cdflib
    import pymsis

    name = SATS[sat][1](d)
    try:
        raw = urlopen(BASE + quote(name, safe=""), timeout=120).read()
    except Exception:
        return None
    if len(raw) < 1000:
        return None
    g = _read(raw, name)
    secs = cdflib.cdfepoch.unixtime(g["time"])
    secs = np.asarray(secs, float)
    ok = (g["validity_flag"] == 0) & np.isfinite(g["density"]) & (g["density"] > 0) & (g["density"] < 1e-8)
    idx = np.nonzero(ok)[0]
    if len(idx) < 50:
        return None
    bins = np.floor(secs[idx] / cadence_s)
    keep = idx[np.r_[True, np.diff(bins) > 0]]  # first valid sample in each cadence bin
    f, fa, ap = drivers
    t = secs[keep]
    times = np.array([np.datetime64(int(x), "s") for x in t])
    n = len(t)
    out = pymsis.calculate(times, g["longitude"][keep], g["latitude"][keep], g["altitude"][keep] / 1e3,
                           f107s=np.full(n, f), f107as=np.full(n, fa), aps=np.full((n, 7), ap))
    rho_m = out[..., pymsis.Variable.MASS_DENSITY].reshape(n)
    return {"t": t, "alt_km": g["altitude"][keep] / 1e3, "lat": g["latitude"][keep], "lst": g["local_solar_time"][keep],
            "logratio": np.log(g["density"][keep] / rho_m), "f107": np.full(n, f), "f107a": np.full(n, fa),
            "ap": np.full(n, ap)}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--step", type=int, default=2, help="take every n-th day")
    ap_.add_argument("--sats", default="gracefo1,swarmC,swarmB")
    ap_.add_argument("--workers", type=int, default=8)
    a = ap_.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sw = load_space_weather()
    last = max(d for d, r in sw.items() if r.get("type", "").startswith("OBS"))
    for sat in a.sats.split(","):
        first, end = SATS[sat][0], min(last, SATS[sat][3]) if len(SATS[sat]) > 3 else last
        days = [first + timedelta(k) for k in range(0, (end - first).days + 1, a.step)]
        path = OUT / f"{sat}.npz"
        have = {}
        if path.exists():
            z = np.load(path)
            have = {k: list(z[k]) for k in FIELDS}
        done = {datetime.fromtimestamp(x, timezone.utc).date() for x in have.get("t", [])}
        todo = [d for d in days if d not in done]
        print(f"{sat}: {len(days)} days, {len(todo)} to fetch", flush=True)
        drv = dict(zip(days, zip(*D._drivers(days, sw))))
        cols = {k: [np.asarray(have.get(k, []), float)] for k in FIELDS}
        with ThreadPoolExecutor(a.workers) as ex:
            jobs = [(sat, d, drv[d], SATS[sat][2]) for d in todo]
            for k, res in enumerate(ex.map(lambda j: reduce_day(*j), jobs)):
                if res:
                    for f in FIELDS:
                        cols[f].append(res[f])
                if k % 100 == 99 or k == len(todo) - 1:
                    np.savez_compressed(path, **{f: np.concatenate(cols[f]) for f in FIELDS})
                    print(f"  {sat} {k + 1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
