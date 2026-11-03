"""Figures and animations for the README (written to assets/).

Static charts read reports/phase3c_results.json. The animations recompute from local data:
- solar_analog.gif: the solar-cycle analog F10.7 forecast issued every 2 months, 2016-2026;
- gto_reentry.gif: a real high-eccentricity (Molniya-type) rocket body with and without Sun/Moon;
- sph_impact.gif: 2-D SPH run of a 2 mm aluminium sphere hitting a Whipple shield at 7 km/s.

Run from the repo root:  .venv/bin/python scripts/make_figures.py [names...]
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

OUT = Path("assets")
RES = Path("reports/phase3c_results.json")

INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
C = {"sgp4": "#9ca3af", "bstar": "#c4b5a5", "persist": "#f59e0b", "analog": "#2563eb", "oracle": "#059669",
     "nols": "#dc2626"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.titlelocation": "left", "figure.facecolor": "white", "legend.frameon": False,
})


def _save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def _res():
    return json.loads(RES.read_text())


def headline():
    r = _res()["all_years"]
    rows = [("SGP4, frozen B*", "sgp4", C["sgp4"]), ("Physics with B* as drag term", "oracle_bstar", C["bstar"]),
            ("Physics, today's solar flux held", "persist", C["persist"]),
            ("SOLACE: physics + solar-cycle forecast", "analog", C["analog"]),
            ("Physics, true future weather (bound)", "oracle", C["oracle"])]
    vals = [100 * r[k]["365"]["median_rel_err"] for _, k, _ in rows]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=[c for *_, c in rows], height=0.62)
    ax.set_xscale("log")
    ax.set_yticks(y, [n for n, *_ in rows])
    for yi, v in zip(y, vals):
        ax.text(v * 1.08, yi, f"{v:.0f} %", va="center", color=INK, fontweight="bold")
    ax.set_xlim(8, 600)
    ax.set_xlabel("Median error in predicted re-entry time (log scale)")
    ax.set_title("1-year re-entry forecasts, 2016–2026 (39,814 re-entries)")
    ax.grid(axis="y", visible=False)
    _save(fig, "headline_error.png")


def by_year():
    r = _res()
    years = sorted(k[5:] for k in r if k.startswith("year_") and r[k]["oracle"]["365"]["median_rel_err"] is not None)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    series = (("persist", "Today's flux held"), ("analog", "Solar-cycle forecast"), ("oracle", "True future weather"))
    for k, lab in series:
        v = [100 * r[f"year_{y}"][k]["365"]["median_rel_err"] for y in years]
        ax.plot(years, v, "o-", color=C[k], lw=2.2, label=lab)
    ax.axvspan(3.5, 5.5, color=C["persist"], alpha=0.08)
    ax.text(4.5, 2, "cycle 25 rises", ha="center", color=C["persist"], fontsize=10)
    ax.set_ylabel("Median timing error, %")
    ax.set_title("Holding today's solar flux fails when the cycle turns")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 62)
    _save(fig, "error_by_year.png")


def coverage():
    r = _res()
    years = [k[5:] for k in r if k.startswith("year_") and r[k]["window_80"]["365"]["coverage_80"] is not None]
    cov = [r[f"year_{y}"]["window_80"]["365"]["coverage_80"] for y in years]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(years, cov, color=[C["analog"] if c >= 0.7 else "#93c5fd" for c in cov], width=0.65)
    ax.axhline(0.8, color=INK, ls="--", lw=1.2, label="Target 80 %")
    ax.axhline(0.40, color=C["nols"], ls=":", lw=1.4, label="First attempt (quantile regression), test years")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of re-entries inside window")
    ax.set_title("Calibrated 80 % re-entry windows (online, no look-ahead)")
    ax.grid(axis="x", visible=False)
    _save(fig, "window_coverage.png")


def lunisolar():
    r = _res()
    bins = [("e < 0.05", "test_e_lt_0.05"), ("0.05–0.2", "test_e_0.05_0.2"), ("e ≥ 0.2\n(GTO-like)", "test_e_ge_0.2")]
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    runs = ((-0.18, "oracle_nols", "Drag + J2 only", C["nols"]), (0.18, "oracle", "+ Sun and Moon", C["oracle"]))
    for dx, k, lab, col in runs:
        v = [100 * r[b][k]["365"]["median_rel_err"] for _, b in bins]
        ax.bar(x + dx, v, 0.34, color=col, label=lab)
        for xi, vi in zip(x + dx, v):
            ax.text(xi, vi + 0.8, f"{vi:.0f}", ha="center", color=INK, fontsize=10)
    ax.set_xticks(x, [n for n, _ in bins])
    ax.set_ylabel("Median timing error, %")
    ax.set_title("Sun and Moon matter for eccentric orbits (test years)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    _save(fig, "lunisolar_ablation.png")


def solar_gif():
    from solace.risk import solar as S
    from solace.risk.drivers import load_space_weather

    sw = load_space_weather()
    last = max(d for d, r in sw.items() if r.get("type", "").startswith("OBS"))
    h = S.SolarHistory(sw, last)
    t0 = date(2008, 1, 1)
    i0 = h.idx(t0)
    days = np.array([t0 + timedelta(int(k)) for k in range(h.idx(last) - i0 + 1)])
    obs = h.f81c[i0:h.idx(last) + 1]
    issues = [date(2016, 6, 1) + timedelta(61 * k) for k in range(62)]
    issues = [d for d in issues if d <= last - timedelta(30)]
    fig, ax = plt.subplots(figsize=(8, 4))

    def frame(n):
        ax.clear()
        d = issues[n]
        mf, _, _ = h.analog(d, 730, fade_days=730.0)
        k = h.idx(d) - i0
        fut = np.array([d + timedelta(j) for j in range(731)])
        ax.plot(days[:k], obs[:k], color=INK, lw=1.6, label="Observed (81-day mean)")
        ax.plot(days[k:], obs[k:], color=GRID, lw=1.6)
        for m in mf:
            ax.plot(fut, m, color=C["analog"], lw=0.8, alpha=0.35)
        ax.plot(fut, np.median(mf, axis=0), color=C["analog"], lw=2.6, label="Solar-cycle analog forecast")
        ax.plot(fut, np.full(731, h.f81[h.idx(d) - 1]), color=C["persist"], lw=2, ls="--", label="Today's flux held")
        ax.axvline(d, color=MUTED, lw=0.8)
        ax.set_xlim(date(2008, 1, 1), date(2028, 12, 31))
        ax.set_ylim(55, 260)
        ax.set_ylabel("F10.7 solar flux, sfu")
        ax.set_title(f"Solar-cycle analog forecast, issued {d:%b %Y}")
        ax.legend(loc="upper left", fontsize=9)

    anim = FuncAnimation(fig, frame, frames=len(issues))
    OUT.mkdir(exist_ok=True)
    anim.save(OUT / "solar_analog.gif", writer=PillowWriter(fps=5), dpi=90)
    plt.close(fig)
    print("wrote", OUT / "solar_analog.gif")


def gto_gif(norad=11550, issue=date(2023, 12, 11)):
    from solace.ingest.cache import DATA_DIR
    from solace.risk import decay as D
    from solace.risk import reentry as R
    from solace.risk.drivers import load_space_weather

    tt = R.TLETable.load(Path(DATA_DIR) / "reentry_tles.npz")
    sw = load_space_weather()
    last = max(d for d, r in sw.items() if r.get("type", "").startswith("OBS"))
    obs = D.observed_table(sw, date(2015, 6, 1), last)
    bodies = D.third_bodies(date(2015, 6, 1), last)
    s = R.issue_set(tt, issue)
    k = int(np.nonzero(s.norad == norad)[0][0])
    t0 = s.jd0[k:k + 1] - R._jd(obs.start)
    t1 = s.jd1[k:k + 1] - R._jd(obs.start)
    actual = (R.decay_dates()[norad] - issue).days
    runs = {}
    for name, b in (("sun_moon", bodies), ("no_sun_moon", None)):
        bc = D.fit_bc(s.a0[k:k + 1], s.e0[k:k + 1], s.ang0[k:k + 1, 0], t0, s.a1[k:k + 1], t1, obs,
                      D.BSTAR_TO_BC * s.bstar[k:k + 1], raan=s.ang0[k:k + 1, 1], argp=s.ang0[k:k + 1, 2], bodies=b)
        a, e, ang, t = s.a1[k:k + 1], s.e1[k:k + 1], s.ang1[k:k + 1], t1.copy()
        hist = []
        for _ in range(140):
            hist.append((float(t[0] - obs.offset(issue)), float(a[0]), float(e[0])))
            a, e, t, down, ang = D.propagate(a, e, ang[:, 0], bc, t, t + 5.0, obs, raan=ang[:, 1], argp=ang[:, 2],
                                             bodies=b)
            if down[0] or t[0] > obs.offset(issue) + 690:
                hist.append((float(t[0] - obs.offset(issue)), float(a[0]), float(e[0])))
                break
        runs[name] = np.array(hist)
        print(name, "reentry day", hist[-1][0] if down[0] else "none", "actual", actual)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.1, 1.5]})
    n_frames = max(len(v) for v in runs.values())
    th = np.linspace(0, 2 * np.pi, 400)

    def frame(n):
        ax0.clear()
        ax1.clear()
        ax0.add_patch(plt.Circle((0, 0), D.RE, color="#93c5fd"))
        ax0.add_patch(plt.Circle((0, 0), D.RE + 120, fill=False, color=MUTED, lw=0.6, ls=":"))
        for name, col in (("no_sun_moon", C["nols"]), ("sun_moon", C["oracle"])):
            hst = runs[name][min(n, len(runs[name]) - 1)]
            a, e = hst[1], hst[2]
            r = a * (1 - e**2) / (1 + e * np.cos(th))
            ax0.plot(r * np.cos(th), r * np.sin(th), color=col, lw=1.4)
        ax0.set_xlim(-50000, 9000)
        ax0.set_ylim(-22000, 22000)
        ax0.set_aspect("equal")
        ax0.axis("off")
        day = runs["sun_moon"][min(n, len(runs["sun_moon"]) - 1)][0]
        ax0.set_title(f"Orbit, day {day:.0f}", fontsize=11)
        for name, col, lab in (("no_sun_moon", C["nols"], "Drag + J2 only"),
                               ("sun_moon", C["oracle"], "+ Sun and Moon")):
            hs = runs[name][: n + 1]
            ax1.plot(hs[:, 0], hs[:, 1] * (1 - hs[:, 2]) - D.RE, color=col, lw=2.2, label=lab)
        ax1.axvline(actual, color=INK, ls="--", lw=1.2)
        ax1.text(actual - 6, 520, "actual\nre-entry", ha="right", color=INK, fontsize=9)
        ax1.axhline(120, color=MUTED, ls=":", lw=0.8)
        ax1.set_xlim(0, 700)
        ax1.set_ylim(0, 600)
        ax1.set_xlabel("Days after forecast")
        ax1.set_ylabel("Perigee altitude, km")
        ax1.set_title(f"SL-6 rocket body {norad}: Sun and Moon lower its perigee", fontsize=11)
        ax1.legend(loc="lower left", fontsize=9)

    anim = FuncAnimation(fig, frame, frames=n_frames + 8)
    anim.save(OUT / "gto_reentry.gif", writer=PillowWriter(fps=12), dpi=85)
    plt.close(fig)
    print("wrote", OUT / "gto_reentry.gif")


def sph_gif():
    from solace.impact.sph import WhippleImpact

    sc = WhippleImpact(resolution=0.15, width=24.0)
    solver = sc.build()
    frames = solver.run(sc.time_to_rear_wall() * 1.6, snapshots=60)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor="#0b0f19")

    def frame(n):
        ax.clear()
        ax.set_facecolor("#0b0f19")
        f = frames[n]
        p, v = f["pos"], np.linalg.norm(f["vel"], axis=1)
        still = v < 0.05
        ax.scatter(p[still, 0], p[still, 1], s=2.2, c="#64748b", linewidths=0)
        ax.scatter(p[~still, 0], p[~still, 1], s=3.0, c=v[~still], cmap="plasma", vmin=0, vmax=7, linewidths=0)
        ax.set_xlim(-14, 14)
        ax.set_ylim(-19, 4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(-13.5, 3.2, "2 mm Al sphere, 7 km/s, Whipple shield (2-D SPH)", color="white", fontsize=10,
                fontweight="bold")
        ax.text(-13.5, 2.0, f"t = {f['t']:.2f} µs    colour = speed (0–7 km/s)", color="#cbd5e1", fontsize=9)
        ax.text(13.8, -1.6, "bumper", color="#cbd5e1", fontsize=8, ha="right")
        ax.text(12.5, -18.3, "rear wall", color="#cbd5e1", fontsize=8, ha="right")

    anim = FuncAnimation(fig, frame, frames=len(frames))
    anim.save(OUT / "sph_impact.gif", writer=PillowWriter(fps=12), dpi=85, savefig_kwargs={"facecolor": "#0b0f19"})
    plt.close(fig)
    print("wrote", OUT / "sph_impact.gif")


ALL = {"headline": headline, "by_year": by_year, "coverage": coverage, "lunisolar": lunisolar,
       "solar_gif": solar_gif, "gto_gif": gto_gif, "sph_gif": sph_gif}

if __name__ == "__main__":
    for name in sys.argv[1:] or ALL:
        ALL[name]()
