"""Cross-check the screening engine against CelesTrak SOCRATES.

SOCRATES runs SGP4 on the same public GP data, so for events whose two objects are both in our
catalog we should reproduce its TCA and minimum range closely. Differences come from TLE updates
between their run and ours.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone

import numpy as np

from .ingest.cache import cached_get
from .ingest.celestrak import TLE, fetch_catalog
from .orbit.conjunction import rescore, screen

SOCRATES_CSV = "https://celestrak.org/SOCRATES/sort-minRange.csv"


def load_socrates() -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(cached_get(SOCRATES_CSV, "socrates_minrange.csv"))))
    for r in rows:
        r["TCA"] = datetime.strptime(r["TCA"], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        for k in ("TCA_RANGE", "TCA_RELATIVE_SPEED", "MAX_PROB", "DSE_1", "DSE_2"):
            r[k] = float(r[k])
    return rows


def compare(max_events: int = 100, catalog: list[TLE] | None = None) -> list[dict]:
    by_id = {t.norad_id: t for t in (catalog or fetch_catalog())}
    now = datetime.now(timezone.utc)
    out = []
    for r in load_socrates():
        if len(out) >= max_events:
            break
        a, b = by_id.get(int(r["NORAD_CAT_ID_1"])), by_id.get(int(r["NORAD_CAT_ID_2"]))
        if a is None or b is None or r["TCA_RELATIVE_SPEED"] < 0.05 or r["TCA"] < now:
            continue
        found = screen(a, [b], start=r["TCA"] - timedelta(minutes=30), days=1 / 24, step_s=20,
                       threshold_km=50, docked_km=0.0)
        if not found:
            out.append({"pair": (a.norad_id, b.norad_id), "found": False, "ref_range_km": r["TCA_RANGE"]})
            continue
        best = min(found, key=lambda c: abs((c.tca - r["TCA"]).total_seconds()))
        # SOCRATES reports days-since-epoch; recover its TLE epochs and compare with ours.
        ref_ep1 = r["TCA"] - timedelta(days=r["DSE_1"])
        ref_ep2 = r["TCA"] - timedelta(days=r["DSE_2"])
        same_tles = (abs((tle_epoch(a) - ref_ep1).total_seconds()) < 600
                     and abs((tle_epoch(b) - ref_ep2).total_seconds()) < 600)
        out.append({
            "pair": (a.norad_id, b.norad_id),
            "found": True,
            "same_tles": same_tles,
            "ref_range_km": r["TCA_RANGE"],
            "our_range_km": best.miss_km,
            "dt_s": (best.tca - r["TCA"]).total_seconds(),
            "ref_speed": r["TCA_RELATIVE_SPEED"],
            "our_speed": best.rel_speed_kms,
        })
    return out


def tle_epoch(t: TLE) -> datetime:
    yy, doy = int(t.line1[18:20]), float(t.line1[20:32])
    year = 2000 + yy if yy < 57 else 1900 + yy
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)


def summarize(results: list[dict]) -> str:
    same = [r for r in results if r["found"] and r["same_tles"]]
    lines = [_summary_line("all events", results)]
    lines.append(_summary_line("identical TLE epochs", same))
    return "\n".join(lines)


def _summary_line(label: str, results: list[dict]) -> str:
    ok = [r for r in results if r["found"]]
    if not ok:
        return f"{label}: 0/{len(results)} events reproduced"
    dr = np.array([abs(r["our_range_km"] - r["ref_range_km"]) for r in ok])
    dt = np.array([abs(r["dt_s"]) for r in ok])
    dv = np.array([abs(r["our_speed"] - r["ref_speed"]) for r in ok])
    return (f"{label}: {len(ok)}/{len(results)} events reproduced | "
            f"|d range| median {np.median(dr):.3f} km, p90 {np.percentile(dr, 90):.3f} km | "
            f"|d TCA| median {np.median(dt):.2f} s, p90 {np.percentile(dt, 90):.2f} s | "
            f"|d speed| median {np.median(dv):.4f} km/s")


def covariance_study(asset_ids=(65053, 20580, 25544, 39634, 62261, 66315),
                     days: float = 3.0, catalog: list[TLE] | None = None) -> dict:
    """How much does covariance realism change Pc and its ranking?

    Screens each asset once, then scores every event twice: assumed RIC sigmas vs Spacebook
    synthetic covariance. Reports rank agreement and how many events change alert tier.
    """
    from scipy.stats import spearmanr

    from .orbit.covariance import spacebook_provider

    cat = catalog or fetch_catalog()
    by_id = {t.norad_id: t for t in cat}
    rows = []
    for norad_id in asset_ids:
        a = by_id.get(norad_id)
        if a is None:
            continue
        name = a.name
        assumed = screen(a, cat, days=days)
        for b, c in zip(assumed, rescore(assumed, spacebook_provider)):
            if c.cov_source != "spacebook":
                continue
            rows.append({"asset": name, "secondary": c.secondary.name, "miss_km": c.miss_km,
                         "pc_assumed": b.pc, "pc_spacebook": c.pc})
    if not rows:
        return {"events": 0}
    lo = lambda x: np.log10(max(x, 1e-30))  # noqa: E731
    pa = np.array([lo(r["pc_assumed"]) for r in rows])
    ps = np.array([lo(r["pc_spacebook"]) for r in rows])
    tier = lambda p: np.digitize(p, [-6, -4])  # <1e-6 | 1e-6..1e-4 | >=1e-4  # noqa: E731
    return {
        "events": len(rows),
        "spearman_rank": float(spearmanr(pa, ps).statistic),
        # Only where both Pc are non-negligible; near-zero values make log ratios meaningless.
        "median_log10_ratio_both_above_1e-12": float(np.median((ps - pa)[(pa > -12) & (ps > -12)]))
        if ((pa > -12) & (ps > -12)).any() else None,
        "pc_up_10x": int((ps - pa > 1).sum()),
        "pc_down_10x": int((pa - ps > 1).sum()),
        "tier_changes": int((tier(pa) != tier(ps)).sum()),
        "above_1e-6_assumed": int((pa >= -6).sum()),
        "above_1e-6_spacebook": int((ps >= -6).sum()),
        "rows": sorted(rows, key=lambda r: -r["pc_spacebook"]),
    }


if __name__ == "__main__":
    print(summarize(compare()))
