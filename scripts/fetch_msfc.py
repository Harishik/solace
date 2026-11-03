"""Download NASA MSFC's archived monthly solar-cycle forecasts (F10.7 and Ap, 13-month smoothed).

MSFC has issued these since 1999 for spacecraft lifetime and re-entry planning; each file gives
the 5/50/95 % percentiles month by month for the rest of the cycle. The archive page lists every
issue: https://www.nasa.gov/solar-cycle-progression-and-forecast/archived-forecast/
From 2019 there are two tables per month; we use the standard one ("prd"), not "prd75".

Writes data/solar/msfc_forecasts.npz with one row per (issue month, forecast month):
issue (ordinal of the 1st of the issue month), t (decimal year), f05, f50, f95, ap05, ap50, ap95.

Run from the repo root:  .venv/bin/python scripts/fetch_msfc.py
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from solace.ingest.cache import DATA_DIR

PAGE = "https://www.nasa.gov/solar-cycle-progression-and-forecast/archived-forecast/"
OUT = Path(DATA_DIR) / "solar"
MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
ROW = re.compile(r"^\s*(\d{4}\.\d+)\s+[A-Z]{3}" + r"\s+(-?[\d.]+)" * 6)


def _get(url: str) -> str:
    return urlopen(Request(url, headers={"User-Agent": "solace"}), timeout=60).read().decode("latin-1")


def _issues(html: str) -> dict[date, str]:
    """Issue month -> URL, preferring the standard table over the "prd75" variant."""
    best: dict[date, tuple[int, str]] = {}
    for url in set(re.findall(r'https?://[^"\s]+f10[^"\s]*\.txt[^"\s]*', html)):
        m = re.search(r"/([a-z]{3})(\d{4})f10([^/?]*)\.txt", url)
        if not m or m.group(1) not in MONTHS:
            continue
        d = date(int(m.group(2)), MONTHS.index(m.group(1)) + 1, 1)
        rank = 1 if "75" in m.group(3) else 0
        if d not in best or rank < best[d][0]:
            best[d] = (rank, url.replace("&amp;", "&"))
    return {d: u for d, (_, u) in sorted(best.items())}


def parse(text: str) -> np.ndarray:
    """(n, 7) rows: t, f95, f50, f05, ap95, ap50, ap05."""
    rows = [tuple(float(x) for x in m.groups()) for m in map(ROW.match, text.splitlines()) if m]
    return np.array(rows, float).reshape(-1, 7)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    issues = _issues(_get(PAGE))
    cols = {k: [] for k in ("issue", "t", "f05", "f50", "f95", "ap05", "ap50", "ap95")}
    for d, url in issues.items():
        try:
            r = parse(_get(url))
        except Exception as e:  # a few archive links are dead; skip them
            print(f"{d:%Y-%m} skipped ({e})")
            continue
        if len(r) == 0:
            print(f"{d:%Y-%m} no table")
            continue
        cols["issue"].append(np.full(len(r), d.toordinal()))
        for k, j in (("t", 0), ("f95", 1), ("f50", 2), ("f05", 3), ("ap95", 4), ("ap50", 5), ("ap05", 6)):
            cols[k].append(r[:, j])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez_compressed(OUT / "msfc_forecasts.npz", **out)
    print(f"{len(np.unique(out['issue']))} issues, {len(out['t'])} rows -> {OUT / 'msfc_forecasts.npz'}")


if __name__ == "__main__":
    main()
