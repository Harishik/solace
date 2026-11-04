"""Cross-check the hindcast's re-entry truth (Spacebook satellite catalogue decay dates) against
CelesTrak's SATCAT, an independently maintained copy of the public catalogue.

Run from the repo root:  .venv/bin/python scripts/validate_truth.py
Writes reports/truth_validation.json.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

from solace.ingest.cache import DATA_DIR, cached_get
from solace.risk.reentry import decay_dates

URL = "https://celestrak.org/pub/satcat.csv"
OUT = Path("reports/truth_validation.json")


def main():
    text = cached_get(URL, "celestrak_satcat.csv", max_age_s=7 * 86400, timeout=120)
    ct = {int(r["NORAD_CAT_ID"]): r["DECAY_DATE"] for r in csv.DictReader(text.splitlines())}
    with np.load(Path(DATA_DIR) / "reentry_tles.npz") as z:
        cohort = set(np.unique(z["norad"]).tolist())
    sb = {k: v for k, v in decay_dates().items() if k in cohort and v >= date(2016, 6, 1)}
    diffs, missing = [], 0
    for k, d in sb.items():
        c = ct.get(k)
        if not c:
            missing += 1
            continue
        diffs.append((date.fromisoformat(c) - d).days)
    diffs = np.array(diffs)
    # Objects CelesTrak lists as decayed but Spacebook does not (would be wrongly treated as still up).
    extra = sum(1 for k in cohort if ct.get(k) and date.fromisoformat(ct[k]) >= date(2016, 6, 1) and k not in sb)
    out = {"n_spacebook_decays": len(sb), "n_compared": int(len(diffs)), "missing_in_celestrak": missing,
           "decayed_in_celestrak_only": extra, "exact_match": float(np.mean(diffs == 0)),
           "within_1_day": float(np.mean(np.abs(diffs) <= 1)), "within_7_days": float(np.mean(np.abs(diffs) <= 7)),
           "median_abs_diff_days": float(np.median(np.abs(diffs))), "max_abs_diff_days": int(np.abs(diffs).max())}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
