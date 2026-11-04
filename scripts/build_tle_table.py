"""Build data/reentry_tles.npz (debris and rocket bodies) from the cached weekly Spacebook snapshots.
With --payloads, build data/reentry_tles_payload.npz instead."""
import sys
import time
from pathlib import Path

from solace.ingest.cache import DATA_DIR
from solace.risk.reentry import build_tle_table

payloads = "--payloads" in sys.argv
files = sorted((Path(DATA_DIR) / "history").glob("sb_*.tle"))
t = time.time()
tt = build_tle_table(files, progress=True, types=("PAYLOAD",) if payloads else ("DEBRIS", "ROCKET BODY"))
tt.save(Path(DATA_DIR) / ("reentry_tles_payload.npz" if payloads else "reentry_tles.npz"))
print(f"{len(tt.norad)} TLEs, {len(set(tt.norad))} objects, {time.time() - t:.0f}s")
