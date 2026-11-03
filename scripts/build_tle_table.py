"""Build reports/../data/reentry_tles.npz from the cached weekly Spacebook snapshots."""
import time
from pathlib import Path

from orbitguard.ingest.cache import DATA_DIR
from orbitguard.risk.reentry import build_tle_table

files = sorted((Path(DATA_DIR) / "history").glob("sb_*.tle"))
t = time.time()
tt = build_tle_table(files, progress=True)
tt.save(Path(DATA_DIR) / "reentry_tles.npz")
print(f"{len(tt.norad)} TLEs, {len(set(tt.norad))} objects, {time.time() - t:.0f}s")
