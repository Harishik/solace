"""Physics baseline forecasts that any learned model has to beat.

* Persistence: the grid in H days equals the grid today.
* SGP4 decay: propagate every object H days with SGP4 (whose B* drag term encodes each object's
  observed decay rate) and rebuild the grid there; objects that re-enter drop out.
* Hybrid (the default prior): payloads persist and only debris and rocket bodies decay. Active
  payloads station-keep and raise their orbits, so pure decay drags whole constellation shells
  down (on 2016-2026 history it is worse than persistence in the Starlink shells). Launches and
  breakups are unknown to all three, which is what a learned model can add.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from ..ingest.celestrak import TLE
from .shells import ShellGrid, build_grid


def sgp4_forecast(tles: list[TLE], start: datetime, horizon_days: float) -> ShellGrid:
    return build_grid(tles, when=start + timedelta(days=horizon_days))


def hybrid_forecast(tles: list[TLE], start: datetime, horizon_days: float, today: ShellGrid | None = None) -> ShellGrid:
    """Payloads from today's grid, debris and rocket bodies from the SGP4 decay forecast."""
    today = today or build_grid(tles, when=start)
    decayed = sgp4_forecast(tles, start, horizon_days)
    counts = decayed.counts.copy()
    counts[..., 0] = today.counts[..., 0]
    return ShellGrid(counts=counts, bstar=decayed.bstar)
