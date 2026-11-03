"""Tiny file cache so daily runs don't hammer public APIs (CelesTrak asks for <= 1 pull per group per 2 h)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

DATA_DIR = Path(os.environ.get("ORBITGUARD_DATA", Path.cwd() / "data"))
USER_AGENT = "orbitguard/0.1 (+https://github.com/; research use)"


def cached_get(url: str, name: str, max_age_s: float = 6 * 3600, timeout: float = 30) -> str:
    """Return the body of ``url``, served from ``DATA_DIR/raw/name`` if fresher than ``max_age_s``.

    If the network fails and a stale copy exists, the stale copy is returned rather than raising,
    so an offline run still works on yesterday's data.
    """
    path = DATA_DIR / "raw" / name
    if path.exists() and time.time() - path.stat().st_mtime < max_age_s:
        return path.read_text()
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
    except requests.RequestException:
        if path.exists():
            return path.read_text()
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(r.text)
    return r.text
