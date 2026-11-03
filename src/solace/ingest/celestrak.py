"""Two-line element sets from CelesTrak's public GP service.

NASA's earthaccess / GIBS serve Earth-observation products (including NISAR SAR scenes), not a
space-object catalog, so orbital states for payloads and debris come from CelesTrak, which
republishes the 18th Space Defense Squadron catalog without a login.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cache import cached_get

GP_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
CATNR_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=tle"

# Fragmentation clouds that dominate LEO debris flux.
DEBRIS_GROUPS = (
    "cosmos-1408-debris",
    "fengyun-1c-debris",
    "iridium-33-debris",
    "cosmos-2251-debris",
)


_ALPHA5 = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # I and O are skipped to avoid confusion with 1 and 0


def decode_catnr(field: str) -> int:
    """Catalog number from a TLE's 5-character field, including Alpha-5 (``A0000`` = 100000)
    used once numbers passed 99999."""
    field = field.strip()
    if field and field[0].isalpha():
        return (_ALPHA5.index(field[0].upper()) + 10) * 10000 + int(field[1:])
    return int(field)


@dataclass(frozen=True)
class TLE:
    name: str
    line1: str
    line2: str

    @property
    def norad_id(self) -> int:
        return decode_catnr(self.line1[2:7])

    @property
    def is_debris(self) -> bool:
        n = self.name.upper()
        return " DEB" in n or n.endswith("DEB") or " R/B" in n


def parse_tle_text(text: str) -> list[TLE]:
    """Parse 3-line (name + 2 lines) TLE text, tolerating blank lines and missing names."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out: list[TLE] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name, l1, l2 = lines[i][2:7].strip(), lines[i], lines[i + 1]
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name, l1, l2 = lines[i].strip(), lines[i + 1], lines[i + 2]
            i += 3
        else:
            i += 1
            continue
        out.append(TLE(name, l1, l2))
    return out


def fetch_group(group: str) -> list[TLE]:
    return parse_tle_text(cached_get(GP_URL.format(group=group), f"tle_{group}.txt"))


def fetch_satellite(norad_id: int) -> TLE:
    tles = parse_tle_text(cached_get(CATNR_URL.format(catnr=norad_id), f"tle_{norad_id}.txt"))
    if not tles:
        raise LookupError(f"No TLE for NORAD {norad_id}")
    return tles[0]


def fetch_catalog(source: str = "full") -> list[TLE]:
    """The on-orbit public catalog.

    ``full`` (default): every tracked object, from COMSPOC Spacebook's mirror of the public
    catalog (~33k objects, ~12k of them debris and rocket bodies), falling back to CelesTrak groups.
    ``celestrak``: CelesTrak's active satellites plus the four largest fragmentation clouds only
    (~19k objects, ~2.7k debris) -- it misses most debris, so it underestimates flux.
    """
    if source == "full":
        from .spacebook import fetch_spacebook_catalog

        try:
            tles = fetch_spacebook_catalog()
            if len(tles) > 10000:
                return tles
            print(f"[solace] Spacebook catalog looks short ({len(tles)}); using CelesTrak")
        except Exception as exc:
            print(f"[solace] Spacebook catalog unavailable ({exc}); using CelesTrak")
    return fetch_celestrak_groups(fallback=source != "full")


def fetch_celestrak_groups(groups=DEBRIS_GROUPS + ("active",), fallback: bool = True) -> list[TLE]:
    """Union of groups, de-duplicated by NORAD id. Groups that fail to download are skipped.

    CelesTrak answers 403 to clients that re-download unchanged data too often (a fresh server
    with an empty cache can trip this). If the large ``active`` group fails and ``fallback`` is
    set, the full public catalog is taken from COMSPOC Spacebook instead.
    """
    seen: dict[int, TLE] = {}
    failed = []
    for g in groups:
        try:
            for t in fetch_group(g):
                seen.setdefault(t.norad_id, t)
        except Exception as exc:  # one dead group shouldn't sink the run
            failed.append(g)
            print(f"[solace] skipping group {g}: {exc}")
    if fallback and "active" in failed:
        from .spacebook import fetch_spacebook_catalog

        try:
            for t in fetch_spacebook_catalog():
                seen.setdefault(t.norad_id, t)
            print("[solace] used Spacebook catalog as fallback")
        except Exception as exc:
            print(f"[solace] Spacebook fallback failed: {exc}")
    return list(seen.values())
