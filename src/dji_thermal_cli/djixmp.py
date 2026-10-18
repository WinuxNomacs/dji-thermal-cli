"""DJI's drone-dji XMP fields (position, attitude, rangefinder, RTK quality, capture time)."""

from __future__ import annotations

import re

from . import tiffout

_FIELD = re.compile(rb'drone-dji:(\w+)(?:\s*=\s*"([^"]*)"|>([^<]*)<)')


def fields(data: bytes) -> dict[str, str]:
    """Every drone-dji field of the JPEG's XMP, written as attributes or as elements; the
    first occurrence wins."""
    out: dict[str, str] = {}
    for m in _FIELD.finditer(tiffout.read_metadata(data).xmp or b""):
        out.setdefault(m.group(1).decode(), (m.group(2) or m.group(3) or b"").decode("latin-1").strip())
    return out


def number(f: dict[str, str], key: str) -> float | None:
    """A field as a finite float, or None when missing or not a number."""
    try:
        v = float(f[key])
    except (KeyError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None
