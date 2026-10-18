"""Write single-channel float32 OpenEXR images (temperatures as HDR data).

A minimal OpenEXR 2 scanline file: one FLOAT channel named "Y" (so viewers show it as
luminance), then the line offset table and the chunks. Compression is ZIP (lossless: blocks
of 16 scanlines, bytes split into even/odd halves, delta-coded, then zlib) or none (one
scanline per chunk). String attributes can carry extra metadata (e.g. the temperature unit).
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_MAGIC = 20000630
_VERSION = 2                # single-part scanline image
_FLOAT = 2                  # channel pixel type
_COMPRESSION = {"none": (0, 1), "zip": (3, 16)}     # EXR compression id, scanlines per chunk
_INCREASING_Y = 0


def _attr(name: str, type_: str, value: bytes) -> bytes:
    return name.encode() + b"\0" + type_.encode() + b"\0" + struct.pack("<i", len(value)) + value


def _zip(raw: bytes) -> bytes:
    """OpenEXR ZIP chunk coding; a chunk that doesn't shrink is stored as is."""
    b = np.frombuffer(raw, np.uint8)
    t = np.concatenate([b[0::2], b[1::2]])
    d = t.copy()
    d[1:] = t[1:] - t[:-1] + 128                     # uint8 arithmetic wraps, as in OpenEXR
    packed = zlib.compress(d.tobytes(), 6)
    return packed if len(packed) < len(raw) else raw


def write_exr(path: Path, pixels: np.ndarray, attributes: dict[str, str] | None = None,
              compression: str = "zip") -> None:
    """Write a 2-D float32 array as a one-channel ("Y") OpenEXR image ("zip" or "none")."""
    pixels = np.asarray(pixels, dtype="<f4")
    if pixels.ndim != 2:
        raise ValueError("EXR pixels must be a 2-D array")
    height, width = pixels.shape
    code, lines = _COMPRESSION[compression]
    box = struct.pack("<4i", 0, 0, width - 1, height - 1)
    channels = b"Y\0" + struct.pack("<iB3xii", _FLOAT, 0, 1, 1) + b"\0"
    header = [
        _attr("channels", "chlist", channels),
        _attr("compression", "compression", bytes([code])),
        _attr("dataWindow", "box2i", box),
        _attr("displayWindow", "box2i", box),
        _attr("lineOrder", "lineOrder", bytes([_INCREASING_Y])),
        _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attr("screenWindowCenter", "v2f", struct.pack("<2f", 0.0, 0.0)),
        _attr("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    ]
    for name, value in (attributes or {}).items():
        header.append(_attr(name, "string", value.encode("utf-8")))
    head = struct.pack("<ii", _MAGIC, _VERSION) + b"".join(header) + b"\0"

    chunks = []
    for y in range(0, height, lines):
        raw = pixels[y:y + lines].tobytes()
        data = _zip(raw) if code else raw
        chunks.append(struct.pack("<ii", y, len(data)) + data)
    offsets, pos = [], len(head) + 8 * len(chunks)
    for chunk in chunks:
        offsets.append(pos)
        pos += len(chunk)
    Path(path).write_bytes(head + struct.pack(f"<{len(offsets)}Q", *offsets) + b"".join(chunks))
