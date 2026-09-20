"""Pure-Python reader for DJI R-JPEG thermal captures (the "iirp" header variant).

An R-JPEG is a normal JPEG marker stream. The raw 16-bit sensor frame is split across
consecutive APP3 segments, and one APP4 segment carries a small header: a magic tag at
byte 4, the sensor resolution, and the environmental measurement parameters. Layout was
established empirically against DJI Matrice 4T captures; other header variants are
rejected rather than guessed at.
"""

from __future__ import annotations

import dataclasses
import struct

import numpy as np

_APP3 = 0xE3
_APP4 = 0xE4
_MAGIC_OFFSET = 4
_WIDTH_OFFSET = 157
_HEIGHT_OFFSET = 159
_PARAM_OFFSETS = {"ambient_temp": 32, "distance": 36, "emissivity": 40, "humidity": 44, "reflection": 48}
_HEADER_MIN_LEN = _HEIGHT_OFFSET + 2
SUPPORTED_VARIANTS = (b"iirp",)


class RJpegError(ValueError):
    """The data is not a DJI R-JPEG this reader can parse."""


class UnsupportedRJpeg(RJpegError):
    """A DJI R-JPEG whose header variant is not supported by the pure-Python reader."""


@dataclasses.dataclass
class MeasurementParams:
    """Environmental parameters. humidity is a percent (0-100); temperatures are Celsius."""

    distance: float
    humidity: float
    emissivity: float
    reflection: float
    ambient_temp: float

    FIELDS = ("distance", "humidity", "emissivity", "reflection", "ambient_temp")

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.FIELDS}


@dataclasses.dataclass
class RJpeg:
    width: int
    height: int
    raw: np.ndarray  # uint16, shape (height, width)
    params: MeasurementParams
    variant: str


def iter_segments(data: bytes):
    """Yield (marker, payload) for each JPEG segment up to the start of the scan."""
    if data[:2] != b"\xff\xd8":
        raise RJpegError("not a JPEG file")
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return
        marker = data[i + 1]
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:
            return
        (length,) = struct.unpack_from(">H", data, i + 2)
        yield marker, data[i + 4:i + 2 + length]
        i += 2 + length


def parse(data: bytes) -> RJpeg:
    raw_parts: list[bytes] = []
    header: bytes | None = None
    for marker, payload in iter_segments(data):
        if marker == _APP3:
            raw_parts.append(payload)
        elif marker == _APP4 and header is None and payload[_MAGIC_OFFSET + 1:_MAGIC_OFFSET + 4] == b"irp":
            header = payload
    if header is None or not raw_parts:
        raise RJpegError("no DJI thermal data found (not an R-JPEG)")

    magic = bytes(header[_MAGIC_OFFSET:_MAGIC_OFFSET + 4])
    if magic not in SUPPORTED_VARIANTS:
        raise UnsupportedRJpeg(f"R-JPEG header variant {magic.decode('ascii', 'replace')!r} is not supported by the pure-Python reader")
    if len(header) < _HEADER_MIN_LEN:
        raise RJpegError("truncated R-JPEG header")

    (width,) = struct.unpack_from("<H", header, _WIDTH_OFFSET)
    (height,) = struct.unpack_from("<H", header, _HEIGHT_OFFSET)
    blob = b"".join(raw_parts)
    if width == 0 or height == 0 or len(blob) != width * height * 2:
        raise RJpegError(f"raw frame size {len(blob)} bytes does not match {width}x{height} 16-bit pixels")

    values = {name: struct.unpack_from("<f", header, off)[0] for name, off in _PARAM_OFFSETS.items()}
    params = MeasurementParams(
        distance=values["distance"],
        humidity=values["humidity"] * 100.0,
        emissivity=values["emissivity"],
        reflection=values["reflection"],
        ambient_temp=values["ambient_temp"],
    )
    raw = np.frombuffer(blob, dtype="<u2").astype(np.uint16, copy=False).reshape(height, width)
    return RJpeg(width=width, height=height, raw=raw, params=params, variant=magic.decode("ascii"))
