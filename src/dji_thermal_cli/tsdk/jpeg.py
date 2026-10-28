"""R-JPEG container parsing: APPn segments, EXIF camera model, infrared ND-filter info."""

from __future__ import annotations

import dataclasses
import struct

from .errors import RJpegError

APP1, APP2, APP3, APP4, APP5, APP7 = 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE7


@dataclasses.dataclass
class Segment:
    marker: int
    payload: bytes
    offset: int = 0          # file offset of the payload
    data: bytes = b""        # the whole file, for readers that run past a segment's end (as the SDK does)

    def read(self, start: int, size: int) -> bytes:
        """`size` bytes from `start` within this payload, continuing into the following file
        bytes when the payload is shorter (the SDK reads fixed-size tables that way)."""
        return self.data[self.offset + start:self.offset + start + size]


def segments(data: bytes) -> list[Segment]:
    """Every marker segment before the start of scan, in file order."""
    if data[:2] != b"\xff\xd8":
        raise RJpegError("not a JPEG file")
    out: list[Segment] = []
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker == 0xFF:          # fill byte
            i += 1
            continue
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:
            break
        (length,) = struct.unpack_from(">H", data, i + 2)
        out.append(Segment(marker, data[i + 4:i + 2 + length], i + 4, data))
        i += 2 + length
    return out


def concat(segs: list[Segment], marker: int) -> bytes:
    return b"".join(s.payload for s in segs if s.marker == marker)


def first(segs: list[Segment], marker: int) -> bytes | None:
    return next((s.payload for s in segs if s.marker == marker), None)


def first_segment(segs: list[Segment], marker: int) -> Segment | None:
    return next((s for s in segs if s.marker == marker), None)


def is_flir(segs: list[Segment]) -> bool:
    """Whether the JPEG carries FLIR's radiometric data (APP1 segments starting "FLIR\\0", as
    FLIR cameras and DJI's Zenmuse XT2 write)."""
    return any(s.marker == APP1 and s.payload.startswith(b"FLIR\0") for s in segs)


def exif_make_model(segs: list[Segment]) -> tuple[str, str]:
    """IFD0 Make (0x010F) and Model (0x0110) from the APP1 Exif segment; '' when absent."""
    for s in segs:
        if s.marker != APP1 or not s.payload.startswith(b"Exif\0\0"):
            continue
        tiff = s.payload[6:]
        if len(tiff) < 8:
            break
        endian = {b"II": "<", b"MM": ">"}.get(tiff[:2])
        if endian is None:
            break
        (ifd,) = struct.unpack_from(endian + "I", tiff, 4)
        if ifd + 2 > len(tiff):
            break
        (count,) = struct.unpack_from(endian + "H", tiff, ifd)
        found = {}
        for k in range(count):
            off = ifd + 2 + 12 * k
            if off + 12 > len(tiff):
                break
            tag, typ, n = struct.unpack_from(endian + "HHI", tiff, off)
            if tag in (0x010F, 0x0110) and typ == 2:
                where = off + 8 if n <= 4 else struct.unpack_from(endian + "I", tiff, off + 8)[0]
                raw = tiff[where:where + n].split(b"\0", 1)[0]
                found[tag] = raw.decode("latin-1").rstrip(" ")
        return found.get(0x010F, ""), found.get(0x0110, "")
    return "", ""


@dataclasses.dataclass
class NdFilter:
    """Infrared density (ND) filter correction that libdirp applies after measuring:
    T' = T * scale + offset * 10 (sic, also for the float output)."""

    installed: bool
    scale: float
    offset: float


def nd_filter(segs: list[Segment]) -> NdFilter:
    """libdirp's TinyEXIF scan: an APP7 'DJI-IR_ND' block, else a legacy APP4 signature."""
    nd = NdFilter(False, 0.0, 0.0)
    for s in segs:
        p = s.payload
        if s.marker == APP7 and len(p) + 2 == 0x18 and p.startswith(b"DJI-IR_ND\0"):
            _, installed, scale, offset = struct.unpack_from("<HHff", p, 10)
            nd = NdFilter(installed != 0, scale, offset)
        elif s.marker == APP4 and len(p) > 0xA5 and p[0xA1] == 1 and p[0xA3] == 1 and p[0xA5] == 0xC1:
            nd = NdFilter(True, struct.unpack("<f", bytes.fromhex("c1a85840"))[0], -74.0)
    return nd
