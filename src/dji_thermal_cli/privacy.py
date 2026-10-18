"""Location-free outputs (`process --strip-location`): files that can be shared without
revealing where, or with which aircraft and camera, they were taken.

Removed from the metadata the outputs copy from the source:
- EXIF: the GPS IFD (pointer, entries and their data, wiped in place, since the TIFF output
  keeps the source's EXIF block byte for byte), and the Exif IFD's serial numbers, owner
  name and image unique ID;
- XMP: every GPS* property in any namespace; DJI's position, altitude, RTK and
  rangefinder-target fields; every *SerialNumber; DJI's SelfData (free-form).
Capture time is kept. The mosaic is still placed using GPS (it needs it), but written
without its georeference and with positions relative to its own corner (see mosaic.write).
"""

from __future__ import annotations

import re
import struct

from . import tiffout, xmpedit

_TAG_GPS_IFD, _TAG_EXIF_IFD = 34853, 34665
# Exif IFD: ImageUniqueID, CameraOwnerName, BodySerialNumber, LensSerialNumber
_EXIF_IDENTIFIERS = {0xA420, 0xA430, 0xA431, 0xA435}
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}

# XMP properties removed, by local name (any namespace)
XMP_REMOVED = re.compile(r"(?i)^(gps\w*|absolutealtitude|relativealtitude|altitudetype|rtk\w*"
                         r"|lrftarget(lat|lon|alt|absalt)|\w*serialnumber|selfdata)$")


def strip(metadata: tiffout.SourceMetadata) -> tiffout.SourceMetadata:
    """The source metadata without location or identifiers."""
    xmp = xmpedit.remove_properties(metadata.xmp, XMP_REMOVED.match) if metadata.xmp else metadata.xmp
    return tiffout.SourceMetadata(strip_exif(metadata.exif_tiff), xmp)


def strip_exif(exif: bytes | None) -> bytes | None:
    """The EXIF block (a TIFF structure) with the GPS IFD and the identifiers wiped, in place:
    every other offset stays valid."""
    if not (exif and len(exif) >= 8 and exif[:4] in (b"II*\x00", b"MM\x00*")):
        return exif
    bo = "<" if exif[:2] == b"II" else ">"
    blob = bytearray(exif)
    (ifd0,) = struct.unpack_from(bo + "I", blob, 4)
    for _tag, _type, _count, raw in _remove(blob, bo, ifd0, {_TAG_GPS_IFD}):
        _wipe_ifd(blob, bo, struct.unpack(bo + "I", raw)[0])
    exif_ifd = next((struct.unpack(bo + "I", raw)[0] for tag, _t, _c, raw in _entries(blob, bo, ifd0)
                     if tag == _TAG_EXIF_IFD), None)
    if exif_ifd is not None:
        _remove(blob, bo, exif_ifd, _EXIF_IDENTIFIERS)
    return bytes(blob)


def _entries(blob: bytearray, bo: str, offset: int) -> list[tuple[int, int, int, bytes]]:
    if offset < 8 or offset + 2 > len(blob):
        return []
    (n,) = struct.unpack_from(bo + "H", blob, offset)
    if offset + 2 + 12 * n + 4 > len(blob):
        return []
    return [struct.unpack_from(bo + "HHI", blob, offset + 2 + 12 * i) + (bytes(blob[offset + 10 + 12 * i:offset + 14 + 12 * i]),)
            for i in range(n)]


def _wipe_value(blob: bytearray, bo: str, type_: int, count: int, raw: bytes) -> None:
    """Zero an entry's out-of-line value (values of 4 bytes or less live in the entry)."""
    size = _TYPE_SIZE.get(type_, 1) * count
    if size > 4:
        (offset,) = struct.unpack(bo + "I", raw)
        if 8 <= offset and offset + size <= len(blob):
            blob[offset:offset + size] = bytes(size)


def _remove(blob: bytearray, bo: str, offset: int, tags: set[int]) -> list[tuple[int, int, int, bytes]]:
    """Remove the entries with these tags from the IFD at offset (values zeroed, the rest moved
    up, the next-IFD pointer kept); returns the removed entries."""
    entries = _entries(blob, bo, offset)
    removed = [e for e in entries if e[0] in tags]
    if not removed:
        return []
    for _tag, type_, count, raw in removed:
        _wipe_value(blob, bo, type_, count, raw)
    kept = [e for e in entries if e[0] not in tags]
    end = offset + 2 + 12 * len(entries)
    next_ifd = bytes(blob[end:end + 4])
    out = struct.pack(bo + "H", len(kept))
    for tag, type_, count, raw in kept:
        out += struct.pack(bo + "HHI", tag, type_, count) + raw
    out += next_ifd
    blob[offset:end + 4] = out + bytes(end + 4 - offset - len(out))
    return removed


def _wipe_ifd(blob: bytearray, bo: str, offset: int) -> None:
    """Zero a whole IFD: its entries' values and the IFD itself."""
    entries = _entries(blob, bo, offset)
    for _tag, type_, count, raw in entries:
        _wipe_value(blob, bo, type_, count, raw)
    if entries:
        size = 2 + 12 * len(entries) + 4
        blob[offset:offset + size] = bytes(size)
