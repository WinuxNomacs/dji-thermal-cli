"""Location-free outputs (`process --strip-location`): files that can be shared without
revealing where, or with which aircraft and camera, they were taken.

Removed from the metadata the outputs copy from the source:
- EXIF: the GPS IFD (pointer, entries and their data, wiped in place, since the TIFF output
  keeps the source's EXIF block byte for byte), the Exif IFD's serial numbers, owner
  name and image unique ID, the free-text fields where a name or an address could be
  (ImageDescription, Artist, Copyright, UserComment, the Windows XP fields), and the
  sensor's serial in DJI's debug records ("[sensor_id:...]");
- XMP: every GPS* property in any namespace; DJI's position, altitude, RTK and
  rangefinder-target fields; every *SerialNumber; DJI's SelfData (free-form); the free-text
  fields (dc:creator, dc:rights, dc:description, dc:title, dc:subject and the EXIF ones).
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
# Free text: IFD0 ImageDescription, Artist, Copyright, XPTitle..XPSubject; Exif IFD UserComment
_IFD0_TEXT = {0x010E, 0x013B, 0x8298, 0x9C9B, 0x9C9C, 0x9C9D, 0x9C9E, 0x9C9F}
_EXIF_TEXT = {0x9286}
# DJI's debug records ("[name:value]", in the EXIF block and in APP7's DJI-DBG): the sensor's serial
DEBUG_SERIAL = re.compile(rb"\[sensor_id:([^\]\x00]*)")
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}

# XMP properties removed, by local name (any namespace)
XMP_REMOVED = re.compile(r"(?i)^(gps\w*|absolutealtitude|relativealtitude|altitudetype|rtk\w*"
                         r"|lrftarget(lat|lon|alt|absalt)|\w*serialnumber|selfdata"
                         r"|creator|rights|description|title|subject|imagedescription|artist|copyright|usercomment)$")


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
    for tag, _type, _count, raw in _remove(blob, bo, ifd0, {_TAG_GPS_IFD} | _IFD0_TEXT):
        if tag == _TAG_GPS_IFD:
            _wipe_ifd(blob, bo, struct.unpack(bo + "I", raw)[0])
    exif_ifd = next((struct.unpack(bo + "I", raw)[0] for tag, _t, _c, raw in _entries(blob, bo, ifd0)
                     if tag == _TAG_EXIF_IFD), None)
    if exif_ifd is not None:
        _remove(blob, bo, exif_ifd, _EXIF_IDENTIFIERS | _EXIF_TEXT)
    wipe_debug_serials(blob)
    return bytes(blob)


def exif_identifiers(exif: bytes | None) -> dict[str, bytes]:
    """The Exif IFD's identifiers (serial numbers, owner name, image unique ID) as their bytes."""
    if not (exif and len(exif) >= 8 and exif[:4] in (b"II*\x00", b"MM\x00*")):
        return {}
    bo = "<" if exif[:2] == b"II" else ">"
    blob = bytearray(exif)
    (ifd0,) = struct.unpack_from(bo + "I", blob, 4)
    exif_ifd = next((struct.unpack(bo + "I", raw)[0] for tag, _t, _c, raw in _entries(blob, bo, ifd0)
                     if tag == _TAG_EXIF_IFD), None)
    out = {}
    for tag, type_, count, raw in _entries(blob, bo, exif_ifd) if exif_ifd is not None else []:
        if tag in _EXIF_IDENTIFIERS:
            size = _TYPE_SIZE.get(type_, 1) * count
            off = struct.unpack(bo + "I", raw)[0] if size > 4 else None
            value = raw[:size] if off is None else bytes(blob[off:off + size])
            out[f"tag {tag:#06x}"] = value.rstrip(b"\0 ")
    return out


def wipe_debug_serials(blob: bytearray) -> None:
    """Zero the value of every "[sensor_id:...]" debug record, in place."""
    for m in DEBUG_SERIAL.finditer(blob):
        blob[m.start(1):m.end(1)] = bytes(m.end(1) - m.start(1))


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
