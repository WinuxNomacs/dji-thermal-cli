"""Write float32 temperature TIFFs that WebODM/ODX accept as thermal frames.

ODM and ODX georeference each frame from its EXIF GPS block, take orientation and altitude
from DJI XMP attributes, and only treat an image as thermal when its XMP carries
Camera:BandName="LWIR". So each output TIFF carries the source R-JPEG's EXIF (with its GPS
and Exif sub-IFDs) and XMP, plus that band name.

The EXIF block of a JPEG is itself a TIFF structure. The output file starts with that block
unchanged, so every internal offset (Exif IFD, GPS IFD, maker notes) stays valid; the pixel
data, the XMP packet and a new IFD0 are appended after it. The new IFD0 copies the original
tags and adds the image-structure tags.
"""

from __future__ import annotations

import dataclasses
import struct
from pathlib import Path

import numpy as np

from .rjpeg import iter_segments

BAND_NAME = "LWIR"
NO_DATA_C = -273.15

_APP1 = 0xE1
_EXIF_PREFIX = b"Exif\x00\x00"
_XMP_PREFIX = b"http://ns.adobe.com/xap/1.0/\x00"
_CAMERA_NS = "http://pix4d.com/camera/1.0"
_TAG_EXIF_IFD = 34665
_TAG_XMP = 700
_TAG_EXIF_WIDTH, _TAG_EXIF_HEIGHT = 40962, 40963
_SHORT, _LONG, _BYTE = 3, 4, 1
_STRUCTURAL_TAGS = {254, 256, 257, 258, 259, 262, 273, 277, 278, 279, 284, 322, 323, 324, 325, 330, 339, _TAG_XMP}
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}

_EMPTY_XMP = (
    '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    f'<rdf:Description rdf:about="" xmlns:Camera="{_CAMERA_NS}" Camera:BandName="{BAND_NAME}"/>'
    '</rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
)


@dataclasses.dataclass
class SourceMetadata:
    exif_tiff: bytes | None
    xmp: bytes | None


def read_metadata(data: bytes) -> SourceMetadata:
    exif = xmp = None
    for marker, payload in iter_segments(data):
        if marker != _APP1:
            continue
        if exif is None and payload.startswith(_EXIF_PREFIX):
            exif = bytes(payload[len(_EXIF_PREFIX):])
        elif xmp is None and payload.startswith(_XMP_PREFIX):
            xmp = bytes(payload[len(_XMP_PREFIX):])
    return SourceMetadata(exif, xmp)


def with_band_name(xmp: bytes | None) -> bytes:
    """The XMP packet with Camera:BandName="LWIR" added (namespace declared if needed)."""
    if not xmp:
        return _EMPTY_XMP.encode("utf-8")
    try:
        text, encoding = xmp.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        text, encoding = xmp.decode("latin-1"), "latin-1"
    if "Camera:BandName" in text:
        return xmp
    if "<rdf:Description" not in text:
        return _EMPTY_XMP.encode("utf-8")
    declare = "" if "xmlns:Camera=" in text else f' xmlns:Camera="{_CAMERA_NS}"'
    text = text.replace("<rdf:Description", f'<rdf:Description{declare} Camera:BandName="{BAND_NAME}"', 1)
    return text.encode(encoding)


def _entry_value(bo: str, type_: int, value: int) -> bytes:
    if type_ == _SHORT:
        return struct.pack(bo + "H", value) + b"\x00\x00"
    return struct.pack(bo + "I", value)


def _read_ifd(blob: bytes | bytearray, bo: str, offset: int) -> list[tuple[int, int, int, bytes]]:
    if offset < 8 or offset + 2 > len(blob):
        return []
    (n,) = struct.unpack_from(bo + "H", blob, offset)
    entries = []
    for i in range(n):
        pos = offset + 2 + 12 * i
        if pos + 12 > len(blob):
            break
        tag, type_, count = struct.unpack_from(bo + "HHI", blob, pos)
        entries.append((tag, type_, count, bytes(blob[pos + 8:pos + 12])))
    return entries


def _patch_exif_dimensions(blob: bytearray, bo: str, entries, width: int, height: int) -> None:
    """Make the Exif IFD's pixel dimensions describe this TIFF, not the JPEG the EXIF came from."""
    exif_ptr = next((e for e in entries if e[0] == _TAG_EXIF_IFD), None)
    if exif_ptr is None:
        return
    (ifd_offset,) = struct.unpack(bo + "I", exif_ptr[3])
    if ifd_offset < 8 or ifd_offset + 2 > len(blob):
        return
    (n,) = struct.unpack_from(bo + "H", blob, ifd_offset)
    for i in range(n):
        pos = ifd_offset + 2 + 12 * i
        if pos + 12 > len(blob):
            return
        tag, type_, count = struct.unpack_from(bo + "HHI", blob, pos)
        if tag in (_TAG_EXIF_WIDTH, _TAG_EXIF_HEIGHT) and count == 1 and type_ in (_SHORT, _LONG):
            value = width if tag == _TAG_EXIF_WIDTH else height
            if type_ == _SHORT and value > 0xFFFF:
                continue
            blob[pos + 8:pos + 12] = _entry_value(bo, type_, value)


def _pad_even(buf: bytearray) -> None:
    if len(buf) % 2:
        buf.append(0)


def write_temperature_tiff(path: Path, temperature: np.ndarray, metadata: SourceMetadata | None = None) -> None:
    """Write `temperature` (2-D Celsius array) as an uncompressed single-strip float32 TIFF.

    NaN pixels are written as -273.15, marking them as having no valid temperature.
    """
    pixels = np.nan_to_num(np.asarray(temperature, dtype=np.float32), nan=NO_DATA_C)
    if pixels.ndim != 2:
        raise ValueError("temperature must be a 2-D array")
    height, width = pixels.shape

    exif = metadata.exif_tiff if metadata else None
    if exif and len(exif) >= 8 and exif[:4] in (b"II*\x00", b"MM\x00*"):
        bo = "<" if exif[:2] == b"II" else ">"
        blob = bytearray(exif)
        (ifd0,) = struct.unpack_from(bo + "I", blob, 4)
        base = _read_ifd(blob, bo, ifd0)
        _patch_exif_dimensions(blob, bo, base, width, height)
    else:
        bo = "<"
        blob = bytearray(b"II*\x00\x00\x00\x00\x00")
        base = []
    _pad_even(blob)

    body = bytearray(blob)
    pixel_offset = len(body)
    body += pixels.astype(bo + "f4").tobytes()
    _pad_even(body)

    xmp_offset = xmp_len = 0
    xmp = with_band_name(metadata.xmp if metadata else None)
    xmp_offset, xmp_len = len(body), len(xmp)
    body += xmp
    _pad_even(body)

    entries = {tag: (type_, count, raw) for tag, type_, count, raw in base if tag not in _STRUCTURAL_TAGS}
    entries.update({
        256: (_LONG, 1, _entry_value(bo, _LONG, width)),
        257: (_LONG, 1, _entry_value(bo, _LONG, height)),
        258: (_SHORT, 1, _entry_value(bo, _SHORT, 32)),
        259: (_SHORT, 1, _entry_value(bo, _SHORT, 1)),
        262: (_SHORT, 1, _entry_value(bo, _SHORT, 1)),
        273: (_LONG, 1, _entry_value(bo, _LONG, pixel_offset)),
        277: (_SHORT, 1, _entry_value(bo, _SHORT, 1)),
        278: (_LONG, 1, _entry_value(bo, _LONG, height)),
        279: (_LONG, 1, _entry_value(bo, _LONG, pixels.nbytes)),
        284: (_SHORT, 1, _entry_value(bo, _SHORT, 1)),
        339: (_SHORT, 1, _entry_value(bo, _SHORT, 3)),
        _TAG_XMP: (_BYTE, xmp_len, _entry_value(bo, _LONG, xmp_offset)),
    })

    ifd_offset = len(body)
    ifd = bytearray(struct.pack(bo + "H", len(entries)))
    for tag in sorted(entries):
        type_, count, raw = entries[tag]
        ifd += struct.pack(bo + "HHI", tag, type_, count) + raw
    ifd += struct.pack(bo + "I", 0)
    body += ifd
    body[4:8] = struct.pack(bo + "I", ifd_offset)
    Path(path).write_bytes(bytes(body))
