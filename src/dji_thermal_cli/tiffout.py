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
import zlib
from pathlib import Path

import numpy as np

from .tsdk.jpeg import segments

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
_STRUCTURAL_TAGS = {254, 256, 257, 258, 259, 262, 273, 277, 278, 279, 284, 317, 322, 323, 324, 325, 330, 339, _TAG_XMP}
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
    for seg in segments(data):
        marker, payload = seg.marker, seg.payload
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


_SAMPLE_FORMAT = {"f": 3, "u": 1, "i": 2}     # IEEE float, unsigned, signed
_COMPRESSION = {"none": 1, "deflate": 8}       # 8: Adobe Deflate (zlib)
_STRIP_BYTES = 64 * 1024                       # uncompressed bytes per strip, about


def exif_for(metadata: SourceMetadata | None, width: int, height: int) -> bytes | None:
    """The source EXIF block with its Exif-IFD pixel dimensions set to width x height."""
    exif = metadata.exif_tiff if metadata else None
    if not (exif and len(exif) >= 8 and exif[:4] in (b"II*\x00", b"MM\x00*")):
        return None
    bo = "<" if exif[:2] == b"II" else ">"
    blob = bytearray(exif)
    (ifd0,) = struct.unpack_from(bo + "I", blob, 4)
    _patch_exif_dimensions(blob, bo, _read_ifd(blob, bo, ifd0), width, height)
    return bytes(blob)


def _strips(pixels: np.ndarray, bo: str, compression: str) -> tuple[list[bytes], int, int]:
    """The pixel data as strips, the rows per strip, and the TIFF predictor used.

    Deflate uses horizontal differencing (predictor 2) on 16-bit integers only: on float
    temperatures and 8-bit RGB it makes the files larger, as their values repeat."""
    height = pixels.shape[0]
    row_bytes = pixels.nbytes // height
    rows = max(1, min(height, _STRIP_BYTES // row_bytes)) if compression != "none" else height
    predictor = 2 if compression != "none" and pixels.dtype.itemsize == 2 and pixels.dtype.kind in "ui" else 1
    data = pixels
    if predictor == 2:
        data = pixels.copy()
        data[:, 1:] = np.diff(pixels, axis=1)          # wraps modulo 2^16, as the predictor does
    data = data.astype(pixels.dtype.newbyteorder(bo))
    strips = [data[y:y + rows].tobytes() for y in range(0, height, rows)]
    if compression == "deflate":
        strips = [zlib.compress(strip, 6) for strip in strips]
    return strips, rows, predictor


_TYPE_CODE = {"B": 1, "s": 2, "H": 3, "I": 4, "d": 12}    # BYTE, ASCII, SHORT, LONG, DOUBLE


def tag(fmt: str, values) -> tuple[int, int, bytes]:
    """An extra tag for write_tiff: fmt one of B, s (ASCII; values a str), H, I, d."""
    if fmt == "s":
        raw = values.encode("utf-8") + b"\0"
        return 2, len(raw), raw
    values = list(values)
    return _TYPE_CODE[fmt], len(values), struct.pack(f"<{len(values)}{fmt}", *values)


def write_tiff(path: Path, pixels: np.ndarray, metadata: SourceMetadata | None = None, band_name: bool = True,
               compression: str = "deflate", extra_tags: dict[int, tuple[int, int, bytes]] | None = None) -> None:
    """Write a TIFF carrying the source EXIF and XMP.

    pixels: 2-D float32, uint16, int16 or uint8 (one band), (height, width, 3) uint8 RGB,
    or (height, width, bands) float32.
    band_name: add Camera:BandName="LWIR" to the XMP (single-band thermal products).
    compression: "deflate" (lossless, in strips of about 64 KB) or "none" (one strip).
    extra_tags: {tag: tag(...)} added to IFD0 (little-endian payloads; e.g. GeoTIFF keys).
    """
    if compression not in _COMPRESSION:
        raise ValueError(f"unknown TIFF compression {compression!r}")
    pixels = np.ascontiguousarray(pixels)
    if pixels.ndim == 3 and pixels.shape[2] == 3 and pixels.dtype == np.uint8:
        spp = 3
    elif pixels.ndim == 3 and pixels.dtype == np.float32:
        spp = pixels.shape[2]
    elif pixels.ndim == 2 and pixels.dtype in (np.float32, np.uint16, np.int16, np.uint8):
        spp = 1
    else:
        raise ValueError(f"unsupported TIFF pixels: {pixels.dtype} {pixels.shape}")
    height, width = pixels.shape[:2]
    bits, sample_format = pixels.dtype.itemsize * 8, _SAMPLE_FORMAT[pixels.dtype.kind]

    exif = exif_for(metadata, width, height)
    if exif is not None:
        bo = "<" if exif[:2] == b"II" else ">"
        blob = bytearray(exif)
        (ifd0,) = struct.unpack_from(bo + "I", blob, 4)
        base = _read_ifd(blob, bo, ifd0)
    else:
        bo = "<"
        blob = bytearray(b"II*\x00\x00\x00\x00\x00")
        base = []
    _pad_even(blob)

    body = bytearray(blob)
    strips, rows_per_strip, predictor = _strips(pixels, bo, compression)
    strip_offsets = []
    for strip in strips:
        strip_offsets.append(len(body))
        body += strip
        _pad_even(body)

    xmp = with_band_name(metadata.xmp if metadata else None) if band_name else (metadata.xmp if metadata else None)
    xmp_offset = len(body)
    if xmp:
        body += xmp
        _pad_even(body)

    def shorts(*values: int) -> tuple[int, int, bytes]:
        """A SHORT entry; values that don't fit in the entry are stored in the body."""
        if len(values) <= 2:
            return _SHORT, len(values), struct.pack(bo + f"{len(values)}H", *values).ljust(4, b"\x00")
        offset = len(body)
        body.extend(struct.pack(bo + f"{len(values)}H", *values))
        _pad_even(body)
        return _SHORT, len(values), _entry_value(bo, _LONG, offset)

    def longs(*values: int) -> tuple[int, int, bytes]:
        if len(values) == 1:
            return _LONG, 1, _entry_value(bo, _LONG, values[0])
        offset = len(body)
        body.extend(struct.pack(bo + f"{len(values)}I", *values))
        return _LONG, len(values), _entry_value(bo, _LONG, offset)

    entries = {tag: (type_, count, raw) for tag, type_, count, raw in base if tag not in _STRUCTURAL_TAGS}
    entries.update({
        256: (_LONG, 1, _entry_value(bo, _LONG, width)),
        257: (_LONG, 1, _entry_value(bo, _LONG, height)),
        258: shorts(*[bits] * spp),
        259: shorts(_COMPRESSION[compression]),
        262: shorts(2 if spp == 3 and pixels.dtype == np.uint8 else 1),
        273: longs(*strip_offsets),
        277: shorts(spp),
        278: (_LONG, 1, _entry_value(bo, _LONG, rows_per_strip)),
        279: longs(*map(len, strips)),
        284: shorts(1),
        339: shorts(*[sample_format] * spp),
    })
    if predictor != 1:
        entries[317] = shorts(predictor)
    if spp > 1 and pixels.dtype != np.uint8:
        entries[338] = shorts(*[0] * (spp - 1))          # ExtraSamples: unspecified data bands
    for tag_id, (type_, count, raw) in (extra_tags or {}).items():
        if bo != "<":
            raise ValueError("extra_tags need a little-endian file")
        if len(raw) <= 4:
            entries[tag_id] = (type_, count, raw.ljust(4, b"\0"))
        else:
            _pad_even(body)
            entries[tag_id] = (type_, count, _entry_value(bo, _LONG, len(body)))
            body.extend(raw)
    if xmp:
        entries[_TAG_XMP] = (_BYTE, len(xmp), _entry_value(bo, _LONG, xmp_offset))

    ifd_offset = len(body)
    ifd = bytearray(struct.pack(bo + "H", len(entries)))
    for tag in sorted(entries):
        type_, count, raw = entries[tag]
        ifd += struct.pack(bo + "HHI", tag, type_, count) + raw
    ifd += struct.pack(bo + "I", 0)
    body += ifd
    body[4:8] = struct.pack(bo + "I", ifd_offset)
    Path(path).write_bytes(bytes(body))


def write_temperature_tiff(path: Path, temperature: np.ndarray, metadata: SourceMetadata | None = None,
                           compression: str = "deflate") -> None:
    """Write `temperature` (2-D Celsius array) as a float32 TIFF for WebODM/ODX.

    NaN pixels are written as -273.15, marking them as having no valid temperature.
    """
    pixels = np.nan_to_num(np.asarray(temperature, dtype=np.float32), nan=NO_DATA_C)
    if pixels.ndim != 2:
        raise ValueError("temperature must be a 2-D array")
    write_tiff(path, pixels, metadata, compression=compression)
