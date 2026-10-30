"""Read the first band of a TIFF's first image, with its tags: enough for a DSM as ODM, GDAL and
the like write one, without a TIFF library.

Classic TIFF and BigTIFF, either byte order; strips or tiles, chunky or planar; integer or
floating-point samples; no compression, Deflate or LZW, with no predictor, the horizontal
one or the floating-point one (GDAL's PREDICTOR=1/2/3). Other compressions (JPEG, ZSTD,
LERC, ...) are refused with a message saying how to rewrite the file.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_WIDTH, _LENGTH, _BITS, _COMPRESSION = 256, 257, 258, 259
_STRIP_OFFSETS, _SAMPLES, _ROWS_PER_STRIP, _STRIP_BYTES = 273, 277, 278, 279
_PLANAR, _PREDICTOR = 284, 317
_TILE_WIDTH, _TILE_LENGTH, _TILE_OFFSETS, _TILE_BYTES = 322, 323, 324, 325
_SAMPLE_FORMAT = 339

_NONE, _LZW, _DEFLATE, _ADOBE_DEFLATE = 1, 5, 8, 32946
_COMPRESSIONS = {6: "old JPEG", 7: "JPEG", 32773: "PackBits", 34887: "LERC", 34925: "LZMA", 50000: "ZSTD",
                 50001: "WebP", 50002: "JPEG XL"}

# type: (struct code, size); ASCII (2) and UNDEFINED (7) are kept as bytes
_TYPES = {1: ("B", 1), 2: ("s", 1), 3: ("H", 2), 4: ("I", 4), 5: ("II", 8), 6: ("b", 1), 7: ("s", 1),
          8: ("h", 2), 9: ("i", 4), 10: ("ii", 8), 11: ("f", 4), 12: ("d", 8), 16: ("Q", 8), 17: ("q", 8),
          18: ("Q", 8)}


class TiffError(ValueError):
    """The file is not a TIFF this module reads."""


def read_band(path: Path) -> tuple[np.ndarray, dict]:
    """The first image's first band, (rows, cols) in its own dtype, and its tags as {code: value}:
    a tuple of numbers, or a str for ASCII. Raises TiffError, or OSError when it can't be read."""
    data = Path(path).read_bytes()
    tags = _tags(data)
    try:
        width, length = int(tags[_WIDTH][0]), int(tags[_LENGTH][0])
    except KeyError:
        raise TiffError("no image size") from None
    samples = int(tags.get(_SAMPLES, (1,))[0])
    bits = tags.get(_BITS, (1,))
    fmt = int(tags.get(_SAMPLE_FORMAT, (1,))[0])
    if len(set(bits)) != 1 or bits[0] not in (8, 16, 32, 64) or fmt not in (1, 2, 3) or (fmt == 3 and bits[0] < 32):
        raise TiffError(f"samples of {bits[0]} bits in format {fmt} are not supported")
    dtype = np.dtype({1: "u", 2: "i", 3: "f"}[fmt] + str(bits[0] // 8)).newbyteorder(_order(data))
    compression = int(tags.get(_COMPRESSION, (_NONE,))[0])
    if compression not in (_NONE, _LZW, _DEFLATE, _ADOBE_DEFLATE):
        name = _COMPRESSIONS.get(compression, f"compression {compression}")
        raise TiffError(f"{name} compression is not supported: rewrite it with Deflate, e.g. "
                        "gdal_translate -co COMPRESS=DEFLATE in.tif out.tif")
    predictor = int(tags.get(_PREDICTOR, (1,))[0])
    if predictor not in (1, 2, 3) or (predictor == 3 and fmt != 3):
        raise TiffError(f"predictor {predictor} is not supported for these samples")
    planar = int(tags.get(_PLANAR, (1,))[0]) == 2
    per_pixel = 1 if planar else samples
    if _TILE_WIDTH in tags:
        cw, ch = int(tags[_TILE_WIDTH][0]), int(tags[_TILE_LENGTH][0])
        offsets, counts = tags.get(_TILE_OFFSETS), tags.get(_TILE_BYTES)
    else:
        cw, ch = width, min(int(tags.get(_ROWS_PER_STRIP, (length,))[0]), length)
        offsets, counts = tags.get(_STRIP_OFFSETS), tags.get(_STRIP_BYTES)
    if not offsets or not counts or len(offsets) != len(counts):
        raise TiffError("no pixel data")
    across, down = -(-width // cw), -(-length // ch)
    if len(offsets) < across * down:          # planar: band 0's chunks come first
        raise TiffError("fewer chunks than the image needs")
    band = np.empty((length, width), dtype.newbyteorder("="))
    for k in range(across * down):
        r, c = divmod(k, across)
        rows = ch if _TILE_WIDTH in tags else min(ch, length - r * ch)
        raw = data[int(offsets[k]):int(offsets[k]) + int(counts[k])]
        if len(raw) != int(counts[k]):
            raise TiffError("the file is cut short")
        need = rows * cw * per_pixel * dtype.itemsize
        chunk = _decompress(raw, compression, need)
        if len(chunk) < need:
            raise TiffError("a chunk holds fewer pixels than it should")
        values = _unpredict(chunk[:need], predictor, dtype, rows, cw, per_pixel)
        values = values.reshape(rows, cw, per_pixel)[..., 0]
        h, w = min(rows, length - r * ch), min(cw, width - c * cw)
        band[r * ch:r * ch + h, c * cw:c * cw + w] = values[:h, :w]
    return band, tags


def _order(data: bytes) -> str:
    return {b"II": "<", b"MM": ">"}.get(data[:2], "?")


def _tags(data: bytes) -> dict:
    bo = _order(data)
    if bo == "?" or len(data) < 16:
        raise TiffError("not a TIFF")
    version = struct.unpack(bo + "H", data[2:4])[0]
    if version == 42:
        offset, count_fmt, entry, value_size = struct.unpack(bo + "I", data[4:8])[0], "H", "HHI", 4
    elif version == 43:
        offset, count_fmt, entry, value_size = struct.unpack(bo + "Q", data[8:16])[0], "Q", "HHQ", 8
    else:
        raise TiffError("not a TIFF")
    head = struct.calcsize(bo + count_fmt)
    if offset + head > len(data):
        raise TiffError("the file is cut short")
    n = struct.unpack_from(bo + count_fmt, data, offset)[0]
    size = struct.calcsize(bo + entry) + value_size
    tags = {}
    for i in range(n):
        at = offset + head + i * size
        if at + size > len(data):
            raise TiffError("the file is cut short")
        code, type_, count = struct.unpack_from(bo + entry, data, at)
        if type_ not in _TYPES:
            continue
        code_fmt, item = _TYPES[type_]
        length = item * count
        where = at + size - value_size
        if length > value_size:
            where = struct.unpack_from(bo + ("I" if version == 42 else "Q"), data, where)[0]
        raw = data[where:where + length]
        if len(raw) != length:
            raise TiffError("the file is cut short")
        if code_fmt == "s":
            tags[code] = raw.decode("latin-1").rstrip("\x00") if type_ == 2 else raw
        else:
            values = struct.unpack(bo + code_fmt * count, raw)
            if type_ in (5, 10):                  # rationals
                values = tuple(a / b if b else float("nan") for a, b in zip(values[::2], values[1::2]))
            tags[code] = values
    return tags


def _decompress(raw: bytes, compression: int, need: int) -> bytes:
    if compression == _NONE:
        return raw
    if compression == _LZW:
        return _lzw(raw, need)
    try:
        return zlib.decompress(raw)
    except zlib.error as exc:
        raise TiffError(f"a chunk could not be decompressed ({exc})") from None


def _lzw(raw: bytes, need: int) -> bytes:
    """TIFF's LZW (MSB-first codes, 9-12 bits, early change), up to need bytes."""
    out = bytearray()
    table = [bytes([i]) for i in range(256)] + [b"", b""]
    width, prev, bitpos, total = 9, None, 0, len(raw) * 8
    while bitpos + width <= total and len(out) < need:
        byte = bitpos >> 3
        word = int.from_bytes(raw[byte:byte + 3].ljust(3, b"\x00"), "big")
        code = (word >> (24 - width - (bitpos & 7))) & ((1 << width) - 1)
        bitpos += width
        if code == 256:                           # clear
            table, width, prev = table[:258], 9, None
            continue
        if code == 257:                           # end of information
            break
        if prev is None:
            entry = table[code] if code < len(table) else b""
        elif code < len(table):
            entry = table[code]
            table.append(prev + entry[:1])
        elif code == len(table):
            entry = prev + prev[:1]
            table.append(entry)
        else:
            raise TiffError("a chunk's LZW data is corrupt")
        out += entry
        prev = entry
        if len(table) + 1 >= (1 << width) and width < 12:
            width += 1
    return bytes(out)


def _unpredict(chunk: bytes, predictor: int, dtype: np.dtype, rows: int, cols: int, samples: int) -> np.ndarray:
    """The chunk's (rows * cols * samples,) samples in native order, with the predictor undone:
    each sample differenced from the same sample of the pixel before it (2), or for floating
    point (3), each row's bytes differenced likewise, then regrouped from byte planes (most
    significant first) into big-endian samples."""
    native = dtype.newbyteorder("=")
    if predictor == 3:
        size = dtype.itemsize
        planes = np.frombuffer(chunk, np.uint8).reshape(rows, cols * size, samples)
        planes = np.cumsum(planes, axis=1, dtype=np.uint8).reshape(rows, size, cols * samples)
        big = np.ascontiguousarray(planes.transpose(0, 2, 1))
        return big.view(dtype.newbyteorder(">")).astype(native).ravel()
    values = np.frombuffer(chunk, dtype).astype(native)
    if predictor == 2:                            # on the bits, as integers (libtiff does floats so too)
        bits = values.view("u%d" % dtype.itemsize).reshape(rows, cols, samples)
        values = np.cumsum(bits, axis=1, dtype=bits.dtype).view(native).ravel()
    return values
