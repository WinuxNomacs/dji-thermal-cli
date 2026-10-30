"""Write TIFFs in the layouts GDAL can (strips or tiles, chunky or planar, Deflate or LZW, the
horizontal or floating-point predictor, BigTIFF, either byte order), for tiffin's tests:
tifffile writes LZW and the floating-point predictor only with imagecodecs."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

COMPRESSION = {"none": 1, "lzw": 5, "deflate": 8}


def write(path: Path, bands: np.ndarray, *, compression: str = "none", predictor: int = 1,
          tile: int | None = None, rows_per_strip: int | None = None, planar: bool = False,
          bigtiff: bool = False, order: str = "<", extra: list[tuple[int, int, tuple | bytes]] = ()) -> Path:
    """bands: (bands, rows, cols). extra: [(code, type, values)] more tags (type 2: bytes)."""
    bands = np.asarray(bands)
    count, rows, cols = bands.shape
    dtype = bands.dtype.newbyteorder(order)
    if tile:
        cw = ch = tile
    else:
        cw, ch = cols, rows_per_strip or rows
    planes = [bands[b:b + 1] for b in range(count)] if planar else [bands]
    chunks = []
    for plane in planes:
        for r in range(0, rows, ch):
            for c in range(0, cols, cw):
                block = plane[:, r:r + ch, c:c + cw]
                h = ch if tile else block.shape[1]
                padded = np.zeros((plane.shape[0], h, cw), bands.dtype)
                padded[:, :block.shape[1], :block.shape[2]] = block
                pixels = padded.transpose(1, 2, 0)                    # (rows, cols, samples)
                chunks.append(_compress(_predict(pixels, predictor, dtype), compression))
    fmt = {"u": 1, "i": 2, "f": 3}[bands.dtype.kind]
    per_pixel = 1 if planar else count
    tags = [(256, 4, (cols,)), (257, 4, (rows,)), (258, 3, (bands.dtype.itemsize * 8,) * per_pixel),
            (259, 3, (COMPRESSION[compression],)), (262, 3, (1,)), (277, 3, (count,)),
            (284, 3, (2 if planar else 1,)), (317, 3, (predictor,)), (339, 3, (fmt,) * per_pixel)]
    if tile:
        tags += [(322, 3, (cw,)), (323, 3, (ch,)), (324, 16 if bigtiff else 4, None), (325, 4, None)]
    else:
        tags += [(273, 16 if bigtiff else 4, None), (278, 4, (ch,)), (279, 4, None)]
    tags += list(extra)
    return Path(path).write_bytes(_layout(sorted(tags), chunks, order, bigtiff)) and Path(path)


def _predict(pixels: np.ndarray, predictor: int, dtype: np.dtype) -> bytes:
    if predictor == 3:
        size = dtype.itemsize
        rows, cols, samples = pixels.shape
        big = np.ascontiguousarray(pixels.astype(dtype.newbyteorder(">"))).view(np.uint8)
        planes = big.reshape(rows, cols * samples, size).transpose(0, 2, 1).reshape(rows, cols * size, samples)
        diff = planes.copy()
        diff[:, 1:] = planes[:, 1:] - planes[:, :-1]
        return diff.tobytes()
    if predictor == 2:
        bits = pixels.view("u%d" % pixels.dtype.itemsize)
        diff = bits.copy()
        diff[:, 1:] = bits[:, 1:] - bits[:, :-1]
        pixels = diff.view(pixels.dtype)
    return pixels.astype(dtype).tobytes()


def _compress(raw: bytes, compression: str) -> bytes:
    if compression == "deflate":
        return zlib.compress(raw)
    if compression == "lzw":
        return lzw(raw)
    return raw


def lzw(raw: bytes) -> bytes:
    """TIFF's LZW: MSB-first codes, 9-12 bits with early change, a clear code before the table fills."""
    out, acc, nbits = bytearray(), 0, 0

    def emit(code, width):
        nonlocal acc, nbits
        acc, nbits = (acc << width) | code, nbits + width
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)

    table, width = {bytes([i]): i for i in range(256)}, 9
    emit(256, width)
    w = b""
    for byte in raw:
        wc = w + bytes([byte])
        if wc in table:
            w = wc
            continue
        emit(table[w], width)
        table[wc] = len(table) + 2
        if len(table) + 2 >= (1 << width) and width < 12:    # the decoder's table, one code behind
            width += 1
        if len(table) + 2 >= 4094:
            emit(256, width)
            table, width = {bytes([i]): i for i in range(256)}, 9
        w = bytes([byte])
    if w:
        emit(table[w], width)
        if len(table) + 3 >= (1 << width) and width < 12:
            width += 1
    emit(257, width)
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def _layout(tags, chunks, order, bigtiff) -> bytes:
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 11: 4, 12: 8, 16: 8}
    codes = {1: "B", 3: "H", 4: "I", 11: "f", 12: "d", 16: "Q"}
    head = 16 if bigtiff else 8
    entry, slot = (20, 8) if bigtiff else (12, 4)
    ifd_size = (8 if bigtiff else 2) + len(tags) * entry + slot
    data = bytearray()
    offsets = []
    at = head + ifd_size
    for chunk in chunks:
        offsets.append(at + len(data))
        data += chunk
        if len(data) % 2:
            data += b"\0"
    resolved = []
    for code, type_, values in tags:
        if values is None:
            values = tuple(offsets) if code in (273, 324) else tuple(len(c) for c in chunks)
        resolved.append((code, type_, values))
    blob = bytearray()
    entries = []
    for code, type_, values in resolved:
        raw = values if type_ == 2 else struct.pack(order + codes[type_] * len(values), *values)
        count = len(raw) // sizes[type_]
        if len(raw) <= slot:
            entries.append((code, type_, count, raw.ljust(slot, b"\0")))
        else:
            where = at + len(data) + len(blob)
            blob += raw + (b"\0" if len(raw) % 2 else b"")
            entries.append((code, type_, count, struct.pack(order + ("Q" if bigtiff else "I"), where)))
    out = bytearray(order.encode().replace(b"<", b"II").replace(b">", b"MM"))
    if bigtiff:
        out += struct.pack(order + "HHHQ", 43, 8, 0, head)
        out += struct.pack(order + "Q", len(entries))
    else:
        out += struct.pack(order + "HI", 42, head)
        out += struct.pack(order + "H", len(entries))
    for code, type_, count, value in entries:
        out += struct.pack(order + ("HHQ" if bigtiff else "HHI"), code, type_, count) + value
    out += b"\0" * slot
    return bytes(out + data + blob)
