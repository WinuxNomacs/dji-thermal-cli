"""R-JPEG copies (the `rjpeg` product, and dji_thermal_cli.rjpeg_copy): a DJI radiometric JPEG
written again, so that DJI's own tools still measure it, with a dataset's measurement parameters
where DJI's SDK reads them, and without the location or identifiers.

Every edit is made in place: a copy has its source's length and layout, since the SDK reads some
tables past their segment's end and the MPF index points into the file by offset. Values are
zeroed, and an XMP packet keeps its length with padding.

Where each camera family keeps its parameters (as the SDK port reads them):
- iirp: five floats at APP4's start (ambient, distance, emissivity, humidity 0-1, reflection);
  before the v0300 header the ambient temperature is the reflection's;
- girp: APP4's AA 55 line (emissivity %, humidity %, distance in 0.1 m, reflection in 0.1 C);
- hirp: APP4's TempParams (Q13 and Q16 values; flags say the reflection and ambient are its own);
- dirp (Zenmuse H20T, XT S) reads none: DJI's SDK starts it from fixed values, so its R-JPEGs
  cannot carry parameters.

A new picture replaces what follows the APP segments (the visible JPEG's tables and scan) and the
MPF preview image after it, and the MPF index is rewritten; the APP segments, where the SDK reads,
stay as they are. (iirp reads its tau table up to 16 bytes past APP5, into the next APP segment.)
The EXIF thumbnail is DJI's still.

A private copy has privacy's removals (see privacy) made in its EXIF and XMP, and every other
occurrence of the sensor's and the thermal module's serials zeroed: in DJI's debug and padding
blocks, and in the hirp calibration block (which the SDK measures the same without it).
"""

from __future__ import annotations

import io
import re
import struct

import numpy as np
from PIL import Image

from . import privacy, xmpedit
from .tsdk import jpeg, open as open_rjpeg
from .tsdk._cnum import F, cvtt, roundf
from .tsdk.base import MeasurementParams
from .tsdk.errors import ParamError

_EXIF = b"Exif\0\0"
_XMP = b"http://ns.adobe.com/xap/1.0/\0"
# The segments the SDK measures from: left alone but for the hirp module serial
_MEASURED = (jpeg.APP3, jpeg.APP4, jpeg.APP5)
_HIDR_MODULE = 4                    # the HIDR block naming the thermal module (HM-TM...)
_MODULE_SERIAL = rb"([0-9A-Z]{8,})"  # what follows the module's name in the calibration block
_MPF = b"MPF\0"
PICTURE_QUALITY = 92

NO_PARAMS = "this camera's R-JPEGs cannot carry measurement parameters (DJI's SDK reads none from them)"


class CopyError(ValueError):
    """An R-JPEG copy that can't be made: the message says why."""


def copy(data: bytes, params: MeasurementParams | None = None, private: bool = False,
         picture: np.ndarray | None = None) -> bytes:
    """The R-JPEG with params written where DJI's SDK reads them (None: its own), without the
    location and identifiers when private, and with picture (uint8 RGB, any size: scaled to
    DJI's) as its visible image (None: DJI's). Raises CopyError, or TsdkError for a file that
    isn't an R-JPEG the port reads."""
    out = bytearray(data)
    if params is not None:
        write_params(out, params)
    if private:
        make_private(out)
    return bytes(out) if picture is None else with_picture(bytes(out), picture)


def with_picture(data: bytes, rgb: np.ndarray) -> bytes:
    """The R-JPEG with rgb as its visible image and its MPF preview, each at DJI's size."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise CopyError("the picture must be uint8 RGB (height, width, 3)")
    segs = jpeg.segments(data)
    first = next((s for s in segs if not 0xE0 <= s.marker <= 0xEF), None)
    if first is None:
        raise CopyError("the R-JPEG has no image after its APP segments")
    head = data[:first.offset - 4]
    mpf = next((s for s in segs if s.marker == jpeg.APP2 and s.payload.startswith(_MPF)), None)
    index = _mpf_index(data, mpf) if mpf else None
    main_end = index[1][2] if index and len(index) > 1 else len(data)
    main = head + _encode(rgb, _size(data[:main_end]))[2:]
    main = main[:len(head)] + _without_app_segments(main[len(head):])
    if not index or len(index) < 2:
        return main
    _, size2, start2 = index[1]
    preview = _encode(rgb, _size(data[start2:start2 + size2]))
    out = bytearray(main + preview + data[start2 + size2:])
    _set_mpf_index(out, mpf, len(main), len(preview))
    return bytes(out)


def _encode(rgb: np.ndarray, size: tuple[int, int]) -> bytes:
    image = Image.fromarray(rgb)                     # uint8 (height, width, 3): RGB
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=PICTURE_QUALITY)
    return buf.getvalue()


def _without_app_segments(body: bytes) -> bytes:
    """A JPEG's segments after its SOI without the APP ones (Pillow's JFIF header): the
    R-JPEG's own APP segments come before them."""
    i = 0
    while body[i:i + 1] == b"\xff" and 0xE0 <= body[i + 1] <= 0xEF:
        i += 2 + struct.unpack_from(">H", body, i + 2)[0]
    return body[i:]


def _size(jpeg_data: bytes) -> tuple[int, int]:
    """(width, height) from a JPEG's frame header."""
    sof = next((s for s in jpeg.segments(jpeg_data) if s.marker in (0xC0, 0xC1, 0xC2)), None)
    if sof is None:
        raise CopyError("the R-JPEG's picture has no frame header")
    height, width = struct.unpack_from(">HH", sof.payload, 1)
    return width, height


def _mpf_index(data: bytes, mpf: jpeg.Segment) -> list[tuple[int, int, int]]:
    """The MPF entries as (attribute, size, file offset); the first image starts the file."""
    base, t = mpf.offset + 4, mpf.payload[4:]
    bo = "<" if t[:2] == b"II" else ">"
    (ifd,) = struct.unpack_from(bo + "I", t, 4)
    (n,) = struct.unpack_from(bo + "H", t, ifd)
    for i in range(n):
        tag, _type, count, value = struct.unpack_from(bo + "HHII", t, ifd + 2 + 12 * i)
        if tag == 0xB002:
            entries = [struct.unpack_from(bo + "III", t, value + 16 * k) for k in range(count // 16)]
            return [(attr, size, base + off if off else 0) for attr, size, off in entries]
    raise CopyError("the R-JPEG's MPF index has no image list")


def _set_mpf_index(data: bytearray, mpf: jpeg.Segment, main_size: int, preview_size: int) -> None:
    """Rewrite the first two MPF entries' sizes and the second's offset, in place."""
    base, t = mpf.offset + 4, mpf.payload[4:]
    bo = "<" if t[:2] == b"II" else ">"
    (ifd,) = struct.unpack_from(bo + "I", t, 4)
    (n,) = struct.unpack_from(bo + "H", t, ifd)
    value = next(struct.unpack_from(bo + "I", t, ifd + 2 + 12 * i + 8)[0] for i in range(n)
                 if struct.unpack_from(bo + "H", t, ifd + 2 + 12 * i)[0] == 0xB002)
    struct.pack_into(bo + "I", data, base + value + 4, main_size)
    struct.pack_into(bo + "II", data, base + value + 16 + 4, preview_size, main_size - base)


def write_params(data: bytearray, p: MeasurementParams) -> None:
    """Write the measurement parameters into the R-JPEG, in place, encoded as the SDK sets them."""
    rj = open_rjpeg(bytes(data))
    if rj.plugin == "dirp":
        raise CopyError(NO_PARAMS)
    try:
        rj.get_measurement_params()
        rj.set_measurement_params(p)             # the camera's own range checks
    except ParamError as e:
        raise CopyError(str(e)) from e
    app4 = jpeg.first_segment(jpeg.segments(bytes(data)), jpeg.APP4).offset
    if rj.plugin == "iirp":
        ambient = p.ambient_temp if rj._v0300 else p.reflection
        struct.pack_into("<5f", data, app4 + (0x20 if rj.magic else 0), float(F(ambient)), float(F(p.distance)),
                         float(F(p.emissivity)), float(F(float(F(p.humidity)) / 100.0)), float(F(p.reflection)))
    elif rj.plugin == "girp":
        line = app4 + 32
        struct.pack_into("<HHHH", data, line + 4, cvtt(float(F(p.humidity))) & 0xFFFF,
                         (cvtt(float(F(p.distance))) * 10) & 0xFFFF,
                         cvtt(float(F(F(p.emissivity) * F(100.0)))) & 0xFFFF,
                         cvtt(float(F(F(p.reflection) * F(10.0)))) & 0xFFFF)
    elif rj.plugin == "hirp":
        tp = app4 + 0x20 + min(struct.unpack_from("<I", data, app4 + 0x24)[0], 0x28)
        struct.pack_into("<H", data, tp + 0x10, (cvtt(float(F(F(p.emissivity) * F(65536.0)))) >> 3) & 0xFFFF)
        struct.pack_into("<ii", data, tp + 0x14, cvtt(float(F(F(p.distance) * F(8192.0)))),
                         cvtt(float(F(F(p.reflection) * F(8192.0)))))
        data[tp + 0x1C] = 1
        struct.pack_into("<i", data, tp + 0x2C, (cvtt(float(roundf(F(p.humidity)))) & 0xFF) * 1000)
        data[tp + 0x3B] = 1
        struct.pack_into("<i", data, tp + 0x3C, cvtt(float(F(F(p.ambient_temp) * F(8192.0)))))
    else:
        raise CopyError(f"writing parameters into {rj.plugin} R-JPEGs is not implemented")


def make_private(data: bytearray) -> None:
    """Remove the location and identifiers from the R-JPEG, in place (see privacy)."""
    segs = jpeg.segments(bytes(data))
    serials = [v for v in identifiers(bytes(data), segs).values()]
    for s in segs:
        start, end = s.offset, s.offset + len(s.payload)
        if s.marker == jpeg.APP1 and s.payload.startswith(_EXIF):
            data[start + len(_EXIF):end] = privacy.strip_exif(s.payload[len(_EXIF):])
        elif s.marker == jpeg.APP1 and s.payload.startswith(_XMP):
            packet = s.payload[len(_XMP):]
            data[start + len(_XMP):end] = _padded(xmpedit.remove_properties(packet, privacy.XMP_REMOVED.match),
                                                  len(packet))
        if s.marker not in _MEASURED:
            block = data[start:end]
            privacy.wipe_debug_serials(block)
            for value in serials:
                block = block.replace(value, bytes(len(value)))
            data[start:end] = block
    for at, size in _module_serials(segs):
        data[at:at + size] = bytes(size)


def identifiers(data: bytes, segs: list[jpeg.Segment] | None = None) -> dict[str, bytes]:
    """The serial numbers in the R-JPEG, as they appear in it: the sensor's (debug records), the
    body's and lens's (EXIF), XMP's *SerialNumber values and the thermal module's."""
    segs = jpeg.segments(data) if segs is None else segs
    found = {}
    for i, m in enumerate(privacy.DEBUG_SERIAL.finditer(data)):
        if m.group(1).strip(b"\0"):
            found[f"sensor_id {i}"] = m.group(1).rstrip(b"\0")
    for s in segs:
        if s.marker == jpeg.APP1 and s.payload.startswith(_XMP):
            text = s.payload[len(_XMP):]
            for name, value in re.findall(rb'(\w*SerialNumber)(?:="|>)([^"<]+)', text):
                found["xmp " + name.decode()] = value
        elif s.marker == jpeg.APP1 and s.payload.startswith(_EXIF):
            for name, value in privacy.exif_identifiers(s.payload[len(_EXIF):]).items():
                found["exif " + name] = value
    for i, (at, size) in enumerate(_module_serials(segs)):
        found[f"module {i}"] = data[at:at + size]
    return {k: v for k, v in found.items() if len(v.strip(b"\0 0")) >= 4}


def _module_serials(segs: list[jpeg.Segment]) -> list[tuple[int, int]]:
    """(file offset, size) of each thermal module serial in a hirp R-JPEG's HIDR blocks: what
    follows the module's name (HIDR block 4) where it appears again."""
    out = []
    for s in segs:
        p = s.payload
        if s.marker != jpeg.APP5 or p[:4] != b"HIDR" or len(p) < 12 + 9 * 12:
            continue
        for i in range(9):
            kind, size, off = struct.unpack_from("<III", p, 12 + 12 * i)
            name = re.search(rb"HM-[\x21-\x7e]{4,}", p[off:off + size]) if kind == _HIDR_MODULE else None
            if name:
                out += [(s.offset + m.start(1), m.end(1) - m.start(1))
                        for m in re.finditer(re.escape(name.group()) + _MODULE_SERIAL, p)]
    return out


def _padded(packet: bytes, size: int) -> bytes:
    """The XMP packet at its former size: padded with spaces before its closing processing
    instruction (XMP's own padding), or at its end."""
    if len(packet) > size:
        raise CopyError("the XMP packet grew")
    pad = b" " * (size - len(packet))
    end = packet.rfind(b"<?xpacket end")
    return packet[:end] + pad + packet[end:] if end >= 0 else packet + pad
