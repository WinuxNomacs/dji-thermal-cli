"""Everything about one dataset file, grouped, for `list` (and later `check`).

describe() reads a thermal R-JPEG without changing anything and returns:

- file:        path, name, size, what the content is, warnings
- camera:      make, model, serials, resolution, focal length (mm, 35 mm, pixels), capture time
- thermal:     SDK plugin and R-JPEG version, the embedded measurement parameters and their
               valid ranges, palette, colour bar, ND filter
- temperature: min / max / mean / median / std of the SDK's measurement with the file's own
               parameters (optional: it measures every pixel)
- geospatial:  position, altitudes, GPS / RTK status and accuracy, laser rangefinder
- attitude:    gimbal and aircraft yaw / pitch / roll
- exif:        optional, every EXIF tag and XMP property, raw

Temperatures are in the requested unit (converted from the SDK's Celsius); a group's fields
are None where the file doesn't record them.
"""

from __future__ import annotations

import io
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ExifTags import GPSTAGS, IFD, TAGS

from . import core, dataset, djixmp, mosaic, tiffout, tsdk, units

GROUPS = ("file", "camera", "thermal", "temperature", "geospatial", "attitude", "exif")


def describe(path: Path, unit: str = "C", stats: bool = True, exif_all: bool = False,
             decode: bool = False) -> tuple[dict | None, dataset.FileStatus | None]:
    """(groups, None) for a thermal R-JPEG, or (None, its FileStatus) for any other file."""
    status = dataset.classify(path, decode=decode)
    if status.kind != dataset.THERMAL:
        return None, status
    data = path.read_bytes()
    try:
        rj = tsdk.open(data)
        info = {
            "file": {"path": str(path), "name": path.name, "size_bytes": len(data), "kind": status.kind,
                     "warnings": status.warnings},
            "camera": None,
            "thermal": _thermal(rj, unit),
            "temperature": _stats(rj, unit) if stats else None,
        }
    except tsdk.TsdkError as e:
        return None, dataset.FileStatus(path, dataset.CORRUPT, f"{type(e).__name__}: {e}", rj.plugin if "rj" in locals() else "")
    exif = Image.open(io.BytesIO(data)).getexif()
    xmp = djixmp.fields(data)
    info["camera"] = _camera(exif, xmp, rj)
    info["geospatial"] = _geospatial(exif, xmp)
    info["attitude"] = _attitude(xmp)
    info["exif"] = _all_tags(exif, data) if exif_all else None
    return info, None


def _thermal(rj, unit: str) -> dict:
    mp = {k: _f32(v) for k, v in rj.get_measurement_params().as_dict().items()}
    ranges = {k: [_f32(x) for x in v] for k, v in rj.get_measurement_params_range().items()}
    for key in ("reflection", "ambient_temp"):
        if ranges[key] == [0.0, 0.0]:          # the camera has no such parameter (dirp, girp: ambient)
            mp[key], ranges[key] = None, None
            continue
        mp[key] = _temp(mp[key], unit)
        ranges[key] = [_temp(v, unit) for v in ranges[key]]
    try:
        cb, index = rj.get_color_bar(), rj.get_pseudo_color()
        color_bar = {"manual": bool(cb.manual_enable), "low": _temp(cb.low, unit), "high": _temp(cb.high, unit)}
        names = core.PSEUDO_COLOR_NAMES
        palette = {"index": index, "name": names[index] if 0 <= index < len(names) else "unknown"}
    except tsdk.TsdkError:
        color_bar = palette = None
    return {
        "plugin": rj.plugin,
        "rjpeg_version": dict(zip(("rjpeg", "header", "curve"), rj.rjpeg_version())),
        "measurement_params": mp,
        "measurement_params_range": ranges,
        "palette": palette,
        "color_bar": color_bar,
        "nd_filter": {"installed": rj.nd.installed, "scale": rj.nd.scale, "offset": rj.nd.offset},
    }


def _stats(rj, unit: str) -> dict:
    t = rj.measure_temperature_c().astype(np.float64)
    valid = t[np.isfinite(t) & (t > float(np.float32(units.ABSOLUTE_ZERO_C)))]
    if not valid.size:
        return {"min": None, "max": None, "mean": None, "median": None, "std": None, "valid_pixels": 0}
    v = units.from_celsius(valid, unit)
    return {"min": _f32(v.min()), "max": _f32(v.max()), "mean": _f32(v.mean()), "median": _f32(np.median(v)),
            "std": _f32(v.std()), "valid_pixels": int(valid.size)}


def _camera(exif, xmp: dict, rj) -> dict:
    sub = exif.get_ifd(IFD.Exif)
    model = str(exif.get(272, "")).strip("\0 ")
    focal, focal35 = _num(sub.get(37386)), _num(sub.get(41989))
    focal_px, focal_source = mosaic.focal_length_px(model, rj.width, rj.height, focal, focal35)
    return {
        "make": _str(exif.get(271)), "model": model or None, "software": _str(exif.get(305)),
        "camera_serial": xmp.get("CameraSerialNumber") or _str(sub.get(42033)),
        "drone_model": xmp.get("DroneModel"), "drone_serial": xmp.get("DroneSerialNumber"),
        "width": rj.width, "height": rj.height,
        "focal_length_mm": focal, "focal_length_35mm": focal35,
        "focal_px": None if focal_px is None else round(focal_px, 2), "focal_source": focal_source or None,
        "capture_time": _str(sub.get(36867)) or _str(exif.get(306)),     # DateTimeOriginal, else DateTime
        "utc_at_exposure": xmp.get("UTCAtExposure"),
    }


def _geospatial(exif, xmp: dict) -> dict:
    lat, lon = djixmp.number(xmp, "GpsLatitude"), djixmp.number(xmp, "GpsLongitude")
    alt = djixmp.number(xmp, "AbsoluteAltitude")
    gps = exif.get_ifd(IFD.GPSInfo)
    if lat is None and gps.get(2):                 # no DJI XMP position: the EXIF GPS block
        lat, lon = _dms(gps.get(2), gps.get(1)), _dms(gps.get(4), gps.get(3))
        alt = _num(gps.get(6))
    lrf = None
    if "LRFStatus" in xmp or "LRFTargetDistance" in xmp:
        lrf = {"status": xmp.get("LRFStatus"), "distance_m": djixmp.number(xmp, "LRFTargetDistance"),
               "target_lat": djixmp.number(xmp, "LRFTargetLat"), "target_lon": djixmp.number(xmp, "LRFTargetLon"),
               "target_abs_alt_m": djixmp.number(xmp, "LRFTargetAbsAlt")}
    return {
        "latitude": lat, "longitude": lon, "absolute_altitude_m": alt,
        "relative_altitude_m": djixmp.number(xmp, "RelativeAltitude"),
        "altitude_type": xmp.get("AltitudeType"), "gps_status": xmp.get("GpsStatus") or _str(gps.get(9)),
        "rtk_flag": xmp.get("RtkFlag"),
        "rtk_std_m": {k: djixmp.number(xmp, f"RtkStd{k.capitalize()}") for k in ("lat", "lon", "hgt")}
        if "RtkStdLat" in xmp else None,
        "rangefinder": lrf,
    }


def _attitude(xmp: dict) -> dict:
    return {f"{who}_{axis}": djixmp.number(xmp, f"{key}{axis.capitalize()}Degree")
            for who, key in (("gimbal", "Gimbal"), ("flight", "Flight")) for axis in ("yaw", "pitch", "roll")}


# --- every tag ------------------------------------------------------------------------------

_XMP_PROPERTY = re.compile(r'([\w.-]+):([\w.-]+)(?:\s*=\s*"([^"]*)"|>([^<]*)</\1:\2>)')


def _all_tags(exif, data: bytes) -> dict:
    """Every EXIF tag (IFD0, Exif, GPS, Interop) by name, and every simple XMP property as
    prefix:name. MakerNote and other binary values are given as hex."""
    out: dict[str, dict] = {"ifd0": {}, "exif": {}, "gps": {}, "interop": {}, "xmp": {}}
    for tag, value in exif.items():
        if tag not in (IFD.Exif, IFD.GPSInfo, IFD.Interop):
            out["ifd0"][TAGS.get(tag, f"0x{tag:04x}")] = _plain(value)
    for group, ifd, names in (("exif", IFD.Exif, TAGS), ("gps", IFD.GPSInfo, GPSTAGS), ("interop", IFD.Interop, TAGS)):
        for tag, value in exif.get_ifd(ifd).items():
            out[group][names.get(tag, f"0x{tag:04x}")] = _plain(value)
    xmp = (tiffout.read_metadata(data).xmp or b"").decode("utf-8", "replace")
    for m in _XMP_PROPERTY.finditer(xmp):
        prefix, name = m.group(1), m.group(2)
        if prefix in ("xmlns", "rdf", "x"):
            continue
        out["xmp"].setdefault(f"{prefix}:{name}", (m.group(3) if m.group(3) is not None else m.group(4)).strip())
    return out


def _plain(value):
    """An EXIF value as JSON-friendly data."""
    if isinstance(value, bytes):
        text = value.rstrip(b"\0")
        if text and all(32 <= b < 127 for b in text):
            return text.decode("ascii")
        return value.hex()
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, str):
        return value.strip("\0 ")
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return None
    return int(f) if isinstance(value, int) else f


def _f32(value: float) -> float:
    """A value at float32 precision, written in its shortest form (0.91, not
    0.9100000262260437). For the SDK's own float32 values this is exactly the value it holds."""
    return float(str(np.float32(value)))


def _temp(celsius: float, unit: str) -> float:
    """An SDK temperature in `unit`, at the float32 precision the SDK works in."""
    return _f32(units.from_celsius(celsius, unit))


def _num(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _str(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip("\0 ")
    return s or None


def _dms(value, ref) -> float | None:
    try:
        d, m, s = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    deg = d + m / 60 + s / 3600
    return -deg if str(ref).upper() in ("S", "W") else deg


# --- flat views (table, CSV) ------------------------------------------------------------------

def flatten(info: dict, prefix: str = "") -> dict:
    """Nested groups as one level of dotted keys (lists joined with ';')."""
    out = {}
    for key, value in info.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        elif isinstance(value, list):
            out[name] = ";".join(str(v) for v in value)
        else:
            out[name] = value
    return out
