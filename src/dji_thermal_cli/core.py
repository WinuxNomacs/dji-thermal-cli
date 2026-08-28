"""Batch discovery, per-file parameter resolution, and the process/list operations.

Does NOT rewrite the original R-JPEG: DJI's R-JPEG embeds its environmental
parameters in a binary APP4 block that is fixed at capture and not writable
by this SDK. `process_file` only ever produces new derivative files.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from PIL import Image

from . import sdk

try:
    import tifffile
    HAVE_TIFFFILE = True
except ImportError:
    HAVE_TIFFFILE = False

THERMAL_SUFFIXES = (".jpg", ".jpeg")


def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def find_thermal_candidates(root: Path):
    """
    Yield candidate R-JPEG files. DJI thermal frames are conventionally
    named with a _T suffix (e.g. DJI_0001_T.JPG), but not every fleet
    follows that -- we filter by suffix as a fast pre-check, then let
    dirp_create_from_rjpeg be the actual source of truth by skipping files
    it rejects. `root` may be a single file or a directory (searched
    recursively).
    """
    if root.is_file():
        if root.suffix.lower() in THERMAL_SUFFIXES:
            yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in THERMAL_SUFFIXES:
            yield path


@dataclasses.dataclass
class MeasurementOverrides:
    """Any field left as None means: keep whatever dirp_get_measurement_params
    reads back as the R-JPEG's own embedded value for that field -- only
    fields the caller actually set on the CLI get overridden.

    humidity is a percent (0-100), not a 0-1 fraction -- see the note on
    sdk.dirp_measurement_params_t. reflection/ambient_temp are Celsius
    (matches dirp_measurement_params_t)."""

    distance: float | None = None
    humidity: float | None = None
    emissivity: float | None = None
    reflection: float | None = None
    ambient_temp: float | None = None

    def apply(self, params: sdk.dirp_measurement_params_t) -> sdk.dirp_measurement_params_t:
        for field in sdk.dirp_measurement_params_t.FIELDS:
            value = getattr(self, field)
            if value is not None:
                setattr(params, field, value)
        return params


@dataclasses.dataclass
class ProcessResult:
    src: Path
    ok: bool
    reason: str | None = None
    temperature_tiff: Path | None = None
    pseudo_color_png: Path | None = None


def process_file(
    dirp: sdk.DirpSDK,
    src: Path,
    dst_root: Path,
    src_root: Path,
    overrides: MeasurementOverrides,
    range_min_c: float | None,
    range_max_c: float | None,
    palette: int | None,
    save_pseudo_color: bool,
    save_temperature: bool,
) -> ProcessResult:
    try:
        rjpeg = dirp.open(src.read_bytes())
    except sdk.DirpError:
        return ProcessResult(src, ok=False, reason="not an R-JPEG / unsupported")

    with rjpeg:
        params = overrides.apply(rjpeg.get_measurement_params())
        rjpeg.set_measurement_params(params)

        rel = src.relative_to(src_root)
        out_dir = dst_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = rel.stem

        result = ProcessResult(src, ok=True)

        if save_temperature:
            temp_arr = rjpeg.measure_temperature_c()
            out_tiff = out_dir / f"{stem}_temp_c.tiff"
            if HAVE_TIFFFILE:
                tifffile.imwrite(out_tiff, temp_arr)
            else:
                Image.fromarray(temp_arr, mode="F").save(out_tiff)
            result.temperature_tiff = out_tiff

        if save_pseudo_color:
            if range_min_c is not None and range_max_c is not None:
                rjpeg.set_color_bar(manual_enable=True, low=range_min_c, high=range_max_c)
            if palette is not None:
                rjpeg.set_pseudo_color(palette)

            rgb = rjpeg.process_pseudo_color()
            out_png = out_dir / f"{stem}_normalized.png"
            Image.fromarray(rgb, mode="RGB").save(out_png)
            result.pseudo_color_png = out_png

        return result


def describe_file(dirp: sdk.DirpSDK, src: Path) -> dict | None:
    """Read-only inspection for `list`: embedded measurement params, their
    valid ranges, resolution, R-JPEG version, and current color-bar/palette
    ISP settings. Returns None if the file isn't an R-JPEG libdirp accepts."""
    try:
        rjpeg = dirp.open(src.read_bytes())
    except sdk.DirpError:
        return None

    with rjpeg:
        width, height = rjpeg.resolution()
        rjpeg_ver, header_ver, curve_ver = rjpeg.rjpeg_version()
        params = rjpeg.get_measurement_params()
        params_range = rjpeg.get_measurement_params_range()
        color_bar = rjpeg.get_color_bar()
        palette = rjpeg.get_pseudo_color()

        return {
            "path": str(src),
            "width": width,
            "height": height,
            "rjpeg_version": {"rjpeg": rjpeg_ver, "header": header_ver, "curve": curve_ver},
            "measurement_params": params.as_dict(),
            "measurement_params_range": params_range.as_dict(),
            "color_bar": {
                "manual_enable": bool(color_bar.manual_enable),
                "low": color_bar.low,
                "high": color_bar.high,
            },
            "palette": {"index": palette, "name": sdk.PSEUDO_COLOR_NAMES[palette] if 0 <= palette < len(sdk.PSEUDO_COLOR_NAMES) else "unknown"},
        }


# --- picklable entry points for isolate.run_isolated ------------------------
#
# Each of these loads its own DirpSDK rather than being handed an
# already-open one: a ctypes CDLL handle isn't picklable, and re-loading
# libdirp fresh in the child also means a crash in one file's call can't
# leave the *library's* internal state corrupted for the next file.

def process_file_isolated(
    dll_dir: str | None,
    src: Path,
    dst_root: Path,
    src_root: Path,
    overrides: MeasurementOverrides,
    range_min_c: float | None,
    range_max_c: float | None,
    palette: int | None,
    save_pseudo_color: bool,
    save_temperature: bool,
) -> dict:
    dirp = sdk.DirpSDK(dll_dir)
    result = process_file(
        dirp, src, dst_root, src_root, overrides,
        range_min_c, range_max_c, palette, save_pseudo_color, save_temperature,
    )
    return dataclasses.asdict(result)


def describe_file_isolated(dll_dir: str | None, src: Path) -> dict | None:
    dirp = sdk.DirpSDK(dll_dir)
    return describe_file(dirp, src)
