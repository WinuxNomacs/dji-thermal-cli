"""Batch discovery, per-file parameter resolution, and the process/list operations.

Does NOT rewrite the original R-JPEG: DJI's R-JPEG embeds its environmental
parameters in a binary APP4 block that is fixed at capture and not writable
by this tool. `process_file` only ever produces new derivative files.

Two backends compute temperatures. "python" is the pure-Python model in
radiometry.py (no DJI SDK needed). "native" calls the DJI Thermal SDK through
ctypes and is also the only source of the pseudo-color rendering. "auto" uses
the pure-Python model for temperatures where it applies and falls back to the
native SDK otherwise.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path

import numpy as np
from PIL import Image

from . import radiometry, rjpeg, sdk, tiffout

THERMAL_SUFFIXES = (".jpg", ".jpeg")
BACKENDS = ("auto", "python", "native")

# Parameter ranges accepted by the DJI SDK (dirp_get_measurement_params_range).
PARAM_RANGES = {
    "distance": (1.0, 300.0),
    "humidity": (1.0, 100.0),
    "emissivity": (0.1, 1.0),
    "reflection": (-40.0, 100.0),
    "ambient_temp": (-40.0, 80.0),
}


def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def find_thermal_candidates(root: Path):
    """
    Yield candidate R-JPEG files. DJI thermal frames are conventionally
    named with a _T suffix (e.g. DJI_0001_T.JPG), but not every fleet
    follows that -- we filter by suffix as a fast pre-check, then let the
    R-JPEG parser be the actual source of truth by skipping files it
    rejects. `root` may be a single file or a directory (searched
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
    """Any field left as None means: keep the R-JPEG's own embedded value for
    that field -- only fields the caller actually set on the CLI get overridden.

    humidity is a percent (0-100), not a 0-1 fraction. reflection/ambient_temp
    are Celsius."""

    distance: float | None = None
    humidity: float | None = None
    emissivity: float | None = None
    reflection: float | None = None
    ambient_temp: float | None = None

    def apply(self, params):
        for field in rjpeg.MeasurementParams.FIELDS:
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
    temperature_backend: str | None = None
    notes: list[str] = dataclasses.field(default_factory=list)


def _write_temperature(path: Path, arr, source_jpeg: bytes) -> None:
    """Float32 TIFF carrying the source EXIF/GPS/XMP and Camera:BandName=LWIR, as WebODM/ODX expect."""
    tiffout.write_temperature_tiff(path, arr, tiffout.read_metadata(source_jpeg))


def process_file(
    dirp: sdk.DirpSDK | None,
    src: Path,
    dst_root: Path,
    src_root: Path,
    overrides: MeasurementOverrides,
    range_min_c: float | None,
    range_max_c: float | None,
    palette: int | None,
    save_pseudo_color: bool,
    save_temperature: bool,
    backend: str = "auto",
) -> ProcessResult:
    data = src.read_bytes()
    rel = src.relative_to(src_root)
    out_dir = dst_root / rel.parent
    stem = rel.stem
    result = ProcessResult(src, ok=True)

    python_reason = None
    temperature = None
    if backend != "native":
        try:
            parsed = rjpeg.parse(data)
        except rjpeg.RJpegError as e:
            python_reason = str(e)
        else:
            if save_temperature:
                params = overrides.apply(dataclasses.replace(parsed.params))
                try:
                    temperature = radiometry.temperature_c(parsed.raw, params)
                except radiometry.ModelDomainError as e:
                    python_reason = str(e)
                else:
                    out_of_range = int(np.isnan(temperature).sum())
                    if out_of_range == temperature.size:
                        temperature = None
                        python_reason = (
                            "no valid temperatures for these parameters (every pixel falls below the model's supported range)"
                        )
                    else:
                        result.temperature_backend = "python"
                        if out_of_range:
                            result.notes.append(f"{out_of_range} pixels are below the model's range (about -45 C) and written as -273.15 (no valid temperature)")

    needs_native_temperature = save_temperature and temperature is None
    if backend == "python":
        if needs_native_temperature:
            return ProcessResult(src, ok=False, reason=python_reason or "not supported by the pure-Python backend")
        if save_pseudo_color:
            result.notes.append("pseudo-color PNG skipped: it needs the native SDK (--backend auto or native)")
            save_pseudo_color = False

    rjpeg_handle = None
    if backend != "python" and (needs_native_temperature or save_pseudo_color):
        if dirp is None:
            if needs_native_temperature:
                why = f"{python_reason}; " if python_reason else ""
                return ProcessResult(src, ok=False, reason=f"{why}native DJI SDK not available")
            result.notes.append("pseudo-color PNG skipped: the native DJI SDK is not available")
            save_pseudo_color = False
        else:
            try:
                rjpeg_handle = dirp.open(data)
            except sdk.UnsupportedSdkVersion as e:
                if needs_native_temperature:
                    return ProcessResult(src, ok=False, reason=str(e))
                result.notes.append(f"pseudo-color PNG skipped: {e}")
                save_pseudo_color = False
            except sdk.DirpError:
                return ProcessResult(src, ok=False, reason=python_reason or "not an R-JPEG / unsupported")

    if temperature is None and rjpeg_handle is None and not save_pseudo_color:
        return ProcessResult(src, ok=False, reason="nothing to do")

    out_dir.mkdir(parents=True, exist_ok=True)
    if temperature is not None:
        out_tiff = out_dir / f"{stem}_temp_c.tiff"
        _write_temperature(out_tiff, temperature, data)
        result.temperature_tiff = out_tiff

    if rjpeg_handle is not None:
        with rjpeg_handle:
            params = overrides.apply(rjpeg_handle.get_measurement_params())
            rjpeg_handle.set_measurement_params(params)

            if needs_native_temperature:
                out_tiff = out_dir / f"{stem}_temp_c.tiff"
                _write_temperature(out_tiff, rjpeg_handle.measure_temperature_c(), data)
                result.temperature_tiff = out_tiff
                result.temperature_backend = "native"

            if save_pseudo_color:
                if range_min_c is not None and range_max_c is not None:
                    rjpeg_handle.set_color_bar(manual_enable=True, low=range_min_c, high=range_max_c)
                if palette is not None:
                    rjpeg_handle.set_pseudo_color(palette)
                rgb = rjpeg_handle.process_pseudo_color()
                out_png = out_dir / f"{stem}_normalized.png"
                Image.fromarray(rgb, mode="RGB").save(out_png)
                result.pseudo_color_png = out_png

    return result


def _describe_python(src: Path, data: bytes) -> dict | None:
    try:
        parsed = rjpeg.parse(data)
    except rjpeg.RJpegError:
        return None
    return {
        "path": str(src),
        "width": parsed.width,
        "height": parsed.height,
        "rjpeg_version": None,
        "measurement_params": parsed.params.as_dict(),
        "measurement_params_range": {k: list(v) for k, v in PARAM_RANGES.items()},
        "color_bar": None,
        "palette": None,
    }


def _describe_native(dirp: sdk.DirpSDK, src: Path, data: bytes) -> dict | None:
    try:
        handle = dirp.open(data)
    except sdk.DirpError:
        return None

    with handle:
        width, height = handle.resolution()
        rjpeg_ver, header_ver, curve_ver = handle.rjpeg_version()
        params = handle.get_measurement_params()
        params_range = handle.get_measurement_params_range()
        color_bar = handle.get_color_bar()
        palette = handle.get_pseudo_color()

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


def describe_file(dirp: sdk.DirpSDK | None, src: Path, backend: str = "auto") -> dict | None:
    """Read-only inspection for `list`: embedded measurement params, resolution and valid
    ranges (plus R-JPEG version and color-bar/palette settings when the native SDK is used).
    Returns None if the file isn't an R-JPEG the chosen backend accepts."""
    data = src.read_bytes()
    if backend == "native" or (backend == "auto" and dirp is not None):
        if dirp is None:
            return None
        try:
            return _describe_native(dirp, src, data)
        except sdk.UnsupportedSdkVersion:
            if backend == "native":
                raise
    return _describe_python(src, data)


# --- picklable entry points for isolate.WorkerPool / run_isolated ------------
#
# Each of these loads its own DirpSDK rather than being handed an
# already-open one: a ctypes CDLL handle isn't picklable. The load is cached
# per process, so a long-lived pool worker pays for it once; a worker that
# crashes is replaced by a fresh process (and a fresh library load).

@functools.lru_cache(maxsize=None)
def _sdk_for(dll_dir: str | None) -> sdk.DirpSDK:
    return sdk.DirpSDK(dll_dir)


def optional_sdk(dll_dir: str | None, backend: str = "auto") -> sdk.DirpSDK | None:
    """The native SDK, or None when it isn't installed (or the pure-Python backend is forced)."""
    if backend == "python":
        return None
    try:
        return _sdk_for(dll_dir)
    except FileNotFoundError:
        if backend == "native":
            raise
        return None


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
    backend: str = "auto",
) -> dict:
    result = process_file(
        optional_sdk(dll_dir, backend), src, dst_root, src_root, overrides,
        range_min_c, range_max_c, palette, save_pseudo_color, save_temperature, backend,
    )
    return dataclasses.asdict(result)


def describe_file_isolated(dll_dir: str | None, src: Path, backend: str = "auto") -> dict | None:
    return describe_file(optional_sdk(dll_dir, backend), src, backend)
