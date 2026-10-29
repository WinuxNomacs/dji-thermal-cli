"""DJI's own Thermal SDK (libdirp) in place of the pure-Python port, where a user has it.

    rj = native.open(data, "/opt/dji_thermal_sdk")     # an unpacked DJI Thermal SDK
    rj.measure_temperature_c()

open() returns an object with the port's RJpeg interface (tsdk.base.RJpeg), so the rest of the
tool measures and renders with either. DJI's library runs only in a child process (client):
a crash or a hang there raises NativeCrash for that call and the caller goes on; the next call
starts a new child. Only the SDK versions the bindings follow are used (bindings.SDK_VERSION):
another one raises NativeSdkError.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .. import tsdk
from ..tsdk import jpeg
from . import bindings
from .client import NativeCrash, NativeError, NativeProcess, NativeSdkError, process_for

__all__ = ["open", "check", "find_library", "NativeRJpeg", "NativeError", "NativeSdkError", "NativeCrash",
           "NativeProcess", "SDK_VERSION", "SUPPORTED_API_VERSIONS"]

SDK_VERSION = bindings.SDK_VERSION
SUPPORTED_API_VERSIONS = bindings.SUPPORTED_API_VERSIONS

_MEASURE = {"dirp_measure", "dirp_measure_ex"}
_ISP = {"dirp_process", "dirp_process_strech", "dirp_set_color_bar", "dirp_get_color_bar",
        "dirp_get_color_bar_adaptive_params", "dirp_set_pseudo_color", "dirp_get_pseudo_color",
        "dirp_get_pseudo_color_lut", "dirp_set_isotherm", "dirp_get_isotherm", "dirp_set_enhancement_params",
        "dirp_get_enhancement_params"}


def find_library(sdk_dir: str | os.PathLike) -> Path:
    """libdirp in an unpacked DJI Thermal SDK (see bindings.find_library)."""
    return bindings.find_library(sdk_dir)


def _supported_text() -> str:
    revisions = ", ".join(str(v) for v in sorted(SUPPORTED_API_VERSIONS))
    return f"dji-thermal-cli supports DJI Thermal SDK v{SDK_VERSION} (API revisions {revisions})"


def open(data: bytes, sdk_dir: str | os.PathLike, process: NativeProcess | None = None) -> NativeRJpeg:  # noqa: A001
    """Open an R-JPEG with DJI's SDK in sdk_dir. Raises the port's errors where DJI's library
    refuses the file (RJpegError, UnsupportedRJpeg), NativeSdkError if the SDK can't be used,
    NativeCrash if it crashed."""
    process = process or process_for(find_library(sdk_dir))
    reply = process.request("open", data, doing="dirp_create_from_rjpeg")
    info = _result(reply, None)
    generation = process.generation
    revision, magic = info["api"]
    if revision not in SUPPORTED_API_VERSIONS:
        process.close_handle(info["id"], generation)
        raise NativeSdkError(f"DJI's Thermal SDK at {process.library} reports API revision {revision} "
                             f"({magic!r}); {_supported_text()}")
    return NativeRJpeg(process, generation, info, data)


def check(sdk_dir: str | os.PathLike, samples: list[str | os.PathLike]) -> dict:
    """Whether DJI's SDK in sdk_dir can be used, from R-JPEGs it is given (the API revision is
    reported per open file): {"library", "sdk_version", "api_revisions", "measured": file names}.
    Raises NativeSdkError (or FileNotFoundError for a folder without the SDK)."""
    library = find_library(sdk_dir)
    process = process_for(library)
    revisions, measured, problems = set(), [], []
    for sample in samples:
        path = Path(sample)
        try:
            rj = open(path.read_bytes(), sdk_dir, process)
            rj.measure_temperature_c()
        except NativeSdkError:
            raise
        except tsdk.TsdkError as e:
            problems.append(f"{path.name}: {e}")
            continue
        revisions.add(rj.api_revision)
        measured.append(path.name)
    if not measured:
        raise NativeSdkError(f"DJI's Thermal SDK at {library} measured none of the sample files"
                             + (f" ({'; '.join(problems)})" if problems else ""))
    return {"library": str(library), "sdk_version": SDK_VERSION, "api_revisions": sorted(revisions),
            "measured": measured}


def _message(func: str, code: int) -> str:
    return f"DJI's Thermal SDK: {func} failed with {bindings.describe_ret(code)}"


def _dirp_error(func: str, code: int, rj: NativeRJpeg | None) -> tsdk.TsdkError:
    """DJI's error code as the port's exception for the same refusal."""
    message = _message(func, code)
    if func == "dirp_create_from_rjpeg":
        e = tsdk.UnsupportedRJpeg(message) if code in (-12, -16) else tsdk.RJpegError(message)
    elif code == -12:
        e = tsdk.UnsupportedFeature(message)
    elif func == "dirp_set_measurement_params" and code == -3:
        return tsdk.ParamError(rj.refused_params() if rj else message, code)
    elif code == -3:
        return tsdk.ParamError(message, code)
    elif func in _MEASURE:
        return tsdk.MeasureError(message, code)
    elif func in _ISP:
        return tsdk.IspError(message, code)
    else:
        return NativeError(message, code)
    e.code = code
    return e


def _result(reply, rj: NativeRJpeg | None):
    kind = reply[0]
    if kind == "ok":
        return reply[1]
    if kind == "dirp":
        raise _dirp_error(reply[1], reply[2], rj)
    raise NativeError(f"DJI's Thermal SDK: {reply[1]}: {reply[2]}")


class NativeRJpeg(tsdk.RJpeg):
    """One R-JPEG open in DJI's SDK, in the child process, with the port's RJpeg interface."""

    def __init__(self, process: NativeProcess, generation: int, info: dict, data: bytes):
        segs = jpeg.segments(data)
        self._process = process
        self._generation = generation
        self._id = info["id"]
        self.plugin = tsdk.plugin_name(segs)
        self.nd = jpeg.nd_filter(segs)
        self.width, self.height = info["width"], info["height"]
        self.api_revision, self._magic = info["api"]
        self._range = None

    def __del__(self):
        process = getattr(self, "_process", None)
        if process is not None:
            process.close_handle(self._id, self._generation)

    def _call(self, func: str, method: str, *args):
        if self._generation != self._process.generation:
            raise NativeCrash("DJI's Thermal SDK was restarted (after a crash, or replaced): open the file again")
        return _result(self._process.request("call", self._id, method, args, doing=func), self)

    def refused_params(self) -> str:
        """Why DJI's SDK refused the parameters last set: those outside the camera's range."""
        p, ranges = self._last_set, self.get_measurement_params_range()
        outside = [f"{name}={p[name]:g} is outside [{lo:g}, {hi:g}]" for name, (lo, hi) in ranges.items()
                   if (lo, hi) != (0.0, 0.0) and not lo <= np.float32(p[name]) <= hi]
        return "; ".join(outside) or _message("dirp_set_measurement_params", -3)

    def get_api_version(self) -> tuple[int, str]:
        return self.api_revision, self._magic

    def infrared_density_filter_installed(self) -> bool:
        return self._call("dirp_get_infared_density_filter_installed", "infrared_density_filter_installed")

    def get_original_raw(self) -> np.ndarray:
        return self._call("dirp_get_original_raw", "get_original_raw")

    def rjpeg_version(self) -> tuple[int, int, int]:
        return tuple(self._call("dirp_get_rjpeg_version", "rjpeg_version"))

    def get_measurement_params(self) -> tsdk.MeasurementParams:
        return tsdk.MeasurementParams(**self._call("dirp_get_measurement_params", "get_measurement_params"))

    def set_measurement_params(self, params: tsdk.MeasurementParams) -> None:
        self._last_set = params.as_dict()
        self._call("dirp_set_measurement_params", "set_measurement_params", self._last_set)

    def get_measurement_params_range(self) -> dict[str, tuple[float, float]]:
        if self._range is None:             # fixed for a file
            self._range = self._call("dirp_get_measurement_params_range", "get_measurement_params_range")
        return dict(self._range)

    def measure(self) -> np.ndarray:
        return self._call("dirp_measure", "measure")

    def measure_temperature_c(self) -> np.ndarray:
        return self._call("dirp_measure_ex", "measure_temperature_c")

    def get_pseudo_color(self) -> int:
        return self._call("dirp_get_pseudo_color", "get_pseudo_color")

    def set_pseudo_color(self, palette: int) -> None:
        self._call("dirp_set_pseudo_color", "set_pseudo_color", int(palette))

    def get_color_bar(self) -> tsdk.ColorBar:
        return tsdk.ColorBar(*self._call("dirp_get_color_bar", "get_color_bar"))

    def get_color_bar_adaptive_params(self) -> tsdk.ColorBar:
        return tsdk.ColorBar(*self._call("dirp_get_color_bar_adaptive_params", "get_color_bar_adaptive_params"))

    def set_color_bar(self, cb: tsdk.ColorBar) -> None:
        self._call("dirp_set_color_bar", "set_color_bar", bool(cb.manual_enable), float(cb.high), float(cb.low))

    def get_isotherm(self) -> tsdk.Isotherm:
        return tsdk.Isotherm(*self._call("dirp_get_isotherm", "get_isotherm"))

    def set_isotherm(self, iso: tsdk.Isotherm) -> None:
        self._call("dirp_set_isotherm", "set_isotherm", bool(iso.enable), float(iso.high), float(iso.low))

    def get_enhancement_params(self) -> int:
        return self._call("dirp_get_enhancement_params", "get_enhancement_params")

    def set_enhancement_params(self, brightness: int) -> None:
        self._call("dirp_set_enhancement_params", "set_enhancement_params", int(brightness))

    def get_pseudo_color_lut(self) -> np.ndarray:
        return self._call("dirp_get_pseudo_color_lut", "get_pseudo_color_lut")

    def process(self) -> np.ndarray:
        return self._call("dirp_process", "process")

    def process_strech(self) -> np.ndarray:
        return self._call("dirp_process_strech", "process_strech")
