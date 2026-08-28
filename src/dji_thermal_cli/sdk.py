"""ctypes bindings for the DJI Thermal SDK's dirp (R-JPEG) API.

These structs and function signatures are transcribed directly from
``lib/tsdk-core/api/dirp_api.h`` (DJI Thermal SDK v1.8, vendored in this
repo under ``lib/``). They intentionally do NOT reuse the third-party
``dji_thermal_sdk`` pip package: that package's ``dirp_measurement_params_t``
targets SDK v1.3 and is missing the ``ambient_temp`` field that v1.8 added,
which would misalign every field after ``reflection`` if used against the
real (newer) ``libdirp`` binary bundled here.
"""

from __future__ import annotations

import ctypes as CT
import os
import platform
from pathlib import Path

import numpy as np

DIRP_SUCCESS = 0

_ERROR_NAMES = {
    0: "DIRP_SUCCESS",
    -1: "DIRP_ERROR_MALLOC",
    -2: "DIRP_ERROR_POINTER_NULL",
    -3: "DIRP_ERROR_INVALID_PARAMS",
    -4: "DIRP_ERROR_INVALID_RAW",
    -5: "DIRP_ERROR_INVALID_HEADER",
    -6: "DIRP_ERROR_INVALID_CURVE",
    -7: "DIRP_ERROR_RJPEG_PARSE",
    -8: "DIRP_ERROR_SIZE",
    -9: "DIRP_ERROR_INVALID_HANDLE",
    -10: "DIRP_ERROR_FORMAT_INPUT",
    -11: "DIRP_ERROR_FORMAT_OUTPUT",
    -12: "DIRP_ERROR_UNSUPPORTED_FUNC",
    -13: "DIRP_ERROR_NOT_READY",
    -14: "DIRP_ERROR_ACTIVATION",
    -15: "DIRP_ERROR_INVALID_INI",
    -16: "DIRP_ERROR_INVALID_SUB_DLL",
    -32: "DIRP_ERROR_ADVANCED",
    -64: "DIRP_ERROR_SUPER_MODE",
}

PSEUDO_COLOR_NAMES = [
    "whitehot", "fulgurite", "ironred", "hotiron", "medical",
    "arctic", "rainbow1", "rainbow2", "tint", "blackhot",
]


def describe_ret(code: int) -> str:
    return f"{code} ({_ERROR_NAMES.get(code, 'unknown')})"


class DirpError(RuntimeError):
    def __init__(self, func: str, code: int):
        self.func = func
        self.code = code
        super().__init__(f"{func} failed: {describe_ret(code)}")


def _check(func_name: str, code: int) -> None:
    if code != DIRP_SUCCESS:
        raise DirpError(func_name, code)


DIRP_HANDLE = CT.c_void_p


# --- structs, matching dirp_api.h's #pragma pack(push, 1) section ---------

class dirp_api_version_t(CT.Structure):
    _pack_ = 1
    _fields_ = [("api", CT.c_uint32), ("magic", CT.c_char * 8)]


class dirp_rjpeg_version_t(CT.Structure):
    _pack_ = 1
    _fields_ = [("rjpeg", CT.c_uint32), ("header", CT.c_uint32), ("curve", CT.c_uint32)]


class dirp_resolution_t(CT.Structure):
    _pack_ = 1
    _fields_ = [("width", CT.c_int32), ("height", CT.c_int32)]


class dirp_color_bar_t(CT.Structure):
    _pack_ = 1
    _fields_ = [("manual_enable", CT.c_bool), ("high", CT.c_float), ("low", CT.c_float)]


class dirp_measurement_params_t(CT.Structure):
    """5-field struct (v1.8): distance, humidity, emissivity, reflection,
    ambient_temp. humidity is a percent (0-100), confirmed empirically
    against the SDK's own sample R-JPEGs (dirp_get_measurement_params_range
    reports e.g. humidity in [20.0, 100.0], and a sample file reads back
    humidity=70.0) -- NOT a 0-1 fraction, despite that being true of the
    older v1.3-era dji_thermal_sdk pip wrapper this project intentionally
    does not use. reflection/ambient_temp are Celsius (per the header's own
    comments)."""

    _pack_ = 1
    _fields_ = [
        ("distance", CT.c_float),
        ("humidity", CT.c_float),
        ("emissivity", CT.c_float),
        ("reflection", CT.c_float),
        ("ambient_temp", CT.c_float),
    ]

    FIELDS = ("distance", "humidity", "emissivity", "reflection", "ambient_temp")

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.FIELDS}


class _MinMax(CT.Structure):
    _pack_ = 1
    _fields_ = [("min", CT.c_float), ("max", CT.c_float)]


class dirp_measurement_params_range_t(CT.Structure):
    _pack_ = 1
    _fields_ = [
        ("distance", _MinMax),
        ("humidity", _MinMax),
        ("emissivity", _MinMax),
        ("reflection", _MinMax),
        ("ambient_temp", _MinMax),
    ]

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            name: (getattr(self, name).min, getattr(self, name).max)
            for name in dirp_measurement_params_t.FIELDS
        }


# --- locating the vendored library -----------------------------------------

def _default_lib_root() -> Path:
    # src/dji_thermal_cli/sdk.py -> <repo root>/lib/tsdk-core/lib
    return Path(__file__).resolve().parents[2] / "lib" / "tsdk-core" / "lib"


def _os_arch() -> tuple[str, str]:
    osname = "windows" if os.name == "nt" else "linux"
    machine = platform.machine().lower()
    arch = "release_x64" if machine in ("amd64", "x86_64") else "release_x86"
    return osname, arch


def resolve_dll_path(dll_dir: str | Path | None) -> Path:
    """Locate libdirp.{so,dll}.

    Search order: an explicit --dll-dir, then $DJI_THERMAL_SDK_LIB_DIR, then
    the lib/tsdk-core/lib tree vendored in this repo. Each root is tried as:
    the file directly, a windows/linux subfolder, or the full
    <os>/release_x64|x86 layout DJI ships the SDK in.
    """
    osname, arch = _os_arch()
    libname = "libdirp.dll" if osname == "windows" else "libdirp.so"

    roots: list[Path] = []
    if dll_dir:
        roots.append(Path(dll_dir))
    env_dir = os.environ.get("DJI_THERMAL_SDK_LIB_DIR")
    if env_dir:
        roots.append(Path(env_dir))
    roots.append(_default_lib_root())

    for root in roots:
        for candidate in (
            root / libname,
            root / osname / libname,
            root / osname / arch / libname,
            root / arch / libname,
        ):
            if candidate.exists():
                return candidate

    searched = ", ".join(str(r) for r in roots)
    raise FileNotFoundError(
        f"Could not find {libname} under any of: {searched}. "
        f"Pass --dll-dir explicitly, or set DJI_THERMAL_SDK_LIB_DIR."
    )


class DirpSDK:
    """One loaded libdirp instance with bound function signatures.

    libdirp dynamically loads sibling plugin libraries (libv_dirp.so,
    libv_girp.so, ...) listed in libv_list.ini next to it, so the resolved
    library must stay inside its original release_x64/release_x86 directory
    alongside those companions -- which is exactly how they're vendored
    under lib/tsdk-core/lib/.
    """

    def __init__(self, dll_dir: str | Path | None = None):
        self.dll_path = resolve_dll_path(dll_dir)
        self._lib = CT.CDLL(str(self.dll_path))
        self._bind()

    def _bind(self) -> None:
        lib = self._lib
        lib.dirp_create_from_rjpeg.argtypes = [CT.c_char_p, CT.c_int32, CT.POINTER(DIRP_HANDLE)]
        lib.dirp_create_from_rjpeg.restype = CT.c_int32
        lib.dirp_destroy.argtypes = [DIRP_HANDLE]
        lib.dirp_destroy.restype = CT.c_int32
        lib.dirp_get_rjpeg_resolution.argtypes = [DIRP_HANDLE, CT.POINTER(dirp_resolution_t)]
        lib.dirp_get_rjpeg_resolution.restype = CT.c_int32
        lib.dirp_get_rjpeg_version.argtypes = [DIRP_HANDLE, CT.POINTER(dirp_rjpeg_version_t)]
        lib.dirp_get_rjpeg_version.restype = CT.c_int32
        lib.dirp_process.argtypes = [DIRP_HANDLE, CT.c_void_p, CT.c_int32]
        lib.dirp_process.restype = CT.c_int32
        lib.dirp_measure_ex.argtypes = [DIRP_HANDLE, CT.c_void_p, CT.c_int32]
        lib.dirp_measure_ex.restype = CT.c_int32
        lib.dirp_set_color_bar.argtypes = [DIRP_HANDLE, CT.POINTER(dirp_color_bar_t)]
        lib.dirp_set_color_bar.restype = CT.c_int32
        lib.dirp_get_color_bar.argtypes = [DIRP_HANDLE, CT.POINTER(dirp_color_bar_t)]
        lib.dirp_get_color_bar.restype = CT.c_int32
        lib.dirp_set_pseudo_color.argtypes = [DIRP_HANDLE, CT.c_int32]
        lib.dirp_set_pseudo_color.restype = CT.c_int32
        lib.dirp_get_pseudo_color.argtypes = [DIRP_HANDLE, CT.POINTER(CT.c_int32)]
        lib.dirp_get_pseudo_color.restype = CT.c_int32
        lib.dirp_set_measurement_params.argtypes = [DIRP_HANDLE, CT.POINTER(dirp_measurement_params_t)]
        lib.dirp_set_measurement_params.restype = CT.c_int32
        lib.dirp_get_measurement_params.argtypes = [DIRP_HANDLE, CT.POINTER(dirp_measurement_params_t)]
        lib.dirp_get_measurement_params.restype = CT.c_int32
        lib.dirp_get_measurement_params_range.argtypes = [DIRP_HANDLE, CT.POINTER(dirp_measurement_params_range_t)]
        lib.dirp_get_measurement_params_range.restype = CT.c_int32

    def open(self, data: bytes) -> "RJpeg":
        return RJpeg(self._lib, data)


class RJpeg:
    """One dirp handle over a single R-JPEG's bytes. Use as a context manager."""

    def __init__(self, lib: CT.CDLL, data: bytes):
        self._lib = lib
        self._handle = DIRP_HANDLE()
        # The SDK reads from this buffer lazily, so it must outlive the handle.
        self._data_buf = CT.create_string_buffer(data, len(data))
        ret = lib.dirp_create_from_rjpeg(self._data_buf, CT.c_int32(len(data)), CT.byref(self._handle))
        _check("dirp_create_from_rjpeg", ret)

    def __enter__(self) -> "RJpeg":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._handle:
            self._lib.dirp_destroy(self._handle)
            self._handle = DIRP_HANDLE()

    def resolution(self) -> tuple[int, int]:
        res = dirp_resolution_t()
        _check("dirp_get_rjpeg_resolution", self._lib.dirp_get_rjpeg_resolution(self._handle, CT.byref(res)))
        return res.width, res.height

    def rjpeg_version(self) -> tuple[int, int, int]:
        v = dirp_rjpeg_version_t()
        _check("dirp_get_rjpeg_version", self._lib.dirp_get_rjpeg_version(self._handle, CT.byref(v)))
        return v.rjpeg, v.header, v.curve

    def get_measurement_params(self) -> dirp_measurement_params_t:
        p = dirp_measurement_params_t()
        _check("dirp_get_measurement_params", self._lib.dirp_get_measurement_params(self._handle, CT.byref(p)))
        return p

    def set_measurement_params(self, params: dirp_measurement_params_t) -> None:
        _check("dirp_set_measurement_params", self._lib.dirp_set_measurement_params(self._handle, CT.byref(params)))

    def get_measurement_params_range(self) -> dirp_measurement_params_range_t:
        r = dirp_measurement_params_range_t()
        _check(
            "dirp_get_measurement_params_range",
            self._lib.dirp_get_measurement_params_range(self._handle, CT.byref(r)),
        )
        return r

    def get_color_bar(self) -> dirp_color_bar_t:
        cb = dirp_color_bar_t()
        _check("dirp_get_color_bar", self._lib.dirp_get_color_bar(self._handle, CT.byref(cb)))
        return cb

    def set_color_bar(self, manual_enable: bool, low: float, high: float) -> None:
        cb = dirp_color_bar_t(manual_enable=manual_enable, high=high, low=low)
        _check("dirp_set_color_bar", self._lib.dirp_set_color_bar(self._handle, CT.byref(cb)))

    def get_pseudo_color(self) -> int:
        pc = CT.c_int32()
        _check("dirp_get_pseudo_color", self._lib.dirp_get_pseudo_color(self._handle, CT.byref(pc)))
        return pc.value

    def set_pseudo_color(self, palette: int) -> None:
        _check("dirp_set_pseudo_color", self._lib.dirp_set_pseudo_color(self._handle, CT.c_int32(palette)))

    def measure_temperature_c(self) -> np.ndarray:
        """float32 Celsius array, shape (height, width), via dirp_measure_ex."""
        w, h = self.resolution()
        size = w * h * CT.sizeof(CT.c_float)
        buf = CT.create_string_buffer(size)
        _check("dirp_measure_ex", self._lib.dirp_measure_ex(self._handle, buf, CT.c_int32(size)))
        return np.frombuffer(buf.raw, dtype=np.float32).reshape(h, w).copy()

    def process_pseudo_color(self) -> np.ndarray:
        """uint8 RGB array, shape (height, width, 3), via dirp_process."""
        w, h = self.resolution()
        size = w * h * 3 * CT.sizeof(CT.c_uint8)
        buf = CT.create_string_buffer(size)
        _check("dirp_process", self._lib.dirp_process(self._handle, buf, CT.c_int32(size)))
        return np.frombuffer(buf.raw, dtype=np.uint8).reshape(h, w, 3).copy()
