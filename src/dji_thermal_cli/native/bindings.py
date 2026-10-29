# Portions of this file -- the struct layouts and function signatures below -- are transcribed
# from DJI's Thermal SDK header dirp_api.h, which DJI distributes under the MIT License. DJI's
# notice, reproduced as that license requires:
#
# Copyright (c) 2020-2023 DJI. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The rest of this file is part of dji-thermal-cli and is licensed under AGPL-3.0-only
# (see the LICENSE file).

"""ctypes bindings of DJI's libdirp (DJI Thermal SDK v1.8, dirp_api.h).

Loaded only in the native child process (see _child): libdirp can crash, and a crash there
must never take the caller's process with it. The tests' own bindings (tests/native_sdk.py)
are separate on purpose: they are the oracle this module is checked against.
"""

from __future__ import annotations

import ctypes as CT
import os
import sys
from pathlib import Path

import numpy as np

DIRP_SUCCESS = 0

ERROR_NAMES = {
    0: "DIRP_SUCCESS", -1: "DIRP_ERROR_MALLOC", -2: "DIRP_ERROR_POINTER_NULL", -3: "DIRP_ERROR_INVALID_PARAMS",
    -4: "DIRP_ERROR_INVALID_RAW", -5: "DIRP_ERROR_INVALID_HEADER", -6: "DIRP_ERROR_INVALID_CURVE",
    -7: "DIRP_ERROR_RJPEG_PARSE", -8: "DIRP_ERROR_SIZE", -9: "DIRP_ERROR_INVALID_HANDLE",
    -10: "DIRP_ERROR_FORMAT_INPUT", -11: "DIRP_ERROR_FORMAT_OUTPUT", -12: "DIRP_ERROR_UNSUPPORTED_FUNC",
    -13: "DIRP_ERROR_NOT_READY", -14: "DIRP_ERROR_ACTIVATION", -15: "DIRP_ERROR_INVALID_INI",
    -16: "DIRP_ERROR_INVALID_SUB_DLL", -32: "DIRP_ERROR_ADVANCED", -64: "DIRP_ERROR_SUPER_MODE",
}

# The SDK release these bindings follow, and the dirp_api_version_t.api values it reports (per
# camera plugin: iirp reports 20 for v0300 headers, M4T and H30T, the others 19). Struct layouts
# change between releases, so another revision is refused rather than read wrongly.
SDK_VERSION = "1.8"
SUPPORTED_API_VERSIONS = frozenset({19, 20})

PSEUDO_COLOR_NUM = 10
LUT_DEPTH = 256

# Where DJI's zip keeps the libraries, below the SDK's root
SDK_LIB_SUBDIR = Path("tsdk-core", "lib")


class DirpError(RuntimeError):
    def __init__(self, func: str, code: int):
        self.func = func
        self.code = code
        super().__init__(f"{func} failed: {describe_ret(code)}")


def describe_ret(code: int) -> str:
    return f"{code} ({ERROR_NAMES.get(code, 'unknown')})"


def _check(func: str, code: int) -> None:
    if code != DIRP_SUCCESS:
        raise DirpError(func, code)


# --- structs: dirp_api.h's #pragma pack(push, 1) section ----------------------------------------

class _Packed(CT.Structure):
    _pack_ = 1
    _layout_ = "ms"         # #pragma pack(1) without bitfields: the same layout as GCC's


class dirp_api_version_t(_Packed):
    _fields_ = [("api", CT.c_uint32), ("magic", CT.c_char * 8)]


class dirp_rjpeg_version_t(_Packed):
    _fields_ = [("rjpeg", CT.c_uint32), ("header", CT.c_uint32), ("curve", CT.c_uint32)]


class dirp_resolution_t(_Packed):
    _fields_ = [("width", CT.c_int32), ("height", CT.c_int32)]


class dirp_isotherm_t(_Packed):
    _fields_ = [("enable", CT.c_bool), ("high", CT.c_float), ("low", CT.c_float)]


class dirp_color_bar_t(_Packed):
    _fields_ = [("manual_enable", CT.c_bool), ("high", CT.c_float), ("low", CT.c_float)]


class dirp_isp_pseudo_color_lut_t(_Packed):
    _fields_ = [(c, CT.c_uint8 * LUT_DEPTH * PSEUDO_COLOR_NUM) for c in ("red", "green", "blue")]


class dirp_enhancement_params_t(_Packed):
    _fields_ = [("brightness", CT.c_int32)]


PARAM_FIELDS = ("distance", "humidity", "emissivity", "reflection", "ambient_temp")


class dirp_measurement_params_t(_Packed):
    """humidity is a percent (0-100); reflection and ambient_temp are Celsius."""

    _fields_ = [(name, CT.c_float) for name in PARAM_FIELDS]


class _MinMax(_Packed):
    _fields_ = [("min", CT.c_float), ("max", CT.c_float)]


class dirp_measurement_params_range_t(_Packed):
    _fields_ = [(name, _MinMax) for name in PARAM_FIELDS]


DIRP_HANDLE = CT.c_void_p


# --- finding the libraries ------------------------------------------------------------------

def library_name() -> str:
    return "libdirp.dll" if os.name == "nt" else "libdirp.so"


def platform_dirs() -> tuple[str, str]:
    """(DJI's folder for this system, for this architecture): ("linux", "release_x64") etc."""
    import platform
    machine = platform.machine().lower()
    return ("windows" if os.name == "nt" else "linux"), ("release_x64" if machine in ("amd64", "x86_64") else "release_x86")


def find_library(sdk_dir: str | os.PathLike) -> Path:
    """libdirp in sdk_dir: an unpacked DJI Thermal SDK (its root, its tsdk-core/lib, or the
    <os>/<arch> folder of its libraries). Raises FileNotFoundError naming where it looked."""
    root = Path(sdk_dir)
    if sys.platform == "darwin":
        raise FileNotFoundError("DJI's Thermal SDK has no macOS build (only Linux and Windows)")
    osname, arch = platform_dirs()
    name = library_name()
    for base in (root / SDK_LIB_SUBDIR, root):
        for candidate in (base / osname / arch / name, base / arch / name, base / name):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no {name} for {osname}/{arch} in {root} (an unpacked DJI Thermal SDK, "
                            f"with {SDK_LIB_SUBDIR / osname / arch / name})")


# --- the library ----------------------------------------------------------------------------

class Dirp:
    """One loaded libdirp. libdirp loads its camera plugins (libv_*, listed in libv_list.ini)
    from its own folder, so the library stays where DJI's zip put it."""

    # The libraries the libv_* plugins link against (DT_NEEDED), in dependency order
    _LINUX_COMPANIONS = ("libexif.so.12", "libMicroJPEG_Release_x64.so", "libMicroJPEG_Release_x86.so",
                         "libMicroTA_Release_x64.so", "libMicroTA_Release_x86.so",
                         "libMicroIA_Release_x64.so", "libMicroIA_Release_x86.so")

    def __init__(self, library: Path):
        self.path = library
        if os.name == "nt":
            self._dll_directory = os.add_dll_directory(str(library.parent.resolve()))
        # On Linux the plugins' own dependencies are found only by LD_LIBRARY_PATH or by being
        # loaded already (by soname). RTLD_LOCAL: with RTLD_GLOBAL the plugins' duplicate C++
        # statics are interposed and the heap is corrupted at exit.
        self._companions = [] if os.name == "nt" else [
            CT.CDLL(str(library.parent / n), mode=CT.RTLD_LOCAL)
            for n in self._LINUX_COMPANIONS if (library.parent / n).exists()]
        self.lib = CT.CDLL(str(library))
        self._bind()

    def _bind(self) -> None:
        P = CT.POINTER
        sigs = {
            "dirp_create_from_rjpeg": [CT.c_char_p, CT.c_int32, P(DIRP_HANDLE)],
            "dirp_destroy": [DIRP_HANDLE],
            "dirp_get_api_version": [DIRP_HANDLE, P(dirp_api_version_t)],
            "dirp_get_rjpeg_version": [DIRP_HANDLE, P(dirp_rjpeg_version_t)],
            "dirp_get_rjpeg_resolution": [DIRP_HANDLE, P(dirp_resolution_t)],
            "dirp_get_original_raw": [DIRP_HANDLE, CT.c_void_p, CT.c_int32],
            "dirp_process": [DIRP_HANDLE, CT.c_void_p, CT.c_int32],
            "dirp_process_strech": [DIRP_HANDLE, CT.c_void_p, CT.c_int32],
            "dirp_measure": [DIRP_HANDLE, CT.c_void_p, CT.c_int32],
            "dirp_measure_ex": [DIRP_HANDLE, CT.c_void_p, CT.c_int32],
            "dirp_set_isotherm": [DIRP_HANDLE, P(dirp_isotherm_t)],
            "dirp_get_isotherm": [DIRP_HANDLE, P(dirp_isotherm_t)],
            "dirp_set_color_bar": [DIRP_HANDLE, P(dirp_color_bar_t)],
            "dirp_get_color_bar": [DIRP_HANDLE, P(dirp_color_bar_t)],
            "dirp_get_color_bar_adaptive_params": [DIRP_HANDLE, P(dirp_color_bar_t)],
            "dirp_set_pseudo_color": [DIRP_HANDLE, CT.c_int32],
            "dirp_get_pseudo_color": [DIRP_HANDLE, P(CT.c_int32)],
            "dirp_get_pseudo_color_lut": [DIRP_HANDLE, P(dirp_isp_pseudo_color_lut_t)],
            "dirp_set_enhancement_params": [DIRP_HANDLE, P(dirp_enhancement_params_t)],
            "dirp_get_enhancement_params": [DIRP_HANDLE, P(dirp_enhancement_params_t)],
            "dirp_set_measurement_params": [DIRP_HANDLE, P(dirp_measurement_params_t)],
            "dirp_get_measurement_params": [DIRP_HANDLE, P(dirp_measurement_params_t)],
            "dirp_get_measurement_params_range": [DIRP_HANDLE, P(dirp_measurement_params_range_t)],
        }
        for name, argtypes in sigs.items():
            f = getattr(self.lib, name)
            f.argtypes, f.restype = argtypes, CT.c_int32
        f = self.lib.dirp_get_infared_density_filter_installed
        f.argtypes, f.restype = [DIRP_HANDLE], CT.c_uint32

    def open(self, data: bytes) -> Handle:
        return Handle(self.lib, data)


class Handle:
    """One dirp handle over one R-JPEG's bytes. Results are plain Python and numpy values."""

    def __init__(self, lib: CT.CDLL, data: bytes):
        self._lib = lib
        self._handle = DIRP_HANDLE()
        self._data = CT.create_string_buffer(data, len(data))   # read lazily: must outlive the handle
        _check("dirp_create_from_rjpeg", lib.dirp_create_from_rjpeg(self._data, len(data), CT.byref(self._handle)))
        self.width, self.height = self.resolution()

    def close(self) -> None:
        if self._handle:
            self._lib.dirp_destroy(self._handle)
            self._handle = DIRP_HANDLE()

    def _get(self, func: str, struct):
        value = struct()
        _check(func, getattr(self._lib, func)(self._handle, CT.byref(value)))
        return value

    def _image(self, func: str, dtype, channels: int = 1) -> np.ndarray:
        shape = (self.height, self.width) + ((channels,) if channels > 1 else ())
        out = np.zeros(shape, dtype)
        _check(func, getattr(self._lib, func)(self._handle, out.ctypes.data_as(CT.c_void_p), out.nbytes))
        return out

    def api_version(self) -> tuple[int, str]:
        v = self._get("dirp_get_api_version", dirp_api_version_t)
        return int(v.api), v.magic.decode("ascii", "replace")

    def rjpeg_version(self) -> tuple[int, int, int]:
        v = self._get("dirp_get_rjpeg_version", dirp_rjpeg_version_t)
        return int(v.rjpeg), int(v.header), int(v.curve)

    def resolution(self) -> tuple[int, int]:
        r = self._get("dirp_get_rjpeg_resolution", dirp_resolution_t)
        return int(r.width), int(r.height)

    def infrared_density_filter_installed(self) -> bool:
        return bool(self._lib.dirp_get_infared_density_filter_installed(self._handle))

    def get_measurement_params(self) -> dict[str, float]:
        p = self._get("dirp_get_measurement_params", dirp_measurement_params_t)
        return {n: float(getattr(p, n)) for n in PARAM_FIELDS}

    def set_measurement_params(self, params: dict[str, float]) -> None:
        p = dirp_measurement_params_t(*(params[n] for n in PARAM_FIELDS))
        _check("dirp_set_measurement_params", self._lib.dirp_set_measurement_params(self._handle, CT.byref(p)))

    def get_measurement_params_range(self) -> dict[str, tuple[float, float]]:
        r = self._get("dirp_get_measurement_params_range", dirp_measurement_params_range_t)
        return {n: (float(getattr(r, n).min), float(getattr(r, n).max)) for n in PARAM_FIELDS}

    def measure(self) -> np.ndarray:
        return self._image("dirp_measure", np.int16)

    def measure_temperature_c(self) -> np.ndarray:
        return self._image("dirp_measure_ex", np.float32)

    def get_original_raw(self) -> np.ndarray:
        return self._image("dirp_get_original_raw", np.uint16)

    def process(self) -> np.ndarray:
        return self._image("dirp_process", np.uint8, 3)

    def process_strech(self) -> np.ndarray:
        return self._image("dirp_process_strech", np.float32)

    def get_pseudo_color(self) -> int:
        return int(self._get("dirp_get_pseudo_color", CT.c_int32).value)

    def set_pseudo_color(self, palette: int) -> None:
        _check("dirp_set_pseudo_color", self._lib.dirp_set_pseudo_color(self._handle, int(palette)))

    def _bar(self, func: str) -> tuple[bool, float, float]:
        cb = self._get(func, dirp_color_bar_t)
        return bool(cb.manual_enable), float(cb.high), float(cb.low)

    def get_color_bar(self) -> tuple[bool, float, float]:
        return self._bar("dirp_get_color_bar")

    def get_color_bar_adaptive_params(self) -> tuple[bool, float, float]:
        return self._bar("dirp_get_color_bar_adaptive_params")

    def set_color_bar(self, manual_enable: bool, high: float, low: float) -> None:
        cb = dirp_color_bar_t(bool(manual_enable), high, low)
        _check("dirp_set_color_bar", self._lib.dirp_set_color_bar(self._handle, CT.byref(cb)))

    def get_isotherm(self) -> tuple[bool, float, float]:
        iso = self._get("dirp_get_isotherm", dirp_isotherm_t)
        return bool(iso.enable), float(iso.high), float(iso.low)

    def set_isotherm(self, enable: bool, high: float, low: float) -> None:
        iso = dirp_isotherm_t(bool(enable), high, low)
        _check("dirp_set_isotherm", self._lib.dirp_set_isotherm(self._handle, CT.byref(iso)))

    def get_enhancement_params(self) -> int:
        return int(self._get("dirp_get_enhancement_params", dirp_enhancement_params_t).brightness)

    def set_enhancement_params(self, brightness: int) -> None:
        e = dirp_enhancement_params_t(int(brightness))
        _check("dirp_set_enhancement_params", self._lib.dirp_set_enhancement_params(self._handle, CT.byref(e)))

    def get_pseudo_color_lut(self) -> np.ndarray:
        lut = self._get("dirp_get_pseudo_color_lut", dirp_isp_pseudo_color_lut_t)
        return np.frombuffer(bytes(lut), np.uint8).reshape(3, PSEUDO_COLOR_NUM, LUT_DEPTH).copy()
