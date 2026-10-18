"""Subprocess helper for the tests: replay image-pipeline calls on ONE native SDK handle.

    native_process.py <file> <steps.json>

steps.json is a list of {"op": ...} dicts (see OPS). Writes an npz to stdout with, for step i,
code_i (the dirp return code) and, where the op returns data, out_i.
"""

import ctypes as CT
import io
import json
import os
import sys

import numpy as np

os.environ.setdefault("DJI_THERMAL_SDK_ALLOW_UNKNOWN_VERSION", "1")
import native_sdk as sdk  # noqa: E402


class Packed3(CT.Structure):
    """dirp_color_bar_t / dirp_isotherm_t: bool + two floats, packed (9 bytes)."""

    _pack_ = 1
    _fields_ = [("flag", CT.c_bool), ("high", CT.c_float), ("low", CT.c_float)]


class Params(CT.Structure):
    _fields_ = [(n, CT.c_float) for n in ("distance", "humidity", "emissivity", "reflection", "ambient_temp")]


lib = sdk.DirpSDK()
L = lib._lib
L.dirp_get_infared_density_filter_installed.restype = CT.c_uint32
for name in ("dirp_set_isotherm", "dirp_get_isotherm", "dirp_set_color_bar", "dirp_get_color_bar",
             "dirp_get_color_bar_adaptive_params", "dirp_set_enhancement_params", "dirp_get_enhancement_params",
             "dirp_get_pseudo_color_lut", "dirp_process", "dirp_process_strech", "dirp_set_pseudo_color",
             "dirp_get_pseudo_color", "dirp_set_measurement_params", "dirp_measure", "dirp_measure_ex",
             "dirp_get_original_raw", "dirp_get_api_version"):
    getattr(L, name).argtypes = None
    getattr(L, name).restype = CT.c_int32
steps = json.load(open(sys.argv[2]))
out = {}
with lib.open(open(sys.argv[1], "rb").read()) as r:
    h = r._handle
    w, hh = r.resolution()
    for i, s in enumerate(steps):
        op, data = s["op"], None
        if op == "set_palette":
            code = L.dirp_set_pseudo_color(h, CT.c_int32(s["v"]))
        elif op in ("set_cb", "set_iso"):
            v = Packed3(s["flag"], s["high"], s["low"])
            code = (L.dirp_set_color_bar if op == "set_cb" else L.dirp_set_isotherm)(h, CT.byref(v))
        elif op == "set_bright":
            code = L.dirp_set_enhancement_params(h, CT.byref(CT.c_int32(s["v"])))
        elif op == "set_params":
            p = Params(*(s[k] for k in ("distance", "humidity", "emissivity", "reflection", "ambient_temp")))
            code = L.dirp_set_measurement_params(h, CT.byref(p))
        elif op == "get":
            pc, cb, ad, iso, b = CT.c_int32(-1), Packed3(), Packed3(), Packed3(), CT.c_int32(-1)
            codes = [L.dirp_get_pseudo_color(h, CT.byref(pc)), L.dirp_get_color_bar(h, CT.byref(cb)),
                     L.dirp_get_color_bar_adaptive_params(h, CT.byref(ad)), L.dirp_get_isotherm(h, CT.byref(iso)),
                     L.dirp_get_enhancement_params(h, CT.byref(b))]
            code = min(codes)
            data = np.array(codes + [pc.value, cb.flag, cb.high, cb.low, ad.high, ad.low, iso.flag, iso.high, iso.low, b.value],
                            np.float64)
        elif op == "lut":
            buf = CT.create_string_buffer(3 * 10 * 256)
            code = L.dirp_get_pseudo_color_lut(h, buf)
            data = np.frombuffer(buf.raw, np.uint8).reshape(3, 10, 256)
        elif op == "process":
            buf = CT.create_string_buffer(w * hh * 3)
            code = L.dirp_process(h, buf, CT.c_int32(w * hh * 3))
            data = np.frombuffer(buf.raw, np.uint8).reshape(hh, w, 3)
        elif op == "strech":
            buf = CT.create_string_buffer(w * hh * 4)
            code = L.dirp_process_strech(h, buf, CT.c_int32(w * hh * 4))
            data = np.frombuffer(buf.raw, np.float32).reshape(hh, w)
        elif op == "measure16":
            buf = np.zeros(w * hh, np.int16)
            code = L.dirp_measure(h, CT.c_void_p(buf.ctypes.data), CT.c_int32(w * hh * 2))
            data = buf.reshape(hh, w)
        elif op == "measure_ex":
            buf = np.zeros(w * hh, np.float32)
            code = L.dirp_measure_ex(h, CT.c_void_p(buf.ctypes.data), CT.c_int32(w * hh * 4))
            data = buf.reshape(hh, w)
        elif op == "raw":
            buf = np.zeros(w * hh, np.uint16)
            code = L.dirp_get_original_raw(h, CT.c_void_p(buf.ctypes.data), CT.c_int32(w * hh * 2))
            data = buf.reshape(hh, w)
        elif op == "api":
            v = sdk.dirp_api_version_t()
            code = L.dirp_get_api_version(h, CT.byref(v))
            data = np.frombuffer(bytes(v), np.uint8).copy()
        elif op == "nd":
            code = 0
            data = np.array([L.dirp_get_infared_density_filter_installed(h)], np.int64)
        else:
            raise SystemExit(f"unknown op {op}")
        out[f"code_{i}"] = code
        if data is not None and (code == 0 or op == "get"):
            out[f"out_{i}"] = data
buf = io.BytesIO()
np.savez(buf, **out)
sys.stdout.buffer.write(buf.getvalue())
