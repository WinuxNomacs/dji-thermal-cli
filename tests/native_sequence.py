"""Subprocess helper for the tests: replay set/measure steps on ONE native SDK handle.

    native_sequence.py <file> <steps.json>

steps.json is a list of parameter dicts (or null for "measure without setting"). Writes an
npz to stdout: for step i, set_code_i, measure_code_i, t_i (float32, empty on failure) and
params_i (what the SDK reports after the step).
"""

import io
import json
import os
import sys

import numpy as np

os.environ.setdefault("DJI_THERMAL_SDK_ALLOW_UNKNOWN_VERSION", "1")
import native_sdk as sdk  # noqa: E402

NAMES = ("distance", "humidity", "emissivity", "reflection", "ambient_temp")
lib = sdk.DirpSDK()
steps = json.load(open(sys.argv[2]))
out = {}
with lib.open(open(sys.argv[1], "rb").read()) as r:
    for i, step in enumerate(steps):
        set_code = 0
        if step is not None:
            p = r.get_measurement_params()
            for k in NAMES:
                setattr(p, k, float(step[k]))
            try:
                r.set_measurement_params(p)
            except sdk.DirpError as e:
                set_code = e.code
        meas_code, t = 0, np.zeros(0, np.float32)
        try:
            t = r.measure_temperature_c()
        except sdk.DirpError as e:
            meas_code = e.code
        try:
            p = r.get_measurement_params()
            params = np.array([getattr(p, k) for k in NAMES], np.float32)
        except sdk.DirpError:
            params = np.zeros(0, np.float32)
        out[f"set_{i}"], out[f"meas_{i}"], out[f"t_{i}"], out[f"params_{i}"] = set_code, meas_code, t, params
buf = io.BytesIO()
np.savez(buf, **out)
sys.stdout.buffer.write(buf.getvalue())
