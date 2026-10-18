"""Subprocess helper for the tests: measure one R-JPEG with the native DJI Thermal SDK.

    native_measure.py <file> [distance humidity emissivity reflection ambient]

Writes an npz to stdout with t (float32 temperatures, empty on failure), params (the
parameters the SDK reports after setting), code (0, or the failing dirp return code),
version (rjpeg, header, curve) and resolution (width, height).
"""

import io
import os
import sys

import numpy as np

os.environ.setdefault("DJI_THERMAL_SDK_ALLOW_UNKNOWN_VERSION", "1")
import native_sdk as sdk  # noqa: E402

lib = sdk.DirpSDK()
code, t = 0, np.zeros(0, np.float32)
with lib.open(open(sys.argv[1], "rb").read()) as r:
    version, resolution = np.array(r.rjpeg_version()), np.array(r.resolution())
    p = r.get_measurement_params()
    try:
        if len(sys.argv) > 2:
            p.distance, p.humidity, p.emissivity, p.reflection, p.ambient_temp = map(float, sys.argv[2:7])
            r.set_measurement_params(p)
        t = r.measure_temperature_c()
    except sdk.DirpError as e:
        code = e.code
    p = r.get_measurement_params()
    params = np.array([p.distance, p.humidity, p.emissivity, p.reflection, p.ambient_temp])
buf = io.BytesIO()
np.savez(buf, t=t, params=params, code=code, version=version, resolution=resolution)
sys.stdout.buffer.write(buf.getvalue())
