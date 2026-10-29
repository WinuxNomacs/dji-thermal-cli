"""A native child process for the tests, without DJI's SDK: the real request loop (native._child)
serving a stand-in library that can crash, hang or print.

    fake_native_child.py <mode>

mode "ok" serves files; "unsupported" reports API revision 21; "fail" can't load.
"""

import ctypes
import os
import sys
import time

import numpy as np

from dji_thermal_cli.native._child import serve


class FakeDirpError(RuntimeError):
    def __init__(self, func, code):
        super().__init__(f"{func} failed")
        self.func, self.code = func, code


class Handle:
    width, height = 2, 1

    def __init__(self, revision):
        self.revision = revision
        self.params = {"distance": 5.0, "humidity": 70.0, "emissivity": 1.0, "reflection": 23.0, "ambient_temp": 25.0}

    def close(self):
        pass

    def api_version(self):
        return self.revision, "relver"

    def get_measurement_params(self):
        return dict(self.params)

    def get_measurement_params_range(self):
        return {"distance": (1.0, 25.0), "humidity": (20.0, 100.0), "emissivity": (0.1, 1.0),
                "reflection": (-40.0, 500.0), "ambient_temp": (0.0, 0.0)}

    def set_measurement_params(self, params):
        if not 1.0 <= params["distance"] <= 25.0:
            raise FakeDirpError("dirp_set_measurement_params", -3)
        self.params = dict(params)

    def measure_temperature_c(self):
        return np.array([[20.5, 21.5]], np.float32)

    def crash(self):
        ctypes.string_at(0)             # a segmentation fault, as a native library's

    def hang(self):
        time.sleep(600)

    def chatter(self):
        os.write(1, b"library log line\n")      # what a native library prints to stdout
        return "answered"


class Library:
    def __init__(self, mode):
        if mode == "fail":
            raise OSError("libdirp.so: cannot open shared object file")
        self.revision = 21 if mode == "unsupported" else 20

    def open(self, data):
        return Handle(self.revision)


if __name__ == "__main__":
    serve(Library, sys.argv[1:])
