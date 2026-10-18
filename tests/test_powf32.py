"""The powf port must match glibc's powf, which the native H20T path calls."""

import ctypes
import ctypes.util
import sys

import numpy as np
import pytest

from dji_thermal_cli.tsdk.powf32 import powf

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="compares against glibc")


@pytest.fixture(scope="module")
def libm_powf():
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    libm.powf.restype = ctypes.c_float
    libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
    return libm.powf


@pytest.mark.parametrize("lo,hi,y", [(200, 2000, 4.0), (1e8, 1e13, 0.25), (1e-44, 1e-37, 0.25), (1e-30, 1e30, 2.5), (0.5, 2.0, -3.0)])
def test_matches_glibc(libm_powf, lo, hi, y):
    xs = np.random.default_rng(int(lo * 7 + y * 13) % 2**32).uniform(lo, hi, 20000).astype(np.float32)
    for x in xs:
        assert np.float32(powf(float(x), y)) == np.float32(libm_powf(float(x), y)), (float(x), y)


def test_zero(libm_powf):
    assert powf(0.0, 0.25) == libm_powf(0.0, 0.25) == 0.0
