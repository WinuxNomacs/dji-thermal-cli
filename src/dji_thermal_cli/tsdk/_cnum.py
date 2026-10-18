"""C numeric semantics needed to reproduce the DJI Thermal SDK bit for bit.

The SDK mixes float32 and double arithmetic, truncating integer conversions, 16-bit
wrap-around and C's toward-zero division; these helpers make each of those explicit.
"""

from __future__ import annotations

import math

import numpy as np

F = np.float32


def s16(x: int) -> int:
    """Wrap to a signed 16-bit value, as a C (short) cast does."""
    return ((int(x) + 0x8000) & 0xFFFF) - 0x8000


def s32(x: int) -> int:
    """Wrap to a signed 32-bit value, as C int arithmetic does on overflow."""
    return ((int(x) + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def cvtt(x: float) -> int:
    """x86 cvttss2si/cvttsd2si: truncate toward zero; NaN and out-of-range give INT_MIN."""
    if math.isnan(x) or not (-(2**31) <= x < 2**31):
        return -(2**31)
    return int(x)


def to_short(x) -> np.ndarray:
    """(short)(int)x over an array: truncation toward zero (NaN and out-of-range give INT_MIN,
    as cvttss2si does), then the 16-bit wrap."""
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | (x < -(2.0 ** 31)) | (x >= 2.0 ** 31)
    return np.where(bad, -(2 ** 31), np.trunc(np.where(bad, 0.0, x))).astype(np.int64).astype(np.int16)


def cdiv(a: int, b: int) -> int:
    """C integer division, which truncates toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def roundf(x) -> np.ndarray:
    """C roundf (half away from zero), exact for float32 input."""
    x = np.asarray(x, dtype=np.float32)
    t = np.trunc(x)
    return (t + np.where(np.abs(x - t) >= F(0.5), np.sign(x), F(0))).astype(np.float32)


def per_value(values: np.ndarray, fn) -> np.ndarray:
    """Evaluate a scalar fn once per distinct value and map the results back (float32)."""
    uniq, inverse = np.unique(values, return_inverse=True)
    out = np.fromiter((fn(int(v)) for v in uniq), dtype=np.float32, count=len(uniq))
    return out[inverse].reshape(values.shape)
