"""Bit-exact port of glibc's x86-64 powf (the FMA variant glibc selects on CPUs with FMA).

The DJI Thermal SDK's H20T path is computed in float32 with libm powf, whose results differ
from numpy's float32 power and from a rounded double pow in rare cases. This reproduces it
exactly: the algorithm and tables are from ARM optimized-routines math/powf.c and
powf_log2_data.c / exp2f_data.c (Copyright (c) 2017-2026, Arm Limited;
SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception), which glibc ships, with
each a*b+c contracted to a fused multiply-add as GCC does when building glibc's FMA variant.
Supports finite non-negative x and finite y.
"""
import math
import struct

import numpy as np

_h = float.fromhex

_LOG2_TAB = [  # (invc, logc)
    (_h("0x1.661ec79f8f3bep+0"), -_h("0x1.efec65b963019p-2")), (_h("0x1.571ed4aaf883dp+0"), -_h("0x1.b0b6832d4fca4p-2")),
    (_h("0x1.49539f0f010bp+0"), -_h("0x1.7418b0a1fb77bp-2")), (_h("0x1.3c995b0b80385p+0"), -_h("0x1.39de91a6dcf7bp-2")),
    (_h("0x1.30d190c8864a5p+0"), -_h("0x1.01d9bf3f2b631p-2")), (_h("0x1.25e227b0b8eap+0"), -_h("0x1.97c1d1b3b7afp-3")),
    (_h("0x1.1bb4a4a1a343fp+0"), -_h("0x1.2f9e393af3c9fp-3")), (_h("0x1.12358f08ae5bap+0"), -_h("0x1.960cbbf788d5cp-4")),
    (_h("0x1.0953f419900a7p+0"), -_h("0x1.a6f9db6475fcep-5")), (_h("0x1p+0"), 0.0),
    (_h("0x1.e608cfd9a47acp-1"), _h("0x1.338ca9f24f53dp-4")), (_h("0x1.ca4b31f026aap-1"), _h("0x1.476a9543891bap-3")),
    (_h("0x1.b2036576afce6p-1"), _h("0x1.e840b4ac4e4d2p-3")), (_h("0x1.9c2d163a1aa2dp-1"), _h("0x1.40645f0c6651cp-2")),
    (_h("0x1.886e6037841edp-1"), _h("0x1.88e9c2c1b9ff8p-2")), (_h("0x1.767dcf5534862p-1"), _h("0x1.ce0a44eb17bccp-2")),
]
_A = (_h("0x1.27616c9496e0bp-2"), -_h("0x1.71969a075c67ap-2"), _h("0x1.ec70a6ca7baddp-2"), -_h("0x1.7154748bef6c8p-1"), _h("0x1.71547652ab82bp0"))
_EXP2_TAB = [
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
    0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
    0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
    0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
]
_C = (_h("0x1.c6af84b912394p-5"), _h("0x1.ebfce50fac4f3p-3"), _h("0x1.62e42ff0c52d6p-1"))
_SHIFT = _h("0x1.8p+52") / 32
_OFF = 0x3F330000
_fma = math.fma


def _f32bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _bits_f32(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def _d_bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _bits_d(u):
    return struct.unpack("<d", struct.pack("<Q", u & 0xFFFFFFFFFFFFFFFF))[0]


def powf(x: float, y: float) -> float:
    """powf(x, y) for float32-representable x > 0 (normal) and finite y; returns a float32 value."""
    ix = _f32bits(x)
    y = float(np.float32(y))
    if ix & 0x7FFFFFFF == 0:                     # +-0: x*x, or 1/(x*x) for negative y
        return math.inf if y < 0 else 0.0
    if ix >= 0x7F800000:
        raise ValueError("powf: only finite positive x supported")
    if ix < 0x00800000:                          # subnormal: normalise so the exponent goes negative
        ix = _f32bits(float(np.float32(x) * np.float32(2.0 ** 23))) & 0x7FFFFFFF
        ix = (ix - (23 << 23)) & 0xFFFFFFFF
    # log2_inline
    tmp = (ix - _OFF) & 0xFFFFFFFF
    i = (tmp >> (23 - 4)) % 16
    top = tmp & 0xFF800000
    iz = (ix - top) & 0xFFFFFFFF
    k = (top - (1 << 32) if top & 0x80000000 else top) >> 23
    invc, logc = _LOG2_TAB[i]
    z = _bits_f32(iz)
    r = _fma(z, invc, -1.0)
    y0 = logc + float(k)
    r2 = r * r
    yy = _fma(_A[0], r, _A[1])
    p = _fma(_A[2], r, _A[3])
    r4 = r2 * r2
    q = _fma(_A[4], r, y0)
    q = _fma(p, r2, q)
    logx = _fma(yy, r4, q)
    ylogx = y * logx
    if abs(ylogx) >= 126.0:                      # round-to-nearest branch of the reference code
        if ylogx <= -150.0:
            return 0.0
        if ylogx > _h("0x1.fffffffd1d571p+6"):
            return math.inf
        if ylogx > _h("0x1.fffffffa3aae2p+6"):
            return float(np.float32(_h("0x1.fffffep127")))
    # exp2_inline
    kd = ylogx + _SHIFT
    ki = _d_bits(kd)
    kd -= _SHIFT
    r = ylogx - kd
    t = _EXP2_TAB[ki % 32]
    t = (t + (ki << (52 - 5))) & 0xFFFFFFFFFFFFFFFF
    s = _bits_d(t)
    zz = _fma(_C[0], r, _C[1])
    r2 = r * r
    yy = _fma(_C[2], r, 1.0)
    yy = _fma(zz, r2, yy)
    return float(np.float32(yy * s))

