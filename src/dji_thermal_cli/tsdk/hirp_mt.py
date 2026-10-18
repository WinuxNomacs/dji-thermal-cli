"""The libMicroTA integer measurement core (MT_*_INT) that the hirp plugin drives.

Memory is modelled byte-for-byte: `h` is the 0x280-byte _MT_PARAM_INT handle, `c` the 0x3800-byte
calibration struct that follows it in the persistent buffer. C integer semantics are reproduced
explicitly (wrapping, truncating division, logical vs arithmetic shifts, float32/double rounding).
"""

from __future__ import annotations

import math
import struct

import numpy as np

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
F32 = np.float32


def s8(x):
    x &= 0xFF
    return x - 0x100 if x & 0x80 else x


def s16(x):
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def s32(x):
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


def s64(x):
    x &= M64
    return x - (1 << 64) if x >> 63 else x


def u64(x):
    return x & M64


def cdiv(a, b):
    """C truncating division."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def f2i_trunc(f, bits=64):
    """cvttsd2si/cvttss2si: truncate; NaN or out of range -> integer indefinite (INT_MIN)."""
    f = float(f)
    if math.isnan(f) or math.isinf(f) or not (-(2.0 ** (bits - 1)) <= f < 2.0 ** (bits - 1)):
        return -(1 << (bits - 1))
    return int(f)


def i2f32(n):
    """cvtsi2ss of a 64-bit integer: correctly rounded (no double rounding through float64)."""
    if abs(n) < (1 << 53):
        return F32(float(n))
    sign = -1 if n < 0 else 1
    a = abs(n)
    shift = a.bit_length() - 24
    q, r = divmod(a, 1 << shift)
    half = 1 << (shift - 1)
    if r > half or (r == half and q & 1):
        q += 1
    return F32(sign * float(q << shift))


class OutsideTable(ArithmeticError):
    """The native code would read outside a lookup table here (garbage or a crash). Internal:
    hirp turns it into a MeasureError, or ignores it where the native result goes unused."""


class Mem:
    __slots__ = ("b",)

    def __init__(self, size):
        self.b = bytearray(size)

    def i8(self, o):
        return s8(self.b[o])

    def u8(self, o):
        return self.b[o]

    def i16(self, o):
        return struct.unpack_from("<h", self.b, o)[0]

    def u16(self, o):
        return struct.unpack_from("<H", self.b, o)[0]

    def i32(self, o):
        return struct.unpack_from("<i", self.b, o)[0]

    def u32(self, o):
        return struct.unpack_from("<I", self.b, o)[0]

    def i64(self, o):
        return struct.unpack_from("<q", self.b, o)[0]

    def w8(self, o, v):
        self.b[o] = v & 0xFF

    def w16(self, o, v):
        struct.pack_into("<H", self.b, o, v & 0xFFFF)

    def w32(self, o, v):
        struct.pack_into("<I", self.b, o, v & M32)

    def w64(self, o, v):
        struct.pack_into("<Q", self.b, o, v & M64)


# ---------------------------------------------------------------- tables (MicroTA's constants)

class Tables:
    """libMicroTA lookup tables (see tables.py: mt_*), as Python lists for fast scalar indexing."""

    def __init__(self, t: dict):
        def lst(name):
            return [int(v) for v in t[name]]
        self.exp_frac = lst("mt_exp_frac")      # ResponseEXP: 16384 * exp(frac / 16384) mantissas
        self.sqrt = lst("mt_sqrt")              # SquareRootCal
        self.sqrt64_a = lst("mt_sqrt64_a")      # SquareRootCal_64
        self.sqrt64_b = lst("mt_sqrt64_b")
        self.inv_a = lst("mt_inv_a")            # INVERSE
        self.inv_b = lst("mt_inv_b")
        self.log = lst("mt_log")                # ResponseEXPInv
        self.atm_k = lst("mt_atm_k")            # CalcAtmTrsmit
        self.atm_b = lst("mt_atm_b")
        self.atm_lim = lst("mt_atm_lim")


# ---------------------------------------------------------------- numeric primitives

def highbit(x):
    if x < 0:
        return -19, 0
    return 0, (x.bit_length() - 1 if x > 0 else 0)


def sqrt_cal(T, x):
    """MT_SquareRootCal_INT(ulong x) -> (rc, value)."""
    x = u64(x)
    if s64(x) < 0:
        return -19, None
    _, hb = highbit(x)
    d6 = 2.0 ** (hb + 1)
    y = u64(x << 12)
    d7 = float(y >> 1) * 2.0 if y >> 63 else float(y)
    idx = ((f2i_trunc(d7 / d6) + 0x3FF800) & 0x1FFFF) >> 1
    if idx >= len(T.sqrt):
        raise OutsideTable(f"MT_SquareRootCal_INT table index {idx}")
    if (hb + 1) & 1 == 0:
        v = 2.0 ** ((hb + 1) >> 1) * float(T.sqrt[idx]) * 0.0001220703125
    else:
        v = 2.0 ** (hb >> 1) * 11585.0 * 0.0001220703125 * float(T.sqrt[idx]) * 0.0001220703125
    return 0, f2i_trunc(v)


def sqrt_cal64(T, x):
    """MT_SquareRootCal_64_INT(long x) -> (rc, value)."""
    if x < 0:
        return -19, None
    _, hb = highbit(x)
    d10 = 2.0 ** (hb + 1)
    l5 = f2i_trunc(float(s64(x << 20)) / d10)
    u2 = ((l5 - 0x7F000) >> 12) & 0xFF
    i6 = u2 - 1
    if not 0 <= i6 < len(T.sqrt64_a):
        raise OutsideTable(f"MT_SquareRootCal_64_INT table index {i6}")
    l4 = ((l5 - s32(u2 * 0x1000 + 0x7F800)) * T.sqrt64_b[i6] >> 17) + T.sqrt64_a[i6] * 8
    l1 = l4 + 1 + l4 * l4
    if l1 + 2 + l4 * 2 < l5 * 0x400000:
        l4 += 2
    elif l1 < l5 * 0x400000:
        l4 += 1
    if (hb + 1) & 1 == 0:
        v = 2.0 ** ((hb + 1) >> 1) * float(l4) * 3.814697265625e-06 * 0.125
    else:
        v = float(l4) * 2.0 ** (hb >> 1) * 46341.0 * 3.0517578125e-05 * 3.814697265625e-06 * 0.125
    return 0, f2i_trunc(v)


def inverse(T, a, b):
    """MT_INVERSE_INT_INT(a, b) ~ a / b (scaled)."""
    sign = 1 if ((a >= 0 or b < 1) and (b >= 0 or a < 1)) else -1
    l4 = abs(b)
    l10 = l4 * 0x8000
    if l4 < 2:
        b9 = 0x1E
    else:
        hb = l4.bit_length() - 1
        b9 = hb + 0x1E
        l10 >>= hb
    if l10 < 0x8000:
        raise ZeroDivisionError("MT_INVERSE_INT_INT reads out of bounds for b == 0")
    u3 = (l10 - 0x8000) >> 7
    lv4 = u3 * 0x80 + 0x8040
    lv4 = T.inv_a[u3] + (T.inv_b[u3] * (lv4 - l10) >> 15)
    r = (s64((s64(0x80000000 - l10 * lv4) * lv4) >> 15) * abs(a)) >> (b9 & 0x3F)
    return 0, s64(s64(r) * sign)


def response_exp(T, p, t, flag):
    """MT_ResponseEXP_INT: gray response of a black body. p = (R, B, O, F) ints."""
    if t <= 0:
        return -19, None
    l2 = cdiv(p[1] * 1000, t)
    k = l2 >> 14
    frac = l2 - (k << 14)
    e = F32(math.exp(k)) if k < 89 else F32(np.inf)
    v = f2i_trunc(F32(F32(e * F32(T.exp_frac[frac])) * F32(0.015625)))
    v = v - (p[3] if flag else 0x100)
    if v == 0:
        return -19, None
    o = p[2]
    return 0, s32(((o + 0xFF if o < 0 else o) >> 8) + s32(cdiv(p[0], v)))


def response_exp_inv(T, p, g):
    """MT_ResponseEXPInv_INT: black-body temperature of a (compensated) gray."""
    i1 = p[2]
    if not i1 < s32(g * 0x100):
        return -19, None
    l6 = s32(g * 0x100 - i1)
    l3 = s32(p[3] * g) + s32(p[0] - (s32(i1 * p[3]) >> 8))
    if not (0 < l3 and l6 <= l3):
        return -19, None
    hb3 = l3.bit_length() - 1
    hb6 = l6.bit_length() - 1
    i_a = ((u64(f2i_trunc(float(l3 * 0x4000) / 2.0 ** hb3)) - 0x4000) & 0xFFFFFFFFF) >> 4
    i_b = ((u64(f2i_trunc(float(s64(l6 << 14)) / 2.0 ** hb6)) - 0x4000) & 0xFFFFFFFFF) >> 4
    v = s32((((hb3 - hb6) & 0xFF) * 0x2C5C + T.log[i_a] - T.log[i_b]) & M32) >> 2
    if v == 0:
        return -19, None
    rc, r = inverse(T, s32(p[1] << 6), v)
    return rc, s32(r)


def calc_atm_trsmit(T, env, cal):
    """MT_CalcAtmTrsmit_INT(out, env, cal). env = (ambient/50, humidity, distance, ...)."""
    amb, hum, dist = env[0], env[1] & 0xFFFF, env[2]
    l9 = cal[1] * s32(dist - cal[2])
    l10 = cal[0] * s32(dist - cal[2])
    l1 = s64(l10 * 0x800000)
    if l1 < 0x186A1:
        l10, l9 = -l10, -l9
    l11 = amb
    if l9 > 20000000:
        l9 = 20000000
    u4 = u64(l11 * l11 * 0x200) // 0x9C4
    a50 = cdiv(l11, 0x32)
    l5 = s64(u4 * a50)
    poly = ((s64(a50 * l5 * 0x2988) >> 32) + (cdiv(l11 * 0x5756, 0x32) >> 7) + 0x98D
            + (s64(u4 * 0x138FB) >> 0x17) + (s64(l5 * 0x2B13) >> 0x1A))
    l5 = s64(cdiv(s64(u64(poly * hum) * u64(s64(l10 * 0x3FE) >> 10)), 0x1312D) * -0x826) >> 0x11
    l6 = s64(l5 * l5 * l5)
    l8 = s64((l6 >> 32) * l5)
    i = 0
    lim = 3
    while True:
        if l9 * 0x10 <= (lim * 100000) & M32:
            k7, i3 = T.atm_b[i], T.atm_k[i]
            break
        i += 1
        if i == 10:
            i3, k7 = 0x27DF, -0x2FD
            break
        lim = T.atm_lim[i]
    l10 = cdiv(l10 * -0x9A, 100000)
    l12 = s64(l10 * l10 * l10)
    l11 = s64(l12 * l10)
    a = s32(s32(l10) + 0x4000 + s32(s64(l10 * l10) >> 0xF) + s32(cdiv(l12 >> 0x1D, 3))
            + s32(cdiv(l11 >> 0x2D, 3)) + s32(cdiv(s64(l10 * l11) >> 0x3B, 0xF))) & M32
    b = s64(l5 + 0x10000 + (s64(l5 * l5) >> 0x11) + cdiv(l6 >> 0x21, 3) + cdiv(l8 >> 0x13, 3)
            + cdiv(s64(l5 * l8) >> 0x23, 0xF))
    c = s32(i3 + s32(cdiv((s64(l9 * 0x1FD1) >> 0xD) * k7, 100000) >> 6)) & M32
    u4 = u64(a * (u64(b * c) >> 0x10)) >> 0xE
    if u4 == 0 or u4 > 0x4000:
        return -305, None
    if l1 < 0x186A1:
        u4 = (u64(u4 * -0x18 + 0x60000) >> 4) + 0x4000
    return 0, s16(u4)


# ---------------------------------------------------------------- the MT handle

HM_TM640 = b"HM-TM640-17"


class MT:
    """Mirror of a libMicroTA _MT_PARAM_INT handle plus its calibration buffer."""

    def __init__(self, T: Tables, width=640, height=512):
        self.T = T
        self.h = Mem(0x280)
        self.c = Mem(0x3800)
        self.scratch = Mem(0x80)
        self.ub = False
        h = self.h
        h.w16(8, width)
        h.w16(10, height)
        self._set_default_param()
        self._restart()

    # -- creation
    def _set_default_param(self):
        h = self.h
        h.w32(0xC, 0x84032BE)
        h.w64(0xE4, 0x1E000003E8)
        h.w16(0xF0, 0x4000)
        h.w64(0x64, 0x3E800004E20)
        h.w64(0x44, 0x200000002000)
        h.w16(0x12C, 0x3CA)
        h.w64(0x10, 0x37F000000406)
        h.w8(1, 0)
        h.w64(0x28, 0x100000001)
        h.w32(0xF4, 2)
        h.w64(0x104, 0x4E2000004E2)
        h.w64(0x10C, 0x4E2000004E2)
        h.w32(0xEC, 10000)
        h.w32(0xCC, 0x2000)
        h.w64(0xD4, 0x4E2)
        h.w32(0xDC, 20000)
        h.w16(0xE0, 0x32)
        h.w32(0x70, 20000)
        h.w16(0x60, 0x3CA)
        h.w64(0x78, 0x4000)
        h.w64(0x80, 0)
        h.w32(0x5C, 0)
        h.w32(0x18, 1)
        h.w32(0x50, 0)
        h.w64(0x30, 0x20)
        h.w32(0x38, 0)
        h.w8(3, 6)
        h.w16(0x150, 1)
        h.w8(0x180, 0)
        h.w64(0x184, 25000)
        h.w8(0x18C, 0)
        h.w32(0x130, 2000)
        h.w32(0x114, 0x4E2)
        h.w8(0x118, 0)
        h.w32(0x128, 0xA00)
        h.w16(0x18F, 0x101)
        h.w16(0x200, 0x3C)
        h.w32(0x1FC, 0)
        h.w64(0x204, 0)
        h.w32(0x11C, h.u16(8) >> 1)
        h.w32(0x120, h.u16(10) >> 1)
        c = self.c
        c.b[:0x37F0] = bytes(0x37F0)
        c.w64(0, 0x37F000000404)
        c.w32(8, 0x20)
        c.w64(80, 0xFFFFD800FFFFD800)
        c.w64(100, 0x28A000000B400)

    def _restart(self):
        h = self.h
        h.w8(0, 2)
        h.w32(0xD8, 0)
        h.w32(0x5C, 0)
        h.w64(0x34, 0)
        h.w32(0x3C, 0)

    # -- configuration (MT_SetConfig_INT)
    def set_calib(self, data: bytes) -> int:
        if len(data) > 0x3800:
            return -13
        ver, size = struct.unpack_from("<II", data, 0)
        self.c.w32(0x2B10, 1)
        if ver > self.h.u32(0x10):
            return -303
        self.c.b[:size] = data[:size]
        return 0

    def is_hm640(self):
        return bytes(self.c.b[0x10:0x10 + 12]) == HM_TM640 + b"\0"

    def check_param(self, pid, v):
        u = v & M32
        ok = {
            5: lambda: (u + 0x222000) & M32 < 0x2EA001,
            6: lambda: u < 0xC8001,
            0xD: lambda: u < 5,
            0xE: lambda: u < 2,
            0xF: lambda: (u - 0x10) & M32 < 0xA3,
            0x14: lambda: u < 0x2001,
            0x15: lambda: (u + 0x50000) & M32 < 0xF0001,
            0x1D: lambda: s32(u) >= 0,
            0x2D: lambda: u < 2,
        }
        if pid not in ok:
            raise NotImplementedError(f"MT param id {pid:#x}")
        return 0 if ok[pid]() else -19

    def set_single(self, pid, v) -> int:
        rc = self.check_param(pid, v)
        if rc:
            return rc
        h = self.h

        def q(div):
            return cdiv(s32((v * 10000) >> 0xD) + 5, div)

        if pid == 5:
            h.w32(0xDC, q(200))
        elif pid == 6:
            h.w16(0xE0, q(10000))
        elif pid == 0xD:
            h.w8(1, v)
        elif pid == 0xE:
            h.w32(0x18, v)
        elif pid == 0xF:
            h.w32(0x30, v)
        elif pid == 0x14:
            h.w32(0x68, q(10))
        elif pid == 0x15:
            h.w32(0x70, q(10))
        elif pid == 0x1D:
            h.w32(0x1C, v)
        elif pid == 0x2D:
            h.w8(0x151, v)
        return 0

    def set_addl(self, addl: bytes) -> int:
        h = self.h
        width = h.u16(8)
        if len(addl) < width * 4:
            return -13
        ver = h.u32(0x30) & 0xFF
        row2 = addl[(width >> 1) * 4:(width >> 1) * 4 + 88]
        if ver == 0xB2:
            raise NotImplementedError("addline layout for calib version 0xb2")
        rc = self._set_addl1(ver, struct.unpack_from("<28H", addl, 0))
        if rc:
            return rc
        h.w32(0x5C, s32(h.i32(0x10C) - h.i32(0xDC)))       # MT_SetADDL2Param_INT
        h.w32(0x1FC, struct.unpack_from("<I", row2, 0x4C)[0])
        rc = self.calc_real_respns_coeff()
        if rc:
            return rc
        rc = self.calc_base_gray()
        if rc:
            return rc
        env = self.handle_env()
        rc = self.set_atm_trsmit(env)
        self.store_handle_env(env)
        return rc

    def _set_addl1(self, ver, w):
        h, c = self.h, self.c
        gain, off = c.i32(0x2B10), c.i32(0x2B14)
        g16, o16 = s16(gain), s16(off)

        def fold(x):
            return x if x < 0x2000 else (0x2000 - x) & 0xFFFF

        p0 = w[0] & 0x3FFF
        if p0 > 0x1FFF:
            p0 = (0x2000 - p0) & 0xFFFF
        u2 = fold(w[1] & 0x3FFF)
        i6 = s16(p0 * g16 + o16)
        h.w32(0x104, i6)
        s3 = s16(u2 * g16 + o16)
        if ver == 0x30:
            if bytes(c.b[0x10:0x20]) != b"DS-2TD2136DT-15\0":
                i6 = s3
        else:
            i6 = s3
        h.w32(0x10C, i6)
        h.w32(0x110, s16(fold(w[2] & 0x3FFF)) * gain + off)
        m = w[5] & 0x3FFF
        if m >= 5:
            return -19
        h.w8(0xF4, m)
        m = w[6] & 0x3FFF
        if m >= 2:
            return -19
        h.w8(0xF6, m)
        m = w[7] & 0x3FFF
        if m >= 2:
            return -19
        h.w8(0xF7, m)
        h.w16(0xFA, w[9] & 0x3FFF)
        h.w16(0xF8, w[8] & 0x3FFF)
        h.w32(0xFC, w[11] & 0x3FFF)
        h.w32(0x100, w[12] & 0x3FFF)
        m = w[13] & 0x3FFF
        if m >= 5:
            return -19
        h.w8(0xF5, m)
        u2 = fold(w[20] & 0x3FFF)
        p13 = w[27] & 0x3FFF
        if p13 > 0x1FFF:
            p13 = (0x2000 - p13) & 0xFFFF
        h.w32(0x114, s32(s16(u2) * gain + off))
        h.w32(0x108, s32(s16(p13) * gain + off))
        return 0

    # -- helpers
    def resp_params(self):
        h = self.h
        return (h.i32(0x88), h.i32(0x8C), h.i32(0x90), h.i32(0x94))

    def resp_flag(self):
        return self.c.u32(0) > 0x402

    def handle_env(self):
        h = self.h
        return [h.i32(0xDC), h.u16(0xE0), h.i32(0xE4), h.i32(0xE8), h.i32(0xEC), h.i16(0xF0)]

    def store_handle_env(self, env):
        self.h.w16(0xF0, env[5])

    def slot(self):
        return self.h.u8(1) * 5 + self.h.u8(0xF5)

    # -- MT_CalcRealRespnsCoeff_INT
    def calc_real_respns_coeff(self):
        h, c = self.h, self.c
        f5, f4, f1 = h.u8(0xF5), h.u8(0xF4), h.u8(1)

        def C(i):
            return c.u32(4 * i)

        u14 = C(f4 + f1 * 0x19 + f5 * 5 + 0x9B)
        p12 = f4 * 0x11 + f5 * 0x55 + 0x195
        i1 = f1 * 0xE1 + f4 * 9 + f5 * 0x2D + 0x33E
        u1 = C(i1)
        l13 = s32(C(i1 + 1))
        u2 = C(0)
        if (C(2) - 0x50) & M32 < 8:
            u8_ = (h.i32(0x10C) - u14) & M32
            cb = 4 * ((f4 + f5 * 5) * 0x10 + 0x934)
            A, B, Cc, D = c.i64(cb), c.i64(cb + 8), c.i64(cb + 16), c.i64(cb + 24)
            sv = s32(u8_)
            if abs(sv) < 0x2EF or sv == -(1 << 31):
                lv = s32(u8_ * 2)
                l7 = cdiv(s64(A * lv), 1000000) + cdiv(s64(B * lv * lv), 1000000000000)
                l9 = cdiv(s64(D * lv * lv), 1000000000000) + cdiv(s64(lv * Cc), 1000000)
            else:
                lv = cdiv(sv, 10)
                l7 = cdiv(s64(A * lv), 50000) + cdiv(s64(B * lv * lv), 2500000000)
                l9 = cdiv(s64(D * lv * lv), 2500000000) + cdiv(s64(lv * Cc), 50000)
            l6 = s64(l7 + 0x100000000)
            if not u64(l7 + 0xFD70A3D7) < 0x4FD70A3D9:
                return -309
            if u64(l9 + 0x1FF) < 0x3FF:
                l9 = 0
        else:
            l6, l9 = 0x100000000, 0
        h.w64(0x78, l6)
        h.w64(0x80, l9)
        P = [s32(C(p12 + k)) for k in range(4)]
        l58 = s32((P[0] * l13) >> 0x14)
        l54 = P[1]
        l50 = s32((s32(u1) * 0x1000 + l13 * P[2]) >> 0x14)
        l4c = P[3]
        if h.u8(0x118) == 0:
            h.w32(0x54, u14)
        else:
            u14 = h.u32(0x54)
        rc, l5c = response_exp(self.T, (l58, l54, l50, l4c), s32(u14 * 0x14 + 0x42AFE), u2 > 0x402)
        if rc == 0 and (l5c & M32) < 0x4000:
            h.w32(0x40, l5c)
            ub = c.u32(0x2B18)
            if ub < 10:
                ub = 10
            u10 = ((h.i32(0x68) * 1000) & M32) // ub
            i4 = s32(cdiv(s64(u64(s32(l50 + l5c * -0x100) * u10)), 1000))
            h.w32(0x90, i4)
            h.w32(0x8C, l54)
            h.w32(0x4C, s32(i4 + 0x100) >> 8)
            h.w32(0x94, l4c)
            h.w32(0x88, s32(cdiv(s64(u64(l58 * u10)), 1000)))
            return 0
        return -309

    # -- MT_CalcBaseGray_INT and its compensations
    def calc_base_gray(self):
        h, c, T = self.h, self.c, self.T
        f5, f4, f1 = h.u8(0xF5), h.u8(0xF4), h.u8(1)

        def C(i):
            return c.u32(4 * i)

        i14 = h.i32(0x10C)
        u5 = C(0)
        l5 = f5 * 5
        i15 = h.i32(0x108)
        pb = (f4 + 0xF4 + l5) * 8 + 3
        P1 = [C(pb + k) for k in range(8)]
        lv = f1 * 0x19 + l5 + f4
        u6, u7, u8_ = C(lv + 0x118), C(lv + 0x9B), C(lv + 0xAF7)
        flag = u5 > 0x402
        rc, l44 = response_exp(T, self.resp_params(), s32(i14 * 0x14 + 0x42AFE), flag)
        if rc:
            return -307
        h78, h80 = h.i64(0x78), h.i64(0x80)
        l44 = s32(u64(s64((h80 * l44 + h78) * l44)) >> 32)
        t = s32(((i14 - u7) & M32) * P1[4])
        t = (cdiv(t, 0x32) + P1[0]) & M32
        t = s32(t * (l44 & M32)) >> 0xE
        u13 = (h.i32(0x40) - t) & M32
        if u5 < 0x405:
            i10 = s32(u7 - u6)
            rc, l50 = hep_hzsk(h.i32(0x5C), i10, P1)
            if rc:
                return rc
        else:
            i10 = s32(u7 - u8_)
            rc, l50 = self.hep_hy(s32(i14 - i15), i10, P1)
            if rc:
                return rc
            l50 = s32(u64(s64(l50 * h78) + s64(s32(l50 * l50) * h80)) >> 32)
        h.w32(0xD8, P1[6])
        rc, l4c = self.zero_set()
        if rc:
            return rc
        i14 = s32(u7 - cdiv(i10, 2))
        if h.u8(4) == 0:
            i15 = cdiv(h.i32(0x70), 0x14)
        else:
            i15 = s32(h.i32(0x10C) - h.u16(0x74) // 0x14)
            if (i15 * 0x14 + 40000) & M32 > 120000:
                return -307
        rc, l48 = compensate_out_opt(T, i14, c.u16(0x2B18), i15, h.u16(0x68), u7, self.resp_params(), flag)
        if rc:
            return -307
        l48 = s32(u64(s64((h80 * l48 + h78) * l48)) >> 32)
        i14 = h.i32(0x1FC)
        mode = h.u8(0x1F8)
        if mode == 0:
            u13 = (u13 - l50 - l4c + l48 - i14) & M32
        else:
            raise NotImplementedError("MT base gray mode %d" % mode)
        if u13 > 0x3FFF:
            return -307
        h.w32(0x44, u13)
        h.w32(0x1FC, i14)
        h.w32(0x34, l50)
        h.w32(0x38, l4c)
        h.w32(0x3C, l48)
        return 0

    def hep_hy(self, a, b, P):
        h, c = self.h, self.c
        i2, i5 = s32(P[2]), s32(P[3])
        if h.u8(0xF5) == 0 and h.u8(0xF4) == 1 and bytes(c.b[0x10:0x1E]) == b"DS-2TP23-10VF\0":
            i5, i2 = -0xC000, 0x22000
        d = s32(a - b)
        v = s32(cdiv(s32(i2 * d), 0x32) + cdiv(s32(s32(i5 * d) * d), 0x9C4)) >> 0xE
        if u64(v + 1000) > 2000:
            return -306, 0
        return 0, v

    def zero_set(self):
        h = self.h
        l4 = h.i32(0xD8)
        i2 = s32(h.i32(0x10C) - h.i32(0x114))
        if u64(l4 * 10000 + 0x4000) > 0x8000 and 500 < (i2 + 0xFA) & M32:
            return -308, None
        if h.u8(0x151) == 1:
            return 0, 0
        l3 = cdiv(i2 * l4, 0x32) >> 0xE
        if l3 >= 0x259:
            return 0, 600
        return 0, (l3 if l3 > -0x259 else -600)

    # -- atmosphere
    def _dist_comp(self, env, lo, hi):
        h, c = self.h, self.c
        idx = self.slot()
        p = [c.i32(0x2DDC + idx * 0x10 + 4 * k) for k in range(4)]
        l = s32(env[2] - c.i32(0x21AC + idx * 0x20 + 8))
        v = s64(cdiv(s64(cdiv(s64(s64(p[2] * l) * l), 1000) * 0x2000), 10000000) + 0x800000
                + cdiv(s64(s64(p[1] * l) * 0x2000), 100000)
                + cdiv(s64(cdiv(s64(cdiv(s64(s64(p[3] * l) * l), 1000) * l), 10000) * 0x2000), 100000000))
        if v > hi:
            v = hi
        if v < lo:
            v = lo
        amb = h.i32(0xDC)
        h.w64(0x198, v)
        l9 = cdiv(s32(p[0] * amb) >> 2, 0x32)
        l5 = (1 << 46) // v
        h.w64(0x1A8, l5)
        h.w64(0x1A0, l9 - cdiv(s64(s64(v * p[0]) * amb) >> 0x19, 0x32))
        h.w64(0x1B0, l9 - cdiv(s64(s64(p[0] * l5) * amb) >> 0x19, 0x32))

    def _use_dist_comp(self):
        idx = self.slot()
        return self.h.u8(0x18F) == 1 and self.c.i32(0x2DDC + idx * 0x10 + 4) != 0

    def _atm_cal(self):
        o = 0x21AC + self.slot() * 0x20
        return [self.c.i32(o + 4 * k) for k in range(3)]

    def set_atm_trsmit(self, env):
        """MT_SetAtmTrsmit_INT: humidity/ambient-dependent coefficient times the distance transmission."""
        h = self.h
        amb = h.i32(0xDC)
        tab = [0] * 26
        tab[6:11] = [0, 0x14, 0x32, 0x50, 100]
        lo = [0x41AF, 0x4102, 0x4000, 0x3F54, 0x3EE1]
        mid = [0x423F, 0x4159, 0x4000, 0x3EFD, 0x3E51]
        hi = [0x42CF, 0x41AF, 0x4000, 0x3E2E, 0x3CF8]
        tab[11:16], tab[16:21], tab[21:26] = lo, mid, hi
        if amb < 0x4E3:
            tab[1:6] = lo
        elif amb < 0x7D1:
            tab[1:6] = [s16(cdiv((m - l) * (amb - 0x4E2), 0x2EE) + l) for l, m in zip(lo, mid)]
        elif amb < 0x9C5:
            tab[1:6] = [s16(cdiv((x - m) * (amb - 2000), 500) + m) for m, x in zip(mid, hi)]
        else:
            tab[1:6] = hi
        hum = h.u16(0xE0)
        r15 = 0
        prev = 0
        k = 1
        done = False
        while not done:
            bp = tab[k + 6]
            if hum <= bp:
                r15 = tab[k] + cdiv((tab[k + 1] - tab[k]) * (hum - prev), bp - prev)
            while True:
                if k == 5:
                    done = True
                    break
                prev = tab[k + 6]
                k += 1
                if not hum < prev:
                    break
        if self._use_dist_comp():
            env[5] = 0x4000
            self._dist_comp(env, 0x59999A, 0x8CCCCC)
            env[5] = s16(r15)
            return 0
        rc, o = calc_atm_trsmit(self.T, env, self._atm_cal())
        if rc == 0:
            env[5] = s16(s32(((r15 & 0xFFFF) * (o & M32)) & M32) >> 0xE)
        return rc

    def set_atm_trsmit_dj20(self, env):
        if self._use_dist_comp():
            env[5] = 0x4000
            self._dist_comp(env, 0xCCCCD, 0x851EB8)
            return 0
        rc, o = calc_atm_trsmit(self.T, env, self._atm_cal())
        if rc == 0:
            env[5] = o
        return rc

    def preprocess(self, dj20):
        rc = self.calc_real_respns_coeff()
        if rc:
            return rc
        m = self.h.u8(0x118)
        if m == 0:
            rc = self.calc_base_gray()
            if rc:
                return rc
        elif m == 1:
            raise NotImplementedError("MT_SetBaseGray_INT")
        env = self.handle_env()
        rc = self.set_atm_trsmit_dj20(env) if dj20 else self.set_atm_trsmit(env)
        self.store_handle_env(env)
        return rc

    # -- per-point
    def pre_ranhe(self, pt, trans):
        h = self.h
        ems, refl = pt[5] & 0xFFFF, pt[6]
        h.w64(0x1C0, 0xF85)
        l5 = cdiv(s32(h.i32(0xDC) << 0xC), 0x32) + 0x111266
        u6 = u64(l5 * l5) >> 0xC
        l4 = s64(((ems * 0x1000) // 1000) * (trans & 0xFFFF)) >> 0xC
        h.w64(0x1B8, l4)
        l5 = s32(cdiv(refl << 0xC, 1000)) + 0x111266
        u7 = u64(u6 * u6) >> 0xC
        u6 = u64(l5 * l5) >> 0xC
        t = s64(u64(u64(((trans & 0xFFFF) - l4) * (u64(u6 * u6) >> 0xC)) + u64(u7 * s32(0x4000 - (trans & 0xFFFF)))))
        h.w64(0x1E0, u64((u64(u7 * 0x7B) >> 0xC) - (t >> 0xE)))
        h.w32(0x1E8, cdiv(1 << 26, l4))

    def bbtemp2objtemp(self, bb):
        h, T = self.h, self.T
        u4 = u64(s64((bb + 0x111266) * (bb + 0x111266))) >> 0xC
        u4 = u64((u64(u64(u4 * u4) >> 0xC) * h.i64(0x1C0) & M64) >> 0xC) + h.i64(0x1E0)
        u4 = u64(u4)
        if u4 & 0x1FFFFFFFFFFFFF == 0:
            return -19, None
        u2 = cdiv(1 << 26, h.i64(0x1B8)) & M32
        rc, r = sqrt_cal(T, u64(u4 * u2) >> 0xC)
        if rc:
            return rc, None
        rc, r = sqrt_cal(T, s64(r << 6))
        if rc:
            return rc, None
        return 0, s32(cdiv(s64(r * 0x40), 16)) - 0x11126

    def bbtemp2gray(self, t):
        h = self.h
        t = s32(t * 1000)
        rc, l24 = response_exp(self.T, self.resp_params(), ((t + 0xFF if t < 0 else t) >> 8) + 0x42AFE, self.resp_flag())
        if rc:
            return None
        l24 = s32(u64(s64((h.i64(0x80) * l24 + h.i64(0x78)) * l24)) >> 32)
        v = s32(l24 + h.i32(0x44))
        return 0 if v < 0 else min(v, 0x3FFF)

    def gray_range(self, pt, trans):
        h, c, T = self.h, self.c, self.T
        f5 = h.u8(0xF5)
        u2 = c.i32(0x64 + f5 * 4)
        i8 = c.i32(0x50 + f5 * 4)
        if u2 > 0x2EED9:
            u2 = 0x2EED9
        l5 = 0x3FFFFFFFFFFFC - cdiv(s64(h.i64(0x1E0) << 0xC), h.i64(0x1B8))
        l48 = 0
        rc, r = sqrt_cal(T, u64(s64(cdiv(l5, h.i64(0x1C0)) * h.i64(0x1B8))) >> 4)
        if rc == 0:
            l48 = r
        rc, r = sqrt_cal(T, s64(l48 << 8))
        if rc == 0:
            l48 = r
        u4 = u64(l48 * 0x40 - 0x111266) >> 4
        u6 = u2 & M32
        if u4 < u64(u2):
            u6 = u4 & M32
        tr = trans & 0xFFFF
        a = cdiv(s32(h.i32(0xDC) << 0xC), 0x32) + 0x111266
        l7 = s64(a * a) >> 0xC
        b = s32(cdiv(pt[6] << 0xC, 1000)) + 0x111266
        b = s64(b * b) >> 0xC
        l7 = s64(l7 * l7) >> 0xC
        e = s64(((((pt[5] & 0xFFFF) * 0x1000) // 1000) * tr)) >> 0xC
        l5 = (s64(l7 * 0x7B) >> 0xC) - (s64(u64(l7 * s32(0x4000 - tr)) + u64((tr - e) * (s64(b * b) >> 0xC)))
                                         >> 0xE)
        if l5 < 0:
            rc, r = sqrt_cal(T, cdiv(l5 * -0x1000, 0xF85))
            if rc == 0:
                l48 = r
            rc, r = sqrt_cal(T, s64(l48 << 6))
            if rc == 0:
                l48 = r
            l48 = s64(l48 << 6)
            i3 = s32(l48) - 0x111266
            if i3 < 0:
                i3 = s32(l48) - 0x111257
            if c.i32(0x50 + f5 * 4) < (i3 >> 4):
                i8 = i3 >> 4
        else:
            i8 = c.i32(0x50 + f5 * 4)
        return self.bbtemp2gray(i8), self.bbtemp2gray(s32(u6))

    def gray2temp(self, pt, hm640):
        h, c, T = self.h, self.c, self.T
        f1, f5 = h.u8(1), h.u8(0xF5)
        tmin = c.i32(0x50 + f5 * 4)
        tmax = c.i32(0x64 + f5 * 4)
        i8 = s32(pt[1] - h.i32(0x44))
        l10, l7 = h.i64(0x80), h.i64(0x78)
        if l10 >> 9 == 0:
            u9 = s32(s32(i8 * s32(f2i_trunc(70368744177664.0 / float(l7)))) >> 0xE)
        else:
            l10 = s64(l10 * 4 * i8 + (s64((l7 >> 5) * (l7 >> 5)) >> 0x16))
            if l10 >> 9 == 0:
                return -19, None
            rc, l48 = sqrt_cal64(T, s64(l10 * 0x100))
            if rc:
                return rc, None
            l48 = s64(l48 * 0x1000)
            l10 = f2i_trunc(4294967296.0 / float(h.i64(0x80)))
            u9 = s64(s64((l48 - h.i64(0x78)) * l10)) >> 0x21
        rc, l50 = response_exp_inv(T, self.resp_params(), s32(u9))
        if rc:
            return rc, None
        l50 = s32(l50 - 0x11126)
        status = 0
        if tmax < l50:
            status, l50 = -300, tmax
        elif l50 < tmin:
            status, l50 = -301, tmin
        l50 = s32(l50 << 4)
        rc, l4c = self.bbtemp2objtemp(l50)
        if rc:
            return rc, None
        if h.u8(0x18F) == 1 and c.i32(0x2DE0 + (f1 * 5 + f5) * 0x10) != 0:
            l4c = s32(s32(s64(l4c * h.i64(0x1A8)) >> 0x17) + s32(h.i64(0x1B0)))
        out = l4c >> 2
        if hm640 and (l4c >> 8) > 0x173:
            k = c.i16(0x2F88)
            if (l4c >> 8) < 0x1F5:
                out = s32((s32((out - 0x5D00) * k) >> 0xD) + out)
            else:
                out = s32(out + k)
        return status, out

    # -- MT_Process_INT (one point)
    def _setup(self, pt):
        """MT_Process_INT up to the per-point conversion: parameter scaling and checks, the
        calibration slot defaults, preprocessing and the atmospheric transmittance.
        Returns (rc, transmittance, hm640)."""
        h, c = self.h, self.c
        ems_q13, refl_q13, dist_q13 = pt[5], pt[6], pt[7]
        ems = (((s32((ems_q13 & 0xFFFF) * 10000 >> 0xD)) + 5) & M32) // 10 & 0xFFFF
        pt[5] = ems
        dist = cdiv(s32(((dist_q13 & M32) * 10000) >> 0xD) + 5, 10) & M32
        pt[7] = s32(dist)
        refl = cdiv(s32((refl_q13 * 10000) >> 0xD) + 5, 10)
        pt[6] = refl
        if ((ems - 10) & 0xFFFF) > 0x3DE or refl < -0x42AFE:
            return -19, 0, False
        hm640 = self.is_hm640()
        f5 = h.u8(0xF5)
        if hm640:
            dist = _hm640_distance(dist, f5)
            pt[7] = s32(dist)
        if c.u8(0x2F8A) == 1:
            idx = f5 + h.u8(1) * 2
            if c.i32(0xC + (idx + 0x2F8) * 0x10) == 0 and c.i32(0x2FAC + idx * 4) == 0:
                c.w32(0x21B0, 0x104)
                c.w32(0x2DEC, 0x19A)
                c.w32(0x2DF0, -0x2843)
                c.w32(0x2DF4, 0)
                c.w32(0x2DF8, 0)
            else:
                for o in (0x2DDC, 0x2DE0, 0x2DE4, 0x2DE8):
                    c.w32(o, c.u32(o + 0x1B0))
                for k in range(4):
                    c.w32(0x2DEC + 4 * k, c.u32(0x2F9C + 4 * k))
                c.w32(0x21B0, c.u32(0x2FAC))
                c.w32(0x21D0, c.u32(0x2FB0))
            o = h.u8(1) * 0xC + f5 * 6
            if c.i16(o + 0x2FDC) == 0 and c.i16(o + 0x2FDE) == 0 and c.i16(o + 0x2FE0) == 0:
                for k, v in enumerate((0x32, 0xFFF1, 0x14, 0, 0, 0)):
                    c.w16(0x2F7C + 2 * k, v)
            else:
                for k in range(6):
                    c.w16(0x2F7C + 2 * k, c.u16(0x2FDC + 2 * k))
            o = f5 * 0x14
            if any(c.i32(o + 0x2FB4 + 4 * k) for k in range(5)):
                for k in range(5):
                    c.w32(0x1ECC + 4 * k, c.u32(0x2FB4 + 4 * k))
            else:
                for k, v in enumerate((0x4000, 0x14000, 0x30000, 0x30000, 3)):
                    c.w32(0x1ECC + 4 * k, v)
                for k, v in enumerate((0x4333, 0x14000, 0x6000, 0x6000, -0x31)):
                    c.w32(0x1F6C + 4 * k, v)
        if h.u8(0x1F8) == 0 and c.u8(8) != h.u32(0x30):
            return -302, 0, False
        if h.u8(0xF6) == 1 or h.u8(0xF7) == 1:
            return 1, 0, False                    # nothing computed, MT_Process_INT returns 0
        rc = self.preprocess(dj20=not hm640)
        if rc:
            return rc, 0, False
        env = [h.i32(0xDC), h.u16(0xE0), -100000, h.i32(0xE8), h.i32(0xEC), 0]
        cal = 0xC + (h.u8(1) * 5 + 0x10D + f5) * 0x20
        if 1 < s32(c.i32(cal + 0x18) * 0x200):
            c.w32(cal + 4, 1)
        if env[2] != (pt[7] & M32) // 10:
            env[2] = (pt[7] & M32) // 10
            rc = self.set_atm_trsmit(env) if hm640 else self.set_atm_trsmit_dj20(env)
            if rc:
                return rc, 0, False
        return 0, env[5] & 0xFFFF, hm640

    def process(self, gray, ems_q13, refl_q13, dist_q13):
        """Mode 0: gray -> temperature. Returns (rc, point); point[0] status, point[4] result."""
        h = self.h
        pt = [0, gray, 0, 0, 0, ems_q13 & 0xFFFF, refl_q13, dist_q13]
        rc, trans, hm640 = self._setup(pt)
        if rc:
            return (0 if rc == 1 else rc), pt
        f5 = h.u8(0xF5)
        self.pre_ranhe(pt, trans)
        try:
            lo, hi = self.gray_range(pt, trans)
        except OutsideTable:
            lo = hi = None          # only feeds the gray-limit outputs; native reads garbage here
            self.ub = True
        if lo is not None:
            h.w32(0x210, lo + 10)
        if hi is not None:
            h.w32(0x20C, hi - 20)
        st, out = self.gray2temp(pt, hm640)
        pt[4] = -0x4449 if st == -19 else (out if out is not None else pt[4])
        pt[0] = st
        if f5 == 2:
            v = pt[1] * 0xC3
            pt[4] = (v - 0xD1790 if v - 0xD1790 >= 0 else v - 0xD1691) >> 8
        return 0, pt

    def temp2gray_point(self, temp, ems_q13, refl_q13, dist_q13):
        """Mode 1: temperature (pt[4], 1/256 C) -> gray. Returns (rc, point); point[0] status,
        point[1] the gray (None where native leaves it uninitialised)."""
        pt = [0, 0, 0, 0, temp, ems_q13 & 0xFFFF, refl_q13, dist_q13]
        rc, trans, hm640 = self._setup(pt)
        if rc:
            return (0 if rc == 1 else rc), pt
        st, g = self.temp2gray(pt, trans, hm640)
        pt[1] = 0 if st == -19 else g
        pt[0] = st
        return 0, pt

    def temp2gray(self, pt, trans, hm640):
        """MT_Temp2Gray_DJ20_INT / MT_Temp2Gray_INT -> (status, gray or None)."""
        h, c = self.h, self.c
        t = pt[4]
        if t < -0x42AFC:
            return -19, None
        f5 = h.u8(0xF5)
        tmin, tmax = c.i32(0x50 + f5 * 4), c.i32(0x64 + f5 * 4)
        if c.u32(8) == 0xB2:
            tmax = s32(tmax + h.i32(0x10C) - c.i32((h.u8(0xF4) + h.u8(1) * 0x19 + f5 * 5 + 0x9B) * 4))
        if hm640 and t >= 0x17400:
            k = c.i16(0x2F88)
            d = k * 4
            if t <= d + 0x1F400:
                d = cdiv(s32((t - 0x17400) * k), k + 0x2000)
            t = s32(t - d)
        if h.u8(0x18F) == 1 and c.i32(0x2DE0 + (f5 + h.u8(1) * 5) * 0x10) != 0:
            t = s32(s32(s64(t * h.i64(0x198)) >> 0x17) + h.i32(0x1A0))
        rc, bb = self.objtemp2bbtemp(t, pt[5] & 0xFFFF, pt[6], trans)
        if rc:
            return rc, None
        status = -300 if not bb < tmax else (-301 if bb < tmin else 0)
        g = self.bbtemp2gray(bb)
        if g is None:
            return -1, None                      # MT_ResponseEXP_INT failed: gray left unset
        return status, g

    def objtemp2bbtemp(self, t, ems, refl, trans):
        """MT_Objtemp2BBtemp_INT -> (rc, black-body temperature)."""
        h, T = self.h, self.T
        e = ((ems * 0x1000) // 1000) & M64
        l10 = s32(s32((trans * e) & M32) >> 0xC)
        l8 = cdiv(s32(h.i32(0xDC) << 0xC), 0x32) + 0x111266
        l6 = s32(s32(t * 0x10) + 0x111266)
        l4 = s32(s32(cdiv(refl << 0xC, 1000)) + 0x111266)
        u7 = u64(l6 * l6) >> 0xC
        u5 = u64(l4 * l4) >> 0xC
        u9 = u64(l8 * l8) >> 0xC
        u7 = u64(u7 * u7)
        u9 = u64(u9 * u9) >> 0xC
        a = s64(u64(u64(s32(0x4000 - trans) * u9) + u64(u64(trans - l10) * (u64(u5 * u5) >> 0xC))))
        l4 = s64((a >> 0xE) - (u64(u9 * 0x7B) >> 0xC))
        if (u7 >> 0xC) < 0x3FFFFFFFFFFFC:
            u5 = u64((u7 >> 0xE) * l10) >> 0xC
        else:
            u5 = u64((u7 >> 0x1A) * l10)
        u9 = u64(l4 + u5)
        if u9 & 0x1FFFFFFFFFFFFF == 0:
            return -19, None
        rc, r = sqrt_cal(T, u64(u9 * 0x107E) >> 0xC)
        if rc:
            return rc, None
        rc, r = sqrt_cal(T, s64(r << 6))
        if rc:
            return rc, None
        v = s64(r * 0x40)
        return 0, s32(cdiv(v, 16)) - 0x11126


def _hm640_distance(d, f5):
    """Distance remap applied by MT_Process_INT for the HM-TM640-17 calibration (d in mm)."""
    state = None
    if (d - 15001) & M32 < 4999:
        d = (cdiv(s32(d - 20000), 5) + 16000) & M32
        state = "a7a"
    elif (d - 20000) & M32 > 19999:
        state = "a7a"
    else:
        if f5 == 0:
            d = (((d * 0x18 - 480000) & M32) // 0x14 + 16000) & M32
        state = "409b"
    if state == "a7a":
        if d < 40000:
            state = "409b"
        else:
            if f5 == 0:
                d = (((d * 0x26 - 0x173180) & M32) // 10 + 40000) & M32
            state = "aad"
    if state == "409b":
        state = "aad" if d > 19999 else None
    if state == "aad" and f5 == 1:
        d = ((((d + 0x7FFFB1E0) * 2) & M32) // 5 + 16000) & M32
    return d


def compensate_out_opt(T, a, b, c_, d, e, p, flag):
    """MT_CompensateOutOpt_INT."""
    b = min(b, 1000)
    if b < 10:
        b = 10
    d = min(d, 1000)
    u6 = d if d >= 10 else 10

    def p4(t):
        x = s32(t * 20 + 0x42AFE)
        q = cdiv(x * x, 1000)
        return u64(s64(q * q)) // 1000

    def root(v):
        f = np.sqrt(i2f32(s64(v)))
        f = np.sqrt(i2f32(s64(f2i_trunc(f) * 1000)))
        return f2i_trunc(f, 32)

    u5 = p4(e)
    ta = root(u64(p4(a) * s32(1000 - b)) + u64(b * u5))
    tb = root(u64(p4(c_) * s32(1000 - u6)) + u64(u5 * u6))
    rc, l48 = response_exp(T, p, ta, flag)
    if rc:
        return -307, None
    rc, l44 = response_exp(T, p, tb, flag)
    if rc:
        return -307, None
    return 0, cdiv(s32((l44 - l48) * 1000), u6)


def hep_hzsk(a, b, P):
    l6, l1, l3, l4, l5 = s32(P[1]), b, a, s32(P[2]), s32(P[3])
    l2 = l6 * 0x32
    if l2 < l1 * 0x4000:
        if l2 < l3 * 0x4000:
            r = cdiv(l5 * (l3 - l1), 0x32) >> 0xE
        else:
            r = (cdiv(l3 * l4 - l1 * l5, 0x32) >> 0xE) + (l6 * (l5 - l4) >> 0x1C)
    elif l2 < l3 * 0x4000:
        r = (cdiv(l3 * l5 - l1 * l4, 0x32) >> 0xE) + (l6 * (l4 - l5) >> 0x1C)
    else:
        r = cdiv(l4 * (l3 - l1), 0x32) >> 0xE
    if u64(r + 1000) < 0x7D1:
        return 0, s32(r)
    return -306, 0
