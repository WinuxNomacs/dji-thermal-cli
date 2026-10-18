"""The iirp plugin's image pipeline (libv_iirp process / process_strech, the "irproc" library).

dirp_process renders the raw frame as a pseudo-colour RGB image:

1. 5x5 range-weighted spatial noise reduction (weights by gain mode);
2. contrast enhancement, one of
   - y14_image_enhance: 3x3 gaussian detail split, clipped-histogram AGC with an optional
     linear stretch (the manual colour bar), and piecewise digital detail enhancement (DDE);
   - y14_image_enhance_300 (gain 0 and the v0300 headers): the same detail split and DDE around
     agc3, an adaptive-platform histogram equalisation;
3. palette lookup (256-entry YUV LUTs), optional isotherm, brightness, YUV -> RGB.

Integer widths, float32/double placement and unsigned 64-bit wraps follow libv_iirp exactly,
so the output is bit-identical to the native SDK. Errors carry the dirp return code (irproc
codes converted as irproc_error_code_convert does). Struct offsets in comments are those of the
native config structs (y14_image_enhance: 0x50 bytes; agc3: 0x168 bytes), which this module
builds byte for byte.
"""

from __future__ import annotations

import struct

import numpy as np

from .errors import IspError, UndefinedBehaviour

F = np.float32


def yuv_lut(t, pal):
    return t[f"isp_yuv_{pal}"].reshape(256, 3)


def map_to_yuv_pseudocolor(t, y14: np.ndarray, pal: int) -> np.ndarray:
    """y14_map_to_yuv_pseudocolor: LUT[y14 >> 6] (the LUT is a 256-entry stack copy)."""
    if f"isp_yuv_{pal}" not in t:
        raise IspError("color mode unsupported", -3)
    idx = y14.reshape(-1) >> 6
    if idx.max(initial=0) > 255:
        raise UndefinedBehaviour("pseudo-colour index past the LUT")
    return yuv_lut(t, pal)[idx]


def y14_to_y8(y14: np.ndarray) -> np.ndarray:
    v = np.minimum(y14.reshape(-1).astype(np.uint32), 0x3FFF)
    return (v * 0xFF // 0x3FFF).astype(np.uint8)


def brightness_adjust(yuv: np.ndarray, k: float) -> np.ndarray:
    """yuv_brightness_adjust (in place on Y): (ushort)(int)((float)y * k), capped at 255."""
    k = F(k)
    if k < F(0.5) or k > F(2.0):
        raise IspError("brightness out of range", -3)
    out = yuv.copy()
    y = (out[:, 0].astype(F) * k).astype(np.int32) & 0xFFFF
    out[:, 0] = np.minimum(y, 0xFF).astype(np.uint8)
    return out


def brightness_factor(b: int) -> float:
    f = F(b)
    return F(F(f / F(100.0)) + F(0.5)) if b < 0x32 else F(f / F(50.0))


def yuv444_to_rgb(yuv: np.ndarray) -> np.ndarray:
    y = yuv[:, 0].astype(np.float64)
    u = yuv[:, 1].astype(np.int32) - 0x80
    v = yuv[:, 2].astype(np.int32) - 0x80
    r = (v * 1.14 + y).astype(np.int32)
    g = ((y - u * 0.394) - v * 0.581).astype(np.int32)
    b = (u * 2.032 + y).astype(np.int32)

    def clamp(c):
        c = c.astype(np.int16)  # (short)(int)
        return np.where(c >= 0x100, 0xFF, np.where(c < 1, 0, c)).astype(np.uint8)

    return np.stack([clamp(r), clamp(g), clamp(b)], axis=1)


def cdiv(a, b):
    """C integer division (truncates toward zero), elementwise."""
    q = np.abs(a) // np.abs(b)
    return np.where((a < 0) != (b < 0), -q, q)


def snr_weights(t, gain):
    """checkout_snr_weight_table: [distance class 0..5][|diff| 0..15]."""
    return t["isp_snr_weights"].reshape(2, 6, 16)[0 if gain == 0 else 1].astype(np.int64)


def spatial_noise_reduction(t, raw: np.ndarray, w: int, h: int, gain: int) -> np.ndarray:
    """y14_image_spatial_noise_reduction: 5x5 range-weighted smoothing,
    out = c - sum(wt*(c-n)) / sum(wt) (C division), clamped to 0..0xffff."""
    tab = snr_weights(t, gain)
    img = raw.reshape(h, w).astype(np.int64)
    pad = np.pad(img, 2)
    valid = np.pad(np.ones((h, w), bool), 2)
    num = np.zeros((h, w), np.int64)
    den = np.zeros((h, w), np.int64)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            ay, ax = abs(dy), abs(dx)
            s = ay + ax
            cls = {0: 0, 1: 1, 3: 4, 4: 5}.get(s, 2 if (ay == 1 and ax == 1) else 3)
            n = pad[2 + dy:2 + dy + h, 2 + dx:2 + dx + w]
            m = valid[2 + dy:2 + dy + h, 2 + dx:2 + dx + w]
            d = img - n
            ad = np.abs(d)
            wt = np.where(ad < 16, tab[cls][np.minimum(ad, 15)], 0)
            wt = np.where(m, wt, 0)
            num += wt * d
            den += wt
    if (den == 0).any():
        raise UndefinedBehaviour("noise reduction divides by zero")
    out = img - cdiv(num, den)
    return np.clip(out, 0, 0xFFFF).astype(np.uint16).reshape(-1)


# ---- y14_image_enhance (classic: detail extraction + histogram AGC + DDE)
def gauss3_kernel(t):
    """get_gau_kernel(size 3): the integer kernel as float32, each divided by the float sum."""
    k = t["isp_gauss3"].astype(F)
    s = F(0)
    for v in k:
        s = F(v + s)
    return (k / s).astype(F)


def gaussian_filter3(t, img: np.ndarray, w: int, h: int) -> np.ndarray:
    """gaussian_filter(1.0, size 3): out-of-image taps take the centre value; float32 sum in
    kernel order; (short)(int) of the result."""
    k = gauss3_kernel(t)
    a = img.reshape(h, w)
    pad = np.pad(a, 1)
    inside = np.pad(np.ones((h, w), bool), 1)
    acc = np.zeros((h, w), F)
    i = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            n = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            m = inside[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            p = np.where(m, n, a).astype(F)
            acc = (p * k[i] + acc).astype(F)
            i += 1
    return acc.astype(np.int32).astype(np.int16).view(np.uint16).reshape(-1)


def extract_detail(t, img: np.ndarray, w: int, h: int) -> np.ndarray:
    """y14_image_extract_detail: int16 (img - blur)."""
    blur = gaussian_filter3(t, img, w, h)
    return (img.astype(np.int32) - blur.astype(np.int32)).astype(np.int16)


def _u32_trunc(x):
    """(uint)(long)(float x)."""
    return int(np.int64(np.trunc(float(x)))) & 0xFFFFFFFF


def auto_gain_control(img: np.ndarray, cfg: bytes) -> np.ndarray:
    """y14_image_auto_gain_control with the AGC struct cfg[0:0x24]."""
    f_hi, f_lo, f_clip_hi, f_clip_lo, gain_i, gain_f, base_f = struct.unpack_from("<ffffiff", cfg, 0)
    enable, _, lower, upper = struct.unpack_from("<BBHH", cfg, 0x1C)
    n = img.size
    if upper < lower:
        raise IspError("invalid parameter", -3)
    if img.max(initial=0) >= 0x4000:
        raise UndefinedBehaviour("AGC map index past 0x3fff")
    # get_histogram: values < 0x3fff
    hist = np.bincount(img[img < 0x3FFF], minlength=0x4000).astype(np.int64)[:0x4000]
    # get_limit
    nf = F(n)
    t_lo = _u32_trunc(F(nf * F(f_lo)))
    t_hi = (n - (_u32_trunc(F(nf * F(f_hi))))) & 0xFFFFFFFF
    cum = np.cumsum(hist) & 0xFFFFFFFF
    lo = hi = 0
    for i in range(0x4000):
        c = int(cum[i])
        if lo == 0 and t_lo < c:
            lo = i & 0xFFFF
        if hi == 0 and t_hi < c:
            hi = 0x3FFF if i == 0 else i - 1
            break
    if hi < lo:
        raise IspError("error source image", -4)
    # get_limited_acc_histogram
    up = _u32_trunc(F(nf * F(f_clip_hi)))
    dn = _u32_trunc(F(nf * F(f_clip_lo)))
    acc = np.zeros(0x4000, np.int64)
    s = 0
    for i in range(lo, hi):
        v = int(hist[i])
        s = (s + (up if up < v else dn if v < dn else v)) & 0xFFFFFFFF
        acc[i] = s
    acc[hi:] = s
    # get_map
    L = (gain_i * ((hi - lo) + 1)) & 0xFFFFFFFF
    if F(F(gain_f) * F(16383.0)) < F(L):
        L = _u32_trunc(F(F(gain_f) * F(16383.0)))
    if acc[0x3FFF] == 0:
        raise UndefinedBehaviour("AGC map left uninitialised")
    off = int(np.int64(np.trunc(float(F(F((0x3FFF - L) & 0xFFFFFFFF) * F(base_f))))))
    mp = ((off + (L * acc) // int(acc[0x3FFF])) & 0xFFFFFFFF).astype(np.int64)
    # stretch limits
    if lower < hi:
        if upper <= lo:
            lower, upper = lo, (lo + 1) & 0xFFFF
    else:
        lower, upper = (hi - 1) & 0xFFFF, hi
    lower, upper = int(mp[lower]) & 0xFFFF, int(mp[upper]) & 0xFFFF
    if enable != 1 or upper < lower:
        enable = 0
    out = (mp[img] & 0xFFFF).astype(np.int64)
    if enable == 1:                      # stretch_frame
        out = np.where(out <= lower, 0, np.where(out >= upper, 0x3FFF,
                       ((out - lower) * 0x3FFF) // max(upper - lower, 1)))
    return out.astype(np.uint16)


def dde_coefficients(d: np.ndarray, cfg: bytes) -> np.ndarray:
    """dde_coefficient_calculate: piecewise-linear gain over |detail| with knots t0..t3."""
    f0, f1, f2, t0, t1, t2, t3 = struct.unpack_from("<fffiiii", cfg, 0)
    f0, f1, f2 = F(f0), F(f1), F(f2)
    a = np.abs(d.astype(np.int64))

    def ramp(lo, hi, flo, fhi):  # ((a-lo)*fhi + flo*(hi-a)) / (hi-lo), float32 steps
        with np.errstate(divide="ignore", invalid="ignore"):
            return ((((a - lo).astype(F) * fhi).astype(F) + (flo * (hi - a).astype(F)).astype(F)).astype(F) / F(hi - lo)).astype(F)

    c = np.full(a.shape, f2, F)
    c = np.where((t2 < a) & (a <= t3), ramp(t2, t3, f1, f2), c)
    c = np.where((t1 < a) & (a <= t2), f1, c)
    c = np.where((t0 < a) & (a <= t1), ramp(t0, t1, f0, f1), c)
    return np.where(a <= t0, f0, c).astype(F)


def digital_detail_enhance(agc: np.ndarray, detail: np.ndarray, cfg: bytes) -> np.ndarray:
    """y14_image_digital_detail_enhance with the DDE struct cfg[0:0x1c]."""
    coef = dde_coefficients(detail, cfg)
    v = ((detail.astype(F) * coef).astype(F) + agc.astype(F)).astype(F)
    v = v.astype(np.int64)  # (int) truncation; values are small
    return np.clip(v, 0, 0x3FFF).astype(np.uint16)


def enhance(t, img: np.ndarray, w: int, h: int, cfg: bytes) -> np.ndarray:
    """y14_image_enhance(img, wh, out, cfg[0x50] by value)."""
    detail = extract_detail(t, img, w, h)
    agc = auto_gain_control(img, cfg[:0x24])
    return digital_detail_enhance(agc, detail, cfg[0x24:0x40])


# ---- y14_image_enhance_300: extract_detail + agc3_process (adaptive-platform histogram
# equalisation, libirprocess.c) + DDE with cfg[0x148:0x164]. agc3 works on one struct (the
# 0x168-byte config passed by value); offsets below are that struct's.
M64 = (1 << 64) - 1


def s64(v):
    v &= M64
    return v - (1 << 64) if v >> 63 else v


class Agc3:
    def __init__(self, cfg: bytes, w: int, h: int):
        self.b = bytearray(cfg[:0x168])
        self.w, self.h = w, h

    def g(self, fmt, off):
        return struct.unpack_from("<" + fmt, self.b, off)[0]

    def p(self, fmt, off, v):
        mask = {"B": 0xFF, "b": 0xFF, "H": 0xFFFF, "h": 0xFFFF, "I": 0xFFFFFFFF, "i": 0xFFFFFFFF, "Q": M64, "q": M64}[fmt]
        v &= mask
        if fmt in "bhiq" and v >> (mask.bit_length() - 1):
            v -= mask + 1
        struct.pack_into("<" + fmt, self.b, off, v)

    def u16(self, off):
        return self.g("H", off)

    # calculate_y14_image_blured_image_and_pix_average
    def blurred_and_average(self, img, detail):
        w = self.w
        y0, x0, y1, x1 = self.u16(0x3A), self.u16(0x3C), self.u16(0x3E), self.u16(0x40)
        a = img.reshape(self.h, w).astype(np.int32)
        d = detail.reshape(self.h, w).astype(np.int32)
        self.blur = np.zeros((self.h, w), np.uint16)  # malloc'd; only the ROI is written
        roi = ((a[y0:y1, x0:x1] - d[y0:y1, x0:x1]) & 0xFFFF).astype(np.uint16)
        self.blur[y0:y1, x0:x1] = roi
        total = int(roi.astype(np.uint64).sum())
        area = ((x1 - x0) * (y1 - y0)) & 0xFFFFFFFF or 1
        self.p("h", 0x48, total // area)
        self.detail = d

    # get_histogram_status = assign_weight_value_for_every_pixel + calculate_histogram_with_pix_weight_value
    def histogram(self):
        y0, x0, y1, x1 = self.u16(0x3A), self.u16(0x3C), self.u16(0x3E), self.u16(0x40)
        sy, sx = self.g("B", 0x38), self.g("B", 0x39)
        t = [self.g("B", 0x14 + i) for i in range(7)]
        wts = [self.g("B", 0x1B + i) for i in range(8)]
        if sy == 0 or sx == 0:
            raise UndefinedBehaviour("agc3 histogram step 0 (endless loop)")
        det = self.detail[y0:y1:sy, x0:x1:sx].astype(np.int16)
        u = np.abs(det.astype(np.int32)) & 0xFFFF  # (ushort)abs((short)d)
        conds = [u < t[0]] + [(t[k - 1] <= u) & (u < t[k]) for k in range(1, 7)]
        wmap = np.select(conds, wts[:7], wts[7]).astype(np.int64)
        bins = self.blur[y0:y1:sy, x0:x1:sx].astype(np.int64)
        if bins.size and bins.max() >= 0x4000:
            raise UndefinedBehaviour("agc3 histogram index past 0x3fff")
        hist = np.minimum(np.bincount(bins.reshape(-1), weights=wmap.reshape(-1), minlength=0x4000)[:0x4000].astype(np.int64), 0x3FFF)
        self.hist = hist  # @0x78 (u16)
        self.p("Q", 0xC0, int(hist.sum()))

    # update_size_and_demarcation_point_value
    def demarcation(self):
        total = self.g("Q", 0xC0)
        acc = np.cumsum(self.hist).astype(np.uint64)
        thr = [(self.u16(off) * total & M64) >> 16 for off in (0x6C, 0x6E, 0x70, 0x72, 0x74)]
        lo_thr, hi_thr, lb_thr, le_thr, dp_thr = thr
        p60 = p62 = p64 = p66 = p68 = 0
        for i in range(0x4000):
            a = int(acc[i])
            if lb_thr < a and p66 == 0:
                p66 = i
            if lo_thr < a and p60 == 0:
                p60 = i
            if dp_thr < a and p64 == 0 and p60 != 0:
                p64 = i
            if hi_thr < a and p62 == 0 and p64 != 0:
                p62 = i
            if le_thr < a and p68 == 0 and p66 != 0:
                p68 = i
        for off, v in ((0x60, p60), (0x62, p62), (0x64, p64), (0x66, p66), (0x68, p68)):
            self.p("H", off, v)
        # calculate_size_of_each_area
        f8, fa = self.u16(0xF8), self.u16(0xFA)
        self.p("Q", 0x118, ((f8 * (p68 - p66)) & M64 & 0x3FFFFFFFFFFFFFFF) >> 2)
        self.p("Q", 0x100, ((f8 * (p62 - p60)) & M64 & 0x3FFFFFFFFFFFFFFF) >> 2)
        self.p("Q", 0x108, ((f8 * (p64 - p60)) & M64 & 0x3FFFFFFFFFFFFFFF) >> 2)
        self.p("Q", 0x110, ((fa * (p62 - p64)) & M64 & 0x3FFFFFFFFFFFFFFF) >> 2)
        # limit_size_of_each_area (unsigned 64-bit compares)
        u1, u2, u3 = self.u16(0x120), self.u16(0x122), self.u16(0x6A)
        cap = (0x3FFF - u1 - u2) & M64
        if cap < self.g("Q", 0x100):
            self.p("Q", 0x100, cap)
        if cap < self.g("Q", 0x118):
            self.p("Q", 0x118, cap)
        c = s64(((u3 - u1) & 0xFFFFFFFF) - (1 << 32) if ((u3 - u1) & 0xFFFFFFFF) >> 31 else (u3 - u1) & 0xFFFFFFFF) & M64
        if c < self.g("Q", 0x108):
            self.p("Q", 0x108, c)
        c = (0x3FFF - u3 - u2) & M64
        if c < self.g("Q", 0x110):
            self.p("Q", 0x110, c)

    # calculate_limited_image_hist (low + high temperature parts)
    def limited_hist(self):
        hist = self.hist
        lim = np.zeros(0x4000, np.int64)
        excl = self.g("i", 4) != 0
        p60, p62, p64 = self.u16(0x60), self.u16(0x62), self.u16(0x64)
        e4 = self.u16(0xE4)

        def part(idx, outer, plat, top, vac):
            h = hist[idx]
            v = np.where(h > e4, np.where(h <= plat, plat, np.where(h <= top, h, top)), vac)
            return np.where(outer, 0 if excl else h, v)

        lo_idx = np.arange(0, min(p64, 0x3FFF) + 1)
        lim[lo_idx] = part(lo_idx, lo_idx < p60, self.u16(0xDA), self.u16(0xD8), self.u16(0xDC))
        hi_idx = np.arange(p64 + 1, 0x4000)
        lim[hi_idx] = part(hi_idx, hi_idx > p62, self.u16(0xE0), self.u16(0xDE), self.u16(0xE2))
        self.lim = lim & 0xFFFF

    # calculate_count_of_low_and_high_temperature_area
    def counts(self):
        i = np.arange(0x4000)
        excl = self.g("i", 4) != 0
        p60, p62, p64 = self.u16(0x60), self.u16(0x62), self.u16(0x64)
        lim = self.lim
        low = (i <= p64) & ~((i < p60) & excl)
        high = (i > p64) & ~((i > p62) & excl)
        self.p("Q", 0xC8, int(lim[low].sum()))
        self.p("Q", 0xD0, int(lim[high].sum()))

    # calculate_adaptive_histogram_plantform_value
    def adaptive_platform(self):
        mode = self.g("i", 8)
        if mode == 0:
            return
        if mode != 1:
            raise IspError("invalid parameter", -3)
        self.p("H", 0xF4, int(float(self.u16(0xB8)) * 0.75))
        self.p("H", 0xF6, int(float(self.u16(0xBA)) * 0.75))
        cap = self.u16(0xF4) if self.u16(0xF6) < self.u16(0xF4) else self.u16(0xF6)
        ea, ec = self.u16(0xEA), self.u16(0xEC)
        m = self.g("i", 0)
        if m == 0:
            d = self.g("Q", 0x108) or 1
            self.p("H", 0xD8, ((self.u16(0xF8) * self.g("q", 0xC8) * 0x40) & M64) // d >> 8)
            if self.u16(0xF4) < self.u16(0xD8):
                self.p("H", 0xD8, self.u16(0xF4))
            if self.u16(0xD8) < ea:
                self.p("H", 0xD8, ea)
            if ec < self.u16(0xD8):
                self.p("H", 0xD8, ec)
            d = self.g("Q", 0x110) or 1
            self.p("H", 0xDE, ((self.u16(0xFA) * self.g("q", 0xD0) * 0x40) & M64) // d >> 8)
            if self.u16(0xF6) < self.u16(0xDE):
                self.p("H", 0xDE, self.u16(0xF6))
            if self.u16(0xDE) < ea:
                self.p("H", 0xDE, ea)
            if ec < self.u16(0xDE):
                self.p("H", 0xDE, ec)
        elif m == 1:
            d = self.g("Q", 0x100) or 1
            self.p("H", 0xD8, ((self.u16(0xF8) * (self.g("q", 0xD0) + self.g("q", 0xC8)) * 4) & M64) // d >> 8)
            if cap < self.u16(0xD8):
                self.p("H", 0xD8, cap)
            if self.u16(0xD8) < ea:
                self.p("H", 0xD8, ea)
            if ec < self.u16(0xD8):
                self.p("H", 0xD8, ec)
            self.p("H", 0xDE, self.u16(0xD8))
        else:
            raise IspError("invalid parameter", -3)
        # update_low_platform_value
        e6, e8 = self.u16(0xE6), self.u16(0xE8)
        for dst, k, src in ((0xDA, 0xEE, 0xD8), (0xE0, 0xF0, 0xDE)):
            self.p("H", dst, (self.u16(k) * self.u16(src)) >> 8)
            if self.u16(dst) < e6:
                self.p("H", dst, e6)
            if e8 < self.u16(dst):
                self.p("H", dst, e8)
        # update_vacant_platform_value
        v = self.g("i", 0xC)
        if v == 0:
            self.p("H", 0xDC, self.u16(0xDA))
            self.p("H", 0xE2, self.u16(0xE0))
        elif v == 1:
            self.p("H", 0xDC, (self.u16(0xF2) * self.u16(0xDA)) >> 8)
            self.p("H", 0xE2, (self.u16(0xF2) * self.u16(0xE0)) >> 8)
        else:
            raise IspError("invalid parameter", -3)

    # update_platform_value_of_current_frame
    def platforms(self):
        p60, p62, p64 = self.u16(0x60), self.u16(0x62), self.u16(0x64)
        i = np.arange(0x4000)
        h = self.hist
        lo = h[(i >= p60) & (i < p64)]
        hi = h[(i < p62) & (i >= p64)]
        self.p("H", 0xB8, int(lo.max(initial=0)))
        self.p("H", 0xBA, int(hi.max(initial=0)))
        self.limited_hist()
        self.counts()
        self.adaptive_platform()
        self.limited_hist()
        self.counts()
        self.adaptive_platform()
        self.limited_hist()
        self.counts()

    # calculate_infinite_impulse_response (FIR over global average_delta_array, IIR factor)
    def iir(self):
        if self.g("i", 0x140) > 2 or self.g("B", 0x58) != 0:
            raise NotImplementedError("agc3 frame-average FIR (multi-frame state)")
        if self.g("i", 0x140) > 6 or self.g("i", 0x134) != 0:
            s138, s128 = self.g("h", 0x138), self.g("h", 0x128)
            hi_ = (s128 < s138) if s138 < 0 else (s138 < s128)
            self.p("i", 0x134, self.g("i", 0x130) if hi_ else self.g("i", 0x12C))

    # calculate_map_function
    def map_function(self):
        p66, p68 = self.u16(0x66), self.u16(0x68)
        span = (p68 - p66) & 0xFFFF or 1
        sz = self.g("Q", 0x118)
        q124, q120, q122 = self.u16(0x124), self.u16(0x120), self.u16(0x122)
        base = s64(q124 * ((0x3FFF - q120 - q122 - (sz & 0xFFFF)) & M64)) >> 8
        top = 0x3FFF - q122
        lin = np.zeros(0x4000, np.int64)
        last = 0
        for i in range(0x4000):
            if i < p66:
                last = min(max(base, q120), top)
                lin[i] = last
            elif i > p68:
                lin[i] = last
            else:
                num = s64((sz & 0xFFFF) * (i - p66))
                q = abs(num) // span
                v = base + q120 + (q if num >= 0 else -q)
                last = min(max(v, q120), top)
                lin[i] = last
        self.lin = lin & 0xFFFF
        # calculate_map_function_of_current_frame
        m = self.g("i", 0)
        lim = self.lim
        cdf = np.zeros(0x4000, np.int64)
        if m == 0:
            c8 = self.g("Q", 0xC8) or 1
            d0 = self.g("Q", 0xD0) or 1
            k0 = ((self.g("q", 0x108) << 16) & M64) // c8
            k1 = ((self.g("q", 0x110) << 16) & M64) // d0
            b2 = (self.u16(0x6A) - self.g("q", 0x108)) & M64
            b3 = self.u16(0x6A)
            p60, p62, p64 = self.u16(0x60), self.u16(0x62), self.u16(0x64)
            a30 = a28 = l20 = 0
            for i in range(0x4000):
                if i < p60:
                    cdf[i] = b2
                elif i > p64:
                    if i > p62:
                        cdf[i] = l20
                    else:
                        a28 = (a28 + k1 * int(lim[i])) & M64
                        l20 = (b3 + (a28 >> 16)) & M64
                        cdf[i] = l20
                else:
                    a30 = (a30 + k0 * int(lim[i])) & M64
                    l20 = (b2 + (a30 >> 16)) & M64
                    cdf[i] = l20
        elif m == 1:
            den = (self.g("q", 0xD0) + self.g("q", 0xC8)) & M64
            if den == 0:
                raise UndefinedBehaviour("agc3 divides by zero")
            k0 = ((self.g("q", 0x100) << 16) & M64) // den
            b2 = (((0x3FFF - q120 - q122 - self.g("q", 0x100)) & M64) * q124 & M64) >> 8
            acc = 0
            for i in range(0x4000):
                acc = (acc + k0 * int(lim[i])) & M64
                cdf[i] = ((acc >> 0x1B) + b2) & 0xFFFFFFFF
        else:
            raise IspError("invalid parameter", -3)
        cdf &= 0xFFFFFFFF  # stored as int, read back as uint
        self.cdf = cdf
        # calculate_new_map_function
        if self.g("i", 0x140) >= 7:
            raise NotImplementedError("agc3 IIR map (multi-frame state)")
        k = self.u16(0xB0)
        lo, hi = q120, 0x3FFF - q122
        out = np.empty(0x4000, np.int64)
        for i in range(0x4000):
            v = (((0x100 - k) & M64) * int(cdf[i]) + int(self.lin[i]) * k) & M64
            v = s64(v >> 8)
            if hi < v:
                v = hi
            if v < lo:
                v = lo
            out[i] = v & 0xFFFF
        self.cur = out

    def run(self, img, detail):
        self.blurred_and_average(img, detail)
        self.histogram()
        self.demarcation()
        self.platforms()
        self.iir()
        self.map_function()
        if img.max(initial=0) >= 0x4000:
            raise UndefinedBehaviour("agc3 map index past 0x3fff")
        return self.cur[img.reshape(-1)].astype(np.uint16)


def enhance_300(t, img: np.ndarray, w: int, h: int, cfg: bytes) -> np.ndarray:
    detail = extract_detail(t, img, w, h)
    agc = Agc3(cfg, w, h).run(img.reshape(-1), detail)
    return digital_detail_enhance(agc, detail, cfg[0x148:0x164])


# ---- process() orchestration (libv_iirp process)

DIRP2IRPROC = {0: 1, 1: 10, 2: 3, 3: 8, 4: 5, 5: 9, 6: 6, 7: 4, 8: 7, 9: 11}
IRPROC2DIRP = {v: k for k, v in DIRP2IRPROC.items()}
# color_convert_ircmd2irproc: v0300 headers store this id (create fails on others)
IRCMD2IRPROC = {0: 1, 1: 11, 2: 3, 3: 7, 4: 4, 5: 6, 6: 8, 7: 10, 8: 9, 9: 5}


class Cfg:
    def __init__(self, size):
        self.b = bytearray(size)

    def put(self, fmt, off, v):
        n = struct.calcsize("<" + fmt)
        self.b[off:off + n] = (v & ((1 << (8 * n)) - 1)).to_bytes(n, "little")
        return self


def cfg_enhance(manual=False, lower=0, upper=0) -> bytes:
    """process()'s 0x50-byte y14_image_enhance config (local_1e8)."""
    c = Cfg(0x50)
    c.b[0:0x1C] = struct.pack("<ffffiff", F(0.0001), F(0.0001), F(0.008), F(0.001), 1000, 1.0, F(0.3))
    c.put("B", 0x1C, 1 if manual else 0).put("H", 0x1E, lower).put("H", 0x20, upper)
    c.b[0x24:0x40] = struct.pack("<fffiiii", 0.0, 250.0, 80.0, 0, 5, 40, 80)
    return bytes(c.b)


def cfg_300_gain0(w, h, wide) -> bytes:
    """process()'s inline agc3 config for gain 0; wide = max-min of the NR image > 1000.
    Offset 0x140 (frame counter) is left uninitialised by the native code; taken as 0."""
    c = Cfg(0x168)
    for fmt, off, v in (("Q", 0, 0x100000000), ("Q", 8, 0x100000001), ("I", 0x10, 2), ("Q", 0x14, 0xE0C0A08060401),
                        ("I", 0x1C, 0x5040302), ("H", 0x20, 0x406), ("B", 0x22, 1), ("I", 0x38, 0x101), ("H", 0x3C, 0),
                        ("H", 0x3E, h), ("H", 0x40, w), ("I", 0x48, 0x20000000), ("H", 0x4C, 0x2000), ("B", 0x58, 0),
                        ("Q", 0x60, 0x282EE03FFE0028), ("Q", 0x68, 0xFFFF000012C03F70), ("H", 0x70, 0), ("H", 0x72, 0xFFFF),
                        ("H", 0x74, 58000), ("H", 0xB0, 0 if wide else 200), ("I", 0xB8, 0x27102710),
                        ("Q", 0xD8, 0xDAC0039023E0DAC), ("Q", 0xE0, 0xA00010039023E), ("Q", 0xE8, 0x1E1B58019007D0),
                        ("Q", 0xF0, 0x3A983A98000500C8), ("H", 0xF8, 400), ("H", 0xFA, 700), ("Q", 0x100, 16000),
                        ("Q", 0x108, 8000), ("Q", 0x110, 8000), ("Q", 0x118, 16000), ("H", 0x120, 0x1E0), ("H", 0x122, 0x50),
                        ("H", 0x124, 0x40), ("Q", 0x12C, 0x177000001770), ("Q", 0x134, 0xC000600000000),
                        ("I", 0x13C, 0x17700800), ("I", 0x148, 0), ("I", 0x14C, 0), ("I", 0x150, 0),
                        ("Q", 0x154, 0x300000000), ("Q", 0x15C, 0x5000000028)):
        c.put(fmt, off, v)
    return bytes(c.b)


def cfg_300(w, h, gain) -> bytes:
    """agc3 config for the v0300 headers (gain != 0 in practice)."""
    c = Cfg(0x168)
    c.put("Q", 0, 0x100000000).put("Q", 8, 0x100000001)
    if gain == 0:
        seq = (("Q", 0x6A, 0xFFFF000012C0), ("H", 0xB0, 200), ("I", 0x10, 2), ("I", 0x62, 0x2EE03FFE),
               ("I", 0x72, 0xE290FFFF), ("H", 0xEA, 400), ("I", 0xF0, 0x500C8), ("I", 0xF8, 0x2BC0190),
               ("H", 0x13E, 6000), ("I", 0x148, 0), ("I", 0x14C, 0), ("I", 0x150, 0))
    else:
        seq = (("Q", 0x6A, 0x64FFDC00641F40), ("I", 0x10, 0), ("H", 0xB0, 0x20), ("I", 0x62, 0x20003F70),
               ("H", 0xEA, 0x28), ("I", 0x72, 0xC350FFDC), ("I", 0xF0, 0x3C), ("I", 0xF8, 0x5140320),
               ("H", 0x13E, 0x600), ("I", 0x148, 0), ("I", 0x14C, 0x43960000), ("I", 0x150, 0x42F00000))
    for fmt, off, v in seq + (("Q", 0x154, 0x300000000), ("Q", 0x15C, 0x5000000028), ("H", 0x3C, 0), ("H", 0x40, w),
                              ("H", 0x3E, h), ("Q", 0xD8, 0xDAC0039023E0DAC), ("Q", 0xE0, 0xA00010039023E),
                              ("Q", 0x12C, 0x177000001770), ("Q", 0x134, 0xC000600000000), ("I", 0x38, 0x101),
                              ("Q", 0x100, 16000), ("Q", 0x108, 8000), ("Q", 0x14, 0xE0C0A08060401), ("B", 0x58, 0),
                              ("I", 0x48, 0x20000000), ("H", 0x4C, 0x2000), ("H", 0x60, 0x28), ("I", 0x66, 0x3F700028),
                              ("I", 0xB8, 0x27102710), ("H", 0xE8, 2000), ("I", 0xEC, 0x1E1B58), ("I", 0xF4, 0x3A983A98),
                              ("H", 0x120, 0x1E0), ("H", 0x122, 0x50), ("H", 0x124, 0x40), ("I", 0x140, 0),
                              ("Q", 0x110, 8000), ("Q", 0x118, 16000), ("H", 0x13C, 0x800), ("I", 0x1C, 0x5040302),
                              ("H", 0x20, 0x406), ("B", 0x22, 1)):
        c.put(fmt, off, v)
    return bytes(c.b)


def temp_to_y14(t) -> int:
    """(u16)cvttss2si((float)((double)t + 273.15) * 16.0f) as process() packs colour-bar/isotherm limits."""
    v = F(F(float(F(t)) + 273.15) * F(16.0))
    if not np.isfinite(v) or not (-2.0 ** 31 <= float(v) < 2.0 ** 31):
        return 0  # cvttss2si -> 0x80000000
    return int(np.trunc(float(v))) & 0xFFFF


def y14_map_to_yuv_colorbar(t, y14, pal):
    yuv = map_to_yuv_pseudocolor(t, y14, pal)
    return (y14.reshape(-1) >> 6).astype(np.uint8), yuv


def image_isotherm(t, raw, y8, pal, lo, hi):
    if f"isp_yuv_{pal}" not in t:
        raise IspError("color mode unsupported", -3)
    inside = (raw.reshape(-1).astype(np.int64) >= lo) & (raw.reshape(-1).astype(np.int64) <= hi)
    return np.where(inside[:, None], yuv_lut(t, pal)[y8], yuv_lut(t, 1)[y8]).astype(np.uint8)


def process(t, y14, w, h, gain, pal_irproc, magic, version, colorbar=(False, 30.0, 25.0),
            isotherm=(False, 30.0, 25.0), brightness=50, reverse=None):
    """libv_iirp process(): returns RGB (h, w, 3). reverse(t) -> raw-domain temperature for the
    manual colour bar (reverse_enhance_distance_temp_correct)."""
    if gain == 3:
        raise IspError("high high gain not supported", -12)
    pal = pal_irproc
    if pal == 7 and h == 0x400 and w == 0x500:
        pal = 0xC
    nr = spatial_noise_reduction(t, y14, w, h, gain)
    if not nr.any():
        raise IspError("process before agc get image all data is 0", -65)
    if not colorbar[0]:
        if gain == 0:
            out = enhance_300(t, nr, w, h, cfg_300_gain0(w, h, int(nr.max()) - int(nr.min()) > 1000))
        elif magic and (version & 0xFFFEFFFF) != 0x200 and version != 0x20200:
            out = enhance_300(t, nr, w, h, cfg_300(w, h, gain))
        else:
            out = enhance(t, nr, w, h, cfg_enhance())
        yuv = map_to_yuv_pseudocolor(t, out, pal)
        y8 = y14_to_y8(out)
    else:
        hi_t, lo_t = reverse(colorbar[1]), reverse(colorbar[2])
        hi_raw, lo_raw = temp_to_y14(hi_t), temp_to_y14(lo_t)
        out = enhance(t, nr, w, h, cfg_enhance(True, lo_raw, hi_raw))
        y8, yuv = y14_map_to_yuv_colorbar(t, out, pal)
    if isotherm[0]:
        ipal = 3 if pal in (1, 0xB) else pal
        yuv = image_isotherm(t, y14, y8, ipal, temp_to_y14(isotherm[2]), temp_to_y14(isotherm[1]))
    yuv = brightness_adjust(yuv, brightness_factor(brightness))
    return yuv444_to_rgb(yuv).reshape(h, w, 3)


def adaptive_params(y14):
    """get_color_bar_adaptive_params: (high, low) from max/min of the raw y14 copy."""
    def conv(v):
        return F(float(F(F(int(v)) * F(0.0625))) - 273.15)
    return conv(y14.max()), conv(y14.min())


def process_strech(t, y14, w, h, gain, colorbar=(False, 30.0, 25.0)):
    """process_strech: NR + classic enhance; manual colour bar limits straight from the temperatures."""
    if gain == 3:
        raise IspError("high high gain not supported", -12)
    nr = spatial_noise_reduction(t, y14, w, h, gain)
    if colorbar[0]:
        cfg = cfg_enhance(True, temp_to_y14(colorbar[2]), temp_to_y14(colorbar[1]))
    else:
        cfg = cfg_enhance()
    return enhance(t, nr, w, h, cfg).astype(np.float32).reshape(h, w)


def rgb_lut(t, pal):
    return t[f"isp_rgb_{pal}"].reshape(256, 3)


def pseudo_color_lut(t):
    """get_pseudo_color_lut: dirp palettes 0..9 as planar red[10][256], green[10][256], blue[10][256]."""
    luts = np.stack([rgb_lut(t, DIRP2IRPROC[i]) for i in range(10)])  # (10, 256, 3)
    return np.ascontiguousarray(luts.transpose(2, 0, 1))                 # (3, 10, 256)
