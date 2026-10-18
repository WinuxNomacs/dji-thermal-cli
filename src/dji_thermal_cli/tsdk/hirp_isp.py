"""The hirp plugin's image pipeline (libv_hirp -> MicroSDK / libMicroIA; newer M3TD / M4T firmware).

The 14-bit frame goes through DJI's MicroSDK infrared ISP:

- IFR_ISP_Process: raw & 0x3fff, then (M4T gain range) an edge-preserving 14-bit spatial filter;
- AGC_RUN: 16x16 block-mean weighted histogram of the mean-shifted 12-bit frame, tail points,
  a cumulative-histogram map curve and the 12-bit -> 8-bit mapping (plus a 14-bit LUT and the frame
  range the colouring stage receives);
- DDE: the raw detail layer (5-tap Gaussian high-pass, compressed and weighted) added to the gray
  image; then (M3TD) a 3x3 thresholded mean filter on the 8-bit image;
- colouring: the 256-colour palette converted to packed YUV, mapped as NV16 (U/V averaged over
  pixel pairs) and converted back to RGB.

All ISP parameters come from the R-JPEG itself: the APP5 "HIDR" container embeds a ZIP with
DJI's AGC/DDE/DNR parameter files, and APP4's TempParams give the gain range. They are turned into
the ISP handle exactly as IFR_ISP_Create / IFR_ISP_SetDefaultParam / IFR_ISP_SetParam /
IFR_ISP_UpdateISPParams / CC_UpdateConfig do. Integer arithmetic, rounding and clamping follow
libMicroIA, so the output is bit-identical to the native SDK.
"""

from __future__ import annotations

import io
import struct
import zipfile

import numpy as np

from .errors import UndefinedBehaviour, UnsupportedFeature


class Unsupported(UnsupportedFeature):
    """A configuration the port does not cover yet (the native SDK handles it)."""


def u8(H, o): return H[o]
def u16(H, o): return struct.unpack_from("<H", H, o)[0]
def s16(H, o): return struct.unpack_from("<h", H, o)[0]
def u32(H, o): return struct.unpack_from("<I", H, o)[0]
def w16(H, o, v): struct.pack_into("<H", H, o, v & 0xFFFF)


def img_mean(a: np.ndarray) -> int:
    """comm_calcImgMeanValue: rounded mean via a 32-bit reciprocal."""
    n = a.size
    if n == 0:
        return 0
    s = int(a.astype(np.uint64).sum())
    return ((s * ((1 << 32) // n) + (1 << 31)) >> 32) & 0xFFFF


def _s16(a):
    return ((a.astype(np.int64) + 0x8000) & 0xFFFF) - 0x8000


def shift_img(a: np.ndarray, mode: int, v: int) -> np.ndarray:
    """comm_shiftImg (in place natively): 0 -> p + 0x800 - v, 1 -> p - v (both as s16, clipped to
    [0, 0xfff]); 2 -> p >> v."""
    if mode == 2:
        return (a.astype(np.int64) >> (v & 0x1F)).astype(np.uint16)
    d = _s16(a.astype(np.int64) + ((0x800 - v) if mode == 0 else -v))
    return np.clip(d, 0, 0xFFF).astype(np.uint16)


# ---------------------------------------------------------------------------------------------
# AGC (libMicroIA AGC_RUN), first-frame path. H: bytearray of the IFR handle; buffers behind
# the handle's pointers are numpy arrays in B (keyed by the pointer's offset).

def b8(H, o): return H[o]
def w8(H, o, v): H[o] = v & 0xFF
def g32(H, o): return struct.unpack_from("<I", H, o)[0]
def w32(H, o, v): struct.pack_into("<I", H, o, v & 0xFFFFFFFF)


def tp_index(hist, total, p3, p4):
    """comm_getHistTPIndex: first bins where the running sum exceeds p3 and max(total - p4, p3)
    (u32 arithmetic); returns (lo, hi, min(hi - lo, 0xffe) + 1)."""
    c = np.cumsum(hist.astype(np.int64)) & 0xFFFFFFFF
    t2 = (total - p4) & 0xFFFFFFFF
    if t2 < p3:
        t2 = p3
    i = np.nonzero(c > p3)[0]
    lo = int(i[0]) if i.size else None
    j = np.nonzero(c > t2)[0]
    hi = int(j[0]) if j.size else None
    return lo, hi


def sum_hist(hist, s):
    """comm_calcSumHist with its parameter struct s (dict: thr, start, top, mn1, mx1, mn2, mx2,
    mn3, mx3, i1, i2)."""
    cum = np.zeros(0x1000, np.int64)
    h = hist.astype(np.int64)
    start = s["start"]
    cum[start] = min(max(h[start], s["mn1"]), s["mx1"])
    k = (start + 1) & 0xFFFF
    if k < s["i1"]:
        for i in range(k, s["i1"]):
            cum[i] = cum[i - 1] + min(max(h[i], s["mn1"]), s["mx1"])
        k = s["i1"]
    if k < s["i2"]:
        for i in range(k, s["i2"]):
            cum[i] = cum[i - 1] if h[i] < s["thr"] else cum[i - 1] + min(max(h[i], s["mn2"]), s["mx2"])
    if s["i2"] < 0x1000:
        for i in range(s["i2"], 0x1000):
            cum[i] = cum[i - 1] if i > s["top"] else cum[i - 1] + min(max(h[i], s["mn3"]), s["mx3"])
    return cum & 0xFFFFFFFF


def agc_run(H: bytearray, raw: np.ndarray, raw0: np.ndarray | None = None, curves: dict | None = None):
    """AGC_RUN on a (h, w) 14-bit frame (raw0: the frame before the 14-bit NR, whose range bounds
    the LUT). Returns (y8, v[4], lut14 (0x4000 bytes), B)."""
    B = {}
    h, w = raw.shape
    n = u32(H, 0xC)
    # --- preProcess
    w16(H, 0x468, img_mean(raw))                                    # comm_calcImgMeanValue
    mv = u16(H, 0x46E) if H[0x401] == 4 else u16(H, 0x468)
    if H[0x444] != 1 or H[0x1A] == 1:
        w16(H, 0x476, mv)
    # agc_calcManualLinearFixBgValue(H, mv) -- parameter is the register value (mean) passed through
    p2 = mv
    if H[0x466] == 0:
        w16(H, 0x472, p2); u3 = p2
    else:
        u3 = u16(H, 0x470)
    k = H[0x415]
    u3 = ((p2 * k + (0x100 - k) * u3) >> 8) & 0xFFFF
    w16(H, 0x470, u3)
    if H[0x57E] != 3:
        w16(H, 0x472, u3)
    if H[0x416] == 0:
        if H[0x1A] == 1:
            w16(H, 0x472, u3)
        else:
            raise Unsupported("bg value tracking (frames > 1)")
    else:
        w16(H, 0x472, u16(H, 0x418))
    if H[0x57E] == 3:
        w16(H, 0x476, u16(H, 0x472))
    flat = raw.reshape(-1)[:n]
    w16(H, 0x46A, int(flat.max()) if n else 0)                      # agc_calcRawMaxMinValue
    w16(H, 0x46C, int(flat.min()) if n else 0xFFFF)
    if H[0x662] == 1 or H[0x463] == 1:
        raise Unsupported("agc_shiftTypeUpdate")
    m = H[0x401]
    if m in (0, 3, 4):
        r12 = shift_img(raw, 0, u16(H, 0x476))
    elif m == 1:                                                    # manual range: p - lower
        r12 = shift_img(raw, 1, u16(H, 0x40E))
    else:
        raise Unsupported(f"agc shift mode {m}")
    roi = np.array(B.setdefault(0x5A8, np.array([0, 0, w, h], np.uint32)))
    agc_raw12_info(H, r12, roi)
    # --- statistics
    wt = agc_weight_table(H, r12)
    B[0x5B0] = wt
    hist = np.zeros(0x1000, np.int64)
    cap = 0xFFFFF if w >= 0x281 else 0x3FFFF
    y0 = u16(H, 0x3FC) & 0xFFFF
    if g32(H, 0x3FC) == 0x501:
        y0 = int(roi[1]) & 0xFFFF
    x0, rw, rh = int(roi[0]), int(roi[2]), int(roi[3])
    rows = r12[y0:rh, x0:x0 + rw:2].reshape(-1)
    wts = wt.reshape(h, w)[y0:rh, x0:x0 + rw:2].reshape(-1)
    # per-bin saturation is applied as the native loop accumulates: emulate exactly when needed
    acc = np.bincount(rows, weights=wts, minlength=0x1000).astype(np.int64)
    if (acc > cap).any():
        acc = np.zeros(0x1000, np.int64)
        for v, wv in zip(rows.tolist(), wts.tolist()):
            acc[v] = min(acc[v] + wv, cap)
    hist = acc
    B[0x600] = hist
    w32(H, 0x52C, int(hist.sum()))
    if H[0x420] == 4:
        raise Unsupported("gradient weight mode (8-bit histogram)")
    agc_get_points(H, hist)
    agc_points_sh(H, hist, B)
    # --- map curve
    agc_calc_dnr(H)
    agc_bg_map_value(H)
    if H[0x57E] != 3:
        if H[0x61] == 1:
            agc_temp_range_div(H)
        elif H[0x432] == 1:
            raise Unsupported("segmented map")
        else:
            agc_nosegm_div(H)
    agc_comp_value(H)
    if H[0x580] != 0:
        raise Unsupported("map curve variant")
    if H[0x61] == 1:
        curve = agc_manual_curve(H, B[0x610])
    elif H[0x57E] == 3 or H[0x432] == 1:
        raise Unsupported("map curve variant")
    else:
        curve = agc_nosegm_curve(H, B[0x610])
    B[0x5D0] = curve
    if H[0x456] == 1:
        raise Unsupported("agc_curveContrastRevise")
    # --- grayMap
    if not (H[0x1A] == 1):
        raise Unsupported("curve smoothing (frames > 1)")
    curves = curves or {}
    if H[0x404] == 1:
        curve = curves.get(0x638, np.arange(256, dtype=np.uint8))[curve]
    if H[0x405] == 1:
        curve = curves.get(0x650, np.arange(256, dtype=np.uint8))[curve]
    B[0x5D0] = curve
    if H[0x407] != 0:
        raise Unsupported("dual curve")
    r = struct.unpack_from("<4I", H, 0x594)
    y8 = np.zeros((h, w), np.uint8)
    y8[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = curve[r12[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]]
    v, lut = agc_lut14(H, raw if raw0 is None else raw0, curve)
    if H[0x44C] == 1:
        raise Unsupported("agc_grayEnhance")
    # agc_updateState
    w16(H, 0x46E, u16(H, 0x468))
    H[0x535] = H[0x534]; H[0x537] = H[0x536]
    if H[0x580] == 0:
        w16(H, 0x4F0, u16(H, 0x49E))
    else:
        w16(H, 0x4F2, u16(H, 0x49E))
    H[0x4C3] = H[0x4C2]
    return y8, v, lut, B


def agc_raw12_info(H, r12, roi):
    lo, hi = (u16(H, 0x504), u16(H, 0x506)) if H[0x580] == 0 else (u16(H, 0x508), u16(H, 0x50A))
    x0, y0, rw, rh = (int(x) for x in roi)
    blk = r12[y0:y0 + rh, x0:x0 + rw]
    if blk.size:
        mx, mn = int(blk.max()), int(blk.min())
        sel = blk[(blk <= hi) & (blk >= lo)]
        cnt = sel.size or 1
        mean = (int(sel.astype(np.int64).sum()) & 0xFFFFFFFF) // cnt
        # native starts the scan with max 0 / min 0xfff
        mx = max(mx, 0); mn = min(mn, 0xFFF)
    else:
        mean, mn, mx = 0, 0xFFF, 0
    w16(H, 0x49E, mean)
    w16(H, 0x48A, mx)
    w16(H, 0x48C, mn)
    off = (u16(H, 0x476) - 0x800) & 0xFFFF
    w16(H, 0x65E, mn + off)
    w16(H, 0x660, mx + off)
    if H[0x44C] == 1 and H[0x580] == 0:
        raise Unsupported("highlight mean")


def agc_weight_table(H, r12):
    h, w = r12.shape
    n = u32(H, 0xC)
    mode = H[0x420]
    if mode in (1, 2):
        if mode != 1:
            raise Unsupported("block mean mode 2")
        gx, gy = H[0x47A], H[0x47B]
        sx, sy = H[0x478], H[0x479]
        means = np.zeros(gx * gy, np.int64)
        for i in range(gy):
            for j in range(gx):
                o = sx * j + sy * w * i
                y, x = divmod(o, w)
                means[i * gx + j] = (int(r12[y:y + 16, x:x + 16].astype(np.int64).sum()) & 0xFFFFFFFF) >> 8
        nb = u16(H, 0x47C)
        srt = np.sort(means[:nb])
        med = srt[nb >> 1]
        bw = np.where(means[:nb] < med, H[0x421], H[0x422]).astype(np.uint8)
        yy = (np.arange(h) >> 4) & 0xFFF
        xx = np.arange(w) >> 4
        return bw[(H[0x47A] * yy)[:, None] + xx[None, :]].reshape(-1)
    if mode == 3 or mode == 4:
        raise Unsupported(f"weight mode {mode}")
    return np.full(n, 5, np.uint8)


def agc_get_points(H, hist):
    total = g32(H, 0x52C)
    lo, hi = tp_index(hist, total, u16(H, 0x50C), g32(H, 0x510))
    if lo is not None: w16(H, 0x48E, lo)
    if hi is not None: w16(H, 0x490, hi)
    p_lo, p_hi = u16(H, 0x48E), u16(H, 0x490)
    u3 = max(p_lo, u16(H, 0x49E))
    w16(H, 0x492, (p_lo + p_hi) >> 1)
    w16(H, 0x49E, min(u3, p_hi))
    lo, hi = tp_index(hist, total, g32(H, 0x514), g32(H, 0x518))
    if lo is not None: w16(H, 0x49A, lo)
    if hi is not None: w16(H, 0x49C, hi)
    if H[0x580] == 0:
        w16(H, 0x504, u16(H, 0x49A)); w16(H, 0x506, u16(H, 0x49C))
        if H[0x44C] == 1:
            raise Unsupported("highlight points")
    else:
        w16(H, 0x508, u16(H, 0x49A)); w16(H, 0x50A, u16(H, 0x49C))
    u6 = u16(H, 0x426)
    if H[0x423] == 1:
        raise Unsupported("agc 0x423 mode")
    w16(H, 0x4B0, u6)
    H[0x541] = 0
    l28 = l26 = 0
    if H[0x434] == 0:
        a = u16(H, 0x49E)
        l28 = max(a - H[0x433], 0) & 0xFFFF
        l26 = min(a + H[0x433], 0xFFF)
        if u16(H, 0x48E) < l28 and l26 < u16(H, 0x490):
            H[0x541] = 1
    elif H[0x434] == 1:
        raise Unsupported("segm endpoint")
    w16(H, 0x4A0, l28); w16(H, 0x4A2, l26)


def agc_points_sh(H, hist, B):
    if H[0x423] == 1:
        raise Unsupported("sum hist variant")
    if H[0x463] == 1 and H[0x401] == 2:
        raise Unsupported("sum hist shifted")
    s = dict(thr=H[0x462], start=u16(H, 0x48C), top=u16(H, 0x48A), mx1=u16(H, 0x42A), mn1=u16(H, 0x428),
             mx2=u16(H, 0x426), mn2=u16(H, 0x424), mx3=u16(H, 0x42E), mn3=u16(H, 0x42C),
             i1=u16(H, 0x48E), i2=u16(H, 0x490))
    if H[0x61] == 1:
        if H[0x401] == 2:
            raise Unsupported("manual range with shift mode 2")
        # manual range: the histogram from 1 up to (upper - lower), clamped to [0, 0xfff]
        d = min(max(u16(H, 0x410) - u16(H, 0x40E), 0), 0xFFF)
        s.update(start=1, i1=1, i2=d, top=d)
    cum = sum_hist(hist, s)
    B[0x610] = cum
    for off, idx in ((0x4C4, 0x48E), (0x4C8, 0x490), (0x4CC, 0x492), (0x4D0, 0x49E), (0x4D4, 0x4A0), (0x4D8, 0x4A2)):
        w32(H, off, int(cum[u16(H, idx)]))
    w32(H, 0x4DC, int(cum[s["top"] if H[0x61] == 1 else u16(H, 0x48A)]))


def agc_temp_range_div(H):
    """agc_calcTempRangeDiv: (curve span << 33) / (pixels in the manual range)."""
    n = g32(H, 0x4DC) or 1
    struct.pack_into("<Q", H, 0x548, ((((H[0x41C] - H[0x41B]) & 0xFFFFFFFF) << 33) // n) & 0xFFFFFFFFFFFFFFFF)


def agc_manual_curve(H, cum):
    """agc_calcManualTempRangeMapCurve: a ramp over the cumulative histogram, clamped to the
    curve's [low, high] grey levels."""
    div = struct.unpack_from("<Q", H, 0x548)[0]
    lo, hi = H[0x41B], H[0x41C]
    comp = s16(H, 0x57C)
    out = np.empty(0x1000, np.uint8)
    for i, c in enumerate(cum.tolist()):
        v = (((int(c) * div + (1 << 32)) & 0xFFFFFFFFFFFFFFFF) >> 33) & 0xFFFF
        v = (v - 0x10000 if v & 0x8000 else v) + comp
        out[i] = min(max(v, lo), hi) & 0xFF
    return out


def agc_calc_dnr(H):
    u7 = min(u16(H, 0x490) - u16(H, 0x48E), 0xFFE) + 1
    w16(H, 0x494, u7)
    if H[0x1A] != 1:
        raise Unsupported("dnr smoothing (frames > 1)")
    w16(H, 0x496, u7); w16(H, 0x498, u7)
    u12 = u7 & 0xFFFFFFFF
    u9 = u7 & 0xFFFF
    if H[0x439] == 1:
        b2 = H[0x43B]
        if H[0x43A] == 1:
            raise Unsupported("dnr mode 0x43a")
        if ((u12 & 0xFFFF) < b2 and H[0x578] == 0) or (u9 < b2 + H[0x43C] and H[0x578] == 1):
            b2 = H[0x43D]; H[0x578] = 1
        else:
            b2 = H[0x438]; H[0x578] = 0
        H[0x51C] = b2; H[0x51D] = b2
    else:
        b2 = H[0x438]; H[0x51C] = H[0x436]; H[0x51D] = H[0x437]
    if (u12 & 0xFFFF) > 0xFF:
        u12 = 0xFF
    u5 = (u12 * b2) & 0xFFFF
    if u5 < 0x3FC:
        u9 = u5 >> 2
    else:
        u9 = 0xFF
    H[0x542] = u9 & 0xFF
    if H[0x41D] == 1:
        i6 = 0 if u16(H, 0x48E) == 0 else max(H[0x41E] - H[0x41B], 0)
        H[0x579] = i6 & 0xFF
        if u16(H, 0x490) == 0xFFF:
            H[0x57A] = 0; i14 = 0; i10 = i6
        else:
            i14 = max(H[0x41C] - H[0x41F], 0)
            H[0x57A] = i14 & 0xFF; i10 = i6 + i14
    else:
        i14 = i6 = i10 = 0; H[0x579] = 0; H[0x57A] = 0
    u7 = (H[0x41C] - H[0x41B]) - i6 - i14
    if u9 < u7:
        u7 = u9
    H[0x542] = u7 & 0xFF
    H[0x4C2] = (u7 + i10) & 0xFF


def agc_bg_map_value(H):
    if H[0x441] == 1:
        raise Unsupported("bg map value")
    H[0x581] = H[0x440]


def agc_nosegm_div(H):
    c8, c4 = g32(H, 0x4C8), g32(H, 0x4C4)
    u7 = (c8 - c4) & 0xFFFFFFFF
    if H[0x41D] != 1:
        q = (H[0x4C2] << 33) // (c8 or 1)
        struct.pack_into("<Q", H, 0x548, q)
        c5 = (((c4 * q) >> 33) + H[0x41B]) & 0xFF
        H[0x53E] = c5
        H[0x53F] = (c5 + ((q * u7) >> 33)) & 0xFF
        return
    c5 = (H[0x579] + H[0x41B]) & 0xFF
    H[0x53E] = c5
    H[0x53F] = (H[0x542] + c5) & 0xFF
    struct.pack_into("<Q", H, 0x548, (H[0x542] << 33) // (u7 or 1))


def agc_comp_value(H):
    m = H[0x443]
    i3 = 0
    if m == 2:
        i3 = max(H[0x581] - H[0x4C2], 0)
    elif m == 1:
        i3 = max(H[0x581] - H[0x41B] - (H[0x4C2] >> 1), 0)
    elif m == 4:
        raise Unsupported("comp mode 4")
    elif m != 0:
        if H[0x4C2] != 0xFF:
            d = H[0x41C] - H[0x41B] - H[0x4C2]
            i3 = d >> 1 if d > 1 else 0
    i3 = s16(H, 0x1DE) + i3
    w16(H, 0x57C, 0xFF if i3 >= 0xFF else (-0x100 if i3 < -0x100 else i3))


def agc_nosegm_curve(H, cum):
    q = struct.unpack_from("<Q", H, 0x548)[0]
    c4, c8 = g32(H, 0x4C4), g32(H, 0x4C8)
    plo, phi = u16(H, 0x48E), u16(H, 0x490)
    b3e, b3f, off = H[0x53E], H[0x53F], s16(H, 0x57C)
    lo_c, hi_c = H[0x41B], H[0x41C]
    M = (1 << 64) - 1
    out = np.zeros(0x1000, np.uint8)
    for i in range(0x1000):
        c = int(cum[i])
        if i < plo:
            s = b3e - (((q * ((c4 - c) & 0xFFFFFFFF)) & M) >> 33 & 0xFFFF)
        elif phi < i:
            s = b3f + (((q * ((c - c8) & 0xFFFFFFFF)) & M) >> 33 & 0xFFFF)
        else:
            s = b3e + (((q * ((c - c4) & 0xFFFFFFFF)) & M) >> 33 & 0xFFFF)
        v = ((s + off + 0x8000) & 0xFFFF) - 0x8000
        if v <= lo_c:
            v = lo_c
        if not v <= hi_c:
            v = hi_c
        out[i] = v & 0xFF
    return out


def agc_lut14(H, raw, curve):
    """agc_grayMap14To8 (mode 3/4): the 14-bit -> 8-bit LUT and the four AGC outputs
    (low, high points and max/min in raw units)."""
    if H[0x401] == 1:
        return _agc_lut14_manual(H, raw, curve)
    if H[0x401] - 3 & 0xFF >= 2:
        raise Unsupported("lut14 for shift modes other than 3/4")
    mx, mn = int(raw.max()), int(raw.min())
    hi = max(mx, u16(H, 0x46A))
    lo = min(mn, u16(H, 0x46C))
    lut = np.zeros(0x4000, np.uint8)
    base = u16(H, 0x476) - 0x800
    r = 0
    for k in range(0x1000):
        r = (base + k) & 0xFFFF
        if lo <= r <= hi and lo < hi:
            lut[(r or 1) - 1] = curve[k]
    if r < hi:
        lut[r:hi] = curve[0xFFF]
    for off, dst in ((0x482, 0x480), (0x484, 0x481)):
        a = u16(H, off)
        if a < mn:
            H[dst] = 0
        elif mx < a:
            H[dst] = 0xFF
        else:
            a = a or 1
            w16(H, off, a)
            H[dst] = lut[a - 1]
    v0 = (u16(H, 0x48E) + u16(H, 0x476) - 0x800) & 0xFFFF
    v1 = (u16(H, 0x476) + u16(H, 0x490) - 0x800) & 0xFFFF
    vmax = u16(H, 0x46A) or 1
    vmin = u16(H, 0x46C) or 1
    w16(H, 0x46A, vmax)
    w16(H, 0x46C, vmin)
    H[0x47E] = lut[vmax - 1]
    H[0x47F] = lut[vmin - 1]
    return [v0, v1, vmax, vmin], lut


def _agc_lut14_manual(H, raw, curve):
    """agc_grayMap14To8, shift mode 1 (manual range from u16 0x40e to 0x410)."""
    L, U = u16(H, 0x40E), u16(H, 0x410)
    lo = min(int(raw.min()), L) or 1
    hi = max(int(raw.max()), U)
    lut = np.zeros(0x4000, np.uint8)
    r = 0
    for k in range(0x1000):
        r = (L + k) & 0xFFFF
        if lo <= r <= hi and lo < hi:
            lut[r - 1] = curve[k]
    if r < hi:
        lut[r:hi] = curve[0xFFF]
    for off, dst in ((0x482, 0x480), (0x484, 0x481)):
        a = u16(H, off)
        if a < L:
            H[dst] = 0
        elif U < a:
            H[dst] = 0xFF
        else:
            a = a or 1
            w16(H, off, a)
            H[dst] = lut[a - 1]
    v0 = (u16(H, 0x48E) + L) & 0xFFFF
    v1 = (u16(H, 0x490) + L) & 0xFFFF
    vmax, vmin = U or 1, L or 1
    w16(H, 0x46A, vmax)
    w16(H, 0x46C, vmin)
    H[0x47E] = lut[vmax - 1]
    H[0x47F] = lut[vmin - 1]
    return [v0, v1, vmax, vmin], lut


# ---------------------------------------------------------------------------------------------
# DDE (detail enhancement): DDE_V3_GetRawDetail / DDE_V3_AddHfDetail

def _s8(a):
    return ((np.asarray(a).astype(np.int64) + 0x80) & 0xFF) - 0x80


def dde_hf(H, raw):
    """DDE_comm_getHF_Gaussian_LH_S08: raw - 5x5 separable Gaussian (Q12 x Q12), as s8."""
    wts = [u16(H, 0x38E + 2 * k) for k in range(5)]
    p = np.pad(raw.astype(np.int64), 2, mode="reflect")
    h, w = raw.shape
    tmp = sum(wts[k] * p[k:k + h, :] for k in range(5))             # vertical (rows 2..h+1)
    tmp = np.pad(tmp[:, 2:w + 2], ((0, 0), (2, 2)), mode="reflect")
    s = sum(wts[k] * tmp[:, k:k + w] for k in range(5))
    c = raw.astype(np.int64)
    d = c - (s >> 24)
    d2 = c - ((s + 0x1000000) >> 24)
    out = np.where(d < 1, np.maximum(d, -0x80), np.where(d2 < 0x7F, np.maximum(d2, -0x80), 0x7F))
    return out.astype(np.int8)


def _hist_spread(hist, thr):
    c = np.cumsum(hist) & 0xFFFFFFFF
    i = np.nonzero(c >= thr)[0]
    lo = int(i[0]) if i.size else 0
    acc, hi = int(hist[255]), 0xFF
    if acc < thr:
        while acc < thr:
            hi -= 1
            acc = (acc + int(hist[hi & 0xFFFF])) & 0xFFFFFFFF
        hi &= 0xFF
    return max(hi - lo, 1) if hi - lo >= 1 else 1


def dde_get_raw_detail(H, raw):
    """DDE_V3_GetRawDetail: the detail layer (s8) the add step uses (handle buffer 0x3e0)."""
    if H[0x2A8] == 0:
        return None
    hf = dde_hf(H, raw)
    if H[0x2C3] == 1:
        raise Unsupported("DDE bilateral filter")
    src = hf
    n = u32(H, 0xC)
    hist = np.bincount((hf.reshape(-1)[:n:2].astype(np.int64) + 0x80) & 0xFF, minlength=256).astype(np.int64)
    spread = _hist_spread(hist, g32(H, 0x2B4))                         # DDE_CalHfCompressRate
    if H[0x2BF] == 1:
        rate = H[0x2C0]
    else:
        rate = ((u16(H, 0x2BC) * 0x20 + ((spread & 0xFF) >> 1)) // spread) & 0xFFFF0FFF
        rate = min(rate & 0xFFFF, 0x20) if (rate & 0xFFFF) > 0x20 else rate
    if H[0x2C1] == 1:
        div = H[0x2C2] or 1
        H[0x2C2] = div
        m = (spread & 0xFF) if (spread & 0xFF) < div else div
        rate = ((rate & 0xFFFF) * m) // div
    H[0x2BE] = rate & 0xFF
    cnt = _hist_spread(hist, g32(H, 0x2B8)) & 0xFF                     # DDE_GetHfCnt
    det = dde_weight(H, src, H[0x2C8] | H[0x2C9] << 8)
    tab = np.frombuffer(bytes(H[0x2E5:0x2E5 + 0x81]), np.int8).astype(np.int64)   # DDE_HfFix
    d = det.astype(np.int64)
    det = np.where(d < 0, -tab[-d], tab[np.abs(d)]).astype(np.int8)
    if H[0x2E4] == 1:
        raise Unsupported("DDE 0x2e4 second weight")
    H[0x2E3] = H[0x2E2]
    H[0x2E2] = cnt
    return det


def dde_weight(H, src, w2c8):
    rate, u2 = H[0x2BE], u16(H, 0x2C6)
    if not (rate < 0x21 and u2 < 0x400):
        raise Unsupported("DDE weight out of range (native returns an error)")
    k = w2c8 * ((rate * u2) & 0xFFFF)
    v = (src.astype(np.int64) * k) >> 0x15
    return np.clip(v, -H[0x2CB], H[0x2CA]).astype(np.int8)


def dde_shoot_ctrl(H, y8, det):
    t = H[0x2D4]
    b1, b2, b3, b5 = H[0x2D5], H[0x2D7], H[0x2D8], H[0x2D6]
    u6, u7, u8_, u9 = u16(H, 0x2DE), u16(H, 0x2E0), u16(H, 0x2DA), u16(H, 0x2DC)
    i15 = u7 * 16 - u6 * 16
    yv = y8.astype(np.int64)
    d = det.astype(np.int64)

    def cdiv(a, b):
        q = abs(a) // abs(b)
        return q if (a >= 0) == (b > 0) else -q

    span = max(b2 - b3, 1)
    span2 = max(b1 - b5, 1)
    # low side (d < -t): gain g from the gray value
    g_lo = np.where(yv < b3, u6, np.where(yv <= b2, (cdiv(i15, span) * (yv - b3) + u6 * 16) >> 4, u7))
    g_hi = np.where(yv < b5, u9, np.where(yv <= b1, ((cdiv(u8_ * 16 - u9 * 16, span2) * (yv - b5) + u9 * 16) >> 4) & 0xFFFF, u8_))
    # d < -t
    g16 = g_lo & 0xFFFF
    g2 = 0x100 - g16
    r = ((d * g2) >> 8) + 1
    r_u = r & 0xFFFFFFFF
    r = np.where(r_u < t, t, r)
    low = np.where(((g_lo & 0xFFFF) ^ 0x8000) - 0x8000 == 0, d, r)
    # d > t
    iv = d * (0x100 - g_hi)
    hi_v = np.where(((iv >> 8) & 0xFFFFFFFF) < t, t, iv >> 8)
    res = np.where(d <= t, np.where(d >= -t, d, low), hi_v)
    return _s8(res).astype(np.int8)


def dde_add(H, raw, y8, det):
    """DDE_V3_AddHfDetail (in place on y8 natively)."""
    if det is None:
        return y8
    if H[0x2D3] == 1:
        det = dde_shoot_ctrl(H, y8, det)
    u2 = u16(H, 0x2D0)
    sh = H[0x2D2] & 0x1F
    u10 = (1 << sh) & 0x7FFF
    b = y8.astype(np.int64)
    c = det.astype(np.int64)
    if H[0x2CE] == 1:
        i8 = max(u2 - u10, 0)
        u6 = np.maximum(raw.astype(np.int64) - i8, 0)
        u6 = np.where(u6 > u10, u10, u6)
        c = _s8((c * u6) >> sh)
    s = b + c
    u5 = np.where(((s + 0x8000) & 0xFFFF) - 0x8000 < 0, 0, s & 0x1FF)
    u5 = np.where((b != 0) & (b <= H[0x2CD]), u5, b)
    return np.minimum(u5, H[0x2CC]).astype(np.uint8)


def nr8(H, y8):
    """NR_V2_1_Filter8bit / nr_SpatialFilter8bit: 3x3 mean of the neighbours within 2*thr."""
    if H[0x731] != 1:
        return y8.copy()
    thr = (H[0x732] + H[0x732] * 4) >> 4
    H[0x739] = thr & 0xFF
    h, w = y8.shape
    p = np.pad(y8.astype(np.int64), 1, mode="reflect")
    c = y8.astype(np.int64)
    s = np.zeros_like(c); n = np.zeros_like(c)
    for dy in range(3):
        for dx in range(3):
            v = p[dy:dy + h, dx:dx + w]
            m = np.abs(v - c) <= 2 * thr
            s += np.where(m, v, 0); n += m
    q = (s & 0xFFFF) // np.maximum(n, 1)
    q16 = ((q + 0x8000) & 0xFFFF) - 0x8000
    return np.where(q16 >= 0xFF, 0xFF, np.where(q16 < 0, 0, q)).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# IFR_ISP_Process: raw protect, (raw DDE enhance), 14-bit NR

def _aniso(H, img, pad, offs):
    tab = np.frombuffer(bytes(H[0x67F:0x67F + 40]), np.uint8).astype(np.int64)
    k = H[0x6A7]
    h, w = img.shape
    p = np.pad(img.astype(np.int64), pad, mode="reflect")
    c = img.astype(np.int64)
    acc = np.zeros_like(c)
    for dy, dx in offs:
        d = np.clip(p[pad + dy:pad + dy + h, pad + dx:pad + dx + w] - c, -39, 39)
        acc += d * tab[np.abs(d)]
    acc = ((acc * k) + (1 << 31) & 0xFFFFFFFF) - (1 << 31)       # int32 wrap
    v = c + (acc >> 14)
    return np.where(v < 0x3FFF, np.maximum(v, 0), 0x3FFF).astype(np.uint16)


_A1 = [(-1, 0), (-2, 0), (1, 0), (2, 0), (0, 1), (0, 2), (0, -1), (0, -2)]
_A2 = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (1, 1), (-1, -1), (1, -1)]


def ifr_process(H, raw):
    """IFR_ISP_Process on the 14-bit frame: returns the frame the AGC and DDE see."""
    img = raw & 0x3FFF                                                  # comm_rawBWProtect
    if H[0x2A9] == 1:
        raise Unsupported("DDE raw enhance")
    if H[0x678] != 1:
        return img
    if H[0x679] == 1:
        raise Unsupported("temporal NR")
    if H[0x402] == 3:
        raise Unsupported("NR history frame")
    if H[0x67B] == 1:
        if H[0x67C] != 1:
            raise Unsupported("FastBF spatial NR")
        img = _aniso(H, img, 2, _A1)
        img = _aniso(H, img, 1, _A2)
    return img


# ---------------------------------------------------------------------------------------------
# colouring: palette (RGB -> packed YUV), PC_ColorMap (NV16), Nv16ToRGBByISP

def _cdiv(a, b):
    """C integer division (truncation toward zero) on int64 arrays."""
    q = np.abs(a) // b
    return np.where(a < 0, -q, q)


def palette_yuv(rgb256: np.ndarray) -> np.ndarray:
    """PseudoColorMap::transformatFormat (ARGB -> packed Y|V<<8|Y<<16|U<<24) for a (256, 3) RGB table."""
    r, g, b = (rgb256[:, k].astype(np.int64) for k in range(3))
    y = (r * 299 + g * 587 + b * 114) // 1000
    v = (r * 5000 + 1280000 - g * 4187 - b * 813) // 10000
    u = (b * 5000 + 1280000 - r * 1687 - g * 3313) // 10000
    return (y * 0x10001 + (v << 8) + (u << 24)) & 0xFFFFFFFF


def color_map_nv16(y8: np.ndarray, table: np.ndarray) -> np.ndarray:
    """PC_ColorMap (mode 0): Y plane, then U, V of each horizontal pixel pair (floor average)."""
    c = table[y8.reshape(-1)]
    Y = (c >> 16) & 0xFF
    a, b = c[0::2], c[1::2]
    U = ((a >> 24) + (b >> 24)) >> 1
    V = (((a >> 8) & 0xFF) + ((b >> 8) & 0xFF)) >> 1
    return np.concatenate([Y, np.stack([U, V], 1).reshape(-1)]).astype(np.uint8)


def nv16_to_rgb(nv: np.ndarray, n: int) -> np.ndarray:
    Y = nv[:n].astype(np.int64)
    uv = nv[n:2 * n].reshape(-1, 2).astype(np.int64)
    u = np.repeat(uv[:, 0] - 0x80, 2)
    v = np.repeat(uv[:, 1] - 0x80, 2)
    r = Y + _cdiv(v * 1375, 1000)
    g = Y + _cdiv(v * -703125 + u * -343750, 1000000)
    b = Y + _cdiv(u * 1734375, 1000000)
    return np.clip(np.stack([r, g, b], 1), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# ISP handle parameters

# --- MicroIA parameter records (record layouts, copy lists, weight curves) ---
# IFR_ISP_Load{AGC,DDE,DNR}Params: (record offset, word index in the parameter set, width)
_AGC_LOAD = (
    (0, 0, 1), (3, 1, 1), (4, 2, 1), (5, 3, 1), (6, 4, 1), (7, 5, 1),
    (8, 6, 1), (9, 7, 1), (10, 8, 1), (11, 9, 1), (12, 10, 1), (13, 11, 1),
    (14, 12, 1), (15, 13, 1), (16, 14, 1), (18, 15, 2), (20, 16, 2), (22, 17, 2),
    (24, 18, 2), (26, 19, 2), (28, 20, 2), (30, 21, 2), (32, 22, 4), (36, 23, 4),
    (40, 24, 4), (44, 25, 1), (45, 26, 1), (46, 27, 1), (47, 28, 1), (48, 29, 1),
    (49, 30, 1), (50, 31, 1), (51, 32, 1), (52, 33, 1), (53, 34, 1), (54, 35, 1),
    (55, 36, 1), (56, 37, 1), (57, 38, 1), (58, 39, 1), (59, 40, 1), (60, 41, 1),
    (62, 42, 2), (64, 43, 1), (66, 44, 2), (68, 45, 1), (69, 46, 1), (70, 47, 1),
    (71, 48, 1), (72, 51, 2), (74, 52, 2), (76, 49, 2), (78, 50, 2), (80, 53, 1),
    (81, 54, 1),
)
_DDE_LOAD = (
    (0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1), (4, 4, 4), (8, 5, 4),
    (12, 6, 2), (14, 7, 1), (15, 8, 1), (16, 9, 1), (17, 10, 1), (18, 11, 2),
    (20, 12, 2), (22, 13, 1), (23, 14, 1), (24, 15, 1), (25, 16, 1), (26, 17, 1),
    (28, 18, 2), (30, 19, 1), (31, 20, 1), (32, 21, 2), (34, 22, 2), (36, 23, 1),
    (37, 24, 1), (38, 25, 1), (39, 26, 1), (40, 27, 1), (41, 28, 1), (42, 29, 2),
    (44, 30, 2), (46, 31, 2), (48, 32, 2),
)
_DNR_LOAD = (
    (0, 0, 1), (1, 1, 1), (2, 2, 2), (4, 3, 1), (5, 4, 1), (6, 5, 2),
    (8, 6, 2), (10, 7, 2), (12, 8, 1), (13, 9, 1), (14, 10, 1), (15, 11, 1),
    (16, 12, 1), (17, 13, 1), (18, 14, 1),
)
# IFR_ISP_Update{Agc,Dde,Dnr}Params: (handle offset, record offset, width)
_AGC_UPD = (
    (1025, 4, 1), (1026, 5, 1), (1028, 6, 1), (1029, 7, 1), (1030, 8, 1), (1053, 9, 1),
    (1051, 10, 1), (1052, 11, 1), (1054, 12, 1), (1055, 13, 1), (1056, 14, 1), (1057, 15, 1),
    (1058, 16, 1), (1060, 18, 2), (1062, 20, 2), (1064, 22, 2), (1066, 24, 2), (1068, 26, 2),
    (1070, 28, 2), (1292, 30, 2), (1296, 32, 4), (1300, 36, 4), (1304, 40, 4), (1088, 44, 1),
    (1089, 45, 1), (1090, 46, 1), (1080, 51, 1), (1081, 52, 1), (1083, 53, 1), (1084, 54, 1),
    (1085, 55, 1), (1092, 56, 1), (1093, 57, 1), (1094, 58, 1), (1095, 59, 1), (1096, 60, 1),
    (1110, 68, 1), (1111, 69, 1), (1113, 70, 1), (1112, 71, 1), (1118, 72, 2), (1120, 74, 2),
    (1114, 76, 2), (1116, 78, 2), (1123, 81, 1), (1091, 47, 1),
)
_DDE_UPD = (
    (680, 0, 1), (686, 1, 1), (707, 2, 1), (708, 3, 1), (692, 4, 4), (696, 8, 4),
    (700, 12, 2), (703, 14, 1), (704, 15, 1), (705, 16, 1), (706, 17, 1), (710, 18, 2),
    (712, 20, 2), (714, 22, 1), (715, 23, 1), (716, 24, 1), (717, 25, 1), (718, 26, 1),
    (720, 28, 2), (722, 30, 1), (681, 31, 1), (682, 32, 2), (684, 34, 2), (723, 36, 1),
    (724, 37, 1), (725, 38, 1), (726, 39, 1), (727, 40, 1), (728, 41, 1), (730, 42, 2),
    (732, 44, 2), (734, 46, 2), (736, 48, 2),
)
_DNR_UPD = (
    (1658, 1, 1), (1736, 2, 2), (1730, 4, 1), (1731, 5, 1), (1732, 6, 2), (1734, 8, 2),
    (1728, 10, 2), (1738, 12, 1), (1739, 13, 1), (1740, 14, 1), (1741, 15, 1), (1742, 16, 1),
    (1659, 17, 1), (1841, 20, 1), (1844, 22, 1), (1845, 23, 1), (1846, 24, 1), (1847, 25, 1),
    (1848, 26, 1),
)
# 40-byte weight curves shared by the DDE and NR refresh switches
_CURVES = (
    bytes.fromhex('ffe19b53230b03010000000000000000000000000000000000000000000000000000000000000000'),
    bytes.fromhex('fff1cc9b694023110703010000000000000000000000000000000000000000000000000000000000'),
    bytes.fromhex('fff9e7ccac8a694c3523160d07040201000000000000000000000000000000000000000000000000'),
    bytes.fromhex('fffaebd5b99b7c60473223170e090503020100000000000000000000000000000000000000000000'),
    bytes.fromhex('fffbefdcc4a98d715943312318100a06040201010000000000000000000000000000000000000000'),
    bytes.fromhex('fffbf1e1ccb49b816953402f2318110b070503020101000000000000000000000000000000000000'),
    bytes.fromhex('fffcf3e5d3bea78f78624e3d2e2319120c0806040201010000000000000000000000000000000000'),
    bytes.fromhex('fffcf5e9d9c6b19b85705c4a3b2d231a130d09060403020101000000000000000000000000000000'),
    bytes.fromhex('fffdf6ebddccb9a5907c695747392d231a140e0a0705030202010100000000000000000000000000'),
    bytes.fromhex('fffdf7eee1d2c0ae9b8775635344372c231b140f0b08060403020101010000000000000000000000'),
    bytes.fromhex('fffdf8f0e4d6c7b6a492806e5e4f42362b231b15100c090705030202010101000000000000000000'),
    bytes.fromhex('fffdf9f1e7dbccbcac9b8a79695a4c40352b231b16110d0a07050403020101010000000000000000'),
    bytes.fromhex('fffef9f3e9ded1c2b3a39382736456493e332a231c16110e0a080604030202010101000000000000'),
    bytes.fromhex('fffefaf4ebe1d5c8b9aa9b8b7c6e6053473c322a231c17120e0b0907050403020201010100000000'),
    bytes.fromhex('fffefbf6efe6dcd0c4b6a99b8d7f7165594d43393129231d1813100d0a0806050403020201010100'),
    bytes.fromhex('fffefbf7f1eae1d7ccc0b4a89b8e8175695d534940372f29231d1814110e0b090706050403020201'),
    bytes.fromhex('fffefcf8f3ede5ddd3c9beb2a79b8f83786c62584e453d352e28231e1915120f0c0a080706040403'),
    bytes.fromhex('fffefcf9f5efe9e1d9cfc6bbb1a69b90857a70665c534a423b342d28231e1a1613100d0b09080605'),
    bytes.fromhex('fffefdfaf6f1ebe5ddd5ccc3b9afa59b90867c726960574f474039322d27231e1a1714110e0c0a09'),
    bytes.fromhex('fffffdfbf7f3eee8e1dad2c9c0b7aea49b91877e756c635b534b443d37312c27231e1b1714120f0d'),
    bytes.fromhex('fff7e1c09b75533723140b0603010100000000000000000000000000000000000000000000000000'),
    bytes.fromhex('fff5d9b1855c3b231309040201000000000000000000000000000000000000000000000000000000'),
)
_NR_UPD_CURVE = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 2: 20}
_NR_REF_CURVE = {1: 1, 2: 20, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19}
_DDE_CURVE = {5: 1, 6: 21, 7: 20, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13}


HSZ = 0x7B8


class Handle:
    def __init__(self):
        self.b = bytearray(HSZ)
        self.buf = {}                  # pointer slot -> bytearray / numpy (pointees the port needs)

    # little helpers (signed/unsigned by format)
    def g(self, fmt, o): return struct.unpack_from("<" + fmt, self.b, o)[0]
    def s(self, fmt, o, v):
        if fmt in "bhiq":
            bits = {"b": 8, "h": 16, "i": 32, "q": 64}[fmt]
            v = ((v + (1 << (bits - 1))) & ((1 << bits) - 1)) - (1 << (bits - 1))
        else:
            bits = {"B": 8, "H": 16, "I": 32, "Q": 64}[fmt]
            v &= (1 << bits) - 1
        struct.pack_into("<" + fmt, self.b, o, v)
    def u8(self, o): return self.b[o]
    def u16(self, o): return self.g("H", o)
    def u32(self, o): return self.g("I", o)
    def w8(self, o, v): self.s("B", o, v)
    def w16(self, o, v): self.s("H", o, v)
    def w32(self, o, v): self.s("I", o, v)
    def w64(self, o, v): self.s("Q", o, v)


def create(w, h):
    """IFR_ISP_Create with the ability flags MicroSDK uses (all 0): IFR_ISP_InitSpec + CC_AlgParamsInit."""
    H = Handle()
    w16, w8, w32, w64 = H.w16, H.w8, H.w32, H.w64
    # IFR_ISP_InitSpec (byte offsets)
    w16(0x0, w); w16(0x20, 0); w16(0x2, h); w16(0x4, w); w16(0x6, h); w16(0x1C, 0); w16(0x22, 0)
    w8(0x1D9, 2); w32(0xC, w * h); w16(0x18, 0x101); w16(0x1A, 0x301); w16(0x3E, 0x400); w16(0x40, 0x50)
    w64(0x30, 0); w16(0x1E, 0); w16(0x42, 0x28); w16(0x38, 0x1F05); w16(0x3B, 0x2828)
    w16(0x66, 0x7F); w16(0x68, 4); w8(0x439, 0); w8(0x63, 0); w8(0x447, 3); w8(0x60, 0)
    w8(0x1D8, 0x32); w16(0x1D0, 0x32); w16(0x1D2, 0x32); w16(0x1D4, 0x32); w16(0x1D6, 0x3202); w16(0x1DE, 0)
    d29 = float(H.u16(0x68)); d25 = (H.u16(0x68) - 1) * 0.0 + 1.0
    import numpy as np
    w16(0x1DC, int(np.float32(np.float32(min(d29, d25)) * np.float32(1024.0))))
    # CC_AlgParamsInit
    w8(0x1D9, 6); w16(0x6, H.u16(0x2)); w16(0x4, H.u16(0x0)); w32(0xC, H.u16(0x2) * H.u16(0x0))
    w16(0x14, 0); w16(0x16, 0); w16(0x18, 0x101); w16(0x1A, 0x301)
    agc_alg_params_init(H)
    return H


def agc_alg_params_init(H):
    w8, w16, w32, w64 = H.w8, H.w16, H.w32, H.w64
    w8(0x464, 1); w8(0x580, 0); w8(0x3F8, 1); w8(0x7B0, 0); w16(0x400, 0x401)
    if H.u8(0x61) == 1:
        w8(0x401, 1)
    w16(0x424, 0)
    n = H.u32(0xC)
    w16(0x414, 0xFF3C); w16(0x403, 0x102); w64(0x40C, 0xB4157C11940200); w64(0x418, 0xF00500F005051000)
    w32(0x5A0, H.u16(0x6)); w16(0x42C, 0); w8(0x405, 1); w32(0x407, 0x1000200); w8(0x416, 0)
    w32(0x420, 0x90102); w32(0x448, 0xFF3C)
    u3 = ((((n * 0x96) & 0xFFFFFFFF) >> 0xC) * 0x97B42F6 >> 0x20) & 0xFFFF
    w64(0x430, 0x808000010000202); w16(0x428, 0); w64(0x438, 0x1919041041000008); w64(0x440, 0x2021900001E006E)
    w16(0x426, u3); w64(0x44E, 0x232010A1480010E); w16(0x42A, u3); w16(0x42E, u3); w8(0x44C, 0)
    w32(0x594, 0); w32(0x598, 0)
    u10 = ((n * 2) & 0xFFFFFFFF) // 2000
    w16(0x4E0, u10); w32(0x4E4, u10 * 9)
    u2 = ((n * 0x19) & 0xFFFFFFFF) // 200
    w32(0x4F4, u2); w32(0x4F8, u2 * 9)
    b11 = (H.u16(0x4) >> 4) & 0xFF
    w8(0x47A, b11); w16(0x478, 0x1010)
    b9 = (H.u16(0x6) >> 4) & 0xFF
    w8(0x47B, b9); w16(0x47C, b9 * b11); w32(0x59C, H.u16(0x4)); w16(0x4E8, 0x35); w32(0x10, 0xD297)
    w64(0x4FC, 0xECE200001A52); w32(0x4EC, 0x1DD); w8(0x4C3, 0); w8(0x466, 0)
    w64(0x584, 0x4300000002); w64(0x58C, 0x8F00000179)
    w16(0x520, (((n * 10) & 0xFFFFFFFF) // 200) * 5); w16(0x522, (n // 200) * 5)
    w16(0x524, (((n * 0x32) & 0xFFFFFFFF) // 200) * 5); w16(0x526, (((n * 2) & 0xFFFFFFFF) // 200) * 5)
    w16(0x57E, 0x401); w16(0x18, 0x101); w16(0x51E, (((n * 0x14) & 0xFFFFFFFF) // 2000) * 5)
    w16(0x1DE, 0); w8(0x1D0, 0x32); w8(0x1D2, 0x32); w8(0x1DA, 0x32); w8(0x1D4, 0x32)
    H.buf[0x638] = gbce_curve("gamma", 0x32)
    H.buf[0x650] = gbce_curve("contrast", H.u8(0x1D2))
    w8(0x662, 0)


def gbce_curve(kind, level):
    """comm_calcGbceCurve(min, max, level): identity at the default level 50."""
    import numpy as np
    if level == 0x32:
        return np.arange(256, dtype=np.uint8)
    raise Unsupported(f"{kind} curve at level {level} (needs DJI's curve tables)")


# ---------------------------------------------------------------------------------------------
# blob loading (IFR_ISP_SetDefaultParam -> IFR_ISP_Load{AGC,DDE,DNR}Params)



MAGIC_AGC, MAGIC_DDE, MAGIC_DNR = 0xAAAAAAAA, 0xDDDDDDDD, 0x3D3D3D3D


def _words(blob):
    return list(struct.unpack_from(f"<{len(blob) // 4}I", blob))


def _per_set(nwords, lo, mid, hi, small):
    n = (nwords * 4 - 8) // 4 & 0xFFFF       # (size - 8) >> 2 as ushort
    if n < hi:
        if n < mid:
            return small if not lo < n else (n >> 1) & 0x7FFF
        return n >> 2
    return n // 6


def _record(words, base, layout, size, extra=()):
    rec = bytearray(size)
    for off, wi, width in layout:
        v = words[base + wi] & ((1 << 8 * width) - 1)
        rec[off:off + width] = v.to_bytes(width, "little")
    return rec


def set_default_param(H, blob):
    w = _words(blob)
    magic = w[0]
    H.w8(0x65C, 1)
    if magic == MAGIC_AGC:
        k = _per_set(len(w), 0x69, 0xD4, 0x13E, 0x37)
        layout = [e for e in _AGC_LOAD if e[0] < 0x50]
        tail = [e for e in _AGC_LOAD if e[0] >= 0x50]
        slots = [0x260, 0x270, 0x268, 0x278]
        for i, slot in enumerate(slots):
            base = 1 + i * k
            rec = _record(w, base, layout, 0x60)
            if k >= 0x36:
                for off, wi, width in tail:
                    rec[off] = w[base + wi] & 0xFF
            H.buf[slot] = rec
        if 1 + 4 * k < len(w) and w[1 + 4 * k] != MAGIC_AGC:
            raise Unsupported("AGC blob with a fifth parameter set")
    elif magic == MAGIC_DDE:
        k = _per_set(len(w), 0x41, 0x84, 0xC6, 0x21)
        for i, slot in ((0, 0x248), (2, 0x250)):
            H.buf[slot] = _record(w, 1 + i * k, _DDE_LOAD, 0x40)
        if 1 + 4 * k < len(w) and w[1 + 4 * k] != MAGIC_DDE:
            raise Unsupported("DDE blob with a third parameter set")
    elif magic == MAGIC_DNR:
        k = _per_set(len(w), 0x2B, 0x58, 0x84, 0x16)
        if k != 0x17:
            raise Unsupported("DNR blob layout other than 23 words per set")
        for i, slot in ((0, 0x290), (2, 0x298)):
            base = 1 + i * k
            rec = _record(w, base, [e for e in _DNR_LOAD if e[0] < 0x13], 0x20)
            for j in range(8):                                     # the 23-word layout
                rec[0x13 + j] = w[base + 15 + j] & 0xFF
            H.buf[slot] = rec
        if 1 + 4 * k < len(w) and w[1 + 4 * k] != MAGIC_DNR:
            raise Unsupported("DNR blob with a third parameter set")
    else:
        raise ValueError("unknown ISP parameter blob")


# ---------------------------------------------------------------------------------------------
# SetConfig keys used by MicroSDK on the hirp path

def set_param(H, k, v):
    w8, w16 = H.w8, H.w16
    flags = lambda bit: w16(0x24, H.u16(0x24) | bit)  # noqa: E731
    if k == 0x529:
        w8(0x18, v)
    elif k == 0x216:
        w16(0x8, v)
    elif k == 0x217:
        w16(0xA, v)
    elif k == 0xA4:
        w16(0x142, v)
    elif k == 0x218:
        w8(0x662, v)
    elif k == 0x20A:
        flags(4); w8(0x60, v); w8(0x400, v); w8(0x57E, v)
    elif k == 0x20B:
        w8(0x61, v)
        if not (v & 0xFF == H.u8(0x62) and H.u32(0x14) != 0):
            w32 = H.w32; w32(0x14, 0); flags(4)
    elif k == 0x20C:                                 # manual AGC upper raw limit
        w16(0x82, v); w16(0x410, v)
    elif k == 0x20D:                                 # manual AGC lower raw limit
        w16(0x80, v); w16(0x40E, v)
    elif k in (0x21A, 0x21B):
        w8({0x21A: 0x2E4, 0x21B: 0x3F9}[k], v)
    elif k == 0x21C:
        w8(0x3FA, v)
        if H.u8(0x407) == 1:
            H.w32(0x3FC, 0x501)
        elif H.u8(0x3F9) == 1:
            H.w32(0x3FC, ((v & 0xFF) * 0x14800) >> 0xE)
        else:
            H.w32(0x3FC, 0)
    elif k == 0x300:
        flags(2); H.w32(0x14, 0); w8(0x1D0, v); w16(0x1DE, ((v - 0x32) * 0x7F) // 0x32 if v >= 0x32 else -(((0x32 - v) * 0x7F) // 0x32))
    elif k == 0x301:
        flags(2); H.w32(0x658, H.u32(0x658) | 0x10); H.w32(0x14, 0); w8(0x1D2, v)
    elif k == 0x302:
        flags(2); w8(0x1D4, v)
    elif k == 0x306:
        w8(0x1D6, v)
    elif k == 0x307:
        w8(0x1D7, v)
    elif k == 0x308:
        w8(0x1D8, v)
    elif k == 0x701:
        w8(0x679, v)
    elif k == 0x740:
        w8(0x67B if H.u8(0x18) == 1 else 0x731, v)
    elif k == 0x800:
        flags(1); w8(0x7B0, v)
    else:
        raise Unsupported(f"ISP key {k:#x}")


# ---------------------------------------------------------------------------------------------
# IFR_ISP_UpdateISPParams (+ Update{Agc,Dde,Dnr,Scene}Params) and CC_UpdateConfig (the refreshes)

def _w40(H, off, b):
    H.b[off:off + 40] = b


def update_isp_params(H):
    lo, hi = H.g("h", 0x8), H.u16(0xA)
    if H.u8(0x65C) == 1:
        if lo < 0 or (lo == 0 and hi in (800, 1000)):
            mode = 1
        elif hi < 0x709:
            mode = 2
        else:
            mode = 3
    else:
        mode = 1 if hi < 0xFB else (3 if hi > 999 else 2)
    H.w8(0x18, mode)
    fl = H.u16(0x24)
    if H.u8(0x19) != mode:
        fl |= 1
        H.w16(0x24, fl)
    if fl:
        if fl & 4:
            update_agc_params(H)
            fl = H.u16(0x24)
        if fl & 1:
            update_agc_params(H)
            update_dde_params(H)
            update_dnr_params(H)
            if H.u8(0x7B0) == 1:
                raise Unsupported("ISP scene mode")
    H.w32(0x14, H.u32(0x14) + 1)
    H.w16(0x24, 0)


def _copy(H, rec, table):
    for dst, src, width in table:
        H.b[dst:dst + width] = rec[src:src + width]


def update_agc_params(H):
    w, h = H.g("h", 0x4), H.g("h", 0x6)
    if H.u8(0x7B0) == 1 or H.u8(0x65C) != 1:
        raise Unsupported("AGC params without the file's parameter blob / scene mode")
    mode, c9 = H.u8(0x18), H.u8(0x57E)
    if mode == 3:
        raise Unsupported("AGC range mode 3")
    if mode == 2:
        rec8, rec11, b3 = H.buf[0x268], H.buf[0x278], True
    else:
        rec8, rec11, b3 = H.buf[0x260], H.buf[0x270], False
    if c9 == 1:
        rec = rec8
    elif c9 == 2:
        rec = rec11
    else:
        rec = None
    if rec is not None:
        _copy(H, rec, _AGC_UPD)
        c10 = rec[0x50]
    else:
        c10 = H.u8(0x462)
    special = (w == 0x180 and h == 0x120 and H.u8(0x463) == 1)
    if c10 == 0x28 and b3 and special:
        raise Unsupported("AGC gamma override for 384x288")
    if c10 == 10 and H.u8(0x18) == 1 and special:
        raise Unsupported("AGC gamma override for 384x288")
    if (w == 0x1E0 and h == 0x168):
        raise Unsupported("AGC rescale for 480x360")
    H.w8(0x406, 1)
    H.w16(0x404, 0x101)
    if H.u8(0x61) == 1:
        H.w8(0x401, 1); H.w8(0x443, 0); H.w8(0x462, 0)
    elif H.u8(0x401) != 2:
        H.w8(0x401, 3)
    if H.u8(0x407) == 1 and w == 0x180:
        raise Unsupported("AGC dual curve")


def update_dde_params(H):
    if H.u8(0x65C) != 1:
        raise Unsupported("DDE params without the file's parameter blob")
    mode = H.u8(0x18)
    rec = H.buf[{1: 0x248, 2: 0x250}.get(mode, 0x258)]
    _copy(H, rec, _DDE_UPD)
    H.b[0x2E5:0x2E5 + 0x80] = bytes(range(0x80))      # the HfFix base table: identity
    H.b[0x38E:0x396] = bytes(8)
    H.w16(0x396, 0)


def update_dnr_params(H):
    if H.u8(0x65C) != 1:
        raise Unsupported("DNR params without the file's parameter blob")
    mode = H.u8(0x18)
    rec = H.buf[{1: 0x290, 2: 0x298}.get(mode, 0x2A0)]
    H.w8(0x678, 1); H.w8(0x730, 1)
    _copy(H, rec, _DNR_UPD)
    b8 = rec[0x12]
    H.w8(0x67D, b8)
    c1 = rec[0x13]
    H.w8(0x6A7, c1)
    u2 = rec[0x15]
    H.w8(0x732, u2)
    if c1 != 0:
        H.w8(0x67C, 1)
    w = H.u16(0x4)
    if H.u8(0x679) == 0 and (((w - 0x180) & 0xFEFF) == 0 or w == 0x1E0):
        b8 = (b8 + 0xF) & 0xFF
        H.w8(0x67D, b8)
    H.w8(0x67E, b8)
    H.w8(0x733, u2)
    idx = ((b8 * 0xCD) & 0xFFFF) >> 8 >> 2
    if idx in _NR_UPD_CURVE:                    # other indices: the switch writes nothing
        _w40(H, 0x67F, _CURVES[_NR_UPD_CURVE[idx]])


def cc_update_config(H):
    agc_params_refresh(H)
    dde_params_refresh(H)
    nr_params_refresh(H)
    if H.u8(0x7B0) == 1:
        H.w32(0x4E4, (H.u32(0xC) * 4 & 0xFFFFFFFF) // 10000)
    H.w8(0x19, H.u8(0x18))


def agc_params_refresh(H):
    if H.u8(0x407) == 1:
        raise Unsupported("AGC dual-curve refresh")
    H.w8(0x580, 0)
    H.w8(0x57E, H.u8(0x400))
    if H.u16(0x142) == 1:
        H.w8(0x582, H.u8(0x445)); H.w8(0x444, 1)
    else:
        H.w8(0x582, 1); H.w8(0x444, 0)
    c11 = H.u8(0x420)
    if H.u8(0x65C) == 0:
        raise Unsupported("AGC refresh without the parameter blob")
    # roi pointer 0x5a8 -> handle + 0x594
    if c11 == 4:
        raise Unsupported("AGC gradient weighting")
    if H.u8(0x404) == 1 and (H.u32(0x658) & 8):
        H.buf[0x638] = gbce_curve("gamma", H.u8(0x1DA)); H.w32(0x658, H.u32(0x658) & 0xFFF7)
    if H.u8(0x405) == 1 and (H.u32(0x658) & 0x10):
        H.buf[0x650] = gbce_curve("contrast", H.u8(0x1D2)); H.w32(0x658, H.u32(0x658) & 0xFFEF)
    H.w8(0x57F, H.u8(0x57E)); H.w8(0x19, H.u8(0x18)); H.w8(0x62, H.u8(0x61))


_DDE_GAUSS = {0: (0xA50EB600A50000, 0), 1: (0x1B40C9601B40001, 1), 2: (0x2A60A9E02A6000B, 0xB),
              3: (0x349091E03490028, 0x28), 4: (0x3A807FC03A8005A, 0x5A), 5: (0x3D7071E03D7009A, 0x9A),
              6: (0x3E8067203E800DF, 0xDF), 7: (0x3E905EA03E90122, 0x122), 8: (0x3E2057E03E2015F, 0x15F),
              9: (0x3D7052803D70195, 0x195), 10: (0x3CA04E403CA01C4, 0x1C4)}
_DDE_SMALL = {0: (0x23FF, 0, 0, 0, 0), 1: (0x9BFF, 0x323, 0, 0, 0), 2: (0xCCFF, 0x2369, 0x107, 0, 0),
              3: (0xE1FF, 0x539B, 0xB23, 0x103, 0), 4: (0xEBFF, 0x7CB9, 0x2347, 0x50E, 2)}


def dde_params_refresh(H):
    b2 = H.u8(0x1D4)
    if b2 < 0x32:
        s4 = (b2 * 0x100) // 0x32
    else:
        u3 = ((b2 * 3 - 0x96) * 0x100) & 0xFFFFFFFF
        u3s = u3 - (1 << 32) if u3 >> 31 else u3
        s4 = 0x3FF if not u3s < 0x9600 else (u3 // 0x32 + 0x100)
    H.w16(0x2C6, s4)
    g = _DDE_GAUSS.get(H.u8(0x2AE))
    if g is not None:
        H.w64(0x38E, g[0]); H.w16(0x396, g[1])
    k = min(H.u8(0x2C4), 0x13)
    if k in _DDE_SMALL:
        a, b, c, d, e = _DDE_SMALL[k]
        H.w64(0x365, a | b << 16 | c << 32 | d << 48)
        H.w64(0x36D, e); H.w64(0x385, 0); H.w64(0x375, 0); H.w64(0x37D, 0)
    else:
        _w40(H, 0x365, _CURVES[_DDE_CURVE[k]])


def nr_params_refresh(H):
    if H.u8(0x1D6) == 0:
        H.w8(0x679, 0); H.w8(0x67D, 0); H.w8(0x734, 0); H.w8(0x731, 0)
        _w40(H, 0x67F, _CURVES[_NR_UPD_CURVE[0]])
        return
    b3 = H.u8(0x1D8)
    H.w8(0x67A, H.u8(0x1D7))
    if H.u8(0x18) == 1:
        b1 = H.u8(0x67E)
        if b3 < 0x33:
            b3 = (b3 * b1) // 0x32
        else:
            b3 = int(((100 - b1) * b3) / 0x32) + (0x32 - b1) * -2
        b3 &= 0xFF
        H.w8(0x67D, b3)
    else:
        b1 = H.u8(0x733)
        if b3 < 0x33:
            H.w8(0x732, (b3 * b1) // 0x32)
        else:
            H.w8(0x732, int(((100 - b1) * b3) / 0x32) + (0x32 - b1) * -2)
        b3 = H.u8(0x67D)
    idx = (((b3 + (b3 + b3 * 4) * 8) * 5) & 0xFFFFFFFF) >> 8 & 0xFF
    idx >>= 2
    if idx == 0:
        _w40(H, 0x67F, _CURVES[_NR_REF_CURVE[1]])
    elif idx in _NR_REF_CURVE:
        _w40(H, 0x67F, _CURVES[_NR_REF_CURVE[idx]])


def build(w, h, blobs, keys):
    """The handle after MicroSDK's ISPProcess setup and the first IFR_ISP_Process parameter update."""
    H = create(w, h)
    for b in blobs:
        set_default_param(H, b)
    for k, v in keys:
        set_param(H, k, v)
    update_isp_params(H)
    cc_update_config(H)
    return H


# ---------------------------------------------------------------------------------------------
# the hirp path

def param_blobs(app5: bytes) -> list[bytes]:
    """The AGC / DDE / DNR parameter files of the ZIP embedded in the APP5 "HIDR" container
    (MicroSDK::ISPProcess loads every entry, in ZIP order)."""
    i = app5.find(b"PK\x03\x04")
    if i < 0:
        raise Unsupported("hirp R-JPEG without the ISP parameter archive")
    try:
        z = zipfile.ZipFile(io.BytesIO(app5[i:]))
        return [z.read(n) for n in z.namelist()]
    except (zipfile.BadZipFile, OSError) as e:
        raise Unsupported(f"unreadable ISP parameter archive: {e}") from e


def _i32(v):
    return ((v + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


def temp_range(tp: bytes) -> tuple[int, int]:
    """The gain mode's temperature range (low, high) in whole degrees from TempParams (MHandle:
    limits << 9, then (int)((float)limit / 512))."""
    lo, hi = struct.unpack_from("<ii", tp, 0x40)
    f = lambda v: int(np.float32(np.float32(_i32(v << 9)) * np.float32(0.001953125)))  # noqa: E731
    return f(lo), f(hi)


def isp_keys(tp: bytes, brightness: int = 50, manual: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    """The SetConfig sequence MicroSDK issues for the ISP that renders the image (ISPProcess
    constructor, initPalette, initDontRepeat, GISPProcess), with the gain range from TempParams
    (MHandle: mode = gain + 1, limits << 9, then (int)((float)limit / 512)). manual: the colour
    bar's (lower, upper) raw limits (ISPProcess::setManualAGCConfig)."""
    mode = (tp[0x34] + 1) & 0xFF
    lo, hi = temp_range(tp)
    rng = [(0x529, mode), (0x216, lo), (0x217, hi)]
    enhance = [(0x302, 50), (0x300, brightness), (0x301, 50)]
    if manual is None:
        mid, limits = enhance + [(0x20A, 2), (0x20B, 0)], []
    else:
        mid, limits = [(0x20A, 2), (0x20B, 1)] + enhance, [(0x20D, manual[0]), (0x20C, manual[1])]
    return (rng + [(0xA4, 1), (0x701, 0), (0x740, 0), (0x306, 1), (0x307, 50), (0x740, 1), (0x308, 50)]
            + mid + [(0x306, 1), (0x307, 50), (0x740, 1), (0x308, 50)]
            + rng + [(0x21B, 0), (0x21C, 0), (0x21A, 0)] + limits + [(0x218, 0), (0x800, 0)])


def render(raw14: np.ndarray, app5: bytes, tp: bytes, brightness: int = 50, manual: tuple[int, int] | None = None):
    """The ISP: (8-bit image, IFR output (14-bit), the AGC's 14->8 bit LUT, AGC outputs v, handle
    bytes). manual: (lower, upper) raw limits of a manual colour bar."""
    h, w = raw14.shape
    H = build(w, h, param_blobs(app5), isp_keys(tp, brightness, manual))
    hb = H.b
    raw = ifr_process(hb, raw14)
    y8, v, lut, _ = agc_run(hb, raw, raw14 & 0x3FFF, H.buf)
    y8 = dde_add(hb, raw, y8, dde_get_raw_detail(hb, raw))
    return nr8(hb, y8), raw, lut, v, hb


def gray(raw14: np.ndarray, app5: bytes, tp: bytes, brightness: int = 50, manual: tuple[int, int] | None = None):
    """The ISP's 8-bit image and the AGC frame range (min, max raw)."""
    y8, _raw, _lut, v, _hb = render(raw14, app5, tp, brightness, manual)
    return y8, (v[3], v[2])


# ---------------------------------------------------------------------------------------------
# Isotherm colouring (CE_ColorBarCalc_Process + CE_ColorMap_Process with alarm type 9): pixels
# whose raw value lies in a band get the band's palette, stretched over the band's grey levels;
# the rest get palette 0.

UNKNOWN = -1      # a LUT entry that depends on the byte the native code reads past the LUT


def ca_preprocess(ext: np.ndarray, lo: list, hi: list, minr: int, maxr: int, c_hi: int, c_lo: int):
    """CE_CA_preProcess: marks the band edges in the 14->8 bit LUT. ext: the LUT as an int array
    with one entry before it (the top byte of the heap chunk's size field: 0) and one after it
    (UNKNOWN); entries that come to depend on the latter become UNKNOWN as well. Writing a
    non-zero value before the LUT corrupts the heap natively."""
    class _Lut:
        def __getitem__(self, i):
            return ext[i + 1]

        def __setitem__(self, i, v):
            ext[i + 1] = v
    lut = _Lut()
    def edge(u, then):
        """lut[u] != 0 ? then() : (lut[u-1] == 0 and lut[u-1] = lut[u]), with unknowns."""
        if lut[u] == UNKNOWN:
            lut[u] = lut[u - 1] = UNKNOWN
        elif lut[u] != 0:
            then()
        elif lut[u - 1] == 0:
            lut[u - 1] = lut[u]

    def put(u, c):
        lut[u] = c
        lut[u - 1] = lut[u]
    for k in range(len(lo)):
        u = lo[k]
        if u < minr:
            edge(u, lambda: put(u, c_lo))
        elif maxr < u:
            edge(u, lambda: put(u, c_hi))
        elif lut[u - 1] == 0:                 # (an UNKNOWN lut[u - 1] stays UNKNOWN)
            lut[u - 1] = lut[u]
        u = hi[k]
        c = int(lut[u])
        if minr <= u:
            if u <= maxr or c == 0:
                lut[u - 1] = c
            elif c == UNKNOWN:
                lut[u] = lut[u - 1] = UNKNOWN
            else:
                put(u, c_hi)
        elif c == UNKNOWN:
            lut[u] = lut[u - 1] = UNKNOWN
        elif c != 0:
            put(u, (c_lo + 1) & 0xFF)
        else:
            lut[u - 1] = c
    if lut[maxr] == UNKNOWN:
        lut[0] = UNKNOWN
    elif lut[maxr] == 0:
        lut[0] = 0xFF
    if ext[0] != 0:
        raise UndefinedBehaviour("isotherm band from gray 0: the native SDK writes before its LUT")


def ca_map_bar_isotherm(lut: np.ndarray, lo: list, hi: list, minr: int, maxr: int):
    """CA_CreateMapBar (type 9): bands clipped to the frame's raw range, and each band's grey
    levels (LUT values at its limits). Returns (lo, hi, grey lo, grey hi); bands outside the
    frame's range are skipped (grey levels None: never written natively)."""
    lo, hi = list(lo), list(hi)
    glo, ghi = [None] * len(lo), [None] * len(lo)
    for k in range(len(lo)):
        if lo[k] == 0 or maxr < lo[k] or hi[k] < minr:
            continue
        if maxr == minr:
            raise UndefinedBehaviour("isotherm colour bar of a frame with a single raw value (division by zero)")
        lo[k] = maxr if max(lo[k], minr) >= maxr else max(lo[k], minr)
        hi[k] = maxr if max(hi[k], minr) >= maxr else max(hi[k], minr)
        glo[k], ghi[k] = int(lut[lo[k] - 1]), int(lut[hi[k] - 1])
    return lo, hi, glo, ghi


def ca_mask_isotherm(raw: np.ndarray, lo: list, hi: list) -> np.ndarray:
    """CA_CreatCAMaskImg (type 9): per horizontal pixel pair, 1 + the last band containing the
    pair's mean raw value (0: none)."""
    r = raw.astype(np.int64)
    m = (r[:, 0::2] + r[:, 1::2]) >> 1
    mask = np.zeros(m.shape, np.uint8)
    for k in range(len(lo)):
        if lo[k] != 0:
            mask[(lo[k] <= m) & (m <= hi[k])] = k + 1
    return mask


def ca_color_map_isotherm(y8: np.ndarray, mask: np.ndarray, glo: list, ghi: list, base: int,
                          outside: np.ndarray, bands: list) -> np.ndarray:
    """CA_ColorMap (colour bar mode 1, type 9) -> NV16. base: the grey level of the frame's
    minimum (the io block's second byte)."""
    h, w = y8.shape
    a = y8[:, 0::2].astype(np.int64)
    b = y8[:, 1::2].astype(np.int64)
    ca, cb = outside[a], outside[b]
    for k in range(len(glo)):
        sel = mask == k + 1
        if not sel.any():
            continue
        lo8, hi8 = glo[k], ghi[k]
        span = (hi8 - lo8) & 0xFF or 1
        top = hi8 + base

        def idx(v):
            v = np.minimum(np.maximum(v, lo8 + base), top)
            return np.minimum(base + ((v - lo8 - base) * 0xFF) // span, 0xFF)
        ca = np.where(sel, bands[k][idx(a)], ca)
        cb = np.where(sel, bands[k][idx(b)], cb)
    Y = np.empty((h, w), np.int64)
    Y[:, 0::2], Y[:, 1::2] = ca & 0xFF, cb & 0xFF
    U = ((ca >> 24) + (cb >> 24)) >> 1
    V = (((ca >> 8) & 0xFF) + ((cb >> 8) & 0xFF)) >> 1
    return np.concatenate([Y.reshape(-1), np.stack([U, V], -1).reshape(-1)]).astype(np.uint8)


def colorize_isotherm(y8: np.ndarray, raw: np.ndarray, lut: np.ndarray, v: list, hb, bands: list,
                      outside_rgb: np.ndarray, band_rgb: np.ndarray) -> np.ndarray:
    """MicroSDK paletteProcess with isotherm bands [(lower, upper) raw]: band pixels in
    band_rgb's colours, the rest in outside_rgb's."""
    lo = [l for l, _ in bands]
    hi = [u for _, u in bands]
    for k in range(len(lo)):
        if hi[k] <= lo[k]:
            lo[k] = hi[k] = 1
    minr, maxr = v[3], v[2]
    ext = np.concatenate([[0], lut.astype(np.int64), [UNKNOWN]])
    ca_preprocess(ext, lo, hi, minr, maxr, hb[0x47E], hb[0x47F])
    lut = ext[1:]
    lo, hi, glo, ghi = ca_map_bar_isotherm(lut, lo, hi, minr, maxr)
    if UNKNOWN in glo + ghi:
        raise UnsupportedFeature("hirp isotherm band whose colours depend on memory the native SDK reads "
                                 "past its LUT")
    mask = ca_mask_isotherm(raw, lo, hi)
    for k in range(len(lo)):
        if glo[k] is None and (mask == k + 1).any():
            raise UnsupportedFeature("hirp isotherm band outside the colour bar's range: the native SDK "
                                     "colours it from uninitialised memory")
    table = palette_yuv(band_rgb).astype(np.int64)
    nv = ca_color_map_isotherm(y8, mask, glo, ghi, hb[0x47F], palette_yuv(outside_rgb).astype(np.int64),
                               [table] * len(lo))
    return nv16_to_rgb(nv, y8.size).reshape(y8.shape + (3,))


def colorize(y8: np.ndarray, rgb256: np.ndarray) -> np.ndarray:
    """PaletteProcessFilter + MicroSDK::toRGB for one (256, 3) RGB palette."""
    nv = color_map_nv16(y8, palette_yuv(rgb256))
    return nv16_to_rgb(nv, y8.size).reshape(y8.shape + (3,))
