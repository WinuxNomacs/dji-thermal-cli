"""The dirp plugin's image pipeline (libv_dirp process / process_strech; Zenmuse H20T and XT S).

libv_dirp runs a small node graph over images of doubles:

- sen: raw / 2^(bits - 8), i.e. the frame in an 8-bit scale (16-bit H20T, 14-bit XT S raw);
- isotherm: a mask of the pixels inside the isotherm band;
- str_pc: a linear stretch of [low, high] onto [0, 255.99609375). In automatic mode low/high come
  from a 1024-bin histogram (0.5 / 255.496 of 256 percentiles, bounded by the frame's min/max)
  and are then snapped onto the raw values of whole tenths of a degree (the "discrete LUT");
- cm_yuv: interpolated lookup in a 256-entry YUV palette (doubles), with the isotherm palette
  inside/outside the band, and the brightness gain on Y;
- y2r: BT.601 YUV -> RGB; camera: truncation to bytes.

process_strech stops after str_pc and returns the stretched frame as float32. Everything is
double precision in the native code, evaluated in the same order here, so results are
bit-identical.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import UndefinedBehaviour

TOP = 255.99609375          # 256 - 1/256: the pipeline's 8-bit full scale
_HIST_LO, _HIST_HI = 0.5, 255.49609375
_MARGIN = 0.25


def to_raw(v: float, bits: int) -> int:
    """v >= 0 as a raw value of `bits` bits: floor(v * 2^(bits-8)), saturating at 2^bits - 1."""
    scale = float(1 << (bits - 8))
    top = 256.0 - 1.0 / scale
    if v < 0.0:
        q = 0.0                     # unsigned mode: negatives become 0
    elif top <= v:
        q = top
    else:
        q = math.floor(v * scale) / scale
    return int(q / (1.0 / scale) + 0.5)


def auto_range(img: np.ndarray) -> tuple[float, float]:
    """(low, high) of the automatic stretch."""
    v = img.reshape(-1)
    vmax = max(float(v.max(initial=0.0)), 0.0)
    vmin = min(float(v.min(initial=TOP)), TOP)
    idx = (v * 0.00390625 * 1024.0).astype(np.int64)
    if idx.size and (idx.min() < 0 or idx.max() > 1023):
        raise UndefinedBehaviour("dirp histogram index outside its 1024 bins")
    counts = np.bincount(idx, minlength=1024).astype(np.float64)
    total = 0.0
    for c in counts:                      # sequential double sum, as natively
        total += c
    if total == 0.0:
        total = 1.0
    norm = (counts * 256.0) / total

    def point(p):
        cum = 0.0
        for i in range(1024):
            cum += norm[i]
            if p <= cum:
                return float(i) * 0.0009765625 * 256.0
        return 1023.0 * 0.0009765625 * 256.0

    lo_p, hi_p = point(_HIST_LO), point(_HIST_HI)
    if vmin == vmax:
        lo = vmin - _MARGIN
        hi = vmax + _MARGIN
        return (lo if 0.0 <= lo else 0.0), (hi if hi <= TOP else TOP)
    if hi_p <= lo_p:
        return vmin, vmax
    return (lo_p if vmin <= lo_p else vmin), (hi_p if hi_p <= vmax else vmax)


def snap_to_lut(lo: float, hi: float, bits: int, table) -> tuple[float, float]:
    """Move the automatic limits onto discrete-LUT raw values, following the SDK's search step
    by step. Unchanged when the LUT is missing or does not cover the range."""
    u1, u2 = to_raw(lo, bits), to_raw(hi, bits)
    n = 0 if table is None else len(table)
    if n == 0 or u2 <= u1 or u1 < int(table[0]) or int(table[n - 1]) < u2:
        return lo, hi
    t = [int(x) for x in table]
    last = n - 1
    u11 = u2
    if last != 0:
        k = 0
        if u1 == t[0]:
            u11, u7 = t[0], u1
        else:
            exact = False
            u8 = t[1]
            if t[0] < u1 <= u8:
                k = 1
            else:
                j = 1
                while True:
                    if last == j:
                        k = -1          # not found
                        break
                    u11 = t[j]
                    if u1 == u11:
                        exact, k = True, j
                        break
                    j += 1
                    u8 = t[j]
                    if not (u8 < u1 or u1 <= u11):
                        k = j
                        break
            if k == -1:
                k = 0
                u11, u7 = t[0], u1
            elif exact:
                u7 = t[k]               # u11 is t[k] already
            else:
                u1 = u8
                if last <= k:
                    return u1 / float(1 << (bits - 8)), u2 / float(1 << (bits - 8))
                u11, u7 = t[k], u1
        u1 = u7
        if u11 != u2:
            j = k + 1
            u7 = t[j]
            while True:
                u8 = u7
                if not (u8 <= u2 or u2 <= u11):
                    break
                if not (u2 <= u11):
                    u11 = u8
                    if u2 == u8:
                        break
                u11 = u2
                if not j < last:
                    break
                u11 = u8
                if u2 == u8:
                    break
                j += 1
                u7 = t[j]
    s = float(1 << (bits - 8))
    return u1 * (1.0 / s), u11 * (1.0 / s)


def stretch(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """str_pc's mapping (float mode)."""
    if not lo < hi:
        return img.copy()
    k = TOP / (hi - lo)
    out = (img - lo) * k
    out = np.where(out <= 0.0, 0.0, np.where(out < TOP, out, TOP))
    out = np.where(img < lo, 0.0, np.where(img <= hi, out, TOP))
    return out


def _interp(v: np.ndarray, tab: np.ndarray) -> np.ndarray:
    i = np.minimum(v.astype(np.int64), 0xFF)
    j = np.where(i == 0xFF, 0xFF, i + 1)
    f = np.minimum(v - i, (j - i).astype(np.float64))
    a = tab[i]
    y = (tab[j] - a) * f + a
    return np.where(y <= 0.0, 0.0, np.where(TOP <= y, TOP, y))


def process(raw: np.ndarray, bits: int, palettes: np.ndarray, y2r: np.ndarray, palette: int, brightness: int,
            iso: tuple, cb: tuple, lut_raws, strech: bool = False):
    """The libv_dirp graph. iso = (enable, high raw, low raw); cb = (automatic, high raw, low raw).
    Returns (rgb uint8 (h, w, 3) or float32 stretched frame, (low raw, high raw) of the stretch)."""
    shape = raw.shape
    scale = 1.0 / float(1 << (bits - 8))
    div = math.pow(2.0, bits - 8)
    v = scale * raw.reshape(-1).astype(np.float64)                         # sen
    iso_on, iso_hi, iso_lo = iso
    mask = (iso_lo / div <= v) & (v <= iso_hi / div)                      # isotherm (always run)
    auto, cb_hi, cb_lo = cb
    if auto:
        lo, hi = auto_range(v)
        lo, hi = snap_to_lut(lo, hi, bits, lut_raws)
    else:
        lo, hi = cb_lo / div, cb_hi / div
    s = stretch(v, lo, hi)                                                # str_pc
    limits = (int(lo * div), int(div * hi))
    if strech:
        return s.astype(np.float32).reshape(shape), limits
    if not 0 <= palette < len(palettes):
        raise UndefinedBehaviour("dirp palette index past its palette table")
    pal = palettes[palette]                                               # (3, 256): Y, U, V
    ipal = palettes[2] if palette in (0, 9) else palettes[0]
    yuv = []
    for c in range(3):                                                    # cm_yuv
        a, b = _interp(s, pal[c]), _interp(s, ipal[c])
        if iso_on:
            a = np.where(mask, b, a) if palette in (0, 9) else np.where(mask, a, b)
        yuv.append(a)
    gain = (float(brightness & 0xFF) * 1.4) / 100.0 + 0.3
    y = gain * yuv[0]
    yuv[0] = np.where(y <= 0.0, 0.0, np.where(TOP <= y, TOP, y))
    m = y2r                                                               # y2r (BT.601)
    yy, uu, vv = yuv[0] - m[9], yuv[1] - m[10], yuv[2] - m[11]
    r = m[0] * yy + m[1] * uu + m[2] * vv
    g = m[3] * yy + m[4] * uu + m[5] * vv
    b = yy * m[6] + uu * m[7] + vv * m[8]
    rgb = np.stack([np.where(c <= 0.0, 0.0, np.where(TOP <= c, TOP, c)) for c in (r, g, b)], axis=1)
    out = np.floor(rgb).astype(np.uint8)                                  # camera
    return out.reshape(shape + (3,)), limits


def pseudo_color_lut(palettes: np.ndarray) -> np.ndarray:
    """The ten YUV palettes as (3, 10, 256) RGB bytes (BT.601, rounded)."""
    y = (palettes[:, 0] - 16.0) * 1.1644
    u = palettes[:, 1] - 128.0
    v = palettes[:, 2] - 128.0
    r = u * 0.0 + y + v * 1.596 + 0.5
    g = u * -0.3918 + y + v * -0.813 + 0.5
    b = v * 0.0 + y + u * 2.0172 + 0.5

    def byte(c):
        return np.where(c > 0.0, np.where(c < TOP, np.trunc(np.where(c > 0.0, c, 0.0)), 255.0), 0.0).astype(np.uint8)

    return np.stack([byte(r), byte(g), byte(b)])
