"""The girp plugin's image pipeline (libv_girp process / process_strech).

A linear 8-bit stretch of the signed raw frame between its 1 % / 99 % points (or the manual
colour bar's limits, converted to raw), widened to at least 128 counts, with the brightness
setting as the offset; then one of eleven 256-entry palettes (stored BGR). An isotherm keeps
the palette inside its band and shows palette 0 elsewhere. After an automatic run the plugin
stores the 0.2 % / 99.8 % temperatures as the colour bar, which get_color_bar then reports.

Double/float placement and C integer division follow libv_girp, so the output is
bit-identical to the native SDK.
"""

from __future__ import annotations

import numpy as np

from .errors import UndefinedBehaviour

F = np.float32


def _tail_points(raw: np.ndarray, count: int) -> tuple[int, int]:
    """(high, low) raw values where the histogram's top/bottom tails first reach `count`
    pixels (libv_girp scans both ends of a 65536-bin histogram at once)."""
    hist = np.bincount(raw.reshape(-1).astype(np.int64) + 0x8000, minlength=0x10000)
    up = np.cumsum(hist[::-1])
    dn = np.cumsum(hist)
    i_hi = int(np.argmax(up >= count)) if (up >= count).any() else None
    i_lo = int(np.argmax(dn >= count)) if (dn >= count).any() else None
    # the scan stops when both are found; one never found leaves its initial 0 (-0x8000 after
    # the shift), which cannot happen for count <= the pixel count
    hi = (0xFFFF - i_hi) if i_hi is not None else 0
    lo = i_lo if i_lo is not None else 0
    return hi - 0x8000, lo - 0x8000


def _stretch(hi: int, lo: int) -> int:
    span = hi - lo
    return 0x80 if span >= 0x80 else (0x80 - span) // 2 + 0x80


def process(palettes: np.ndarray, raw: np.ndarray, palette: int, brightness: int,
            color_bar: tuple, isotherm: tuple, raw_of, temp_of):
    """The girp image pipeline. color_bar / isotherm: (flag, high, low). raw_of(t) -> raw (raises on a
    rejected conversion), temp_of(raw) -> Celsius. Returns (rgb (h, w, 3), new colour bar
    (high, low) as the plugin stores it afterwards)."""
    flat = raw.reshape(-1).astype(np.int64)
    n = flat.size
    cb_manual, cb_high, cb_low = color_bar
    iso_on, iso_high, iso_low = isotherm
    hi, lo = _tail_points(raw, n // 100)
    if not cb_manual:
        # the 0.2 % points, reported back as the colour bar
        c = int(n * 0.2 / 100.0)
        h2, l2 = _tail_points(raw, c)
        stored = (F(temp_of(h2)), F(temp_of(l2)))
    else:
        lo, hi = raw_of(cb_low), raw_of(cb_high)
        stored = (F(cb_high), F(cb_low))
    r_iso_low, r_iso_high = raw_of(iso_low), raw_of(iso_high)

    k = ((150.0 / float(_stretch(hi, lo) + (hi - lo))) * 75.0) / 50.0
    b = float((brightness * 100) // 0x32) - (float(lo + hi) * k) / 2.0

    def to8(v):
        return min(max(int(float(v) * k + b), 0), 0xFF)     # (short)(int)(double), clamped

    s6a, s6c, s66, s68 = to8(r_iso_high), to8(r_iso_low), to8(hi), to8(lo)
    v = np.clip((flat.astype(np.float64) * k + b).astype(np.int64), 0, 0xFF)
    if palette < 0 or palette >= len(palettes):
        raise UndefinedBehaviour("girp palette index past its palette table")
    lut = palettes[palette][:, ::-1]                      # stored BGR
    if not iso_on:
        if cb_manual:
            v = np.where(v < s68, 0, np.where(s66 < v, 0xFF, v))
        rgb = lut[v]
    else:
        if cb_manual:
            s6c = max(s6c, s68)
            s6a = min(s6a, s66)
        inside = (s6c < v) & (v < s6a)
        rgb = np.where(inside[:, None], lut[v], palettes[0][:, ::-1][v])
    return rgb.astype(np.uint8).reshape(raw.shape + (3,)), stored


def process_strech(raw: np.ndarray) -> np.ndarray:
    """The same 1 % stretch in double at brightness 50, clamped to [0, 255] and
    stored as float32."""
    flat = raw.reshape(-1)
    hi, lo = _tail_points(raw, flat.size // 100)
    k = ((150.0 / float(_stretch(hi, lo) + (hi - lo))) * 75.0) / 50.0
    off = (50 * 100.0) / 50.0 - (float(lo + hi) * k) / 2.0
    v = flat.astype(np.float64) * k + off
    return np.clip(v, 0.0, 255.0).astype(F).reshape(raw.shape)


def pseudo_color_lut(palettes: np.ndarray) -> np.ndarray:
    """Palettes 0..9 as (3, 10, 256) red, green, blue."""
    p = palettes[:10]
    return np.ascontiguousarray(np.stack([p[:, :, 2], p[:, :, 1], p[:, :, 0]]))
