"""The "girp" plugin (libv_girp): some Matrice 30T captures.

The raw frame is signed 16-bit. An integer pipeline driven by the APP4 parameter line (an
AA 55 block) maps each raw value to an index into one of ten gray->temperature LUTs taken
from libv_girp, then applies an emissivity correction in tenths of a kelvin. Output is in
0.1 C steps. Integer widths, C truncation and the plugin's quirks (distance truncated to
whole metres, ambient fixed at 23.0 C, a quadratic correction at exactly 95 % emissivity)
are reproduced, so results are bit-identical to the native SDK.
"""

from __future__ import annotations

import math
import struct

import numpy as np

from ._cnum import F, cdiv, cvtt, per_value, s16, s32, to_short
from . import girp_isp
from .base import ColorBar, Isotherm, MeasurementParams, RJpeg
from .errors import IspError, MeasureError, ParamError, RJpegError, UndefinedBehaviour
from .jpeg import APP3, APP4, NdFilter, concat, first

# (flag == 0, flag != 0) LUT index per lens/mode byte; unknown modes behave like mode 1
_LUT_FOR_MODE = {1: (3, 2), 2: (5, 4), 3: (1, 0), 10: (8, 9), 11: (6, 7)}
_RANGES = {"distance": (1.0, 25.0), "humidity": (20.0, 100.0), "emissivity": (float(F(0.1)), 1.0), "reflection": (-40.0, 500.0)}


class Girp(RJpeg):
    plugin = "girp"
    width, height = 640, 512

    def __init__(self, segs, tables: dict[str, np.ndarray], nd: NdFilter):
        self.nd = nd
        self.luts = tables["girp_luts"].reshape(10, 0x4000)
        app4 = first(segs, APP4)
        if app4 is None or len(app4) < 32 + 0x10C or app4[32:34] != b"\xaa\x55":
            raise RJpegError("girp R-JPEG without its AA 55 parameter line")
        self.pd = app4[32:]
        raw = np.frombuffer(concat(segs, APP3), dtype="<i2")
        if raw.size != self.width * self.height:
            raise RJpegError("girp raw frame size mismatch")
        self.raw = raw.reshape(self.height, self.width)
        h = self._u16
        # params[7]: emissivity %, humidity %, distance (0.1 m), reflection (0.1 C), ambient (0.1 C), gain %, offset
        self.p = [h(8), h(4), h(6), s16(h(10)), 230, 100, 0]
        self.palettes = tables["girp_palettes"]
        # image pipeline state (libv_girp keeps it in globals, reset by create)
        self.palette = h(0x9C)
        self.color_bar = ColorBar(False, 0.0, 255.0)
        self.isotherm = Isotherm(False, 0.0, 0.0)
        self.brightness = 50

    def rjpeg_version(self):
        v = struct.unpack_from("<H", self.pd, 0xBA)[0]
        return v, v, self.pd[0xEA]             # libv_girp: parameter-line fields

    def get_measurement_params(self) -> MeasurementParams:
        p = self.p
        # ambient_temp is not reported by this plugin; libdirp leaves the caller's value (0)
        return MeasurementParams(float(F(p[2]) / F(10.0)), float(p[1]), float(F(p[0]) / F(100.0)),
                                 float(F(s16(p[3])) / F(10.0)), 0.0)

    def get_measurement_params_range(self):
        return dict(_RANGES, ambient_temp=(0.0, 0.0))

    def set_measurement_params(self, mp: MeasurementParams) -> None:
        for name, (lo, hi) in _RANGES.items():
            v = F(getattr(mp, name))
            if v < F(lo) or v > F(hi):
                raise ParamError(f"{name}={float(v):g} is outside [{lo:g}, {hi:g}]")
        self.p[0] = s16(cvtt(float(F(F(mp.emissivity) * F(100.0)))))
        self.p[2] = s16(s16(cvtt(float(F(mp.distance)))) * 10)     # whole metres, in 0.1 m
        self.p[1] = s16(cvtt(float(F(mp.humidity))))
        self.p[3] = s16(cvtt(float(F(F(mp.reflection) * F(10.0)))))

    def _measure(self) -> np.ndarray:
        return per_value(self.raw, self._converter())

    def _measure_tenths(self, t: np.ndarray) -> np.ndarray:
        """libv_girp measure: (short)(int)(t * 10.0 +- 0.5) in double, half away from zero."""
        d = t.astype(np.float64) * 10.0
        return to_short(np.where(t < 0, d - 0.5, d + 0.5))

    # --- image pipeline -----------------------------------------------------------------
    def _isp_get_pseudo_color(self) -> int:
        # values outside 0..9 are not written back (the caller's variable is left as it was)
        return self.palette if 0 <= self.palette <= 9 else -1

    def _isp_set_pseudo_color(self, palette: int) -> None:
        self.palette = 9 if (palette & 0xFFFFFFFF) > 10 else palette

    def _isp_get_color_bar(self) -> ColorBar:
        return ColorBar(**vars(self.color_bar))

    def _isp_set_color_bar(self, cb: ColorBar) -> None:
        self.color_bar = cb

    _isp_color_bar_adaptive = _isp_get_color_bar     # the plugin answers with its colour bar

    def _isp_get_isotherm(self) -> Isotherm:
        return Isotherm(**vars(self.isotherm))

    def _isp_set_isotherm(self, iso: Isotherm) -> None:
        self.isotherm = iso

    def _isp_get_brightness(self) -> int:
        return self.brightness

    def _isp_set_brightness(self, brightness: int) -> None:
        if not 0 <= brightness <= 100:
            raise ParamError(f"brightness range [0,100] :{brightness}", -3)
        self.brightness = brightness

    def _isp_palettes(self) -> np.ndarray:
        return self.palettes.reshape(-1, 256, 3)

    def _isp_pseudo_color_lut(self) -> np.ndarray:
        return girp_isp.pseudo_color_lut(self._isp_palettes())

    def _isp_process(self) -> np.ndarray:
        cb, iso = self.color_bar, self.isotherm
        temp = self._converter()
        try:
            rgb, (high, low) = girp_isp.process(self._isp_palettes(), self.raw, self.palette, self.brightness,
                                                (cb.manual_enable, cb.high, cb.low), (iso.enable, iso.high, iso.low),
                                                self._raw_of, temp)
        except (IspError, MeasureError) as e:
            if isinstance(e, UndefinedBehaviour):
                raise
            raise IspError(f"paramline parse error ({e})", -7) from None
        self.color_bar = ColorBar(cb.manual_enable, float(high), float(low))
        return rgb

    def _isp_process_strech(self) -> np.ndarray:
        return girp_isp.process_strech(self.raw)

    def _i32(self, off: int) -> int:
        return struct.unpack_from("<i", self.pd, off)[0]

    def _i16(self, off: int) -> int:
        return struct.unpack_from("<h", self.pd, off)[0]

    def _u16(self, off: int) -> int:
        return struct.unpack_from("<H", self.pd, off)[0]

    def _raw_of(self, t_c) -> int:
        """The raw value whose temperature is t_c (colour bar and isotherm
        limits). Returns the clamped raw, or raises IspError with the plugin's status."""
        pd, p = self.pd, self.p
        i32, i16, u16 = self._i32, self._i16, self._u16
        ems, hum, dist = p[0] & 0xFFFF, p[1] & 0xFFFF, p[2] & 0xFFFF
        refl, amb, gain, off = s16(p[3]), s16(p[4]), p[5] & 0xFFFF, s16(p[6])
        for bad, code in ((ems == 0 or ems > 100, -2), (hum > 100, -3), (not 5 <= dist <= 5000, -4),
                          (not -400 <= refl <= 0x157C, -5), (not -400 <= amb <= 1000, -6),
                          (gain == 0 or gain > 200, -7), (not -100 <= off <= 100, -8)):
            if bad:
                raise IspError(f"girp: temperature conversion rejected the parameters ({code})", code)
        ems = max(ems, 50)
        d50 = dist - 50
        atm = s16(cdiv(s32(s32(d50 * i32(0xF0) + i32(0xF4)) * d50), 10000))
        u6 = cdiv(s32(((0x13C4 - hum) & 0xFFFFFFFF) * ((i16(0xF8) + atm) & 0xFFFF) & 0xFFFFFFFF), 5000) & 0xFFFF
        k = cdiv(s32(u6 * u16(0x104)), 1000) & 0xFFFF
        base = s16(i16(0x102) + s16(cdiv(s32(s32(d50 * i32(0xFA) + i32(0xFE)) * d50), 10000))
                   + s16(cdiv(s32(u6 * i32(0x106)), 1000)) + i16(0x10A))
        mode = pd[0xE8] if pd[0xE8] in _LUT_FOR_MODE else 1
        lut = self.luts[_LUT_FOR_MODE[mode][0 if u16(0x24) == 0 else 1]]

        t10 = s16(cvtt(float(F(F(t_c) * F(10.0)))))
        num = cdiv(s32((t10 - off) * 100), s16(gain))
        v = s16(num)
        if ems == 95:
            q = s16(cvtt(math.sqrt(float(num) * 1000000.0 + 12000000.0 + 260685809476.0) - 510574.0))
            if q >= 0x60F:
                v = q
        if ems < 98:
            tr = refl if abs(refl - amb) > 20 else amb
            tr4 = math.pow((tr + 2735) / 10.0, 4.0)
            q = (tr4 * 100.0 + (math.pow(s16(v + 2735) / 10.0, 4.0) - tr4) * (ems + 1)) / 100.0
            v = s16(cvtt(math.pow(q, 0.25) * 10.0 - 2735.0)) if q >= 0 else s16(cvtt(math.nan))
        target = s16(v - i16(0x16))
        hits = np.nonzero(lut >= target)[0]
        idx = int(hits[0]) if hits.size else 0
        if k == 0:
            raise IspError("girp: temperature conversion has a zero gain", -1)
        raw = s16(cvtt(float(idx - base) * (8192.0 / float(k))))
        return min(max(raw, 0), 0x3FFF)

    def _converter(self):
        """The plugin's raw -> Celsius conversion for the current parameters (one value at a
        time)."""
        pd, p = self.pd, self.p
        i32, i16, u16 = self._i32, self._i16, self._u16
        ems = p[0] & 0xFFFF
        if ems == 0 or ems > 100:
            raise MeasureError("girp: emissivity out of range", -7)
        ems = max(ems, 50)
        dist, hum, refl, amb, gain, off = p[2] & 0xFFFF, p[1] & 0xFFFF, s16(p[3]), s16(p[4]), p[5], s16(p[6])
        if hum > 100 or not 5 <= dist <= 5000:
            raise MeasureError("girp: humidity or distance out of range", -7)
        d50 = dist - 50
        atm = s16(cdiv(s32(s32(d50 * i32(0xF0) + i32(0xF4)) * d50), 10000))
        u6 = s32(((0x13C4 - hum) & 0xFFFFFFFF) * ((i16(0xF8) + atm) & 0xFFFF) & 0xFFFFFFFF)
        u6 = cdiv(u6, 5000) & 0xFFFF
        k = cdiv(s32(u6 * u16(0x104)), 1000) & 0xFFFF
        base = (i16(0x102) + s16(cdiv(s32(s32(d50 * i32(0xFA) + i32(0xFE)) * d50), 10000))
                + s16(cdiv(s32(u6 * i32(0x106)), 1000)) + i16(0x10A))
        mode = pd[0xE8] if pd[0xE8] in _LUT_FOR_MODE else 1
        lut = self.luts[_LUT_FOR_MODE[mode][0 if u16(0x24) == 0 else 1]]
        toff = i16(0x16)
        tr = amb if abs(refl - amb) < 21 else refl

        def one(v: int) -> float:
            g = s32(k * v)
            if g < 0:
                g += 0x1FFF
            idx = min(max(s16(base + s16(g >> 13)), 0), 0x3FFF)
            t = s16(int(lut[idx]) + toff)
            if ems < 100:
                tr4 = math.pow((tr + 2735) / 10.0, 4.0)
                q = ((math.pow((t + 2735) / 10.0, 4.0) - tr4) * 100.0) / (ems + 1) + tr4
                t38 = math.pow(q, 0.25) * 10.0 if q >= 0 else math.nan    # C pow of a negative -> NaN
            else:
                t38 = float(t) + 2735.0
            s13 = s16(cvtt(t38 - 2735.0))
            extra = F(0.0)
            if ems == 95 and s13 >= 0x60F:
                extra = F((F(s13) * F(21148.0) + F(s13) * F(s13)) / F(1e6) - F(12.0))
            return float(F((F(off) + (F(gain) * (F(s13) + extra)) / F(100.0)) / F(10.0)))

        return one
