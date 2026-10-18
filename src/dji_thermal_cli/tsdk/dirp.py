"""The "dirp" plugin (libv_dirp): Zenmuse H20T ("class D") and Zenmuse XT S ("class G").

Both carry a 16k-entry temperature curve (0.1 C steps) and look the raw value up in it:

* H20T: raw (APP3) is converted to a curve index with gains and offsets derived from the
  camera's internal temperatures (APP4 calibration block), then corrected for emissivity in
  float32 with libm powf.
* XT S: raw (APP4) goes through an integer/double pipeline driven by a parameter line that
  libv_dirp rebuilds from the APP2 "Plug" block, then the same emissivity correction as girp.

Both classes share libv_dirp's parameter handling: distance is kept as whole metres,
emissivity in percent and reflection in tenths of a degree; on load the H20T calibration
block's own parameters are replaced by the plugin defaults (5 m, 70 %, 1.0, 23 C), and the
XT S always measures with an atmospheric temperature of 23 C.
"""

from __future__ import annotations

import math
import struct

import numpy as np

from ._cnum import F, cdiv, cvtt, per_value, roundf, s16
from . import dirp_isp
from .base import ColorBar, Isotherm, MeasurementParams, RJpeg
from .errors import IspError, MeasureError, ParamError, RJpegError, UndefinedBehaviour
from .jpeg import APP2, APP3, APP4, APP5, NdFilter, concat, first
from .powf32 import powf

_RANGES = {"distance": (1.0, 25.0), "humidity": (20.0, 100.0), "emissivity": (float(F(0.1)), 1.0), "reflection": (-40.0, 500.0)}
DEFAULT_PARAMS = (5, 70, 100, 230)       # distance m, humidity %, emissivity %, reflection 0.1 C


class _Dirp(RJpeg):
    plugin = "dirp"
    width, height = 640, 512

    def __init__(self, nd: NdFilter, tables: dict | None = None):
        self.nd = nd
        self.t = tables or {}
        self.mp = list(DEFAULT_PARAMS)    # handle+200 shorts
        # image pipeline state (object +0xa1 brightness, +0xa4 palette, +0xa6 isotherm,
        # +0xb9 colour bar automatic flag; limits in 0.1 C)
        self.palette = 2                  # the subclasses read the file's palette
        self.brightness = 50
        self.iso = [False, 340, 300]
        self.cb_auto, self.cb_high, self.cb_low = True, 350, 290
        self.auto_range = None            # (high, low) Celsius after an automatic run
        self._lut = None                  # built on first use / after each parameter change

    def get_measurement_params(self) -> MeasurementParams:
        d, h, e, r = self.mp
        return MeasurementParams(float(d & 0xFFFF), float(h & 0xFFFF), float(F(e & 0xFFFF) / F(100.0)),
                                 float(F(s16(r)) / F(10.0)), 0.0)

    def get_measurement_params_range(self):
        return dict(_RANGES, ambient_temp=(0.0, 0.0))

    # --- image pipeline -----------------------------------------------------------------
    def _lut_now(self):
        """The discrete LUT as the object holds it: rebuilt on parameter changes, None when the
        last refresh failed (the native code then has no LUT)."""
        if self._lut is None:
            try:
                self._lut = self.discrete_lut()
            except ParamError:
                self._lut = False
        return self._lut or None

    def _refresh_lut(self):
        self._lut = None
        try:
            self._lut = self.discrete_lut()
        except ParamError:
            self._lut = False
            raise

    def _t2raw(self, t10: int) -> int:
        """The LUT raw for a temperature given in tenths of a degree, clamped to its range."""
        lut = self._lut_now()
        if lut is None:
            raise UndefinedBehaviour("no colour LUT (its refresh failed); the native SDK dereferences NULL here")
        temps, raws = lut
        t = s16(cvtt(float(F(F(F(t10) / F(10.0)) * F(10.0)))))
        if temps[0] < t:
            return int(raws[t - temps[0]]) if t < temps[-1] else int(raws[-1])
        return int(raws[0])

    def _raw2t(self, raw: int) -> float:
        """The temperature of a raw value through the LUT."""
        lut = self._lut_now()
        if lut is None:
            raise UndefinedBehaviour("no colour LUT (its refresh failed); the native SDK dereferences NULL here")
        temps, raws = lut
        if raws[0] < raw:
            if raw < raws[-1]:
                k = 0
                for i in range(1, len(raws)):
                    if raws[i - 1] < raw <= raws[i]:
                        k = i
                t = temps[k]
            else:
                t = temps[-1]
        else:
            t = temps[0]
        return float(F(F(int(t)) / F(10.0)))

    def _isp_get_pseudo_color(self) -> int:
        return self.palette & 0xFF

    def _isp_set_pseudo_color(self, palette: int) -> None:
        if not palette < 10:
            raise ParamError(f"pseudo color {palette} exceeds the range [0, 10)", -3)
        self.palette = palette & 0xFF

    def _isp_get_color_bar(self) -> ColorBar:
        return ColorBar(not self.cb_auto, float(F(F(self.cb_high) / F(10.0))), float(F(F(self.cb_low) / F(10.0))))

    def _isp_set_color_bar(self, cb: ColorBar) -> None:
        if cb.manual_enable and F(cb.high) <= F(cb.low):
            raise ParamError("Color bar thresh_high must be bigger than thresh_low", -3)
        self.cb_auto = not cb.manual_enable
        self.cb_high = s16(cvtt(float(F(F(cb.high) * F(10.0)))))
        self.cb_low = s16(cvtt(float(F(F(cb.low) * F(10.0)))))

    def _isp_color_bar_adaptive(self) -> ColorBar:
        if self.auto_range is None:
            raise IspError("run process at least once in color bar automatic mode", -13)
        return ColorBar(False, *self.auto_range)

    def _isp_get_isotherm(self) -> Isotherm:
        en, hi, lo = self.iso
        return Isotherm(en, float(F(F(hi) / F(10.0))), float(F(F(lo) / F(10.0))))

    def _isp_set_isotherm(self, iso: Isotherm) -> None:
        if iso.enable and F(iso.high) <= F(iso.low):
            raise ParamError("Isotherm thresh_high must be bigger than thresh_low", -3)
        self.iso = [bool(iso.enable), s16(cvtt(float(F(F(iso.high) * F(10.0))))), s16(cvtt(float(F(F(iso.low) * F(10.0)))))]

    def _isp_get_brightness(self) -> int:
        return self.brightness

    def _isp_set_brightness(self, brightness: int) -> None:
        if not 0 <= brightness <= 100:
            raise ParamError(f"brightness {brightness} exceeds the range [0, 100]", -3)
        self.brightness = brightness

    def _isp_tables(self):
        return self.t["dirp_palettes"].reshape(10, 3, 256), self.t["dirp_y2r"]

    def _isp_pseudo_color_lut(self) -> np.ndarray:
        return dirp_isp.pseudo_color_lut(self._isp_tables()[0])

    def _run(self, strech: bool):
        palettes, y2r = self._isp_tables()
        en, ihi, ilo = self.iso
        iso = (en, self._t2raw(ihi), self._t2raw(ilo)) if en else (False, 0, 0)
        cb = (True, 0, 0) if self.cb_auto else (False, self._t2raw(self.cb_high), self._t2raw(self.cb_low))
        lut = self._lut_now()
        out, (lo, hi) = dirp_isp.process(self.raw.view(np.uint16), self._bits, palettes, y2r, self.palette,
                                         self.brightness, iso, cb, None if lut is None else lut[1], strech)
        if self.cb_auto:
            self.auto_range = (self._raw2t(hi & 0xFFFF), self._raw2t(lo & 0xFFFF))
        return out

    def _isp_process(self) -> np.ndarray:
        return self._run(False)

    def _isp_process_strech(self) -> np.ndarray:
        return self._run(True)

    def set_measurement_params(self, p: MeasurementParams) -> None:
        for name in ("distance", "humidity", "emissivity", "reflection"):
            lo, hi = _RANGES[name]
            v = F(getattr(p, name))
            if v < F(lo) or v > F(hi):
                raise ParamError(f"{name}={float(v):g} is outside [{lo:g}, {hi:g}]")
        self.mp = [s16(cvtt(float(F(p.distance)))), s16(cvtt(float(F(p.humidity)))),
                   s16(cvtt(float(F(F(p.emissivity) * F(100.0))))), s16(cvtt(float(F(F(p.reflection) * F(10.0)))))]
        if self._PARAMS_RESET_AUTO:
            self.auto_range = None        # the adaptive colour bar is gone until the next automatic run
        self._refresh_lut()               # libv_dirp rebuilds its colour LUT; failure fails the set

    # --- discrete colour LUT (libv_dirp discrete_lut_refresh) ----------------------------
    def discrete_lut(self):
        """(temps in 0.1 C, raw per temp) between the temperatures of the lowest and highest raw
        value; raises ParamError(-6 / -1) where the native refresh fails."""
        lo, hi = self._lut_range()
        if not lo < hi:
            raise ParamError("dirp: no usable temperature range for the colour LUT", self._LUT_FAIL)
        temps = np.arange(lo, hi + 1, dtype=np.int64)
        raws = np.array([self._raw_of_t10(int(t)) for t in temps], dtype=np.int64)
        if (np.diff(raws) < 0).any():
            raise ParamError("dirp: colour LUT is not monotonic", self._LUT_FAIL)
        return temps, raws


class H20T(_Dirp):
    """Class D: raw in APP3, calibration block (AA 55 12 ..) in APP4, curve in APP5."""

    _bits = 16

    def rjpeg_version(self):
        return 1, self.header_version, 1        # the header version is at +0x148 in the APP4 block

    def __init__(self, segs, nd: NdFilter, tables: dict | None = None):
        super().__init__(nd, tables)
        app4, app5 = first(segs, APP4), first(segs, APP5)
        if app4 is None or app5 is None or len(app4) < 0xA00 or len(app5) < 0x8000:
            raise RJpegError("H20T R-JPEG without its calibration block or curve")
        self.s = bytearray(app4[:0xA00])
        self.header_version = struct.unpack_from("<I", app4, 0x148)[0]
        self.palette = app4[0x584] if app4[0x584] <= 9 else 2
        self.lut = np.frombuffer(app5[:0x8000], dtype="<i2")
        raw = np.frombuffer(concat(segs, APP3), dtype="<u2")
        if raw.size != self.width * self.height:
            raise RJpegError("H20T raw frame size mismatch")
        self.raw = raw.reshape(self.height, self.width)

    def _f(self, off):
        return F(struct.unpack_from("<f", self.s, off)[0])

    def _h(self, off):
        return struct.unpack_from("<h", self.s, off)[0]

    def _H(self, off):
        return struct.unpack_from("<H", self.s, off)[0]

    def _measure(self) -> np.ndarray:
        return per_value(self.raw, self._converter())

    def _s10(self) -> int:
        """The curve offset both directions use: where the internal temperature sits on the curve."""
        lut = self.lut
        s9 = s16(int(roundf(F(self._f(0xA4) * F(10.0)))))
        if not lut[0x3FFF] <= s9:
            if lut[0] < s9 and (k := _step(lut, s9)) is not None:
                return int(roundf(F((k + 1) << 2)))
            return 0
        return -4

    _LUT_FAIL = -6
    _PARAMS_RESET_AUTO = True     # a parameter set (even one whose LUT refresh fails) drops the adaptive range

    def _lut_range(self):
        try:
            conv = self._converter()
            t0, t1 = conv(0), conv(0xFFFF)
        except MeasureError as e:
            raise ParamError(f"dirp: colour LUT refresh failed ({e})", -6) from None
        return s16(cvtt(float(F(F(t0) * F(10.0))))), s16(cvtt(float(F(F(t1) * F(10.0)))))

    def _raw_of_t10(self, t10: int) -> int:
        return self._raw_of(F(F(t10) / F(10.0)))

    def _raw_of(self, t_c) -> int:
        """The raw count whose temperature is t_c (one value)."""
        self._load_params()
        lut = self.lut
        p0, p1, p2, p3, p4 = self._prep()
        s10 = F(self._s10() & 0xFFFF)
        refl = F(F(F(self._h(0x4A)) / F(10.0)) + F(273.0))
        r4 = F(powf(float(refl), 4.0))
        ems, kj, db = F(self._H(0x48)), F(self._H(0x54)), F(F(self._h(0x56)) / F(10.0))
        x = F(powf(float(F(F(F(F(F(t_c) - db) * F(100.0)) / kj) + F(273.0))), 4.0))
        x = F(powf(float(F(F(F(ems * F(x - r4)) / F(100.0)) + r4)), 0.25))
        x = F(x - F(273.0))
        s = s16(int(roundf(F(x * F(10.0)))))
        if not s < lut[0x3FFF]:
            g = F(65532.0)
        else:
            g = F(0.0)
            if lut[0] < s and (k := _step(lut, s)) is not None:
                g = F(((k + 1) & 0xFFFF) << 2)
        v = roundf(F(F(F(F(F(g * p4) + p0) + p1) - s10) - p2))
        v = F(F(v / p3) + F(self._H(0x146)))
        if v > F(65535.0):
            return 0xFFFF
        return int(roundf(v)) & 0xFFFF

    def _load_params(self):
        d, h, e, r = self.mp
        struct.pack_into("<HHHh", self.s, 0x44, d & 0xFFFF, h & 0xFFFF, e & 0xFFFF, s16(r))

    def _converter(self):
        self._load_params()
        for off in (0x64, 0xA4, 0x88, 0xA8, 0x84):
            if self._f(off) < F(-273.5):
                raise MeasureError("H20T: invalid internal temperature in the calibration block", -0x27)
        lut = self.lut
        s9 = s16(int(roundf(F(self._f(0xA4) * F(10.0)))))
        s10 = -4
        if not lut[0x3FFF] <= s9:
            s10 = 0
            if lut[0] < s9 and (k := _step(lut, s9)) is not None:
                s10 = int(roundf(F((k + 1) << 2)))
            if not s10:
                raise MeasureError("H20T: internal temperature outside the curve", -0x24)
        p0, p1, p2, p3, p4 = self._prep()
        raw_off = self._H(0x146)
        refl = F(F(F(self._h(0x4A)) / F(10.0)) + F(273.0))
        r4 = F(powf(float(refl), 4.0))
        ems = F(self._H(0x48))
        kj, db = F(self._H(0x54)), F(F(self._h(0x56)) / F(10.0))

        def one(v: int) -> float:
            g = F(F(F(F(F(F(v - raw_off) * p3) + F(s10 & 0xFFFF)) + p2) - p0) - p1)
            g = F(g / p4)
            if g < 0:
                idx = 0
            else:
                u = int(roundf(g)) & 0xFFFFFFFF
                idx = 0x3FFF if u >= 0xFFFE else int(math.floor(u * 0.25 + 0.5)) & 0xFFFF
            t = F(F(int(lut[idx])) / F(10.0))
            t4 = F(powf(float(F(t + F(273.0))), 4.0))
            q = F(F(F(F(t4 - r4) * F(100.0)) / ems) + r4)
            tt = F(powf(float(max(q, F(0.0))), 0.25))
            return float(F(db + F(F(F(tt - F(273.0)) * kj) / F(100.0))))

        return one

    def _prep(self):
        """The measurement coefficients from the calibration block."""
        f = self._f
        kk = self._h(0x58)
        fv6 = f(0x64) if kk < -0xAAF else F(F(kk) / F(10.0))
        k1, k2, k3, k4, kf, b1, b2 = (f(o) for o in (0x24, 0x28, 0x2C, 0x30, 0x34, 0x38, 0x3C))
        f7 = F(f(0xA8) - fv6)
        p0 = F(F(F(f(0xA4) - fv6) * k1) + b1)
        f6 = F(f(0x88) - fv6)
        p1 = F(F(F(F(f6 * k3) + F(F(f6 * k2) * f6)) + k4) - F(F(F(k3 * f7) + F(F(k2 * f7) * f7)) + k4))
        dist5 = self._H(0x44) - 50
        d1, d2 = struct.unpack_from("<ii", self.s, 0x4C)
        hum = F(F(F(self._H(0x46) - 0x46) / F(6.4)) + F(70.0))
        p4 = F(F(hum / F(70.0)) * F(F(F(F(d1 * dist5 * dist5) / F(100000.0)) + F(1.0)) - F(F(dist5 * d2) / F(100000.0))))
        return p0, p1, b2, F(kf / F(79.0)), p4


# Parameter line (offset, size) <- Plug block offset, as rebuilt by libv_dirp
_PL_MAP = [(4, 0x494, 2), (6, 0x496, 2), (8, 0x498, 2), (10, 0x49A, 2), (0xC, 0x4A2, 2), (0xE, 0x4A4, 2), (0x10, 0x4A6, 2),
           (0x12, 0x4A8, 2), (0x14, 0x4AA, 2), (0x16, 0x4AC, 2), (0x18, 0x4B0, 4), (0x1C, 0x4B4, 4), (0x20, 0x4B8, 2),
           (0x22, 0x4BA, 2), (0x24, 0x4BC, 2), (0x26, 0x4BE, 2), (0x28, 0x4C0, 2), (0x2A, 0x4C2, 2), (0x2C, 0x4C4, 4),
           (0x30, 0x4C8, 2), (0x32, 0x4CA, 2), (0x34, 0x4CC, 2), (0x36, 0x4CE, 2), (0x38, 0x4D0, 2), (0x3A, 0x4D2, 2),
           (0x3C, 0x4D4, 2), (0x3E, 0x4D6, 2), (0x40, 0x4D8, 4), (0x44, 0x4DC, 2), (0x46, 0x4DE, 2), (0x48, 0x4E0, 2),
           (0x4A, 0x4E2, 2), (0x4C, 0x4E4, 2), (0x4E, 0x4E6, 2), (0x50, 0x4E8, 2), (0x52, 0x4EA, 2)]


class XTS(_Dirp):
    """Class G: "Plug" block in APP2, curve in APP3, raw in APP4."""

    _bits = 14

    def rjpeg_version(self):
        return 1, 1, 1

    def __init__(self, segs, nd: NdFilter, tables: dict | None = None):
        super().__init__(nd, tables)
        plug = first(segs, APP2)
        if plug is None or len(plug) < 0x4EC or not plug.startswith(b"Plug"):
            raise RJpegError("XT S R-JPEG without its Plug block")
        pl = bytearray(0x58)
        struct.pack_into("<HH", pl, 0, 0x55AA, 0x38)
        for dst, src, n in _PL_MAP:
            pl[dst:dst + n] = plug[src:src + n]
        struct.pack_into("<HH", pl, 0x54, 0xF0, 0x6666)
        self.pl = bytes(pl)
        self.palette = plug[0xC] if plug[0xC] <= 9 else 2
        curve = concat(segs, APP3)
        if len(curve) < 0x8000:
            raise RJpegError("XT S curve is truncated")
        self.curve = np.frombuffer(curve[:0x8000], dtype="<i2")
        raw = np.frombuffer(concat(segs, APP4), dtype="<i2")
        if raw.size != self.width * self.height:
            raise RJpegError("XT S raw frame size mismatch")
        self.raw = raw.reshape(self.height, self.width)

    def _g(self, off):
        """The 16-bit parameter-line field at byte offset off."""
        return struct.unpack_from("<h", self.pl, off)[0]

    def _g32(self, off):
        """The 32-bit parameter-line field at byte offset off."""
        return struct.unpack_from("<i", self.pl, off)[0]

    def _point_fn(self):
        """The per-gray measurement: gray -> tenths of a degree."""
        dist, hum, ems = (v & 0xFFFF for v in self.mp[:3])
        refl, atm = s16(self.mp[3]), 230
        if ems == 0 or ems > 100 or hum > 100 or not 5 <= dist <= 5000 or not -400 <= refl <= 0x157C:
            raise MeasureError("XT S: measurement parameters out of range", -1)
        g, g32 = self._g, self._g32
        f8, f9, f10 = (F(F(g(a)) / F(100.0)) for a in (0x30, 0x44, 0x52))
        s2 = s16(cvtt(float(f9) * (g(0x14) / 1000.0) + (g(0x12) / 1000.0) * float(f9) * float(f9)) + g(0x26))
        i4, i5, i6 = g(0x34), g(0x10), g(0x0E)
        d = F(f9 - f10)
        s3 = s16(cvtt((float(F(F(g32(0x40)) * F(d * d))) / 10000.0
                       + ((g(0x3C) / 1000.0) + float(F(F(g(0x3E)) * d)) / 10000.0)) * float(abs(g(0x4E)))))
        if f9 <= f8:
            d80 = s16(cvtt(float(F(f8 - f9)) * (s3 / 10.0)))
        else:
            d80 = s16(cvtt(float(F(f8 - f9)) * ((s3 - 0x15E) / 10.0)))
        t10 = F(f9 * F(10.0))
        hits = np.flatnonzero(self.curve.astype(np.float32) >= t10)
        l6c = int(hits[0]) if hits.size else 0
        off = s16(cvtt(float(F(F(i6) + F(F(i5) * F(f8 - F(F(i4) / F(100.0))))))))
        k_a8 = g(0x0C) / 100.0
        hum_f = F(1.0 - (hum - 60) / 5000.0)
        dpoly = F(((g32(0x18) * (dist - 5) * (dist - 5) + 10000) - g32(0x1C) * (dist - 5)) / 10000.0)
        b2 = cdiv(g(0x16), 10)
        tr = refl if abs(refl - atm) > 0x14 else atm
        l30 = (tr + 0xAAF) / 10.0
        be, d6, ec = g(0x22), g(0x3A), g(0x50)
        curve = self.curve

        def point(v: int) -> int:
            inner = s16(s16(s16(v + s2) - off) + d80 - d6)
            a = s16(cvtt((k_a8 * inner + float(l6c)) + float(ec)))
            b = s16(cvtt(float(F(F(a) * hum_f))))
            c = min(max(s16(cvtt(float(F(F(b) / dpoly)))), 0), 0x3FFF)
            t = s16(int(curve[c]) + b2)
            if ems < 100:
                p2 = math.pow(l30, 4.0)
                q = ((math.pow((t + 0xAAF) / 10.0, 4.0) - p2) * 100.0) / (ems + 1) + p2
                t38 = (math.pow(q, 0.25) if q >= 0 else math.nan) * 10.0
            else:
                t38 = float(t) + 2735.0
            return s16(cvtt(float(be) + t38) - 0xAAF)

        return point

    _LUT_FAIL = -1
    _PARAMS_RESET_AUTO = False

    def _lut_range(self):
        try:
            point = self._point_fn()
            return point(0), point(0x3FFF)
        except MeasureError:
            return 0, 0              # the point function returns without writing its output

    def _raw_of_t10(self, t10: int) -> int:
        """The gray whose temperature is t10 (0.1 C): the inverse of _point_fn."""
        dist, hum, ems = (v & 0xFFFF for v in self.mp[:3])
        refl, atm = s16(self.mp[3]), 230
        if ems == 0 or ems > 100 or hum > 100 or not 5 <= dist <= 5000 or not -400 <= refl <= 0x157C:
            raise MeasureError("XT S: measurement parameters out of range", -3)
        g, g32 = self._g, self._g32
        f8, f9, f10 = (F(F(g(a)) / F(100.0)) for a in (0x30, 0x44, 0x52))
        t = s16(t10 + 0xAAF - g(0x22))
        if ems < 100:
            tr = refl if abs(refl - atm) > 0x14 else atm
            p6 = math.pow((tr + 0xAAF) / 10.0, 4.0)
            q = (p6 * 100.0 + (math.pow(t / 10.0, 4.0) - p6) * (ems + 1)) / 100.0
            t = s16(cvtt((math.pow(q, 0.25) if q >= 0 else math.nan) * 10.0 - 2735.0))
        else:
            t = s16(t - 0xAAF)
        curve = self.curve
        target = s16(t - cdiv(g(0x16), 10))
        hits = np.flatnonzero(curve >= target)
        if not hits.size:
            raise UndefinedBehaviour("XT S temperature above the curve (native index uninitialised)")
        l60 = int(hits[0])
        t10f = F(f9 * F(10.0))
        hits = np.flatnonzero(curve.astype(np.float32) >= t10f)
        l5c = int(hits[0]) if hits.size else 0
        d = F(f9 - f10)
        s3 = s16(cvtt((float(F(F(g32(0x40)) * F(d * d))) / 10000.0
                       + ((g(0x3C) / 1000.0) + float(F(F(g(0x3E)) * d)) / 10000.0)) * float(abs(g(0x4E)))))
        if f9 <= f8:
            d80 = s16(cvtt(float(F(f8 - f9)) * (s3 / 10.0)))
        else:
            d80 = s16(cvtt(float(F(f8 - f9)) * ((s3 - 0x15E) / 10.0)))
        hum_f = F(1.0 - (hum - 60) / 5000.0)
        dpoly = F(((g32(0x18) * (dist - 5) * (dist - 5) + 10000) - g32(0x1C) * (dist - 5)) / 10000.0)
        a = s16(cvtt(float(F(F(l60) / hum_f))))
        c = s16(cvtt(float(F(F(a) * dpoly))))
        e = s16(cdiv(s16(c - (l5c + g(0x50))) * 100, g(0x0C)))
        s2 = s16(cvtt(float(f9) * (g(0x14) / 1000.0) + (g(0x12) / 1000.0) * float(f9) * float(f9)) + g(0x26))
        off = s16(cvtt(float(F(F(g(0x0E)) + F(F(g(0x10)) * F(f8 - F(F(g(0x34)) / F(100.0))))))))
        v = s16(e + s16(s16(s16(g(0x3A) - s2) + off) - d80))
        return min(max(v, 0), 0x3FFF)

    def _points(self) -> np.ndarray:
        """The per-gray measurement over the frame, tenths of a degree (int16)."""
        try:
            point = self._point_fn()
        except MeasureError:
            # the SDK's per-gray measurement returns early without writing; measure / measure_ex still succeed
            # on the (zero-initialised) output buffer, so the native SDK returns all zeros
            return np.zeros((self.height, self.width), dtype=np.int16)
        return per_value(self.raw, point).astype(np.int16)

    def _measure(self) -> np.ndarray:
        # the int16 tenths, converted to float
        return (self._points().astype(np.float32) / F(10.0)).astype(np.float32)

    def measure(self) -> np.ndarray:
        v = self._points()
        if self.nd.installed:
            return self._nd_tenths(v)
        return v


def _step(lut: np.ndarray, s: int) -> int | None:
    """The first k < 0x3FFF with lut[k] < s <= lut[k + 1] (libv_dirp's linear curve scan), or None."""
    hits = np.flatnonzero((lut[:-1] < s) & (lut[1:] >= s))
    return int(hits[0]) if hits.size else None


def is_class_d(segs) -> bool:
    app4 = first(segs, APP4)
    return app4 is not None and len(app4) > 0x9FF and app4[:3] == b"\xaa\x55\x12"


def is_class_g(segs) -> bool:
    app2 = first(segs, APP2)
    return app2 is not None and app2.startswith(b"Plug")
