"""The "hirp" plugin (libv_hirp): newer Matrice 3TD / Matrice 4T firmware.

libv_hirp wraps DJI's MicroSDK: the R-JPEG's APP5 "HIDR" container carries the camera's full
calibration, APP4 a TempParams block with the capture's measurement environment, APP3 the raw
14-bit frame. Temperatures come from libMicroTA's integer measurement core (MT_*_INT, ported in
hirp_mt), evaluated once per distinct gray value through MicroSDK's IntegerCalculator::measure:

- every measurement first re-applies the distance/emissivity/reflection ("expert config"),
  which runs one probe conversion of gray 5000;
- the gray limits of the last successful MT conversion and a per-gain-mode temperature range
  clamp the result (grays outside the limits are not converted at all);
- the measurement fails (dirp -4) when the last newly converted gray failed in MT;
- each value is truncated to 0.1 C and returned in Q13 fixed point.

Ambient temperature and humidity go to MT through IntegerCalculator::setModelConfig when the
parameters are set; a rejected value makes set fail with -3 and leaves the values already
applied in place, exactly like the native SDK. Results are bit-identical to libv_hirp.
"""

from __future__ import annotations

import struct

import numpy as np

from . import hirp_isp, hirp_mt
from ._cnum import F, cvtt, roundf, to_short
from .base import ColorBar, Isotherm, MeasurementParams, RJpeg
from .errors import IspError, MeasureError, ParamError, RJpegError, UnsupportedFeature
from .iirp_isp import DIRP2IRPROC
from .jpeg import APP3, APP4, APP5, NdFilter, concat, first

_RANGE = {"distance": (0.0, 1000.0), "humidity": (0.0, 100.0), "emissivity": (float(F(0.1)), 1.0),
          "reflection": (-40.0, 1000.0), "ambient_temp": (-100.0, 100.0)}
# IntegerCalculator::getGrayLimit: temperature clamp (Q13) per TempRange mode (TempParams gain + 1)
_TEMP_RANGE = {1: (-0x3C000, 0x14A000), 2: (-0x3C000, 0x4B0000)}
_TEMP_RANGE_NONE = (-0x2224CC, 0x7FFFFFFF)
_UNCOMPUTED = -0x222333          # table entries below this are recomputed on every use
_PROBE_GRAY = 5000
_MICROSDK_VERSION = 0x10000    # get_rjpeg_version packs MicroSDK version() 1.0.0 as major<<16 | minor<<8 | patch


def _q13_milli(v: int) -> int:
    """(int)((float)v / 1000.0f * 8192.0f): a thousandths value to Q13."""
    return cvtt(float(F(F(F(v) / F(1000.0)) * F(8192.0))))


def _hidr_block(app5: bytes, kind: int) -> bytes | None:
    """Payload of a typed block in the APP5 "HIDR" (MicroRImageV1 add-info) container."""
    if len(app5) < 12 + 9 * 12 or app5[:4] != b"HIDR":
        return None
    for i in range(9):
        typ, size, off = struct.unpack_from("<III", app5, 12 + 12 * i)
        if typ == kind and size > 0x1C and off + 0x1C <= len(app5):
            return app5[off + 0x1C:off + size]
    return None


class Hirp(RJpeg):
    plugin = "hirp"

    def __init__(self, segs, tables: dict, nd: NdFilter):
        self.nd = nd
        app4 = first(segs, APP4)
        if app4 is None or len(app4) < 0x28:
            raise RJpegError("hirp R-JPEG without its APP4 parameter block")
        # MHandle::parseApp4: two length-prefixed structs, ImageParams at 0x20 and TempParams after it
        isize = min(struct.unpack_from("<I", app4, 0x24)[0], 0x28)
        if isize == 0 or 0x20 + isize + 8 > len(app4):
            raise RJpegError("hirp APP4 ImageParams block is truncated")
        tp_off = 0x20 + isize
        tsize = min(struct.unpack_from("<I", app4, tp_off + 4)[0], 0x48)
        if tsize < 0x48 or tp_off + tsize > len(app4):
            raise RJpegError("hirp APP4 TempParams block is truncated")
        ip = app4[0x20:0x20 + isize].ljust(0x28, b"\0")
        self.tp = tp = app4[tp_off:tp_off + tsize]
        self.image_params_version = struct.unpack_from("<I", ip, 0)[0]
        self.width, self.height = struct.unpack_from("<II", ip, 8)
        raw = np.frombuffer(concat(segs, APP3), dtype="<u2")
        if self.width == 0 or self.height == 0 or raw.size != self.width * self.height:
            raise RJpegError("hirp raw frame size mismatch")
        self.raw = raw.reshape(self.height, self.width)
        self.app5 = concat(segs, APP5)
        calib = _hidr_block(self.app5, 7)
        if calib is None or len(calib) < 0x3000:
            raise RJpegError("hirp R-JPEG without its APP5 calibration block")
        self.gain_mode = tp[0x34]

        i32 = lambda o: struct.unpack_from("<i", tp, o)[0]  # noqa: E731
        u16 = lambda o: struct.unpack_from("<H", tp, o)[0]  # noqa: E731
        # MHandle::MHandle: the MeasureRule (Q13 / Q16) and MeasurementEnv defaults
        self.distance = i32(0x14) & 0xFFFFFFFF
        self.emissivity = (u16(0x10) << 3) & 0xFFFFFFFF
        self.reflection = i32(0x18) if tp[0x1C] == 1 else _q13_milli(i32(0x28))
        self.ambient = _q13_milli(i32(0x28)) if tp[0x3B] == 0 else i32(0x3C)
        self.humidity = cvtt(float(roundf(F(F(i32(0x2C)) / F(1000.0))))) & 0xFF
        self.model20 = i32(0x08)
        self.model21 = cvtt(float(F(F(i32(0x0C) * 1000) * F(2.0 ** -13))))
        self.addlines = self._addlines(tp)
        self.t = tables
        # image pipeline state: palette from ImageParams (MicroSDK PaletteType -> dirp), libv_hirp defaults
        self.palette = _PALETTE_TYPE.get(ip[0x14], 0)
        self.alarm = ip[0x15] == 1
        self.brightness = 50
        self.color_bar = ColorBar(False, 0.0, 0.0)
        # isotherm: on when the capture was saved with the temperature alarm; its limits are
        # the alarm thresholds in ImageParams (tenths of a degree above -100 C)
        hi, lo = struct.unpack_from("<ii", ip, 0x18)
        self.isotherm = Isotherm(self.alarm, _q13(float(F(F(hi - 1000) / F(10.0)))),
                                 _q13(float(F(F(lo - 1000) / F(10.0)))))
        self.auto_range = None            # (max, min) Celsius of the last automatic run
        # process_strech swaps in a grey palette (id 0x7f, isotherm off) and restores the
        # palette parameters only when it succeeds; until set_pseudo_color / set_isotherm
        # rebuild them, process then renders with palette 0's table; set_pseudo_color restores
        # only the palette, the isotherm stays off until set_isotherm
        self.strech_palette = False
        self.strech_iso_off = False
        # a process refused by setManualAGCConfig after a successful one leaves libv_hirp
        # reusing that earlier ISP (multi-frame state, not ported) until the palette is
        # rebuilt (set_pseudo_color, set_isotherm, process_strech)
        self.processed = False
        self.isp_stale = False

        self.mt = hirp_mt.MT(hirp_mt.Tables(tables), self.width, self.height)
        rc = self.mt.set_calib(bytes(calib[:0x3800]))
        for pid, val in ((13, 0), (45, 1), (15, calib[8]), (29, 0)):
            rc = rc or self.mt.set_single(pid, val)
        rc = rc or self._model_config()
        if rc:
            raise RJpegError(f"hirp: the measurement core rejected this R-JPEG's calibration (MT {rc})")

    def _addlines(self, tp: bytes) -> bytes:
        """MHandle::createAddlines: two raw-width rows carrying the shutter/lens readings."""
        w = self.width
        buf = bytearray(w * 4)
        for dst, src in ((2, 0x30), (10, 0x32), (0x1A, 0x34), (0x28, 0x36), (0x36, 0x38)):
            buf[dst:dst + 2] = tp[src:src + 2]
        struct.pack_into("<HH", buf, 0x10, w & 0xFFFF, self.height & 0xFFFF)
        struct.pack_into("<IiIIii", buf, w * 2, 0xAABBCCDD, *struct.unpack_from("<i", tp, 0x24),
                         w, self.height, *struct.unpack_from("<ii", tp, 0x28))
        return bytes(buf)

    def _model_config(self) -> int:
        """IntegerCalculator::setModelConfig: stops at the first value MT rejects."""
        mt = self.mt
        for pid, val in ((14, 0), (5, self.ambient), (6, self.humidity << 13), (20, self.model20), (21, self.model21)):
            rc = mt.set_single(pid, val)
            if rc:
                return rc
        return mt.set_addl(self.addlines)

    # -- dirp API
    def rjpeg_version(self):
        # rjpeg: the MicroSDK library version (1.0.0 in SDK v1.8); header/curve: the ImageParams
        # and TempParams struct versions from APP4
        return _MICROSDK_VERSION, self.image_params_version, struct.unpack_from("<I", self.tp, 0)[0]

    def get_measurement_params(self) -> MeasurementParams:
        if self.gain_mode == 2:
            raise ParamError("hirp: measurement parameters are not available in this gain mode", -64)
        q13, q16 = F(2.0 ** -13), F(2.0 ** -16)
        return MeasurementParams(
            distance=float(F(self.distance) * q13),
            humidity=float(F(self.humidity)),
            emissivity=float(F(self.emissivity) * q16),
            reflection=float(F(self.reflection) * q13),
            ambient_temp=float(F(self.ambient) * q13),
        )

    def get_measurement_params_range(self):
        return dict(_RANGE)

    def set_measurement_params(self, mp: MeasurementParams) -> None:
        # set_measurement_params: no range check of its own; MT validates ambient and humidity
        self.reflection = cvtt(float(F(F(mp.reflection) * F(8192.0))))
        self.distance = cvtt(float(F(F(mp.distance) * F(8192.0)))) & 0xFFFFFFFF
        self.emissivity = cvtt(float(F(F(mp.emissivity) * F(65536.0)))) & 0xFFFFFFFF
        self.humidity = cvtt(float(roundf(F(mp.humidity)))) & 0xFF
        self.ambient = cvtt(float(F(F(mp.ambient_temp) * F(8192.0))))
        rc = self._model_config()
        if rc:
            raise ParamError(f"hirp: the measurement core rejected the ambient temperature or humidity (MT {rc})", -3)

    def _measure(self) -> np.ndarray:
        return self._gray_temps(self.raw.reshape(-1)).reshape(self.height, self.width)

    def _measure_tenths(self, t: np.ndarray) -> np.ndarray:
        """libv_hirp measure: (short)(int)roundf(q * 2^-13 * 10.0f); t is q * 2^-13 exactly."""
        return to_short(roundf(t.astype(np.float32) * F(10.0)))

    def _gray_temps(self, flat: np.ndarray) -> np.ndarray:
        """IntegerCalculator::measure over a sequence of grays (the frame, in raster order)."""
        return (self._gray_q13(flat).astype(np.float32) * F(2.0 ** -13)).astype(np.float32)

    def _gray_q13(self, flat: np.ndarray, strict: bool = True, with_rc: bool = False, unclamped: bool = False):
        """The temperatures of a sequence of grays as TempAnalyzer::calculate returns them (Q13).
        strict: fail like dirp_measure when the last conversion reports a bad status;
        with_rc: also return each gray's MT_Process_INT return code; unclamped: also return
        the values before the camera's temperature-range clamp."""
        mt, h = self.mt, self.mt.h
        ems = (self.emissivity >> 3) & 0xFFFF
        refl, dist = self.reflection, self.distance

        def convert(gray):
            try:
                return mt.process(gray, ems, refl, dist)
            except hirp_mt.OutsideTable as e:
                raise MeasureError(f"hirp: {e} (the native SDK reads out of bounds here)", -4) from None

        mt.ub = False
        convert(_PROBE_GRAY)                      # IntegerCalculator::setExpertConfig probe
        mode = (self.gain_mode + 1) & 0xFF
        if mode == 3:
            lo, hi = 0, 0x3FFF
        else:
            lo, hi = h.u16(0x210), h.u16(0x20C)  # MT_Set_Gray_Thr_INT
        tmin, tmax = _TEMP_RANGE.get(mode, _TEMP_RANGE_NONE)

        grays, first_idx, inverse = np.unique(flat, return_index=True, return_inverse=True)
        values = np.empty(len(grays), dtype=np.int64)
        free = np.empty(len(grays), dtype=np.int64)
        rcs = np.zeros(len(grays), dtype=np.int64)
        for i, g in enumerate(grays.tolist()):
            if g < lo:
                v, rc = tmin, 0
            elif g > hi:
                v, rc = tmax, 0
            else:
                rc, pt = convert(g)
                t = F(F(F(pt[4]) * F(0.015625)) * F(10.0))
                v = cvtt(float(F(F(F(cvtt(float(t))) / F(10.0)) * F(8192.0))))
                free[i] = v
                v = min(max(v, tmin), tmax)
            if g < lo or g > hi:
                free[i] = v
            values[i], rcs[i] = v, rc
        if mt.ub:
            raise MeasureError("hirp: these parameters make the native SDK read outside a lookup table "
                               "(undefined result; it usually crashes)", -4)
        # the measurement's status is that of the last conversion in raster order: the first
        # occurrence of each gray, or every occurrence of grays whose value stays "uncomputed"
        last = first_idx.copy()
        sticky = values < _UNCOMPUTED
        if sticky.any():
            for i in np.nonzero(sticky)[0]:
                last[i] = np.nonzero(inverse == i)[0][-1]
        if strict and rcs[int(np.argmax(last))] != 0:
            raise MeasureError(f"hirp: temperature conversion failed (MT {int(rcs[int(np.argmax(last))])})", -4)
        out = (values[inverse],)
        if with_rc:
            out += (rcs[inverse],)
        if unclamped:
            out += (free[inverse],)
        return out if len(out) > 1 else out[0]


    # --- image pipeline -----------------------------------------------------------------
    def _isp_get_pseudo_color(self) -> int:
        return self.palette

    def _isp_set_pseudo_color(self, palette: int) -> None:
        if palette > 9:                              # signed check: negatives are accepted
            raise ParamError(f"pseudo color {palette} is not supported", -3)
        self.palette = palette
        self.strech_palette = self.processed = self.isp_stale = False

    def _isp_get_color_bar(self) -> ColorBar:
        return ColorBar(**vars(self.color_bar))

    def _isp_set_color_bar(self, cb: ColorBar) -> None:
        self.color_bar = ColorBar(cb.manual_enable, _q13(cb.high), _q13(cb.low))

    def _isp_color_bar_adaptive(self) -> ColorBar:
        if self.auto_range is None:
            raise IspError("run process at least once in color bar automatic mode", -13)
        return ColorBar(False, *self.auto_range)

    def _isp_get_isotherm(self) -> Isotherm:
        iso = Isotherm(**vars(self.isotherm))
        if self.strech_iso_off:               # the grey strech palette left behind
            iso.enable = False
        return iso

    def _isp_set_isotherm(self, iso: Isotherm) -> None:
        self.isotherm = Isotherm(iso.enable, _q13(iso.high), _q13(iso.low))
        self.strech_palette = self.strech_iso_off = self.processed = self.isp_stale = False

    def _isp_get_brightness(self) -> int:
        return self.brightness

    def _isp_set_brightness(self, brightness: int) -> None:
        self.brightness = brightness                 # libv_hirp does not range-check it

    def _isp_rgb_palettes(self) -> list[np.ndarray]:
        """PseudoColorMap::pseudoColorRGB: libv_hirp's RGB tables are libv_iirp's palettes."""
        return [self.t[f"isp_rgb_{DIRP2IRPROC[i]}"].reshape(256, 3) for i in range(10)]

    def _isp_pseudo_color_lut(self) -> np.ndarray:
        pal = self._isp_rgb_palettes()
        return np.ascontiguousarray(np.stack([np.stack([p[:, c] for p in pal]) for c in range(3)]))

    def _temp_to_gray(self, q13: int, line) -> int:
        """MicroSDK::GrayCalculator::gray: IntegerCalculator::gray (MT_Process_INT mode 1), or
        when that fails or reports an out-of-range status, the straight line through the
        frame's (raw min, temperature) and (raw max, temperature)."""
        t = int(float(F(q13)) * 2.0 ** -13 * 256.0)
        try:
            rc, pt = self.mt.temp2gray_point(t, (self.emissivity >> 3) & 0xFFFF, self.reflection, self.distance)
        except hirp_mt.OutsideTable:
            rc, pt = -1, None                     # only reached for nonsense parameters
        if rc == 0 and pt[0] == 0:
            return pt[1] or 1
        g1, t1, g2, t2 = line
        with np.errstate(all="ignore"):           # equal end temperatures divide by zero (IEEE)
            d1 = np.float64(F(F(t1) * F(2.0 ** -13)))
            d0 = (np.float64(g2) - np.float64(g1)) / (np.float64(F(F(t2) * F(2.0 ** -13))) - d1)
            v = (np.float64(F(F(q13) * F(2.0 ** -13))) * d0 + np.float64(g1)) - d0 * d1
        if v < 0.0:
            return 1
        if v > 16384.0:
            return 0x4000
        return 1 if np.isnan(v) else ((int(v) & 0xFFFF) or 1)   # (int)NaN is INT_MIN

    def _isp_gray(self) -> np.ndarray:
        return self._isp_render(isotherm=False)[0]

    def _isp_render(self, isotherm: bool):
        """The ISP output (y8, IFR raw, LUT, AGC outputs, handle) and the isotherm bands (raw
        limits) when isotherm is set."""
        raw = self.raw & 0x3FFF
        rmax, rmin = int(raw.max()), int(raw.min())
        # with a manual colour bar or an isotherm process re-applies the measurement
        # parameters; when TempAnalyzer::setModelConfig refuses them (after a rejected
        # set_measurement_params) there is no GrayCalculator: a manual bar spans the whole
        # 14-bit range, an isotherm band is dropped
        stats = self.color_bar.manual_enable or isotherm
        model_ok = not stats or self._model_config() == 0
        try:
            # TempAnalyzer::calculate of the frame's extremes
            tq, rcs = self._gray_q13(np.array([rmin, rmax], dtype=np.uint16), strict=False, with_rc=True)
        except MeasureError as e:
            raise IspError(str(e), -32) from None
        extremes_ok = not rcs.any()
        line = None
        if stats and model_ok and extremes_ok:
            # TempAnalyzer::regionStatsETC builds the GrayCalculator from the frame's coldest
            # and hottest grays (ties: the lowest gray for the coldest, the highest for the
            # hottest)
            grays = np.unique(raw)
            q = self._gray_q13(grays, strict=False)
            i = int(np.flatnonzero(q == q.min())[0])
            j = int(np.flatnonzero(q == q.max())[-1])
            line = (int(grays[i]), int(q[i]), int(grays[j]), int(q[j]))

        def to_gray(t):
            return self._temp_to_gray(cvtt(float(F(F(t) * F(8192.0)))), line)
        manual = None
        if self.color_bar.manual_enable and line is None:
            manual = (0, 0x4000)              # TempAnalyzer::regionStatsETC failed
        elif self.color_bar.manual_enable:
            # libv_hirp: the colour bar's limits as grays, then ISPProcess::setManualAGCConfig
            lo, hi = to_gray(self.color_bar.low), to_gray(self.color_bar.high)
            if hi <= lo:
                hi = lo + 1
            if hi > 0x4000:                        # setManualAGCConfig refuses it
                raise IspError("hirp: the manual colour bar lies above the camera's range", -4)
            manual = (lo, hi)
        bands = None
        if isotherm:
            # the isotherm limits as grays (no GrayCalculator: lower 0, which drops the band)
            bands = [(to_gray(self.isotherm.low), to_gray(self.isotherm.high)) if line else (0, 0x3FFF)]
        out = hirp_isp.render(self.raw, self.app5, self.tp, self.brightness, manual)
        # the adaptive range: the temperatures of the frame's raw max / min
        # the adaptive range: TempAnalyzer::calculate of the raw max, then of the raw min; a
        # failed conversion (MT left half configured by a refused set_addl) reads 0, and the
        # range is unavailable when both failed
        if rcs.all():
            self.auto_range = None
        else:
            t = np.where(rcs == 0, tq, 0).astype(np.float32) * F(2.0 ** -13)
            self.auto_range = (float(t[1]), float(t[0]))
        return out + (bands,)

    def _isp_process(self) -> np.ndarray:
        pal = self._isp_rgb_palettes()
        # PseudoColorMap::pseudoColorRGB: anything but 1..9 selects palette 0's table
        pc = 0 if self.strech_palette else self.palette
        # the grey palette process_strech can leave behind has the isotherm switched off
        iso = self.isotherm.enable and not self.strech_iso_off
        try:
            y8, raw, lut, v, hb, bands = self._isp_render(isotherm=iso)
        except IspError as e:
            if e.code == -4 and self.processed:
                self.isp_stale = True
            raise
        if self.isp_stale:
            raise UnsupportedFeature("hirp: rendering after a refused manual colour bar reuses native ISP "
                                     "state that is not ported to Python")
        self.processed = True
        rgb = pal[pc if 1 <= pc <= 9 else 0]
        if bands is None:
            return hirp_isp.colorize(y8, rgb)
        # isotherm: the band in the selected palette, everything else in white hot (0); the
        # grey palettes white hot / black hot (9) keep themselves outside and use iron (2)
        # inside, and an invalid palette id stays palette 0's table throughout
        outside, inside = {0: (0, 2), 9: (9, 2)}.get(pc, (0, pc if 1 <= pc <= 9 else 0))
        return hirp_isp.colorize_isotherm(y8, raw, lut, v, hb, bands, pal[outside], pal[inside])

    def _isp_process_strech(self) -> np.ndarray:
        self.processed = self.isp_stale = False
        try:
            return self._isp_gray().astype(np.float32)
        except IspError as e:
            if e.code == -4:
                self.strech_palette = self.strech_iso_off = True
            raise


def _q13(v: float) -> float:
    """libv_hirp keeps colour-bar / isotherm limits as (int)(v * 8192) and returns them as float."""
    return float(F(F(cvtt(float(F(F(v) * F(8192.0))))) * F(1.0 / 8192.0)))


# libv_hirp fromPaletteType: MicroSDK palette type (ImageParams) -> dirp_pseudo_color_e
_PALETTE_TYPE = {2: 9, 10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6, 16: 7, 17: 8}
