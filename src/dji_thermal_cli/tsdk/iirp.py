"""The "iirp" plugin (libv_iirp): M2EA, M30T, M3T, M3TD, H20N, H30T, M4T.

Each raw count is already a blackbody temperature: T0 = (raw >> 2) / 16 - 273.15. The plugin
corrects it per pixel for emissivity, reflected and atmospheric temperature and atmospheric
transmission (tau). Tau comes from the file's own APP5 table, interpolated over (target
temperature, distance) and adjusted for humidity; emissivity is optionally remapped through a
DJI correction table. The correction runs twice: tau is re-read at the first-pass estimate.

Float32/double placement, integer truncations and table edge behaviour follow libv_iirp
exactly, so results are bit-identical to the native SDK.
"""

from __future__ import annotations

import math
import struct

import numpy as np

from ._cnum import F, cvtt, per_value
from . import iirp_isp
from .base import ColorBar, Isotherm, MeasurementParams, RJpeg
from .errors import MeasureError, ParamError, RJpegError, UndefinedBehaviour, UnsupportedRJpeg
from .jpeg import APP3, APP4, APP5, NdFilter, concat, first, first_segment

MAGIC = 0xFFD1D2FF          # APP4 bytes FF D2 D1 FF, followed by the "iirp" tag


class Iirp(RJpeg):
    plugin = "iirp"

    def __init__(self, segs, tables: dict[str, np.ndarray], nd: NdFilter):
        self.t = tables
        self.nd = nd
        app4, seg5 = first(segs, APP4), first_segment(segs, APP5)
        if app4 is None or seg5 is None:
            raise RJpegError("iirp R-JPEG without APP4/APP5")
        self.magic = len(app4) >= 4 and struct.unpack_from("<I", app4, 0)[0] == MAGIC
        off = 0x20 if self.magic else 0
        if len(app4) < off + 0x80:
            raise RJpegError("truncated iirp header")
        # handle+0x38: ambient, distance, emissivity, humidity (0-1), reflection
        self.amb, self.dist, self.ems, self.hum, self.refl = (F(x) for x in struct.unpack_from("<5f", app4, off))
        blk = off + 0x60                                  # handle+0x98 block
        self.gain = app4[blk]
        if self.magic:
            self.version = struct.unpack_from("<I", app4, blk + 0x19)[0]
            self.width, self.height = struct.unpack_from("<HH", app4, blk + 0x1D)
        else:
            self.version, self.width, self.height = 0x100, 640, 512
        if self.gain == 3:
            raise UnsupportedRJpeg("iirp 'high high gain' captures cannot be measured (the native SDK refuses them too)")
        if self.width == 0 or self.height == 0:
            raise RJpegError("iirp header has a zero image size")
        raw = np.frombuffer(concat(segs, APP3), dtype="<u2")
        if raw.size != self.width * self.height:
            raise RJpegError(f"iirp raw frame has {raw.size} pixels, expected {self.width}x{self.height}")
        self.y14 = (raw >> 2).astype(np.uint16).reshape(self.height, self.width)

        # image pipeline settings (handle+0x99 palette, +0x844e isotherm, +0x8457 colour bar,
        # +0x8460 brightness). v0300 headers store the palette as an "ircmd" id, converted on open.
        self.palette = app4[blk + 1]
        if self._v0300:
            if self.palette not in iirp_isp.IRCMD2IRPROC:
                raise RJpegError(f"iirp header has an unknown palette id {self.palette}")
            self.palette = iirp_isp.IRCMD2IRPROC[self.palette]
        self.color_bar = ColorBar(False, 30.0, 25.0)
        self.isotherm = Isotherm(False, 30.0, 25.0)
        self.brightness = 50

        app5 = seg5.payload
        head = app5[:0x100]
        if head[:4] == b"\xff\xff\xff\xff":
            name = "default_tau_gain1" if self.gain == 1 else "default_tau_gain0"
            tau = tables[name].tobytes().ljust(0x1F00, b"\0")
            self.dist_max = 25.0
        else:
            if ((self.version - 0x100) & 0xFFFFFEFF) == 0:   # old 0x1c00-byte tau table
                tau = seg5.read(0x100, 0x1C00).ljust(0x1F00, b"\0")
                self.dist_max = 25.0
            else:
                tau = seg5.read(0x100, 0x1F00)     # may run past APP5 into the next segment (sic)
                self.dist_max = 300.0 if (head[3] << 8 | head[2]) > 0x3F else 25.0
        if len(tau) != 0x1F00:
            raise RJpegError("truncated iirp tau table")
        table = np.frombuffer(head.ljust(0x100, b"\0") + tau, dtype="<u2")
        h = table[:0x80]
        if h[2] == 0xFFFF and h[3] == 0xFFFF:
            self.hdr = dict(ver=int(h[1]), ems_on=int(h[4]), ems_ver=int(h[5]), hum_on=int(h[6]), hum_ver=int(h[7]),
                            refl_on=int(h[10]), refl_ver=int(h[11]))
            self.tau = table[0x80:]
        else:
            self.hdr = dict(ver=0, ems_on=0, ems_ver=0, hum_on=0, hum_ver=0, refl_on=0, refl_ver=0)
            self.tau = table
        self._grids()

    # --- parameters -------------------------------------------------------------------
    @property
    def _v0300(self) -> bool:
        return self.magic and (self.version & 0xFFFEFFFF) != 0x200 and self.version != 0x20200

    def rjpeg_version(self):
        return self.version, 1, 1              # libv_iirp get_rjpeg_version: header/curve fixed at 1

    def get_measurement_params(self) -> MeasurementParams:
        return MeasurementParams(float(self.dist), float(F(self.hum * F(100.0))), float(self.ems), float(self.refl), float(self.amb))

    def get_measurement_params_range(self):
        return {"distance": (1.0, self.dist_max), "humidity": (1.0, 100.0), "emissivity": (float(F(0.1)), 1.0),
                "reflection": (-40.0, 100.0), "ambient_temp": (-40.0, 80.0)}

    def set_measurement_params(self, p: MeasurementParams) -> None:
        rng = self.get_measurement_params_range()
        names = ["distance", "humidity", "emissivity", "reflection"] + (["ambient_temp"] if self._v0300 else [])
        for name in names:
            lo, hi = rng[name]
            v = F(getattr(p, name))
            if v > F(hi) or v < F(lo):
                raise ParamError(f"{name}={float(v):g} is outside [{lo:g}, {hi:g}]")
        self.refl = F(p.reflection)
        self.dist = F(p.distance)
        self.hum = F(float(F(p.humidity)) / 100.0)
        self.ems = F(p.emissivity)
        # Before the v0300 header the plugin takes the ambient temperature from the
        # reflection value, ignoring the ambient argument.
        self.amb = F(p.ambient_temp) if self._v0300 else F(p.reflection)

    # --- image pipeline -----------------------------------------------------------------
    def _isp_get_pseudo_color(self) -> int:
        if self.palette not in iirp_isp.IRPROC2DIRP:
            raise ParamError("color mode unsupported", -3)
        return iirp_isp.IRPROC2DIRP[self.palette]

    def _isp_set_pseudo_color(self, palette: int) -> None:
        if palette not in iirp_isp.DIRP2IRPROC:
            raise ParamError(f"pseudo color {palette} is not supported", -3)
        self.palette = iirp_isp.DIRP2IRPROC[palette]

    def _isp_get_color_bar(self) -> ColorBar:
        return ColorBar(**vars(self.color_bar))

    def _isp_set_color_bar(self, cb: ColorBar) -> None:
        self.color_bar = cb

    def _isp_color_bar_adaptive(self) -> ColorBar:
        high, low = iirp_isp.adaptive_params(self.y14)
        return ColorBar(False, float(high), float(low))

    def _isp_get_isotherm(self) -> Isotherm:
        return Isotherm(**vars(self.isotherm))

    def _isp_set_isotherm(self, iso: Isotherm) -> None:
        self.isotherm = iso

    def _isp_get_brightness(self) -> int:
        return self.brightness

    def _isp_set_brightness(self, brightness: int) -> None:
        if not 0 <= brightness <= 100:
            raise ParamError("brightness out of range", -3)
        self.brightness = brightness

    def _isp_tables(self):
        return self.t

    def _isp_pseudo_color_lut(self) -> np.ndarray:
        return iirp_isp.pseudo_color_lut(self._isp_tables())

    def _isp_process(self) -> np.ndarray:
        cb, iso = self.color_bar, self.isotherm
        return iirp_isp.process(self._isp_tables(), self.y14.reshape(-1), self.width, self.height, self.gain,
                                self.palette, self.magic, self.version, (cb.manual_enable, cb.high, cb.low),
                                (iso.enable, iso.high, iso.low), self.brightness, self._reverse_correct)

    def _isp_process_strech(self) -> np.ndarray:
        cb = self.color_bar
        return iirp_isp.process_strech(self._isp_tables(), self.y14.reshape(-1), self.width, self.height, self.gain,
                                       (cb.manual_enable, cb.high, cb.low))

    def _reverse_correct(self, t_c) -> np.float32:
        """reverse_enhance_distance_temp_correct: the raw-domain temperature that measures as t_c
        (the manual colour bar's limits). Its status is ignored by process(); a refusal would leave
        the value uninitialised there."""
        if not (0.0 <= self.ems <= 1.0 and 0.0 <= self.hum <= 1.0 and self.dist >= 0.0
                and -273.15 <= self.amb <= 100.0 and -273.15 <= self.refl <= 100.0):
            raise UndefinedBehaviour("colour bar conversion refused for these parameters (native result undefined)")
        t = F(t_c)
        tau = self._hum_tau(t, self._read_tau(t))
        r1 = self._reverse_temp_correct(self._read_ems(t), t, tau)
        return self._reverse_temp_correct(self._read_ems(r1), t, tau)

    def _reverse_temp_correct(self, ems, t_c, tau):
        """reverse_temp_correct / reverse_temp_correct_with_reflected_temp (Q14 tau)."""
        if ems < 0.0001 or tau == 0:
            raise UndefinedBehaviour("colour bar conversion refused (native result undefined)")
        e = float(ems)
        a = math.pow(float(F(t_c)) + 273.15, 4.0)
        c = math.pow(float(self.amb) + 273.15, 4.0)
        if self.hdr["refl_on"] == 1:
            if self.hdr["refl_ver"] != 1:
                raise UndefinedBehaviour("reflected-temperature table version mismatch")
            b = math.pow(float(self.refl) + 273.15, 4.0)
            v = c * (1.0 - tau / 16384.0) + (b * (1.0 - e) * tau) / 16384.0 + (tau * e * a) / 16384.0
        else:
            v = c * (1.0 - (tau * e) / 16384.0) + (tau * e * a) / 16384.0
        return F(float(F(math.sqrt(math.sqrt(v)))) - 273.15)

    # --- measurement --------------------------------------------------------------------
    def _measure(self) -> np.ndarray:
        if not (0.0 <= self.ems <= 1.0 and 0.0 <= self.hum <= 1.0 and 0.0 <= self.dist <= 1000.0
                and -273.15 <= self.amb <= 100.0 and -273.15 <= self.refl <= 100.0):
            raise MeasureError("measurement parameters out of the plugin's range")
        out = per_value(self.y14, lambda v: self._correct(F(float(F(v)) * 0.0625 - 273.15)))
        back = ((out.astype(np.float64) + 273.15) * 16.0).astype(np.int64) & 0xFFFF
        if not back.any():
            raise MeasureError("after measure_ex all data is 0", -65)
        return out

    def _raw(self) -> np.ndarray:
        return self.y14

    def _grids(self):
        t, ver = self.t, self.hdr["ver"]
        if ver < 0x40:
            self.tg, self.dg = t["target_temp_table"], t["dist_table"]
        elif ver < 0x101:
            self.tg, self.dg = t["new_target_temp_table"], t["new_dist_table"]
        else:
            self.tg, self.dg = t["target_temp_table_v3"], t["dist_table_v3"]

    def _correct(self, t_raw):
        """enhance_distance_temp_correct."""
        e2 = self._read_ems(t_raw)
        tau1 = self._read_tau(t_raw)
        t1 = self._temp_correct(e2, t_raw, tau1)
        tau2 = self._hum_tau(t1, self._read_tau(t1))
        return float(self._temp_correct(e2, t_raw, tau2))

    def _read_tau(self, t_c) -> int:
        tg, dg = self.tg, self.dg
        x = float(F(float(F(t_c)) + 273.15))
        y = float(self.dist)
        i, j = _idx_lin(x, tg), _idx_lin(y, dg)
        nt, nd = len(tg), len(dg)
        cell = lambda a, b: int(self.tau[b + a * nd])
        if i < nt - 1:
            if j < nd - 1:
                area = abs(dg[j + 1] - dg[j]) * abs(tg[i + 1] - tg[i])
                v = (cell(i, j) * (((dg[j + 1] - y) * (tg[i + 1] - x)) / area)
                     + cell(i, j + 1) * (((y - dg[j]) * (tg[i + 1] - x)) / area)
                     + cell(i + 1, j) * (((dg[j + 1] - y) * (x - tg[i])) / area)
                     + cell(i + 1, j + 1) * (((y - dg[j]) * (x - tg[i])) / area) + 0.5)
                return int(v) & 0xFFFF
            span = abs(tg[i + 1] - tg[i])
            return int(cell(i + 1, j) * ((x - tg[i]) / span) + cell(i, j) * ((tg[i + 1] - x) / span)) & 0xFFFF
        if j < nd - 1:          # last temperature row: both weights land on the same cell (sic)
            span = abs(dg[j + 1] - dg[j])
            return int(cell(i, j) * ((y - dg[j]) / span) + cell(i, j) * ((dg[j + 1] - y) / span)) & 0xFFFF
        return cell(i, j)

    def _read_ems(self, t_c):
        hdr, ems = self.hdr, self.ems
        on, ev = hdr["ems_on"], hdr["ems_ver"]
        if hdr["ver"] == 0x101:
            on, ev = 1, 1
        if on != 1:
            return ems
        e = float(ems)
        tk = float(F(float(F(t_c)) + 273.15))
        if ev == 1:
            org, tt, tab = self.t["org_ems_list_of_ems_table_v1"], self.t["target_temp_list_of_ems_table_v1"], self.t["ems_correct_table_v1"]
            if e > org[-1] or e < org[0]:
                return ems
            return F(tab[_idx_ems(e, org, 1) + _idx_ems(tk, tt, 1) * 20])
        if ev not in (2, 3):
            raise MeasureError(f"unknown emissivity table version {ev}")
        org, tt, tab = self.t["org_ems_list_of_ems_table_v2"], self.t["target_temp_list_of_ems_table_v2"], self.t["ems_correct_table_v2"]
        cell = lambda a, b: float(F(tab[b + a * 100]))
        inside = org[0] <= e <= org[-1]
        if inside:
            i, j = _idx_ems(tk, tt, 2), _idx_ems(e, org, 2)
            if i < 80:
                if j < 99:
                    area = abs(org[j + 1] - org[j]) * abs(tt[i + 1] - tt[i])
                    w00 = ((org[j + 1] - e) * (tt[i + 1] - tk)) / area
                    w01 = ((e - org[j]) * (tt[i + 1] - tk)) / area
                    w10 = ((org[j + 1] - e) * (tk - tt[i])) / area
                    w11 = ((e - org[j]) * (tk - tt[i])) / area
                    return F(cell(i + 1, j + 1) * w11 + cell(i, j) * w00 + cell(i, j + 1) * w01 + cell(i + 1, j) * w10)
                span = abs(tt[i + 1] - tt[i])
                return F(cell(i + 1, j) * ((tk - tt[i]) / span) + cell(i, j) * ((tt[i + 1] - tk) / span))
            if j < 99:
                span = abs(org[j + 1] - org[j])
                return F(cell(i, j + 1) * ((e - org[j]) / span) + cell(i, j) * ((org[j + 1] - e) / span))
            return F(cell(i, j))
        if ev == 2:
            return ems
        i = _idx_ems(tk, tt, 2)
        if e <= org[99]:            # below the table: scale the first column
            if i < 80:
                span = abs(tt[i + 1] - tt[i])
                c = float(F(cell(i + 1, 0) * ((tk - tt[i]) / span) + cell(i, 0) * ((tt[i + 1] - tk) / span)))
            else:
                c = cell(i, 0)
            return F(e * (c / org[0]))
        if i < 80:                  # above the table: blend the last column towards 1
            span = abs(tt[i + 1] - tt[i])
            c = float(F(cell(i + 1, 99) * ((tk - tt[i]) / span) + cell(i, 99) * ((tt[i + 1] - tk) / span)))
        else:
            c = cell(i, 99)
        return F((e - org[99]) * ((1.0 - c) / (1.0 - org[99])) + c)

    def _temp_correct(self, ems, t_c, tau):
        """temp_correct / temp_correct_with_reflected_temp (Q14 tau)."""
        if ems < 0.0001 or tau == 0:
            raise MeasureError("emissivity or tau is zero")
        a = math.pow(float(F(float(t_c) + 273.15)), 4.0)
        c = math.pow(float(self.amb) + 273.15, 4.0)
        if self.hdr["refl_on"] == 1:
            if self.hdr["refl_ver"] != 1:
                raise MeasureError("reflected-temperature table version mismatch")
            b = math.pow(float(self.refl) + 273.15, 4.0)
            v = (a - (b * (1.0 - float(ems)) * tau) / 16384.0) - c * (1.0 - tau / 16384.0)
        else:
            v = a - c * (1.0 - (tau * float(ems)) / 16384.0)
        v = max(v, 0.0)
        r = F(math.sqrt(math.sqrt(v / float(F(F(F(tau) * ems) / F(16384.0))))))
        return F(float(r) - 273.15)

    def _hum_tau(self, t_first, tau) -> int:
        """recalc_tau_under_influence_of_hum."""
        ver, on, hv = self.hdr["ver"], self.hdr["hum_on"], self.hdr["hum_ver"]
        if ver < 0x102:
            on, hv = 1, 1
        if ver == 0x110:
            on, hv = 1, 2
        if ver in (0x104, 0x113, 0x116, 0x119):
            on, hv = 1, 3
        if on != 1:
            return tau
        if not 1 <= hv <= 3:
            raise MeasureError(f"unknown humidity correction version {hv}")
        ta = float(self.amb)
        vp = F(float(self.hum) * 611.2 * math.exp(17.67 * (ta / (243.5 + ta))))
        vp = F(min(max(float(vp), 800.0), 3000.0 if hv == 1 else 4000.0))
        tk = float(F(float(F(t_first)) + 273.15))
        if tk < 273.15:
            coef = F(0.0)
        elif tk < 410.15:
            coef = F((math.pow(410.15 / tk, 4.0) - 1.0) * 0.01 - 0.04)
        elif tk < 1273.1501:
            coef = F((math.pow(tk / 410.15, 4.0) - 1.0) * -0.003 - 0.04)
        else:
            coef = F(0.0)
        if hv == 2:
            coef = F(0.6 * float(coef))
        elif hv == 3:
            coef = F(0.3 * float(coef))
        d = max(min(float(self.dist), 1000.0), 1.0)
        if hv in (2, 3) and d > 10.0:
            d = 10.0
        base = F(F(cvtt(float(vp))) / F(1425.0))     # vapour pressure is passed on as an integer
        p = math.pow(float(base), math.log(float(F(d))) * 1.4427 * float(coef))
        return cvtt(float(F(F(tau) * F(p)))) & 0xFFFF


def _idx_lin(x, grid) -> int:
    n = len(grid)
    if x < grid[0]:
        return 0
    if x >= grid[n - 1]:
        return n - 1
    k = 0
    while k < n and grid[k] - 0.0001 < x:
        k += 1
    return max(k - 1, 0)


def _idx_ems(x, grid, mode) -> int:
    n = len(grid)
    if x < grid[0]:
        return 0
    if x > grid[n - 1]:
        return n - 1
    k = 0
    while k < n and grid[k] - 0.0001 < x:
        k += 1
    if mode == 1:
        k = min(k, n - 1)
        d2 = grid[k] - grid[k - 1]
        d3 = grid[k] - x
        if abs(d3 - d2 / 2.0) >= 1e-05:
            return k - 1 if d2 / 2.0 <= d3 else k
        return k
    return max(k - 1, 0)
