"""Common interface of the per-format backends (the SDK's libv_* plugins)."""

from __future__ import annotations

import dataclasses

import numpy as np

from ._cnum import F, to_short
from .errors import ParamError, UnsupportedFeature
from .jpeg import NdFilter


@dataclasses.dataclass
class MeasurementParams:
    """dirp_measurement_params_t. humidity is a percent (0-100); temperatures are Celsius."""

    distance: float
    humidity: float
    emissivity: float
    reflection: float
    ambient_temp: float

    FIELDS = ("distance", "humidity", "emissivity", "reflection", "ambient_temp")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.FIELDS}


@dataclasses.dataclass
class ColorBar:
    """dirp_color_bar_t (also the adaptive range). Temperatures are Celsius."""

    manual_enable: bool
    high: float
    low: float


@dataclasses.dataclass
class Isotherm:
    """dirp_isotherm_t."""

    enable: bool
    high: float
    low: float


PSEUDO_COLOR_NUM = 10       # dirp_pseudo_color_e: white hot .. black hot


class RJpeg:
    """One opened R-JPEG. Subclasses implement the camera family's own plugin."""

    plugin = "?"
    width = 0
    height = 0
    nd = NdFilter(False, 0.0, 0.0)
    api_revision = 20           # set by open() from the camera model, as libdirp does

    def get_api_version(self) -> tuple[int, str]:
        """dirp_get_api_version: (API revision, magic)."""
        return self.api_revision, "relver"

    def infrared_density_filter_installed(self) -> bool:
        """dirp_get_infared_density_filter_installed."""
        return bool(self.nd.installed)

    def get_original_raw(self) -> np.ndarray:
        """dirp_get_original_raw: the raw frame as the plugin keeps it, uint16 (height, width)."""
        return self._raw().astype(np.uint16, copy=True)

    def _raw(self) -> np.ndarray:
        return self.raw.view(np.uint16)

    def rjpeg_version(self) -> tuple[int, int, int]:
        """dirp_get_rjpeg_version: (rjpeg, header, curve) as the camera's plugin reports them."""
        raise NotImplementedError

    def resolution(self) -> tuple[int, int]:
        """dirp_get_rjpeg_resolution: (width, height)."""
        return self.width, self.height

    def get_measurement_params(self) -> MeasurementParams:
        raise NotImplementedError

    def set_measurement_params(self, params: MeasurementParams) -> None:
        raise NotImplementedError

    def get_measurement_params_range(self) -> dict[str, tuple[float, float]]:
        raise NotImplementedError

    def _measure(self) -> np.ndarray:
        raise NotImplementedError

    def _measure_tenths(self, t: np.ndarray) -> np.ndarray:
        """The plugin's dirp_measure output from its own temperatures: (short)(int)(10.0f * t)."""
        return to_short(t.astype(np.float32) * F(10.0))

    def measure(self) -> np.ndarray:
        """dirp_measure: int16 tenths of a degree Celsius, shape (height, width)."""
        v = self._measure_tenths(self._measure())
        return self._nd_tenths(v) if self.nd.installed else v

    def _nd_tenths(self, v: np.ndarray) -> np.ndarray:
        """libdirp's ND-filter correction of dirp_measure output, in float32."""
        return to_short((v.astype(np.float32) * F(self.nd.scale)).astype(np.float32) + F(F(self.nd.offset) * F(10.0)))

    def measure_temperature_c(self) -> np.ndarray:
        """dirp_measure_ex: float32 Celsius, shape (height, width)."""
        t = self._measure()
        if self.nd.installed:            # applied by libdirp on top of every plugin
            t = (t * F(self.nd.scale)).astype(np.float32) + F(F(self.nd.offset) * F(10.0))
        return t.astype(np.float32, copy=False)

    # --- image pipeline (dirp_process and its settings) --------------------------------
    # The public methods carry libdirp's own layer (threshold checks, the ND filter mapping of
    # colour-bar temperatures); the _isp_* hooks are the camera plugin's.

    def get_pseudo_color(self) -> int:
        return self._isp_get_pseudo_color()

    def set_pseudo_color(self, palette: int) -> None:
        self._isp_set_pseudo_color(int(palette))

    def get_color_bar(self) -> ColorBar:
        return self._nd_out(self._isp_get_color_bar())

    def get_color_bar_adaptive_params(self) -> ColorBar:
        return self._nd_out(self._isp_color_bar_adaptive())

    def set_color_bar(self, cb: ColorBar) -> None:
        high, low = F(cb.high), F(cb.low)
        if cb.manual_enable and high <= low:
            raise ParamError(f"color bar's upper threshold ({float(high):f}) must be greater than the lower threshold ({float(low):f})")
        if self.nd.installed:
            scale, offset = F(self.nd.scale), F(self.nd.offset)
            if abs(float(scale)) < 1e-07:
                high, low = F(high - offset), F(low - offset)
            else:
                high, low = F(F(high - offset) / scale), F(F(low - offset) / scale)
            adaptive = self._isp_color_bar_adaptive()
            high = high if high <= F(adaptive.high) else F(adaptive.high)
            low = low if F(adaptive.low) <= low else F(adaptive.low)
        self._isp_set_color_bar(ColorBar(bool(cb.manual_enable), float(high), float(low)))

    def _nd_out(self, cb: ColorBar) -> ColorBar:
        if self.nd.installed:
            scale, offset = F(self.nd.scale), F(self.nd.offset)
            cb = ColorBar(cb.manual_enable, float(F(F(F(cb.high) * scale) + offset)), float(F(F(scale * F(cb.low)) + offset)))
        return cb

    def get_isotherm(self) -> Isotherm:
        return self._isp_get_isotherm()

    def set_isotherm(self, iso: Isotherm) -> None:
        if iso.enable and F(iso.high) <= F(iso.low):
            raise ParamError(f"isotherm's upper threshold ({float(F(iso.high)):f}) must be greater than the lower threshold ({float(F(iso.low)):f})")
        self._isp_set_isotherm(Isotherm(bool(iso.enable), float(F(iso.high)), float(F(iso.low))))

    def get_enhancement_params(self) -> int:
        """dirp_enhancement_params_t.brightness (0-100)."""
        return self._isp_get_brightness()

    def set_enhancement_params(self, brightness: int) -> None:
        self._isp_set_brightness(int(brightness))

    def get_pseudo_color_lut(self) -> np.ndarray:
        """dirp_isp_pseudo_color_lut_t as uint8 (3, PSEUDO_COLOR_NUM, 256): red, green, blue."""
        return self._isp_pseudo_color_lut()

    def process(self) -> np.ndarray:
        """dirp_process: the pseudo-colour rendering, uint8 RGB (height, width, 3)."""
        return self._isp_process()

    def process_strech(self) -> np.ndarray:
        """dirp_process_strech: the enhanced (stretched) frame as float32 (height, width)."""
        return self._isp_process_strech()

    def _unsupported(self, *_):
        raise UnsupportedFeature(f"pseudo-colour rendering of {self.plugin} R-JPEGs is not ported to Python yet")

    _isp_get_pseudo_color = _isp_set_pseudo_color = _isp_get_color_bar = _isp_set_color_bar = _unsupported
    _isp_color_bar_adaptive = _isp_get_isotherm = _isp_set_isotherm = _isp_get_brightness = _unsupported
    _isp_set_brightness = _isp_pseudo_color_lut = _isp_process = _isp_process_strech = _unsupported
