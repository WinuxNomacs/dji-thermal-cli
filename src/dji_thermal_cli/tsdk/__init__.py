"""Pure-Python port of the DJI Thermal SDK v1.8 temperature measurement.

Mirrors the native SDK's structure: libdirp picks a per-camera plugin from the EXIF camera
model and the APP4 header tag, each plugin turns the raw frame into temperatures, and
libdirp then applies the infrared ND-filter correction. Every implemented plugin is
bit-identical to the native SDK. The dirp API maps onto RJpeg methods:

    dirp_measure_ex / dirp_measure          measure_temperature_c() / measure()
    dirp_process / dirp_process_strech      process() / process_strech()
    dirp_get_original_raw                   get_original_raw()
    dirp_get_api_version                    get_api_version()
    dirp_get_infared_density_filter_installed   infrared_density_filter_installed()
    dirp_get/set_measurement_params(_range), _color_bar(_adaptive_params), _isotherm,
    _pseudo_color(_lut), _enhancement_params, rjpeg_version, rjpeg_resolution: same names

(dirp_set_verbose_level and dirp_set_logger_file only configure the native library's log.)

    rj = tsdk.open(data)
    params = rj.get_measurement_params()
    params.emissivity = 0.95
    rj.set_measurement_params(params)
    temps = rj.measure_temperature_c()      # float32 Celsius, (height, width)

Implemented: iirp (M2EA, M30T, M3T, M3TD, H20N, H30T, M4T), girp (M30T), dirp (H20T, XT S),
hirp (newer M3TD/M4T firmware).
"""

from __future__ import annotations

from . import dirp, jpeg, tables
from .base import ColorBar, Isotherm, MeasurementParams, RJpeg
from .errors import (IspError, MeasureError, ParamError, RJpegError, TsdkError, UndefinedBehaviour,
                     UnsupportedFeature, UnsupportedRJpeg)

__all__ = [
    "open", "plugin_name", "MeasurementParams", "RJpeg", "TsdkError", "RJpegError", "UnsupportedRJpeg",
    "ParamError", "MeasureError", "tables", "ColorBar", "Isotherm", "IspError",
    "UndefinedBehaviour", "UnsupportedFeature",
]

_DIRP_MODELS = ("ZH20T", "XT S")
_M2EA = "MAVIC2-ENTERPRISE-ADVANCED"
# libdirp reports API revision 19 for these camera models (EXIF Model) and 20 for the rest
_REVISION_19_MODELS = ("XT S", "ZH20T", _M2EA, "ZH20N", "M30T", "M3T", "M3TD")


def plugin_name(segs: list[jpeg.Segment]) -> str:
    """libdirp's plugin choice: camera model first, then the APP4 header tag overrides it."""
    _, model = jpeg.exif_make_model(segs)
    plugin = "none"
    if model in _DIRP_MODELS:
        plugin = "dirp"
    elif model == _M2EA:
        plugin = "iirp" if _resolution(segs) == (640, 512) else "unknown"
    app4 = jpeg.first(segs, jpeg.APP4)
    if app4 is not None and len(app4) > 0x20 and app4[0] == 0xFF and app4[3] == 0xFF and app4[1:3] == b"\xd2\xd1":
        tag = app4[4:12].split(b"\0", 1)[0].decode("latin-1")
        if tag:
            plugin = tag
    return plugin


def _resolution(segs) -> tuple[int, int] | None:
    """EXIF PixelXDimension/PixelYDimension are not needed: M2EA thermal frames are 640x512
    whenever the raw APP3 payload has 640*512 16-bit samples."""
    n = len(jpeg.concat(segs, jpeg.APP3))
    return (640, 512) if n == 640 * 512 * 2 else None


def open(data: bytes) -> RJpeg:  # noqa: A001 - mirrors dirp_create_from_rjpeg
    """Parse an R-JPEG and return its plugin. Raises RJpegError / UnsupportedRJpeg."""
    segs = jpeg.segments(data)
    rj = _open_plugin(segs, tables.load())
    rj.api_revision = 19 if jpeg.exif_make_model(segs)[1] in _REVISION_19_MODELS else 20
    return rj


def _open_plugin(segs, t: dict):
    plugin = plugin_name(segs)
    nd = jpeg.nd_filter(segs)
    if plugin == "iirp":
        from .iirp import Iirp
        return Iirp(segs, t, nd)
    if plugin == "girp":
        from .girp import Girp
        return Girp(segs, t, nd)
    if plugin == "dirp":
        if dirp.is_class_d(segs):
            return dirp.H20T(segs, nd, t)
        if dirp.is_class_g(segs):
            return dirp.XTS(segs, nd, t)
        raise UnsupportedRJpeg("unknown dirp R-JPEG class")
    if plugin == "hirp":
        from .hirp import Hirp
        return Hirp(segs, t, nd)
    if plugin in ("none", "unknown"):
        if jpeg.is_flir(segs):
            model = jpeg.exif_make_model(segs)[1]
            raise UnsupportedRJpeg("a FLIR radiometric JPEG" + (f" ({model})" if model else "") +
                                   ", not a DJI R-JPEG format")
        raise RJpegError("no DJI thermal data found (not an R-JPEG)")
    raise UnsupportedRJpeg(f"R-JPEG plugin {plugin!r} is not supported")
