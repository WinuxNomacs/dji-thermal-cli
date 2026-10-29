"""Batch discovery, per-file parameter resolution, and the process/list operations.

Does NOT rewrite the original R-JPEG: DJI's R-JPEG embeds its environmental
parameters in a binary APP4 block that is fixed at capture and not writable
by this tool. `process_file` only ever produces new derivative files.

Temperatures and pseudo-color images come from the pure-Python port of the DJI
Thermal SDK in the tsdk package: bit-identical to the SDK, with no native library
at run time. Anything the port cannot do is reported as a skipped file or a note,
with the reason.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from . import adjust, djixmp, mosaic, outputs, privacy, rjpeg, tiffout, tsdk, units

PSEUDO_COLOR_NAMES = [
    "whitehot", "fulgurite", "ironred", "hotiron", "medical",
    "arctic", "rainbow1", "rainbow2", "tint", "blackhot",
]


@dataclasses.dataclass
class MeasurementOverrides:
    """Any field left as None means: keep the R-JPEG's own embedded value for
    that field -- only fields the caller actually set on the CLI get overridden.

    humidity is a percent (0-100), not a 0-1 fraction. reflection/ambient_temp
    are Celsius.

    use_rangefinder: when distance is not set, take each file's laser-rangefinder
    target distance (DJI XMP) instead of the embedded distance, where the file has
    a valid reading."""

    distance: float | None = None
    humidity: float | None = None
    emissivity: float | None = None
    reflection: float | None = None
    ambient_temp: float | None = None
    use_rangefinder: bool = True

    def apply(self, params):
        for field in tsdk.MeasurementParams.FIELDS:
            value = getattr(self, field)
            if value is not None:
                setattr(params, field, value)
        return params


@dataclasses.dataclass
class Rangefinder:
    """A file's laser-rangefinder reading. distance (metres) is set only for a valid
    reading: status Normal and a positive distance."""

    status: str | None
    distance: float | None


def read_rangefinder(data: bytes) -> Rangefinder | None:
    """The LRFStatus / LRFTargetDistance in the file's DJI XMP, or None when it has neither."""
    f = djixmp.fields(data)
    if "LRFStatus" not in f and "LRFTargetDistance" not in f:
        return None
    status = f.get("LRFStatus")
    distance = djixmp.number(f, "LRFTargetDistance")
    if status is None or status.lower() != "normal" or distance is None or distance <= 0:
        distance = None
    return Rangefinder(status, distance)


@dataclasses.dataclass
class ProcessResult:
    src: Path
    ok: bool
    reason: str | None = None
    outputs: list[Path] = dataclasses.field(default_factory=list)
    distance: float | None = None
    distance_source: str | None = None      # "option", "rangefinder" or "embedded"
    notes: list[str] = dataclasses.field(default_factory=list)


def explain(e: tsdk.TsdkError) -> str:
    """A one-line reason for a skipped file or output."""
    if isinstance(e, tsdk.UnsupportedRJpeg):
        return f"unsupported R-JPEG: {e}"
    if isinstance(e, tsdk.RJpegError):
        return str(e)
    if isinstance(e, tsdk.UnsupportedFeature):
        return f"not supported: {e}"
    if isinstance(e, tsdk.IspError):
        return f"these color settings are refused or undefined in the DJI SDK too: {e}"
    if isinstance(e, tsdk.ParamError):
        return f"refused: {e}"
    return str(e)


def process_file(
    src: Path,
    dst_root: Path,
    src_root: Path,
    overrides: MeasurementOverrides,
    range_min_c: float | None,
    range_max_c: float | None,
    palette: int | None,
    wanted: tuple[tuple[str, str], ...] = outputs.DEFAULT_OUTPUTS,
    compression: str = "deflate",
    adj: adjust.Adjustments = adjust.Adjustments(),
    offset_c: float = 0.0,
    max_pitch: float | None = None,
    rjpeg_picture: str = "palette",
) -> ProcessResult:
    """Write each wanted (product, extension) output of one R-JPEG. A product that can't be
    made is noted and the others are still written; the file is skipped only when none can.
    adj: the gimbal override and out-of-range replacement (see adjust). offset_c: subtracted
    from the CLI's temperature products (normalization; before any replacement). max_pitch:
    skip the file when its gimbal pitch (after adj's override) is above it. rjpeg_picture: the
    rjpeg product's picture, "palette" (coloured as pseudo) or "original" (DJI's)."""
    data = src.read_bytes()
    if max_pitch is not None:
        pitch = adj.gimbal_pitch
        if pitch is None:
            pitch = djixmp.number(djixmp.fields(data), "GimbalPitchDegree")
        if pitch is not None and pitch > max_pitch:
            return ProcessResult(src, ok=False, reason=f"oblique (gimbal pitch {pitch:g} above {max_pitch:g})")
    rel = src.relative_to(src_root)
    out_dir = dst_root / rel.parent
    result = ProcessResult(src, ok=True)

    try:
        rj = _open_with_params(data, overrides, result)
    except tsdk.TsdkError as e:
        return ProcessResult(src, ok=False, reason=explain(e))

    replaced = adjust.Replaced()
    products = dict.fromkeys(p for p, _ in wanted if p != "rjpeg")
    asked_rjpeg = any(p == "rjpeg" for p, _ in wanted)
    if asked_rjpeg and rjpeg_picture == "palette":
        products["pseudo"] = None                   # the R-JPEG's picture (written only if asked for)
    made, failed = _make_products(rj, products, range_min_c, range_max_c, palette, result,
                                  adj, replaced, offset_c)
    if asked_rjpeg:
        picture = made["pseudo"][0] if rjpeg_picture == "palette" and "pseudo" in made else None
        try:
            made["rjpeg"] = (_rjpeg_copy(data, rj, adj, offset_c, picture), {})
        except (rjpeg.CopyError, tsdk.TsdkError) as e:
            failed["rjpeg"] = explain(e) if isinstance(e, tsdk.TsdkError) else str(e)
    made = {p: v for p, v in made.items() if p in dict(wanted)}
    failed = {p: v for p, v in failed.items() if p in dict(wanted)}
    if not made:
        return ProcessResult(src, ok=False, reason="; ".join(failed.values()) or "nothing to do")
    for product, reason in failed.items():
        result.notes.append(f"{product} skipped: {reason}")

    source = tiffout.read_metadata(data)
    if adj.strip_location:
        source = privacy.strip(source)
    xmp = djixmp.fields(data)
    original_gimbal = {a: djixmp.number(xmp, f"Gimbal{a.capitalize()}Degree") for a in adjust.GIMBAL_AXES}
    if adj.replaces() and (replaced.above or replaced.below):
        result.notes.append(f"replaced {replaced.above} pixel(s) above and {replaced.below} below the thresholds")
    metadata = tiffout.SourceMetadata(source.exif_tiff, adjust.record(
        source.xmp, params=_used_params(rj), distance_source=result.distance_source, adj=adj,
        replaced=replaced if adj.replaces() else None, original_gimbal=original_gimbal, offset_c=offset_c))
    extra_text = {k: str(v) for k, v in adjust.describe(adj).items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for product, ext in wanted:
        if product in made:
            pixels, text = made[product]
            path = out_dir / outputs.file_name(rel.stem, product, ext)
            outputs.write(path, product, ext, pixels, metadata, {**text, **extra_text}, compression)
            result.outputs.append(path)
    return result


def _rjpeg_copy(data: bytes, rj, adj: adjust.Adjustments, offset_c: float, picture: np.ndarray | None) -> bytes:
    """The source R-JPEG written again (see rjpeg): with the parameters rj measures with where they
    differ from its own, private with strip_location. Raises CopyError for what it can't carry."""
    if adj.replaces() or offset_c:
        raise rjpeg.CopyError("an R-JPEG can't carry normalized or replaced temperatures (temp_c can)")
    used = rj.get_measurement_params()
    try:
        own = tsdk.open(data).get_measurement_params()
    except tsdk.ParamError:
        own = None
    return rjpeg.copy(data, None if used == own else used, private=adj.strip_location, picture=picture)


def _open_with_params(data: bytes, overrides: MeasurementOverrides, result: ProcessResult):
    """The R-JPEG opened with the overrides applied (and the rangefinder distance, if asked
    for); result gets the distance and its source. Raises TsdkError."""
    rj = tsdk.open(data)
    params = overrides.apply(rj.get_measurement_params())
    result.distance_source = "option" if overrides.distance is not None else "embedded"
    if overrides.distance is None and overrides.use_rangefinder:
        _rangefinder_distance(rj, params, data, result)
    rj.set_measurement_params(params)
    result.distance = params.distance
    return rj


@dataclasses.dataclass
class Measured:
    """One file measured in memory (measure_file)."""

    temperature_c: np.ndarray           # float32 Celsius, adjustments applied; no data = -273.15
    params: dict                        # the measurement parameters the SDK used
    distance_source: str
    replaced: adjust.Replaced
    notes: list[str]


def measure_file(src: Path, overrides: MeasurementOverrides = MeasurementOverrides(),
                 adj: adjust.Adjustments = adjust.Adjustments(), offset_c: float = 0.0) -> Measured:
    """Measure one R-JPEG as process_file does, without writing anything. Raises TsdkError."""
    data = src.read_bytes()
    result = ProcessResult(src, ok=True)
    rj = _open_with_params(data, overrides, result)
    t = np.nan_to_num(rj.measure_temperature_c(), nan=tiffout.NO_DATA_C).astype(np.float32, copy=False)
    if offset_c:
        t = adjust.remove_offset(t, offset_c)
    replaced = adjust.Replaced()
    if adj.replaces():
        t, replaced = adjust.replace(t, adj)
    return Measured(t, _used_params(rj), result.distance_source, replaced, result.notes)


def _used_params(rj) -> dict:
    """The measurement parameters the SDK holds now, without those the camera doesn't have."""
    params = rj.get_measurement_params().as_dict()
    ranges = rj.get_measurement_params_range()
    return {k: v for k, v in params.items() if tuple(ranges.get(k, (1, 1))) != (0.0, 0.0)}


def _make_products(rj, products, range_min_c, range_max_c, palette, result,
                   adj: adjust.Adjustments = adjust.Adjustments(), replaced: adjust.Replaced | None = None,
                   offset_c: float = 0.0):
    """{product: (pixels, text metadata)} for the products that could be made, and
    {product: reason} for those that couldn't. The pseudo-colour image comes last: it sets
    the colour bar and palette. The CLI's temperature products carry adj's replacement
    (counted into replaced); temp_dc, raw and pseudo are the SDK's own."""
    made, failed, cache = {}, {}, {}

    def temp_c():
        if "t" not in cache:
            t = np.nan_to_num(rj.measure_temperature_c(), nan=tiffout.NO_DATA_C)
            if offset_c:
                t = adjust.remove_offset(t, offset_c)
            if adj.replaces():
                t, counts = adjust.replace(t, adj)
                if replaced is not None:
                    replaced.above, replaced.below = counts.above, counts.below
            cache["t"] = t
        return cache["t"]

    def make(product):
        if product == "temp_c":
            return temp_c(), {"units": "Celsius"}
        if product in ("temp_f", "temp_k"):
            unit = product[-1].upper()
            return units.float32_in(temp_c(), unit), {"units": units.UNITS[unit]}
        if product == "temp_ck":
            pixels, clipped = outputs.centikelvin(temp_c())
            if clipped:
                result.notes.append(f"temp_ck: {clipped} pixels above 382.2 C clipped to 65535")
            return pixels, {"units": "centikelvin"}
        if product == "temp_dc":
            return rj.measure(), {"units": "decicelsius"}
        if product == "raw":
            return rj.get_original_raw(), {"units": "raw sensor counts"}
        if product == "gray16":
            pixels, low, high = outputs.gray16(temp_c(), range_min_c, range_max_c)
            return pixels, {"range_min_c": f"{low:.4f}", "range_max_c": f"{high:.4f}"}
        if product == "pseudo":
            return _render(rj, range_min_c, range_max_c, palette), {}
        raise ValueError(product)

    for product in sorted(products, key=lambda p: p == "pseudo"):
        try:
            made[product] = make(product)
        except tsdk.TsdkError as e:
            failed[product] = explain(e)
    return made, failed


def _rangefinder_distance(rj, params, data: bytes, result: ProcessResult) -> None:
    """Use the file's laser-rangefinder distance, clamped to the camera's accepted range."""
    lrf = read_rangefinder(data)
    if lrf is None:
        return
    if lrf.distance is None:
        result.notes.append(f"rangefinder reading unusable (status {lrf.status or 'missing'}); "
                            f"kept the embedded distance {params.distance:.2f} m")
        return
    lo, hi = rj.get_measurement_params_range()["distance"]
    params.distance = min(max(lrf.distance, lo), hi)
    result.distance_source = "rangefinder"
    if params.distance != lrf.distance:
        result.notes.append(f"rangefinder distance {lrf.distance:.2f} m is outside this camera's "
                            f"{lo:g}-{hi:g} m range; used {params.distance:.2f} m")


def _render(rj, range_min_c, range_max_c, palette):
    """The pseudo-colour image, set up in the order the SDK's own sample does."""
    if range_min_c is not None and range_max_c is not None:
        rj.set_color_bar(tsdk.ColorBar(True, range_max_c, range_min_c))
    if palette is not None:
        rj.set_pseudo_color(palette)
    return rj.process()


def mosaic_frame(src: Path, src_root: Path, overrides: MeasurementOverrides, max_pitch: float,
                 adj: adjust.Adjustments = adjust.Adjustments()) -> dict:
    """One frame for the mosaic: {"frame": mosaic.Frame, "notes", "palette": (the file's
    palette index, the camera's colour tables (3, palettes, 256)) or None}, or {"reason"} with
    "not_thermal" or "oblique" set when that is why. Frames looking less steeply down than
    max_pitch (degrees) are left out unmeasured."""
    data = src.read_bytes()
    try:
        rj = tsdk.open(data)
    except tsdk.RJpegError as e:
        return {"reason": explain(e), "not_thermal": not isinstance(e, tsdk.UnsupportedRJpeg)}
    pose, why = mosaic.read_pose(data, (rj.width, rj.height))
    if pose is None:
        return {"reason": why}
    for axis, value in adj.gimbal().items():                 # the override decides placement too
        pose.record[f"original_gimbal_{axis}"] = getattr(pose, axis)
        setattr(pose, axis, value)
    if pose.pitch > max_pitch:
        return {"reason": f"oblique (gimbal pitch {pose.pitch:g} deg)", "oblique": True}
    result = ProcessResult(src, ok=True)
    try:
        params = overrides.apply(rj.get_measurement_params())
        result.distance_source = "option" if overrides.distance is not None else "embedded"
        if overrides.distance is None and overrides.use_rangefinder:
            _rangefinder_distance(rj, params, data, result)
        rj.set_measurement_params(params)
        temperature = rj.measure_temperature_c()
    except tsdk.TsdkError as e:
        return {"reason": explain(e)}
    pose.record.update({"measurement": params.as_dict(), "distance_source": result.distance_source})
    temperature = np.nan_to_num(temperature, nan=tiffout.NO_DATA_C)
    if adj.replaces():
        temperature, counts = adjust.replace(temperature, adj)
        pose.record.update({"replaced_above_pixels": counts.above, "replaced_below_pixels": counts.below})
    name = src.relative_to(src_root).as_posix()
    try:
        palette = rj.get_pseudo_color(), rj.get_pseudo_color_lut()
    except tsdk.TsdkError:
        palette = None
    return {"frame": mosaic.Frame(name, temperature, pose),
            "notes": result.notes, "palette": palette}


def embedded_ambient(src: Path) -> tuple[float, float] | None:
    """The R-JPEG's embedded (ambient, reflected) temperatures, or None if it isn't one."""
    try:
        params = tsdk.open(src.read_bytes()).get_measurement_params()
    except tsdk.TsdkError:
        return None
    return params.ambient_temp, params.reflection


def process_file_dict(*args) -> dict:
    """process_file with the result as plain data (for worker processes)."""
    return dataclasses.asdict(process_file(*args))
