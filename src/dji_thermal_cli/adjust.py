"""Dataset-wide adjustments applied on top of the SDK's measurement (the SDK is untouched).

- Gimbal override: one yaw, pitch and/or roll for every file, for when the recorded angle
  is known to be wrong. It places frames in the mosaic and replaces the drone-dji
  Gimbal*Degree values in the outputs' XMP (the originals are kept under dtc:).
- Out-of-range replacement, per pixel: every pixel above (or below) a threshold gets a given
  value, or becomes no data. It applies to the CLI's temperature outputs (temp_c, temp_f,
  temp_k, temp_ck, gray16) and the mosaic; temp_dc, raw and pseudo are the SDK's own and are
  left as they are.
- Location-free outputs (strip_location): see privacy.
- record(): every output's XMP gets a dtc: record of what was done: the measurement
  parameters actually used, the gimbal override and the replacement thresholds and counts.
"""

from __future__ import annotations

import dataclasses
import numpy as np

from . import tiffout, xmpedit
from ._version import __version__

NO_DATA = tiffout.NO_DATA_C             # a replacement value of "no data"
GIMBAL_AXES = ("yaw", "pitch", "roll")


@dataclasses.dataclass(frozen=True)
class Adjustments:
    """None = not set. Temperatures are Celsius; a replacement value of NO_DATA marks the
    pixels as no data."""

    gimbal_yaw: float | None = None
    gimbal_pitch: float | None = None
    gimbal_roll: float | None = None
    replace_above: float | None = None
    replace_above_value: float | None = None
    replace_below: float | None = None
    replace_below_value: float | None = None
    strip_location: bool = False                # outputs without location or identifiers (privacy)

    def gimbal(self) -> dict[str, float]:
        """The overridden axes, {"pitch": -90.0, ...}."""
        return {axis: v for axis in GIMBAL_AXES if (v := getattr(self, f"gimbal_{axis}")) is not None}

    def replaces(self) -> bool:
        return self.replace_above is not None or self.replace_below is not None


@dataclasses.dataclass
class Replaced:
    above: int = 0
    below: int = 0


def replace(temp_c: np.ndarray, adj: Adjustments) -> tuple[np.ndarray, Replaced]:
    """A copy of the float32 Celsius frame with out-of-range pixels replaced, and the counts.
    Pixels that are already no data (the SDK's floor) are left alone."""
    out = np.array(temp_c, dtype=np.float32, copy=True)
    counts = Replaced()
    valid = out > np.float32(NO_DATA)
    if adj.replace_above is not None:
        hit = valid & (out > np.float32(adj.replace_above))
        counts.above = int(hit.sum())
        out[hit] = np.float32(adj.replace_above_value)
    if adj.replace_below is not None:
        hit = valid & (temp_c < np.float32(adj.replace_below))          # judged on the measured values
        counts.below = int(hit.sum())
        out[hit] = np.float32(adj.replace_below_value)
    return out, counts


def remove_offset(temp_c: np.ndarray, offset_c: float) -> np.ndarray:
    """The frame minus its normalization offset (computed in double, rounded once to float32);
    no-data pixels stay no data."""
    t = np.asarray(temp_c, dtype=np.float32)
    out = (t.astype(np.float64) - offset_c).astype(np.float32)
    return np.where(t > np.float32(NO_DATA), out, t)


def _num(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _value(v: float) -> str:
    return "nodata" if v == NO_DATA else _num(v)


def record(xmp: bytes | None, *, params: dict, distance_source: str | None, adj: Adjustments,
           replaced: Replaced | None, original_gimbal: dict[str, float | None], offset_c: float = 0.0) -> bytes:
    """The output XMP: the gimbal override applied to drone-dji, and the dtc: record."""
    gimbal = adj.gimbal()
    if gimbal:
        xmp = xmpedit.set_properties(xmp, xmpedit.DRONE_DJI,
                                     {f"Gimbal{a.capitalize()}Degree": f"{v:+.2f}" for a, v in gimbal.items()})
    props = {"Software": f"dji-thermal-cli {__version__}"}
    names = {"distance": "Distance", "humidity": "Humidity", "emissivity": "Emissivity",
             "reflection": "ReflectedTemperatureC", "ambient_temp": "AmbientTemperatureC"}
    for key, name in names.items():
        if params.get(key) is not None:
            props[name] = _num(float(params[key]))
    if distance_source:
        props["DistanceSource"] = distance_source
    for axis, v in gimbal.items():
        props[f"Gimbal{axis.capitalize()}Override"] = f"{v:+.2f}"
        original = original_gimbal.get(axis)
        props[f"OriginalGimbal{axis.capitalize()}Degree"] = "" if original is None else f"{original:+.2f}"
    if offset_c:
        props["NormalizationOffsetC"] = _num(offset_c)
    if adj.strip_location:
        props["LocationRemoved"] = "True"
    if adj.replace_above is not None:
        props["ReplaceAboveC"] = _num(adj.replace_above)
        props["ReplaceAboveValueC"] = _value(adj.replace_above_value)
        props["ReplacedAbovePixels"] = str(replaced.above if replaced else 0)
    if adj.replace_below is not None:
        props["ReplaceBelowC"] = _num(adj.replace_below)
        props["ReplaceBelowValueC"] = _value(adj.replace_below_value)
        props["ReplacedBelowPixels"] = str(replaced.below if replaced else 0)
    return xmpedit.set_properties(xmp, xmpedit.DTC, props)


def describe(adj: Adjustments) -> dict:
    """The adjustments as plain data, for reports."""
    out = {f"gimbal_{a}": v for a, v in adj.gimbal().items()}
    if adj.replace_above is not None:
        out["replace_above_c"] = adj.replace_above
        out["replace_above_value_c"] = _value(adj.replace_above_value)
    if adj.replace_below is not None:
        out["replace_below_c"] = adj.replace_below
        out["replace_below_value_c"] = _value(adj.replace_below_value)
    if adj.strip_location:
        out["location_removed"] = True
    return out
