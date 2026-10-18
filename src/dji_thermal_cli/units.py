"""Temperature units for the command line: Celsius, Fahrenheit and Kelvin.

The SDK works in degrees Celsius only: its parameters (reflected and ambient temperature,
colour-bar limits) are Celsius, and it measures float32 Celsius. The CLI converts what it is
given to Celsius before calling the SDK, and converts the SDK's Celsius results to the
requested unit afterwards. Every conversion is computed in double precision and rounded once,
so a float32 result is the float32 nearest the exact conversion of the SDK's value.
"""

from __future__ import annotations

import numpy as np

UNITS = {"C": "Celsius", "F": "Fahrenheit", "K": "Kelvin"}
SYMBOLS = {"C": "deg C", "F": "deg F", "K": "K"}
KELVIN = 273.15                         # 0 C in kelvin
ABSOLUTE_ZERO_C = -KELVIN


def to_celsius(value: float, unit: str) -> float:
    """A temperature in `unit` as Celsius (double precision)."""
    value = float(value)
    if unit == "C":
        return value
    if unit == "F":
        return (value - 32.0) * 5.0 / 9.0
    if unit == "K":
        return value - KELVIN
    raise ValueError(f"unknown temperature unit {unit!r}")


def from_celsius(celsius, unit: str):
    """Celsius (a number or an array) in `unit`, in double precision."""
    c = np.asarray(celsius, dtype=np.float64)
    if unit == "C":
        out = c
    elif unit == "F":
        out = c * 9.0 / 5.0 + 32.0
    elif unit == "K":
        out = c + KELVIN
    else:
        raise ValueError(f"unknown temperature unit {unit!r}")
    return float(out) if out.ndim == 0 else out


NO_DATA = {"C": ABSOLUTE_ZERO_C, "F": -459.67, "K": 0.0}     # absolute zero in each unit


def float32_in(celsius: np.ndarray, unit: str) -> np.ndarray:
    """A float32 Celsius array in `unit` as float32, rounded once from the exact conversion.
    NaN stays NaN, and the SDK's no-data floor, float32(-273.15) (really -273.149994 C),
    becomes absolute zero in `unit` exactly (0 K, -459.67 F) rather than 0.000006 K."""
    c = np.asarray(celsius, dtype=np.float32)
    if unit == "C":
        return c
    out = from_celsius(c, unit).astype(np.float32)
    out[c == np.float32(ABSOLUTE_ZERO_C)] = np.float32(NO_DATA[unit])
    return out
