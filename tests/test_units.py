"""Temperature unit conversions: exact to one float32 rounding."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from dji_thermal_cli import units

EXACT = {"C": lambda c: c, "F": lambda c: c * Fraction(9, 5) + 32, "K": lambda c: c + Fraction("273.15")}


def _nearest_float32(x: Fraction) -> np.float32:
    """The float32 nearest x (ties to even), by exact comparison with its neighbours."""
    f = np.float32(float(x))
    lo, hi = np.nextafter(f, np.float32(-np.inf)), np.nextafter(f, np.float32(np.inf))
    return min((lo, f, hi), key=lambda g: (abs(Fraction(float(g)) - x), int(g.view(np.uint32)) & 1))


@pytest.mark.parametrize("unit", list(units.UNITS))
def test_float32_in_is_correctly_rounded(unit):
    """Over a spread of SDK-like float32 Celsius values, every converted value is the float32
    nearest the exact conversion."""
    rng = np.random.default_rng(0)
    c = np.concatenate([rng.uniform(-40, 550, 3000), [-273.0, 0, 25, 100, 36.6]]).astype(np.float32)
    got = units.float32_in(c, unit)
    assert got.dtype == np.float32
    want = np.array([_nearest_float32(EXACT[unit](Fraction(float(v)))) for v in c], np.float32)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("unit, value, celsius", [("C", 20, 20.0), ("F", 68, 20.0), ("F", -40, -40.0),
                                                  ("K", 273.15, 0.0), ("K", 0, -273.15)])
def test_to_celsius(unit, value, celsius):
    assert units.to_celsius(value, unit) == pytest.approx(celsius, abs=1e-12)


@pytest.mark.parametrize("unit", list(units.UNITS))
def test_round_trip(unit):
    for c in (-273.15, -40.0, 0.0, 21.37, 500.0):
        assert units.to_celsius(units.from_celsius(c, unit), unit) == pytest.approx(c, abs=1e-12)


def test_nan_and_no_data():
    t = np.array([np.nan, -273.15], np.float32)
    assert np.isnan(units.float32_in(t, "K")[0]) and units.float32_in(t, "K")[1] == np.float32(0.0)
    assert units.float32_in(t, "F")[1] == np.float32(-459.67)
    assert units.float32_in(t, "C")[1] == np.float32(-273.15)
