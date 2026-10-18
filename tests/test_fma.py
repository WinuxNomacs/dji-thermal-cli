"""The fused multiply-add powf falls back on before Python 3.13 must round exactly like math.fma."""

import math

import numpy as np
import pytest

from dji_thermal_cli.tsdk import powf32

needs_math_fma = pytest.mark.skipif(not hasattr(math, "fma"), reason="math.fma is Python 3.13+")


@needs_math_fma
def test_fallback_matches_math_fma():
    rng = np.random.default_rng(13)
    a, b, c = (rng.standard_normal(20000) * 10.0 ** rng.integers(-30, 30, 20000) for _ in range(3))
    for x, y, z in zip(a.tolist(), b.tolist(), c.tolist()):
        assert powf32.fma_exact(x, y, z) == math.fma(x, y, z), (x, y, z)


@needs_math_fma
def test_fallback_matches_math_fma_where_rounding_twice_differs():
    # c cancels a*b almost exactly, so only the single rounding of a fused multiply-add gets the low bits
    rng = np.random.default_rng(7)
    rounded_twice_differs = 0
    for x, y in rng.uniform(0.5, 2.0, (20000, 2)).tolist():
        z = -(x * y)
        assert powf32.fma_exact(x, y, z) == math.fma(x, y, z), (x, y, z)
        rounded_twice_differs += math.fma(x, y, z) != x * y + z
    assert rounded_twice_differs > 10000


def test_powf_with_the_fallback_gives_the_same_results(monkeypatch):
    xs = np.random.default_rng(3).uniform(1e-30, 1e30, 3000).astype(np.float32).tolist()
    ys = (4.0, 0.25, 2.5, -3.0)
    expected = [powf32.powf(x, y) for x in xs for y in ys]
    monkeypatch.setattr(powf32, "_fma", powf32.fma_exact)
    assert [powf32.powf(x, y) for x in xs for y in ys] == expected
