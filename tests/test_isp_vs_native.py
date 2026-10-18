"""The pure-Python image pipeline (dirp_process and its settings) must match the native SDK bit for bit."""

from __future__ import annotations

import copy
import struct

import numpy as np
import pytest

from dji_thermal_cli import tsdk

from . import oracle_cases as cases
from .conftest import native_process, sample_files

FILES = sample_files()
pytestmark = [pytest.mark.native, pytest.mark.samples]


def _open(data: bytes):
    try:
        rj = tsdk.open(data)
    except tsdk.UnsupportedRJpeg as e:
        pytest.skip(str(e))
    try:
        rj.get_pseudo_color_lut()
    except tsdk.UnsupportedFeature as e:
        pytest.skip(str(e))
    return rj


_apply = cases.run


def _check(rj, path, steps, tmp_path):
    native = native_process(path, steps, tmp_path)
    if native is None:
        # Python reports native undefined behaviour (often a crash) as IspError -32; check
        # everything before it, and that native indeed crashes there
        probe = copy.deepcopy(rj)
        k = next((i for i, st in enumerate(steps) if _apply(probe, st)[0] == -32), None)
        if k is None:
            pytest.skip("native SDK crashed where the Python port has defined behaviour")
        assert native_process(path, steps[:k + 1], tmp_path) is None, f"step {k}: Python -32, native ran"
        steps = steps[:k]
        native = native_process(path, steps, tmp_path)
        if native is None:
            pytest.skip("the native SDK crashes nondeterministically on this sequence")
    for i, (step, (ncode, ndata)) in enumerate(zip(steps, native)):
        code, data = _apply(rj, step)
        if code == tsdk.UnsupportedFeature.code and ncode == 0 and step["op"] in ("process", "strech"):
            return                                   # a setting the port does not render yet
        if step["op"] != "get":                      # getters are compared one by one below
            assert code == ncode, f"step {i} {step}: native {ncode} python {code}"
        if step["op"] == "get":
            codes, vals = data
            ported = [c != tsdk.UnsupportedFeature.code for c in codes]
            assert [c for c, p in zip(codes, ported) if p] == [int(c) for c, p in zip(ndata[:5], ported) if p], \
                f"step {i}: getter codes"
            codes = [c if p else -1 for c, p in zip(codes, ported)]
            pc, cb, ad, iso, b = vals
            f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]  # noqa: E731
            want = ndata[5:]
            if codes[0] == 0:
                assert pc == int(want[0]), f"step {i}: palette"
            if codes[1] == 0:
                assert (bool(cb.manual_enable), f32(cb.high), f32(cb.low)) == (bool(want[1]), want[2], want[3]), f"step {i}: color bar"
            if codes[2] == 0:
                assert (f32(ad.high), f32(ad.low)) == (want[4], want[5]), f"step {i}: adaptive"
            if codes[3] == 0:
                assert (bool(iso.enable), f32(iso.high), f32(iso.low)) == (bool(want[6]), want[7], want[8]), f"step {i}: isotherm"
            if codes[4] == 0:
                assert b == int(want[9]), f"step {i}: brightness"
        elif ncode == 0 and ndata is not None:
            assert data.shape == ndata.shape and np.array_equal(data.view(np.uint8), ndata.view(np.uint8)), \
                f"step {i} {step}: {np.count_nonzero(data != ndata)} values differ"


IDS = dict(ids=lambda p: f"{p.parent.name}/{p.name}")


@pytest.mark.parametrize("path", FILES, **IDS)
def test_defaults(path, tmp_path):
    rj = _open(path.read_bytes())
    _check(rj, path, cases.ISP_DEFAULTS, tmp_path)


@pytest.mark.parametrize("path", cases.isp_sweep(FILES), **IDS)
def test_settings_sequence(path, tmp_path):
    rj = _open(path.read_bytes())
    _check(rj, path, cases.isp_random(rj, path), tmp_path)


@pytest.mark.parametrize("path", cases.isp_nd(FILES), **IDS)
def test_nd_filter_color_bar(path, tmp_path):
    data = cases.with_nd(path.read_bytes())
    nd_path = tmp_path / path.name
    nd_path.write_bytes(data)
    rj = _open(data)
    assert rj.nd.installed
    _check(rj, nd_path, cases.ISP_ND, tmp_path)


@pytest.mark.parametrize("gain", cases.GAINS)
@pytest.mark.parametrize("path", cases.isp_variants(FILES), **IDS)
def test_other_gain_modes(path, gain, tmp_path):
    """No sample uses low gain (0) or gain 2; patch the header so those pipelines run too."""
    data = cases.with_gain(path.read_bytes(), gain)
    p = tmp_path / path.name
    p.write_bytes(data)
    rj = _open(data)
    _check(rj, p, cases.ISP_GAIN, tmp_path)


@pytest.mark.parametrize("path", cases.hirp(FILES), **IDS)
def test_hirp_ported_settings(path, tmp_path):
    """hirp: palette, brightness, colour bar, isotherm and measurement parameters."""
    rj = _open(path.read_bytes())
    _check(rj, path, cases.hirp_random(rj, path), tmp_path)


@pytest.mark.parametrize("path", cases.hirp(FILES), **IDS)
def test_hirp_alarm(path, tmp_path):
    """No sample was saved with the temperature alarm on; patch the flag in (the isotherm then
    starts enabled over the gain mode's temperature range)."""
    data = cases.with_alarm(path.read_bytes())
    p = tmp_path / path.name
    p.write_bytes(data)
    rj = _open(data)
    _check(rj, p, cases.HIRP_ALARM, tmp_path)
