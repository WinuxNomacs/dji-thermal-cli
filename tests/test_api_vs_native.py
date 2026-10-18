"""The rest of the dirp API (dirp_measure, dirp_get_original_raw, dirp_get_api_version and the
ND-filter flag) must match the native SDK, including after parameter changes and refusals."""

from __future__ import annotations

import numpy as np
import pytest

from dji_thermal_cli import tsdk

from . import oracle_cases as cases
from .conftest import _flight_scan, native_process, sample_files

FILES = sample_files()
pytestmark = [pytest.mark.native, pytest.mark.samples]


_ours = cases.run


def _check(data: bytes, path, steps, tmp_path):
    try:
        rj = tsdk.open(data)
    except tsdk.UnsupportedRJpeg as e:
        pytest.skip(str(e))
    native = native_process(path, steps, tmp_path)
    if native is None:
        pytest.skip("native SDK crashed on this sequence")
    for i, (step, (ncode, ndata)) in enumerate(zip(steps, native)):
        code, ours = _ours(rj, step)
        assert (code != 0) == (ncode != 0), f"step {i} {step}: native {ncode} python {code}"
        if ncode == 0 and ndata is not None and step["op"] != "set_params":
            assert ours.shape == ndata.shape, f"step {i} {step}"
            assert np.array_equal(ours.view(np.uint8), ndata.view(np.uint8)), f"step {i} {step}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_api(path, tmp_path):
    data = path.read_bytes()
    rj = tsdk.open(data)
    for seed in cases.api_seeds(path):
        _check(data, path, cases.api_steps(rj, seed), tmp_path)


@pytest.mark.parametrize("path", cases.api_nd(FILES), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_api_nd_filter(path, tmp_path):
    data = cases.with_nd(path.read_bytes())
    nd_path = tmp_path / path.name
    nd_path.write_bytes(data)
    rj = tsdk.open(data)
    assert rj.infrared_density_filter_installed()
    _check(data, nd_path, cases.api_steps(rj, cases.api_seeds(path, "nd")[0]), tmp_path)


@pytest.mark.flight
@pytest.mark.parametrize("path", _flight_scan()[0], ids=lambda p: p.name)
def test_flight_frame(path, tmp_path):
    """Every frame of the flight: real scenes beyond the SDK's samples."""
    _check(path.read_bytes(), path, cases.FLIGHT, tmp_path)
