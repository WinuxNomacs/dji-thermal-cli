"""The pure-Python port must be bit-identical to the native DJI Thermal SDK."""

from __future__ import annotations

import numpy as np
import pytest

from dji_thermal_cli import tsdk

from . import oracle_cases as cases
from .conftest import native_info, native_measure, native_sequence, sample_files

FILES = sample_files()
pytestmark = [pytest.mark.native, pytest.mark.samples]


def _open(path):
    try:
        return tsdk.open(path.read_bytes())
    except tsdk.UnsupportedRJpeg as e:
        pytest.skip(str(e))


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a.astype(np.float32).view(np.uint32), b.astype(np.float32).view(np.uint32))


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_embedded_params(path):
    rj = _open(path)
    native, native_params, code = native_measure(path)
    assert code == 0
    assert _same_bits(rj.measure_temperature_c(), native)
    ours = rj.get_measurement_params().as_dict()
    for name in ("distance", "humidity", "emissivity", "reflection"):
        assert ours[name] == pytest.approx(native_params[name], abs=1e-5), name


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_random_params(path):
    rj = _open(path)
    for params in cases.random_params(rj, path):
        native, _, code = native_measure(path, params)
        rj = _open(path)
        try:
            rj.set_measurement_params(tsdk.MeasurementParams(**params))
            ours = rj.measure_temperature_c()
        except tsdk.TsdkError:
            ours = None
        if code != 0:
            # native crashed (hirp: an out-of-bounds table read in libMicroTA) -> python must refuse too
            assert ours is None, f"native failed with {code} but python measured ({params})"
            continue
        assert ours is not None, f"python failed where native succeeded ({params})"
        assert _same_bits(ours, native), params


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_digest_params(path):
    """The parameter sets test_port_digest records results for, the file's ambient kept."""
    for params in cases.DIGEST_PARAMS[1:]:
        rj = _open(path)
        p = rj.get_measurement_params()
        p.distance, p.humidity, p.emissivity, p.reflection = params
        try:
            rj.set_measurement_params(p)
            ours = rj.measure_temperature_c()
        except tsdk.TsdkError:
            ours = None
        native, _, code = native_measure(path, p.as_dict())
        if code != 0:
            assert ours is None, f"native failed with {code} but python measured ({params})"
            continue
        assert ours is not None, f"python failed where native succeeded ({params})"
        assert _same_bits(ours, native), params


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_parameter_sequence(path, tmp_path):
    """Set/measure several times on one handle: state left by earlier (possibly failed) calls
    must carry over exactly as in the native SDK."""
    rj = _open(path)
    steps = cases.parameter_sequence(rj, path)
    native = native_sequence(path, steps, tmp_path)
    if native is None:
        pytest.skip("native SDK crashed on this sequence")
    for i, (step, (set_code, meas_code, t, params)) in enumerate(zip(steps, native)):
        ours_set = 0
        if step is not None:
            try:
                rj.set_measurement_params(tsdk.MeasurementParams(**step))
            except tsdk.TsdkError as e:
                ours_set = e.code
        assert ours_set == set_code, f"step {i} {step}: set native {set_code} python {ours_set}"
        try:
            ours = rj.measure_temperature_c()
            ours_meas = 0
        except tsdk.TsdkError as e:
            ours, ours_meas = None, e.code
        assert (ours_meas != 0) == (meas_code != 0), f"step {i} {step}: measure native {meas_code} python {ours_meas}"
        if meas_code == 0:
            assert _same_bits(ours, t.reshape(ours.shape)), f"step {i} {step}"
        if params.size:
            got = np.array([rj.get_measurement_params().as_dict()[k] for k in ("distance", "humidity", "emissivity", "reflection", "ambient_temp")], np.float32)
            assert np.array_equal(got[:4], params[:4]), f"step {i}: params {got} vs native {params}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_version_and_resolution(path):
    rj = _open(path)
    version, resolution = native_info(path)
    assert tuple(rj.rjpeg_version()) == version
    assert tuple(rj.resolution()) == resolution
