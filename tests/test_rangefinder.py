"""Object distance from the DJI XMP laser-rangefinder fields: parsing, precedence over the
embedded distance, --distance lrf|embedded|N, and clamping to the camera's range."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from PIL import Image

from dji_thermal_cli import core, tsdk

from .conftest import sample
from .test_cli import _cli

_XMP_PREFIX = b"http://ns.adobe.com/xap/1.0/\x00"


def _xmp(body: str) -> bytes:
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" ' + body + '</rdf:Description>'
        '</rdf:RDF></x:xmpmeta>'
    ).encode()


def _with_xmp(data: bytes, xmp: bytes) -> bytes:
    """The JPEG with an XMP APP1 segment inserted right after SOI."""
    payload = _XMP_PREFIX + xmp
    return data[:2] + b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + data[2:]


def _lrf(status: str, distance: str) -> bytes:
    return _xmp(f'drone-dji:LRFStatus="{status}" drone-dji:LRFTargetDistance="{distance}">')


JPEG = b"\xff\xd8\xff\xd9"


@pytest.mark.parametrize("xmp, expected", [
    (_lrf("Normal", "21.400"), core.Rangefinder("Normal", 21.4)),
    (_xmp("><drone-dji:LRFStatus>Normal</drone-dji:LRFStatus>"
          "<drone-dji:LRFTargetDistance>8.9</drone-dji:LRFTargetDistance>"), core.Rangefinder("Normal", 8.9)),
    (_lrf("TooFar", "0.000"), core.Rangefinder("TooFar", None)),
    (_lrf("Normal", "0"), core.Rangefinder("Normal", None)),
    (_lrf("Normal", "nan"), core.Rangefinder("Normal", None)),
    (_lrf("Normal", "x"), core.Rangefinder("Normal", None)),
    (_xmp('drone-dji:LRFTargetDistance="12">'), core.Rangefinder(None, None)),
    (_xmp('drone-dji:RelativeAltitude="+21.1">'), None),
])
def test_read_rangefinder(xmp, expected):
    assert core.read_rangefinder(_with_xmp(JPEG, xmp)) == expected


def test_no_xmp():
    assert core.read_rangefinder(JPEG) is None


def _sample(camera: str):
    return sample(camera).read_bytes()


def _temps_at(data: bytes, distance: float) -> np.ndarray:
    rj = tsdk.open(data)
    params = rj.get_measurement_params()
    params.distance = distance
    rj.set_measurement_params(params)
    return rj.measure_temperature_c()


def _process(tmp_path, data, overrides):
    src = tmp_path / "src" / "DJI_0001_T.JPG"
    src.parent.mkdir(exist_ok=True)
    src.write_bytes(data)
    r = core.process_file(src, tmp_path / "out", src.parent, overrides, None, None, None, (("temp_c", "tiff"),))
    assert r.ok, r.reason
    return r, np.asarray(Image.open(r.outputs[0]))


@pytest.mark.parametrize("overrides, distance, source", [
    (core.MeasurementOverrides(), 21.4, "rangefinder"),
    (core.MeasurementOverrides(distance=7.0), 7.0, "option"),
    (core.MeasurementOverrides(use_rangefinder=False), None, "embedded"),
])
@pytest.mark.samples
def test_precedence(overrides, distance, source, tmp_path):
    data = _sample("M4T")
    embedded = tsdk.open(data).get_measurement_params().distance
    r, temps = _process(tmp_path, _with_xmp(data, _lrf("Normal", "21.4")), overrides)
    expected = embedded if distance is None else distance
    assert (r.distance, r.distance_source, r.notes) == (pytest.approx(expected), source, [])
    assert np.array_equal(temps, _temps_at(data, expected))


@pytest.mark.samples
def test_unusable_reading_keeps_embedded(tmp_path):
    data = _sample("M4T")
    r, _ = _process(tmp_path, _with_xmp(data, _lrf("TooFar", "0")), core.MeasurementOverrides())
    assert r.distance_source == "embedded"
    assert r.notes and "status TooFar" in r.notes[0]


@pytest.mark.parametrize("reading, used", [("40", 25.0), ("0.5", 1.0)])
@pytest.mark.samples
def test_clamped_to_camera_range(reading, used, tmp_path):
    data = _sample("M30T")
    assert tsdk.open(data).get_measurement_params_range()["distance"] == (1.0, 25.0)
    r, temps = _process(tmp_path, _with_xmp(data, _lrf("Normal", reading)), core.MeasurementOverrides())
    assert (r.distance, r.distance_source) == (used, "rangefinder")
    assert "outside this camera's 1-25 m range" in r.notes[0]
    assert np.array_equal(temps, _temps_at(data, used))


@pytest.mark.samples
def test_cli(tmp_path):
    src = tmp_path / "in" / "DJI_0001_T.JPG"
    src.parent.mkdir()
    src.write_bytes(_with_xmp(_sample("M4T"), _lrf("Normal", "21.4")))
    proc = _cli("process", "--input-dir", src.parent, "--output-dir", tmp_path / "out", "--jobs", "1")
    assert "Object distance source: 1 laser rangefinder" in proc.stdout, proc.stderr
    proc = _cli("process", "--input-dir", src.parent, "--output-dir", tmp_path / "out", "--jobs", "1", "--distance", "embedded")
    assert "Object distance source: 1 embedded" in proc.stdout, proc.stderr
    proc = _cli("process", "--input-dir", src.parent, "--output-dir", tmp_path / "out", "--jobs", "1", "--distance", "7")
    assert "Object distance source: 1 --distance" in proc.stdout, proc.stderr
    proc = _cli("process", "--input-dir", src.parent, "--output-dir", tmp_path / "out", "--distance", "far")
    assert proc.returncode != 0 and "must be lrf, embedded or a distance in metres" in proc.stderr
    proc = _cli("list", "--input-dir", src.parent)
    assert "21.40" in proc.stdout, proc.stderr
