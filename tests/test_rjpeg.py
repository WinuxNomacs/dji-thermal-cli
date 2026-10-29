"""R-JPEG copies: DJI's tools measure a copy with parameters as they measure its source with the same
parameters set, a private copy measures as its source and carries no identifier, and every copy
keeps its source's length and layout."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from dji_thermal_cli import privacy, rjpeg, tsdk, xmpedit
from dji_thermal_cli.tsdk import jpeg

from .conftest import flight_files, native_measure, sample_files
from .test_privacy import _found, _needles

FILES = sample_files()
PARAMS = dict(distance=12.0, humidity=55.0, emissivity=0.92, reflection=18.5, ambient_temp=15.0)
ids = lambda p: f"{p.parent.name}/{p.name}"  # noqa: E731


def _open(data: bytes):
    try:
        return tsdk.open(data)
    except tsdk.UnsupportedRJpeg as e:
        pytest.skip(str(e))


def _measured_with(data: bytes, params: dict) -> np.ndarray:
    rj = _open(data)
    rj.set_measurement_params(tsdk.MeasurementParams(**params))
    return rj.measure_temperature_c()


def _layout_kept(source: bytes, copy: bytes) -> None:
    """Same length, the same segments at the same places, the MPF index and the picture untouched."""
    assert len(copy) == len(source)
    a, b = jpeg.segments(source), jpeg.segments(copy)
    assert [(s.marker, s.offset, len(s.payload)) for s in a] == [(s.marker, s.offset, len(s.payload)) for s in b]
    assert jpeg.concat(a, jpeg.APP2) == jpeg.concat(b, jpeg.APP2)
    scan = a[-1].offset + len(a[-1].payload)
    assert copy[scan:] == source[scan:]


@pytest.mark.samples
@pytest.mark.parametrize("path", FILES, ids=ids)
def test_a_copy_with_parameters_measures_as_its_source_with_them_set(path):
    data = path.read_bytes()
    rj = _open(data)
    if rj.plugin == "dirp":
        with pytest.raises(rjpeg.CopyError, match="cannot carry measurement parameters"):
            rjpeg.copy(data, tsdk.MeasurementParams(**PARAMS))
        return
    copy = rjpeg.copy(data, tsdk.MeasurementParams(**PARAMS))
    _layout_kept(data, copy)
    assert np.array_equal(_open(copy).measure_temperature_c(), _measured_with(data, PARAMS))
    read = _open(copy).get_measurement_params().as_dict()
    for name in ("distance", "humidity", "emissivity", "reflection"):
        assert read[name] == pytest.approx(PARAMS[name], abs=1e-4), name


@pytest.mark.samples
@pytest.mark.parametrize("path", FILES, ids=ids)
def test_a_private_copy_measures_as_its_source_and_carries_no_serial(path):
    data = path.read_bytes()
    found = rjpeg.identifiers(data)
    copy = rjpeg.copy(data, private=True)
    _layout_kept(data, copy)
    assert np.array_equal(_open(copy).measure_temperature_c(), _open(data).measure_temperature_c())
    assert [name for name, value in found.items() if value in copy] == []
    assert rjpeg.identifiers(copy) == {}
    with Image.open(io.BytesIO(copy)) as im:
        assert not im.getexif().get_ifd(0x8825) and im.getexif()[0x0110] == jpeg.exif_make_model(jpeg.segments(data))[1]


@pytest.mark.samples
def test_the_samples_serials_are_found():
    found = {}
    for path in FILES:
        found.setdefault(path.parent.name, set()).update(k.split()[0] for k in rjpeg.identifiers(path.read_bytes()))
    assert "sensor_id" in found.get("M4T", set()) and "module" in found.get("M3TD", set()), found


@pytest.mark.samples
def test_parameters_out_of_the_cameras_range_are_refused():
    path = next((p for p in FILES if _open(p.read_bytes()).plugin == "iirp"), None)
    if path is None:
        pytest.skip("no iirp sample")
    with pytest.raises(rjpeg.CopyError, match="emissivity"):
        rjpeg.copy(path.read_bytes(), tsdk.MeasurementParams(**dict(PARAMS, emissivity=0.01)))


@pytest.mark.native
@pytest.mark.samples
@pytest.mark.parametrize("path", FILES, ids=ids)
def test_dji_s_sdk_measures_a_copy_with_its_parameters(path, tmp_path):
    data = path.read_bytes()
    if _open(data).plugin == "dirp":
        pytest.skip(rjpeg.NO_PARAMS)
    copy = tmp_path / "copy.jpg"
    copy.write_bytes(rjpeg.copy(data, tsdk.MeasurementParams(**PARAMS), private=True))
    want, _, code = native_measure(path, PARAMS)
    got, params, copy_code = native_measure(copy)
    assert code == copy_code == 0
    assert np.array_equal(want, got)
    for name in ("distance", "humidity", "emissivity", "reflection"):
        assert params[name] == pytest.approx(PARAMS[name], abs=1e-4), name


@pytest.mark.native
@pytest.mark.samples
@pytest.mark.parametrize("path", FILES, ids=ids)
def test_dji_s_sdk_measures_a_private_copy_as_its_source(path, tmp_path):
    copy = tmp_path / "copy.jpg"
    copy.write_bytes(rjpeg.copy(path.read_bytes(), private=True))
    want, _, code = native_measure(path)
    got, _, copy_code = native_measure(copy)
    assert code == copy_code == 0
    assert np.array_equal(want, got)


@pytest.mark.flight
def test_a_private_copy_of_a_flight_frame_has_no_location(tmp_path):
    src = flight_files()[0]
    needles = _needles(src)
    copy = tmp_path / "copy.jpg"
    copy.write_bytes(rjpeg.copy(src.read_bytes(), private=True))
    assert _found(copy, needles) == []
    _layout_kept(src.read_bytes(), copy.read_bytes())


def test_an_xmp_packet_keeps_its_length():
    packet = b'<x:xmpmeta><rdf:Description dc:creator="Me" a:Kept="1"/></x:xmpmeta>   <?xpacket end="w"?>'
    out = rjpeg._padded(xmpedit.remove_properties(packet, privacy.XMP_REMOVED.match), len(packet))
    assert len(out) == len(packet) and b"Me" not in out and b'a:Kept="1"' in out
    assert out.endswith(b'<?xpacket end="w"?>')


def test_the_exif_free_text_and_the_sensor_serial_are_removed():
    exif = Image.Exif()
    exif[0x010F], exif[0x013B], exif[0x8298], exif[0x010E] = "DJI", "A. Person", "(c) A. Person", "Roof, 12 Main St"
    exif.get_ifd(0x8769)[0xA431] = "BODY12345"
    exif[0x927C] = b"[sensor_id:4WYAK1C02AA02J\0\0]"
    blob = exif.tobytes()[6:]
    out = privacy.strip_exif(blob)
    assert len(out) == len(blob)
    for gone in (b"A. Person", b"Main St", b"BODY12345", b"4WYAK1C02AA02J"):
        assert gone not in out, gone
    assert b"DJI" in out and b"[sensor_id:" in out
    assert privacy.exif_identifiers(blob) == {"tag 0xa431": b"BODY12345"}
