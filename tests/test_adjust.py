"""Dataset-wide adjustments: gimbal override and per-pixel out-of-range replacement."""

from __future__ import annotations

import re

import numpy as np
import pytest
import tifffile
from PIL import Image

from dji_thermal_cli import adjust, core, djixmp, mosaic, tsdk, xmpedit

from .conftest import flight_files
from .test_cli import _cli

def _flight_files():
    return flight_files()


def test_replace_counts_and_no_data():
    t = np.array([[-273.15, 10.0, 20.0], [30.0, 40.0, 50.0]], np.float32)
    adj = adjust.Adjustments(replace_above=35.0, replace_above_value=adjust.NO_DATA,
                             replace_below=15.0, replace_below_value=15.0)
    out, counts = adjust.replace(t, adj)
    assert (counts.above, counts.below) == (2, 1)
    np.testing.assert_array_equal(out, np.array([[-273.15, 15.0, 20.0], [30.0, -273.15, -273.15]], np.float32))
    np.testing.assert_array_equal(t[1], [30.0, 40.0, 50.0])             # the input is untouched


def test_replace_below_judged_on_measured_values():
    """A pixel replaced 'above' with a low value is not then caught by 'below'."""
    t = np.array([100.0, 20.0], np.float32)
    adj = adjust.Adjustments(replace_above=50.0, replace_above_value=0.0, replace_below=10.0, replace_below_value=99.0)
    out, counts = adjust.replace(t, adj)
    np.testing.assert_array_equal(out, [0.0, 20.0])
    assert (counts.above, counts.below) == (1, 0)


def test_xmp_edits():
    attr = b'<x:xmpmeta><rdf:RDF><rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" drone-dji:GimbalPitchDegree="-45.00" drone-dji:GimbalYawDegree="+3.0"/></rdf:RDF></x:xmpmeta>'
    out = xmpedit.set_properties(attr, xmpedit.DRONE_DJI, {"GimbalPitchDegree": "-90.00"}).decode()
    assert 'drone-dji:GimbalPitchDegree="-90.00"' in out and 'drone-dji:GimbalYawDegree="+3.0"' in out
    elem = b'<rdf:Description><drone-dji:GimbalPitchDegree>-45.00</drone-dji:GimbalPitchDegree></rdf:Description>'
    assert b"<drone-dji:GimbalPitchDegree>-90.00</drone-dji:GimbalPitchDegree>" in \
        xmpedit.set_properties(elem, xmpedit.DRONE_DJI, {"GimbalPitchDegree": "-90.00"})
    out = xmpedit.set_properties(attr, xmpedit.DTC, {"Note": 'a "quoted" <value>'}).decode()
    assert 'xmlns:dtc="urn:dji-thermal-cli:xmp:1.0"' in out and "dtc:Note=" in out and "&lt;value&gt;" in out
    fresh = xmpedit.set_properties(None, xmpedit.DTC, {"Software": "x"}).decode()
    assert 'dtc:Software="x"' in fresh and fresh.startswith("<?xpacket")


@pytest.mark.flight
def test_outputs(tmp_path):
    """temp_c carries the replacement; temp_dc and pseudo stay the SDK's own; the XMP records it."""
    src = _flight_files()[0]
    rj = tsdk.open(src.read_bytes())
    t = rj.measure_temperature_c()
    hi, lo = (round(float(np.percentile(t, q)), 1) for q in (75, 25))     # this frame's own range
    adj = adjust.Adjustments(gimbal_pitch=-88.0, replace_above=hi, replace_above_value=adjust.NO_DATA,
                             replace_below=lo, replace_below_value=lo)
    plain = core.process_file(src, tmp_path / "plain", src.parent, core.MeasurementOverrides(), None, None, None,
                              (("temp_c", "tiff"), ("temp_dc", "tiff"), ("pseudo", "png")))
    r = core.process_file(src, tmp_path / "adj", src.parent, core.MeasurementOverrides(), None, None, None,
                          (("temp_c", "tiff"), ("temp_dc", "tiff"), ("pseudo", "png")), adj=adj)
    assert r.ok and any("replaced" in n for n in r.notes)
    base, got = tifffile.imread(plain.outputs[0]), tifffile.imread(r.outputs[0])
    want = np.where(base > np.float32(hi), np.float32(-273.15), np.where(base < np.float32(lo), np.float32(lo), base))
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(tifffile.imread(r.outputs[1]), tifffile.imread(plain.outputs[1]))
    np.testing.assert_array_equal(np.asarray(Image.open(r.outputs[2])), np.asarray(Image.open(plain.outputs[2])))
    with tifffile.TiffFile(r.outputs[0]) as tf:
        xmp = tf.pages[0].tags["XMP"].value
    text = xmp.decode()
    f = dict(re.findall(r'drone-dji:(\w+)="([^"]*)"', text))
    assert f["GimbalPitchDegree"] == "-88.00" and f["GimbalYawDegree"] == djixmp.fields(src.read_bytes())["GimbalYawDegree"]
    pitch = djixmp.number(djixmp.fields(src.read_bytes()), "GimbalPitchDegree")
    assert f'dtc:OriginalGimbalPitchDegree="{pitch:+.2f}"' in text and 'dtc:ReplaceAboveValueC="nodata"' in text
    above = int(re.search(r'dtc:ReplacedAbovePixels="(\d+)"', text).group(1))
    assert above == int((base > np.float32(hi)).sum()) > 0
    assert "dtc:DistanceSource=" in text and "dtc:Emissivity=" in text
    assert rj.width == got.shape[1]


@pytest.mark.flight
def test_record_without_adjustments(tmp_path):
    """Every output records the parameters used, even with nothing overridden."""
    src = _flight_files()[0]
    r = core.process_file(src, tmp_path, src.parent, core.MeasurementOverrides(emissivity=0.97), None, None, None,
                          (("temp_c", "tiff"),))
    with tifffile.TiffFile(r.outputs[0]) as tf:
        text = tf.pages[0].tags["XMP"].value.decode()
    assert 'dtc:Emissivity="0.97"' in text and "Override" not in text and "Replace" not in text
    assert 'Camera:BandName="LWIR"' in text


@pytest.mark.flight
def test_mosaic_frame_gimbal_override():
    """Forcing pitch -90 brings an oblique frame into the mosaic; the original is recorded."""
    oblique = next((p for p in _flight_files() if (djixmp.number(djixmp.fields(p.read_bytes()), "GimbalPitchDegree")
                                                     or -90) > -80), None)
    if oblique is None:
        pytest.skip("the flight has no oblique frame")
    assert core.mosaic_frame(oblique, oblique.parent, core.MeasurementOverrides(), -80.0).get("oblique")
    got = core.mosaic_frame(oblique, oblique.parent, core.MeasurementOverrides(), -80.0,
                            adjust.Adjustments(gimbal_pitch=-90.0))
    pose = got["frame"].pose
    assert pose.pitch == -90.0 and pose.record["original_gimbal_pitch"] > -80


def test_mosaic_no_data_falls_back_to_other_frames():
    """A no-data pixel in the best frame leaves the spot to the next-best frame."""
    from .test_mosaic import _frame
    frames = [_frame(f"f{i}", 2.0 * i, 0.0, yaw=0.0) for i in range(2)]
    holed = mosaic.Frame(frames[0].name, frames[0].temperature.copy(), frames[0].pose)
    holed.temperature[:, :] = np.float32(adjust.NO_DATA)
    m = mosaic.build([holed, frames[1]], align=False, check_misalignment=False)
    covered = np.isfinite(m.source)
    assert covered.any() and (m.source[covered] == 1).all()
    assert (m.temperature[covered] > -273).all()


@pytest.mark.flight
def test_cli_options(tmp_path):
    src = _flight_files()[0]
    proc = _cli("process", "--input-dir", src, "--output-dir", tmp_path, "--replace-above", "100")
    assert proc.returncode != 0 and "must be given together" in proc.stderr
    proc = _cli("process", "--input-dir", src, "--output-dir", tmp_path, "--replace-above", "20",
                "--replace-above-value", "nodata", "--replace-below", "30", "--replace-below-value", "30")
    assert proc.returncode != 0 and "--replace-below must be lower" in proc.stderr
    proc = _cli("process", "--input-dir", src, "--output-dir", tmp_path, "--replace-above-value", "warm")
    assert proc.returncode != 0 and "must be a temperature or 'nodata'" in proc.stderr
    proc = _cli("process", "--input-dir", src, "--output-dir", tmp_path / "f", "--jobs", "1", "-o", "temp_f",
                "--input-unit", "F", "--replace-above", "77", "--replace-above-value", "77")
    assert proc.returncode == 0, proc.stderr
    assert tifffile.imread(next((tmp_path / "f").glob("*_temp_f.tiff"))).max() == pytest.approx(77.0, abs=1e-4)
