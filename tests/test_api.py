"""The Python API: the CLI's functionality in-process, with identical results."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

import dji_thermal_cli as dtc
from dji_thermal_cli import checks, core, djixmp, tsdk

from .conftest import flight_counts, flight_dir, flight_files, flight_visual, samples_dir
from .test_cli import _cli


def _flight():
    return flight_files()


@pytest.mark.parametrize("kwargs, message", [
    ({"range_min": 10}, "range_min and range_max must be given together"),
    ({"range_min": 40, "range_max": 30}, "range_max must be greater than range_min"),
    ({"replace_above": 50}, "replace_above and replace_above_value must be given together"),
    ({"replace_above": 50, "replace_above_value": "warm"}, "must be a temperature or 'nodata'"),
    ({"replace_above": 20, "replace_above_value": 0, "replace_below": 30, "replace_below_value": 0},
     "replace_below must be lower than replace_above"),
    ({"unit": "K", "reflected_temp": -1}, "below absolute zero"),
    ({"unit": "R"}, "unit must be C, F or K"),
    ({"distance": "far"}, "distance must be"),
    ({"palette": "ironbow"}, "palette must be 0-9 or one of"),
])
def test_settings_validation(kwargs, message):
    with pytest.raises(dtc.SettingsError, match=message):
        dtc.Settings(**kwargs)


def test_settings_conversion():
    s = dtc.Settings(unit="f", reflected_temp=68, range_min=59, range_max=113, palette="hotiron",
                     replace_above=212, replace_above_value="nodata", distance=12)
    o = s.overrides()
    assert s.unit == "F" and s.palette == 3 and o.distance == 12.0 and not o.use_rangefinder
    assert o.reflection == pytest.approx(20.0) and s.range_c() == pytest.approx((15.0, 45.0))
    a = s.adjustments()
    assert a.replace_above == pytest.approx(100.0) and a.replace_above_value == dtc.api.adjust.NO_DATA
    assert dtc.Settings().overrides().use_rangefinder


@pytest.mark.flight
def test_measure_in_memory():
    src = _flight()[0]
    m = dtc.measure(src)
    rj = tsdk.open(src.read_bytes())
    if m.distance_source == "rangefinder":                  # the reading, clamped to the SDK's 1-25 m
        lrf = float(djixmp.fields(src.read_bytes())["LRFTargetDistance"])
        assert m.params["distance"] == pytest.approx(min(max(lrf, 1.0), 25.0), abs=0.05)
    assert m.temperature.dtype == np.float32 and m.temperature.shape == (rj.height, rj.width)
    k = dtc.measure(src, unit="K")
    np.testing.assert_array_equal(k.temperature, (m.temperature.astype(np.float64) + 273.15).astype(np.float32))
    threshold = float(np.median(m.temperature))
    adj = dtc.measure(src, dtc.Settings(replace_above=threshold, replace_above_value="nodata"))
    assert adj.replaced_above == int((m.temperature > np.float32(threshold)).sum()) > 0
    assert adj.temperature.min() == np.float32(-273.15)


@pytest.mark.flight
def test_process_same_as_cli(tmp_path):
    """The API writes byte-identical files to the command line's."""
    src = _flight()[0]
    events = []
    settings = dtc.Settings(unit="F", emissivity=0.95, replace_above=80, replace_above_value=80)
    report = dtc.process(src, tmp_path / "api", settings, outputs=["temp_c", "temp_f:exr"], jobs=1,
                         progress=events.append)
    assert report.processed == 1 and report.distance_sources() == {dtc.measure(src).distance_source: 1}
    assert [e.kind for e in events] == ["file"] and events[0].data["result"].ok
    proc = _cli("process", "--input-dir", src, "--output-dir", tmp_path / "cli", "--jobs", "1", "-o", "temp_c,temp_f:exr",
                "--input-unit", "F", "--emissivity", "0.95", "--replace-above", "80", "--replace-above-value", "80")
    assert proc.returncode == 0, proc.stderr
    for path in report.results[0].outputs:
        assert path.read_bytes() == (tmp_path / "cli" / path.name).read_bytes(), path.name


def test_process_reports_bad_files(tmp_path, capsys):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "bad.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really")
    (tmp_path / "in" / "notes.txt").write_text("hello")
    events = []
    report = dtc.process(tmp_path / "in", tmp_path / "out", jobs=1, progress=events.append)
    assert report.processed == 0 and not report.results[0].ok
    assert [e.kind for e in events] == ["non_jpeg", "file"]
    assert capsys.readouterr().out == ""                     # the API never prints


@pytest.mark.flight
def test_list_and_describe():
    src = _flight()[0]
    res = dtc.list_dataset(flight_dir(), unit="K", stats=False, jobs=2)
    assert len(res.rows) == len(_flight()) and res.not_thermal == len(flight_visual()) and not res.problems
    one = dtc.describe(src, unit="K", stats=False)
    assert one == next(r for r in res.rows if r["file"]["path"] == str(src))
    if flight_visual():
        with pytest.raises(dtc.FileProblem) as e:
            dtc.describe(flight_visual()[0])
        assert e.value.status.kind == "not_thermal"


@pytest.mark.flight
def test_check():
    """A flight with no errors; strict fails exactly when there are warnings."""
    res = dtc.check(flight_dir())
    assert not res.failed(), [f.message for f in res.findings if f.severity == checks.ERROR]
    assert res.failed(strict=True) == bool(res.summary[checks.WARNING])


@pytest.mark.flight
def test_make_mosaic(tmp_path):
    report = dtc.make_mosaic(flight_dir(), tmp_path, dtc.Settings(),
                             dtc.MosaicOptions(align=False, units=("c", "k")))
    assert (report.frames, report.oblique) == flight_counts()
    assert [p.name for p, _ in report.files] == ["mosaic_temp_c.tif", "mosaic_temp_k.tif"]
    k = tifffile.imread(report.files[1][0])[..., 0]
    ok = np.isfinite(report.mosaic.temperature)
    np.testing.assert_allclose(k[ok], report.mosaic.temperature[ok] + 273.15, atol=3e-5)


@pytest.mark.samples
def test_make_mosaic_without_pose(tmp_path):
    with pytest.raises(dtc.MosaicError, match="No frames to mosaic") as e:
        dtc.make_mosaic(samples_dir("M30T"), tmp_path, jobs=1)
    assert e.value.counts["frames"] == 0 and e.value.counts["unusable"] > 0


def test_missing_path(tmp_path):
    for call in (lambda: dtc.scan(tmp_path / "nope"), lambda: dtc.measure(tmp_path / "nope"),
                 lambda: dtc.process(tmp_path / "nope", tmp_path / "out")):
        with pytest.raises(FileNotFoundError):
            call()


@pytest.mark.flight
def test_core_measure_matches_process(tmp_path):
    src = _flight()[0]
    r = core.process_file(src, tmp_path, src.parent, core.MeasurementOverrides(use_rangefinder=True), None, None, None,
                          (("temp_c", "tiff"),))
    np.testing.assert_array_equal(tifffile.imread(r.outputs[0]), dtc.measure(src).temperature)
