"""`check`: findings from synthetic datasets, and the command on real and damaged files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dji_thermal_cli import checks, dataset

from .conftest import flight_counts, flight_dir, sample
from .test_cli import _cli


def _report(name, *, pitch=-90.0, tmax=30.0, tmin=15.0, tmean=22.0, emissivity=0.95, humidity=50.0,
            distance=5.0, ambient=25.0, reflection=25.0, alt=21.0, lrf=21.0, lrf_status="Normal",
            gps="RTK", rtk=0.01, model="M4T", lat=42.0, sha1=None, warnings=()):
    info = {
        "file": {"warnings": list(warnings)},
        "camera": {"model": model, "width": 640, "height": 512},
        "thermal": {"plugin": "iirp", "measurement_params": {
            "emissivity": emissivity, "humidity": humidity, "distance": distance,
            "ambient_temp": ambient, "reflection": reflection}},
        "temperature": {"min": tmin, "max": tmax, "mean": tmean},
        "geospatial": {"latitude": lat, "relative_altitude_m": alt, "gps_status": gps,
                       "rtk_std_m": {"lat": rtk, "lon": rtk, "hgt": rtk},
                       "rangefinder": {"status": lrf_status, "distance_m": lrf}},
        "attitude": {"gimbal_pitch": pitch},
    }
    status = dataset.FileStatus(Path(name), dataset.THERMAL, "", "iirp", list(warnings))
    return checks.FileReport(name, status, info, sha1 or name)


def _flight(n=20, **overrides):
    return [_report(f"f{i:02d}", **{k: (v(i) if callable(v) else v) for k, v in overrides.items()}) for i in range(n)]


def _codes(findings):
    return {f.code: f for f in findings}


def test_clean_dataset():
    assert checks.analyze(_flight(distance=21.0), []) == []


def test_parameters_that_differ():
    reports = _flight(emissivity=lambda i: 0.98 if i == 3 else 0.95, humidity=lambda i: 60.0 if i < 2 else 50.0)
    c = _codes(checks.analyze(reports, []))
    assert c["emissivity_differs"].files == ["f03"] and "--emissivity" in c["emissivity_differs"].suggestion
    assert c["humidity_differs"].files == ["f00", "f01"]


def test_ambient_drift_merged_when_identical():
    reports = _flight(ambient=lambda i: 25.0 + i * 0.1, reflection=lambda i: 25.0 + i * 0.1)
    c = _codes(checks.analyze(reports, []))
    assert "ambient and reflected" in c["ambient_temp_varies"].message and "reflection_varies" not in c
    assert "--mosaic-normalize" in c["ambient_temp_varies"].suggestion
    c = _codes(checks.analyze(_flight(ambient=lambda i: 25.0 + i * 0.1), []))
    assert "ambient_temp_varies" in c and "reflection_varies" not in c


def test_temperature_outliers_and_obliques():
    reports = _flight(tmax=lambda i: 80.0 if i == 7 else 30.0 + i * 0.1,
                      tmin=lambda i: -10.0 if i in (1, 2) else 15.0 + i * 0.1,
                      pitch=lambda i: -45.0 if i in (1, 2) else -90.0)
    c = _codes(checks.analyze(reports, []))
    assert c["temperature_max_outlier"].files == ["f07"] and "oblique" not in c["temperature_max_outlier"].message
    assert sorted(c["temperature_min_outlier"].files) == ["f01", "f02"]
    assert "2 of them are oblique frames (sky in view?)" in c["temperature_min_outlier"].message
    assert c["oblique"].severity == checks.INFO and sorted(c["oblique"].files) == ["f01", "f02"]


def test_small_differences_are_not_outliers():
    """A perfectly uniform dataset has MAD 0: a value 1 C off must not count as an outlier."""
    reports = _flight(tmax=lambda i: 31.0 if i == 4 else 30.0)
    assert "temperature_max_outlier" not in _codes(checks.analyze(reports, []))


def test_pose_findings():
    reports = _flight(pitch=lambda i: -70.0 if i == 5 else -90.0, alt=lambda i: 5.0 if i == 9 else 21.0,
                      gps=lambda i: "GPS" if i == 0 else "RTK", rtk=lambda i: 0.3 if i == 11 else 0.01,
                      lat=lambda i: None if i == 13 else 42.0)
    c = _codes(checks.analyze(reports, [], max_pitch=-60.0))
    assert c["pitch_outlier"].files == ["f05"]
    assert c["altitude_outlier"].files == ["f09"]
    assert c["gps_status_differs"].files == ["f00"]
    assert c["rtk_accuracy"].files == ["f11"]
    assert c["no_pose"].files == ["f13"]


def test_rangefinder_findings():
    reports = _flight(lrf=lambda i: 40.0 if i == 6 else 21.0 + i * 0.01,
                      lrf_status=lambda i: "TooFar" if i == 8 else "Normal")
    c = _codes(checks.analyze(reports, []))
    assert c["rangefinder_invalid"].files == ["f08"]
    assert c["rangefinder_outlier"].files == ["f06"]
    assert "rangefinder reads 21.1 m" in c["rangefinder_distance"].message


def test_file_errors():
    reports = _flight(5)
    reports.append(_report("dup", sha1="f00"))
    reports.append(checks.FileReport("bad", dataset.FileStatus(Path("bad"), dataset.CORRUPT, "file cut short"), None, None))
    reports.append(_report("w", warnings=["preview image data damaged"]))
    findings = checks.analyze(reports, [Path("notes.JPG"), Path("clip.MP4")])
    c = _codes(findings)
    assert findings[0].severity == checks.ERROR
    assert sorted(c["duplicate"].files) == ["dup", "f00"]
    assert c["corrupt"].files == ["bad"] and c["not_jpeg"].files == ["notes.JPG"]
    assert c["damaged_preview"].severity == checks.WARNING
    assert "1 other files" in c["other_files"].message


def test_mixed_cameras():
    reports = _flight(model=lambda i: "M30T" if i < 3 else "M4T")
    assert _codes(checks.analyze(reports, []))["mixed_cameras"].files == ["f00", "f01", "f02"]


@pytest.mark.samples
def test_command_on_damaged_dataset(tmp_path):
    data = sample("M4T").read_bytes()
    (tmp_path / "a.JPG").write_bytes(data)
    (tmp_path / "copy.JPG").write_bytes(data)
    (tmp_path / "cut.JPG").write_bytes(data[:300_000])
    (tmp_path / "readme.jpg").write_text("not an image")
    proc = _cli("check", "--input-dir", tmp_path, "--format", "json")
    assert proc.returncode == 1
    out = json.loads(proc.stdout)
    codes = {f["code"] for f in out["findings"]}
    assert {"duplicate", "corrupt", "not_jpeg"} <= codes
    assert out["summary"]["files"] == {"thermal": 2, "corrupt": 1}


@pytest.mark.flight
def test_command_on_flight():
    flight = flight_dir()
    proc = _cli("check", "--input-dir", flight)
    assert proc.returncode == 0, proc.stdout
    assert "0 error(s)" in proc.stdout
    oblique = flight_counts()[1]
    assert (f"{oblique} frame(s) look less steeply down" in proc.stdout) == (oblique > 0)
    warnings = json.loads(_cli("check", "--input-dir", flight, "--format", "json").stdout)["summary"]["warning"]
    strict = _cli("check", "--input-dir", flight, "--strict")
    assert strict.returncode == (1 if warnings else 0)
