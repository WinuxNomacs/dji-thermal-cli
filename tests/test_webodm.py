"""Phase 6: per-frame normalization, frame selection, and check --webodm."""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest
import tifffile

import dji_thermal_cli as dtc
from dji_thermal_cli import webodm

from .conftest import flight_counts, flight_dir, flight_files
from .test_cli import _cli
from .test_mosaic import _frame

def _flight():
    return flight_dir()


def test_inside_either_winding_and_repeated_vertices():
    square = np.array([[0, 0], [1, 0], [1, 0], [1, 1], [0, 1]], float)      # a repeated corner
    x, y = np.array([0.5, 1.5]), np.array([0.5, 0.5])
    np.testing.assert_array_equal(webodm._inside(square, x, y), [True, False])
    np.testing.assert_array_equal(webodm._inside(square[::-1], x, y), [True, False])


def test_overlap_formula():
    assert webodm._overlap(1) == 0.0
    assert webodm._overlap(16) == pytest.approx(75.0)
    assert webodm._overlap(4) == pytest.approx(50.0)


def test_coverage_of_stacked_frames():
    """Two frames at the same spot: every covered point is seen twice."""
    a, b = _frame("a", 0.0, 0.0, yaw=0.0), _frame("b", 0.0, 0.0, yaw=0.0)
    c = webodm._coverage([a.pose, b.pose])
    assert c["images_per_point_median"] == 2 and c["images_per_point_p10"] == 2 and c["frames"] == 2
    lone = webodm._coverage([a.pose])
    assert lone["images_per_point_median"] == 1 and lone["equivalent_overlap_pct"] == 0.0


@pytest.mark.flight
def test_check_webodm_plan():
    res = dtc.check(_flight(), webodm=True)
    plan = res.webodm
    nadir, oblique = flight_counts()
    args = plan.process_args
    assert args[:2] == ["-o", "temp_c:tiff"]
    assert ("--max-pitch" in args) == (oblique > 0) and ("--normalize" in args) == (nadir >= 2)
    assert plan.recommended["radiometric-calibration"].value == "none"
    gsd_cm = plan.coverage["gsd_m"] * 100
    assert plan.recommended["orthophoto-resolution"].value == math.ceil(gsd_cm * 10) / 10
    if "gps-accuracy" in plan.recommended:                   # RTK positions
        assert plan.recommended["gps-accuracy"].value >= 0.02
    assert plan.coverage["frames"] == nadir and plan.coverage["images_per_point_median"] >= 1
    assert "--radiometric-calibration" in plan.odm_arguments()
    assert any(f.code == "webodm_low_overlap" for f in res.findings) == (plan.coverage["images_per_point_median"] < 9)


@pytest.mark.flight
def test_check_webodm_cli():
    proc = _cli("check", "--input-dir", _flight(), "--webodm", "--format", "json")
    out = json.loads(proc.stdout)
    assert out["webodm"]["process_command"].startswith("dji-thermal-cli process --input-dir INPUT")
    assert "findings" not in out["webodm"] and out["webodm"]["odm_arguments"][:2] == ["--radiometric-calibration", "none"]
    table = _cli("check", "--input-dir", _flight(), "--webodm").stdout
    assert "WebODM / ODM" in table and "orthophoto-resolution" in table


@pytest.mark.flight
def test_process_normalize_and_max_pitch(tmp_path):
    """Normalized frames: the flight-median ambient, minus each frame's overlap offset."""
    flight = _flight()
    nadir, oblique = flight_counts()
    norm = dtc.normalization(flight, options=dtc.MosaicOptions(align=False))
    assert norm.frames == nadir and norm.without_offset == oblique
    report = dtc.process(flight, tmp_path, outputs=["temp_c"], normalize=norm, max_pitch=-80)
    done = [r for r in report.results if r.ok]
    thermal = set(flight_files())
    assert len(done) == nadir and all("oblique" in r.reason for r in report.results if not r.ok and r.src in thermal)
    r = done[len(done) // 2]
    settings = dtc.Settings(ambient_temp=norm.ambient_c, reflected_temp=norm.reflected_c)
    want = dtc.measure(r.src, settings).temperature.astype(np.float64) - norm.offsets_c[r.src]
    got = tifffile.imread(r.outputs[0])
    valid = got > -273
    np.testing.assert_array_equal(got[valid], want.astype(np.float32)[valid])
    with tifffile.TiffFile(r.outputs[0]) as tf:
        text = tf.pages[0].tags["XMP"].value.decode()
    recorded = float(re.search(r'dtc:NormalizationOffsetC="([-0-9.]+)"', text).group(1))
    assert recorded == pytest.approx(norm.offsets_c[r.src], abs=1e-4)
    ambient = float(re.search(r'dtc:AmbientTemperatureC="([-0-9.]+)"', text).group(1))
    assert ambient == pytest.approx(norm.ambient_c, abs=1e-3)
