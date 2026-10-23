"""The mosaic: attitude convention, geometry end to end on a synthetic flight over a known
temperature field, GeoTIFF tags, and the command line."""

from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest
import tifffile
from PIL import Image

from dji_thermal_cli import api, geo, mosaic, units

from .conftest import flight_counts, flight_dir, sample_files, samples_dir
from .test_cli import _cli
from .test_rangefinder import _with_xmp, _xmp

LAT, LON, GROUND = 42.1034, -88.0767, 200.0


def test_rotation_axes():
    np.testing.assert_allclose(mosaic.rotation(0, 0, 0), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(mosaic.rotation(90, 0, 0)[:, 0], [0, 1, 0], atol=1e-12)    # facing east
    np.testing.assert_allclose(mosaic.rotation(0, -90, 0)[:, 0], [0, 0, 1], atol=1e-12)   # looking down


def test_gimbal_lock_equivalence():
    """DJI nadir frames report gimbal yaw y with roll 180, the same pointing as yaw y+180."""
    np.testing.assert_allclose(mosaic.rotation(5.4, -90, 180), mosaic.rotation(-174.6, -90, 0), atol=1e-12)


_RNG = np.random.default_rng(7)
_WAVES = [(_RNG.uniform(1.0, 8.0), _RNG.uniform(0, 2 * np.pi), _RNG.uniform(0, 2 * np.pi)) for _ in range(40)]
FIELD_SLOPE = 5 / 3.0 + 0.3 * sum(k for k, _, _ in _WAVES)       # bound on the field's gradient, C/m


def _field(e, n):
    """A known ground temperature field over UTM coordinates (degrees C): a smooth trend plus
    aperiodic texture (40 waves of random direction and wavelength), so that images of it
    align unambiguously."""
    t = 20 + 5 * np.sin(e / 3.0)
    for k, direction, phase in _WAVES:
        t = t + 0.3 * np.sin(k * (e * np.cos(direction) + n * np.sin(direction)) + phase)
    return t.astype(np.float32)


def _frame(name, east_m, north_m, yaw, roll=0.0, alt=21.0, size=(320, 256), focal=500.0):
    """A synthetic nadir frame seeing _field, from local offsets around LAT/LON."""
    w, h = size
    lat = LAT + north_m / 111132.0
    lon = LON + east_m / (111320.0 * math.cos(math.radians(LAT)))
    zone, north = geo.utm_zone(LAT, LON)
    c = np.array(geo.utm(lat, lon, zone, north))
    j = geo.local_to_utm(lat, lon, zone, north)
    r = mosaic.rotation(yaw, -90.0, roll)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    rays = np.tensordot(r, np.stack([np.ones((h, w)), (u - (w - 1) / 2) / focal, (v - (h - 1) / 2) / focal]), 1)
    t = alt / rays[2]
    e, n = c[:, None, None] + np.tensordot(j, np.stack([rays[1] * t, rays[0] * t]), 1)
    pose = mosaic.Pose(lat, lon, GROUND + alt, yaw, -90.0, roll, focal, w, h, GROUND, None, {})
    return mosaic.Frame(name, _field(e, n), pose)


@pytest.fixture(scope="module")
def synthetic():
    frames = [_frame(f"f{i}", 4.0 * (i % 3), -4.0 * (i // 3), yaw=(5.4 if i < 3 else -174.6), roll=(180.0 if i < 3 else 0.0))
              for i in range(6)]
    return frames, mosaic.build(frames, align=False)


def _max_field_error(m):
    g = m.grid
    e = g.e0 + (np.arange(g.width) + 0.5) * g.gsd
    n = g.n0 - (np.arange(g.height) + 0.5) * g.gsd
    ok = np.isfinite(m.temperature)
    return float(np.abs(m.temperature[ok] - _field(*np.meshgrid(e, n))[ok]).max())


def test_alignment_keeps_perfect_frames(synthetic):
    """With nothing to correct, alignment stays put: the mosaic still matches the field."""
    frames, _ = synthetic
    m = mosaic.build(frames)
    assert _max_field_error(m) <= FIELD_SLOPE * 1.5 * m.grid.gsd * math.sqrt(2)
    assert m.report["alignment"]["frame_shift_m"]["max"] < 2 * m.grid.gsd


def test_synthetic_flight_geometry(synthetic):
    frames, m = synthetic
    g = m.grid
    assert g.epsg == 32616 and m.ground_alt == GROUND and m.report["frames_contributing"] == 6
    e = g.e0 + (np.arange(g.width) + 0.5) * g.gsd
    n = g.n0 - (np.arange(g.height) + 0.5) * g.gsd
    truth = _field(*np.meshgrid(e, n))
    ok = np.isfinite(m.temperature)
    assert ok.mean() > 0.5
    # nearest-neighbour sampling: off by at most the field's change over about one pixel
    assert np.abs(m.temperature[ok] - truth[ok]).max() <= FIELD_SLOPE * 1.5 * g.gsd * math.sqrt(2)
    mis = m.report["misalignment"]
    assert mis["pairs"] >= 3 and mis["max_m"] < g.gsd     # frames agree to under a pixel


def test_values_are_original_measurements(synthetic):
    frames, m = synthetic
    ok = np.isfinite(m.temperature)
    for k, f in enumerate(frames):
        mine = m.temperature[ok & (m.source == k)]
        assert np.isin(mine, f.temperature).all()
    assert np.nanmax(m.view_angle) < 45


def test_geotiff(synthetic, tmp_path):
    _, m = synthetic
    tif, sidecar = mosaic.write(m, tmp_path / "mosaic_temp_c.tif")
    with tifffile.TiffFile(tif) as tf:
        page = tf.pages[0]
        assert page.asarray().shape == (m.grid.height, m.grid.width, 3)
        assert tf.geotiff_metadata["ProjectedCSTypeGeoKey"] == 32616
        assert page.tags["ModelPixelScaleTag"].value[:2] == (m.grid.gsd, m.grid.gsd)
        assert page.tags["ModelTiepointTag"].value[3:5] == (m.grid.e0, m.grid.n0)
        assert "temperature_c" in page.tags["GDAL_METADATA"].value and page.tags["GDAL_NODATA"].value == "nan"
        np.testing.assert_array_equal(page.asarray()[..., 0], m.temperature)
    info = json.loads(sidecar.read_text())
    assert [f["file"] for f in info["frames"]] == [f"f{i}" for i in range(6)]


@pytest.mark.parametrize("unit, band, convert", [("F", "temperature_f", lambda c: c * 9 / 5 + 32),
                                                 ("K", "temperature_k", lambda c: c + 273.15)])
def test_geotiff_units(synthetic, tmp_path, unit, band, convert):
    """--mosaic-unit: the temperature band converted once from the Celsius mosaic, NaN kept."""
    _, m = synthetic
    tif, sidecar = mosaic.write(m, tmp_path / f"mosaic_{band}.tif", unit=unit)
    with tifffile.TiffFile(tif) as tf:
        page = tf.pages[0]
        assert band in page.tags["GDAL_METADATA"].value
        want = convert(m.temperature.astype(np.float64)).astype(np.float32)
        np.testing.assert_array_equal(page.asarray()[..., 0], want)
        np.testing.assert_array_equal(page.asarray()[..., 1], m.source)
    assert json.loads(sidecar.read_text())["bands"][0] == {"name": band, "unit": units.SYMBOLS[unit]}


def test_focal_length():
    """Focal length over pixel pitch, scaled for super-resolution images; the rounded 35 mm
    equivalent only for cameras not in SENSORS."""
    assert mosaic.focal_length_px("M4T", 640, 512, 12.0, 52.0) == (pytest.approx(1000.0), "focal length / pixel pitch")
    assert mosaic.focal_length_px("M30T", 1280, 1024, 9.1, 40.0)[0] == pytest.approx(9.1 / 0.012 * 2)
    assert mosaic.focal_length_px("XT S", 640, 512, 19.0, None)[0] == pytest.approx(19.0 / 0.017)
    f, source = mosaic.focal_length_px("other", 640, 512, 12.0, 52.0)
    assert f == pytest.approx(52.0 * math.hypot(640, 512) / mosaic.FULL_FRAME_DIAGONAL_MM) and source.startswith("35 mm")
    assert mosaic.focal_length_px("M2EA", 640, 512, None, None) == (None, "")


def test_pose_takes_the_measured_size_not_the_upscaled_jpegs():
    """A super-resolution frame (M4T: a 1280x1024 JPEG over 640x512 measurements) is posed at
    the size it measures, with the focal length scaled to it; a JPEG that is no R-JPEG keeps
    its own size."""
    exif = Image.Exif()
    exif[272] = "M4T"
    exif.get_ifd(0x8769)[37386] = 12.0
    buf = io.BytesIO()
    Image.new("L", (1280, 1024)).save(buf, "JPEG", exif=exif)
    data = _with_xmp(buf.getvalue(), _xmp(
        'drone-dji:GpsLatitude="42.1" drone-dji:GpsLongitude="-88.1" drone-dji:AbsoluteAltitude="250" '
        'drone-dji:GimbalYawDegree="0" drone-dji:GimbalPitchDegree="-90" drone-dji:GimbalRollDegree="0">'))
    pose, why = mosaic.read_pose(data, (640, 512))
    assert why is None and (pose.width, pose.height) == (640, 512) and pose.focal_px == pytest.approx(1000.0)
    pose, _ = mosaic.read_pose(data)
    assert (pose.width, pose.height) == (1280, 1024) and pose.focal_px == pytest.approx(2000.0)


@pytest.mark.samples
def test_focal_length_matches_35mm_equivalent():
    """For every SDK sample camera with both EXIF values, the two focal lengths agree to the
    35 mm equivalent's rounding plus the spread of DJI's nominal figures (H20T: 2.4%)."""
    checked = set()
    for path in sample_files():
        with Image.open(path) as image:
            exif, size = image.getexif(), image.size
        lens = exif.get_ifd(0x8769)
        model = str(exif.get(272, "")).strip("\0 ")
        assert model in mosaic.SENSORS, model
        if lens.get(37386) and lens.get(41989):
            f, _ = mosaic.focal_length_px(model, *size, float(lens[37386]), None)
            f35, _ = mosaic.focal_length_px("", *size, None, float(lens[41989]))
            assert abs(f / f35 - 1) < 0.03, (path, f, f35)
            checked.add(model)
    assert checked, "no sample records both focal lengths"


def test_preview_palette(synthetic, tmp_path):
    """The preview maps one temperature range for the whole mosaic through the palette."""
    _, m = synthetic
    lut = np.stack([np.arange(256), 255 - np.arange(256), np.zeros(256)], axis=1).astype(np.uint8)
    t = m.temperature[np.isfinite(m.temperature)]
    low, high = float(t.min()), float(t.max())
    path, used_low, used_high = mosaic.write_preview(m, tmp_path / "p.png", lut, low, high)
    assert (used_low, used_high) == (low, high)
    rgba = np.asarray(Image.open(path))
    ok = np.isfinite(m.temperature)
    level = np.rint((m.temperature[ok] - low) / (high - low) * 255).astype(int)
    np.testing.assert_array_equal(rgba[ok][:, 0], level)
    np.testing.assert_array_equal(rgba[ok][:, 1], 255 - level)
    assert (rgba[ok][:, 3] == 255).all() and (rgba[~ok][:, 3] == 0).all()
    gray, _, _ = mosaic.write_preview(m, tmp_path / "g.png")
    with Image.open(gray) as im:
        assert im.mode == "LA"


@pytest.mark.samples
def test_no_pose_data(tmp_path):
    proc = _cli("process", "--input-dir", samples_dir("M30T"), "--output-dir", tmp_path, "--mosaic", "--jobs", "1")
    assert proc.returncode != 0 and "No frames to mosaic." in proc.stderr
    assert "no camera pose in the DJI XMP" in proc.stdout


@pytest.mark.flight
def test_flight(tmp_path):
    proc = _cli("process", "--input-dir", flight_dir(), "--output-dir", tmp_path, "--mosaic")
    assert proc.returncode == 0, proc.stderr
    nadir, oblique = flight_counts()
    assert f"Mosaic: {nadir} nadir frames, {oblique} oblique left out" in proc.stdout
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mosaic_preview.png", "mosaic_temp_c.json", "mosaic_temp_c.tif"]
    assert "mosaic_preview.png  (ironred, " in proc.stdout
    with Image.open(tmp_path / "mosaic_preview.png") as im:
        assert im.mode == "RGBA"
    steps = [line for line in proc.stderr.splitlines() if line.startswith("Mosaic step ")]
    assert steps[0] == f"Mosaic step 1 of 5: reading frames, {len(api.scan(flight_dir()).jpegs)} files"
    assert steps[-1] == "Mosaic step 5 of 5: writing, 2 files"
    assert sum("aligning (round" in line for line in steps) == 6     # pairs, then solve steps, for 3 rounds


@pytest.mark.flight
def test_flight_quiet_prints_the_results_without_progress(tmp_path):
    proc = _cli("process", "--input-dir", flight_dir(), "--output-dir", tmp_path, "--mosaic", "--no-mosaic-align", "--quiet")
    assert proc.returncode == 0, proc.stderr
    assert "Mosaic step" not in proc.stderr + proc.stdout
    assert "mosaic_preview.png  (ironred, " in proc.stdout


def test_alignment_recovers_boresight(monkeypatch):
    """Frames taken with a camera boresight (0.6, -0.4 deg) and up to ~0.8 deg of heading
    jitter that the metadata doesn't report: alignment finds the boresight and brings the
    mosaic back onto the known field."""
    bias = (math.radians(0.6), math.radians(-0.4))
    rng = np.random.default_rng(3)
    true_rotation = mosaic.rotation
    frames = []
    for i in range(9):
        yaw = 5.4 if (i // 3) % 2 == 0 else -174.6          # two opposite passes, as DJI reports them
        monkeypatch.setattr(mosaic, "rotation", lambda y, p, r: true_rotation(y, p, r) @ mosaic._boresight(*bias))
        f = _frame(f"f{i}", 3.5 * (i % 3), -3.5 * (i // 3), yaw=yaw + rng.normal(0, 0.4), roll=180.0)
        monkeypatch.setattr(mosaic, "rotation", true_rotation)
        f.pose.yaw = yaw
        frames.append(f)

    def field_error(m):
        g = m.grid
        e = g.e0 + (np.arange(g.width) + 0.5) * g.gsd
        n = g.n0 - (np.arange(g.height) + 0.5) * g.gsd
        ok = np.isfinite(m.temperature)
        return float(np.median(np.abs(m.temperature[ok] - _field(*np.meshgrid(e, n))[ok])))

    plain = mosaic.build(frames, align=False)
    aligned = mosaic.build(frames)
    ali = aligned.report["alignment"]
    assert ali["boresight_deg"]["pitch"] == pytest.approx(0.6, abs=0.05)
    assert ali["boresight_deg"]["yaw"] == pytest.approx(-0.4, abs=0.05)
    assert aligned.report["misalignment"]["median_m"] < 0.5 * plain.report["misalignment"]["median_m"]
    assert field_error(aligned) < 0.25 * field_error(plain)


def test_frame_offsets_normalize(synthetic):
    """Frames read 0.8 C warm or cold (camera drift between shutter recalibrations): the
    offsets are found, and --mosaic-normalize removes them."""
    frames, _ = synthetic
    drift = np.array([0.0, 0.8, -0.8, 0.4, 0.0, -0.4])
    shifted = [mosaic.Frame(f.name, f.temperature + np.float32(d), f.pose) for f, d in zip(frames, drift)]
    m = mosaic.build(shifted, align=False, check_misalignment=False)
    found = np.array([f["temperature_offset_c"] for f in m.frames])
    np.testing.assert_allclose(found - np.median(found), drift - np.median(drift), atol=0.05)
    assert not m.report["frame_offsets"]["normalized"]
    h = mosaic.build(shifted, align=False, check_misalignment=False, normalize=True)
    assert h.report["radiometric_seams"]["median_abs_c"] < 0.5 * m.report["radiometric_seams"]["median_abs_c"]


def test_build_reports_each_stage_counting_up_to_its_total(synthetic):
    """build() reports alignment (pairs, then solve steps, per round), compositing and the
    alignment check, each from 0 up to its total, and the mosaic is the same as without progress."""
    frames, _ = synthetic
    events = []
    m = mosaic.build(frames, progress=lambda stage, done, total, unit, **extra: events.append((stage, unit, done, total, extra)))
    runs = {}
    for stage, unit, done, total, extra in events:
        runs.setdefault((stage, unit, extra.get("round")), []).append((done, total))
    assert list(dict.fromkeys(s for s, *_ in events)) == ["align", "composite", "check"]
    assert {r for s, _, r in runs if s == "align"} == {1, 2, 3}
    assert all(extra == {"round": extra["round"], "rounds": 3} for s, *_, extra in events if s == "align")
    assert runs[("composite", "frames", None)][-1] == (6, 6)
    assert runs[("align", "solve steps", 1)] == [(i, 8) for i in range(9)]
    for counts in runs.values():
        done = [d for d, _ in counts]
        assert done[0] == 0 and done[-1] == counts[0][1] and done == sorted(done)
    plain = mosaic.build(frames)
    np.testing.assert_array_equal(m.temperature, plain.temperature)
    assert m.report == plain.report


def test_build_without_alignment_reports_no_align_stage(synthetic):
    frames, _ = synthetic
    stages = []
    mosaic.build(frames, align=False, progress=lambda stage, *a, **k: stages.append(stage))
    assert list(dict.fromkeys(stages)) == ["composite", "check"]


def test_build_in_several_threads_gives_the_same_mosaic(synthetic):
    frames, _ = synthetic
    one = mosaic.build(frames, jobs=1)
    four = mosaic.build(frames, jobs=4)
    np.testing.assert_array_equal(one.temperature, four.temperature)
    assert one.report == four.report
