"""The mosaic: attitude convention, geometry end to end on a synthetic flight over a known
temperature field, GeoTIFF tags, and the command line."""

from __future__ import annotations

import collections
import dataclasses
import io
import json
import math
import re
import subprocess
import sys

import numpy as np
import pytest
import tifffile
from PIL import Image

from dji_thermal_cli import api, geo, mosaic, units

from .conftest import flight_counts, flight_dir, sample_files, samples_dir
from .test_cli import _cli
from . import tiffs
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


def _frame(name, east_m, north_m, yaw, roll=0.0, alt=21.0, size=(320, 256), focal=500.0, surface=None):
    """A synthetic nadir frame seeing _field, from local offsets around LAT/LON; on the ground
    plane, or on surface(e, n) (the height above it, crossed once by each ray), whose height
    under the frame's centre is what the rangefinder reports."""
    w, h = size
    lat = LAT + north_m / 111132.0
    lon = LON + east_m / (111320.0 * math.cos(math.radians(LAT)))
    zone, north = geo.utm_zone(LAT, LON)
    c = np.array(geo.utm(lat, lon, zone, north))
    j = geo.local_to_utm(lat, lon, zone, north)
    r = mosaic.rotation(yaw, -90.0, roll)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    rays = np.tensordot(r, np.stack([np.ones((h, w)), (u - (w - 1) / 2) / focal, (v - (h - 1) / 2) / focal]), 1)
    def ground(t):
        return c[:, None, None] + np.tensordot(j, np.stack([rays[1] * t, rays[0] * t]), 1)

    t = alt / rays[2]
    lrf = GROUND
    if surface is not None:                  # bisect along each ray for where it meets the surface
        lo, hi = np.zeros_like(t), t
        for _ in range(40):
            mid = (lo + hi) / 2
            above = alt - mid * rays[2] > surface(*ground(mid))
            lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
        t = (lo + hi) / 2
        lrf = GROUND + float(surface(*c))
    e, n = ground(t)
    pose = mosaic.Pose(lat, lon, GROUND + alt, yaw, -90.0, roll, focal, w, h, lrf, None, {})
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


def test_each_pixel_keeps_where_in_its_source_frame_it_was_read(synthetic):
    frames, m = synthetic
    ok = np.isfinite(m.temperature)
    assert (np.isfinite(m.source_px).all(axis=-1) == ok).all()
    for k, f in enumerate(frames):
        mine = ok & (m.source == k)
        u, v = (m.source_px[mine][:, i].astype(np.intp) for i in (0, 1))
        np.testing.assert_array_equal(f.temperature[v, u], m.temperature[mine])
        assert u.min() >= 0 and u.max() < f.pose.width and v.min() >= 0 and v.max() < f.pose.height


def test_geotiff(synthetic, tmp_path):
    _, m = synthetic
    tif, sidecar = mosaic.write(m, tmp_path / "mosaic_temp_c.tif")
    with tifffile.TiffFile(tif) as tf:
        page = tf.pages[0]
        assert page.asarray().shape == (m.grid.height, m.grid.width, 5)
        assert tf.geotiff_metadata["ProjectedCSTypeGeoKey"] == 32616
        assert page.tags["ModelPixelScaleTag"].value[:2] == (m.grid.gsd, m.grid.gsd)
        assert page.tags["ModelTiepointTag"].value[3:5] == (m.grid.e0, m.grid.n0)
        assert "temperature_c" in page.tags["GDAL_METADATA"].value and page.tags["GDAL_NODATA"].value == "nan"
        np.testing.assert_array_equal(page.asarray()[..., 0], m.temperature)
        np.testing.assert_array_equal(page.asarray()[..., 3:], m.source_px)
        for name in ("source_frame", "view_angle_deg", "source_u", "source_v"):
            assert f">{name}<" in page.tags["GDAL_METADATA"].value
    info = json.loads(sidecar.read_text())
    assert [f["file"] for f in info["frames"]] == [f"f{i}" for i in range(6)]
    assert [b["name"] for b in info["bands"]] == ["temperature_c", "source_frame", "view_angle_deg", "source_u",
                                                  "source_v"]


def test_a_mosaic_without_source_pixels_is_written_with_three_bands(synthetic, tmp_path):
    _, m = synthetic
    tif, sidecar = mosaic.write(dataclasses.replace(m, source_px=None), tmp_path / "mosaic_temp_c.tif")
    with tifffile.TiffFile(tif) as tf:
        assert tf.pages[0].asarray().shape == (m.grid.height, m.grid.width, 3)
    assert len(json.loads(sidecar.read_text())["bands"]) == 3


def test_heights_geotiff_holds_the_surface_altitudes_on_its_cells(synthetic, tmp_path):
    _, m = synthetic
    assert mosaic.write_heights(m, tmp_path / "plane.tif") is None and not (tmp_path / "plane.tif").exists()
    g, step = m.grid, 8
    rows, cols = -(-g.height // step), -(-g.width // step)
    z = np.arange(rows * cols, dtype=float).reshape(rows, cols) / 10
    temperature = m.temperature.copy()
    temperature[:step, :step] = np.nan              # a cell no frame covers
    m = dataclasses.replace(m, temperature=temperature, heights=mosaic.Heights(g.e0, g.n0, step * g.gsd, z),
                            report={**m.report, "heights": {"method": "parallax", "cell_m": step * g.gsd}})
    tif = mosaic.write_heights(m, tmp_path / "mosaic_heights.tif")
    with tifffile.TiffFile(tif) as tf:
        page = tf.pages[0]
        got = page.asarray()
        assert got.dtype == np.float32 and got.shape == (rows, cols)
        assert tf.geotiff_metadata["ProjectedCSTypeGeoKey"] == 32616
        assert page.tags["ModelPixelScaleTag"].value[:2] == (step * g.gsd, step * g.gsd)
        assert page.tags["ModelTiepointTag"].value[3:5] == (g.e0, g.n0)
        assert "altitude_m" in page.tags["GDAL_METADATA"].value and "parallax" in page.tags["GDAL_METADATA"].value
    covered = np.isfinite(got)
    assert not covered[0, 0] and covered.sum() > 1
    np.testing.assert_allclose(got[covered], (z + m.ground_alt)[covered], atol=1e-3)
    with tifffile.TiffFile(mosaic.write_heights(m, tmp_path / "unlocated.tif", located=False)) as tf:
        assert "ModelTiepointTag" not in tf.pages[0].tags
        np.testing.assert_allclose(tf.pages[0].asarray()[covered], z[covered], atol=1e-6)


def _lonlat(m, rows, cols):
    """The lon/lat of mosaic pixel centres."""
    g = m.grid
    lat, lon = geo.utm_inverse(g.e0 + (cols + 0.5) * g.gsd, g.n0 - (rows + 0.5) * g.gsd, g.zone, g.north)
    return lon, lat


def _check_frame_models(frames, m, tmp_path):
    """The models read back from the sidecar find every mosaic pixel at the source frame pixel
    the mosaic read it from, and every frame that saw it saw the same ground."""
    _, sidecar = mosaic.write(m, tmp_path / "mosaic_temp_c.tif")
    models = api.read_frame_models(sidecar)
    assert models.files == tuple(f["file"] for f in m.frames)
    rows, cols = np.nonzero(np.isfinite(m.temperature))
    pick = np.random.default_rng(1).choice(rows.size, 400, replace=False)
    rows, cols = rows[pick], cols[pick]
    lon, lat = _lonlat(m, rows, cols)
    px = models.pixels(lon, lat)
    source = m.source[rows, cols].astype(np.intp)
    used = px[source, np.arange(rows.size)]
    # the same pixel, but for points within the lon/lat round trip's error (well under 0.01
    # pixel) of a pixel's edge
    assert np.abs(used - m.source_px[rows, cols]).max() <= 0.51
    assert (np.floor(used + 0.5) == m.source_px[rows, cols]).all(axis=1).mean() > 0.98
    by_name = {f.name: f for f in frames}
    g = m.grid
    truth = _field(g.e0 + (cols + 0.5) * g.gsd, g.n0 - (rows + 0.5) * g.gsd)
    seen = np.isfinite(px[..., 0])
    assert (seen.sum(axis=0) > 1).mean() > 0.5         # most points were seen by more than the frame used
    frame_gsd = max(pl.h / pl.frame.pose.focal_px for pl in m.placed)
    for k, name in enumerate(models.files):
        u, v = (np.floor(px[k, seen[k], i] + 0.5).astype(np.intp) for i in (0, 1))
        got = by_name[name].temperature[v, u]
        assert np.abs(got - truth[seen[k]]).max() <= FIELD_SLOPE * (frame_gsd + g.gsd) * math.sqrt(2)
    angles = models.view_angles(lon, lat)
    np.testing.assert_allclose(angles[source, np.arange(rows.size)], m.view_angle[rows, cols], atol=1e-3)
    assert (np.isnan(angles) == ~seen).all()
    return models


def test_frame_models_find_a_point_in_every_frame_that_saw_it(synthetic, tmp_path):
    frames, _ = synthetic
    m = mosaic.build(frames, check_misalignment=False)
    models = _check_frame_models(frames, m, tmp_path)
    assert m.report["alignment"] is not None
    assert any(pl.shift.any() or pl.rot for pl in models._placed)    # the alignment's corrections are kept


def test_frame_models_find_points_on_a_surface_from_parallax(tmp_path):
    zone, north = geo.utm_zone(LAT, LON)
    centre = np.array(geo.utm(LAT - 4.0 / 111132.0, LON + 4.0 / (111320.0 * math.cos(math.radians(LAT))), zone, north))

    def mound(e, n):
        return 1.5 * np.exp(-((e - centre[0]) ** 2 + (n - centre[1]) ** 2) / 8.0)

    frames = [_frame(f"f{i}", 4.0 * (i % 3), -4.0 * (i // 3), yaw=5.4 if (i // 3) % 2 == 0 else -174.6,
                     roll=180.0, surface=mound) for i in range(6)]
    m = mosaic.build(frames, check_misalignment=False)
    assert m.heights is not None
    _check_frame_models(frames, m, tmp_path)


def test_frame_models_say_which_frames_saw_a_point_or_meet_an_area(synthetic, tmp_path):
    frames, m = synthetic
    models = api.read_frame_models(mosaic.write(m, tmp_path / "mosaic_temp_c.tif")[0])
    centre = models.footprints[4].mean(axis=0)
    assert 4 in models.covering(*centre) and set(models.covering(*centre)) <= set(range(6))
    lon, lat = models.footprints[0].min(axis=0) - 0.001
    assert models.covering(lon, lat) == [] and np.isnan(models.pixels(lon, lat)).all()
    assert models.pixels([lon, centre[0]], [lat, centre[1]]).shape == (6, 2, 2)
    every = np.concatenate(models.footprints)
    lo, hi = every.min(axis=0), every.max(axis=0)
    assert models.meeting([lo, (hi[0], lo[1]), hi, (lo[0], hi[1])]) == list(range(6))
    d = 1e-6
    assert models.meeting([centre - d, centre + (d, -d), centre + d]) == models.covering(*centre)
    far = lo - 0.01
    assert models.meeting([far, far + (d, 0), far + d]) == []
    # a thin area crossing frame 0's edge without a vertex inside it
    left = models.footprints[0][:, 0].min()
    mid = models.footprints[0][:, 1].mean()
    assert 0 in models.meeting([(left - 1e-4, mid), (left + 1e-4, mid - 1e-7), (left - 1e-4, mid - 2e-7)])
    with pytest.raises(ValueError, match="at least 3"):
        models.meeting([lo, hi])


def test_frame_models_are_in_the_located_sidecar_only(synthetic, tmp_path):
    _, m = synthetic
    tif, sidecar = mosaic.write(m, tmp_path / "mosaic_temp_c.tif")
    info = json.loads(sidecar.read_text())
    assert info["frame_models"]["version"] == 1 and info["frame_models"]["surface"] is None
    assert len(info["frame_models"]["frames"][0]["footprint_lonlat"]) == 4
    with tifffile.TiffFile(tif) as tf:
        assert "frame_models" not in tf.pages[0].tags["GDAL_METADATA"].value
    del info["frame_models"]                                        # a sidecar from before 1.5.0
    assert mosaic.FrameModels.from_sidecar(info) is None
    _, unlocated = mosaic.write(m, tmp_path / "unlocated.tif", located=False)
    assert api.read_frame_models(unlocated) is None
    with pytest.raises(ValueError, match="version 2"):
        mosaic.FrameModels.from_sidecar({"frame_models": {"version": 2}})


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


def _preview_palette(stdout: str) -> str:
    """The palette the mosaic's preview line names ("mosaic_preview.png  (whitehot, 9.6 to ...")."""
    m = re.search(r"mosaic_preview\.png  \((\w+), ", stdout)
    assert m, stdout
    return m.group(1)


@pytest.mark.flight
def test_flight(tmp_path):
    proc = _cli("process", "--input-dir", flight_dir(), "--output-dir", tmp_path, "--mosaic")
    assert proc.returncode == 0, proc.stderr
    nadir, oblique = flight_counts()
    assert f"Mosaic: {nadir} nadir frames, {oblique} oblique left out" in proc.stdout
    names = ["mosaic_heights.tif", "mosaic_preview.png", "mosaic_temp_c.json", "mosaic_temp_c.tif"]
    assert sorted(p.name for p in tmp_path.iterdir()) == names
    assert _preview_palette(proc.stdout) in api.PALETTES                 # the flight's own, whichever it recorded
    assert re.search(r"mosaic_heights\.tif  \(\d+ cm cells, [+-]\d+\.\d to [+-]\d+\.\d m above the ground plane\)", proc.stdout)
    with Image.open(tmp_path / "mosaic_preview.png") as im:
        assert im.mode == "RGBA"
    steps = [line for line in proc.stderr.splitlines() if line.startswith("Mosaic step ")]
    assert steps[0] == f"Mosaic step 1 of 5: reading frames, {len(api.scan(flight_dir()).jpegs)} files"
    assert steps[-1] == "Mosaic step 5 of 5: writing, 3 files"
    # the heights written are a DSM the next run can project onto
    again = _cli("process", "--input-dir", flight_dir(), "--output-dir", tmp_path / "dsm", "--mosaic", "--quiet",
                 "--no-mosaic-align", "--mosaic-dsm", tmp_path / "mosaic_heights.tif")
    assert again.returncode == 0, again.stderr
    info = json.loads((tmp_path / "dsm" / "mosaic_temp_c.json").read_text())["report"]["heights"]
    assert info["method"] == "DSM" and info["file"] == "mosaic_heights.tif" and info["observed_cells"] > 0
    assert abs(info["offset_m"]) < 0.5 and "rangefinder" in info["offset_source"]
    rounds = collections.Counter(re.search(r"aligning \(round (\d+)/(\d+)\)", line).groups()
                                 for line in steps if "aligning (round" in line)
    assert rounds and set(rounds.values()) == {2}     # pairs, then solve steps, for each round
    for total in {t for _, t in rounds}:     # 4 rounds with heights, 3 more on the plane if they fall back
        assert {int(r) for r, t in rounds if t == total} == set(range(1, int(total) + 1))


@pytest.mark.flight
def test_flight_quiet_prints_the_results_without_progress(tmp_path):
    proc = _cli("process", "--input-dir", flight_dir(), "--output-dir", tmp_path, "--mosaic", "--no-mosaic-align", "--quiet")
    assert proc.returncode == 0, proc.stderr
    assert "Mosaic step" not in proc.stderr + proc.stdout
    assert _preview_palette(proc.stdout) in api.PALETTES                 # the flight's own, whichever it recorded


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


def test_alignment_recovers_a_roof_from_parallax():
    """M4T-sized frames over a 2.5 m high, 8 m wide block: on the ground plane the roof's parallax
    misaligns them; with heights, alignment finds the block's height from that parallax,
    the ground around it stays near the plane, and the frames line up on both."""
    zone, north = geo.utm_zone(LAT, LON)
    centre = np.array(geo.utm(LAT - 5.0 / 111132.0, LON + 5.0 / (111320.0 * math.cos(math.radians(LAT))), zone, north))

    def roof(e, n):
        def edge(x):
            return 0.5 * (np.tanh((x + 4.0) / 0.5) - np.tanh((x - 4.0) / 0.5))
        return 2.5 * edge(e - centre[0]) * edge(n - centre[1])

    frames = [_frame(f"f{i}", 5.0 * (i % 3), -5.0 * (i // 3), yaw=5.4 if (i // 3) % 2 == 0 else -174.6,
                     roll=180.0, size=(640, 512), surface=roof) for i in range(9)]
    plane = mosaic.build(frames, heights="plane")
    m = mosaic.build(frames)
    h = m.report["alignment"]["heights"]
    assert plane.heights is None and plane.report["alignment"]["heights"] == {"method": "plane", "reason": "not asked for"}
    assert h["method"].startswith("parallax") and h["anchors"] == 9
    on_roof, beside = m.heights.at_points(centre[0] + np.array([-1.0, 9.0]), centre[1] + np.array([1.0, 9.0]))
    assert on_roof == pytest.approx(2.5, abs=0.5)
    assert abs(beside) < 0.4
    assert m.report["misalignment"]["median_m"] < 0.5 * plane.report["misalignment"]["median_m"]


def test_utm_inverse_returns_the_latitude_and_longitude():
    lat = np.array([LAT, -33.9, 0.0, 71.2, -60.5])
    lon = np.array([LON, 18.4, 2.9, 25.7, -64.1])
    for la, lo in zip(lat, lon):
        zone, north = geo.utm_zone(la, lo)
        # out to the zone's edges, where the series need most terms
        for d in (0.0, 2.9, -2.9):
            e, n = geo.utm(la, lo + d, zone, north)
            got = geo.utm_inverse(e, n, zone, north)
            np.testing.assert_allclose(got, (la, lo + d), atol=1e-9)


def _write_dsm(path, z, x0, y0, dx, dy, epsg, point=False, nodata="-9999"):
    """A north-up GeoTIFF as GDAL writes one: z's pixel (0, 0) has its top-left (or with point,
    its centre) at (x0, y0)."""
    geographic = epsg == 4326
    keys = [1, 1, 0, 4, 1024, 0, 1, 2 if geographic else 1, 1025, 0, 1, 2 if point else 1,
            2048 if geographic else 3072, 0, 1, epsg, 4099 if geographic else 3076, 0, 1, 9102 if geographic else 9001]
    tifffile.imwrite(path, np.asarray(z, np.float32), extratags=[
        (33550, 12, 3, (dx, dy, 0.0)), (33922, 12, 6, (0.0, 0.0, 0.0, x0, y0, 0.0)),
        (34735, 3, len(keys), keys), (42113, 2, 0, nodata)])
    return path


def test_dsm_is_read_from_utm_and_latitude_longitude_geotiffs(tmp_path):
    zone, north = geo.utm_zone(LAT, LON)
    e0, n0 = (round(v) for v in geo.utm(LAT, LON, zone, north))
    z = np.arange(12.0).reshape(3, 4) + 200
    z[2, 3] = -9999
    dsm = mosaic.read_dsm(_write_dsm(tmp_path / "utm.tif", z, e0, n0, 0.5, 0.5, geo.epsg(zone, north)))
    assert (dsm.epsg, dsm.x0, dsm.y0, dsm.dx) == (32616, e0, n0, 0.5) and np.isnan(dsm.z[2, 3])
    got = dsm.at_utm(np.array([e0 + 0.25, e0 + 1.75, e0 - 0.1]), np.array([n0 - 0.25, n0 - 0.75, n0 - 0.25]), zone, north)
    np.testing.assert_array_equal(got[:2], [200, 207])
    assert np.isnan(got[2])
    # PixelIsPoint: the tiepoint is pixel (0, 0)'s centre
    point = mosaic.read_dsm(_write_dsm(tmp_path / "point.tif", z, e0, n0, 0.5, 0.5, 32616, point=True))
    assert (point.x0, point.y0) == (e0 - 0.25, n0 + 0.25)
    # lat/lon: the same altitudes found through the inverse projection
    step = 1e-5
    lonlat = mosaic.read_dsm(_write_dsm(tmp_path / "ll.tif", z, LON - 2 * step, LAT + 1.5 * step, step, step, 4326))
    e, n = geo.utm(np.array([LAT + step, LAT - step]), np.array([LON - 1.5 * step, LON + 0.5 * step]), zone, north)
    np.testing.assert_array_equal(lonlat.at_utm(e, n, zone, north), [200, 210])
    assert lonlat.pixel_m() == pytest.approx(step * 111320 * math.cos(math.radians(LAT)), rel=1e-3)
    tifffile.imwrite(tmp_path / "plain.tif", z.astype(np.float32))
    with pytest.raises(ValueError, match="no georeference"):
        mosaic.read_dsm(tmp_path / "plain.tif")
    with pytest.raises(ValueError, match="CRS must be"):
        mosaic.read_dsm(_write_dsm(tmp_path / "web.tif", z, 0, 0, 1, 1, 3857))


def test_a_dsm_as_odm_writes_it_is_read_without_a_tiff_library(tmp_path):
    """ODM's dsm.tif: tiled, Deflate with the floating-point predictor, which tifffile reads only
    with imagecodecs; in a process that can't import tifffile."""
    zone, north = geo.utm_zone(LAT, LON)
    e0, n0 = (round(v) for v in geo.utm(LAT, LON, zone, north))
    z = (np.arange(40 * 30.0).reshape(30, 40) / 7 + 200).astype(np.float32)
    z[2, 3] = -9999
    keys = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, geo.epsg(zone, north))
    path = tiffs.write(tmp_path / "dsm.tif", z[None], compression="deflate", predictor=3, tile=16, extra=[
        (33550, 12, (0.05, 0.05, 0.0)), (33922, 12, (0.0, 0.0, 0.0, e0, n0, 0.0)), (34735, 3, keys),
        (42113, 2, b"-9999\0")])
    code = ("import sys; sys.modules['tifffile'] = None; import numpy as np; from dji_thermal_cli import mosaic; "
            f"d = mosaic.read_dsm({str(path)!r}); print(d.epsg, d.x0, d.y0, d.dx, int(np.isnan(d.z).sum()), d.z[29, 39])")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == [str(geo.epsg(zone, north)), str(float(e0)), str(float(n0)), "0.05", "1",
                                   str(float(z[29, 39]))]


def test_a_dsm_is_read_once_while_its_file_is_unchanged(tmp_path):
    z = np.full((3, 4), 200.0)
    path = _write_dsm(tmp_path / "dsm.tif", z, 0, 0, 1, 1, 32616)
    first = mosaic.read_dsm(path)
    assert mosaic.read_dsm(path) is first and not first.z.flags.writeable
    _write_dsm(path, z + 1, 0, 0, 1, 1, 32616, nodata="-99999")     # a longer file: a new size
    assert mosaic.read_dsm(path).z[0, 0] == 201


def test_a_dsm_that_cannot_be_read_fails_before_any_frame_is_converted(tmp_path):
    (tmp_path / "bad.tif").write_bytes(b"not a tiff")
    options = api.MosaicOptions(dsm=tmp_path / "bad.tif")
    for call in (lambda: api.make_mosaic(tmp_path / "no-such-flight", tmp_path / "out", options=options),
                 lambda: api.normalization(tmp_path / "no-such-flight", options=options)):
        with pytest.raises(api.SettingsError, match="dsm: .*bad.tif: not a readable GeoTIFF"):
            call()
    flight = tmp_path / "flight dsm"
    flight.mkdir()
    proc = _cli("process", "--input-dir", flight, "--output-dir", tmp_path / "out", "--mosaic",
                "--mosaic-dsm", tmp_path / "bad.tif")
    assert proc.returncode != 0 and "Done." not in proc.stdout
    assert proc.stderr.strip().startswith(f"--mosaic-dsm: {tmp_path / 'bad.tif'}: not a readable GeoTIFF (not a TIFF)")
    assert "Normalization failed" not in proc.stderr


def test_option_names_replace_field_names_but_not_paths_holding_them():
    from dji_thermal_cli.cli import _fail
    for message, shown in [("dsm: /data/dsm/dsm.tif: not a readable GeoTIFF",
                            "--mosaic-dsm: /data/dsm/dsm.tif: not a readable GeoTIFF"),
                           (r"dsm: C:\dsm\dsm.tif: bad", r"--mosaic-dsm: C:\dsm\dsm.tif: bad"),
                           ("dsm_offset needs a dsm", "--mosaic-dsm-offset needs a --mosaic-dsm"),
                           ("height_cell must be positive.", "--mosaic-height-cell must be positive.")]:
        with pytest.raises(SystemExit) as exc:
            _fail(ValueError(message))
        assert exc.value.code == shown


def test_frames_are_projected_onto_a_dsm_with_its_offset_estimated_from_the_rangefinder(tmp_path):
    """The roof of test_alignment_recovers_a_roof_from_parallax, given as a DSM whose altitudes
    are 3 m above the frames' (another vertical datum): the offset is found from the
    rangefinder, the frames are projected onto the roof and line up as with parallax heights."""
    zone, north = geo.utm_zone(LAT, LON)
    centre = np.array(geo.utm(LAT - 5.0 / 111132.0, LON + 5.0 / (111320.0 * math.cos(math.radians(LAT))), zone, north))

    def roof(e, n):
        def edge(x):
            return 0.5 * (np.tanh((x + 4.0) / 0.5) - np.tanh((x - 4.0) / 0.5))
        return 2.5 * edge(e - centre[0]) * edge(n - centre[1])

    frames = [_frame(f"f{i}", 5.0 * (i % 3), -5.0 * (i // 3), yaw=5.4 if (i // 3) % 2 == 0 else -174.6,
                     roll=180.0, size=(640, 512), surface=roof) for i in range(9)]
    px, x0, y0 = 0.1, centre[0] - 20, centre[1] + 20
    e, n = np.meshgrid(x0 + (np.arange(400) + 0.5) * px, y0 - (np.arange(400) + 0.5) * px)
    path = _write_dsm(tmp_path / "dsm.tif", GROUND + 3.0 + roof(e, n), x0, y0, px, px, geo.epsg(zone, north))
    dsm = mosaic.read_dsm(path)
    plane = mosaic.build(frames, heights="plane")
    m = mosaic.build(frames, dsm=dsm)
    h = m.report["heights"]
    assert h["method"] == "DSM" and h["file"] == "dsm.tif" and h["anchors"] == 9
    assert h["offset_m"] == pytest.approx(-3.0, abs=0.05) and h["anchor_residual_m"] < 0.1
    assert m.report["alignment"]["heights"] == h
    on_roof, beside = m.heights.at_points(centre[0] + np.array([-1.0, 9.0]), centre[1] + np.array([1.0, 9.0]))
    assert on_roof == pytest.approx(2.5, abs=0.3) and abs(beside) < 0.1
    assert m.report["misalignment"]["median_m"] < 0.5 * plane.report["misalignment"]["median_m"]
    # an offset given is used as it is, and without alignment the report is still there
    given = mosaic.build(frames, align=False, dsm=dsm, dsm_offset=-2.0)
    assert given.report["alignment"] is None
    assert given.report["heights"]["offset_m"] == -2.0 and given.report["heights"]["offset_source"] == "given"
    assert given.heights.at_points(centre[0:1] - 1.0, centre[1:2] + 1.0)[0] == pytest.approx(3.5, abs=0.3)
    far = mosaic.read_dsm(_write_dsm(tmp_path / "far.tif", np.zeros((2, 2)), x0 + 1000, y0, px, px, 32616))
    with pytest.raises(ValueError, match="does not cover the flight"):
        mosaic.build(frames, align=False, dsm=far)


def test_dsm_options_are_checked(tmp_path):
    with pytest.raises(api.SettingsError, match="dsm needs heights 'auto'"):
        api.MosaicOptions(dsm=tmp_path, heights="plane")
    with pytest.raises(api.SettingsError, match="no such file"):
        api.MosaicOptions(dsm=tmp_path / "missing.tif")
    with pytest.raises(api.SettingsError, match="dsm_offset needs a dsm"):
        api.MosaicOptions(dsm_offset=1.0)
    (tmp_path / "x.tif").write_bytes(b"not a tiff")
    assert api.MosaicOptions(dsm=str(tmp_path / "x.tif")).dsm == tmp_path / "x.tif"
    with pytest.raises(api.SettingsError, match="dsm: .*not a readable GeoTIFF"):
        api._height_options(api.MosaicOptions(dsm=tmp_path / "x.tif"))


def test_heights_interpolate_bilinearly_between_cell_centres():
    """Heights.at and point_weights agree, give each cell's value at its centre, interpolate
    between centres, and hold the edge value beyond the outer centres."""
    hs = mosaic.Heights(100.0, 200.0, 2.0, np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
    e = np.array([101.0, 102.0, 103.0, 99.0, 108.0])     # centres at E 101, 103, 105
    n = np.array([199.0, 198.0, 197.0, 201.0, 190.0])    # centres at N 199, 197
    np.testing.assert_allclose(hs.at_points(e, n), [0.0, 2.0, 4.0, 0.0, 5.0])
    idx, w = hs.point_weights(e, n)
    np.testing.assert_allclose((hs.z.ravel()[idx] * w).sum(1), hs.at_points(e, n))
    np.testing.assert_allclose(w.sum(1), 1.0)
    np.testing.assert_allclose(hs.at(np.array([101.0, 104.0]), np.array([199.0, 198.0])), [[0.0, 1.5], [1.5, 3.0]])


def test_conjugate_gradients_solve_the_normal_equations_as_the_dense_solver_does():
    rng = np.random.default_rng(3)
    m, rows = 60, 400
    idx = rng.integers(0, m, (rows, 4))
    val = rng.normal(size=(rows, 4))
    ww = rng.uniform(0.2, 2.0, rows)
    diag = np.full(m, 0.01)
    y = rng.normal(size=m)
    a = np.zeros((rows, m))
    np.add.at(a, (np.repeat(np.arange(rows), 4), idx.ravel()), val.ravel())
    dense = np.linalg.solve(a.T @ (ww[:, None] * a) + np.diag(diag), y)
    assert np.allclose(mosaic._cg(idx, val, ww, diag, y, np.zeros(m)), dense, atol=1e-6)


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
    assert {r for s, _, r in runs if s == "align"} == {1, 2, 3, 4}
    assert all(extra == {"round": extra["round"], "rounds": 4} for s, *_, extra in events if s == "align")
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



def test_build_both_gives_the_normalized_and_the_as_converted_mosaic_from_one_layout(synthetic):
    """build_both gives the normalized mosaic of build(normalize=True), and the other frames
    composited as they are over the same layout (as build(normalize=False) without alignment);
    a frame the other set lacks is left out of its mosaic only."""
    frames, _ = synthetic
    drift = [0.0, 0.8, -0.8, 0.4, 0.0, -0.4]
    other = [mosaic.Frame(f.name, f.temperature + np.float32(d), f.pose) for f, d in zip(frames, drift)]
    for align in (False, True):
        normalized, converted = mosaic.build_both(frames, other, align=align)
        alone = mosaic.build(frames, align=align, normalize=True)
        np.testing.assert_array_equal(normalized.temperature, alone.temperature)
        assert normalized.report == alone.report
        assert converted.report["misalignment"] == normalized.report["misalignment"]
        assert not converted.report["frame_offsets"]["normalized"]
    normalized, converted = mosaic.build_both(frames, other, align=False)
    as_built = mosaic.build(other, align=False)
    np.testing.assert_array_equal(converted.temperature, as_built.temperature)
    assert converted.frames == as_built.frames
    _, partial = mosaic.build_both(frames, other[:5], align=False)
    assert [f["file"] for f in partial.frames] == [f.name for f in frames[:5]]


def test_build_both_reports_compositing_both_mosaics_as_one_stage(synthetic):
    frames, _ = synthetic
    events = []
    mosaic.build_both(frames, frames, align=False,
                      progress=lambda stage, done, total, unit, **extra: events.append((stage, done, total)))
    assert list(dict.fromkeys(s for s, *_ in events)) == ["composite", "check"]
    composite = [(d, t) for s, d, t in events if s == "composite"]
    assert composite[0] == (0, 12) and composite[-1] == (12, 12)
    assert [d for d, _ in composite] == sorted(d for d, _ in composite)
