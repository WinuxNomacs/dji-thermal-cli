"""The visual mosaic: reading and pairing visual frames, placing them by the thermal mosaic
with the cameras' offset measured, on a synthetic flight whose visual camera is a picture of
the same known field, the GeoTIFF, and the API and command line on a flight."""

from __future__ import annotations

import dataclasses
import datetime
import io
import json
import math

import numpy as np
import pytest
import tifffile
from PIL import Image

import dji_thermal_cli as dtc
from dji_thermal_cli import geo, mosaic, visual

from .conftest import flight_counts, flight_dir, flight_visual
from .test_cli import _cli
from .test_mosaic import GROUND, LAT, LON, _field, _frame
from .test_rangefinder import _with_xmp, _xmp

OFFSET_DEG = (0.4, -0.3)        # the synthetic visual camera's pitch and yaw against the thermal one
START = datetime.datetime(2026, 10, 22, 21, 33, 12)


def _colour(t):
    """A picture of the temperature field: every channel follows it, so their edges agree."""
    s = np.clip((t - 15) / 15, 0, 1)
    return np.stack([60 + 180 * s, 40 + 120 * s, 200 - 150 * s], axis=-1)


def _visual_frame(path, thermal, offset_deg=OFFSET_DEG, size=(480, 360), focal=400.0, exposure=None, k1=0.0):
    """The visual frame taken with a synthetic thermal one: the same position and attitude, the
    camera turned by offset_deg, a wider lens (distorting by k1, see mosaic._Placed); saved as a
    PNG at path."""
    p = thermal.pose
    w, h = size
    zone, north = geo.utm_zone(LAT, LON)
    c = np.array(geo.utm(p.lat, p.lon, zone, north))
    j = geo.local_to_utm(p.lat, p.lon, zone, north)
    r = mosaic.rotation(p.yaw, p.pitch, p.roll) @ mosaic._boresight(*np.radians(offset_deg))
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x, y = mosaic._undistort((u - (w - 1) / 2) / focal, (v - (h - 1) / 2) / focal, k1)
    rays = np.tensordot(r, np.stack([np.ones((h, w)), x, y]), 1)
    t = (p.abs_alt - GROUND) / rays[2]
    e, n = c[:, None, None] + np.tensordot(j, np.stack([rays[1] * t, rays[0] * t]), 1)
    Image.fromarray(np.rint(_colour(_field(e, n))).astype(np.uint8)).save(path)
    pose = mosaic.Pose(p.lat, p.lon, p.abs_alt, p.yaw, p.pitch, p.roll, focal, w, h, p.ground_alt, None, {})
    return visual.VisualFrame(path.name, path, pose, exposure, None)


@pytest.fixture(scope="module")
def flight(tmp_path_factory):
    """Six thermal frames exposed 2 s apart, their mosaic, and the visual frames taken with them."""
    folder = tmp_path_factory.mktemp("visual")
    thermal = [_frame(f"f{i}_T", 4.0 * (i % 3), -4.0 * (i // 3), yaw=(5.4 if i < 3 else -174.6),
                      roll=(180.0 if i < 3 else 0.0)) for i in range(6)]
    for i, f in enumerate(thermal):
        f.pose.record["UTCAtExposure"] = (START + datetime.timedelta(seconds=2 * i)).isoformat()
    m = mosaic.build(thermal, align=False)
    frames = [_visual_frame(folder / f"f{i}_V.png", f, exposure=START + datetime.timedelta(seconds=2 * i, milliseconds=30))
              for i, f in enumerate(thermal)]
    return thermal, m, frames


def _colour_error(vm):
    """The median difference between the mosaic's colours and the field's, where it has any."""
    g = vm.grid
    e = g.e0 + (np.arange(g.width) + 0.5) * g.gsd
    n = g.n0 - (np.arange(g.height) + 0.5) * g.gsd
    truth = _colour(_field(*np.meshgrid(e, n)))
    covered = vm.source >= 0
    return float(np.median(np.abs(vm.rgb[covered].astype(float) - truth[covered])))


def test_the_visual_mosaic_lines_up_with_the_thermal_one(flight, monkeypatch):
    """The cameras' offset is measured from the pairs and corrected: the visual mosaic shows
    the field where it is, which it doesn't with the visual frames placed as the thermal ones."""
    _, m, frames = flight
    vm = visual.build(m, frames)
    off = vm.report["camera_offset"]
    assert off["applied"] and off["pairs_used"] >= 5
    assert off["boresight_deg"]["pitch"] == pytest.approx(OFFSET_DEG[0], abs=0.05)
    assert off["boresight_deg"]["yaw"] == pytest.approx(OFFSET_DEG[1], abs=0.05)
    assert off["residual_m"]["median"] < off["offset_before_m"]["median"] / 4
    assert vm.grid.gsd == pytest.approx(21.0 / 400, abs=0.001) and vm.grid.epsg == m.grid.epsg
    assert vm.report["frames_used"] == vm.report["frames_contributing"] == 6
    corrected = _colour_error(vm)
    monkeypatch.setattr(visual, "MIN_OFFSET_PAIRS", 100)
    uncorrected = visual.build(m, frames)
    assert not uncorrected.report["camera_offset"]["applied"]
    assert corrected < 3 and corrected < _colour_error(uncorrected) / 2


def test_the_visual_frames_take_the_thermal_frames_alignment(flight):
    """Each visual frame is shifted and rotated as its thermal frame was aligned, and sees the
    ground through the same surface."""
    _, m, frames = flight
    shifted = [dataclasses.replace(pl, shift=np.array([0.3, -0.2]), rot=0.002) for pl in m.placed]
    vm = visual.build(dataclasses.replace(m, placed=shifted), frames)
    for pl, row in zip(vm.placed, vm.frames):
        np.testing.assert_allclose(pl.shift, [0.3, -0.2])
        assert pl.rot == 0.002 and row["alignment_shift_m"] == [0.3, -0.2]


def test_any_number_of_threads_gives_the_same_mosaic(flight):
    _, m, frames = flight
    one, three = visual.build(m, frames, jobs=1), visual.build(m, frames, jobs=3)
    np.testing.assert_array_equal(one.rgb, three.rgb)
    np.testing.assert_array_equal(one.source, three.source)


def test_a_coarser_mosaic_decodes_its_jpegs_smaller(flight, tmp_path):
    """At four times the frames' pixel size, a JPEG is decoded at a quarter of its size, and the
    colours still come from where the frame saw them."""
    _, m, frames = flight
    jpegs = []
    for f in frames:
        with Image.open(f.path) as im:
            im.save(tmp_path / (f.path.stem + ".jpg"), quality=95)
        jpegs.append(dataclasses.replace(f, path=tmp_path / (f.path.stem + ".jpg")))
    assert visual._scale(visual.build(m, jpegs[:1]).placed[0], 4 * 21.0 / 400) == 4
    image, factor = visual._decode(jpegs[0], 4)
    assert image.shape == (90, 120, 3) and factor == 4
    coarse = visual.build(m, jpegs, gsd=4 * 21.0 / 400)
    assert coarse.grid.gsd == pytest.approx(4 * 21.0 / 400) and _colour_error(coarse) < 6


@pytest.fixture(scope="module")
def alone(flight, tmp_path_factory):
    """The flight's visual frames as JPEGs, their camera pointing where the gimbal says."""
    thermal, _, _ = flight
    folder = tmp_path_factory.mktemp("alone")
    frames = []
    for i, f in enumerate(thermal):
        v = _visual_frame(folder / f"f{i}_V.png", f, offset_deg=(0.0, 0.0))
        with Image.open(v.path) as im:
            im.save(folder / f"f{i}_V.jpg", quality=95)
        frames.append(dataclasses.replace(v, name=f"f{i}_V.jpg", path=folder / f"f{i}_V.jpg"))
    return frames


def test_a_flight_of_visual_frames_alone_is_laid_out_by_its_own_overlaps(alone):
    """Without thermal frames, the visual frames are aligned on their brightness, and the mosaic
    shows the field where it is."""
    vm = visual.build_alone(alone, heights="plane")
    assert vm.report["frames_used"] == vm.report["frames_contributing"] == 6
    assert vm.report["work_size"] == [480, 360] and vm.report["frames_oblique"] == []
    assert vm.report["misalignment"]["median_m"] < 0.02
    assert vm.grid.gsd == pytest.approx(21.0 / 400, abs=0.001)
    assert _colour_error(vm) < 4


def test_visual_frames_alone_are_aligned_smaller_and_composited_at_full_size(alone):
    """work_px caps the size the frames are matched at; the mosaic keeps their own pixel size."""
    vm = visual.build_alone(alone, heights="plane", work_px=240)
    assert vm.report["work_size"] == [240, 180]
    assert vm.grid.gsd == pytest.approx(21.0 / 400, abs=0.001) and _colour_error(vm) < 4
    assert all(pl.frame.pose.width == 480 for pl in vm.placed)


def test_a_wide_lens_distortion_is_found_and_corrected(flight, tmp_path):
    """Frames from a lens distorting by k1 = 0.08: the alignment finds it, and the mosaic shows
    the field where it is, which it doesn't with the distortion left unmodelled."""
    thermal, _, _ = flight
    frames = [_visual_frame(tmp_path / f"f{i}_V.png", f, offset_deg=(0.0, 0.0), k1=0.08) for i, f in enumerate(thermal)]
    vm = visual.build_alone(frames, heights="plane")
    assert vm.report["alignment"]["distortion_k1"] == pytest.approx(0.08, abs=0.02)
    assert all(pl.k1 == pytest.approx(0.08, abs=0.02) for pl in vm.placed)
    plain = visual.build_alone(frames, heights="plane", distortion=False)
    assert "distortion_k1" not in plain.report["alignment"]
    assert _colour_error(vm) < 4 and _colour_error(vm) < _colour_error(plain)
    assert vm.report["misalignment"]["median_m"] < plain.report["misalignment"]["median_m"]


def test_a_flight_of_its_own_estimates_its_heights_on_cells_half_the_thermal_mosaics(flight, tmp_path):
    """build_alone's height cells are ALONE_HEIGHT_CELL_PX work pixels wide, half the thermal
    mosaics' unless asked otherwise."""
    thermal, _, _ = flight
    frames = [_visual_frame(tmp_path / f"f{i}_V.png", f, offset_deg=(0.0, 0.0)) for i, f in enumerate(thermal)]
    fine = visual.build_alone(frames, check_misalignment=False)
    coarse = visual.build_alone(frames, check_misalignment=False, height_cell=mosaic.HEIGHT_CELL_PX)
    assert visual.ALONE_HEIGHT_CELL_PX == mosaic.HEIGHT_CELL_PX // 2
    assert fine.report["heights"]["cell_m"] == pytest.approx(coarse.report["heights"]["cell_m"] / 2, rel=0.01)


def test_without_distortion_frames_are_projected_as_before(flight):
    """k1 = 0 (the thermal mosaics') leaves the projection as the pinhole's, to the bit."""
    _, m, _ = flight
    pl = m.placed[0]
    e, n = np.meshgrid(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7))
    plain = mosaic._project(pl, e, n, pl.h)
    assert pl.k1 == 0.0
    distorted = mosaic._project(dataclasses.replace(pl, k1=0.05), e, n, pl.h)
    for a, b in zip(plain, mosaic._project(dataclasses.replace(pl, k1=0.0), e, n, pl.h)):
        np.testing.assert_array_equal(a, b)
    assert not np.allclose(plain[0], distorted[0])
    x, y = np.array([0.1, -0.3]), np.array([0.2, 0.25])
    assert mosaic._undistort(x, y, 0.0) == (x, y)
    xu, yu = mosaic._undistort(x, y, 0.05)
    np.testing.assert_allclose(xu * (1 + 0.05 * (xu ** 2 + yu ** 2)), x, atol=1e-9)


def test_a_cell_s_rangefinder_prior_is_the_median_of_the_nearest_heights():
    hs = mosaic.Heights(0.0, 10.0, 5.0, np.zeros((2, 2)))       # cell centres (2.5, 7.5), (7.5, 7.5), (2.5, 2.5), (7.5, 2.5)
    anchors = np.array([[2.0, 8.0, -5.0], [3.0, 7.0, -5.5], [2.5, 9.0, -6.0], [8.0, 2.0, 1.0], [7.0, 3.0, 0.0],
                        [9.0, 3.0, 0.5]])
    prior = mosaic._anchor_surface(hs, anchors, k=3)
    assert prior[0, 0] == -5.5 and prior[1, 1] == 0.5


def test_oblique_visual_frames_are_left_out_of_a_flight_alone(alone):
    frames = alone
    tilted = dataclasses.replace(frames[0], pose=dataclasses.replace(frames[0].pose, pitch=-60.0))
    vm = visual.build_alone([tilted] + frames[1:], heights="plane")
    assert vm.report["frames_oblique"] == [tilted.name] and vm.report["frames_used"] == 5
    with pytest.raises(ValueError, match="pitched"):
        visual.build_alone([tilted])


def test_pairs_are_the_frames_exposed_together(flight, tmp_path):
    """A visual frame pairs with the thermal frame exposed closest to it, within 0.25 s, once;
    without exposure times, with the same capture number in the same folder."""
    thermal, _, frames = flight
    later = dataclasses.replace(frames[0], name="late", exposure=frames[0].exposure + datetime.timedelta(seconds=0.2))
    far = dataclasses.replace(frames[1], name="far", exposure=frames[1].exposure + datetime.timedelta(seconds=0.5))
    pairs = visual.pair(thermal, [later, frames[0], far] + frames[2:])
    assert pairs["f0_T"].name == "f0_V.png" and "f1_T" not in pairs
    assert [pairs[f"f{i}_T"].name for i in range(2, 6)] == [f"f{i}_V.png" for i in range(2, 6)]

    def named(name, exposure=None):
        return mosaic.Frame(name, np.zeros((0, 0)), mosaic.Pose(0, 0, 0, 0, 0, 0, 1, 1, 1, None, None,
                                                                {} if exposure is None else {"UTCAtExposure": exposure}))
    t = [named("a/DJI_20261022163317_0024_T.JPG"), named("b/DJI_20261022163317_0024_T.JPG")]
    v = [visual.VisualFrame(n, tmp_path, t[0].pose, None, visual._capture(n))
         for n in ("a/DJI_20261022163316_0024_V.JPG", "a/DJI_20261022163318_0025_V.JPG")]
    assert {k: p.name for k, p in visual.pair(t, v).items()} == {t[0].name: v[0].name}


def _jpeg(xmp_body, model="M4T", size=(64, 48)):
    exif = Image.Exif()
    exif[272] = model
    exif.get_ifd(0x8769).update({37386: 6.72, 41989: 24})
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, "JPEG", exif=exif)
    return buf.getvalue() if xmp_body is None else _with_xmp(buf.getvalue(), _xmp(xmp_body))


POSE = ('drone-dji:GpsLatitude="42.1" drone-dji:GpsLongitude="-88.1" drone-dji:AbsoluteAltitude="250" '
        'drone-dji:GimbalYawDegree="0" drone-dji:GimbalPitchDegree="-90" drone-dji:GimbalRollDegree="0" '
        'drone-dji:UTCAtExposure="2026-10-22T21:33:12.242196" ')


def test_a_visual_frame_is_posed_at_its_own_size_with_the_35mm_focal_length(tmp_path):
    """The thermal cameras' pixel pitch (SENSORS, by model) is not the visual camera's: its
    focal length is the 35 mm equivalent's, over the JPEG's own size."""
    path = tmp_path / "DJI_20261022163316_0024_V.JPG"
    path.write_bytes(_jpeg(POSE + 'drone-dji:ImageSource="WideCamera">'))
    v, why = visual.read_visual(path, "x/" + path.name)
    assert why == "" and (v.pose.width, v.pose.height) == (64, 48)
    assert v.pose.focal_px == pytest.approx(24 * math.hypot(64, 48) / mosaic.FULL_FRAME_DIAGONAL_MM)
    assert v.exposure == datetime.datetime(2026, 10, 22, 21, 33, 12, 242196) and v.capture == ("x", 24)
    assert v.pose.record["ImageSource"] == "WideCamera"


@pytest.mark.parametrize("data, why", [
    (_jpeg(POSE + 'drone-dji:ImageSource="ZoomCamera">'), "a zoom camera's image (ZoomCamera)"),
    (_jpeg(None), "no DJI XMP"),
    (_jpeg('drone-dji:GpsLatitude="42.1">'), "no camera pose in the DJI XMP"),
])
def test_zoom_frames_and_frames_without_a_pose_are_not_visual_frames(tmp_path, data, why):
    path = tmp_path / "x.JPG"
    path.write_bytes(data)
    v, reason = visual.read_visual(path, "x.JPG")
    assert v is None and reason.startswith(why)


def test_geotiff(flight, tmp_path):
    """Red, green, blue and alpha (0 where no frame covers), georeferenced, with the frames'
    models in the sidecar: they find a point in the visual frames that saw it."""
    _, m, frames = flight
    vm = visual.build(m, frames)
    tif, sidecar = visual.write(vm, tmp_path / "mosaic_visual.tif")
    with tifffile.TiffFile(tif) as tf:
        page = tf.pages[0]
        pixels = page.asarray()
        assert pixels.shape == (vm.grid.height, vm.grid.width, 4) and pixels.dtype == np.uint8
        assert page.tags["ExtraSamples"].value == (2,) and tf.geotiff_metadata["ProjectedCSTypeGeoKey"] == 32616
        assert page.tags["ModelTiepointTag"].value[3:5] == (vm.grid.e0, vm.grid.n0)
        for name in visual.BANDS:
            assert f">{name}<" in page.tags["GDAL_METADATA"].value
    np.testing.assert_array_equal(pixels[..., :3], vm.rgb)
    np.testing.assert_array_equal(pixels[..., 3] == 255, vm.source >= 0)
    info = json.loads(sidecar.read_text())
    assert [f["file"] for f in info["frames"]] == [f"f{i}_V.png" for i in range(6)]
    assert [f["thermal_file"] for f in info["frames"]] == [f"f{i}_T" for i in range(6)]
    models = mosaic.read_frame_models(sidecar)
    row, col = vm.grid.height // 2, vm.grid.width // 2
    lat, lon = geo.utm_inverse(vm.grid.e0 + (col + 0.5) * vm.grid.gsd, vm.grid.n0 - (row + 0.5) * vm.grid.gsd,
                               vm.grid.zone, vm.grid.north)
    assert int(vm.source[row, col]) in models.covering(float(lon), float(lat))
    path, reduction = visual.write_preview(vm, tmp_path / "preview.png", longest=100)
    with Image.open(path) as im:
        assert im.mode == "RGBA" and max(im.size) <= 100 and reduction == math.ceil(max(vm.grid.width, vm.grid.height) / 100)


def test_a_geotiff_without_location_has_no_position(flight, tmp_path):
    _, m, frames = flight
    tif, sidecar = visual.write(visual.build(m, frames), tmp_path / "mosaic_visual.tif", located=False)
    info = json.loads(sidecar.read_text())
    assert info["location_removed"] and "crs" not in info and "frame_models" not in info
    assert all("lat" not in f and "x_m" in f and "thermal_file" in f for f in info["frames"])
    with tifffile.TiffFile(tif) as tf:
        assert "ModelTiepointTag" not in tf.pages[0].tags


def test_options_are_checked():
    with pytest.raises(dtc.SettingsError, match="visual_gsd needs visual"):
        dtc.MosaicOptions(visual_gsd=0.01)
    with pytest.raises(dtc.SettingsError, match="positive"):
        dtc.MosaicOptions(visual=True, visual_gsd=0)
    assert dtc.MosaicOptions(visual_only=True, visual_gsd=0.01).visual_gsd == 0.01


@pytest.mark.parametrize("options, why", [
    ({"align": False}, "needs align"),
    ({"dsm": __file__}, "DSM is used for thermal mosaics only"),
])
def test_a_mosaic_of_the_visual_frames_alone_refuses_the_thermal_mosaics_options(options, why):
    with pytest.raises(dtc.SettingsError, match=f"visual_only: .*{why}"):
        dtc.MosaicOptions(visual_only=True, **options)


def _rgb_jpeg(path, thermal, exposure):
    """The visual frame taken with a synthetic thermal one, as the drone writes it: a JPEG
    whose EXIF and DJI XMP give its pose, focal length and rangefinder."""
    w, h = 480, 360
    v = _visual_frame(path.with_suffix(".png"), thermal, offset_deg=(0.0, 0.0), size=(w, h),
                      focal=24 * math.hypot(w, h) / mosaic.FULL_FRAME_DIAGONAL_MM)
    p = thermal.pose
    exif = Image.Exif()
    exif[272] = "M4T"
    exif.get_ifd(0x8769).update({37386: 6.72, 41989: 24})
    buf = io.BytesIO()
    with Image.open(v.path) as im:
        im.save(buf, "JPEG", quality=95, exif=exif)
    v.path.unlink()
    path.write_bytes(_with_xmp(buf.getvalue(), _xmp(
        f'drone-dji:GpsLatitude="{p.lat:.9f}" drone-dji:GpsLongitude="{p.lon:.9f}" '
        f'drone-dji:AbsoluteAltitude="{p.abs_alt:.3f}" drone-dji:GimbalYawDegree="{p.yaw}" '
        f'drone-dji:GimbalPitchDegree="{p.pitch}" drone-dji:GimbalRollDegree="{p.roll}" '
        f'drone-dji:LRFStatus="Normal" drone-dji:LRFTargetAbsAlt="{GROUND:.3f}" '
        f'drone-dji:UTCAtExposure="{exposure.isoformat()}" drone-dji:ImageSource="WideCamera">')))


@pytest.fixture(scope="module")
def rgb_flight(flight, tmp_path_factory):
    """A folder of the flight's visual frames alone, as JPEGs from the drone (no thermal frames)."""
    thermal, _, _ = flight
    folder = tmp_path_factory.mktemp("rgb")
    for i, f in enumerate(thermal):
        _rgb_jpeg(folder / f"DJI_2026102221331{i}_000{i}_V.JPG", f, START + datetime.timedelta(seconds=2 * i))
    return folder


def test_make_mosaic_lays_out_a_flight_without_thermal_frames_by_its_own_overlaps(rgb_flight, tmp_path):
    """A flight of visual frames only: make_mosaic makes their mosaic alone (no thermal mosaic),
    with its stages and its files, and the frames' lens distortion in its frame models."""
    events = []
    report = dtc.make_mosaic(rgb_flight, tmp_path, dtc.Settings(), dtc.MosaicOptions(heights="plane"),
                             progress=lambda e: e.kind == "mosaic" and events.append(e.data))
    assert report.visual_only and report.mosaic is None and report.preview is None and report.files == []
    assert report.alignment is None and (report.frames, report.not_thermal) == (0, 6)
    v = report.visual
    assert (v.frames, v.skipped) == (6, 0) and v.mosaic.report["laid_out"] == "by the visual frames' own overlaps"
    assert v.mosaic.report["frames_used"] == 6 and v.mosaic.report["misalignment"]["median_m"] < 0.02
    stages = list(dict.fromkeys(d["stage"] for d in events))
    assert stages == ["frames", "visual_frames", "visual_decode", "align", "check", "visual_composite", "write"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "mosaic_visual.json", "mosaic_visual.tif", "mosaic_visual_preview.png"]
    models = json.loads(v.sidecar.read_text())["frame_models"]["frames"]
    k1 = v.mosaic.report["alignment"]["distortion_k1"]
    assert all(f["lens_k1"] == pytest.approx(k1, abs=1e-5) for f in models)


def test_a_mosaic_of_the_visual_frames_alone_has_its_own_height_cells(rgb_flight, tmp_path):
    """visual_only's height cells are build_alone's unless height_cell says otherwise; a flight
    with thermal frames gets no thermal mosaic with it."""
    report = dtc.make_mosaic(rgb_flight, tmp_path / "a", options=dtc.MosaicOptions(visual_only=True))
    cell = report.visual.mosaic.report["heights"]["cell_m"]
    wide = dtc.make_mosaic(rgb_flight, tmp_path / "b", options=dtc.MosaicOptions(visual_only=True, height_cell=64))
    assert wide.visual.mosaic.report["heights"]["cell_m"] == pytest.approx(2 * cell, rel=0.01)
    assert report.visual_only and not list((tmp_path / "a").glob("mosaic_temp_*"))


def test_frame_models_find_a_point_through_the_lens_distortion(flight, tmp_path):
    """read_frame_models of a mosaic whose lens distorts finds a ground point at the pixel the
    mosaic projected it to."""
    thermal, _, _ = flight
    frames = [_visual_frame(tmp_path / f"f{i}_V.png", f, offset_deg=(0.0, 0.0), k1=0.08) for i, f in enumerate(thermal)]
    vm = visual.build_alone(frames, heights="plane", check_misalignment=False)
    _, sidecar = visual.write(vm, tmp_path / "mosaic_visual.tif")
    models = dtc.read_frame_models(sidecar)
    pl = vm.placed[0]
    g = vm.grid
    e, n = pl.c[0] + 1.5, pl.c[1] - 2.0
    lat, lon = geo.utm_inverse(np.array([e]), np.array([n]), g.zone, g.north)
    u, v, _, _ = mosaic._project(pl, np.array([e - pl.c[0]]), np.array([n - pl.c[1]]), pl.h)
    assert pl.k1 != 0 and models.pixels(lon, lat)[0, 0] == pytest.approx([u[0], v[0]], abs=0.01)


@pytest.mark.flight
def test_make_mosaic_with_the_visual_frames(tmp_path):
    """The visual mosaic of the flight's visual frames, its stages after the thermal mosaic's
    check, and its files; a flight without visual frames says so."""
    events = []
    report = dtc.make_mosaic(flight_dir(), tmp_path, dtc.Settings(),
                             dtc.MosaicOptions(align=False, visual=True, visual_gsd=0.03),
                             progress=lambda e: e.kind == "mosaic" and events.append(e.data))
    stages = list(dict.fromkeys(d["stage"] for d in events))
    if not flight_visual():
        assert report.visual is None and report.visual_note.startswith("no visual frames")
        return
    assert stages == ["frames", "composite", "check", "visual_frames", "visual_offset", "visual_composite", "write"]
    last = {d["stage"]: d for d in events}
    assert last["visual_frames"]["total"] == len(flight_visual()) and last["write"]["total"] == 4 + (report.heights is not None)
    v = report.visual
    assert v.frames + v.skipped == len(flight_visual()) and 0 < v.mosaic.report["frames_used"] <= flight_counts()[0]
    off = v.mosaic.report["camera_offset"]
    assert off["applied"] == (off["pairs_measured"] >= visual.MIN_OFFSET_PAIRS) and v.mosaic.grid.gsd == 0.03
    assert sorted(p.name for p in tmp_path.glob("mosaic_visual*")) == [
        "mosaic_visual.json", "mosaic_visual.tif", "mosaic_visual_preview.png"]


@pytest.mark.flight
def test_the_command_line_prints_the_visual_mosaic(tmp_path):
    proc = _cli("process", "--input-dir", flight_dir(), "--output-dir", tmp_path, "--mosaic", "--no-mosaic-align",
                "--quiet", "--mosaic-visual", "--mosaic-visual-gsd", "0.03")
    assert proc.returncode == 0, proc.stderr
    if not flight_visual():
        assert "Visual mosaic: none, no visual frames" in proc.stdout
        return
    assert "other JPEGs (not thermal R-JPEGs) for the visual mosaic" in proc.stdout
    assert "mosaic_visual.tif  (" in proc.stdout and "3.0 cm/px" in proc.stdout
    assert "cameras' offset: pitch " in proc.stdout or "cameras' offset: not corrected, " in proc.stdout


def test_the_command_line_mosaics_a_flight_without_thermal_frames(rgb_flight, tmp_path):
    proc = _cli("process", "--input-dir", rgb_flight, "--output-dir", tmp_path, "--mosaic", "--quiet",
                "--mosaic-heights", "plane")
    assert proc.returncode == 0, proc.stderr
    assert "Visual mosaic, laid out by the visual frames' own overlaps: 6 visual frames." in proc.stdout
    assert "mosaic_visual.tif  (" in proc.stdout and "lens distortion k1 " in proc.stdout
    assert "Mosaic: 0 nadir frames" not in proc.stdout


def test_the_command_line_refuses_a_visual_mosaic_alone_without_alignment(tmp_path):
    proc = _cli("process", "--input-dir", tmp_path, "--output-dir", tmp_path / "out", "--mosaic-visual-only",
                "--no-mosaic-align")
    assert proc.returncode != 0 and "--mosaic-visual-only: " in proc.stderr and "needs --mosaic-align" in proc.stderr


def test_the_visual_pixel_size_needs_the_visual_mosaic(tmp_path):
    proc = _cli("process", "--input-dir", tmp_path, "--output-dir", tmp_path / "out", "--mosaic",
                "--mosaic-visual-gsd", "0.02")
    assert proc.returncode != 0 and "--mosaic-visual-gsd needs visual or --mosaic-visual-only" in proc.stderr
