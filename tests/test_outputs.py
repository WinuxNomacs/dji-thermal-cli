"""Every output product in every format, read back with independent readers (tifffile,
Pillow, OpenEXR), must hold exactly the port's arrays and carry the source metadata."""

from __future__ import annotations


import numpy as np
import OpenEXR
import pytest
import tifffile
from PIL import Image

from dji_thermal_cli import core, outputs, tsdk

from .conftest import flight_files, sample
from .test_cli import _cli

CAMERAS = ("M4T", "M30T", "H20T", "XTS", "M3TD")
ALL_OUTPUTS = [(p, f) for p, prod in outputs.PRODUCTS.items() for f in prod.formats] + [("temp_c", "tif")]


def _sample(camera: str):
    return sample(camera)


def _read(path):
    if path.suffix == ".exr":
        f = OpenEXR.File(str(path))
        return f.channels()["Y"].pixels, f.header()
    if path.suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            return page.asarray(), {"gps": "GPSTag" in page.tags, "xmp": page.tags["XMP"].value if "XMP" in page.tags else b"",
                                    "compression": int(page.compression)}
    with Image.open(path) as im:
        return np.asarray(im), {**im.info, "gps": 34853 in im.getexif()}


def _expected(src, range_c=None):
    rj = tsdk.open(src.read_bytes())
    t = rj.measure_temperature_c()
    lo, hi = (float(t.min()), float(t.max())) if range_c is None else range_c
    if range_c is not None:
        rj.set_color_bar(tsdk.ColorBar(True, hi, lo))
    return {
        "temp_c": t,
        "temp_f": (t.astype(np.float64) * 9 / 5 + 32).astype(np.float32),
        "temp_k": (t.astype(np.float64) + 273.15).astype(np.float32),
        "temp_ck": np.rint((t.astype(np.float64) + 273.15) * 100).astype(np.uint16),
        "temp_dc": rj.measure(),
        "raw": rj.get_original_raw(),
        "gray16": np.clip(np.rint((t.astype(np.float64) - lo) / (hi - lo) * 65535), 0, 65535).astype(np.uint16),
        "pseudo": rj.process(),
    }


@pytest.mark.parametrize("compression", outputs.COMPRESSIONS)
@pytest.mark.parametrize("range_c", [None, (10.0, 40.0)], ids=["auto", "fixed"])
@pytest.mark.parametrize("camera", CAMERAS)
@pytest.mark.samples
def test_all_outputs(camera, range_c, compression, tmp_path):
    src = _sample(camera)
    lo, hi = range_c or (None, None)
    r = core.process_file(src, tmp_path, src.parent, core.MeasurementOverrides(use_rangefinder=False), lo, hi, None,
                          tuple(ALL_OUTPUTS), compression)
    assert r.ok and r.notes == [], (r.reason, r.notes)
    assert [p.name for p in r.outputs] == [f"{src.stem}_{p}.{f}" for p, f in ALL_OUTPUTS]
    expected = _expected(src, range_c)
    with Image.open(src) as im:
        source_gps = 34853 in im.getexif()
    for (product, _), path in zip(ALL_OUTPUTS, r.outputs):
        pixels, meta = _read(path)
        want = expected[product]
        assert pixels.dtype == want.dtype and np.array_equal(pixels, want), path.name
        if path.suffix == ".exr":
            assert meta["units"] == {"temp_c": "Celsius", "temp_f": "Fahrenheit", "temp_k": "Kelvin"}[product]
            assert str(meta["compression"]).endswith("ZIP_COMPRESSION" if compression == "deflate" else "NO_COMPRESSION")
            continue
        assert meta["gps"] == source_gps, path.name
        if path.suffix in (".tif", ".tiff"):
            assert meta["compression"] == (8 if compression == "deflate" else 1), path.name
        xmp = meta["xmp"] if "xmp" in meta else meta.get("XML:com.adobe.xmp", "").encode()
        assert (b'BandName="LWIR"' in xmp) == (product != "pseudo"), path.name


@pytest.mark.parametrize("shape, dtype", [((509, 637), np.float32), ((509, 637), np.uint16), ((3, 5), np.int16),
                                          ((300, 7, 3), np.uint8), ((1, 1), np.uint16)])
def test_tiff_strips_round_trip(shape, dtype, tmp_path):
    """Partial last strips, the 16-bit predictor's wrap-around and tiny images."""
    info = np.iinfo(dtype) if dtype != np.float32 else None
    rng = np.random.default_rng(0)
    a = (rng.integers(info.min, info.max, shape, endpoint=True) if info else rng.standard_normal(shape)).astype(dtype)
    for compression in outputs.COMPRESSIONS:
        path = tmp_path / f"{compression}.tiff"
        core.tiffout.write_tiff(path, a, None, compression=compression)
        assert np.array_equal(tifffile.imread(path), a) and np.array_equal(np.asarray(Image.open(path)), a)


@pytest.mark.parametrize("shape", [(509, 637), (3, 5), (16, 4), (17, 1)])
def test_exr_chunks_round_trip(shape, tmp_path):
    a = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    for compression in ("zip", "none"):
        path = tmp_path / f"{compression}.exr"
        core.outputs.exrout.write_exr(path, a, compression=compression)
        assert np.array_equal(OpenEXR.File(str(path)).channels()["Y"].pixels, a)


def test_centikelvin_clip_and_no_data():
    t = np.array([[-273.15, 0.0, 382.2, 400.0]], np.float32)
    v, clipped = outputs.centikelvin(t)
    assert v.tolist() == [[0, 27315, 65535, 65535]] and clipped == 1


def test_gray16_flat_image():
    v, lo, hi = outputs.gray16(np.full((2, 2), 20.0, np.float32), None, None)
    assert (v == 0).all() and lo == hi == 20.0


@pytest.mark.parametrize("spec, expected", [
    ("temp_c", ("temp_c", "tiff")), ("temp_c:tif", ("temp_c", "tif")), ("temp_c:EXR", ("temp_c", "exr")),
    ("gray16", ("gray16", "png")), ("pseudo:tiff", ("pseudo", "tiff")),
])
def test_parse_output(spec, expected):
    assert outputs.parse_output(spec) == expected


@pytest.mark.parametrize("spec, message", [
    ("heat", "unknown product 'heat'"), ("temp_dc:png", "temp_dc can be written as tiff/tif, not 'png'"),
    ("temp_c:jpg", "temp_c can be written as tiff/tif, exr, not 'jpg'"),
])
def test_bad_output(spec, message, tmp_path):
    proc = _cli("process", "--input-dir", tmp_path, "--output-dir", tmp_path / "out", "-o", spec)
    assert proc.returncode != 0 and message in proc.stderr


@pytest.mark.samples
def test_cli_outputs(tmp_path):
    src = _sample("M4T")
    proc = _cli("process", "--input-dir", src, "--output-dir", tmp_path, "--jobs", "1",
                "-o", "temp_ck:png,raw", "--output", "temp_ck:png", "--compression", "none")
    assert proc.returncode == 0, proc.stderr
    with tifffile.TiffFile(tmp_path / f"{src.stem}_raw.tiff") as tf:
        assert int(tf.pages[0].compression) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{src.stem}_raw.tiff", f"{src.stem}_temp_ck.png"]


@pytest.mark.flight
def test_gps_carried(tmp_path):
    """A flight image's GPS survives into TIFF and PNG outputs."""
    src = flight_files()[0]
    r = core.process_file(src, tmp_path, src.parent, core.MeasurementOverrides(), None, None, None,
                          (("temp_c", "tiff"), ("temp_ck", "png"), ("pseudo", "png")))
    assert r.ok and all(_read(p)[1]["gps"] for p in r.outputs)


@pytest.mark.samples
def test_units_cli(tmp_path):
    """The same reflected temperature given in C, F or K measures identically, and temp_f /
    temp_k are temp_c converted."""
    src = _sample("M4T")
    made = {}
    for unit, value in (("C", "20"), ("F", "68"), ("K", "293.15")):
        out = tmp_path / unit
        proc = _cli("process", "--input-dir", src, "--output-dir", out, "--jobs", "1", "--input-unit", unit,
                    "--reflected-temp", value, "--ambient-temp", value, "-o", "temp_c,temp_f,temp_k:exr")
        assert proc.returncode == 0, proc.stderr
        made[unit] = {p: _read(out / f"{src.stem}_{p}.{ext}")[0] for p, ext in
                      (("temp_c", "tiff"), ("temp_f", "tiff"), ("temp_k", "exr"))}
    for unit in "FK":
        for product in made["C"]:
            np.testing.assert_array_equal(made[unit][product], made["C"][product])
    c = made["C"]["temp_c"].astype(np.float64)
    np.testing.assert_array_equal(made["C"]["temp_f"], (c * 9 / 5 + 32).astype(np.float32))
    np.testing.assert_array_equal(made["C"]["temp_k"], (c + 273.15).astype(np.float32))


@pytest.mark.samples
def test_below_absolute_zero(tmp_path):
    src = _sample("M4T")
    proc = _cli("process", "--input-dir", src, "--output-dir", tmp_path, "--input-unit", "K", "--reflected-temp", "-1")
    assert proc.returncode != 0 and "below absolute zero" in proc.stderr


def test_write_is_all_or_nothing(tmp_path):
    """A write that fails part-way leaves the previous file (or none), never a partial one."""
    from dji_thermal_cli import files

    path = tmp_path / "x_temp_c.tiff"
    path.write_bytes(b"previous")
    with pytest.raises(RuntimeError), files.replaced_when_done(path) as tmp:
        tmp.write_bytes(b"half")
        raise RuntimeError("interrupted")
    assert path.read_bytes() == b"previous" and sorted(p.name for p in tmp_path.iterdir()) == ["x_temp_c.tiff"]
    with files.replaced_when_done(path) as tmp:
        tmp.write_bytes(b"new")
    assert path.read_bytes() == b"new" and sorted(p.name for p in tmp_path.iterdir()) == ["x_temp_c.tiff"]
