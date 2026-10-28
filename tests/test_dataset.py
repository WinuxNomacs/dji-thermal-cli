"""Files are found and classified by their content, never by their extension."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from dji_thermal_cli import dataset

from .conftest import sample
from .test_cli import _cli


def _thermal():
    return sample("M4T")


def _visual_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 80, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def _unsupported(data: bytes) -> bytes:
    """The thermal file with its APP4 plugin tag changed to one the SDK doesn't know."""
    i = data.index(b"\xff\xe4") + 4                    # APP4 payload
    assert data[i + 4:i + 8] == b"iirp"
    return data[:i + 4] + b"xirp" + data[i + 8:]


def _flir(model: str = "XT2") -> bytes:
    """A JPEG with FLIR's radiometric data (an APP1 "FLIR" segment), and its camera in Exif."""
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = "DJI", model
    Image.fromarray(np.zeros((64, 80, 3), dtype=np.uint8)).save(buf, "JPEG", exif=exif)
    data = buf.getvalue()
    flir = b"FLIR\0\x01\x00\x00FFF\0" + bytes(32)
    return data[:2] + b"\xff\xe1" + (len(flir) + 2).to_bytes(2, "big") + flir + data[2:]


@pytest.fixture
def mixed(tmp_path):
    src = _thermal()
    data = src.read_bytes()
    files = {
        "thermal_renamed.bin": data,                   # thermal, odd extension
        "thermal_no_extension": data,
        "DJI_0002_T.JPG": data,
        "notes.jpg": b"this is not an image\n",        # a .jpg that isn't a JPEG
        "video.MP4": b"\x00\x00\x00\x18ftypmp42",
        "visual_V.JPG": _visual_jpeg(),
        "cut_thermal.jpg": data[: len(data) // 3],     # raw thermal data cut short
        "cut_image.jpg": data[:800_000],               # metadata intact, main image data cut short
        "other_plugin.jpg": _unsupported(data),
        "DJI_0001_R.JPG": _flir(),                     # a Zenmuse XT2's radiometric JPEG
    }
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    return tmp_path


@pytest.mark.samples
def test_scan_by_content(mixed):
    found = dataset.scan(mixed)
    assert sorted(p.name for p in found.others) == ["notes.jpg", "video.MP4"]
    assert "thermal_renamed.bin" in {p.name for p in found.jpegs}
    assert "thermal_no_extension" in {p.name for p in found.jpegs}
    assert not found.unreadable


@pytest.mark.parametrize("name, kind", [
    ("thermal_renamed.bin", dataset.THERMAL), ("thermal_no_extension", dataset.THERMAL),
    ("DJI_0002_T.JPG", dataset.THERMAL), ("notes.jpg", dataset.NOT_JPEG), ("video.MP4", dataset.NOT_JPEG),
    ("visual_V.JPG", dataset.NOT_THERMAL), ("cut_thermal.jpg", dataset.CORRUPT),
    ("other_plugin.jpg", dataset.UNSUPPORTED), ("DJI_0001_R.JPG", dataset.UNSUPPORTED),
])
@pytest.mark.samples
def test_classify(mixed, name, kind):
    status = dataset.classify(mixed / name)
    assert status.kind == kind, status


def test_flir_is_unsupported_not_ordinary(tmp_path):
    """FLIR's radiometric JPEG (DJI's Zenmuse XT2) is a thermal image of a format not read here:
    unsupported, saying what it is, not an ordinary JPEG."""
    (tmp_path / "xt2.jpg").write_bytes(_flir())
    status = dataset.classify(tmp_path / "xt2.jpg")
    assert status.kind == dataset.UNSUPPORTED
    assert status.reason == "a FLIR radiometric JPEG (XT2), not a DJI R-JPEG format"
    (tmp_path / "plain.jpg").write_bytes(_visual_jpeg())
    assert dataset.classify(tmp_path / "plain.jpg").kind == dataset.NOT_THERMAL


@pytest.mark.samples
def test_decode_finds_damaged_image_data(mixed):
    """Decoding finds a cut inside the image data: a warning for a thermal R-JPEG (only its
    preview is damaged; DJI's own XT S sample DJI_0001_R.jpg is one), corrupt otherwise."""
    assert dataset.classify(mixed / "cut_image.jpg").warnings == []
    status = dataset.classify(mixed / "cut_image.jpg", decode=True)
    assert status.kind == dataset.THERMAL and "image data damaged" in status.warnings[0]
    assert dataset.classify(mixed / "DJI_0002_T.JPG", decode=True).warnings == []
    visual = (mixed / "visual_V.JPG").read_bytes()
    (mixed / "cut_visual.jpg").write_bytes(visual[:len(visual) // 2])
    assert dataset.classify(mixed / "cut_visual.jpg", decode=True).kind == dataset.CORRUPT


@pytest.mark.samples
def test_structure_problem():
    data = _thermal().read_bytes()
    assert dataset.structure_problem(data) == ""
    assert "runs past the end" in dataset.structure_problem(data[:400_000])
    end_of_app2 = 29366 + 178                          # a cut exactly between two segments
    assert "no image data" in dataset.structure_problem(data[:end_of_app2])


@pytest.mark.samples
def test_single_file_always_a_candidate(mixed):
    assert dataset.scan(mixed / "notes.jpg").jpegs == [mixed / "notes.jpg"]


@pytest.mark.samples
def test_process_by_content(mixed, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    proc = _cli("process", "--input-dir", mixed, "--output-dir", out, "--jobs", "1", "-o", "temp_c")
    assert proc.returncode == 0, proc.stderr
    assert "Ignored 2 non-JPEG file(s)" in proc.stdout
    made = sorted(p.name for p in out.iterdir())
    assert "thermal_renamed_temp_c.tiff" in made and "thermal_no_extension_temp_c.tiff" in made
    assert "[skip, unsupported R-JPEG" in proc.stdout and "visual_V.JPG" in proc.stdout


def test_output_clashes(tmp_path):
    """a.jpg / a.jpeg / A.JPG clash (names ignore case); other folders, and files that make no
    outputs, don't."""
    names = ["a.jpg", "A.JPG", "a.jpeg", "b.jpg", "sub/a.jpg", "c.jpg", "c.jpeg"]
    paths = [tmp_path / n for n in names]
    thermal = set(paths) - {tmp_path / "c.jpg"}
    clashes = dataset.output_clashes(paths, tmp_path, thermal.__contains__)
    keeper = min(tmp_path / "a.jpg", tmp_path / "A.JPG", tmp_path / "a.jpeg")
    assert clashes == {p: keeper for p in (tmp_path / "a.jpg", tmp_path / "A.JPG", tmp_path / "a.jpeg") if p != keeper}


@pytest.mark.samples
def test_clashing_outputs_skipped(tmp_path):
    data = _thermal().read_bytes()
    (tmp_path / "in").mkdir()
    for name in ("x.jpg", "x.jpeg"):
        (tmp_path / "in" / name).write_bytes(data)
    (tmp_path / "in" / "x.txt").write_text("not a JPEG")
    proc = _cli("process", "--input-dir", tmp_path / "in", "--output-dir", tmp_path / "out", "--jobs", "1", "-o", "temp_c")
    assert proc.returncode == 0, proc.stderr
    assert "its outputs would overwrite those of x.jpeg" in proc.stdout
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["x_temp_c.tiff"]
    check = _cli("check", "--input-dir", tmp_path / "in", "--format", "json")
    found = {f["code"]: f for f in json.loads(check.stdout)["findings"]}
    assert found["output_name_clash"]["files"] == [str(tmp_path / "in" / "x.jpg")]
