"""The command line end to end: its outputs must be what the native SDK produces."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from dji_thermal_cli.cli import _palette

from .conftest import SAMPLES, native_measure, native_process, sample_files

FILES = sample_files()


def _cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "dji_thermal_cli", *map(str, args)], capture_output=True, text=True)


@pytest.mark.parametrize("settings", [None, (15.0, 45.0, 6)], ids=["defaults", "fixed-range"])
@pytest.mark.native
@pytest.mark.samples
def test_process_matches_native(settings, tmp_path):
    extra = [] if settings is None else ["--range-min", settings[0], "--range-max", settings[1], "--palette", settings[2]]
    out = tmp_path / "out"
    proc = _cli("process", "--input-dir", SAMPLES, "--output-dir", out, *extra)
    assert proc.returncode == 0, proc.stderr
    assert f"Done. {len(FILES)}/{len(FILES)} candidate files processed." in proc.stdout
    steps = [] if settings is None else [
        {"op": "set_cb", "flag": True, "high": settings[1], "low": settings[0]},
        {"op": "set_palette", "v": settings[2]},
    ]
    for src in FILES:
        rel = src.relative_to(SAMPLES)
        native = native_process(src, steps + [{"op": "process"}], tmp_path)
        if native is None:
            continue                     # the native SDK crashed on this file
        code, rgb = native[-1]
        assert code == 0
        png = np.asarray(Image.open(out / rel.parent / f"{rel.stem}_pseudo.png"))
        assert np.array_equal(png, rgb), rel
        temps, _, code = native_measure(src)
        if code == 0:
            tiff = np.asarray(Image.open(out / rel.parent / f"{rel.stem}_temp_c.tiff"))
            assert np.array_equal(tiff.view(np.uint32), temps.view(np.uint32)), rel


@pytest.mark.native
@pytest.mark.samples
def test_list():
    proc = _cli("list", "--input-dir", SAMPLES, "--format", "json")
    assert proc.returncode == 0, proc.stderr
    rows = json.loads(proc.stdout)
    assert len(rows) == len(FILES)
    assert all(row["thermal"]["palette"]["name"] and row["thermal"]["color_bar"] is not None for row in rows)


def test_not_an_rjpeg(tmp_path):
    bad = tmp_path / "x.jpg"
    bad.write_bytes(b"not a jpeg")
    proc = _cli("process", "--input-dir", bad, "--output-dir", tmp_path / "out", "--jobs", "1")
    assert "[skip, not a JPEG file]" in proc.stdout
    assert not (tmp_path / "out" / "x_temp_c.tiff").exists()
    proc = _cli("list", "--input-dir", bad)
    assert "[skip, not a JPEG file (by content)]" in proc.stderr


@pytest.mark.parametrize("args, message", [
    (["--range-min", "10"], "--range-min and --range-max must be given together"),
    (["--range-min", "40", "--range-max", "30"], "--range-max must be greater than --range-min"),
])
def test_bad_range(args, message, tmp_path):
    proc = _cli("process", "--input-dir", tmp_path, "--output-dir", tmp_path / "out", *args)
    assert proc.returncode != 0 and message in proc.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("text, index", [("3", 3), ("hotiron", 3), ("HotIron", 3), ("0", 0), ("blackhot", 9)])
def test_palette_by_number_or_name(text, index):
    assert _palette(text) == index


@pytest.mark.parametrize("text", ["10", "-1", "ironbow", ""])
def test_palette_rejected(text):
    with pytest.raises(argparse.ArgumentTypeError, match="must be 0-9 or one of whitehot"):
        _palette(text)


@pytest.mark.samples
def test_palette_name_same_as_number(tmp_path):
    src = FILES[0]
    made = []
    for palette in ("3", "hotiron"):
        out = tmp_path / palette
        proc = _cli("process", "--input-dir", src, "--output-dir", out, "--palette", palette, "-o", "pseudo", "--jobs", "1")
        assert proc.returncode == 0, proc.stderr
        made.append(np.asarray(Image.open(next(out.rglob("*_pseudo.png")))))
    np.testing.assert_array_equal(*made)
