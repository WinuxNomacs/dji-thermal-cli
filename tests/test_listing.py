"""`list`: every group per file, in each output format and unit."""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest

from dji_thermal_cli import djixmp, listing, tsdk, units

from .conftest import flight_files, sample
from .test_cli import _cli


def _flight_file():
    return flight_files()[0]


def _sample(camera):
    return sample(camera)


@pytest.mark.flight
def test_groups_match_the_file():
    src = _flight_file()
    info, status = listing.describe(src, exif_all=True)
    assert status is None and set(info) == set(listing.GROUPS)
    xmp = djixmp.fields(src.read_bytes())
    geo, att, cam = info["geospatial"], info["attitude"], info["camera"]
    assert geo["latitude"] == float(xmp["GpsLatitude"]) and geo["longitude"] == float(xmp["GpsLongitude"])
    assert geo["rangefinder"]["distance_m"] == float(xmp["LRFTargetDistance"])
    assert geo["rtk_std_m"]["hgt"] == float(xmp["RtkStdHgt"])
    assert att["gimbal_pitch"] == float(xmp["GimbalPitchDegree"]) and att["flight_yaw"] == float(xmp["FlightYawDegree"])
    assert cam["model"] == "M4T" and cam["focal_px"] == 1000.0 and cam["camera_serial"] == xmp["CameraSerialNumber"]
    assert info["exif"]["xmp"]["drone-dji:GimbalPitchDegree"] == xmp["GimbalPitchDegree"]
    assert info["exif"]["exif"]["FocalLength"] == 12.0 and info["exif"]["ifd0"]["Model"] == "M4T"


@pytest.mark.flight
def test_stats_are_the_sdk_measurement():
    src = _flight_file()
    t = tsdk.open(src.read_bytes()).measure_temperature_c()
    stats = listing.describe(src)[0]["temperature"]
    assert stats["min"] == float(str(t.min())) and stats["max"] == float(str(t.max()))
    assert stats["mean"] == pytest.approx(float(t.astype(np.float64).mean()), rel=1e-6)
    assert stats["valid_pixels"] == t.size
    assert listing.describe(src, stats=False)[0]["temperature"] is None


@pytest.mark.parametrize("unit", ["F", "K"])
@pytest.mark.flight
def test_units(unit):
    src = _flight_file()
    c, other = listing.describe(src)[0], listing.describe(src, unit)[0]
    assert other["temperature"]["max"] == pytest.approx(units.from_celsius(c["temperature"]["max"], unit), abs=1e-4)
    mp_c, mp = c["thermal"]["measurement_params"], other["thermal"]["measurement_params"]
    assert mp["reflection"] == pytest.approx(units.from_celsius(mp_c["reflection"], unit), abs=1e-4)
    assert mp["emissivity"] == mp_c["emissivity"] and mp["distance"] == mp_c["distance"]


@pytest.mark.parametrize("camera", ["XTS", "H20T"])
@pytest.mark.samples
def test_no_ambient_parameter(camera):
    """dirp cameras have no ambient temperature: it's absent, not a converted 0 C."""
    info = listing.describe(_sample(camera), "F")[0]
    assert info["thermal"]["measurement_params"]["ambient_temp"] is None
    assert info["thermal"]["measurement_params_range"]["ambient_temp"] is None


@pytest.mark.samples
def test_missing_metadata_is_none():
    """The H20T sample has no GPS or DJI XMP: those fields are None, the rest is filled."""
    info = listing.describe(_sample("H20T"))[0]
    assert all(v is None for v in info["geospatial"].values())
    assert all(v is None for v in info["attitude"].values())
    assert info["camera"]["model"] == "ZH20T" and info["temperature"]["valid_pixels"] > 0


def test_rational_over_zero_is_missing():
    """Pillow reads an EXIF rational over 0 as NaN, which float() raises for before Python 3.14."""
    from PIL.TiffImagePlugin import IFDRational

    from dji_thermal_cli import mosaic

    nan = IFDRational(1, 0)
    assert listing._plain(nan) is None and listing._num(nan) is None
    assert listing._dms((nan, IFDRational(0), IFDRational(0)), "N") is None
    assert listing._dms((IFDRational(45), IFDRational(30), IFDRational(0)), "S") == -45.5
    assert mosaic._exif_number(nan) is None and mosaic._exif_number(IFDRational(12)) == 12.0


@pytest.mark.flight
def test_cli_formats():
    src = _flight_file()
    table = _cli("list", "--input-dir", src)
    assert table.returncode == 0 and "gimbal_pitch" in table.stdout and "max_c" in table.stdout
    rows = json.loads(_cli("list", "--input-dir", src, "--format", "json", "--unit", "K").stdout)
    assert len(rows) == 1 and "exif" not in rows[0] and rows[0]["temperature"]["max"] > 273
    out = _cli("list", "--input-dir", src, "--format", "csv", "--exif", "all").stdout
    records = list(csv.DictReader(io.StringIO(out)))
    assert len(records) == 1 and records[0]["attitude.gimbal_pitch"] == "-90.0"
    assert "exif.xmp.drone-dji:UTCAtExposure" in records[0]


@pytest.mark.samples
def test_skips_summarised(tmp_path):
    (tmp_path / "a.jpg").write_bytes(sample("M4T").read_bytes())
    from PIL import Image
    Image.new("RGB", (8, 8)).save(tmp_path / "visual.jpg")
    (tmp_path / "broken.jpg").write_bytes((tmp_path / "a.jpg").read_bytes()[:300_000])
    proc = _cli("list", "--input-dir", tmp_path)
    assert "1 ordinary JPEG(s) without thermal data skipped" in proc.stderr
    assert "1 problem file(s) skipped" in proc.stderr and "broken.jpg" in proc.stderr
    assert "visual.jpg" not in proc.stderr
