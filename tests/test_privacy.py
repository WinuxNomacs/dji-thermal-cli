"""Location-free outputs: no output of `process --strip-location` carries the source's
position, altitude or identifiers, in any encoding, and everything else is kept."""

from __future__ import annotations

import json
import re
import struct

import pytest
import tifffile
from PIL import Image

from dji_thermal_cli import djixmp, outputs, privacy, tiffout, xmpedit

from .conftest import flight_files
from .test_cli import _cli

ALL = ",".join(f"{p}:{f}" for p, prod in outputs.PRODUCTS.items() for f in prod.formats)


def _needles(src) -> dict[str, bytes]:
    """What must not survive, as the bytes it appears as in the source: XMP text values, the
    EXIF serial number, and the GPS IFD's raw values (rationals, altitude)."""
    data = src.read_bytes()
    xmp = djixmp.fields(data)
    out = {}
    for key, value in xmp.items():
        # values long enough to be specific: coordinates, altitudes, serial numbers (a status
        # such as "RTK" or a flag such as "50" would match unrelated bytes)
        if privacy.XMP_REMOVED.match(key) and len(value.strip("+- ")) >= 6:
            out[f"xmp {key}"] = value.encode()
            if key.startswith(("Gps", "LRFTarget")) and "." in value:     # the same number, shortened
                whole, frac = value.lstrip("+-").split(".")
                out[f"xmp {key} (4 decimals)"] = f"{whole}.{frac[:4]}".encode()
    exif = tiffout.read_metadata(data).exif_tiff
    bo = "<" if exif[:2] == b"II" else ">"
    blob = bytearray(exif)
    ifd0 = struct.unpack_from(bo + "I", blob, 4)[0]
    entries = {t: (ty, c, raw) for t, ty, c, raw in privacy._entries(blob, bo, ifd0)}
    gps = struct.unpack(bo + "I", entries[34853][2])[0]
    for tag, type_, count, raw in privacy._entries(blob, bo, gps):
        size = privacy._TYPE_SIZE[type_] * count
        if size >= 8:
            off = struct.unpack(bo + "I", raw)[0]
            out[f"exif GPS tag {tag}"] = bytes(blob[off:off + size])
    with Image.open(src) as im:
        serial = im.getexif().get_ifd(0x8769).get(0xA431)
    if serial:
        out["exif BodySerialNumber"] = str(serial).strip("\0 ").encode()
    return out


def _found(path, needles) -> list[str]:
    data = path.read_bytes()
    return [name for name, needle in needles.items() if needle in data]


def _in_text(text: str, needles) -> list[str]:
    """Needles that appear in console output (text needles only)."""
    return [name for name, needle in needles.items() if name.startswith("xmp") and needle.decode() in text]


@pytest.mark.flight
def test_outputs_free_of_location(tmp_path):
    src = flight_files()[0]
    needles = _needles(src)
    assert any("Gps" in n for n in needles) and any("Serial" in n for n in needles), needles
    plain = _cli("process", "--input-dir", src, "--output-dir", tmp_path / "plain", "-o", ALL, "--jobs", "1")
    stripped = _cli("process", "--input-dir", src, "--output-dir", tmp_path / "strip", "-o", ALL, "--jobs", "1",
                    "--strip-location")
    assert plain.returncode == 0 and stripped.returncode == 0, stripped.stderr
    assert "Location removed (--strip-location)" in stripped.stdout
    assert _in_text(stripped.stdout + stripped.stderr, needles) == []
    made = sorted((tmp_path / "strip").iterdir())
    assert len(made) == len(ALL.split(","))
    for path in made:
        assert _found(path, needles) == [], path.name
    # the needles do find the location in ordinary outputs that carry metadata
    assert all(_found(p, needles) for p in (tmp_path / "plain").iterdir() if p.suffix in (".tiff", ".png"))


@pytest.mark.flight
def test_everything_else_kept(tmp_path):
    src = flight_files()[0]
    _cli("process", "--input-dir", src, "--output-dir", tmp_path, "-o", "temp_c:tiff,pseudo:png", "--jobs", "1",
         "--strip-location")
    tiff = next(tmp_path.glob("*_temp_c.tiff"))
    with tifffile.TiffFile(tiff) as tf:
        page = tf.pages[0]
        xmp = page.tags["XMP"].value.decode()
        assert "GPSTag" not in page.tags and page.tags["Make"].value == "DJI"
        assert page.tags["DateTime"].value
    for key in ("GimbalYawDegree", "GimbalPitchDegree", "FlightYawDegree", "LRFTargetDistance", "UTCAtExposure"):
        assert f"drone-dji:{key}=" in xmp, key
    assert 'Camera:BandName="LWIR"' in xmp and 'dtc:LocationRemoved="True"' in xmp
    with Image.open(tiff) as im:                    # the EXIF block is still well formed
        exif = im.getexif()
        assert not exif.get_ifd(0x8825) and exif.get_ifd(0x8769)
    with Image.open(next(tmp_path.glob("*_pseudo.png"))) as im:
        assert not im.getexif().get_ifd(0x8825) and im.getexif()[0x010F] == "DJI"


@pytest.mark.flight
def test_mosaic_free_of_location(tmp_path):
    files = flight_files()
    needles = {k: v for f in files[:3] for k, v in _needles(f).items()}
    proc = _cli("process", "--input-dir", files[0].parent, "--output-dir", tmp_path, "--mosaic", "--strip-location")
    assert proc.returncode == 0, proc.stderr
    console = proc.stdout + proc.stderr
    assert "Location removed (--strip-location)" in console and "EPSG" not in console
    assert not re.search(r"ground plane [-0-9]", console) and _in_text(console, needles) == []
    for path in tmp_path.iterdir():
        assert _found(path, needles) == [], path.name
    tif = tmp_path / "mosaic_temp_c.tif"
    with tifffile.TiffFile(tif) as tf:
        tags = tf.pages[0].tags
        assert not {33550, 33922, 34735} & {t.code for t in tags.values()}
    info = json.loads((tmp_path / "mosaic_temp_c.json").read_text(encoding="utf-8"))
    assert info["location_removed"] and not {"crs", "origin_utm", "ground_alt_m"} & set(info)
    assert all({"x_m", "y_m"} <= set(f) and not {"lat", "lon", "utm_e", "utm_n", "abs_alt_m"} & set(f)
               for f in info["frames"])
    text = json.dumps(info)
    assert not re.search(r"SerialNumber|Rtk|GpsStatus|AltitudeType", text)


def test_remove_xmp_properties():
    x = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="r"><rdf:Description rdf:about="" '
         b'xmlns:drone-dji="d" xmlns:exif="e" drone-dji:GpsLatitude="+42.1" drone-dji:GimbalYawDegree="+5.4" '
         b'drone-dji:DroneSerialNumber="X1" drone-dji:LRFTargetDistance="21.4" drone-dji:LRFTargetLat="42.1">'
         b'<exif:GPSLatitude>42,6.2N</exif:GPSLatitude><exif:GPSDest><rdf:Seq><rdf:li>1</rdf:li></rdf:Seq></exif:GPSDest>'
         b'<aux:SerialNumber/><drone-dji:UTCAtExposure>2026</drone-dji:UTCAtExposure></rdf:Description></rdf:RDF></x:xmpmeta>')
    out = xmpedit.remove_properties(x, privacy.XMP_REMOVED.match).decode()
    assert "42" not in out and "X1" not in out and "Serial" not in out and "GPS" not in out
    for kept in ('xmlns:drone-dji="d"', 'GimbalYawDegree="+5.4"', 'LRFTargetDistance="21.4"', ">2026<", "</rdf:Description>"):
        assert kept in out, kept
