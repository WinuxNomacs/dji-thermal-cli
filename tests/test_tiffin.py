import itertools

import numpy as np
import pytest

from dji_thermal_cli import tiffin

from . import tiffs


def _bands(dtype, count=1, rows=37, cols=53, seed=0):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        return rng.normal(200, 30, (count, rows, cols)).astype(dtype)
    return rng.integers(0, 120, (count, rows, cols)).astype(dtype)


@pytest.mark.parametrize("compression, predictor, dtype", [
    ("none", 1, "f4"), ("deflate", 1, "f4"), ("deflate", 3, "f4"), ("deflate", 3, "f8"), ("lzw", 3, "f4"),
    ("deflate", 2, "i2"), ("lzw", 2, "u2"), ("lzw", 1, "u1"), ("deflate", 2, "f4"), ("lzw", 2, "i4")])
def test_the_first_band_is_read_whatever_its_compression_and_predictor(tmp_path, compression, predictor, dtype):
    bands = _bands(dtype, count=3)
    for k, (tile, planar, bigtiff, order) in enumerate(itertools.product((None, 16), (False, True), (False, True),
                                                                         ("<", ">"))):
        path = tiffs.write(tmp_path / f"{k}.tif", bands, compression=compression, predictor=predictor, tile=tile,
                           rows_per_strip=None if tile else 5, planar=planar, bigtiff=bigtiff, order=order)
        got, _ = tiffin.read_band(path)
        assert got.dtype == np.dtype(dtype) and got.dtype.isnative, (tile, planar, bigtiff, order)
        np.testing.assert_array_equal(got, bands[0], err_msg=str((tile, planar, bigtiff, order)))


def test_lzw_with_a_full_table_is_read(tmp_path):
    """Enough varied data for the codes to grow to 12 bits and the table to be cleared."""
    bands = _bands("f4", rows=300, cols=400, seed=3)
    got, _ = tiffin.read_band(tiffs.write(tmp_path / "lzw.tif", bands, compression="lzw", predictor=3, tile=256))
    np.testing.assert_array_equal(got, bands[0])


def test_tags_are_read_as_numbers_and_text(tmp_path):
    path = tiffs.write(tmp_path / "tags.tif", _bands("f4"), bigtiff=True, extra=[
        (33550, 12, (0.05, 0.05, 0.0)), (34735, 3, (1, 1, 0, 1, 1024, 0, 1, 1)), (42113, 2, b"-9999\0")])
    _, tags = tiffin.read_band(path)
    assert tags[33550] == (0.05, 0.05, 0.0) and tags[34735] == (1, 1, 0, 1, 1024, 0, 1, 1)
    assert tags[42113] == "-9999" and tags[256] == (53,)


def test_what_it_cannot_read_is_refused_saying_why(tmp_path):
    with pytest.raises(tiffin.TiffError, match="not a TIFF"):
        tiffin.read_band((tmp_path / "x.tif").write_bytes(b"\xff\xd8\xff" * 10) and tmp_path / "x.tif")
    path = tiffs.write(tmp_path / "jpeg.tif", _bands("u1"))
    data = bytearray(path.read_bytes())
    at = data.index(bytes([3, 1, 3, 0, 1, 0, 0, 0, 1, 0]))      # compression (259), SHORT, 1: none
    data[at + 8] = 7                                            # JPEG
    path.write_bytes(bytes(data))
    with pytest.raises(tiffin.TiffError, match="JPEG compression is not supported: rewrite it with Deflate"):
        tiffin.read_band(path)
    path = tiffs.write(tmp_path / "short.tif", _bands("f4"), compression="deflate")
    path.write_bytes(path.read_bytes()[:-2000])
    with pytest.raises(tiffin.TiffError, match="cut short"):
        tiffin.read_band(path)
