"""The built-in tables must be byte-identical to the ones in the DJI Thermal SDK v1.8 libraries."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from dji_thermal_cli.tsdk import tables

from . import sdk_tables
from .conftest import SDK_DIR, SDK_ENV

BUILTIN = tables.load()


@pytest.mark.parametrize("name", list(sdk_tables.TABLES))
def test_matches_sdk_digest(name):
    _lib, size, dtype, _anchor, digest = sdk_tables.TABLES[name]
    a = BUILTIN[name]
    assert a.dtype == np.dtype(dtype)
    raw = a.tobytes()
    assert len(raw) == size
    assert hashlib.sha256(raw).hexdigest() == digest


def test_no_extra_tables():
    assert set(BUILTIN) == set(sdk_tables.TABLES)


def test_matches_sdk_libraries():
    """The digests were recorded from the SDK's Linux libraries, which every SDK download
    includes; they are only read, so this runs on any system. (The Windows DLLs store some
    tables in other widths.)"""
    linux = SDK_DIR / "tsdk-core" / "lib" / "linux" if SDK_DIR else None
    if not (linux and linux.is_dir()):
        pytest.skip(f"no Linux libraries of the DJI Thermal SDK: set {SDK_ENV}" if SDK_DIR is None
                    else f"no Linux libraries under {SDK_DIR} (the full SDK download includes them)")
    found = sdk_tables.extract(linux)
    assert set(found) == set(BUILTIN)
    for name, a in found.items():
        assert np.array_equal(a.view(np.uint8), BUILTIN[name].view(np.uint8)), name
