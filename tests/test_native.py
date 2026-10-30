"""DJI's own SDK in place of the port (dji_thermal_cli.native): its library runs in a child
process, so a crash or a hang there fails one call and never the caller.

Without the SDK, a stand-in child (fake_native_child.py) runs the real request loop; with it
(the native marker), the native backend is compared with the port on every sample."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

import dji_thermal_cli as dtc
from dji_thermal_cli import native, tsdk
from dji_thermal_cli.native import client

from .conftest import SDK_DIR, sample, sample_files

FAKE_CHILD = Path(__file__).with_name("fake_native_child.py")
DATA = b"\xff\xd8\xff\xda\x00\x02"          # a JPEG with no segments: enough for the proxy


def _fake(mode: str = "ok", **kwargs) -> native.NativeProcess:
    # the child runs in its library's folder: here the stand-in's own
    return native.NativeProcess(FAKE_CHILD, command=[sys.executable, str(FAKE_CHILD), mode], **kwargs)


@pytest.fixture
def fake():
    p = _fake()
    yield p
    p.shutdown()


def test_a_crash_in_the_library_fails_only_that_call(fake):
    rj = native.open(DATA, "unused", fake)
    first = fake._proc.pid
    crashed = "ended by exception 0xC0000005 \\(access violation\\); it wrote: exception: access violation" \
        if os.name == "nt" else "killed by SIG"
    with pytest.raises(native.NativeCrash, match="crashed .*" + crashed):
        rj._call("crash", "crash")
    assert isinstance(native.NativeCrash("x"), tsdk.TsdkError)
    again = native.open(DATA, "unused", fake)               # a new child serves the next call
    assert fake._proc.pid != first
    assert np.array_equal(again.measure_temperature_c(), [[20.5, 21.5]])


def test_a_file_opened_before_a_crash_says_to_open_it_again(fake):
    rj = native.open(DATA, "unused", fake)
    with pytest.raises(native.NativeCrash):
        rj._call("crash", "crash")
    native.open(DATA, "unused", fake)
    with pytest.raises(native.NativeCrash, match="open the file again"):
        rj.measure_temperature_c()


def test_a_call_that_hangs_stops_the_child_after_the_timeout():
    p = _fake(request_timeout=1)
    try:
        rj = native.open(DATA, "unused", p)
        with pytest.raises(native.NativeCrash, match="stopped answering"):
            rj._call("hang", "hang")
        assert not p.alive()
        assert native.open(DATA, "unused", p).width == 2
    finally:
        p.shutdown()


def test_a_library_replaced_in_place_is_loaded_again(tmp_path):
    lib = tmp_path / "libdirp.so"
    lib.write_bytes(b"one")
    p = native.NativeProcess(lib, command=[sys.executable, str(FAKE_CHILD), "ok"])
    try:
        rj = native.open(DATA, "unused", p)
        first = p._proc.pid
        lib.write_bytes(b"a newer one")                         # an SDK updated in its folder
        assert native.open(DATA, "unused", p).width == 2
        assert p._proc.pid != first
        with pytest.raises(native.NativeCrash, match="open the file again"):
            rj.measure_temperature_c()
    finally:
        p.shutdown()


def test_what_the_library_prints_does_not_reach_the_replies(fake, capfd):
    rj = native.open(DATA, "unused", fake)
    assert rj._call("chatter", "chatter") == "answered"
    assert "library log line" not in capfd.readouterr().out


def test_an_sdk_version_the_bindings_do_not_follow_is_refused():
    p = _fake("unsupported")
    try:
        with pytest.raises(native.NativeSdkError, match="API revision 21 .*supports DJI Thermal SDK v1.8"):
            native.open(DATA, "unused", p)
    finally:
        p.shutdown()


def test_a_library_that_does_not_load_is_reported():
    p = _fake("fail")
    with pytest.raises(native.NativeSdkError, match="does not load: OSError: libdirp.so: cannot open"):
        native.open(DATA, "unused", p)


def test_refused_parameters_are_named_with_the_cameras_range(fake):
    rj = native.open(DATA, "unused", fake)
    params = rj.get_measurement_params()
    params.distance = 100.0
    with pytest.raises(tsdk.ParamError, match=r"distance=100 is outside \[1, 25\]"):
        rj.set_measurement_params(params)
    assert rj.get_measurement_params().distance == 5.0


def test_a_forked_process_starts_its_own_child(monkeypatch, tmp_path):
    lib = tmp_path / "libdirp.so"
    first = client.process_for(lib)
    assert client.process_for(lib) is first
    monkeypatch.setattr(client, "_owner", -1)                # as in a process forked after it
    assert client.process_for(lib) is not first


@pytest.mark.parametrize("layout", [
    "tsdk-core/lib/linux/release_x64", "linux/release_x64", "release_x64", "",
])
def test_the_library_is_found_in_djis_layout(tmp_path, layout, monkeypatch):
    monkeypatch.setattr(native.bindings.sys, "platform", "linux")
    monkeypatch.setattr(native.bindings, "platform_dirs", lambda: ("linux", "release_x64"))
    monkeypatch.setattr(native.bindings, "library_name", lambda: "libdirp.so")
    lib = tmp_path / layout / "libdirp.so"
    lib.parent.mkdir(parents=True, exist_ok=True)
    lib.write_bytes(b"")
    assert native.find_library(tmp_path) == lib


def test_a_folder_without_the_sdk_is_a_settings_error(tmp_path):
    with pytest.raises(dtc.SettingsError, match="no libdirp|no macOS build"):
        dtc.Settings(sdk=tmp_path)


# --- with DJI's SDK ---------------------------------------------------------------------------

def _maps() -> str:
    return Path("/proc/self/maps").read_text() if os.path.exists("/proc/self/maps") else ""


@pytest.mark.native
@pytest.mark.parametrize("settings", [
    {"distance": "embedded"},
    {"distance": 8.0, "emissivity": 0.9, "humidity": 40.0, "reflected_temp": 30.0},
])
def test_djis_sdk_measures_every_sample_as_the_port_does(settings):
    for src in sample_files():
        try:
            port = dtc.measure(src, dtc.Settings(**settings))
        except tsdk.TsdkError as e:
            with pytest.raises(type(e)):
                dtc.measure(src, dtc.Settings(sdk=SDK_DIR, **settings))
            continue
        ours = dtc.measure(src, dtc.Settings(sdk=SDK_DIR, **settings))
        assert np.array_equal(ours.temperature.view(np.uint32), port.temperature.view(np.uint32)), src
        assert ours.params == port.params, src
    assert "libdirp" not in _maps()                          # loaded only in the child


@pytest.mark.native
def test_djis_sdk_describes_a_file_as_the_port_does():
    src = sample("M4T")
    assert dtc.describe(src, sdk=SDK_DIR)["thermal"] == dtc.describe(src)["thermal"]


@pytest.mark.native
def test_a_crash_in_djis_library_leaves_the_caller_running():
    src = sample("H20T")
    rj = native.open(src.read_bytes(), SDK_DIR)
    rj.set_pseudo_color(-1)                                   # libdirp reads outside its palettes
    with pytest.raises(native.NativeCrash, match="dirp_process.*SIGSEGV"):
        rj.process()
    assert dtc.measure(src, dtc.Settings(sdk=SDK_DIR)).temperature.shape == (512, 640)
    assert "libdirp" not in _maps()


@pytest.mark.native
def test_processing_with_djis_sdk_writes_the_ports_temperatures(tmp_path):
    src = sample("M4T")
    for sdk, out in ((None, tmp_path / "port"), (SDK_DIR, tmp_path / "native")):
        report = dtc.process(src, out, dtc.Settings(sdk=sdk), outputs=["temp_c:tiff"], jobs=1)
        assert report.processed == 1
    name = f"{src.stem}_temp_c.tiff"
    assert np.array_equal(tifffile.imread(tmp_path / "port" / name), tifffile.imread(tmp_path / "native" / name))


@pytest.mark.native
def test_check_sdk_reports_the_revisions_djis_sdk_gives():
    found = dtc.check_sdk(SDK_DIR, [sample("M4T"), sample("H20T")])
    assert found["api_revisions"] == [19, 20]
    assert found["sdk_version"] == "1.8"
    assert found["measured"] == ["DJI_0001_R.JPG", "DJI_0001_R.JPG"]
