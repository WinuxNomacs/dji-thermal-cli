"""Test configuration: what the tests need, and clean skips without it.

The suite runs with nothing configured. Three optional settings, environment variables,
add the tests that need real data or the native SDK:

  DJI_THERMAL_SDK_DIR      an unpacked DJI Thermal SDK. Its Linux x64 libraries are the
                           oracle the pure-Python port is compared with (on Linux), and its
                           dataset/
                           folder supplies the sample images.
  DJI_THERMAL_SAMPLES_DIR  sample R-JPEGs laid out like the SDK's dataset/ (one folder per
                           camera); used instead of the SDK's dataset/.
  DJI_THERMAL_FLIGHT_DIR   the images of one flight: overlapping nadir thermal frames with GPS,
                           visual images may be mixed in (mosaic, normalization, WebODM).

Markers say what a test needs: `samples`, `flight`, `native` (the SDK's libraries). A test
whose need isn't configured is skipped, with the variable to set as the reason. To leave out
the native comparisons: pytest -m "not native". See CONTRIBUTING.md.
"""

from __future__ import annotations

import functools
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from . import native_sdk

SDK_ENV, SAMPLES_ENV, FLIGHT_ENV = "DJI_THERMAL_SDK_DIR", "DJI_THERMAL_SAMPLES_DIR", "DJI_THERMAL_FLIGHT_DIR"


def _dir(env: str) -> Path | None:
    value = os.environ.get(env, "").strip()
    return Path(value).expanduser().resolve() if value else None


SDK_DIR = _dir(SDK_ENV)
LIB_DIR = SDK_DIR / "tsdk-core" / "lib" / "linux" / "release_x64" if SDK_DIR else None
SAMPLES = _dir(SAMPLES_ENV) or (SDK_DIR / "dataset" if SDK_DIR else None)
FLIGHT = _dir(FLIGHT_ENV)
NATIVE_HELPER = Path(__file__).with_name("native_measure.py")
JPEG_SUFFIXES = (".jpg", ".jpeg")


# --- what is configured ---------------------------------------------------------------

def sample_files() -> list[Path]:
    """Every sample R-JPEG (SAMPLES/<camera>/<file>), or [] without samples."""
    if not (SAMPLES and SAMPLES.is_dir()):
        return []
    return sorted(p for p in SAMPLES.glob("*/*") if p.suffix.lower() in JPEG_SUFFIXES)


def sample_key(path: Path) -> str:
    """A sample's name independent of where it is: "<camera>/<file>" (seeds, recorded digests)."""
    return f"{path.parent.name}/{path.name}"


def sample(camera: str) -> Path:
    """The first sample of a camera (DJI_0001_R.JPG in the SDK's dataset/); skips the test when
    there is none."""
    files = sorted(p for p in (SAMPLES / camera).glob("*") if p.suffix.lower() in JPEG_SUFFIXES) \
        if SAMPLES and (SAMPLES / camera).is_dir() else []
    if not files:
        pytest.skip(f"no {camera} sample: set {SAMPLES_ENV} or {SDK_ENV}")
    return files[0]


def samples_dir(camera: str | None = None) -> Path:
    """The samples folder (or one camera's), skipping the test without it."""
    path = SAMPLES / camera if SAMPLES and camera else SAMPLES
    if not (path and path.is_dir()):
        pytest.skip(f"no {camera or 'sample'} images: set {SAMPLES_ENV} or {SDK_ENV}")
    return path


@functools.cache
def _flight_scan() -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """(thermal R-JPEGs, ordinary JPEGs) of the flight, by content."""
    from dji_thermal_cli import dataset

    if not (FLIGHT and FLIGHT.is_dir()):
        return (), ()
    kinds = [(p, dataset.classify(p).kind) for p in dataset.scan(FLIGHT).jpegs]
    return (tuple(sorted(p for p, kind in kinds if kind == dataset.THERMAL)),
            tuple(sorted(p for p, kind in kinds if kind == dataset.NOT_THERMAL)))


def flight_files() -> list[Path]:
    """The flight's thermal R-JPEGs (by content), skipping the test without them."""
    thermal, _ = _flight_scan()
    if not thermal:
        pytest.skip(f"no flight images: set {FLIGHT_ENV}")
    return list(thermal)


def flight_dir() -> Path:
    flight_files()
    return FLIGHT


def flight_visual() -> list[Path]:
    """The flight's ordinary (visual) JPEGs; may be empty."""
    return list(_flight_scan()[1])


@functools.cache
def flight_pitches() -> tuple[float, ...]:
    """Every flight thermal frame's gimbal pitch, for expectations that depend on the flight."""
    from dji_thermal_cli import djixmp

    return tuple(djixmp.number(djixmp.fields(p.read_bytes()), "GimbalPitchDegree") for p in flight_files())


def flight_counts(max_pitch: float = -80.0) -> tuple[int, int]:
    """(nadir, oblique) frames of the flight at a pitch threshold; frames without a recorded
    pitch are neither."""
    pitches = [p for p in flight_pitches() if p is not None]
    nadir = sum(p <= max_pitch for p in pitches)
    return nadir, len(pitches) - nadir


def _native_libraries() -> str | None:
    """Why the native libraries can't be used, or None."""
    if SDK_DIR is None:
        return f"native DJI Thermal SDK not configured: set {SDK_ENV}"
    if not sys.platform.startswith("linux"):
        # DJI's Windows build differs from its Linux build where results are not fully defined
        # (H20T float rounding, out-of-range settings); the port reproduces the Linux build, and
        # test_port_digest checks it gives the same results here
        return "native comparisons run on Linux, against the DJI build the port reproduces"
    if not (LIB_DIR / "libdirp.so").exists():
        return f"no libdirp.so under {LIB_DIR}"
    return None


@functools.cache
def native_status() -> tuple[str | None, str]:
    """(why native tests can't run or None, a one-line description). The API revision is
    probed once, on one sample per camera."""
    problem = _native_libraries()
    if problem:
        return problem, problem
    probe = [sorted(d.glob("*"))[0] for d in sorted(SAMPLES.iterdir()) if d.is_dir() and any(d.glob("*"))] \
        if SAMPLES and SAMPLES.is_dir() else []
    if not probe:
        return None, f"{SDK_DIR} (API revision not checked: no samples)"
    proc = subprocess.run([sys.executable, str(Path(__file__).with_name("native_probe.py")), *map(str, probe)],
                          env=_native_env(), capture_output=True, text=True)
    if proc.returncode != 0:
        why = f"the native SDK at {SDK_DIR} failed to load: {proc.stderr.strip().splitlines()[-1:]}"
        return why, why
    revisions = json.loads(proc.stdout)["revisions"]
    described = f"{SDK_DIR} (API revisions {', '.join(map(str, revisions))})"
    unknown = sorted(set(revisions) - native_sdk.SUPPORTED_API_VERSIONS)
    if unknown and os.environ.get(native_sdk.ALLOW_UNKNOWN_ENV) != "1":
        why = (f"the native SDK reports API revision(s) {unknown}; the port and its bindings mirror DJI Thermal "
               f"SDK v{native_sdk.SDK_VERSION} (revisions {sorted(native_sdk.SUPPORTED_API_VERSIONS)}). Set "
               f"{native_sdk.ALLOW_UNKNOWN_ENV}=1 to compare anyway")
        return why, why
    return None, described


# --- markers and the report header ------------------------------------------------------

def _missing(name: str) -> str | None:
    if name == "samples":
        return None if sample_files() else f"no sample images: set {SAMPLES_ENV} or {SDK_ENV}"
    if name == "flight":
        return None if _flight_scan()[0] else f"no flight images: set {FLIGHT_ENV}"
    return native_status()[0]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    for name in ("native", "samples", "flight"):
        if item.get_closest_marker(name) and (why := _missing(name)):
            pytest.skip(why)


def pytest_report_header(config):
    samples = sample_files()
    flight, _ = _flight_scan()
    return [
        f"samples: {f'{len(samples)} file(s) in {SAMPLES}' if samples else 'not configured'}",
        f"flight: {f'{len(flight)} thermal frame(s) in {FLIGHT}' if flight else 'not configured'}",
        f"native SDK: {native_status()[1]}",
    ]


# --- running the native SDK ---------------------------------------------------------------
# libdirp can crash, so every native call runs in a subprocess.

NATIVE_CRASHED = -1000


def _native_env() -> dict:
    return dict(os.environ, LD_LIBRARY_PATH=str(LIB_DIR), DJI_THERMAL_SDK_LIB_DIR=str(LIB_DIR))


def _crashed(proc: subprocess.CompletedProcess) -> bool:
    """The native SDK crashed (killed by a signal)."""
    return proc.returncode < 0


def _check(proc: subprocess.CompletedProcess) -> None:
    """A native helper that failed without crashing: raise with its error output."""
    if proc.returncode:
        err = proc.stderr.decode("utf-8", "replace") if isinstance(proc.stderr, bytes) else proc.stderr
        raise RuntimeError(f"native helper exited with {proc.returncode}:\n{err[-3000:]}")


def native_info(path: Path) -> tuple[tuple, tuple]:
    """(rjpeg_version, resolution) as the native SDK reports them."""
    with np.load(io.BytesIO(_run_native(path, None).stdout)) as z:
        return tuple(int(v) for v in z["version"]), tuple(int(v) for v in z["resolution"])


def _run_native(path: Path, params: dict | None):
    args = [sys.executable, str(NATIVE_HELPER), str(path)]
    if params is not None:
        args += [repr(params[k]) for k in ("distance", "humidity", "emissivity", "reflection", "ambient_temp")]
    return subprocess.run(args, env=_native_env(), capture_output=True)


def native_measure(path: Path, params: dict | None = None):
    """Run the native SDK in a subprocess: returns (temps, params, error code). A native crash
    is reported as NATIVE_CRASHED with no params."""
    proc = _run_native(path, params)
    if _crashed(proc):
        return None, None, NATIVE_CRASHED
    _check(proc)
    with np.load(io.BytesIO(proc.stdout)) as z:
        code = int(z["code"])
        return (z["t"] if code == 0 else None), dict(zip(("distance", "humidity", "emissivity", "reflection", "ambient_temp"), z["params"])), code


NATIVE_SEQUENCE_HELPER = Path(__file__).with_name("native_sequence.py")


def native_sequence(path: Path, steps: list, tmp_path: Path):
    """Replay set/measure steps on one native handle: list of (set code, measure code, temps,
    params) per step, or None when the native SDK crashed."""
    spec = tmp_path / "steps.json"
    spec.write_text(json.dumps(steps))
    proc = subprocess.run([sys.executable, str(NATIVE_SEQUENCE_HELPER), str(path), str(spec)], env=_native_env(), capture_output=True)
    if _crashed(proc):
        return None
    _check(proc)
    with np.load(io.BytesIO(proc.stdout)) as z:
        return [(int(z[f"set_{i}"]), int(z[f"meas_{i}"]), z[f"t_{i}"], z[f"params_{i}"]) for i in range(len(steps))]


NATIVE_PROCESS_HELPER = Path(__file__).with_name("native_process.py")


def native_process(path: Path, steps: list, tmp_path: Path):
    """Replay image-pipeline ops on one native handle: list of (code, data or None) per step,
    or None when the native SDK crashed."""
    spec = tmp_path / "isp_steps.json"
    spec.write_text(json.dumps(steps))
    proc = subprocess.run([sys.executable, str(NATIVE_PROCESS_HELPER), str(path), str(spec)], env=_native_env(), capture_output=True)
    if _crashed(proc):
        return None
    _check(proc)
    with np.load(io.BytesIO(proc.stdout)) as z:
        return [(int(z[f"code_{i}"]), z[f"out_{i}"] if f"out_{i}" in z.files else None) for i in range(len(steps))]
