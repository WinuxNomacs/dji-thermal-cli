"""The port keeps every result the native comparisons checked, without the native SDK.

Each call sequence the *_vs_native tests replay (tests/oracle_cases.py) is replayed on the
port alone and digested: return codes, images, temperatures, getter values. The digests were
recorded on Linux right after the native comparisons passed on the same sequences, so a
difference here is a change from DJI's results. Like test_port_digest, it runs on every
system with the samples (and the flight, for its frames) configured.

After an intended change to the port's results, rerecord on Linux with the native SDK, the
samples and the flight configured; the recorder runs the native comparisons first and
records only if they pass:  uv run --env-file .env python -m tests.test_oracle_digest
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from dji_thermal_cli import tsdk

from . import oracle_cases as cases
from .conftest import _flight_scan, native_status, sample_files, sample_key

RECORDED = Path(__file__).with_name("oracle_digests.json")
NATIVE_TESTS = ["tests/test_tsdk_vs_native.py", "tests/test_isp_vs_native.py", "tests/test_api_vs_native.py"]

# (case id, input bytes, the sequences to replay given the opened file: one per fresh handle)
Case = tuple[str, Callable[[], bytes], Callable[[object], list[list[dict]]]]


def _measure_steps(sets: list[dict | None]) -> list[dict]:
    steps = []
    for params in sets:
        if params is not None:
            steps.append({"op": "set_params", **params})
        steps += [{"op": "measure_ex"}, {"op": "params"}]
    return steps


def sample_cases(files: list[Path]) -> Iterator[Case]:
    """The samples' sequences, as test_tsdk_vs_native, test_isp_vs_native and test_api_vs_native
    run them."""
    def read(p):
        return p.read_bytes

    for p in files:
        k = sample_key(p)
        yield f"embedded {k}", read(p), lambda rj: [[{"op": "info"}, {"op": "params"}, {"op": "measure_ex"}]]
        yield (f"random-params {k}", read(p),
               lambda rj, p=p: [[{"op": "set_params", **s}, {"op": "measure_ex"}, {"op": "params"}]
                                for s in cases.random_params(rj, p)])
        yield f"parameter-sequence {k}", read(p), lambda rj, p=p: [_measure_steps(cases.parameter_sequence(rj, p))]
        yield f"api {k}", read(p), lambda rj, p=p: [cases.api_steps(rj, seed) for seed in cases.api_seeds(p)]
        yield f"isp-defaults {k}", read(p), lambda rj: [cases.ISP_DEFAULTS]
    for p in cases.isp_sweep(files):
        yield f"isp-random {sample_key(p)}", read(p), lambda rj, p=p: [cases.isp_random(rj, p)]
    for p in cases.isp_nd(files):
        yield f"isp-nd {sample_key(p)}", lambda p=p: cases.with_nd(p.read_bytes()), lambda rj: [cases.ISP_ND]
    for p in cases.isp_variants(files):
        for gain in cases.GAINS:
            yield (f"isp-gain{gain} {sample_key(p)}", lambda p=p, g=gain: cases.with_gain(p.read_bytes(), g),
                   lambda rj: [cases.ISP_GAIN])
    for p in cases.hirp(files):
        yield f"hirp-random {sample_key(p)}", read(p), lambda rj, p=p: [cases.hirp_random(rj, p)]
        yield f"hirp-alarm {sample_key(p)}", lambda p=p: cases.with_alarm(p.read_bytes()), lambda rj: [cases.HIRP_ALARM]
    for p in cases.api_nd(files):
        yield (f"api-nd {sample_key(p)}", lambda p=p: cases.with_nd(p.read_bytes()),
               lambda rj, p=p: [cases.api_steps(rj, seed) for seed in cases.api_seeds(p, "nd")])


def flight_cases(files: list[Path]) -> Iterator[Case]:
    for p in files:
        yield f"flight {p.name}", p.read_bytes, lambda rj: [cases.FLIGHT]


def _feed(h, v) -> None:
    """Add one value to the digest, unambiguously and independent of the system."""
    if v is None:
        h.update(b"N")
    elif isinstance(v, np.ndarray):
        h.update(f"A{v.dtype.str}{v.shape}".encode())
        h.update(np.ascontiguousarray(v).tobytes())
    elif dataclasses.is_dataclass(v):
        h.update(f"D{type(v).__name__}".encode())
        for f in dataclasses.fields(v):
            _feed(h, getattr(v, f.name))
    elif isinstance(v, (tuple, list)):
        h.update(f"T{len(v)}".encode())
        for x in v:
            _feed(h, x)
    elif isinstance(v, (bool, np.bool_)):
        h.update(b"B1" if v else b"B0")
    elif isinstance(v, (int, np.integer)):
        h.update(f"I{int(v)}".encode())
    elif isinstance(v, (float, np.floating)):
        h.update(b"F" + struct.pack("<d", float(v)))
    elif isinstance(v, str):
        h.update(b"S" + v.encode())
    else:
        raise TypeError(f"no digest for {type(v).__name__}")


def digest(case: Case) -> str:
    _, data_fn, sequences = case
    data = data_fn()
    h = hashlib.sha256()
    try:
        seqs = sequences(tsdk.open(data))
    except tsdk.TsdkError as e:                   # a file the port refuses: record how
        _feed(h, ("open", e.code))
        return h.hexdigest()
    for steps in seqs:
        rj = tsdk.open(data)
        for step in steps:
            code, value = cases.run(rj, step)
            _feed(h, (step["op"], code, value))
    return h.hexdigest()


def _compare(found: list[Case]) -> None:
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))
    common = [c for c in found if c[0] in recorded]
    if not common:
        pytest.skip("none of the recorded cases' files are configured")
    differ = [c[0] for c in common if digest(c) != recorded[c[0]]]
    assert not differ, f"the port's results differ from those checked against the native SDK for {differ}"


@pytest.mark.samples
def test_samples_as_recorded():
    _compare(list(sample_cases(sample_files())))


@pytest.mark.flight
def test_flight_as_recorded():
    _compare(list(flight_cases(list(_flight_scan()[0]))))


if __name__ == "__main__":
    reason, _ = native_status()
    if reason:
        sys.exit(f"recording needs the native SDK to check against: {reason}")
    found = list(sample_cases(sample_files())) + list(flight_cases(list(_flight_scan()[0])))
    if not found:
        sys.exit("no samples or flight configured")
    print("running the native comparisons first ...", flush=True)
    if subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "native", *NATIVE_TESTS]).returncode:
        sys.exit("the native comparisons failed: nothing recorded")
    RECORDED.write_text(json.dumps({c[0]: digest(c) for c in found}, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"recorded {len(found)} case(s) in {RECORDED}")
