"""The port gives the same results on every system.

Digests of its results on the SDK's sample images (temperatures under several measurement
parameter sets, and the processed image in every palette) are recorded on Linux, where the
native comparisons hold the port to DJI's SDK bit for bit; on any system with the samples
configured, the port must reproduce them exactly.

After an intended change to the port's results, rerecord on Linux with the native tests
passing:  uv run python -m tests.test_port_digest
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from dji_thermal_cli import tsdk

from . import oracle_cases as cases
from .conftest import sample_files, sample_key

RECORDED = Path(__file__).with_name("port_digests.json")
PARAMS = cases.DIGEST_PARAMS


def digest(path: Path) -> str:
    """The port's results for one file, as one SHA-256."""
    h = hashlib.sha256()
    data = path.read_bytes()

    def add(fn):
        try:
            h.update(fn().tobytes())
        except tsdk.TsdkError as e:
            h.update(f"{type(e).__name__} {e.code}".encode())

    for params in PARAMS:
        rj = tsdk.open(data)
        if params is not None:
            p = rj.get_measurement_params()
            p.distance, p.humidity, p.emissivity, p.reflection = params
            try:
                rj.set_measurement_params(p)
            except tsdk.TsdkError as e:
                h.update(f"{type(e).__name__} {e.code}".encode())
                continue
        add(rj.measure_temperature_c)
    for palette in range(10):
        rj = tsdk.open(data)
        rj.set_pseudo_color(palette)
        add(rj.process)
    return h.hexdigest()


def _openable(files: list[Path]) -> list[Path]:
    out = []
    for p in files:
        try:
            tsdk.open(p.read_bytes())
        except tsdk.TsdkError:
            continue
        out.append(p)
    return out


@pytest.mark.samples
def test_same_results_as_recorded():
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))
    files = {sample_key(p): p for p in sample_files()}
    common = sorted(set(recorded) & set(files))
    if not common:
        pytest.skip("none of the recorded samples (the SDK's dataset/) are configured")
    differ = [k for k in common if digest(files[k]) != recorded[k]]
    assert not differ, f"the port's results differ from those recorded on Linux for {differ}"


if __name__ == "__main__":
    files = _openable(sample_files())
    if not files:
        sys.exit("no samples: set DJI_THERMAL_SAMPLES_DIR or DJI_THERMAL_SDK_DIR")
    RECORDED.write_text(json.dumps({sample_key(p): digest(p) for p in files}, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"recorded {len(files)} sample(s) in {RECORDED}")
