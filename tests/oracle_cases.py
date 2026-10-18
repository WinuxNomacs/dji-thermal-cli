"""The call sequences the native comparisons replay, defined once: the *_vs_native tests check
them against the native SDK, and test_oracle_digest holds the port to the results it gave
while those checks passed, so they stay checked without the SDK.

A sequence is a list of steps, {"op": ...} dicts as tests/native_process.py takes them, plus
"params" and "info" (measurement parameters; rjpeg version and resolution), which the native
comparisons check through their own helpers. Seeds derive from the sample's key, so every
system replays the same sequences.
"""

from __future__ import annotations

import random
import struct
from pathlib import Path

import numpy as np

from dji_thermal_cli import tsdk

from .conftest import sample_key

PARAMS = ("distance", "humidity", "emissivity", "reflection", "ambient_temp")
PALETTES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1]


def run(rj, step: dict):
    """The port's counterpart of one step: (dirp return code, data or None). "get" returns
    the lowest getter code and (codes, values) of the five getters."""
    op = step["op"]
    try:
        if op == "set_palette":
            rj.set_pseudo_color(step["v"])
        elif op == "set_cb":
            rj.set_color_bar(tsdk.ColorBar(step["flag"], step["high"], step["low"]))
        elif op == "set_iso":
            rj.set_isotherm(tsdk.Isotherm(step["flag"], step["high"], step["low"]))
        elif op == "set_bright":
            rj.set_enhancement_params(step["v"])
        elif op == "set_params":
            rj.set_measurement_params(tsdk.MeasurementParams(**{k: step[k] for k in PARAMS}))
        elif op == "get":
            vals, codes = [], []
            for fn in (rj.get_pseudo_color, rj.get_color_bar, rj.get_color_bar_adaptive_params, rj.get_isotherm,
                       rj.get_enhancement_params):
                try:
                    vals.append(fn())
                    codes.append(0)
                except tsdk.TsdkError as e:
                    vals.append(None)
                    codes.append(e.code)
            return min(codes), (codes, vals)
        elif op == "lut":
            return 0, rj.get_pseudo_color_lut()
        elif op == "process":
            return 0, rj.process()
        elif op == "strech":
            return 0, rj.process_strech()
        elif op == "measure16":
            return 0, rj.measure()
        elif op == "measure_ex":
            return 0, rj.measure_temperature_c()
        elif op == "raw":
            return 0, rj.get_original_raw()
        elif op == "api":
            rev, magic = rj.get_api_version()
            return 0, np.frombuffer(struct.pack("<I8s", rev, magic.encode()), np.uint8)
        elif op == "nd":
            return 0, np.array([int(rj.infrared_density_filter_installed())])
        elif op == "params":
            return 0, rj.get_measurement_params()
        elif op == "info":
            return 0, (tuple(rj.rjpeg_version()), tuple(rj.resolution()))
        else:
            raise ValueError(f"unknown op {op!r}")
    except tsdk.TsdkError as e:
        return e.code, None
    return 0, None


# --- the sample selections ----------------------------------------------------------------

def _named(files: list[Path], keys) -> list[Path]:
    return [p for p in files if sample_key(p) in keys]


def isp_sweep(files: list[Path]) -> list[Path]:
    return [p for p in files if p.name in ("DJI_0001_R.JPG", "DJI_0005_R.JPG", "DJI_0007_R.JPG")
            or p.parent.name in ("M30T", "H20T", "XTS")]


def isp_variants(files: list[Path]) -> list[Path]:
    """One classic-enhance and two v0300 (agc3) iirp captures."""
    return _named(files, ("H20N/DJI_0001_R.JPG", "H30T/DJI_0005_R.JPG", "M4T/DJI_0001_R.JPG"))


def isp_nd(files: list[Path]) -> list[Path]:
    return isp_variants(files) + _named(files, ("M30T/DJI_0003_R.JPG", "H20T/DJI_0002_R.JPG", "XTS/DJI_0003_R.JPG"))


def hirp(files: list[Path]) -> list[Path]:
    return _named(files, {f"M3TD/DJI_000{i}_R.JPG" for i in range(1, 6)} | {f"M4T/DJI_000{i}_R.JPG" for i in (3, 4, 5)})


def api_nd(files: list[Path]) -> list[Path]:
    return _named(files, ("M30T/DJI_0001_R.JPG", "M30T/DJI_0003_R.JPG", "H20T/DJI_0002_R.JPG", "XTS/DJI_0003_R.JPG",
                          "M3TD/DJI_0001_R.JPG", "H30T/DJI_0001_R.JPG"))


GAINS = (0, 2)

# Sequences found by running the random comparisons under many more seeds: each once gave a
# result one step off the native SDK's (an integer transmission rounding the other way), so
# they stay in the regular runs. (distance, humidity, emissivity, reflection, ambient)
EXTRA_PARAMS = {
    "H30T/DJI_0002_R.JPG": [(90.30352311158961, 89.24558401052104, 0.6095231411007017, 41.163718157998844,
                             4.069414381907734)],
    "H30T/DJI_0005_R.JPG": [(201.85677338512272, 86.32062615431536, 0.9080948256399913, 30.95059195544721,
                             46.08308308494384)],
}
# test_port_digest's parameter sets (distance, humidity %, emissivity, reflected C); None:
# the file's own parameters. Compared with the native SDK by test_tsdk_vs_native.
DIGEST_PARAMS = [None, (5.0, 70.0, 0.95, 23.0), (1.0, 20.0, 0.5, -10.0), (25.0, 100.0, 1.0, 60.0), (12.5, 45.0, 0.8, 0.0)]
EXTRA_API_SEEDS = {"M2EA/DJI_0003_R.JPG": ["sweep8:apiM2EA/DJI_0003_R.JPG"]}


# --- patched inputs -----------------------------------------------------------------------

def with_nd(data: bytes, scale: float = 1.37, offset: float = -7.5) -> bytes:
    """Insert an APP7 'DJI-IR_ND' block (an installed infrared density filter) after the last
    APPn segment (libv_girp rejects a frame whose segment order it does not expect)."""
    payload = b"DJI-IR_ND\0" + struct.pack("<HHff", 0, 1, scale, offset)
    i = 2
    while data[i] == 0xFF and 0xE0 <= data[i + 1] <= 0xEF:
        i += 2 + struct.unpack_from(">H", data, i + 2)[0]
    return data[:i] + b"\xff\xe7" + struct.pack(">H", len(payload) + 2) + payload + data[i:]


def with_gain(data: bytes, gain: int) -> bytes:
    """Rewrite the iirp header's gain-mode byte (handle+0x98)."""
    i = data.index(b"\xff\xe4") + 4
    magic = struct.unpack_from("<I", data, i)[0] == 0xFFD1D2FF
    off = i + (0x20 if magic else 0) + 0x60
    return data[:off] + bytes([gain]) + data[off + 1:]


def with_alarm(data: bytes) -> bytes:
    """Set the hirp ImageParams temperature-alarm flag (APP4 ImageParams + 0x15)."""
    i = data.index(b"\xff\xe4") + 4 + 0x20 + 0x15
    return data[:i] + b"\x01" + data[i + 1:]


# --- the sequences ------------------------------------------------------------------------

ISP_DEFAULTS = [{"op": "get"}, {"op": "lut"}, {"op": "process"}, {"op": "strech"}]
ISP_ND = [{"op": "get"}, {"op": "set_cb", "flag": True, "high": 60.0, "low": 10.0}, {"op": "get"}, {"op": "process"},
          {"op": "set_cb", "flag": True, "high": 500.0, "low": -100.0}, {"op": "get"}, {"op": "process"},
          {"op": "strech"}]
ISP_GAIN = [{"op": "process"}, {"op": "strech"}, {"op": "set_cb", "flag": True, "high": 50.0, "low": 20.0},
            {"op": "process"}]
HIRP_ALARM = [{"op": "get"}, {"op": "process"}, {"op": "get"}, {"op": "strech"},
              {"op": "set_palette", "v": 5}, {"op": "process"},
              {"op": "set_iso", "flag": True, "high": 40.0, "low": 25.0}, {"op": "process"},
              {"op": "set_iso", "flag": False, "high": 40.0, "low": 25.0}, {"op": "process"}]
# a flight frame: temperatures both ways, the raw frame, and the image in two palettes
FLIGHT = [{"op": "measure_ex"}, {"op": "measure16"}, {"op": "raw"}, {"op": "process"},
          {"op": "set_palette", "v": 3}, {"op": "process"}, {"op": "strech"}]


def isp_random(rj, path: Path) -> list[dict]:
    rng = random.Random("isp" + sample_key(path))
    steps = []
    for _ in range(3):
        steps.append({"op": "set_palette", "v": rng.choice(PALETTES)})
        lo = rng.uniform(-20, 60)
        hi = lo + rng.choice([-5.0, 0.0, rng.uniform(0.5, 80)])
        steps.append({"op": "set_cb", "flag": rng.random() < 0.6, "high": hi, "low": lo})
        lo = rng.uniform(-20, 60)
        steps.append({"op": "set_iso", "flag": rng.random() < 0.5, "high": lo + rng.uniform(-2, 30), "low": lo})
        steps.append({"op": "set_bright", "v": rng.choice([0, 1, 25, 49, 50, 51, 99, 100, 101, -1])})
        if rng.random() < 0.5:
            rngs = rj.get_measurement_params_range()
            steps.append({"op": "set_params", **{k: rng.uniform(*rngs[k]) for k in PARAMS}})
        steps += [{"op": "get"}, {"op": "process"}, {"op": "strech"}]
    return steps


def hirp_random(rj, path: Path) -> list[dict]:
    """hirp: palette, brightness, colour bar, isotherm and measurement parameters."""
    rng = random.Random("hirp" + sample_key(path))
    steps = []
    for _ in range(4):
        steps.append({"op": "set_palette", "v": rng.choice(PALETTES)})
        steps.append({"op": "set_bright", "v": rng.choice([0, 13, 50, 77, 100, 101, -5])})
        lo = rng.choice([rng.uniform(-30, 60), rng.uniform(-300, 700)])
        hi = lo + rng.choice([-5.0, 0.0, 0.01, rng.uniform(0.5, 80), rng.uniform(80, 900)])
        steps.append({"op": "set_cb", "flag": rng.random() < 0.7, "high": hi, "low": lo})
        il = rng.choice([rng.uniform(-30, 60), rng.uniform(-300, 700)])
        ih = il + rng.choice([-1.0, 0.0, 0.01, rng.uniform(0.5, 30), rng.uniform(30, 900)])
        steps.append({"op": "set_iso", "flag": rng.random() < 0.7, "high": ih, "low": il})
        if rng.random() < 0.5:
            rngs = rj.get_measurement_params_range()
            steps.append({"op": "set_params", **{k: rng.uniform(*rngs[k]) for k in PARAMS}})
        steps += [{"op": "process"}, {"op": "get"}, {"op": "strech"}]
    return steps


def step_values(rng: random.Random, ranges: dict) -> dict:
    """Mostly in-range parameters, sometimes just outside the advertised range."""
    out = {}
    for k in PARAMS:
        lo, hi = ranges[k]
        if hi <= lo:
            out[k] = 25.0 if k == "ambient_temp" else lo
            continue
        r = rng.random()
        out[k] = lo - (hi - lo) * 0.05 * rng.random() if r < 0.08 else hi + (hi - lo) * 0.05 * rng.random() if r < 0.16 else rng.uniform(lo, hi)
    return out


def api_seeds(path: Path, prefix: str = "api") -> list[str]:
    """The seeds of a sample's api_steps: its own, and EXTRA_API_SEEDS."""
    return [prefix + sample_key(path)] + (EXTRA_API_SEEDS.get(sample_key(path), []) if prefix == "api" else [])


def api_steps(rj, seed: str) -> list[dict]:
    rng = random.Random(seed)
    ranges = rj.get_measurement_params_range()
    steps = [{"op": "api"}, {"op": "nd"}, {"op": "raw"}, {"op": "measure16"}]
    for _ in range(4):
        steps += [{"op": "set_params", **step_values(rng, ranges)}, {"op": "measure16"}, {"op": "measure_ex"}]
    return steps + [{"op": "raw"}]


def random_params(rj, path: Path) -> list[dict]:
    """Three parameter sets, each measured on a fresh handle, and the file's EXTRA_PARAMS."""
    rng = random.Random(path.name + path.parent.name)
    ranges = rj.get_measurement_params_range()
    out = []
    for _ in range(3):
        params = {k: rng.uniform(*ranges[k]) for k in ("distance", "humidity", "emissivity", "reflection")}
        lo, hi = ranges["ambient_temp"]
        params["ambient_temp"] = rng.uniform(lo, hi) if hi > lo else 25.0
        out.append(params)
    return out + [dict(zip(PARAMS, p)) for p in EXTRA_PARAMS.get(sample_key(path), ())]


def parameter_sequence(rj, path: Path) -> list[dict | None]:
    """Parameter sets to set on one handle in turn (None: measure without setting)."""
    rng = random.Random("seq" + path.name + path.parent.name)
    ranges = rj.get_measurement_params_range()
    return [None] + [step_values(rng, ranges) for _ in range(4)] + [None]
