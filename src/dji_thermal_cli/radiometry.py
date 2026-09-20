"""Pure-Python radiometric model: raw sensor counts to temperature, without the DJI SDK.

The mapping is an empirical characterisation of the SDK's behaviour, stored as a compact
lookup model (data/iirp_v1.djtm). Each entry is the inverse curve raw(T) for one point of a
parameter grid (distance, humidity, ambient, reflection, emissivity); curves are stored as two
anchors plus 1-count second-difference codes in one lzma stream. A query interpolates the
neighbouring curves, which is linear in the stored codes, so they are summed before a single
cumulative-sum pass reconstructs the curve. The table is never expanded.

Supported for Matrice 4T data at emissivity 0.4-1.0; emissivity below 0.4 is refused.
"""

from __future__ import annotations

import functools
import json
import lzma
import struct
from importlib import resources

import numpy as np

from .rjpeg import MeasurementParams

MODEL_RESOURCE = "data/iirp_v1.djtm"
MIN_EMISSIVITY = 0.4

_PARAM_RANGES = {
    "distance": (1.0, 300.0),
    "humidity": (1.0, 100.0),
    "emissivity": (MIN_EMISSIVITY, 1.0),
    "reflection": (-40.0, 100.0),
    "ambient_temp": (-40.0, 80.0),
}
_DIST_BREAKS = (20, 22, 30, 50, 60, 70, 80, 100)
_AMB_BREAKS = (14, 41)
_DEGREE = {"dist": 3, "hum": 1, "amb": 3, "refl": 3, "emis": 3}
_MAGIC = b"DJTM1\n"


class ModelDomainError(ValueError):
    """The requested parameters fall outside the range this model supports."""


def _lagrange(nodes: np.ndarray, x: float, degree: int) -> tuple[np.ndarray, np.ndarray]:
    x = float(np.clip(x, nodes[0], nodes[-1]))
    n = len(nodes)
    degree = min(degree, n - 1)
    j = int(np.clip(np.searchsorted(nodes, x) - 1, 0, n - 2))
    start = int(np.clip(j - (degree - 1) // 2, 0, n - degree - 1))
    ids = np.arange(start, start + degree + 1)
    weights = np.ones(len(ids))
    for a, ia in enumerate(ids):
        for ib in ids:
            if ib != ia:
                weights[a] *= (x - nodes[ib]) / (nodes[ia] - nodes[ib])
    return ids, weights


def _lagrange_segment(nodes: np.ndarray, x: float, degree: int, breaks: tuple[int, ...]):
    """Interpolate only within the smooth segment (between breakpoints) that contains x."""
    x = float(np.clip(x, nodes[0], nodes[-1]))
    edges = [nodes[0], *[b for b in breaks if nodes[0] < b < nodes[-1]], nodes[-1]]
    k = 0
    while k < len(edges) - 2 and x > edges[k + 1]:
        k += 1
    idx = np.flatnonzero((nodes >= edges[k] - 1e-9) & (nodes <= edges[k + 1] + 1e-9))
    ids, weights = _lagrange(nodes[idx], x, degree)
    return idx[ids], weights


class ThermalModel:
    def __init__(self, blob: bytes) -> None:
        if blob[: len(_MAGIC)] != _MAGIC:
            raise ValueError("not a thermal model file")
        (header_len,) = struct.unpack_from("<I", blob, len(_MAGIC))
        start = len(_MAGIC) + 4
        header = json.loads(blob[start:start + header_len])
        payload = lzma.decompress(blob[start + header_len:])

        self._q = float(header["q"])
        self._shape = tuple(header["shape"])
        self._nt = int(header["nt"])
        self._t_grid = np.asarray(header["t_grid"], dtype=np.float64)
        self._dist, self._hum, self._amb, self._refl, self._emis = (
            np.asarray(header[k], dtype=np.float64) for k in ("dist", "hum", "amb", "refl", "emis")
        )
        self._hum_floor = {
            float(k): (100.0 if v is None else float(v)) for k, v in header["hum_thr"].items()
        }

        sizes, n_esc, n = header["sizes"], header["n_esc"], int(np.prod(self._shape))
        width = self._nt - 2
        offset = 0
        self._codes = np.frombuffer(payload, dtype=np.int8, count=sizes["codes"], offset=offset).reshape(n, width)
        offset += sizes["codes"]
        esc_pos = np.frombuffer(payload, dtype="<i8", count=n_esc, offset=offset)
        offset += sizes["esc_pos"]
        esc_val = np.frombuffer(payload, dtype="<i4", count=n_esc, offset=offset)
        offset += sizes["esc_val"]
        self._anchors = np.frombuffer(payload, dtype="<f4", count=2 * n, offset=offset).reshape(n, 2).astype(np.float64)
        self._escapes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if n_esc:
            curve_ids, cols = esc_pos // width, esc_pos % width
            for cid in np.unique(curve_ids):
                sel = curve_ids == cid
                self._escapes[int(cid)] = (cols[sel], esc_val[sel].astype(np.float64))

    def raw_at_temperature(self, params: MeasurementParams) -> np.ndarray:
        """Raw count reaching each temperature of the model's T grid, for these parameters."""
        amb_ids, amb_w = _lagrange_segment(self._amb, params.ambient_temp, _DEGREE["amb"], _AMB_BREAKS)
        dist_ids, dist_w = _lagrange_segment(self._dist, params.distance, _DEGREE["dist"], _DIST_BREAKS)
        refl_ids, refl_w = _lagrange(self._refl, params.reflection, _DEGREE["refl"])
        emis_ids, emis_w = _lagrange(self._emis, params.emissivity, _DEGREE["emis"])

        ids_parts, w_parts = [], []
        for ai, wa in zip(amb_ids, amb_w):
            # humidity has a dead zone below an ambient-dependent threshold; clamp per ambient node
            effective = max(float(params.humidity), self._hum_floor[float(self._amb[ai])])
            hum_ids, hum_w = _lagrange(self._hum, effective, _DEGREE["hum"])
            weights = wa * np.einsum("d,h,r,e->dhre", dist_w, hum_w, refl_w, emis_w)
            grid = np.meshgrid(dist_ids, hum_ids, [ai], refl_ids, emis_ids, indexing="ij")
            ids_parts.append(np.ravel_multi_index([g.ravel() for g in grid], self._shape))
            w_parts.append(weights.ravel())
        ids, w = np.concatenate(ids_parts), np.concatenate(w_parts)

        code_sum = w @ self._codes[ids].astype(np.float64)
        for k, cid in enumerate(ids):
            esc = self._escapes.get(int(cid))
            if esc is not None:
                cols, vals = esc
                code_sum[cols] += w[k] * (vals - np.clip(vals, -127, 127))
        r0, r1 = w @ self._anchors[ids, 0], w @ self._anchors[ids, 1]
        step = (r1 - r0) + self._q * np.cumsum(code_sum)
        curve = np.concatenate([[r0, r1], r1 + np.cumsum(step)])
        return np.maximum.accumulate(curve)

    def lookup_table(self, params: MeasurementParams) -> np.ndarray:
        """float32 temperature for every possible uint16 raw count; NaN outside the model's window."""
        curve = self.raw_at_temperature(params)
        counts = np.arange(65536, dtype=np.float64)
        table = np.interp(counts, curve, self._t_grid)
        table[(counts < curve[0]) | (counts > curve[-1])] = np.nan
        return table.astype(np.float32)


@functools.cache
def get_model() -> ThermalModel:
    return ThermalModel((resources.files(__package__) / MODEL_RESOURCE).read_bytes())


def validate_params(params: MeasurementParams) -> None:
    for name, (lo, hi) in _PARAM_RANGES.items():
        value = getattr(params, name)
        if not (lo <= value <= hi):
            hint = ""
            if name == "emissivity" and 0.1 <= value < MIN_EMISSIVITY:
                hint = f" (the pure-Python backend supports emissivity from {MIN_EMISSIVITY}; use --backend native for lower values)"
            raise ModelDomainError(f"{name}={value:g} is outside the supported range [{lo:g}, {hi:g}]{hint}")


def temperature_c(raw: np.ndarray, params: MeasurementParams) -> np.ndarray:
    """float32 Celsius array shaped like `raw`. Pixels outside the model's window (below about
    -45 C) are NaN."""
    validate_params(params)
    return get_model().lookup_table(params)[raw]
