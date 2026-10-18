"""The constant tables the DJI Thermal SDK v1.8 libraries carry, rebuilt bit for bit.

Three kinds:

- generated: tables the SDK computes from a formula (fixed-point exp/log/sqrt/reciprocal
  tables of the hirp measurement core, the BT.601 YUV->RGB matrix, the iirp table axes).
  They are produced here from that formula;
- parameters: short sets of tuning constants (a 3x3 blur kernel, noise-reduction weights,
  atmospheric-transmission segments), written out below;
- data: measured calibration and designed palettes (emissivity corrections, default
  atmospheric-transmission tables, girp gray->temperature curves, the colour palettes), in
  data/dji_tsdk_v18.npz, stored in compact exact encodings (e.g. emissivity in 1/10000).

load() returns every table under its SDK name, in the SDK's own layout and type.
"""

from __future__ import annotations

import functools
import math
from decimal import Decimal
from pathlib import Path

import numpy as np

DATA_FILE = Path(__file__).with_name("data") / "dji_tsdk_v18.npz"

IIRP_PALETTES_YUV = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
IIRP_PALETTES_RGB = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11)


# --- generated ----------------------------------------------------------------------------

def _ladder(*runs: tuple[str, str, str]) -> np.ndarray:
    """Concatenated evenly stepped runs (start, step, last), as the doubles nearest the
    decimal values (what the C literals compile to)."""
    out = []
    for start, step, last in runs:
        v, s, e = Decimal(start), Decimal(step), Decimal(last)
        while v <= e:
            out.append(float(v))
            v += s
    return np.array(out, dtype="<f8")


def _rounded(values) -> np.ndarray:
    return np.array([math.floor(v + 0.5) for v in values], dtype=np.int64)


_DIST = (("0.25", "0.05", "1.2"), ("1.3", "0.1", "2.0"), ("2.2", "0.2", "4.0"), ("4.5", "0.5", "8.0"),
         ("9", "1", "14"), ("16", "2", "30"), ("35", "5", "50"))
_DIST_FAR = (("60", "10", "100"), ("120", "20", "300"), ("350", "50", "500"), ("600", "100", "1000"))


def _generated() -> dict[str, np.ndarray]:
    i16k, i1k, i256, i128 = np.arange(16384), np.arange(1024), np.arange(256), np.arange(128)
    t = {
        # iirp tau-table axes: target temperature (K) x distance (m)
        "target_temp_table": _ladder(("248.15", "25", "1623.15")),
        "dist_table": _ladder(*_DIST),
        "new_target_temp_table": _ladder(("248.15", "25", "1273.15")),
        "new_dist_table": _ladder(*_DIST, *_DIST_FAR),
        "target_temp_table_v3": _ladder(("248.15", "25", "823.15"), ("873.15", "50", "1873.15")),
        "dist_table_v3": _ladder(*_DIST, *_DIST_FAR),
        # iirp emissivity-correction axes: emissivity x target temperature (K)
        "org_ems_list_of_ems_table_v1": _ladder(("0.80", "0.01", "0.99")),
        "target_temp_list_of_ems_table_v1": _ladder(("223.15", "10", "1073.15")),
        "org_ems_list_of_ems_table_v2": _ladder(("0.800", "0.002", "0.998")),
        "target_temp_list_of_ems_table_v2": _ladder(("223.15", "10", "823.15"), ("873.15", "50", "1823.15")),
        # libv_dirp's BT.601 (limited range) YUV -> RGB: 3x3 matrix, then the Y/U/V offsets
        "dirp_y2r": np.array([1.1644, 0.0, 1.596, 1.1644, -0.3918, -0.813, 1.1644, 2.0172, 0.0,
                              16.0, 128.0, 128.0], dtype="<f8"),
        # libMicroTA fixed-point helpers
        "mt_exp_frac": _rounded(16384 * math.exp(i / 16384) for i in i16k).astype("<u2"),      # 2^14 e^(i/2^14)
        "mt_log": _rounded(16384 * math.log((1024 + i) / 1024) for i in i1k).astype("<u2"),    # 2^14 ln(1 + i/2^10)
        "mt_sqrt": _rounded(4096 * math.sqrt((1024 + i) / 512) for i in i1k).astype("<u2"),    # 2^12 sqrt((2^10+i)/2^9)
        "mt_sqrt64_a": _rounded(16384 * math.sqrt(128.5 + i) for i in i128).astype("<i8"),     # 2^14 sqrt(128.5+i)
        "mt_sqrt64_b": _rounded(2 ** 21 / math.sqrt(128.5 + i) for i in i128).astype("<i8"),   # 2^21 / sqrt(128.5+i)
        "mt_inv_a": np.array([2 ** 24 // (513 + 2 * i) for i in i256], dtype="<i8"),          # floor(2^23 / (256.5+i))
        "mt_inv_b": _rounded(2 ** 31 / (256.5 + i) ** 2 for i in i256).astype("<i8"),         # 2^31 / (256.5+i)^2
    }
    return t


# --- parameters ---------------------------------------------------------------------------

_PARAMETERS = {
    # libv_iirp spatial noise reduction: 3x3 blur kernel (sum 256)
    "isp_gauss3": np.array([13, 36, 13, 36, 60, 36, 13, 36, 13], dtype="<i4"),
    # libMicroTA CalcAtmTrsmit: piecewise-linear transmission per distance segment (Q14 slope/intercept)
    "mt_atm_k": np.array([16384, 16357, 16204, 16073, 15439, 14615, 13926, 12900, 11436, 10207], dtype="<i4"),
    "mt_atm_b": np.array([-78643, -69905, -50332, -41943, -21671, -11115, -6711, -3426, -1552, -765], dtype="<i4"),
    "mt_atm_lim": np.array([3, 8, 16, 32, 80, 160, 320, 800, 1600, 3200], dtype="<i4"),
}

# libv_iirp noise-reduction weights: two 6x16 blocks, each a centre weight row then five rows
_SNR_ROWS = (
    (12,), (7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5), (7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5),
    (6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 4, 4), (6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4),
    (5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3),
    (12,), (7, 7, 6, 6, 5, 4, 4, 3, 3, 2, 1, 1), (6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1),
    (5, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1), (4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1), (3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1),
)


def _snr_weights() -> np.ndarray:
    return np.array([v for row in _SNR_ROWS for v in row + (0,) * (16 - len(row))], dtype="<i4")


# --- data -----------------------------------------------------------------------------------

def _data() -> dict[str, np.ndarray]:
    with np.load(DATA_FILE) as z:
        d = {k: z[k] for k in z.files}
    t = {
        "ems_correct_table_v1": np.array([k / 100 for k in d["ems_correct_v1_centi"].ravel().tolist()], dtype="<f8"),
        "ems_correct_table_v2": np.array([k / 10000 for k in d["ems_correct_v2_permyriad"].ravel().tolist()], dtype="<f8"),
        "default_tau_gain0": d["default_tau_gain0"].ravel().astype("<u2"),
        "default_tau_gain1": d["default_tau_gain1"].ravel().astype("<u2"),
        "girp_luts": d["girp_luts"].ravel().astype("<i2"),
        "girp_palettes": d["girp_palettes_bgr"].ravel().astype("u1"),
        "dirp_palettes": (d["dirp_palettes_yuv_quarters"].astype("<f8") / 4.0).ravel(),
    }
    for n in IIRP_PALETTES_YUV:
        t[f"isp_yuv_{n}"] = d[f"iirp_palette_yuv_{n}"].ravel().astype("u1")
    for n in IIRP_PALETTES_RGB:
        t[f"isp_rgb_{n}"] = d[f"iirp_palette_rgb_{n}"].ravel().astype("u1")
    return t


@functools.cache
def load() -> dict[str, np.ndarray]:
    """Every SDK table by name (read-only arrays)."""
    t = {**_generated(), **_PARAMETERS, "isp_snr_weights": _snr_weights(), **_data()}
    for a in t.values():
        a.flags.writeable = False
    return t
