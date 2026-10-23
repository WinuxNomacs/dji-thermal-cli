"""WGS84 geodesy for the mosaic: UTM projection and local metric frames.

utm() is the transverse Mercator projection by Krueger's series (Karney 2011, terms to
n^3: sub-millimetre within a UTM zone), utm_inverse() its inverse. local_to_utm() gives the affine map from a local
east/north frame in metres (true north) at a point to UTM there; it carries the meridian
convergence (grid north vs true north) and the projection's scale factor.
"""

from __future__ import annotations

import math

import numpy as np

A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)
K0 = 0.9996

_N = F / (2 - F)
_A_RECT = A / (1 + _N) * (1 + _N ** 2 / 4 + _N ** 4 / 64)
_ALPHA = (_N / 2 - 2 * _N ** 2 / 3 + 5 * _N ** 3 / 16, 13 * _N ** 2 / 48 - 3 * _N ** 3 / 5, 61 * _N ** 3 / 240)
_BETA = (_N / 2 - 2 * _N ** 2 / 3 + 37 * _N ** 3 / 96, _N ** 2 / 48 + _N ** 3 / 15, 17 * _N ** 3 / 480)
_E = math.sqrt(E2)


def utm_zone(lat: float, lon: float) -> tuple[int, bool]:
    """(zone number, northern hemisphere). The Norway/Svalbard exceptions are not applied."""
    return int((lon + 180) // 6) % 60 + 1, lat >= 0


def epsg(zone: int, north: bool) -> int:
    return (32600 if north else 32700) + zone


def utm(lat, lon, zone: int, north: bool):
    """(easting, northing) in metres of WGS84 lat/lon (degrees; scalars or arrays) in a zone."""
    phi = np.radians(lat)
    lam = np.radians(np.asarray(lon, dtype=float) - (zone * 6 - 183))
    s = np.sin(phi)
    t = np.sinh(np.arctanh(s) - _E * np.arctanh(_E * s))
    xi = np.arctan2(t, np.cos(lam))
    eta = np.arctanh(np.sin(lam) / np.sqrt(1 + t * t))
    x, y = eta.copy(), xi.copy()
    for j, a in enumerate(_ALPHA, start=1):
        x = x + a * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        y = y + a * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    return 500000.0 + K0 * _A_RECT * x, (0.0 if north else 10000000.0) + K0 * _A_RECT * y


def utm_inverse(e, n, zone: int, north: bool):
    """(lat, lon) in degrees of UTM easting/northing in metres (scalars or arrays) in a zone."""
    xi = (np.asarray(n, dtype=float) - (0.0 if north else 10000000.0)) / (K0 * _A_RECT)
    eta = (np.asarray(e, dtype=float) - 500000.0) / (K0 * _A_RECT)
    x, y = eta.copy(), xi.copy()
    for j, b in enumerate(_BETA, start=1):
        x = x - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        y = y - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    lam = np.arctan2(np.sinh(x), np.cos(y))
    conformal = np.sin(y) / np.hypot(np.sinh(x), np.cos(y))        # tan of the conformal latitude
    tau = conformal.copy()
    for _ in range(5):                                              # Newton on Karney's (2011) eq. 7
        sigma = np.sinh(_E * np.arctanh(_E * tau / np.sqrt(1 + tau * tau)))
        t = tau * np.sqrt(1 + sigma * sigma) - sigma * np.sqrt(1 + tau * tau)
        tau = tau + (conformal - t) / np.sqrt(1 + t * t) * (1 + (1 - E2) * tau * tau) / ((1 - E2) * np.sqrt(1 + tau * tau))
    return np.degrees(np.arctan(tau)), np.degrees(lam) + (zone * 6 - 183)


def local_to_utm(lat: float, lon: float, zone: int, north: bool) -> np.ndarray:
    """2x2 matrix J with d(E, N) = J @ d(east, north) for metre offsets around lat/lon."""
    s = math.sin(math.radians(lat))
    w = math.sqrt(1 - E2 * s * s)
    m = A * (1 - E2) / w ** 3                    # meridian radius of curvature
    n = A / w                                    # prime vertical radius of curvature
    dlat = math.degrees(1 / m)                   # degrees per metre north
    dlon = math.degrees(1 / (n * math.cos(math.radians(lat))))
    h = 1.0                                      # metres; the map is linear to ~1e-9 over this
    de = (np.array(utm(lat, lon + h * dlon, zone, north)) - np.array(utm(lat, lon - h * dlon, zone, north))) / (2 * h)
    dn = (np.array(utm(lat + h * dlat, lon, zone, north)) - np.array(utm(lat - h * dlat, lon, zone, north))) / (2 * h)
    return np.column_stack([de, dn])
