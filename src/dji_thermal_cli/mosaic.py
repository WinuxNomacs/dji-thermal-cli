"""Mosaic of a flight's thermal frames by direct georeferencing (level 1).

Each frame is placed from its own metadata only: the RTK camera position (DJI XMP
GpsLatitude/GpsLongitude/AbsoluteAltitude), the gimbal attitude (GimbalYaw/Pitch/RollDegree,
absolute, aerospace Z-Y-X order) and a pinhole lens whose focal length in pixels is the EXIF
focal length over the sensor's pixel pitch (SENSORS), or else comes from the EXIF 35 mm-equivalent
focal length, which the cameras round to a whole millimetre (about 1%). Rays are intersected with a horizontal ground plane, so
objects above or below that plane (roofs, trees) are displaced. Lens distortion is not
modelled: on a 21 m M4T flight it was not measurable from the overlaps (estimates from two
halves of the data disagreed in sign around zero), and the mosaic takes mostly frame centres.

The mosaic is a north-up UTM grid. Every mosaic pixel takes the temperature of exactly one
frame, the one that saw that ground point closest to straight down, by nearest-neighbour
sampling: values are the frames' own measurements, never blended or interpolated. Two more
bands record which frame each pixel came from and the off-nadir angle it was seen at.

build() also measures how well the frames agree where they overlap: the temperature
difference between the best and second-best frame per pixel (radiometric seams), and the
image shift between overlapping frames found by phase correlation (geometric error).
"""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import math
import threading
import warnings
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

from . import djixmp, files, geo, tiffout, tsdk, units

FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)
DEFAULT_MAX_PITCH = -80.0
HEIGHT_CELL_PX = 64             # the cell of the surface heights (see _align), in mosaic pixels

# Thermal sensors by EXIF Model: (native width in pixels, pixel pitch in mm), from DJI's spec
# pages (enterprise.dji.com/<product>/specs; XT S: its spec sheet). Super-resolution images
# (M30T, M3TD: 1280x1024 from a 640x512 sensor) have proportionally smaller pixels.
SENSORS = {
    "M4T": (640, 0.012),
    "M30T": (640, 0.012),
    "M3T": (640, 0.012),
    "M3TD": (640, 0.012),
    "H30T": (1280, 0.012),
    "ZH20T": (640, 0.012),
    "ZH20N": (640, 0.012),        # both its wide (12 mm) and tele (44.5 mm) thermal cameras
    "MAVIC2-ENTERPRISE-ADVANCED": (640, 0.012),
    "XT S": (640, 0.017),
}


def focal_length_px(model: str, width: int, height: int, focal_mm: float | None,
                    focal35_mm: float | None) -> tuple[float | None, str]:
    """The pinhole focal length in pixels and where it came from: the EXIF focal length over
    the sensor's pixel pitch when the camera is in SENSORS, else the 35 mm-equivalent one."""
    sensor = SENSORS.get(model)
    if sensor and focal_mm:
        native_width, pitch = sensor
        return focal_mm / pitch * width / native_width, "focal length / pixel pitch"
    if focal35_mm:
        return focal35_mm * math.hypot(width, height) / FULL_FRAME_DIAGONAL_MM, "35 mm-equivalent focal length"
    return None, ""


@dataclasses.dataclass
class Pose:
    lat: float
    lon: float
    abs_alt: float              # camera altitude, metres (DJI AbsoluteAltitude)
    yaw: float                  # gimbal attitude, degrees
    pitch: float
    roll: float
    focal_px: float
    width: int
    height: int
    ground_alt: float | None    # rangefinder target altitude (LRFTargetAbsAlt), when valid
    takeoff_alt: float | None   # AbsoluteAltitude - RelativeAltitude
    record: dict                # everything else worth keeping per frame


@dataclasses.dataclass
class Frame:
    name: str
    temperature: np.ndarray     # float32 degrees C, (height, width)
    pose: Pose


def _exif_number(value) -> float | None:
    """A positive, finite EXIF number, or None. A rational over 0 is NaN in Pillow, which float()
    raises for before Python 3.14."""
    try:
        f = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def read_pose(data: bytes, size: tuple[int, int] | None = None) -> tuple[Pose | None, str | None]:
    """The frame's camera pose from its DJI XMP and EXIF, or (None, why not), for an image of
    the frame's measured size (width, height): its R-JPEG's, read when not given. That differs
    from the JPEG's in super-resolution images (M4T: a 1280x1024 JPEG over 640x512 measurements)."""
    f = djixmp.fields(data)
    need = ("GpsLatitude", "GpsLongitude", "AbsoluteAltitude", "GimbalYawDegree", "GimbalPitchDegree",
            "GimbalRollDegree")
    values = {k: djixmp.number(f, k) for k in need}
    missing = [k for k, v in values.items() if v is None]
    if missing:
        return None, f"no camera pose in the DJI XMP (missing {', '.join(missing)})"
    image = Image.open(io.BytesIO(data))
    if size is None:
        try:
            rj = tsdk.open(data)
            size = rj.width, rj.height
        except tsdk.RJpegError:
            size = image.size
    width, height = size
    exif = image.getexif()
    lens = exif.get_ifd(0x8769)
    model = str(exif.get(272, "")).strip("\0 ")
    focal_px, focal_source = focal_length_px(model, width, height, _exif_number(lens.get(37386)),
                                             _exif_number(lens.get(41989)))
    if focal_px is None:
        return None, "no focal length in the EXIF"
    rel = djixmp.number(f, "RelativeAltitude")
    lrf_ok = f.get("LRFStatus", "").lower() == "normal"
    record = {k: f[k] for k in (
        "UTCAtExposure", "GpsStatus", "AltitudeType", "RtkFlag", "RtkStdLat", "RtkStdLon", "RtkStdHgt",
        "RelativeAltitude", "FlightYawDegree", "FlightPitchDegree", "FlightRollDegree", "LRFStatus",
        "LRFTargetDistance", "LRFTargetAbsAlt", "CameraSerialNumber", "DroneModel") if k in f}
    record.update({"focal_px": round(focal_px, 2), "focal_source": focal_source})
    return Pose(
        lat=values["GpsLatitude"], lon=values["GpsLongitude"], abs_alt=values["AbsoluteAltitude"],
        yaw=values["GimbalYawDegree"], pitch=values["GimbalPitchDegree"], roll=values["GimbalRollDegree"],
        focal_px=focal_px,
        width=width, height=height,
        ground_alt=djixmp.number(f, "LRFTargetAbsAlt") if lrf_ok else None,
        takeoff_alt=values["AbsoluteAltitude"] - rel if rel is not None else None,
        record=record,
    ), None


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Camera body (forward, right, down) -> north/east/down, aerospace Z-Y-X angles in degrees."""
    y, p, r = map(math.radians, (yaw, pitch, roll))
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    return rz @ ry @ rx


@dataclasses.dataclass
class _Placed:
    """A frame placed over the ground plane, with its alignment correction: the frame is
    sampled at X + shift + rotation * perp(X - c) for mosaic point X (level 2)."""
    frame: Frame
    h: float                    # camera height above the plane, metres
    c: np.ndarray               # camera position, UTM (E, N)
    j: np.ndarray               # local east/north metres -> UTM offsets
    j_inv: np.ndarray           # UTM offsets -> local east/north metres
    r_t: np.ndarray             # NED -> camera body
    cx: float
    cy: float
    footprint: np.ndarray       # (k, 2) UTM outline, before the correction
    shift: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(2))
    rot: float = 0.0            # radians
    heights: Heights | None = None      # the surface the frame is projected onto (None: the plane)

    def corrected_footprint(self) -> np.ndarray:
        fp = self.footprint
        if self.heights is not None and (low := float(self.heights.z.min())) < 0:
            fp = self.c + (fp - self.c) * self.h / (self.h - low)    # rays reach further below the plane
        d = fp - self.c
        return fp - self.shift - self.rot * np.column_stack([-d[:, 1], d[:, 0]])


def _boresight(pitch: float, yaw: float) -> np.ndarray:
    """A small camera-body rotation (radians): pitch about the right axis, then yaw about the
    body's down axis. For a camera looking straight down these are the two tilts that move the
    image on the ground (body roll would spin it about the optical axis)."""
    ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return ry @ rz


def _place(frame: Frame, ground: float, zone: int, north: bool, bias: tuple[float, float] = (0.0, 0.0)) -> _Placed | None:
    p = frame.pose
    h = p.abs_alt - ground
    if h <= 0:
        return None
    r = rotation(p.yaw, p.pitch, p.roll) @ _boresight(*bias)
    cx, cy = (p.width - 1) / 2, (p.height - 1) / 2
    c = np.array(geo.utm(p.lat, p.lon, zone, north))
    j = geo.local_to_utm(p.lat, p.lon, zone, north)
    edge = np.linspace(0, 1, 9)
    u = np.concatenate([edge * (p.width - 1), np.full(9, p.width - 1.0), (1 - edge) * (p.width - 1), np.zeros(9)])
    v = np.concatenate([np.zeros(9), edge * (p.height - 1), np.full(9, p.height - 1.0), (1 - edge) * (p.height - 1)])
    rays = r @ np.stack([np.ones_like(u), (u - cx) / p.focal_px, (v - cy) / p.focal_px])
    if (rays[2] <= 1e-6).any():
        return None             # part of the frame sees the horizon: no finite footprint
    t = h / rays[2]
    local = np.stack([rays[1] * t, rays[0] * t])          # east, north
    return _Placed(frame, h, c, j, np.linalg.inv(j), r.T, cx, cy, (c[:, None] + j @ local).T)


def _centre_ground(pl: _Placed) -> np.ndarray:
    """UTM point the frame's optical axis hits."""
    ray = pl.r_t.T[:, 0]
    return pl.c + pl.j @ (np.array([ray[1], ray[0]]) * pl.h / ray[2])


@dataclasses.dataclass
class Grid:
    e0: float                   # UTM of the top-left corner of the top-left pixel
    n0: float
    gsd: float
    width: int
    height: int
    zone: int
    north: bool

    @property
    def epsg(self) -> int:
        return geo.epsg(self.zone, self.north)


@dataclasses.dataclass(frozen=True)
class Heights:
    """The surface above (or below) the ground plane, z metres, on a north-up UTM grid of cells:
    z[i, j] is the height at the cell centre (e0 + (j + 0.5) cell, n0 - (i + 0.5) cell),
    interpolated bilinearly between centres and held constant beyond the outer ones."""
    e0: float
    n0: float
    cell: float
    z: np.ndarray               # (rows, cols) float64

    @classmethod
    def flat(cls, grid: Grid, cell: float) -> Heights:
        """Zero over the cells covering grid."""
        return cls(grid.e0, grid.n0, cell, np.zeros((max(1, math.ceil(grid.height * grid.gsd / cell)),
                                                     max(1, math.ceil(grid.width * grid.gsd / cell)))))

    def _weights(self, coord: np.ndarray, axis: int):
        """For 1-D coordinates along an axis (0: north, rows; 1: east, columns): the lower cell
        index and the weight of the upper one."""
        count = self.z.shape[axis]
        f = ((coord - self.e0) if axis else (self.n0 - coord)) / self.cell - 0.5
        f = np.clip(f, 0, count - 1)
        i0 = np.minimum(np.floor(f), max(count - 2, 0)).astype(np.intp)
        return i0, f - i0 if count > 1 else np.zeros_like(f)

    def _matrix(self, coord: np.ndarray, axis: int) -> np.ndarray:
        i0, w = self._weights(coord, axis)
        m = np.zeros((coord.size, self.z.shape[axis]))
        at = np.arange(coord.size)
        m[at, i0] = 1 - w
        m[at, np.minimum(i0 + 1, self.z.shape[axis] - 1)] += w
        return m

    def at(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        """The heights at every (n[i], e[j]) for 1-D e and n: an (n.size, e.size) array."""
        return self._matrix(n, 0) @ self.z @ self._matrix(e, 1).T

    def point_weights(self, e: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For points (e[k], n[k]): the 4 cells (flat indices) their height interpolates and the
        weights, each (k, 4)."""
        i0, wi = self._weights(n, 0)
        j0, wj = self._weights(e, 1)
        rows, cols = self.z.shape
        i1, j1 = np.minimum(i0 + 1, rows - 1), np.minimum(j0 + 1, cols - 1)
        idx = np.column_stack([i0 * cols + j0, i0 * cols + j1, i1 * cols + j0, i1 * cols + j1])
        w = np.column_stack([(1 - wi) * (1 - wj), (1 - wi) * wj, wi * (1 - wj), wi * wj])
        return idx, w

    def at_points(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        idx, w = self.point_weights(e, n)
        return (self.z.ravel()[idx] * w).sum(1)


@dataclasses.dataclass(frozen=True)
class Dsm:
    """A surface model read from a GeoTIFF (see read_dsm): altitudes in metres on a north-up
    raster, pixel (i, j) centred at (x0 + (j + 0.5) dx, y0 - (i + 0.5) dy) in its CRS, which
    is WGS84 UTM (epsg 326zz/327zz: x, y = easting, northing) or WGS84 lat/lon (epsg 4326:
    x, y = longitude, latitude). NaN where it has no data."""
    path: Path
    z: np.ndarray               # (rows, cols) float64
    x0: float
    y0: float
    dx: float
    dy: float
    epsg: int

    def pixel_m(self) -> float:
        """The pixel size in metres (roughly, for lat/lon)."""
        if self.epsg != 4326:
            return self.dx
        lat = self.y0 - self.z.shape[0] * self.dy / 2
        return self.dx * 111_320.0 * max(math.cos(math.radians(lat)), 0.01)

    def at_utm(self, e: np.ndarray, n: np.ndarray, zone: int, north: bool) -> np.ndarray:
        """The altitude at UTM points (nearest pixel), NaN off the raster."""
        if self.epsg == geo.epsg(zone, north):
            x, y = e, n
        else:
            lat, lon = geo.utm_inverse(e, n, zone, north)
            if self.epsg == 4326:
                x, y = lon, lat
            else:
                x, y = geo.utm(lat, lon, self.epsg % 100, self.epsg < 32700)
        j = np.floor((np.asarray(x) - self.x0) / self.dx).astype(np.intp)
        i = np.floor((self.y0 - np.asarray(y)) / self.dy).astype(np.intp)
        rows, cols = self.z.shape
        inside = (i >= 0) & (i < rows) & (j >= 0) & (j < cols)
        return np.where(inside, self.z[np.clip(i, 0, rows - 1), np.clip(j, 0, cols - 1)], np.nan)


def read_dsm(path: Path) -> Dsm:
    """A DSM GeoTIFF's first band, as written by ODM, GDAL and the like: north-up (ModelPixelScale +
    one ModelTiepoint), in WGS84 UTM or lat/lon, altitudes in metres, GDAL_NODATA for no data."""
    import tifffile

    path = Path(path)
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            tags = {t.code: t.value for t in page.tags.values()}
            z = page.asarray()
    except (OSError, ValueError, tifffile.TiffFileError) as exc:
        raise ValueError(f"{path}: not a readable GeoTIFF ({exc})") from exc
    if 33550 not in tags or 33922 not in tags:
        raise ValueError(f"{path}: no georeference (ModelPixelScale and ModelTiepoint): a transformed "
                         "(rotated) raster is not supported")
    if z.ndim == 3:
        z = z[..., 0] if page.planarconfig == 1 else z[0]      # contiguous: (rows, cols, bands)
    z = z.astype(np.float64)
    dx, dy = float(tags[33550][0]), float(tags[33550][1])
    tie = [float(v) for v in tags[33922][:6]]
    keys = {}
    gk = list(tags.get(34735, ()))
    for k in range(4, 4 + 4 * int(gk[3]) if len(gk) >= 4 else 4, 4):
        keys[gk[k]] = gk[k + 3] if gk[k + 1] == 0 else None
    if keys.get(1024) == 2 and keys.get(2048) == 4326:
        epsg = 4326
    elif keys.get(1024) == 1 and keys.get(3072) and 32601 <= keys[3072] <= 32760 and keys[3072] % 100 <= 60 \
            and keys[3072] // 100 in (326, 327):
        epsg = int(keys[3072])
    else:
        raise ValueError(f"{path}: the DSM's CRS must be WGS84 UTM (EPSG:326zz/327zz) or WGS84 lat/lon (EPSG:4326)")
    # the tiepoint maps raster (i, j) to model (x, y); PixelIsPoint puts it at the pixel's centre
    half = 0.5 if keys.get(1025) == 2 else 0.0
    x0, y0 = tie[3] - (tie[0] + half) * dx, tie[4] + (tie[1] + half) * dy
    nodata = tags.get(42113)
    if nodata not in (None, ""):
        try:
            z[z == float(str(nodata).strip("\x00 "))] = np.nan
        except ValueError:
            pass
    z[~np.isfinite(z)] = np.nan
    return Dsm(path, z, x0, y0, dx, dy, epsg)


def _dsm_surface(dsm: Dsm, placed: list[_Placed], grid: Grid, cell: float, ground: float,
                 offset: float | None) -> tuple[Heights, dict]:
    """The DSM on the height cells, as heights above the ground plane, and its report. Each
    cell is the median of the DSM over the cell (sampled on a sub-grid about as fine as the
    DSM's pixels, 2-8 per side); cells the DSM doesn't cover are the plane. offset: added to
    the DSM's altitudes to bring them to the frames' (None: the median of the rangefinder
    targets' altitudes minus the DSM there; 0 without a rangefinder)."""
    surface = Heights.flat(grid, cell)
    rows, cols = surface.z.shape
    sub = int(np.clip(round(cell / dsm.pixel_m()), 2, 8))
    f = (np.arange(sub) + 0.5) / sub
    e = surface.e0 + (np.arange(cols)[:, None] + f).ravel() * cell
    n = surface.n0 - (np.arange(rows)[:, None] + f).ravel() * cell
    ee, nn = np.meshgrid(e, n)
    alt = dsm.at_utm(ee, nn, grid.zone, grid.north).reshape(rows, sub, cols, sub)
    with warnings.catch_warnings():       # cells off the DSM: all-NaN slices
        warnings.simplefilter("ignore", RuntimeWarning)
        cells = np.nanmedian(alt.transpose(0, 2, 1, 3).reshape(rows, cols, sub * sub), axis=2)
    covered = np.isfinite(cells)
    if not covered.any():
        raise ValueError(f"{dsm.path}: the DSM does not cover the flight")
    anchors = _anchors(placed, ground)
    at_anchors = dsm.at_utm(anchors[:, 0], anchors[:, 1], grid.zone, grid.north) if len(anchors) else np.zeros(0)
    on_dsm = np.isfinite(at_anchors)
    if offset is not None:
        source = "given"
    elif on_dsm.any():
        offset = float(np.median(anchors[on_dsm, 2] + ground - at_anchors[on_dsm]))
        source = f"median rangefinder target altitude minus the DSM, at {int(on_dsm.sum())} targets"
    else:
        offset, source = 0.0, "none: no rangefinder targets on the DSM, its altitudes taken as the frames'"
    z = np.where(covered, cells + offset - ground, 0.0)
    surface = dataclasses.replace(surface, z=z)
    resid = surface.at_points(anchors[on_dsm, 0], anchors[on_dsm, 1]) - anchors[on_dsm, 2] if on_dsm.any() else None
    report = {
        "method": "DSM", "file": dsm.path.name, "offset_m": round(offset, 3) + 0.0, "offset_source": source,
        "cell_m": round(cell, 3), "cells": int(z.size), "observed_cells": int(covered.sum()),
        "range_m": [round(float(z[covered].min()), 2) + 0.0, round(float(z[covered].max()), 2) + 0.0],
        "anchors": int(on_dsm.sum()),
        "anchor_residual_m": round(float(np.median(np.abs(resid))), 3) if resid is not None else None,
    }
    return surface, report


def _sample(pl: _Placed, grid: Grid, rows: slice, cols: slice, bilinear: bool = False, pixel: bool = False):
    """For grid pixels in rows x cols: (temperature, off-nadir angle in degrees, valid mask),
    and with pixel=True the frame's pixel each was read from (column, row; nearest only).
    Nearest neighbour, so mosaic values are the frames' own; bilinear is for measuring
    misalignment only (nearest-neighbour renderings carry a staircase pattern tied to each
    frame's pixel grid, which biases image matching between frames)."""
    p = pl.frame.pose
    e = grid.e0 + (np.arange(cols.start, cols.stop) + 0.5) * grid.gsd
    n = grid.n0 - (np.arange(rows.start, rows.stop) + 0.5) * grid.gsd
    de, dn = np.meshgrid(e - pl.c[0], n - pl.c[1])
    if pl.rot or pl.shift.any():
        de, dn = de + pl.shift[0] - pl.rot * dn, dn + pl.shift[1] + pl.rot * de
    east = pl.j_inv[0, 0] * de + pl.j_inv[0, 1] * dn
    north = pl.j_inv[1, 0] * de + pl.j_inv[1, 1] * dn
    # the camera's height over the surface at each point (taken where the point is in the
    # mosaic, before the correction)
    hz = pl.h if pl.heights is None else np.maximum(pl.h - pl.heights.at(e, n), 0.1)
    rt = pl.r_t
    fwd = rt[0, 0] * north + rt[0, 1] * east + rt[0, 2] * hz
    right = rt[1, 0] * north + rt[1, 1] * east + rt[1, 2] * hz
    down = rt[2, 0] * north + rt[2, 1] * east + rt[2, 2] * hz
    with np.errstate(divide="ignore", invalid="ignore"):
        uf = pl.cx + p.focal_px * right / fwd
        vf = pl.cy + p.focal_px * down / fwd
    angle = np.degrees(np.arctan2(np.hypot(east, north), hz)).astype(np.float32)
    temperature = pl.frame.temperature
    if bilinear:
        valid = (fwd > 0) & (uf >= 0) & (uf <= p.width - 1) & (vf >= 0) & (vf <= p.height - 1)
        u0 = np.clip(np.floor(np.where(valid, uf, 0)), 0, p.width - 2).astype(np.intp)
        v0 = np.clip(np.floor(np.where(valid, vf, 0)), 0, p.height - 2).astype(np.intp)
        fu, fv = np.where(valid, uf, 0) - u0, np.where(valid, vf, 0) - v0
        t = ((temperature[v0, u0] * (1 - fu) + temperature[v0, u0 + 1] * fu) * (1 - fv)
             + (temperature[v0 + 1, u0] * (1 - fu) + temperature[v0 + 1, u0 + 1] * fu) * fv)
        return t.astype(np.float32), angle, valid
    u, v = np.floor(uf + 0.5), np.floor(vf + 0.5)
    valid = (fwd > 0) & (u >= 0) & (u < p.width) & (v >= 0) & (v < p.height)
    iu, iv = np.where(valid, u, 0).astype(np.intp), np.where(valid, v, 0).astype(np.intp)
    t = temperature[iv, iu]
    valid &= t > np.float32(tiffout.NO_DATA_C)          # a no-data pixel leaves the spot to other frames
    if pixel:
        return t, angle, valid, u.astype(np.float32), v.astype(np.float32)
    return t, angle, valid


def _window(pl: _Placed, grid: Grid) -> tuple[slice, slice]:
    fp = pl.corrected_footprint()
    c0 = max(0, int((fp[:, 0].min() - grid.e0) / grid.gsd) - 1)
    c1 = min(grid.width, int((fp[:, 0].max() - grid.e0) / grid.gsd) + 2)
    r0 = max(0, int((grid.n0 - fp[:, 1].max()) / grid.gsd) - 1)
    r1 = min(grid.height, int((grid.n0 - fp[:, 1].min()) / grid.gsd) + 2)
    return slice(r0, r1), slice(c0, c1)


@dataclasses.dataclass
class Mosaic:
    grid: Grid
    temperature: np.ndarray     # float32, NaN where no frame covers
    source: np.ndarray          # float32 frame index, NaN where none
    view_angle: np.ndarray      # float32 off-nadir degrees, NaN where none
    frames: list[dict]          # per-frame table (index = source band value)
    ground_alt: float
    ground_alt_source: str
    report: dict
    heights: Heights | None = None      # the surface the frames were projected onto (None: the plane)
    source_px: np.ndarray | None = None  # float32 (height, width, 2): the source frame's column and row, NaN where none


def _grid(placed: list[_Placed], gsd: float, zone: int, north: bool) -> Grid:
    allfp = np.concatenate([pl.corrected_footprint() for pl in placed])
    e0 = math.floor(allfp[:, 0].min() / gsd) * gsd
    n0 = math.ceil(allfp[:, 1].max() / gsd) * gsd
    return Grid(e0, n0, gsd, math.ceil((allfp[:, 0].max() - e0) / gsd), math.ceil((n0 - allfp[:, 1].min()) / gsd),
                zone, north)


def build(frames: list[Frame], gsd: float | None = None, ground_alt: float | None = None,
          check_misalignment: bool = True, align: bool = True, normalize: bool = False,
          progress: Callable[..., None] | None = None, jobs: int = 1, heights: str = "auto",
          height_cell: int = HEIGHT_CELL_PX, dsm: Dsm | None = None, dsm_offset: float | None = None) -> Mosaic:
    """Composite the frames, projected onto the ground plane. align: first correct the
    frames' placement from their overlaps (see _align), and with heights="auto" estimate the
    surface's heights above the plane (roofs, trees) on cells height_cell pixels wide and
    project onto that instead ("plane": the plane only). dsm: project onto this surface model
    instead (then alignment only corrects the frames; see _dsm_surface for dsm_offset). normalize:
    remove each frame's temperature offset against its neighbours (see _frame_offsets).
    progress: called as progress(stage, done, total, unit, **extra) as the "align",
    "composite" and "check" stages go (see api.Event); it doesn't change the result. jobs:
    threads matching overlaps (see _measure_blocks); nor does it."""
    layout = _layout(frames, gsd, ground_alt, align, progress, jobs, heights, height_cell, dsm, dsm_offset)
    composite = _reporter(progress, "composite", "frames")
    if composite:
        composite(0, len(layout.placed))
    m = _composite(layout, layout.placed, normalize, composite, 0, len(layout.placed))
    m.report["misalignment"] = _check(layout, check_misalignment, progress, jobs)
    return m


def build_both(frames: list[Frame], as_converted: list[Frame], gsd: float | None = None,
               ground_alt: float | None = None, check_misalignment: bool = True, align: bool = True,
               progress: Callable[..., None] | None = None, jobs: int = 1, heights: str = "auto",
               height_cell: int = HEIGHT_CELL_PX, dsm: Dsm | None = None,
               dsm_offset: float | None = None) -> tuple[Mosaic, Mosaic]:
    """The normalized mosaic of frames (as build(normalize=True)) and the mosaic of the same
    frames measured another way (as_converted, matched by name), laid out once: the second
    gets the first's placement, alignment and misalignment check, and only its compositing
    is its own (a frame missing from as_converted is left out of it)."""
    layout = _layout(frames, gsd, ground_alt, align, progress, jobs, heights, height_cell, dsm, dsm_offset)
    by_name = {f.name: f for f in as_converted}
    other = [dataclasses.replace(pl, frame=by_name[pl.frame.name]) for pl in layout.placed
             if pl.frame.name in by_name]
    total = len(layout.placed) + len(other)
    composite = _reporter(progress, "composite", "frames")
    if composite:
        composite(0, total)
    normalized = _composite(layout, layout.placed, True, composite, 0, total)
    converted = _composite(layout, other, False, composite, len(layout.placed), total)
    normalized.report["misalignment"] = _check(layout, check_misalignment, progress, jobs)
    converted.report["misalignment"] = copy.deepcopy(normalized.report["misalignment"])
    return normalized, converted


@dataclasses.dataclass
class _Layout:
    placed: list[_Placed]
    dropped: list[str]          # frames with no ground footprint
    grid: Grid
    alignment: dict | None
    ground_alt: float
    ground_source: str
    heights: Heights | None = None
    heights_report: dict | None = None


def _reporter(progress, stage, unit, **extra):
    return (lambda done, total: progress(stage, done, total, unit, **extra)) if progress else None


def _layout(frames, gsd, ground_alt, align, progress, jobs, heights="auto", height_cell=HEIGHT_CELL_PX,
            dsm=None, dsm_offset=None) -> _Layout:
    """The frames placed over the ground plane (and aligned) and the grid they cover."""
    if not frames:
        raise ValueError("no frames to mosaic")
    lat = float(np.mean([f.pose.lat for f in frames]))
    lon = float(np.mean([f.pose.lon for f in frames]))
    zone, north = geo.utm_zone(lat, lon)

    if ground_alt is not None:
        ground_source = "given"
    elif lrf := [f.pose.ground_alt for f in frames if f.pose.ground_alt is not None]:
        ground_alt, ground_source = float(np.median(lrf)), "median laser-rangefinder target altitude"
    elif takeoff := [f.pose.takeoff_alt for f in frames if f.pose.takeoff_alt is not None]:
        ground_alt, ground_source = float(np.median(takeoff)), "take-off altitude"
    else:
        raise ValueError("no ground altitude: no rangefinder or relative-altitude data")

    placed, dropped = [], []
    for f in frames:
        pl = _place(f, ground_alt, zone, north)
        (placed if pl else dropped).append(pl or f.name)
    if not placed:
        raise ValueError("no frame can be projected onto the ground plane")

    if gsd is None:
        gsd = float(np.median([pl.h / pl.frame.pose.focal_px for pl in placed]))
        gsd = max(round(gsd, 3), 0.001)
    grid = _grid(placed, gsd, zone, north)

    alignment = surface = heights_report = None
    if dsm is not None:
        surface, heights_report = _dsm_surface(dsm, placed, grid, height_cell * gsd, ground_alt, dsm_offset)
        for pl in placed:
            pl.heights = surface
    elif not align:
        heights_report = {"method": "plane", "reason": "the heights are estimated when aligning"}
    if align:
        placed, surface, alignment = _align(frames, placed, grid, ground_alt, zone, north, progress=progress,
                                            jobs=jobs, heights=heights, height_cell=height_cell, fixed=surface)
        if heights_report:
            alignment["heights"] = heights_report
        heights_report = alignment["heights"]
        grid = _grid(placed, gsd, zone, north)
    return _Layout(placed, dropped, grid, alignment, ground_alt, ground_source, surface, heights_report)


def _check(layout: _Layout, check_misalignment: bool, progress, jobs) -> dict | None:
    """How well the laid-out frames still agree where they overlap (the "check" stage)."""
    if not check_misalignment:
        return None
    return _summary(_measure_blocks(layout.placed, layout.grid, report=_reporter(progress, "check", "pairs"),
                                    jobs=jobs), layout.grid)


def _composite(layout: _Layout, placed: list[_Placed], normalize: bool, composite, done: int,
               total: int) -> Mosaic:
    """Each mosaic pixel from the frame seeing it closest to nadir (the runner-up kept for the
    seams and offsets), without the misalignment check. composite: reported as (done + frames
    composited, total)."""
    grid = layout.grid
    shape = (grid.height, grid.width)
    best_angle = np.full(shape, np.inf, np.float32)
    second_angle = np.full(shape, np.inf, np.float32)
    temp = np.full(shape, np.nan, np.float32)
    second = np.full(shape, np.nan, np.float32)
    source = np.full(shape, -1, np.int32)
    second_source = np.full(shape, -1, np.int32)
    source_px = np.full(shape + (2,), np.nan, np.float32)
    for k, pl in enumerate(placed):
        rows, cols = _window(pl, grid)
        t, angle, valid, u, v = _sample(pl, grid, rows, cols, pixel=True)
        angle = np.where(valid, angle, np.inf)
        ba, sa = best_angle[rows, cols], second_angle[rows, cols]
        bt, st, bs, ss = temp[rows, cols], second[rows, cols], source[rows, cols], second_source[rows, cols]
        better = angle < ba
        runner = ~better & (angle < sa)
        sa[:] = np.where(better, ba, np.where(runner, angle, sa))
        st[:] = np.where(better, bt, np.where(runner, t, st))
        ss[:] = np.where(better, bs, np.where(runner, k, ss))
        px = source_px[rows, cols]
        px[:] = np.where(better[..., None], np.stack([u, v], axis=-1), px)
        ba[:] = np.where(better, angle, ba)
        bt[:] = np.where(better, t, bt)
        bs[:] = np.where(better, k, bs)
        if composite:
            composite(done + k + 1, total)

    covered = source >= 0
    offsets, radiometry = _frame_offsets(temp, second, source, second_source, len(placed))
    if normalize:
        has_second = second_source >= 0
        temp[covered] -= offsets[source[covered]]
        second[has_second] -= offsets[second_source[has_second]]
    radiometry["normalized"] = normalize
    used = np.bincount(source[covered], minlength=len(placed))
    table = []
    for k, pl in enumerate(placed):
        p = pl.frame.pose
        table.append({
            "index": k, "file": pl.frame.name, "pixels_used": int(used[k]),
            "lat": p.lat, "lon": p.lon, "abs_alt_m": p.abs_alt, "height_above_ground_m": round(pl.h, 3),
            "utm_e": round(float(pl.c[0]), 3), "utm_n": round(float(pl.c[1]), 3),
            "gimbal_yaw": p.yaw, "gimbal_pitch": p.pitch, "gimbal_roll": p.roll,
            "focal_px": round(p.focal_px, 2), "width": p.width, "height": p.height,
            "alignment_shift_m": [round(float(v), 3) for v in pl.shift],
            "alignment_rotation_deg": round(math.degrees(pl.rot), 3),
            "temperature_offset_c": round(float(offsets[k]), 3),
            **p.record,
        })
    report = {
        "frames_used": len(placed),
        "frames_contributing": int((used > 0).sum()),
        "frames_dropped_no_ground_footprint": layout.dropped,
        "radiometric_seams": _seam_stats(temp, second, covered),
        "frame_offsets": radiometry,
        "alignment": layout.alignment,
        "heights": layout.heights_report,
        "misalignment": None,
    }
    return Mosaic(
        grid, temp, np.where(covered, source, np.nan).astype(np.float32),
        np.where(covered, best_angle, np.nan).astype(np.float32), table, layout.ground_alt, layout.ground_source,
        report, layout.heights, source_px,
    )


def _seam_stats(temp, second, covered) -> dict:
    """Best minus second-best frame temperature where two frames see the same ground."""
    both = covered & np.isfinite(second) & (temp > tiffout.NO_DATA_C + 1) & (second > tiffout.NO_DATA_C + 1)
    d = (temp - second)[both]
    if d.size == 0:
        return {"pixels": 0}
    a = np.abs(d)
    return {"pixels": int(d.size), "median_abs_c": round(float(np.median(a)), 3),
            "p95_abs_c": round(float(np.percentile(a, 95)), 3), "mean_c": round(float(d.mean()), 3)}


def _frame_offsets(temp, second, source, second_source, n: int, max_samples: int = 400_000):
    """Each frame's temperature offset against the frames overlapping it, from the pixels two
    frames both see: T_best - T_second = o_best - o_second, solved by robust (Huber) least
    squares with the median offset fixed at 0. Pixels at strong edges are left out, as
    residual misalignment there shows up as a temperature difference.

    Uncooled thermal cameras drift between their periodic shutter recalibrations (flat-field
    corrections), so a frame's offset typically rises steadily and drops back at each one;
    the embedded ambient temperature, which DJI's distance correction stretches the
    temperatures about, adds per-frame offsets of its own."""
    gy, gx = np.gradient(np.nan_to_num(temp, nan=0.0))
    ok = ((source >= 0) & (second_source >= 0) & np.isfinite(second) & (np.hypot(gx, gy) < 0.3)
          & (temp > tiffout.NO_DATA_C + 1) & (second > tiffout.NO_DATA_C + 1))
    idx = np.flatnonzero(ok)
    if idx.size < 100:
        return np.zeros(n), {"pixels": int(idx.size)}
    if idx.size > max_samples:
        idx = np.random.default_rng(0).choice(idx, max_samples, replace=False)
    a, b = second_source.ravel()[idx], source.ravel()[idx]
    d = (temp.ravel()[idx] - second.ravel()[idx]).astype(np.float64)
    w = np.ones_like(d)
    o = np.zeros(n)
    for _ in range(8):
        ata = np.zeros((n, n))
        atb = np.zeros(n)
        np.add.at(ata, (b, b), w)
        np.add.at(ata, (a, a), w)
        np.add.at(ata, (a, b), -w)
        np.add.at(ata, (b, a), -w)
        np.add.at(atb, b, w * d)
        np.add.at(atb, a, -w * d)
        o = np.linalg.solve(ata + 1e-6 * np.eye(n), atb)
        r = np.abs(d - (o[b] - o[a]))
        scale = 1.4826 * np.median(r) + 1e-6
        w = np.where(r <= 2 * scale, 1.0, 2 * scale / np.maximum(r, 1e-12))
    o -= np.median(o)
    after = np.abs(d - (o[b] - o[a]))
    return o, {
        "pixels": int(idx.size),
        "overlap_difference_c": {"median_abs": round(float(np.median(np.abs(d))), 3),
                                 "median_abs_after_offsets": round(float(np.median(after)), 3)},
        "offset_c": {"min": round(float(o.min()), 3), "max": round(float(o.max()), 3),
                     "p5_p95_span": round(float(np.percentile(o, 95) - np.percentile(o, 5)), 3)},
    }


def _phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For stacks of blocks (k, h, w): the shift (dy, dx) of each b relative to its a by phase
    correlation, and the peak strength."""
    k, h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.rfft2((a - a.mean(axis=(1, 2), keepdims=True)) * win)
    fb = np.fft.rfft2((b - b.mean(axis=(1, 2), keepdims=True)) * win)
    cross = fa.conj() * fb
    corr = np.fft.irfft2(cross / (np.abs(cross) + 1e-12), s=(h, w))
    iy, ix = np.unravel_index(np.argmax(corr.reshape(k, -1), axis=1), (h, w))
    at = np.arange(k)
    peak = corr[at, iy, ix]

    def sub(c, l, r, i):        # parabolic sub-pixel refinement
        den = l - 2 * c + r
        with np.errstate(divide="ignore", invalid="ignore"):
            return i + np.where(den != 0, 0.5 * (l - r) / den, 0.0)

    dy = sub(peak, corr[at, (iy - 1) % h, ix], corr[at, (iy + 1) % h, ix], iy)
    dx = sub(peak, corr[at, iy, (ix - 1) % w], corr[at, iy, (ix + 1) % w], ix)
    dy = np.where(dy > h / 2, dy - h, dy)
    dx = np.where(dx > w / 2, dx - w, dx)
    return dy, dx, peak


def _measure_blocks(placed: list[_Placed], grid: Grid, block: int = 128, min_peak: float = 0.1,
                    report: Callable[[int, int], None] | None = None, jobs: int = 1) -> np.ndarray:
    """Misalignment between overlapping frames, in blocks (half-overlapping) fully covered by
    both frames of a pair. Rows: (a, b, block centre E, N, shift E, N), the shift being where
    frame b shows the block's content relative to frame a, in metres. report(done, total):
    the overlapping pairs measured, counted by the overlap area each samples so that the
    count moves evenly. jobs: threads matching pairs at once (numpy's FFTs and array work
    release the GIL); the result is the same for any number."""
    step = block // 2
    windows = [_window(pl, grid) for pl in placed]
    overlaps = []
    for a in range(len(placed)):
        for b in range(a + 1, len(placed)):
            (ra, ca), (rb, cb) = windows[a], windows[b]
            rows = slice(max(ra.start, rb.start), min(ra.stop, rb.stop))
            cols = slice(max(ca.start, cb.start), min(ca.stop, cb.stop))
            if rows.stop - rows.start >= block and cols.stop - cols.start >= block:
                overlaps.append((a, b, rows, cols))
    area = sum((r.stop - r.start) * (c.stop - c.start) for _, _, r, c in overlaps)
    # each frame is rendered once, over its window, by the first pair that needs it, as the
    # gradient magnitude that blocks are matched on, and its pairs' overlaps are cut from that;
    # a render is dropped once all its pairs are measured
    uses = {}
    for a, b, _, _ in overlaps:
        uses[a] = uses.get(a, 0) + 1
        uses[b] = uses.get(b, 0) + 1
    renders, lock = {}, threading.Lock()

    def render(k):
        with lock:
            fut = renders.get(k)
            mine = fut is None
            if mine:
                fut = renders[k] = Future()
        if mine:
            wr, wc = windows[k]
            try:
                t, _, valid = _sample(placed[k], grid, wr, wc, bilinear=True)
                gy, gx = np.gradient(t.astype(np.float64))
                fut.set_result((wr.start, wc.start, np.sqrt(gy * gy + gx * gx), valid))
            except BaseException as e:
                fut.set_exception(e)
        return fut.result()

    def release(k):
        with lock:
            uses[k] -= 1
            if not uses[k]:
                del renders[k]

    def cut(k, rows, cols):
        r0, c0, g, valid = render(k)
        rs, cs = slice(rows.start - r0, rows.stop - r0), slice(cols.start - c0, cols.stop - c0)
        return g[rs, cs], valid[rs, cs]

    def measure(pair):
        a, b, rows, cols = pair
        try:
            ga, va = cut(a, rows, cols)
            gb, vb = cut(b, rows, cols)
        finally:
            release(a)
            release(b)
        both = va & vb
        # blocks fully covered by both frames, from a summed-area table of the uncovered pixels
        gaps = np.zeros((both.shape[0] + 1, both.shape[1] + 1), np.int64)
        gaps[1:, 1:] = (~both).cumsum(0).cumsum(1)
        ys = np.arange(0, both.shape[0] - block + 1, step)
        xs = np.arange(0, both.shape[1] - block + 1, step)
        y, x = (g.ravel() for g in np.meshgrid(ys, xs, indexing="ij"))
        full = (gaps[y + block, x + block] - gaps[y, x + block] - gaps[y + block, x] + gaps[y, x]) == 0
        y, x = y[full], x[full]
        if not y.size:
            return None
        win = np.lib.stride_tricks.sliding_window_view
        dy, dx, peak = _phase_shift(win(ga, (block, block))[y, x], win(gb, (block, block))[y, x])
        keep = peak >= min_peak
        e = grid.e0 + (cols.start + x[keep] + block / 2) * grid.gsd
        n = grid.n0 - (rows.start + y[keep] + block / 2) * grid.gsd
        return np.column_stack([np.full(e.size, a), np.full(e.size, b), e, n,
                                dx[keep] * grid.gsd, -dy[keep] * grid.gsd])

    obs, sampled, done = [], 0, 0
    if report:
        report(0, len(overlaps))
    jobs = max(1, min(jobs, len(overlaps)))
    pool = ThreadPoolExecutor(jobs) if jobs > 1 else None
    try:
        # the workers take the pairs in order, so renders live about as long as measured one by one
        results = pool.map(measure, overlaps) if pool else map(measure, overlaps)
        for (_, _, rows, cols), m in zip(overlaps, results):
            if m is not None:
                obs.append(m)
            if report:
                sampled += (rows.stop - rows.start) * (cols.stop - cols.start)
                if (now := len(overlaps) * sampled // area) != done:
                    done = now
                    report(done, len(overlaps))
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    return np.concatenate(obs).astype(np.float64) if obs else np.zeros((0, 6))


def _summary(obs: np.ndarray, grid: Grid) -> dict:
    """Per-pair offsets (median over the pair's blocks) and their distribution."""
    pairs = []
    for a, b in sorted({(int(r[0]), int(r[1])) for r in obs}):
        sel = obs[(obs[:, 0] == a) & (obs[:, 1] == b)]
        de, dn = np.median(sel[:, 4:6], axis=0)
        pairs.append({"frames": [a, b], "blocks": len(sel), "shift_m": [round(float(de), 3), round(float(dn), 3)],
                      "offset_m": round(float(math.hypot(de, dn)), 3)})
    if not pairs:
        return {"pairs": 0}
    m = np.array([p["offset_m"] for p in pairs])
    return {"pairs": len(pairs), "median_m": round(float(np.median(m)), 3), "p90_m": round(float(np.percentile(m, 90)), 3),
            "max_m": round(float(m.max()), 3), "pairs_detail": pairs}


# --- alignment (level 2) -----------------------------------------------------------------

PRIOR_SHIFT_M = 0.15            # per-frame placement error expected from RTK + gimbal attitude
PRIOR_ROT_DEG = 0.25            # tight: looser, frame rotations absorb roof parallax and bend straight edges
PRIOR_BORESIGHT_DEG = 3.0
OBS_SIGMA_PX = 2.0
HEIGHT_SMOOTH_M = 0.5           # expected height difference between neighbouring cells (robust: edges)
HEIGHT_ANCHOR_M = 0.3           # rangefinder target altitude error
HEIGHT_PRIOR_M = 5.0            # weak pull of unobserved cells to the plane


def _align(frames, placed, grid, ground, zone, north, iterations: int | None = None,
           progress: Callable[..., None] | None = None, jobs: int = 1, heights: str = "auto",
           height_cell: int = HEIGHT_CELL_PX, fixed: Heights | None = None):
    """Self-calibrate the flight from its overlaps.

    Unknowns: one boresight (camera-body pitch and yaw) for the whole flight, the angle between where the
    gimbal says the camera points and where it does (its ground effect flips with heading,
    which makes it observable), for each frame a small shift and rotation on the ground
    plane, and (heights="auto") the height of the surface above the plane on a grid of cells
    height_cell mosaic pixels wide, from the parallax between frames seeing it from different
    places. Observations: the block offsets between overlapping frames, and the rangefinder's
    target altitudes. Each frame's correction is held near zero by a prior (the RTK position
    and gimbal attitude are good), and a robust (Huber) weighting stops blocks the model
    can't explain (moving things, trees) from pulling the rest out of line. A pair's blocks
    are strongly correlated, so each pair counts about as one observation (block weight
    1/sqrt(blocks)). Solved by iteratively reweighted linear least squares, re-measuring the
    offsets after each round (iterations: 4 with heights, 3 without). (On a 21 m M4T
    flight, fitting half of the overlapping pairs cut the other half's median offset from
    0.37 m to 0.12 m.) When the heights can't be estimated (too few cells seen by two
    frames, or a surface reaching the cameras) the flight is aligned on the plane instead,
    and the report says why. fixed: a surface (from a DSM) the frames are projected onto
    throughout, the heights not estimated (the report's "heights" left to the caller).
    progress: as build's, for the "align" stage: per round, the pairs measured, then the
    solve's reweighting steps.
    Returns the placed frames, their heights (None on the plane) and the report.
    """
    if heights not in ("auto", "plane"):
        raise ValueError(f"heights must be 'auto' or 'plane', not {heights!r}")
    use_heights = heights == "auto" and fixed is None
    if iterations is None:
        iterations = 4 if use_heights else 3
    by_name = {f.name: f for f in frames}
    n = len(placed)
    bias = np.zeros(2)
    shifts = np.zeros((n, 2))
    rots = np.zeros(n)
    surface = Heights.flat(grid, height_cell * grid.gsd) if use_heights else fixed
    observed = None
    failed = None
    before = None
    obs = np.zeros((0, 6))
    inliers = 1.0
    for it in range(iterations):
        def report(unit, it=it):
            return ((lambda done, total: progress("align", done, total, unit, round=it + 1, rounds=iterations))
                    if progress else None)
        obs = _measure_blocks(placed, grid, report=report("pairs"), jobs=jobs)
        if it == 0:
            before = _summary(obs, grid)
        if len(obs) < 3:
            break
        step, inliers, observed = _solve(placed, obs, grid.gsd, bias, report("solve steps"),
                                         surface if use_heights else None,
                                         _anchors(placed, ground) if use_heights else None)
        if not np.isfinite(step).all():
            failed = "the solve diverged"
            break
        bias += step[3 * n:3 * n + 2]
        shifts += step[:3 * n].reshape(n, 3)[:, :2]
        rots += step[:3 * n].reshape(n, 3)[:, 2]
        if use_heights:
            surface = dataclasses.replace(surface, z=surface.z + step[3 * n + 2:].reshape(surface.z.shape))
        replaced = [_place(by_name[pl.frame.name], ground, zone, north, tuple(bias)) for pl in placed]
        for k, pl in enumerate(replaced):
            pl.shift, pl.rot, pl.heights = shifts[k].copy(), float(rots[k]), surface
        placed = replaced

    heights_report = None
    if use_heights:
        seen = observed > 0.5 if observed is not None else np.zeros(surface.z.size, bool)
        if failed is None and seen.sum() < 4:
            failed = f"{int(seen.sum())} height cells seen by two frames (4 needed)"
        if failed is None and surface.z.max() >= min(pl.h for pl in placed) - 1:
            failed = f"the surface reaches {surface.z.max():.1f} m above the plane, within 1 m of a camera"
        if failed:
            placed = [_place(by_name[pl.frame.name], ground, zone, north) for pl in placed]
            placed, _, result = _align(frames, placed, grid, ground, zone, north, progress=progress, jobs=jobs,
                                       heights="plane")
            result["heights"] = {"method": "plane", "reason": failed}
            return placed, None, result
        anchors = _anchors(placed, ground)
        z_seen = surface.z.ravel()[seen]
        heights_report = {
            "method": "parallax between overlapping frames, anchored to the rangefinder",
            "cell_m": round(surface.cell, 3), "cells": int(surface.z.size), "observed_cells": int(seen.sum()),
            "range_m": [round(float(z_seen.min()), 2) + 0.0, round(float(z_seen.max()), 2) + 0.0],
            "anchors": len(anchors),
            "anchor_residual_m": (round(float(np.median(np.abs(surface.at_points(anchors[:, 0], anchors[:, 1])
                                                               - anchors[:, 2]))), 3) if len(anchors) else None),
        }
    result = {
        "method": "boresight + per-frame ground shift/rotation from overlap offsets (robust least squares)",
        "boresight_deg": {"pitch": round(math.degrees(bias[0]), 3), "yaw": round(math.degrees(bias[1]), 3)},
        "frame_shift_m": {"median": round(float(np.median(np.linalg.norm(shifts, axis=1))), 3),
                          "max": round(float(np.linalg.norm(shifts, axis=1).max()), 3)},
        "frame_rotation_deg": {"median": round(float(np.degrees(np.median(np.abs(rots)))), 3),
                               "max": round(float(np.degrees(np.abs(rots).max())), 3)},
        "observations": len(obs), "inlier_fraction": round(float(inliers), 3),
        "before": {k: v for k, v in (before or {}).items() if k != "pairs_detail"},
        "heights": heights_report or {"method": "plane", "reason": "not asked for"},
    }
    return placed, surface, result


def _anchors(placed: list[_Placed], ground: float) -> np.ndarray:
    """Where each frame's rangefinder hit, in mosaic coordinates, and that point's height
    above the plane: rows (E, N, height)."""
    rows = []
    for pl in placed:
        lrf = pl.frame.pose.ground_alt
        ray = pl.r_t.T[:, 0]
        if lrf is None or ray[2] <= 1e-6:
            continue
        drop = pl.frame.pose.abs_alt - lrf
        if drop <= 0:
            continue
        x = pl.c + pl.j @ (np.array([ray[1], ray[0]]) * drop / ray[2])
        d = x - pl.c
        x = x - pl.shift - pl.rot * np.array([-d[1], d[0]])
        rows.append([x[0], x[1], lrf - ground])
    return np.array(rows).reshape(-1, 3)


def _solve(placed: list[_Placed], obs: np.ndarray, gsd: float, bias: np.ndarray,
           report: Callable[[int, int], None] | None = None, heights: Heights | None = None,
           anchors: np.ndarray | None = None) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Correction increments [shift E, shift N, rotation] per frame + [boresight pitch, yaw]
    (+ with heights, one per height cell) that cancel the observed offsets, the fraction of
    observations kept at full weight, and (with heights) each cell's observation weight.

    Moving frame k's content: its correction (sampling at X + t + r perp(X - c)) moves it by
    -(t + r perp(X - c)); a boresight b moves it by G_k b (G_k: ground shift of the optical
    axis per radian); raising the surface by dz at X moves it by -(X - c_k) dz / (h_k - z(X)),
    away from the frame's nadir. For each block, (move_b - move_a)(X) = -observed shift.
    The heights are held to their neighbours' (smoothness, robust so that roof edges stay
    sharp), to the rangefinder's (anchors: rows of E, N, height above the plane) and weakly to
    the plane.
    report(done, total): the reweighting steps solved.
    """
    n = len(placed)
    n_global = 2
    eps = 1e-4
    g = []
    for pl in placed:
        cols = []
        for d in ((eps, 0.0), (0.0, eps)):
            moved = dataclasses.replace(pl, r_t=(pl.r_t.T @ _boresight(*d)).T)
            cols.append((_centre_ground(moved) - _centre_ground(pl)) / eps)
        g.append(np.column_stack(cols))
    # each observation row touches 8 unknowns: [t comp, r] of frames a and b, and the boresight
    # (and the 4 height cells its block centre interpolates)
    a, b = obs[:, 0].astype(int), obs[:, 1].astype(int)
    _, pair, count = np.unique(obs[:, :2].astype(int), axis=0, return_inverse=True, return_counts=True)
    c = np.array([pl.c for pl in placed])
    h = np.array([pl.h for pl in placed])
    g = np.array(g)
    m0 = 3 * n + n_global
    if heights is not None:
        cells, cell_w = heights.point_weights(obs[:, 2], obs[:, 3])
        z_obs = (heights.z.ravel()[cells] * cell_w).sum(1)
    idx, val, rhs = [], [], []
    for comp, target in ((0, -obs[:, 4]), (1, -obs[:, 5])):
        cols, vals = [], []
        for k, sign in ((b, 1.0), (a, -1.0)):
            ve, vn = obs[:, 2] - c[k, 0], obs[:, 3] - c[k, 1]
            cols += [3 * k + comp, 3 * k + 2]
            vals += [np.full(len(k), -sign), sign * (vn if comp == 0 else -ve)]   # -t, -r perp(v): (r vn, -r ve)
        cols += [np.full(len(a), 3 * n), np.full(len(a), 3 * n + 1)]
        vals += list((g[b, comp] - g[a, comp]).T)
        if heights is not None:
            v = obs[:, 2 + comp]
            dz = -(v - c[b, comp]) / (h[b] - z_obs) + (v - c[a, comp]) / (h[a] - z_obs)
            cols += list((m0 + cells).T)
            vals += list((dz[:, None] * cell_w).T)
        idx.append(np.column_stack(cols))
        val.append(np.column_stack(vals))
        rhs.append(target)
    width = idx[0].shape[1]
    idx = np.stack(idx, 1).reshape(-1, width)                                 # rows in (block, comp) order
    val = np.stack(val, 1).reshape(-1, width)
    b_rows = np.stack(rhs, 1).ravel()
    base = np.repeat(1 / np.sqrt(count[pair.ravel()]), 2)
    sigma = np.full(len(b_rows), OBS_SIGMA_PX * gsd)
    n_obs = len(b_rows)

    prior = np.concatenate([np.tile([1 / PRIOR_SHIFT_M, 1 / PRIOR_SHIFT_M, 1 / math.radians(PRIOR_ROT_DEG)], n),
                            np.full(2, 1 / math.radians(PRIOR_BORESIGHT_DEG))])
    current = np.concatenate([np.column_stack([np.array([pl.shift for pl in placed]),
                                               [pl.rot for pl in placed]]).ravel(), bias])
    observed = None
    if heights is not None:
        z = heights.z.ravel()
        rows_, cols_ = heights.z.shape
        observed = np.bincount(cells.ravel(), (cell_w * base[::2, None]).ravel(), minlength=z.size)
        # constraints on the total heights: the row is val . increment = target - val . current
        grid_i = np.arange(z.size).reshape(rows_, cols_)
        pairs = np.concatenate([np.column_stack([grid_i[:, :-1].ravel(), grid_i[:, 1:].ravel()]),
                                np.column_stack([grid_i[:-1].ravel(), grid_i[1:].ravel()])])
        extra_idx = [m0 + pairs]
        extra_val = [np.tile([1.0, -1.0], (len(pairs), 1))]
        extra_rhs = [z[pairs[:, 1]] - z[pairs[:, 0]]]
        extra_sigma = [np.full(len(pairs), HEIGHT_SMOOTH_M)]
        if anchors is not None and len(anchors):
            a_cells, a_w = heights.point_weights(anchors[:, 0], anchors[:, 1])
            extra_idx.append(m0 + a_cells)
            extra_val.append(a_w)
            extra_rhs.append(anchors[:, 2] - (z[a_cells] * a_w).sum(1))
            extra_sigma.append(np.full(len(anchors), HEIGHT_ANCHOR_M))
        pad = lambda arr: np.pad(arr, ((0, 0), (0, width - arr.shape[1])))   # noqa: E731
        idx = np.concatenate([idx] + [pad(i) for i in extra_idx])
        val = np.concatenate([val] + [pad(v) for v in extra_val])
        b_rows = np.concatenate([b_rows] + extra_rhs)
        sigma = np.concatenate([sigma] + extra_sigma)
        base = np.concatenate([base, np.ones(len(b_rows) - len(base))])
        prior = np.concatenate([prior, np.full(z.size, 1 / HEIGHT_PRIOR_M)])
        current = np.concatenate([current, z])

    m = len(prior)
    dense = m <= 3000
    if dense:
        outer = (idx[:, :, None] * m + idx[:, None, :]).ravel()
    w = np.ones(len(b_rows))
    x = np.zeros(m)
    steps = 8
    if report:
        report(0, steps)
    for i in range(steps):
        ww = (w * base / sigma) ** 2
        # the normal equations of the weighted rows and the priors, which act on the total correction
        y = np.bincount(idx.ravel(), (val * (ww * b_rows)[:, None]).ravel(), minlength=m) - current * prior ** 2
        if dense:
            normal = np.bincount(outer, (ww[:, None, None] * val[:, :, None] * val[:, None, :]).ravel(),
                                 minlength=m * m).reshape(m, m) + np.diag(prior ** 2)
            x = np.linalg.solve(normal, y)
        else:
            x = _cg(idx, val, ww, prior ** 2, y, x)
        r = np.abs((val * x[idx]).sum(1) - b_rows) / sigma
        w = np.where(r <= 2.0, 1.0, np.sqrt(2.0 / np.maximum(r, 1e-12)))   # Huber, k = 2
        if report:
            report(i + 1, steps)
    return x, float((w[:n_obs] == 1.0).mean()), observed


def _cg(idx: np.ndarray, val: np.ndarray, ww: np.ndarray, diag: np.ndarray, y: np.ndarray, x: np.ndarray,
        rtol: float = 1e-8, max_iter: int = 500) -> np.ndarray:
    """Solve (A' W A + diag) x = y, A's rows being sparse (idx, val), by Jacobi-preconditioned
    conjugate gradients from x: for systems too large for the dense normal equations."""
    m = len(y)

    def apply(p):
        return np.bincount(idx.ravel(), (val * (ww * (val * p[idx]).sum(1))[:, None]).ravel(), minlength=m) + diag * p

    precond = 1 / (np.bincount(idx.ravel(), (val * val * ww[:, None]).ravel(), minlength=m) + diag)
    r = y - apply(x)
    zr = precond * r
    p = zr.copy()
    rz = r @ zr
    stop = rtol * np.linalg.norm(y)
    for _ in range(max_iter):
        if np.linalg.norm(r) <= stop:
            break
        ap = apply(p)
        alpha = rz / (p @ ap)
        x = x + alpha * p
        r = r - alpha * ap
        zr = precond * r
        rz, rz_old = r @ zr, rz
        p = zr + (rz / rz_old) * p
    return x


# --- output ---------------------------------------------------------------------------------

BANDS = (("temperature_c", "deg C"), ("source_frame", "index into the frames table"),
         ("view_angle_deg", "off-nadir degrees"), ("source_u", "source frame pixel column"),
         ("source_v", "source frame pixel row"))


# what a location-free mosaic keeps of each frame (see privacy): no position, altitude or identifier
_LOCATION_FREE_FRAME_KEYS = (
    "index", "file", "pixels_used", "height_above_ground_m", "gimbal_yaw", "gimbal_pitch", "gimbal_roll", "focal_px",
    "focal_source", "width", "height", "alignment_shift_m", "alignment_rotation_deg", "temperature_offset_c",
    "UTCAtExposure", "FlightYawDegree", "FlightPitchDegree", "FlightRollDegree", "LRFStatus", "LRFTargetDistance",
    "DroneModel", "measurement", "distance_source")


def write(m: Mosaic, path: Path, compression: str = "deflate", unit: str = "C",
          located: bool = True) -> tuple[Path, Path]:
    """The mosaic GeoTIFF (5 float32 bands, NaN = no data; 3 for a mosaic without source_px)
    and its JSON sidecar. unit: the temperature band's unit, C, F or K (converted from the
    SDK's Celsius, rounded once).
    located=False: a plain TIFF without georeference, and a sidecar without the CRS, origin,
    ground altitude or any frame's position or identifiers; frames are placed by x_m/y_m,
    metres east and south of the mosaic's top-left corner."""
    g = m.grid
    bands = ((f"temperature_{unit.lower()}", units.SYMBOLS[unit]),) + BANDS[1:3 if m.source_px is None else 5]
    onto = ("a horizontal plane" if m.heights is None else "a surface model (see write_heights)"
            if (m.report.get("heights") or {}).get("method") == "DSM"
            else "the surface estimated from parallax (see write_heights)")
    if located:
        info = {
            "method": f"direct georeferencing onto {onto}; one frame per pixel (most nadir), nearest neighbour",
            "crs": f"EPSG:{g.epsg}", "gsd_m": g.gsd, "width": g.width, "height": g.height,
            "origin_utm": [g.e0, g.n0], "ground_alt_m": round(m.ground_alt, 3), "ground_alt_source": m.ground_alt_source,
            "bands": [{"name": n, "unit": u} for n, u in bands],
            "report": m.report, "frames": m.frames,
        }
    else:
        frames = [{**{k: f[k] for k in _LOCATION_FREE_FRAME_KEYS if k in f},
                   "x_m": round(f["utm_e"] - g.e0, 3), "y_m": round(g.n0 - f["utm_n"], 3)} for f in m.frames]
        info = {
            "method": f"placed on {onto} from the frames' positions (removed); one frame per pixel "
                      "(most nadir), nearest neighbour",
            "location_removed": True, "gsd_m": g.gsd, "width": g.width, "height": g.height,
            "bands": [{"name": n, "unit": u} for n, u in bands],
            "report": m.report, "frames": frames,
        }
    items = [f'<Item name="DESCRIPTION" sample="{i}" role="description">{n}</Item>'
             f'<Item name="UNITTYPE" sample="{i}" role="unittype">{escape(u)}</Item>' for i, (n, u) in enumerate(bands)]
    items.append(f'<Item name="dji_thermal_mosaic">{escape(json.dumps(info, separators=(",", ":")))}</Item>')
    geokeys = [1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, g.epsg, 3076, 0, 1, 9001]
    extra = {
        42112: tiffout.tag("s", "<GDALMetadata>" + "".join(items) + "</GDALMetadata>"),
        42113: tiffout.tag("s", "nan"),
    }
    if located:
        extra.update({
            33550: tiffout.tag("d", (g.gsd, g.gsd, 0.0)),
            33922: tiffout.tag("d", (0.0, 0.0, 0.0, g.e0, g.n0, 0.0)),
            34735: tiffout.tag("H", geokeys),
        })
    layers = [units.float32_in(m.temperature, unit), m.source, m.view_angle]
    if m.source_px is not None:
        layers += [m.source_px[..., 0], m.source_px[..., 1]]
    pixels = np.dstack(layers).astype(np.float32)
    with files.replaced_when_done(path) as tmp:
        tiffout.write_tiff(tmp, pixels, None, band_name=False, compression=compression, extra_tags=extra)
    sidecar = path.with_suffix(".json")
    with files.replaced_when_done(sidecar) as tmp:
        tmp.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return path, sidecar


def write_heights(m: Mosaic, path: Path, compression: str = "deflate", located: bool = True) -> Path | None:
    """The surface the frames were projected onto, as a float32 GeoTIFF of one band on the
    height cells (pixel centres at the cell centres; NaN where no frame covers the cell):
    altitudes in metres, the ground plane's plus the estimated height above it. located=False:
    the heights above the plane, without georeference. None (nothing written) on the plane."""
    h = m.heights
    if h is None:
        return None
    g = m.grid
    rows, cols = h.z.shape
    # the cell each mosaic row and column falls in (the grid may have moved since the cells were laid)
    ci = np.floor((h.n0 - (g.n0 - (np.arange(g.height) + 0.5) * g.gsd)) / h.cell).astype(np.intp)
    cj = np.floor(((g.e0 + (np.arange(g.width) + 0.5) * g.gsd) - h.e0) / h.cell).astype(np.intp)
    seen_cols = np.zeros((rows, g.width), bool)
    seen = np.isfinite(m.temperature)
    for i in range(rows):
        if (ci == i).any():
            seen_cols[i] = seen[ci == i].any(axis=0)
    covered = np.zeros((rows, cols), bool)
    for j in range(cols):
        if (cj == j).any():
            covered[:, j] = seen_cols[:, cj == j].any(axis=1)
    z = h.z + (m.ground_alt if located else 0.0)
    pixels = np.where(covered, z, np.nan).astype(np.float32)
    name, unit = ("altitude_m", "m") if located else ("height_above_plane_m", "m")
    info = {**m.report["heights"], "interpolation": "bilinear between cell centres"}
    if located:
        info.update({"crs": f"EPSG:{g.epsg}", "origin_utm": [h.e0, h.n0], "ground_alt_m": round(m.ground_alt, 3)})
    items = [f'<Item name="DESCRIPTION" sample="0" role="description">{name}</Item>'
             f'<Item name="UNITTYPE" sample="0" role="unittype">{unit}</Item>',
             f'<Item name="dji_thermal_heights">{escape(json.dumps(info, separators=(",", ":")))}</Item>']
    extra = {
        42112: tiffout.tag("s", "<GDALMetadata>" + "".join(items) + "</GDALMetadata>"),
        42113: tiffout.tag("s", "nan"),
    }
    if located:
        extra.update({
            33550: tiffout.tag("d", (h.cell, h.cell, 0.0)),
            33922: tiffout.tag("d", (0.0, 0.0, 0.0, h.e0, h.n0, 0.0)),
            34735: tiffout.tag("H", [1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, g.epsg, 3076, 0, 1, 9001]),
        })
    with files.replaced_when_done(path) as tmp:
        tiffout.write_tiff(tmp, pixels, None, band_name=False, compression=compression, extra_tags=extra)
    return path


def write_preview(m: Mosaic, path: Path, lut: np.ndarray | None = None,
                  low: float | None = None, high: float | None = None) -> tuple[Path, float, float]:
    """8-bit quick look, transparent where no data: lut (256, 3) uint8 colours from cold to
    hot (default: grayscale), over low..high degrees C for the whole mosaic (default: the
    2nd-98th percentile). Returns the path and the range used."""
    t = m.temperature
    ok = np.isfinite(t) & (t > tiffout.NO_DATA_C + 1)
    if low is None or high is None:
        low, high = (float(v) for v in np.percentile(t[ok], (2, 98))) if ok.any() else (0.0, 1.0)
    level = np.clip((np.nan_to_num(t, nan=low) - low) / max(high - low, 1e-6) * 255, 0, 255)
    level = np.rint(level).astype(np.uint8)
    alpha = np.where(ok, 255, 0).astype(np.uint8)
    rgba = np.dstack([level, alpha]) if lut is None else np.dstack([np.asarray(lut, np.uint8)[level], alpha])
    with files.replaced_when_done(path) as tmp:
        Image.fromarray(rgba).save(tmp)                                     # LA or RGBA
    return path, low, high
