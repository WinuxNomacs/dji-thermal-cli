"""Mosaic of a flight's visual (RGB) frames, laid out by its thermal mosaic.

DJI's thermal drones (M4T, M30T, M3T, H20T, H30T) can save the wide camera's visual image at
each thermal capture, from the same gimbal. Each visual frame is paired with the thermal frame
exposed with it (DJI's UTCAtExposure within PAIR_S; DJI's capture number in the file names
when either has no exposure time) and placed as that thermal frame is in the thermal mosaic:
its attitude with the alignment's boresight, its alignment correction (shift and rotation)
and the surface the thermal frames were projected onto. Only the visual frame's own position
and lens are its own. The two cameras don't point exactly the same way: that difference, one
small rotation between them (pitch and yaw), is measured from the pairs by phase correlation
of the two frames' edges on ground both see, and applied to every visual frame, so that the
two mosaics line up.

The mosaic is a north-up UTM grid, by default at the visual frames' own ground sample
distance. Every pixel takes its colour from one frame, the one that saw it closest to
straight down, interpolated bilinearly (colours are a picture, not measurements). Zoom
frames are left out: they see a small part of the wide frame at another scale.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

from . import djixmp, files, mosaic, tiffout

PAIR_S = 0.25                   # a visual frame exposed this close to a thermal one is its pair
MIN_OFFSET_PAIRS = 5            # pairs measured to correct the cameras' offset
OFFSET_MIN_PEAK = 0.05          # weakest phase-correlation peak trusted (edges of two kinds of image)
STRIP_ROWS = 256                # mosaic rows composited at once
PREVIEW_PX = 2048               # the preview's longer side, at most
BANDS = ("red", "green", "blue", "alpha")
_NO_VALUES = np.zeros((0, 0), np.float32)
_CAPTURE = re.compile(r"_(\d{4,})_[A-Z]+\.[^./]+$", re.IGNORECASE)


@dataclasses.dataclass
class VisualFrame:
    name: str                   # the path relative to the dataset, as the thermal frames are named
    path: Path
    pose: mosaic.Pose           # its own position, attitude and lens (the JPEG's size)
    exposure: datetime.datetime | None
    capture: tuple[str, int] | None     # (folder, DJI capture number) from the file name


@dataclasses.dataclass
class VisualMosaic:
    grid: mosaic.Grid
    rgb: np.ndarray             # uint8 (height, width, 3)
    source: np.ndarray          # int32 frame index, -1 where no frame covers
    frames: list[dict]          # per-frame table (index = source value)
    report: dict
    ground_alt: float
    ground_alt_source: str
    heights: mosaic.Heights | None      # the thermal mosaic's surface
    placed: list                # the frames as placed (index = source value), for the frame models


def read_visual(path: Path, name: str) -> tuple[VisualFrame | None, str]:
    """A visual frame for the mosaic, or (None, why not): a JPEG with a DJI camera pose that
    isn't a zoom camera's."""
    data = Path(path).read_bytes()
    f = djixmp.fields(data)
    if not f:
        return None, "no DJI XMP"
    if "zoom" in (source := f.get("ImageSource", "")).lower():
        return None, f"a zoom camera's image ({source})"
    try:
        pose, why = mosaic.read_pose(data, visual=True)
    except OSError as e:
        return None, f"not a readable image ({e})"
    if pose is None:
        return None, why
    if source:
        pose.record["ImageSource"] = source
    return VisualFrame(name, Path(path), pose, _exposure(f.get("UTCAtExposure")), _capture(name)), ""


def _exposure(text: str | None) -> datetime.datetime | None:
    try:
        return datetime.datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def _capture(name: str) -> tuple[str, int] | None:
    m = _CAPTURE.search(name)
    return (name.rsplit("/", 1)[0] if "/" in name else "", int(m.group(1))) if m else None


def pair(thermal: list[mosaic.Frame], visual: list[VisualFrame]) -> dict[str, VisualFrame]:
    """Each thermal frame's visual frame, by the thermal frame's name: the one exposed closest
    to it, within PAIR_S, or (either without an exposure time) the one of the same capture
    number in the same folder. Each visual frame pairs once."""
    candidates = []
    for t in thermal:
        te, tc = _exposure(t.pose.record.get("UTCAtExposure")), _capture(t.name)
        for k, v in enumerate(visual):
            if te and v.exposure:
                if (dt := abs((te - v.exposure).total_seconds())) <= PAIR_S:
                    candidates.append((dt, t.name, k))
            elif tc and tc == v.capture:
                candidates.append((PAIR_S, t.name, k))
    pairs, used = {}, set()
    for _, name, k in sorted(candidates):
        if name not in pairs and k not in used:
            pairs[name] = visual[k]
            used.add(k)
    return pairs


def build(thermal: mosaic.Mosaic, visual: list[VisualFrame], gsd: float | None = None,
          progress: Callable[..., None] | None = None, jobs: int = 1) -> VisualMosaic:
    """The visual mosaic of the frames paired with thermal's (see the module). gsd: its pixel
    size (default: the visual frames' median ground sample distance). progress: as
    mosaic.build's, for the "visual_offset" (pairs) and "visual_composite" (frames) stages.
    jobs: threads decoding and sampling frames; the result is the same for any number."""
    if thermal.placed is None:
        raise ValueError("the thermal mosaic has no placed frames")
    pairs = pair([pl.frame for pl in thermal.placed], visual)
    if not pairs:
        raise ValueError("no visual frame was taken with a frame of the thermal mosaic")
    g = thermal.grid
    thermal_placed = [pl for pl in thermal.placed if pl.frame.name in pairs]
    frames = [pairs[pl.frame.name] for pl in thermal_placed]

    def place(offset):
        return [_place(tpl, v, thermal.ground_alt, g, offset) for tpl, v in zip(thermal_placed, frames)]

    offset, offset_report = _camera_offset(thermal_placed, place((0.0, 0.0)), frames, thermal.ground_alt, g,
                                           progress, jobs)
    placed = place(offset)
    keep = [k for k, pl in enumerate(placed) if pl is not None]
    placed, frames, thermal_placed = [placed[k] for k in keep], [frames[k] for k in keep], [thermal_placed[k] for k in keep]
    if not placed:
        raise ValueError("no visual frame can be projected onto the ground plane")
    if gsd is None:
        gsd = max(round(float(np.median([pl.h / pl.frame.pose.focal_px for pl in placed])), 3), 0.001)
    grid = mosaic._grid(placed, gsd, g.zone, g.north)
    rgb, source = _composite(placed, frames, grid, mosaic._reporter(progress, "visual_composite", "frames"), jobs)

    covered = source >= 0
    used = np.bincount(source[covered], minlength=len(placed))
    table = []
    for k, (pl, v, tpl) in enumerate(zip(placed, frames, thermal_placed)):
        p = v.pose
        table.append({
            "index": k, "file": v.name, "thermal_file": tpl.frame.name, "pixels_used": int(used[k]),
            "lat": p.lat, "lon": p.lon, "abs_alt_m": p.abs_alt, "height_above_ground_m": round(pl.h, 3),
            "utm_e": round(float(pl.c[0]), 3), "utm_n": round(float(pl.c[1]), 3),
            "gimbal_yaw": p.yaw, "gimbal_pitch": p.pitch, "gimbal_roll": p.roll,
            "focal_px": round(p.focal_px, 2), "width": p.width, "height": p.height,
            "alignment_shift_m": [round(float(s), 3) for s in pl.shift],
            "alignment_rotation_deg": round(math.degrees(pl.rot), 3),
            **p.record,
        })
    paired = {v.name for v in frames}
    report = {
        "frames_used": len(placed),
        "frames_contributing": int((used > 0).sum()),
        "pairing": f"exposure time (UTCAtExposure) within {PAIR_S:g} s, else DJI's capture number",
        "thermal_frames_without_visual": [pl.frame.name for pl in thermal.placed if pl.frame.name not in pairs],
        "visual_frames_unused": [v.name for v in visual if v.name not in paired],
        "camera_offset": offset_report,
        "heights": thermal.report.get("heights"),
    }
    return VisualMosaic(grid, rgb, source, table, report, thermal.ground_alt, thermal.ground_alt_source,
                        thermal.heights, placed)


def _place(tpl, v: VisualFrame, ground: float, g: mosaic.Grid, offset: tuple[float, float]):
    """The visual frame placed as its thermal frame tpl is: tpl's attitude (with the thermal
    alignment's boresight) turned by the cameras' offset, its alignment correction and surface."""
    pl = mosaic._place(mosaic.Frame(v.name, _NO_VALUES, v.pose), ground, g.zone, g.north,
                       r=tpl.r_t.T @ mosaic._boresight(*offset))
    if pl is not None:
        pl.shift, pl.rot, pl.heights = tpl.shift.copy(), tpl.rot, tpl.heights
    return pl


def _coords(pl, grid: mosaic.Grid, rows: slice, cols: slice):
    """For grid pixels in rows x cols: where the frame saw them (column, row; continuous), how
    far in front of the camera (positive: seen) and the off-nadir angle, as mosaic._sample."""
    e = grid.e0 + (np.arange(cols.start, cols.stop) + 0.5) * grid.gsd
    n = grid.n0 - (np.arange(rows.start, rows.stop) + 0.5) * grid.gsd
    de, dn = np.meshgrid(e - pl.c[0], n - pl.c[1])
    hz = pl.h if pl.heights is None else np.maximum(pl.h - pl.heights.at(e, n), 0.1)
    return mosaic._project(pl, de, dn, hz)


def _decode(v: VisualFrame, scale: int, mode: str = "RGB") -> tuple[np.ndarray, float]:
    """The frame's image, decoded at about 1/scale of its size (JPEG draft mode: fast), and the
    factor from its full-size pixels to the decoded ones."""
    with Image.open(v.path) as im:
        if scale > 1:
            im.draft(mode, (im.width // scale, im.height // scale))
        pixels = np.asarray(im.convert(mode))
    return pixels, v.pose.width / pixels.shape[1]


def _scale(pl, gsd: float) -> int:
    """The JPEG reduction (1, 2, 4 or 8) whose pixels are still no larger than gsd on the ground."""
    native = pl.h / pl.frame.pose.focal_px
    scale = 1
    while scale < 8 and native * scale * 2 <= gsd * 1.0001:
        scale *= 2
    return scale


def _inside_image(image: np.ndarray, uf, vf, fwd, factor: float):
    """The decoded image's continuous pixel for full-size (uf, vf), and which were seen (the
    nearest pixel inside the image)."""
    h, w = image.shape[:2]
    u, v = (uf + 0.5) / factor - 0.5, (vf + 0.5) / factor - 0.5
    seen = (fwd > 0) & (u >= -0.5) & (u < w - 0.5) & (v >= -0.5) & (v < h - 0.5)
    return u, v, seen


def _bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """image (h, w) or (h, w, c) interpolated at points (u, v) (pixel centres at whole numbers,
    held at the edge pixels beyond them), as float32."""
    h, w = image.shape[:2]
    u, v = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
    u0 = np.minimum(np.floor(u).astype(np.intp), max(w - 2, 0))
    v0 = np.minimum(np.floor(v).astype(np.intp), max(h - 2, 0))
    u1, v1 = np.minimum(u0 + 1, w - 1), np.minimum(v0 + 1, h - 1)
    fu, fv = (u - u0).astype(np.float32), (v - v0).astype(np.float32)
    if image.ndim == 3:
        fu, fv = fu[:, None], fv[:, None]
    top = image[v0, u0] * (1 - fu) + image[v0, u1] * fu
    bottom = image[v1, u0] * (1 - fu) + image[v1, u1] * fu
    return (top * (1 - fv) + bottom * fv).astype(np.float32)


def _composite(placed, frames: list[VisualFrame], grid: mosaic.Grid, report, jobs: int):
    """The colour of each grid pixel from the frame seeing it closest to nadir (ties: the
    lower index, so any number of threads gives the same mosaic), and that frame's index."""
    shape = (grid.height, grid.width)
    rgb = np.zeros(shape + (3,), np.uint8)
    best = np.full(shape, np.inf, np.float32)
    source = np.full(shape, -1, np.int32)
    lock = threading.Lock()

    def one(k: int) -> None:
        pl = placed[k]
        image, factor = _decode(frames[k], _scale(pl, grid.gsd))
        rows, cols = mosaic._window(pl, grid)
        for r0 in range(rows.start, rows.stop, STRIP_ROWS):
            rs = slice(r0, min(r0 + STRIP_ROWS, rows.stop))
            uf, vf, fwd, angle = _coords(pl, grid, rs, cols)
            u, v, seen = _inside_image(image, uf, vf, fwd, factor)
            # a first cut without the lock: best only ever falls, so this keeps every pixel the frame can win
            maybe = seen & (angle <= best[rs, cols])
            if not maybe.any():
                continue
            colour = np.rint(_bilinear(image, u[maybe], v[maybe])).clip(0, 255).astype(np.uint8)
            a = angle[maybe]
            with lock:
                b, s = best[rs, cols], source[rs, cols]
                win = (a < b[maybe]) | ((a == b[maybe]) & (k < s[maybe]))
                take = np.zeros_like(maybe)
                take[maybe] = win
                rgb[rs, cols][take] = colour[win]
                b[take] = a[win]
                s[take] = k

    if report:
        report(0, len(placed))
    with ThreadPoolExecutor(max(1, jobs)) as pool:
        for done, f in enumerate(as_completed([pool.submit(one, k) for k in range(len(placed))]), 1):
            f.result()
            if report:
                report(done, len(placed))
    return rgb, source


def _camera_offset(thermal_placed, placed, frames: list[VisualFrame], ground: float, g: mosaic.Grid, progress,
                   jobs: int):
    """The rotation (pitch, yaw, radians, as mosaic._boresight) that turns the thermal camera
    into the visual one, and its report. Each pair's offset is where the visual frame shows the
    ground relative to its thermal frame (ground metres, east and north); a rotation b moves a
    frame's picture on the ground by about J b (J: how its optical axis' ground point moves
    with the rotation, found numerically), so the rotation is the robust least-squares
    solution of J b = -offset over the pairs."""
    report = mosaic._reporter(progress, "visual_offset", "pairs")
    measurable = [k for k, pl in enumerate(placed) if pl is not None]
    if report:
        report(0, len(measurable))
    measured = {}
    with ThreadPoolExecutor(max(1, jobs)) as pool:
        futures = {pool.submit(_pair_offset, thermal_placed[k], placed[k], frames[k], g): k for k in measurable}
        for done, f in enumerate(as_completed(futures), 1):
            if (result := f.result()) is not None:
                measured[futures[f]] = result
            if report:
                report(done, len(measurable))
    strong = sorted(k for k, (_, peak) in measured.items() if peak >= OFFSET_MIN_PEAK)
    base = {"method": "phase correlation of each pair's edges on ground both see, robust least squares",
            "pairs_measured": len(strong)}
    if len(strong) < MIN_OFFSET_PAIRS:
        return (0.0, 0.0), {**base, "applied": False,
                            "reason": f"{len(strong)} pairs measured (at least {MIN_OFFSET_PAIRS} needed)"}
    eps = 1e-4
    rows, offsets = [], []
    for k in strong:
        tpl, v = thermal_placed[k], frames[k]
        centre = [mosaic._centre_ground(_place(tpl, v, ground, g, b))
                  for b in ((0.0, 0.0), (eps, 0.0), (0.0, eps))]
        rows.append(np.column_stack([(centre[1] - centre[0]) / eps, (centre[2] - centre[0]) / eps]))
        offsets.append(measured[k][0])
    jac, d = np.stack(rows), np.stack(offsets)                  # (k, 2, 2), (k, 2)
    used = np.ones(len(strong), bool)
    b = np.zeros(2)
    for _ in range(3):                                           # refit without the pairs it can't explain
        b = np.linalg.lstsq(jac[used].reshape(-1, 2), -d[used].reshape(-1), rcond=None)[0]
        residual = np.linalg.norm(np.einsum("kij,j->ki", jac, b) + d, axis=1)
        limit = max(3 * 1.4826 * float(np.median(residual[used])), g.gsd)
        used = residual <= limit
    before, after = np.linalg.norm(d, axis=1), residual[used]
    return (float(b[0]), float(b[1])), {
        **base, "applied": True, "pairs_used": int(used.sum()),
        "boresight_deg": {"pitch": round(math.degrees(b[0]), 3), "yaw": round(math.degrees(b[1]), 3)},
        "offset_before_m": {"median": round(float(np.median(before)), 3)},
        "residual_m": {"median": round(float(np.median(after)), 3), "p90": round(float(np.percentile(after, 90)), 3)},
    }


def _pair_offset(tpl, vpl, v: VisualFrame, g: mosaic.Grid):
    """Where the visual frame shows the ground relative to its thermal frame, (east, north)
    metres, and the phase correlation's peak; None when they share too little ground. Measured
    on the largest north-up square inside the thermal frame's footprint, at the thermal
    mosaic's pixel size, from the gradients' magnitudes (edges are where a picture and a
    temperature image agree)."""
    outline = tpl.corrected_footprint()
    centre = outline.mean(axis=0)
    half = float(np.min(outline.max(axis=0) - outline.min(axis=0))) / 2
    for _ in range(40):
        corners = centre + half * np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
        if mosaic._inside(corners, outline).all():
            break
        half *= 0.9
    size = int(2 * half / g.gsd)
    if size < 32:
        return None
    square = mosaic.Grid(float(centre[0] - half), float(centre[1] + half), g.gsd, size, size, g.zone, g.north)
    whole = slice(0, size)
    t, _, t_seen = mosaic._sample(tpl, square, whole, whole, bilinear=True)
    gray, factor = _decode(v, _scale(vpl, g.gsd), "L")
    uf, vf, fwd, _ = _coords(vpl, square, whole, whole)
    u, vv, v_seen = _inside_image(gray, uf, vf, fwd, factor)
    both = t_seen & v_seen
    if both.mean() < 0.95:
        return None
    pic = np.zeros((size, size), np.float32)
    pic[v_seen] = _bilinear(gray.astype(np.float32), u[v_seen], vv[v_seen])
    t = np.where(both, t, np.median(t[both]))
    pic = np.where(both, pic, np.median(pic[both]))
    edges_t, edges_v = np.hypot(*np.gradient(t)), np.hypot(*np.gradient(pic))
    dy, dx, peak = mosaic._phase_shift(edges_t[None], edges_v[None])
    return np.array([dx[0] * g.gsd, -dy[0] * g.gsd]), float(peak[0])


def write(m: VisualMosaic, path: Path, compression: str = "deflate", located: bool = True) -> tuple[Path, Path]:
    """The visual mosaic GeoTIFF (4 uint8 bands: red, green, blue, alpha; alpha 0 where no
    frame covers) and its JSON sidecar, with the frames' models as the thermal mosaic's.
    located=False: as mosaic.write's."""
    g = m.grid
    onto = "a horizontal plane" if m.heights is None else "the thermal mosaic's surface (see its heights)"
    method = (f"each frame placed as its thermal frame in the thermal mosaic, the cameras' offset corrected, "
              f"onto {onto}; one frame per pixel (most nadir), bilinear")
    if located:
        info = {
            "method": method, "crs": f"EPSG:{g.epsg}", "gsd_m": g.gsd, "width": g.width, "height": g.height,
            "origin_utm": [g.e0, g.n0], "ground_alt_m": round(m.ground_alt, 3), "ground_alt_source": m.ground_alt_source,
            "bands": [{"name": n} for n in BANDS], "report": m.report, "frames": m.frames,
            "frame_models": mosaic._frame_models(m),
        }
    else:
        keys = mosaic._LOCATION_FREE_FRAME_KEYS + ("thermal_file", "ImageSource")
        frames = [{**{k: f[k] for k in keys if k in f},
                   "x_m": round(f["utm_e"] - g.e0, 3), "y_m": round(g.n0 - f["utm_n"], 3)} for f in m.frames]
        info = {
            "method": method, "location_removed": True, "gsd_m": g.gsd, "width": g.width, "height": g.height,
            "bands": [{"name": n} for n in BANDS], "report": m.report, "frames": frames,
        }
    items = [f'<Item name="DESCRIPTION" sample="{i}" role="description">{n}</Item>' for i, n in enumerate(BANDS)]
    embedded = {k: v for k, v in info.items() if k != "frame_models"}
    items.append(f'<Item name="dji_thermal_visual_mosaic">{escape(json.dumps(embedded, separators=(",", ":")))}</Item>')
    extra = {42112: tiffout.tag("s", "<GDALMetadata>" + "".join(items) + "</GDALMetadata>")}
    if located:
        extra.update({
            33550: tiffout.tag("d", (g.gsd, g.gsd, 0.0)),
            33922: tiffout.tag("d", (0.0, 0.0, 0.0, g.e0, g.n0, 0.0)),
            34735: tiffout.tag("H", [1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, g.epsg, 3076, 0, 1, 9001]),
        })
    with files.replaced_when_done(path) as tmp:
        tiffout.write_tiff(tmp, _rgba(m), None, band_name=False, compression=compression, extra_tags=extra)
    sidecar = path.with_suffix(".json")
    with files.replaced_when_done(sidecar) as tmp:
        tmp.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return path, sidecar


def _rgba(m: VisualMosaic) -> np.ndarray:
    return np.dstack([m.rgb, np.where(m.source >= 0, 255, 0).astype(np.uint8)])


def write_preview(m: VisualMosaic, path: Path, longest: int = PREVIEW_PX) -> tuple[Path, int]:
    """A PNG quick look, transparent where no frame covers, reduced by a whole factor until its
    longer side is at most `longest` pixels. Returns the path and the factor."""
    factor = max(1, math.ceil(max(m.grid.width, m.grid.height) / longest))
    image = Image.fromarray(_rgba(m))
    if factor > 1:
        image = image.reduce(factor)
    with files.replaced_when_done(path) as tmp:
        image.save(tmp)
    return path, factor
