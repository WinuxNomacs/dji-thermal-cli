"""`check --webodm`: how to prepare a thermal dataset for WebODM / ODM, and which task options to use.

From the dataset's check reports and camera poses it works out:

- which frames to give ODM: nadir frames for an orthophoto (obliques help a 3D model but
  blur an orthophoto's temperatures), and none without a position;
- coverage: how many frames see each ground point, from every nadir frame's footprint on
  the ground plane, with the equivalent front/side overlap. Thermal images are small and low
  in texture, so ODM needs a lot of overlap to match them;
- the `dji-thermal-cli process` command that prepares the images: float32 Celsius TIFFs
  carrying GPS and Camera:BandName=LWIR, consistent measurement parameters, normalized frames;
- ODM task options with the reason for each (names and units as in docs.opendronemap.org).

Recommendations are split into `recommended` (set these) and `consider` (depends on the site
and what you want from it).
"""

from __future__ import annotations

import dataclasses
import math
import shlex

import numpy as np

from . import checks, geo, mosaic

MIN_IMAGES_PER_POINT = 9         # about 67 % front and side overlap; fewer and matching suffers
GOOD_IMAGES_PER_POINT = 16       # about 75 % front and side overlap, the usual thermal advice
MIN_FRAMES = 5
COVERAGE_CELL_GSD = 8            # coverage raster cell, in ground sample distances


@dataclasses.dataclass
class Option:
    value: object
    reason: str


@dataclasses.dataclass
class Plan:
    process_args: list[str]                     # dji-thermal-cli process arguments (without paths)
    process_command: str
    recommended: dict[str, Option]              # ODM task options to set
    consider: dict[str, Option]                 # ODM task options that depend on the site
    notes: list[str]
    findings: list[checks.Finding]              # WebODM-specific findings, also in the check result
    coverage: dict | None

    def odm_arguments(self) -> list[str]:
        """The recommended options as ODM command-line arguments."""
        out = []
        for name, opt in self.recommended.items():
            if opt.value is True:
                out.append(f"--{name}")
            else:
                out += [f"--{name}", str(opt.value)]
        return out


def plan(reports: list[checks.FileReport], poses: dict[str, mosaic.Pose], findings: list[checks.Finding],
         max_pitch: float = mosaic.DEFAULT_MAX_PITCH) -> Plan:
    """The WebODM plan for a dataset: its check reports, the poses of its thermal files (by
    path) and the check's findings."""
    thermal = [r for r in reports if r.info is not None]
    codes = {f.code for f in findings}
    args = ["-o", "temp_c:tiff"]
    notes, own = [], []
    recommended: dict[str, Option] = {
        "radiometric-calibration": Option("none", "the TIFFs already hold temperatures in degrees C; ODM's "
                                          "camera calibration is for raw thermal images"),
    }
    consider: dict[str, Option] = {}

    nadir = [r for r in thermal if r.path in poses and poses[r.path].pitch <= max_pitch]
    oblique = [r for r in thermal if r.path in poses and poses[r.path].pitch > max_pitch]
    no_pose = [r for r in thermal if r.path not in poses]
    if oblique:
        args += ["--max-pitch", f"{max_pitch:g}"]
        notes.append(f"{len(oblique)} oblique frame(s) left out: they help a 3D model but mix viewing angles "
                     "(and sky) into an orthophoto's temperatures; drop --max-pitch to keep them for 3D")
    if no_pose:
        own.append(checks.Finding(checks.WARNING, "webodm_no_gps", f"{len(no_pose)} thermal file(s) have no "
                                  "position: ODM can't georeference them", [r.path for r in no_pose],
                                  "leave them out of the WebODM task"))
    if len(nadir) < MIN_FRAMES:
        own.append(checks.Finding(checks.ERROR if not nadir else checks.WARNING, "webodm_too_few",
                                  f"only {len(nadir)} nadir frame(s) with a position: ODM needs at least "
                                  f"{MIN_FRAMES} overlapping images", [r.path for r in nadir],
                                  "fly more overlapping nadir frames"))

    # measurement parameters: one value for every file
    for key, option in (("emissivity", "--emissivity"), ("humidity", "--humidity")):
        if f"{key}_differs" in codes:
            values = [r.info["thermal"]["measurement_params"][key] for r in nadir or thermal]
            common = checks._mode([v for v in values if v is not None])
            args += [option, f"{common:g}"]
            notes.append(f"{option} {common:g}: the files differ, and frames stitched together must be "
                         "measured alike")
    notes.append("don't add --strip-location here: ODM needs the GPS to place the frames; to share images "
                 "without their location, make a separate --strip-location copy")
    if len(nadir) >= 2:
        args.append("--normalize")
        why = "the embedded ambient temperature drifts between files, and " if "ambient_temp_varies" in codes else ""
        notes.append(f"--normalize: {why}uncooled thermal cameras drift between shutter recalibrations; "
                     "ODM blends overlapping frames, so their offsets become visible seams")
    for stat, side in (("max", "above"), ("min", "below")):
        hits = next((f for f in findings if f.code == f"temperature_{stat}_outlier"), None)
        if hits and any(p in {r.path for r in nadir} for p in hits.files):
            notes.append(f"consider --replace-{side} TEMP --replace-{side}-value TEMP: some nadir frames have "
                         f"outlying {stat}imum temperatures (sun glint, hot exhausts or sky); see the check above")

    # geometry: resolution, GPS accuracy, coverage
    coverage = _coverage([poses[r.path] for r in nadir]) if nadir else None
    if coverage:
        gsd_cm = coverage["gsd_m"] * 100
        recommended["orthophoto-resolution"] = Option(
            math.ceil(gsd_cm * 10) / 10, f"the frames' ground sample distance is {gsd_cm:.1f} cm/px; finer only "
            "interpolates (ODM's default is 5 cm/px)")
        n = coverage["images_per_point_median"]
        if n < MIN_IMAGES_PER_POINT:
            own.append(checks.Finding(checks.WARNING, "webodm_low_overlap",
                                      f"each ground point is seen by {n:g} nadir frame(s) (median), about "
                                      f"{coverage['equivalent_overlap_pct']:.0f} % front and side overlap: thermal "
                                      "images may fail to match", [],
                                      f"fly with at least {_overlap(GOOD_IMAGES_PER_POINT):.0f} % front and side overlap"))
        if n < GOOD_IMAGES_PER_POINT:
            consider["feature-quality"] = Option("ultra", "thermal images are small and low in texture; more "
                                                 "features help ODM match them at this overlap")
            consider["min-num-features"] = Option(20000, "more features per image for low-texture thermal frames "
                                                  "(ODM's default is 10000)")
    rtk = [max(v for v in r.info["geospatial"]["rtk_std_m"].values() if v is not None)
           for r in nadir if r.info["geospatial"]["rtk_std_m"]]
    if rtk and len(rtk) == len(nadir):
        accuracy = max(0.02, round(2 * float(np.median(rtk)), 2))
        recommended["gps-accuracy"] = Option(accuracy, f"RTK positions, {float(np.median(rtk)) * 100:.1f} cm accuracy "
                                             "(median); ODM may set this from RTK tags itself, setting it is explicit")
    elif nadir:
        consider["gps-accuracy"] = Option(3, "no RTK accuracy recorded; ODM's default of 3 m suits ordinary GPS")
    consider["fast-orthophoto"] = Option(True, "for an orthophoto of mostly flat ground: skips the dense "
                                         "reconstruction, which low-texture thermal images rarely support well")
    consider["skip-3dmodel"] = Option(True, "when only the orthophoto is wanted")

    command = "dji-thermal-cli process --input-dir INPUT --output-dir OUTPUT " + shlex.join(args)
    return Plan(args, command, recommended, consider, notes, own, coverage)


def _coverage(poses: list[mosaic.Pose]) -> dict | None:
    """Frames per ground point over the area the nadir frames cover, from their footprints on
    the ground plane (the mosaic's: median rangefinder target altitude, else take-off)."""
    lrf = [p.ground_alt for p in poses if p.ground_alt is not None]
    takeoff = [p.takeoff_alt for p in poses if p.takeoff_alt is not None]
    if not (lrf or takeoff):
        return None
    ground = float(np.median(lrf or takeoff))
    lat, lon = float(np.mean([p.lat for p in poses])), float(np.mean([p.lon for p in poses]))
    zone, north = geo.utm_zone(lat, lon)
    placed = [pl for p in poses if (pl := mosaic._place(mosaic.Frame("", np.zeros((1, 1), np.float32), p),
                                                         ground, zone, north)) is not None]
    if not placed:
        return None
    gsd = float(np.median([pl.h / pl.frame.pose.focal_px for pl in placed]))
    cell = max(gsd * COVERAGE_CELL_GSD, 0.05)
    allfp = np.concatenate([pl.footprint for pl in placed])
    e0, n0 = allfp[:, 0].min(), allfp[:, 1].max()
    w = int(math.ceil((allfp[:, 0].max() - e0) / cell)) + 1
    h = int(math.ceil((n0 - allfp[:, 1].min()) / cell)) + 1
    if w * h > 25_000_000:
        return None
    ee, nn = np.meshgrid(e0 + (np.arange(w) + 0.5) * cell, n0 - (np.arange(h) + 0.5) * cell)
    count = np.zeros((h, w), np.int32)
    for pl in placed:
        count += _inside(pl.footprint, ee, nn)
    covered = count > 0
    if not covered.any():
        return None
    per_point = count[covered]
    median = float(np.median(per_point))
    return {
        "frames": len(placed), "gsd_m": round(gsd, 4), "area_m2": round(float(covered.sum()) * cell * cell, 1),
        "images_per_point_median": median,
        "images_per_point_p10": float(np.percentile(per_point, 10)),
        "area_seen_by_3_or_more_pct": round(100.0 * float((per_point >= 3).mean()), 1),
        "equivalent_overlap_pct": round(_overlap(median), 1),
    }


def _overlap(images_per_point: float) -> float:
    """The equal front and side overlap (percent) at which each point is seen by this many
    images: n = 1 / (1 - overlap)^2."""
    return 100.0 * max(0.0, 1.0 - 1.0 / math.sqrt(max(images_per_point, 1.0)))


def _inside(poly: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Points inside a convex polygon of either winding. Repeated vertices (zero-length edges)
    give a cross product of 0, which counts as inside for both windings."""
    left = np.ones(x.shape, bool)
    right = np.ones(x.shape, bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        left &= cross >= 0
        right &= cross <= 0
    return left | right
