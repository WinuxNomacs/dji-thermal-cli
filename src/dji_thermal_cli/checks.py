"""`check`: a dataset's errors and anomalies, each with the options that address it.

inspect() looks at one file (its content-based kind, its listing, a hash of its bytes);
analyze() compares the files with each other and returns findings:

- error:   a file that can't be used (named like an image but isn't one, unreadable, corrupt,
           an unsupported format) or an exact duplicate of another file
- warning: something that changes the measurement or the mosaic and that an option can fix
           or that deserves a look (parameters that differ between files, outlying frames,
           missing pose or rangefinder readings, a damaged preview)
- info:    facts worth knowing that need no action (ordinary JPEGs mixed in, oblique frames
           the mosaic leaves out)

Outliers are found with robust statistics: a value is an outlier when it is more than
OUTLIER_Z robust standard deviations (1.4826 x the median absolute deviation) from the
dataset's median, and also more than a minimum difference that matters in practice.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np

from . import dataset, listing, mosaic, units

ERROR, WARNING, INFO = "error", "warning", "info"
SEVERITIES = (ERROR, WARNING, INFO)

OUTLIER_Z = 3.5                 # robust z-score beyond which a value is an outlier
MIN_TEMPERATURE_DIFF_C = 5.0    # ... and at least this far from the median (image temperatures)
MIN_PITCH_DIFF_DEG = 5.0        # gimbal pitch among the frames the mosaic uses
MIN_ALTITUDE_DIFF_M = 3.0       # relative altitude
MIN_LRF_DIFF_M = 3.0            # rangefinder distance
AMBIENT_SPREAD_C = 0.5          # embedded ambient/reflected temperatures varying more than this
RTK_STD_M = 0.05                # RTK position accuracy worse than this (any axis)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".jpe", ".jfif")


@dataclasses.dataclass
class Finding:
    severity: str
    code: str
    message: str
    files: list[str] = dataclasses.field(default_factory=list)
    suggestion: str = ""


@dataclasses.dataclass
class FileReport:
    path: str
    status: dataset.FileStatus
    info: dict | None                   # listing.describe() groups, for thermal files
    sha1: str | None


def inspect(path: Path, unit: str = "C") -> FileReport:
    """One file: what it is (with a full image decode), its listing and a hash of its bytes."""
    try:
        sha1 = hashlib.sha1(path.read_bytes(), usedforsecurity=False).hexdigest()
    except OSError:
        sha1 = None
    info, status = listing.describe(path, unit, stats=True, exif_all=False, decode=True)
    if info is not None:
        status = dataset.FileStatus(path, dataset.THERMAL, "", info["thermal"]["plugin"], info["file"]["warnings"])
    return FileReport(str(path), status, info, sha1)


# --- analysis -----------------------------------------------------------------------------

def analyze(reports: list[FileReport], others: list[Path], unit: str = "C",
            max_pitch: float = mosaic.DEFAULT_MAX_PITCH) -> list[Finding]:
    """Every finding for a dataset: its JPEG files' reports and the non-JPEG files beside them."""
    findings = _file_findings(reports, others)
    thermal = [r for r in reports if r.info is not None]
    if thermal:
        findings += _camera_findings(thermal)
        findings += _parameter_findings(thermal, unit)
        findings += _temperature_findings(thermal, unit, max_pitch)
        findings += _pose_findings(thermal, max_pitch)
        findings += _rangefinder_findings(thermal)
    order = {s: i for i, s in enumerate(SEVERITIES)}
    return sorted(findings, key=lambda f: order[f.severity])


def _file_findings(reports: list[FileReport], others: list[Path]) -> list[Finding]:
    out = []
    named = [p for p in others if p.suffix.lower() in IMAGE_EXTENSIONS]
    if named:
        out.append(Finding(ERROR, "not_jpeg", "named like JPEG images but their content isn't a JPEG",
                           [str(p) for p in named], "remove or replace them; they are ignored"))
    by_kind: dict[str, list[FileReport]] = {}
    for r in reports:
        by_kind.setdefault(r.status.kind, []).append(r)
    for kind, label in ((dataset.UNREADABLE, "can't be read"), (dataset.CORRUPT, "are damaged"),
                        (dataset.NOT_JPEG, "aren't JPEGs"),
                        (dataset.UNSUPPORTED, "are DJI thermal images in a format this tool can't measure")):
        for r in by_kind.get(kind, []):
            out.append(Finding(ERROR, kind, f"{label}: {r.status.reason}", [r.path], "they are skipped"))
    for r in by_kind.get(dataset.THERMAL, []):
        for w in r.status.warnings:
            out.append(Finding(WARNING, "damaged_preview", w, [r.path],
                               "temperatures are unaffected; the camera's own preview image is incomplete"))
    hashes: dict[str, list[str]] = {}
    for r in reports:
        if r.sha1:
            hashes.setdefault(r.sha1, []).append(r.path)
    for paths in hashes.values():
        if len(paths) > 1:
            out.append(Finding(ERROR, "duplicate", f"{len(paths)} files are byte-for-byte identical", paths,
                               "keep one: duplicates count twice in statistics and the mosaic"))
    rest = len(by_kind.get(dataset.NOT_THERMAL, [])) + len(others) - len(named)
    if rest:
        out.append(Finding(INFO, "other_files", f"{rest} file(s) without thermal data are ignored "
                           f"({len(by_kind.get(dataset.NOT_THERMAL, []))} ordinary JPEGs, "
                           f"{len(others) - len(named)} other files)"))
    return out


def _camera_findings(thermal: list[FileReport]) -> list[Finding]:
    groups: dict[tuple, list[str]] = {}
    for r in thermal:
        c = r.info["camera"]
        groups.setdefault((c["model"], c["width"], c["height"]), []).append(r.path)
    if len(groups) < 2:
        return []
    ranked = sorted(groups.items(), key=lambda g: -len(g[1]))          # the largest group first
    detail = ", ".join(f"{m} {w}x{h}: {len(p)}" for (m, w, h), p in ranked)
    return [Finding(WARNING, "mixed_cameras", f"the thermal images come from different cameras or resolutions ({detail})",
                    [p for _, ps in ranked[1:] for p in ps],
                    "process each camera's images separately; a mosaic needs one camera")]


_PARAMETERS = (  # (key, label, option, tolerance)
    ("emissivity", "emissivity", "--emissivity", 0.005),
    ("humidity", "humidity", "--humidity", 0.5),
    ("distance", "embedded distance", "--distance", 0.05),
)


def _parameter_findings(thermal: list[FileReport], unit: str) -> list[Finding]:
    out = []
    for key, label, option, tol in _PARAMETERS:
        values = [(r.path, r.info["thermal"]["measurement_params"][key]) for r in thermal]
        values = [(p, v) for p, v in values if v is not None]
        if not values:
            continue
        common = _mode([v for _, v in values])
        odd = [(p, v) for p, v in values if abs(v - common) > tol]
        if odd:
            seen = sorted({v for _, v in odd})
            out.append(Finding(WARNING, f"{key}_differs",
                               f"{label} differs between files: {len(values) - len(odd)} at {common:g}, "
                               f"{len(odd)} at {', '.join(f'{v:g}' for v in seen[:5])}{'...' if len(seen) > 5 else ''}",
                               [p for p, _ in odd],
                               f"set one value for every file with {option}" if key != "distance" else
                               "--distance lrf (the default) uses each file's rangefinder reading; or set one with --distance METRES"))
    sym = units.SYMBOLS[unit]
    temps = {key: [r.info["thermal"]["measurement_params"][key] for r in thermal] for key in ("ambient_temp", "reflection")}
    pairs = [("ambient_temp", "ambient", "--ambient-temp"), ("reflection", "reflected", "--reflected-temp")]
    if temps["ambient_temp"] == temps["reflection"]:       # one embedded value serves both (older iirp headers)
        pairs = [("ambient_temp", "ambient and reflected", "--ambient-temp and --reflected-temp")]
    for key, label, option in pairs:
        values = [v for v in temps[key] if v is not None]
        if len(values) < 2:
            continue
        spread = max(values) - min(values)
        spread_c = spread if unit != "F" else spread * 5 / 9
        if spread_c > AMBIENT_SPREAD_C:
            out.append(Finding(WARNING, f"{key}_varies",
                               f"embedded {label} temperature varies by {spread:.2f} {sym} across files "
                               f"({min(values):.2f} to {max(values):.2f}); it scales each image's temperatures "
                               "with distance, so frames of the same scene disagree",
                               [], f"--mosaic-normalize (uses the flight median) or set one value with {option}"))
    return out


def _temperature_findings(thermal: list[FileReport], unit: str, max_pitch: float) -> list[Finding]:
    out = []
    oblique = {r.path for r in thermal if (p := r.info["attitude"]["gimbal_pitch"]) is not None and p > max_pitch}
    to_unit = (lambda d: d) if unit in ("C", "K") else (lambda d: d * 9 / 5)
    for stat, label in (("max", "maximum"), ("min", "minimum"), ("mean", "mean")):
        values = [(r.path, r.info["temperature"][stat]) for r in thermal
                  if r.info["temperature"] and r.info["temperature"][stat] is not None]
        flagged = _outliers(values, to_unit(MIN_TEMPERATURE_DIFF_C))
        if flagged:
            med = float(np.median([v for _, v in values]))
            detail = ", ".join(f"{Path(p).name} {v:.1f}" for p, v in flagged[:5])
            hint = ("--replace-above/--replace-above-value (or --replace-below/...) to replace or blank hot or cold spots such as sun glint; "
                    "--range-min/--range-max for a fixed colour scale") if stat != "mean" else \
                   "check what these frames see; --mosaic-normalize evens out per-frame offsets"
            n_oblique = sum(p in oblique for p, _ in flagged)
            seen = " (sky in view?)" if stat == "min" else ""
            why = f"; {n_oblique} of them are oblique frames{seen}, which a mosaic leaves out" if n_oblique else ""
            out.append(Finding(WARNING, f"temperature_{stat}_outlier",
                               f"{len(flagged)} image(s) with an unusual {label} temperature "
                               f"(dataset median {med:.1f} {units.SYMBOLS[unit]}): {detail}{why}",
                               [p for p, _ in flagged], hint))
    return out


def _pose_findings(thermal: list[FileReport], max_pitch: float) -> list[Finding]:
    out = []
    no_pose = [r.path for r in thermal if r.info["geospatial"]["latitude"] is None
               or r.info["attitude"]["gimbal_pitch"] is None]
    if no_pose:
        out.append(Finding(WARNING, "no_pose", f"{len(no_pose)} file(s) have no position or gimbal attitude",
                           no_pose, "they can be processed but not placed in a mosaic"))
    posed = [r for r in thermal if r.path not in set(no_pose)]
    oblique = [r.path for r in posed if r.info["attitude"]["gimbal_pitch"] > max_pitch]
    nadir = [r for r in posed if r.info["attitude"]["gimbal_pitch"] <= max_pitch]
    if oblique:
        out.append(Finding(INFO, "oblique", f"{len(oblique)} frame(s) look less steeply down than {max_pitch:g} deg "
                           f"and are left out of a mosaic ({len(nadir)} nadir frames remain)", oblique,
                           "--mosaic-max-pitch DEG to include them"))
    pitches = [(r.path, r.info["attitude"]["gimbal_pitch"]) for r in nadir]
    flagged = _outliers(pitches, MIN_PITCH_DIFF_DEG)
    if flagged:
        out.append(Finding(WARNING, "pitch_outlier", f"{len(flagged)} mosaic frame(s) with an unusual gimbal pitch: "
                           + ", ".join(f"{Path(p).name} {v:g} deg" for p, v in flagged[:5]), [p for p, _ in flagged],
                           "--gimbal-pitch DEG if the recorded angle is wrong, or --mosaic-max-pitch to leave them out"))
    altitudes = [(r.path, r.info["geospatial"]["relative_altitude_m"]) for r in nadir
                 if r.info["geospatial"]["relative_altitude_m"] is not None]
    flagged = _outliers(altitudes, MIN_ALTITUDE_DIFF_M)
    if flagged:
        out.append(Finding(WARNING, "altitude_outlier", f"{len(flagged)} mosaic frame(s) at an unusual altitude: "
                           + ", ".join(f"{Path(p).name} {v:.1f} m" for p, v in flagged[:5]), [p for p, _ in flagged],
                           "their ground resolution differs from the rest (take-off or landing frames?)"))
    statuses = {}
    for r in posed:
        statuses.setdefault(r.info["geospatial"]["gps_status"], []).append(r.path)
    if len(statuses) > 1:
        best = max(statuses, key=lambda k: len(statuses[k]))
        odd = [p for k, ps in statuses.items() if k != best for p in ps]
        detail = ", ".join(f"{k or 'none'}: {len(v)}" for k, v in statuses.items())
        out.append(Finding(WARNING, "gps_status_differs", f"GPS fix differs between files ({detail})", odd,
                           "positions without RTK are metres less accurate; the mosaic's alignment absorbs some of it"))
    loose = [r.path for r in posed if (std := r.info["geospatial"]["rtk_std_m"])
             and max(v for v in std.values() if v is not None) > RTK_STD_M]
    if loose:
        out.append(Finding(WARNING, "rtk_accuracy", f"{len(loose)} file(s) report RTK accuracy worse than "
                           f"{RTK_STD_M * 100:.0f} cm", loose, "check the RTK link for that part of the flight"))
    return out


def _rangefinder_findings(thermal: list[FileReport]) -> list[Finding]:
    out = []
    readings = [(r.path, r.info["geospatial"]["rangefinder"]) for r in thermal]
    with_lrf = [(p, lrf) for p, lrf in readings if lrf is not None]
    if not with_lrf:
        return out
    bad = [p for p, lrf in with_lrf if (lrf["status"] or "").lower() != "normal" or not lrf["distance_m"]]
    if bad:
        out.append(Finding(WARNING, "rangefinder_invalid", f"{len(bad)} file(s) have no valid rangefinder reading",
                           bad, "their embedded distance is used instead; set one with --distance METRES if that's wrong"))
    good = [(p, lrf["distance_m"]) for p, lrf in with_lrf if p not in set(bad)]
    embedded = [r.info["thermal"]["measurement_params"]["distance"] for r in thermal]
    if good and embedded:
        lrf_med, emb_med = float(np.median([v for _, v in good])), float(np.median(embedded))
        if abs(lrf_med - emb_med) > 1.0:
            out.append(Finding(INFO, "rangefinder_distance",
                               f"the embedded object distance is {emb_med:g} m but the rangefinder reads "
                               f"{lrf_med:.1f} m (median); --distance lrf (the default) measures with the rangefinder's",
                               [], "--distance embedded keeps the embedded distance instead"))
    flagged = _outliers(good, MIN_LRF_DIFF_M)
    if flagged:
        out.append(Finding(INFO, "rangefinder_outlier", f"{len(flagged)} file(s) with an unusual rangefinder distance: "
                           + ", ".join(f"{Path(p).name} {v:.1f} m" for p, v in flagged[:5]), [p for p, _ in flagged],
                           "the laser hit something nearer or farther (a roof, a tree, the horizon); "
                           "--distance METRES sets one distance for every file"))
    return out


# --- statistics -----------------------------------------------------------------------------

def _outliers(values: list[tuple[str, float]], min_diff: float) -> list[tuple[str, float]]:
    """The (path, value) pairs more than OUTLIER_Z robust standard deviations and more than
    min_diff away from the median, farthest first."""
    if len(values) < 5:
        return []
    v = np.array([x for _, x in values], dtype=np.float64)
    med = np.median(v)
    sigma = 1.4826 * np.median(np.abs(v - med))
    far = np.abs(v - med) > max(min_diff, OUTLIER_Z * sigma)
    picked = [(p, x) for (p, x), f in zip(values, far) if f]
    return sorted(picked, key=lambda px: -abs(px[1] - med))


def _mode(values: list[float]) -> float:
    uniq, counts = np.unique(np.array(values, dtype=np.float64), return_counts=True)
    return float(uniq[np.argmax(counts)])


def clash_findings(clashes: dict) -> list[Finding]:
    """Files whose outputs would overwrite another's (see dataset.output_clashes)."""
    if not clashes:
        return []
    return [Finding(WARNING, "output_name_clash",
                    f"{len(clashes)} file(s) would write outputs with the same names as another file's "
                    "(names are compared ignoring case, as on Windows and macOS): process skips them",
                    [str(p) for p in clashes], "rename them, e.g. a.jpg and a.jpeg, or A.jpg and a.jpg")]


def summary(findings: list[Finding], reports: list[FileReport]) -> dict:
    kinds: dict[str, int] = {}
    for r in reports:
        kinds[r.status.kind] = kinds.get(r.status.kind, 0) + 1
    return {"files": kinds, **{s: sum(f.severity == s for f in findings) for s in SEVERITIES}}
