"""The Python API: everything the command line does, callable in-process.

    import dji_thermal_cli as dtc

    report = dtc.check("flight/")                        # errors and anomalies
    rows = dtc.list_dataset("flight/", unit="F").rows    # every file, grouped
    m = dtc.measure("flight/DJI_0001_T.JPG", dtc.Settings(emissivity=0.95), unit="K")
    m.temperature                                        # float32 kelvin, in memory
    dtc.process("flight/", "out/", outputs=["temp_c:tiff", "temp_f:exr"])
    dtc.make_mosaic("flight/", "out/", dtc.Settings(), dtc.MosaicOptions(normalize=True))

Nothing here prints or exits. Settings that can't be used raise SettingsError; a path that
doesn't exist raises FileNotFoundError; a mosaic that can't be built raises MosaicError. A
file that can't be read is not an exception in dataset functions: it is reported in the
result, and the rest of the dataset goes on. `progress` callbacks receive an Event per file
(and per step) as the work happens; the command line prints from them.

The SDK port underneath (dji_thermal_cli.tsdk) is DJI's Thermal SDK v1.8, bit for bit; this
layer only converts units, applies the dataset-wide settings and writes files.
"""

from __future__ import annotations

import dataclasses
import os
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import adjust, checks, core, dataset, listing, mosaic as _mosaic, outputs as _outputs, units, webodm as _webodm

__all__ = [
    "Settings", "MosaicOptions", "Event", "SettingsError", "MosaicError", "FileProblem",
    "ListResult", "CheckResult", "Measurement", "ProcessReport", "MosaicReport",
    "Normalization", "scan", "classify", "describe", "list_dataset", "check", "measure", "process",
    "make_mosaic", "normalization",
    "PALETTES", "palette_index", "parse_outputs",
]

PALETTES = tuple(core.PSEUDO_COLOR_NAMES)
DEFAULT_MAX_WORKERS = 8          # past this, worker startup and disk I/O outweigh the parallelism

ProgressFn = Callable[["Event"], None]


# --- errors ---------------------------------------------------------------------------------

class SettingsError(ValueError):
    """Settings that can't be used. The message names the Settings fields involved."""


class MosaicError(RuntimeError):
    """No mosaic could be built (no usable frames, no ground altitude, ...). counts: the
    frames found, oblique, unusable and not thermal."""

    def __init__(self, message: str, counts: dict | None = None):
        super().__init__(message)
        self.counts = counts or {}


class FileProblem(ValueError):
    """describe() of a file that isn't a thermal R-JPEG this package can measure."""

    def __init__(self, status: dataset.FileStatus):
        super().__init__(f"{status.path}: {status.reason or status.kind}")
        self.status = status


# --- settings -------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Settings:
    """How to measure a dataset. None keeps each file's own embedded value.

    Temperatures (reflected_temp, ambient_temp, range_min/max, replace_*) are in `unit`
    (C, F or K) and are converted to Celsius for the SDK. distance is "lrf" (each file's
    laser-rangefinder distance where valid, else its embedded one), "embedded", or metres.
    humidity is a percent. palette is a number 0-9 or a name (see PALETTES).
    replace_*_value is a temperature or "nodata".
    """

    distance: str | float = "lrf"
    humidity: float | None = None
    emissivity: float | None = None
    reflected_temp: float | None = None
    ambient_temp: float | None = None
    unit: str = "C"
    range_min: float | None = None
    range_max: float | None = None
    palette: int | str | None = None
    gimbal_yaw: float | None = None
    gimbal_pitch: float | None = None
    gimbal_roll: float | None = None
    replace_above: float | None = None
    replace_above_value: float | str | None = None
    replace_below: float | None = None
    replace_below_value: float | str | None = None
    strip_location: bool = False        # outputs without location or identifiers (see privacy)

    def __post_init__(self):
        unit = str(self.unit).upper()
        if unit not in units.UNITS:
            raise SettingsError(f"unit must be C, F or K, not {self.unit!r}")
        object.__setattr__(self, "unit", unit)
        if not (self.distance in ("lrf", "embedded") or isinstance(self.distance, (int, float))):
            raise SettingsError("distance must be 'lrf', 'embedded' or a distance in metres")
        if isinstance(self.distance, (int, float)) and not isinstance(self.distance, bool):
            object.__setattr__(self, "distance", float(self.distance))
        if self.palette is not None:
            object.__setattr__(self, "palette", palette_index(self.palette))
        if (self.range_min is None) != (self.range_max is None):
            raise SettingsError("range_min and range_max must be given together")
        if self.range_min is not None and self.range_min >= self.range_max:
            raise SettingsError("range_max must be greater than range_min")
        for side in ("above", "below"):
            threshold, value = getattr(self, f"replace_{side}"), getattr(self, f"replace_{side}_value")
            if (threshold is None) != (value is None):
                raise SettingsError(f"replace_{side} and replace_{side}_value must be given together")
            if isinstance(value, str) and value.lower() != "nodata":
                raise SettingsError(f"replace_{side}_value must be a temperature or 'nodata'")
        if self.replace_above is not None and self.replace_below is not None \
                and self.replace_below >= self.replace_above:
            raise SettingsError("replace_below must be lower than replace_above")
        for name in ("reflected_temp", "ambient_temp", "range_min", "range_max", "replace_above",
                     "replace_below", "replace_above_value", "replace_below_value"):
            self._celsius(name)                      # rejects values below absolute zero

    def _celsius(self, name: str) -> float | None:
        value = getattr(self, name)
        if value is None:
            return None
        if isinstance(value, str):                   # "nodata"
            return adjust.NO_DATA
        c = units.to_celsius(value, self.unit)
        if c < units.ABSOLUTE_ZERO_C:
            raise SettingsError(f"{name} {value:g} {self.unit} is below absolute zero")
        return c

    def overrides(self) -> core.MeasurementOverrides:
        """The SDK-side measurement parameters (Celsius)."""
        return core.MeasurementOverrides(
            distance=self.distance if isinstance(self.distance, float) else None,
            humidity=self.humidity, emissivity=self.emissivity,
            reflection=self._celsius("reflected_temp"), ambient_temp=self._celsius("ambient_temp"),
            use_rangefinder=self.distance == "lrf")

    def adjustments(self) -> adjust.Adjustments:
        """The gimbal override and replacement (Celsius)."""
        return adjust.Adjustments(
            gimbal_yaw=self.gimbal_yaw, gimbal_pitch=self.gimbal_pitch, gimbal_roll=self.gimbal_roll,
            replace_above=self._celsius("replace_above"), replace_above_value=self._celsius("replace_above_value"),
            replace_below=self._celsius("replace_below"), replace_below_value=self._celsius("replace_below_value"),
            strip_location=self.strip_location)

    def range_c(self) -> tuple[float | None, float | None]:
        """The fixed colour-bar range in Celsius, or (None, None)."""
        return self._celsius("range_min"), self._celsius("range_max")


@dataclasses.dataclass(frozen=True)
class MosaicOptions:
    """How to build the mosaic. units: one GeoTIFF per unit (C, F, K). heights: "auto" projects
    the frames onto the surface estimated from their parallax (with align; see mosaic._align),
    on cells height_cell mosaic pixels wide; "plane" onto the flat ground."""

    max_pitch: float = _mosaic.DEFAULT_MAX_PITCH
    align: bool = True
    normalize: bool = False
    gsd: float | None = None
    units: tuple[str, ...] = ("C",)
    heights: str = "auto"
    height_cell: int = _mosaic.HEIGHT_CELL_PX

    def __post_init__(self):
        chosen = tuple(dict.fromkeys(str(u).upper() for u in self.units))
        if not chosen or any(u not in units.UNITS for u in chosen):
            raise SettingsError(f"units must be C, F and/or K, not {self.units!r}")
        object.__setattr__(self, "units", chosen)
        if self.gsd is not None and self.gsd <= 0:
            raise SettingsError("gsd must be positive")
        if self.heights not in ("auto", "plane"):
            raise SettingsError(f"heights must be 'auto' or 'plane', not {self.heights!r}")
        if not 8 <= self.height_cell <= 1024:
            raise SettingsError("height_cell must be 8-1024 pixels")


def palette_index(palette: int | str) -> int:
    """A palette by number (0-9) or name (e.g. 'hotiron') as its number."""
    key = str(palette).strip().lower()
    if key.isdigit() and int(key) < len(PALETTES):
        return int(key)
    if key in PALETTES:
        return PALETTES.index(key)
    raise SettingsError(f"palette must be 0-{len(PALETTES) - 1} or one of {', '.join(PALETTES)}")


def parse_outputs(specs: Iterable[str | tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Output specs ('temp_c', 'temp_f:exr', or (product, format)) as (product, extension) pairs."""
    pairs = []
    for spec in specs:
        if isinstance(spec, tuple):
            spec = ":".join(spec)
        for part in str(spec).split(","):
            if part.strip():
                try:
                    pairs.append(_outputs.parse_output(part))
                except ValueError as e:
                    raise SettingsError(str(e)) from None
    return tuple(dict.fromkeys(pairs))


# --- progress -------------------------------------------------------------------------------

@dataclasses.dataclass
class Event:
    """Something that happened while working through a dataset.

    kind: "unreadable" (path, detail = reason), "non_jpeg" (data["count"]), "file" (one file
    done: path, data["result"]), "note" (path, detail), "normalize" (data["ambient_c"],
    data["reflected_c"]), "mosaic_skip" (path, detail), "mosaic" (a mosaic's progress, below).

    A mosaic (make_mosaic, or normalization for process(normalize=...)) goes through stages,
    each counted in its own unit; data = {"stage", "step", "steps", "done", "total", "unit"},
    step being the stage's place among the run's steps (1-based). The stages, in order:
    "ambient" (files: the embedded ambient temperatures read, when normalizing without
    them given), "frames" (files measured), "align" (per round, data["round"] of
    data["rounds"]: the overlapping pairs measured, counted by the area they overlap so the
    count moves evenly, then the solve's reweighting steps, unit "solve steps"),
    "composite" (frames), "check" (pairs, the misalignment report; make_mosaic only) and
    "write" (files; make_mosaic only). With make_mosaic's as_converted_dir, "frames",
    "composite" and "write" count both mosaics. Each stage starts with done = 0. There is no overall
    percentage: the stages' costs vary with the flight.
    """

    kind: str
    path: Path | None = None
    detail: str = ""
    data: dict = dataclasses.field(default_factory=dict)


def _emit(progress: ProgressFn | None, *args, **kwargs) -> None:
    if progress:
        progress(Event(*args, **kwargs))


def _stages(progress: ProgressFn | None, stages: list[str]):
    """A mosaic progress reporter, report(stage, done, total, unit, **extra), sending "mosaic"
    events numbered among stages; None without a progress callback."""
    if not progress:
        return None

    def report(stage: str, done: int, total: int, unit: str, **extra) -> None:
        progress(Event("mosaic", data={"stage": stage, "step": stages.index(stage) + 1, "steps": len(stages),
                                       "done": done, "total": total, "unit": unit, **extra}))

    return report


# --- datasets -------------------------------------------------------------------------------

def scan(path: str | os.PathLike) -> dataset.Scan:
    """The JPEG files under path by content (or path itself), the non-JPEG files beside them,
    and those that couldn't be read."""
    return dataset.scan(_existing(path))


def classify(path: str | os.PathLike, decode: bool = False) -> dataset.FileStatus:
    """What one file is, from its content (see dataset.classify)."""
    return dataset.classify(_existing(path), decode=decode)


def describe(path: str | os.PathLike, unit: str = "C", stats: bool = True, exif: bool = False) -> dict:
    """Every group about one thermal R-JPEG (see listing). Raises FileProblem otherwise."""
    info, status = listing.describe(_existing(path), _unit(unit), stats=stats, exif_all=exif)
    if info is None:
        raise FileProblem(status)
    if not exif:
        del info["exif"]
    return info


@dataclasses.dataclass
class ListResult:
    rows: list[dict]                            # describe() of every thermal file, in path order
    not_thermal: int                            # ordinary JPEGs skipped
    problems: list[dataset.FileStatus]          # files that couldn't be described
    scan: dataset.Scan


def list_dataset(path: str | os.PathLike, unit: str = "C", stats: bool = True, exif: bool = False,
                 jobs: int | None = None, progress: ProgressFn | None = None) -> ListResult:
    """describe() for every file of a dataset."""
    found = _scan(path, progress)
    unit = _unit(unit)
    rows, problems, not_thermal = [], [], 0
    arglist = [(src, unit, stats, exif) for src in found.jpegs]
    for i, ok, payload in _run_all(listing.describe, arglist, jobs):
        src = found.jpegs[i]
        info, status = payload if ok else (None, dataset.FileStatus(src, dataset.CORRUPT, payload))
        if info is not None:
            if not exif:
                del info["exif"]
            rows.append(info)
        elif status.kind == dataset.NOT_THERMAL:
            not_thermal += 1
        else:
            problems.append(status)
        _emit(progress, "file", src, data={"info": info, "status": status})
    rows.sort(key=lambda r: r["file"]["path"])
    problems.sort(key=lambda st: str(st.path))
    return ListResult(rows, not_thermal, problems, found)


@dataclasses.dataclass
class CheckResult:
    findings: list[checks.Finding]              # most severe first
    summary: dict                               # file kinds and finding counts
    reports: list[checks.FileReport]
    scan: dataset.Scan
    webodm: _webodm.Plan | None = None          # with check(webodm=True)

    def failed(self, strict: bool = False) -> bool:
        """True when there are errors (or, strict, warnings)."""
        return bool(self.summary[checks.ERROR] or (strict and self.summary[checks.WARNING]))


def check(path: str | os.PathLike, unit: str = "C", max_pitch: float = _mosaic.DEFAULT_MAX_PITCH,
          jobs: int | None = None, progress: ProgressFn | None = None, webodm: bool = False) -> CheckResult:
    """A dataset's errors and anomalies, each with the options that address it. webodm: also
    plan a WebODM / ODM task (see webodm): frame selection, coverage, the process command to
    prepare the images and the ODM options to use."""
    found = dataset.scan(_existing(path))
    unit = _unit(unit)
    reports = [checks.FileReport(str(p), dataset.FileStatus(p, dataset.UNREADABLE, why), None, None)
               for p, why in found.unreadable]
    for i, ok, payload in _run_all(checks.inspect, [(src, unit) for src in found.jpegs], jobs):
        src = found.jpegs[i]
        report = payload if ok else checks.FileReport(str(src), dataset.FileStatus(src, dataset.CORRUPT, payload), None, None)
        reports.append(report)
        _emit(progress, "file", src, data={"report": report})
    reports.sort(key=lambda r: r.path)
    root = _existing(path)
    thermal = {Path(r.path) for r in reports if r.info is not None}
    clashes = dataset.output_clashes(sorted(thermal), root if root.is_dir() else root.parent, thermal.__contains__)
    findings = checks.analyze(reports, found.others, unit, max_pitch) + checks.clash_findings(clashes)
    order = {s: i for i, s in enumerate(checks.SEVERITIES)}
    findings.sort(key=lambda f: order[f.severity])
    plan = None
    if webodm:
        poses = {}
        for r in reports:
            if r.info is not None:
                pose, _ = _mosaic.read_pose(Path(r.path).read_bytes())
                if pose is not None:
                    poses[r.path] = pose
        plan = _webodm.plan(reports, poses, findings, max_pitch)
        findings = sorted(findings + plan.findings, key=lambda f: order[f.severity])
    return CheckResult(findings, checks.summary(findings, reports), reports, found, plan)


# --- measuring and writing ------------------------------------------------------------------

@dataclasses.dataclass
class Measurement:
    path: Path
    temperature: np.ndarray                     # float32 in `unit`; no data = absolute zero
    unit: str
    params: dict                                # the parameters the SDK measured with (Celsius)
    distance_source: str                        # "option", "rangefinder" or "embedded"
    replaced_above: int
    replaced_below: int
    notes: list[str]


def measure(path: str | os.PathLike, settings: Settings = Settings(), unit: str = "C") -> Measurement:
    """One file's temperatures in memory, measured as process() would (settings applied),
    converted to `unit`. Raises the SDK's TsdkError family for files it can't measure."""
    src = _existing(path)
    unit = _unit(unit)
    m = core.measure_file(src, settings.overrides(), settings.adjustments())
    return Measurement(src, units.float32_in(m.temperature_c, unit), unit, m.params, m.distance_source,
                       m.replaced.above, m.replaced.below, m.notes)


@dataclasses.dataclass
class ProcessReport:
    results: list[core.ProcessResult]           # one per JPEG file, in path order
    scan: dataset.Scan
    normalization: Normalization | None = None

    @property
    def processed(self) -> int:
        return sum(r.ok for r in self.results)

    def distance_sources(self) -> dict[str, int]:
        return dict(Counter(r.distance_source for r in self.results if r.ok))


def process(path: str | os.PathLike, output_dir: str | os.PathLike, settings: Settings = Settings(),
            outputs: Iterable[str | tuple[str, str]] = ("temp_c:tiff", "pseudo:png"),
            compression: str = "deflate", jobs: int | None = None,
            progress: ProgressFn | None = None, normalize: bool | Normalization = False,
            max_pitch: float | None = None, mosaic_options: MosaicOptions = MosaicOptions()) -> ProcessReport:
    """Write the chosen outputs (see outputs.PRODUCTS) for every thermal file of a dataset,
    mirroring the dataset's folders under output_dir. Source files are never modified.

    normalize: make the frames' temperatures consistent with each other (see normalization();
    True computes it with mosaic_options, or pass one computed before). max_pitch: process only
    frames whose gimbal pitch is at or below it (e.g. -80 for nadir frames only)."""
    if compression not in _outputs.COMPRESSIONS:
        raise SettingsError(f"compression must be one of {', '.join(_outputs.COMPRESSIONS)}")
    wanted = parse_outputs(outputs)
    root = _existing(path)
    found = _scan(root, progress)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    src_root = root if root.is_dir() else root.parent
    lo, hi = settings.range_c()
    overrides, adj = settings.overrides(), settings.adjustments()
    norm = None
    if normalize:
        norm = normalize if isinstance(normalize, Normalization) else \
            normalization(root, settings, mosaic_options, jobs, progress, _found=found)
        overrides = norm.apply(overrides)
    clashes = dataset.output_clashes(found.jpegs, src_root)
    run = [src for src in found.jpegs if src not in clashes]
    arglist = [(src, out, src_root, overrides, lo, hi, settings.palette, wanted, compression, adj,
                norm.offsets_c.get(src, 0.0) if norm else 0.0, max_pitch)
               for src in run]
    results: dict[Path, core.ProcessResult] = {}
    for src, keeper in clashes.items():
        results[src] = core.ProcessResult(src, ok=False, reason=f"its outputs would overwrite those of "
                                          f"{keeper.relative_to(src_root).as_posix()} (names ignore case); rename one")
        _emit(progress, "file", src, data={"result": results[src]})
    for i, ok, payload in _run_all(core.process_file, arglist, jobs):
        result = payload if ok else core.ProcessResult(run[i], ok=False, reason=payload)
        results[run[i]] = result
        _emit(progress, "file", run[i], data={"result": result})
    return ProcessReport([results[src] for src in found.jpegs], found, norm)


@dataclasses.dataclass
class MosaicReport:
    mosaic: _mosaic.Mosaic                      # temperature (Celsius), source, view angle, report
    files: list[tuple[Path, Path]]              # (GeoTIFF, JSON sidecar) per unit
    preview: Path
    palette: str                                # the preview's palette ("grayscale" without one)
    preview_range_c: tuple[float, float]
    frames: int
    oblique: int
    unusable: int
    not_thermal: int
    normalize_ambient_c: tuple[float, float] | None   # (ambient, reflected) when normalize set them
    heights: Path | None = None                 # mosaic_heights.tif (None on the plane)
    as_converted: MosaicReport | None = None    # the mosaic written to as_converted_dir


def make_mosaic(path: str | os.PathLike, output_dir: str | os.PathLike, settings: Settings = Settings(),
           options: MosaicOptions = MosaicOptions(), compression: str = "deflate",
           jobs: int | None = None, progress: ProgressFn | None = None,
           as_converted_dir: str | os.PathLike | None = None) -> MosaicReport:
    """Stitch a dataset's nadir frames into one georeferenced mosaic in output_dir:
    mosaic_temp_<unit>.tif (+ .json) per unit, mosaic_preview.png and, when the frames were
    projected onto estimated heights, mosaic_heights.tif.

    as_converted_dir (with options.normalize): also write the mosaic without the normalization
    there, from the same layout (alignment and misalignment check), in about the time of one
    mosaic instead of two; the frames, composite and write stages then count both."""
    if as_converted_dir is not None and not options.normalize:
        raise SettingsError("as_converted_dir needs options.normalize")
    root = _existing(path)
    found = _scan(root, progress)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    src_root = root if root.is_dir() else root.parent
    adj = settings.adjustments()
    report = _stages(progress, _build_stages(settings, options, options.normalize, check=True) + ["write"])
    parts = 1 if as_converted_dir is None else 2
    frames, palettes, counts, overrides, normalized = _mosaic_frames(
        found, src_root, settings, options, adj, options.normalize, jobs, progress, _share(report, "frames", 0, parts))
    if not frames:
        raise MosaicError("No frames to mosaic.", counts)
    if as_converted_dir is not None:
        converted, _, converted_counts, _, _ = _mosaic_frames(
            found, src_root, settings, options, adj, False, jobs, progress, _share(report, "frames", 1, parts))
    try:
        if as_converted_dir is None:
            m = _mosaic.build(frames, gsd=options.gsd, align=options.align, normalize=options.normalize,
                              progress=report, jobs=worker_count(jobs, len(frames)), **_height_options(options))
        else:
            m, mc = _mosaic.build_both(frames, converted, gsd=options.gsd, align=options.align, progress=report,
                                       jobs=worker_count(jobs, len(frames)), **_height_options(options))
    except ValueError as e:
        raise MosaicError(f"Mosaic failed: {e}", counts) from None
    result = _write_mosaic(m, out, settings, options, compression, adj, palettes, _share(report, "write", 0, parts),
                           normalize_ambient_c=normalized, **counts)
    if as_converted_dir is not None:
        result.as_converted = _write_mosaic(mc, Path(as_converted_dir), settings, options, compression, adj, palettes,
                                            _share(report, "write", 1, parts), normalize_ambient_c=None,
                                            **converted_counts)
    return result


def _height_options(options: MosaicOptions) -> dict:
    return {"heights": options.heights, "height_cell": options.height_cell}


def _share(report, stage: str, part: int, parts: int):
    """report with stage counted as part (0-based) of parts equal runs of it."""
    if not report or parts == 1:
        return report

    def shared(s: str, done: int, total: int, unit: str, **extra) -> None:
        if s == stage:
            done, total = done + part * total, parts * total
        report(s, done, total, unit, **extra)

    return shared


def _write_mosaic(m: _mosaic.Mosaic, out: Path, settings: Settings, options: MosaicOptions, compression: str,
                  adj: adjust.Adjustments, palettes, report, **fields) -> MosaicReport:
    """Write a built mosaic's GeoTIFFs and preview in out (the "write" stage)."""
    out.mkdir(parents=True, exist_ok=True)
    m.report["adjustments"] = adjust.describe(adj)
    writes = len(options.units) + 1 + (m.heights is not None)   # each unit's GeoTIFF (+ sidecar), the heights, the preview
    if report:
        report("write", 0, writes, "files")
    files = []
    for u in options.units:
        files.append(_mosaic.write(m, out / f"mosaic_temp_{u.lower()}.tif", compression, u,
                                   located=not adj.strip_location))
        if report:
            report("write", len(files), writes, "files")
    lut, palette_name = None, "grayscale"
    if palettes:
        index = settings.palette if settings.palette is not None else Counter(p for p, _ in palettes).most_common(1)[0][0]
        luts = palettes[0][1]
        if 0 <= index < luts.shape[1]:
            lut, palette_name = luts[:, index, :].T, PALETTES[index]
    lo, hi = settings.range_c()
    heights = _mosaic.write_heights(m, out / "mosaic_heights.tif", compression, located=not adj.strip_location)
    if heights and report:
        report("write", len(files) + 1, writes, "files")
    preview, low, high = _mosaic.write_preview(m, out / "mosaic_preview.png", lut, lo, hi)
    if report:
        report("write", writes, writes, "files")
    return MosaicReport(m, files, preview, palette_name, (low, high), heights=heights, **fields)


@dataclasses.dataclass
class Normalization:
    """What makes a dataset's frames consistent with each other: one ambient and reflected
    temperature for every frame (the flight medians of the embedded values, unless Settings
    give them), and each frame's offset against its overlapping neighbours (camera drift
    between shutter recalibrations), found from the frames' overlaps as the mosaic finds them.
    Frames the mosaic can't place (oblique, no pose) get no offset."""

    ambient_c: float | None
    reflected_c: float | None
    offsets_c: dict[Path, float]                # per frame, to subtract
    frames: int                                 # frames placed and compared
    without_offset: int                         # thermal frames that got no offset
    report: dict                                # the mosaic's frame_offsets report

    def apply(self, overrides: core.MeasurementOverrides) -> core.MeasurementOverrides:
        """overrides with the normalization's ambient and reflected temperature filled in."""
        return dataclasses.replace(
            overrides,
            ambient_temp=self.ambient_c if overrides.ambient_temp is None else overrides.ambient_temp,
            reflection=self.reflected_c if overrides.reflection is None else overrides.reflection)


def normalization(path: str | os.PathLike, settings: Settings = Settings(), options: MosaicOptions = MosaicOptions(),
                  jobs: int | None = None, progress: ProgressFn | None = None, *, _found=None) -> Normalization:
    """The per-frame normalization of a dataset (see Normalization), for process(normalize=...)
    or make_mosaic(MosaicOptions(normalize=True)). Uses options' max_pitch and align; the
    replacement settings are not applied while comparing frames."""
    root = _existing(path)
    found = _found or _scan(root, progress)
    src_root = root if root.is_dir() else root.parent
    adj = dataclasses.replace(settings.adjustments(), replace_above=None, replace_above_value=None,
                              replace_below=None, replace_below_value=None)
    report = _stages(progress, _build_stages(settings, options, True, check=False))
    frames, _, counts, overrides, _ = _mosaic_frames(found, src_root, settings, options, adj, True, jobs, progress,
                                                     report)
    if not frames:
        raise MosaicError("No frames to compare for normalization.", counts)
    try:
        m = _mosaic.build(frames, gsd=options.gsd, align=options.align, normalize=False, check_misalignment=False,
                          progress=report, jobs=worker_count(jobs, len(frames)), **_height_options(options))
    except ValueError as e:
        raise MosaicError(f"Normalization failed: {e}", counts) from None
    offsets = {src_root / row["file"]: float(row["temperature_offset_c"]) for row in m.frames}
    thermal = counts["frames"] + counts["oblique"] + counts["unusable"]
    return Normalization(overrides.ambient_temp, overrides.reflection, offsets, len(offsets),
                         thermal - len(offsets), m.report["frame_offsets"])


def _build_stages(settings: Settings, options: MosaicOptions, median_ambient: bool, check: bool) -> list[str]:
    """The stages of a mosaic run up to its compositing (and check), as Event describes them."""
    return ((["ambient"] if _reads_ambient(settings, median_ambient) else []) + ["frames"]
            + (["align"] if options.align else []) + ["composite"] + (["check"] if check else []))


def _reads_ambient(settings: Settings, median_ambient: bool) -> bool:
    overrides = settings.overrides()
    return median_ambient and (overrides.ambient_temp is None or overrides.reflection is None)


def _mosaic_frames(found: dataset.Scan, src_root: Path, settings: Settings, options: MosaicOptions,
                   adj: adjust.Adjustments, median_ambient: bool, jobs: int | None, progress: ProgressFn | None,
                   report=None):
    """Every placeable frame measured for a mosaic: (frames, palettes, counts, overrides,
    (ambient, reflected) when median_ambient filled them in). report: see _stages."""
    overrides = settings.overrides()
    normalized = None
    total = len(found.jpegs)
    if _reads_ambient(settings, median_ambient):
        if report:
            report("ambient", 0, total, "files")
        embedded = []
        for done, (_, ok, p) in enumerate(_run_all(core.embedded_ambient, [(s,) for s in found.jpegs], jobs), 1):
            if ok and p:
                embedded.append(p)
            if report:
                report("ambient", done, total, "files")
        if embedded:
            amb, refl = (float(v) for v in np.median(np.array(embedded), axis=0))
            overrides = dataclasses.replace(
                overrides, ambient_temp=amb if overrides.ambient_temp is None else overrides.ambient_temp,
                reflection=refl if overrides.reflection is None else overrides.reflection)
            normalized = (overrides.ambient_temp, overrides.reflection)
            _emit(progress, "normalize", data={"ambient_c": normalized[0], "reflected_c": normalized[1]})
    frames, palettes, unusable, oblique, other = [], [], 0, 0, 0
    arglist = [(src, src_root, overrides, options.max_pitch, adj) for src in found.jpegs]
    if report:
        report("frames", 0, total, "files")
    for done, (i, ok, payload) in enumerate(_run_all(core.mosaic_frame, arglist, jobs), 1):
        src = found.jpegs[i]
        if ok and "frame" in payload:
            frames.append(payload["frame"])
            if payload["palette"]:
                palettes.append(payload["palette"])
            for note in payload["notes"]:
                _emit(progress, "note", src, note)
        elif ok and payload.get("oblique"):
            oblique += 1
        elif ok and payload.get("not_thermal"):
            other += 1
        else:
            unusable += 1
            _emit(progress, "mosaic_skip", src, payload["reason"] if ok else payload)
        if report:
            report("frames", done, total, "files")
    frames.sort(key=lambda f: f.name)
    counts = {"frames": len(frames), "oblique": oblique, "unusable": unusable, "not_thermal": other}
    return frames, palettes, counts, overrides, normalized


# --- helpers --------------------------------------------------------------------------------

def _existing(path: str | os.PathLike) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    return p


def _unit(unit: str) -> str:
    u = str(unit).upper()
    if u not in units.UNITS:
        raise SettingsError(f"unit must be C, F or K, not {unit!r}")
    return u


def _scan(path, progress: ProgressFn | None) -> dataset.Scan:
    found = dataset.scan(_existing(path))
    for p, why in found.unreadable:
        _emit(progress, "unreadable", p, why)
    if found.others:
        _emit(progress, "non_jpeg", data={"count": len(found.others), "paths": found.others})
    return found


def worker_count(requested: int | None, n_jobs: int) -> int:
    """Workers for n_jobs items (processes for files, threads for the mosaic's matching):
    requested, else min(CPU count, 8); at least 1."""
    n = requested if requested else min(os.cpu_count() or 1, DEFAULT_MAX_WORKERS)
    return max(1, min(n, n_jobs))


def _run_all(fn, arglist: list[tuple], jobs: int | None):
    """Yield (index, ok, result or error text) for fn(*args) over arglist, as each finishes,
    in jobs worker processes (1: in this process). An unexpected exception only fails its
    own item."""
    workers = worker_count(jobs, len(arglist))
    if workers <= 1:
        for i, a in enumerate(arglist):
            try:
                yield i, True, fn(*a)
            except Exception as e:  # noqa: BLE001 -- report and go on with the dataset
                yield i, False, f"{type(e).__name__}: {e}"
        return
    pool = ProcessPoolExecutor(workers)
    try:
        futures = {pool.submit(fn, *a): i for i, a in enumerate(arglist)}
        for f in as_completed(futures):
            e = f.exception()
            yield (futures[f], False, f"{type(e).__name__}: {e}") if e else (futures[f], True, f.result())
    finally:                                    # interrupted: don't start what is still queued
        pool.shutdown(wait=True, cancel_futures=True)
