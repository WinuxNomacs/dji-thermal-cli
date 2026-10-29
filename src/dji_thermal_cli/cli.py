"""
dji-thermal-cli

Recursively walks a directory (or reads a single file) for DJI R-JPEG
thermal images.

  `process` writes derivative files for each R-JPEG, chosen with
  --output PRODUCT[:FORMAT] (repeatable or comma-separated; default
  temp_c:tiff,pseudo:png), named <stem>_<product>.<format>:
{products}
  Any of --distance/--humidity/--emissivity/--reflected-temp/--ambient-temp
  you omit is left as whatever value is already embedded in that R-JPEG
  -- you only need to pass the ones you want to change. This does NOT
  rewrite the original R-JPEG: DJI bakes its environmental parameters into
  a binary APP4 block at capture time that is not writable by the SDK.

  Object distance (--distance): "lrf" (default) uses a file's laser-
  rangefinder reading where it has a valid one (DJI XMP LRFStatus=Normal and
  LRFTargetDistance, e.g. the M4T), clamped to the range the camera accepts,
  else its embedded distance; "embedded" always keeps the embedded distance;
  a number sets that distance in metres for every file.

  `list` prints each R-JPEG's embedded measurement parameters (and the
  valid range for each, resolution, R-JPEG version, and current color-bar
  /palette settings) without writing anything.

Units: humidity is a percent, e.g. 50 for 50%. The SDK works in Celsius;
temperatures you give (--input-unit) and write (temp_c/temp_f/temp_k,
--mosaic-unit) may be Celsius, Fahrenheit or Kelvin, converted in double
precision and rounded once.

Everything is computed by a pure-Python port of the DJI Thermal SDK v1.8; its
temperatures and pseudo-color images are identical to the SDK's. It covers the
H20N, H20T, H30T, M2EA, M30T, M3T, M3TD, M4T and XT S. Files it can't read, and
the few color settings the SDK itself doesn't handle (it reads uninitialised
memory there), are reported with the reason.

Files are processed in parallel by --jobs worker processes (--jobs 1: one at a
time, in this process), and the mosaic matches its overlaps in as many threads. A file that can't be converted is reported and skipped;
the rest of the batch goes on.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import re
import sys
from pathlib import Path

from . import api, core, listing, mosaic, outputs, units
from ._version import __version__


def _units(text: str) -> list[str]:
    chosen = [u.strip().upper() for u in text.split(",") if u.strip()]
    bad = [u for u in chosen if u not in units.UNITS]
    if bad or not chosen:
        raise argparse.ArgumentTypeError(f"must be C, F or K (comma-separated), not {text!r}")
    return chosen


def _replacement(text: str) -> str | float:
    if text.strip().lower() == "nodata":
        return "nodata"
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a temperature or 'nodata'") from None


def _positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def _distance(text: str) -> str | float:
    if text in ("lrf", "embedded"):
        return text
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be lrf, embedded or a distance in metres") from None


PALETTE_LOOKS = (
    "grayscale, hot = white",
    "black, red, yellow, white",
    "DJI's ironbow: dark blue, purple, orange, yellow-white",
    "black, green, red",
    "multi-colour, for small temperature differences",
    "blues for cold, gold for hot",
    "full spectrum",
    "full spectrum, a different ramp",
    "grayscale with hot areas tinted red",
    "grayscale, hot = black",
)


def _palette(text: str) -> int:
    """A palette by index (0-9) or name, e.g. 3 or hotiron."""
    names = core.PSEUDO_COLOR_NAMES
    key = text.strip().lower()
    if key.isdigit() and int(key) < len(names):
        return int(key)
    if key in names:
        return names.index(key)
    raise argparse.ArgumentTypeError(f"must be 0-{len(names) - 1} or one of {', '.join(names)}")


def _palettes_help() -> str:
    return "\n".join(f"    {i}  {name:<10} {look}" for i, (name, look) in enumerate(zip(core.PSEUDO_COLOR_NAMES, PALETTE_LOOKS)))


def _outputs(text: str) -> list[tuple[str, str]]:
    try:
        return [outputs.parse_output(spec) for spec in text.split(",") if spec.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _products_help() -> str:
    lines = []
    for name, product in outputs.PRODUCTS.items():
        formats = "|".join(f + ("/tif" if f == "tiff" else "") for f in product.formats)
        lines.append(f"    {name:<8} {formats:<14} {product.description}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dji-thermal-cli",
        description=__doc__.format(products=_products_help()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    proc = sub.add_parser(
        "process", help="Write temperature / pseudo-color derivative files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Write derivative files for each R-JPEG (the source files are never modified).\n\n"
                    "Outputs (--output PRODUCT[:FORMAT], first format is the default), named <stem>_<product>.<format>:\n"
                    + _products_help()
                    + "\n\nPalettes (--palette, by number or name), the DJI SDK's own colour tables:\n"
                    + _palettes_help(),
    )
    proc.add_argument("--input-dir", required=True, type=Path, help="Directory (searched recursively) or single R-JPEG file")
    proc.add_argument("--output-dir", required=True, type=Path)
    proc.add_argument("--distance", type=_distance, default="lrf", metavar="{lrf,embedded,METRES}", help="Object distance: 'lrf' = each file's laser-rangefinder distance where valid, else embedded (default); 'embedded' = each file's embedded value; or a distance in metres for every file")
    proc.add_argument("--humidity", type=float, default=None, help="Relative humidity as a percent, e.g. 50 for 50%% (default: keep embedded)")
    proc.add_argument("--emissivity", type=float, default=None, help="Emissivity, 0.0-1.0; the accepted range depends on the camera, see `list --format json` (default: keep embedded)")
    proc.add_argument("--reflected-temp", type=float, default=None, dest="reflected_temp", help="Reflected apparent temperature (default: keep embedded)")
    proc.add_argument("--ambient-temp", type=float, default=None, dest="ambient_temp", help="Ambient temperature (default: keep embedded)")
    proc.add_argument("--input-unit", type=str.upper, choices=list(units.UNITS), default="C", help="Unit of --reflected-temp/--ambient-temp/--range-min/--range-max: C, F or K (default: C). They are converted to Celsius for the SDK")
    proc.add_argument("--range-min", type=float, default=None, help="Fixed color-bar minimum temperature (enables manual color bar)")
    proc.add_argument("--range-max", type=float, default=None, help="Fixed color-bar maximum temperature (enables manual color bar)")
    proc.add_argument("--gimbal-yaw", type=float, default=None, metavar="DEG", help="Use this gimbal yaw for every file instead of the recorded one (mosaic placement, and written into the outputs' XMP; the original is kept as dtc:OriginalGimbalYawDegree)")
    proc.add_argument("--gimbal-pitch", type=float, default=None, metavar="DEG", help="Use this gimbal pitch for every file (e.g. -90 for straight down); as --gimbal-yaw. The mosaic's --mosaic-max-pitch test uses it too")
    proc.add_argument("--gimbal-roll", type=float, default=None, metavar="DEG", help="Use this gimbal roll for every file; as --gimbal-yaw")
    proc.add_argument("--replace-above", type=float, default=None, metavar="TEMP", help="Pixels measuring above TEMP (--input-unit) get --replace-above-value instead, in the temperature outputs (temp_c/f/k/ck, gray16) and the mosaic. temp_dc, raw and pseudo stay the SDK's own")
    proc.add_argument("--replace-above-value", type=_replacement, default=None, metavar="TEMP|nodata", help="The value for pixels above --replace-above (--input-unit), or 'nodata'")
    proc.add_argument("--replace-below", type=float, default=None, metavar="TEMP", help="Pixels measuring below TEMP get --replace-below-value instead; as --replace-above")
    proc.add_argument("--replace-below-value", type=_replacement, default=None, metavar="TEMP|nodata", help="The value for pixels below --replace-below (--input-unit), or 'nodata'")
    proc.add_argument("--strip-location", action="store_true", help="Write outputs that can be shared without revealing where, or with which aircraft, they were taken: no GPS, DJI position/altitude/RTK/rangefinder-target fields or serial numbers in any output's metadata (capture time is kept), and a mosaic without its georeference (positions relative to its corner). Recorded as dtc:LocationRemoved. ODM/WebODM can't use such frames: they need the GPS")
    proc.add_argument("--normalize", action="store_true", help="Make the per-frame outputs consistent with each other (e.g. for WebODM): every frame measured with the flight's median embedded ambient/reflected temperature (unless given), and each frame's temperature offset against its overlapping neighbours (camera drift between shutter recalibrations) removed from the temperature outputs. Offsets come from the frames' overlaps, as the mosaic finds them (--mosaic-max-pitch, --mosaic-align); frames it can't place get none. Recorded as dtc:NormalizationOffsetC")
    proc.add_argument("--max-pitch", type=float, default=None, metavar="DEG", help="Process only frames looking at least this steeply down (gimbal pitch at or below DEG, after --gimbal-pitch), e.g. -80 for nadir frames only; the others are skipped")
    proc.add_argument("--palette", type=_palette, default=None, metavar="0-9|NAME", help="Pseudo-color palette for pseudo images and the mosaic preview, by number or name (e.g. 3 or hotiron; see the palette list above). Default: keep each file's own")
    proc.add_argument("--output", "-o", action="append", type=_outputs, dest="outputs", metavar="PRODUCT[:FORMAT]", help="Output to write; repeat or comma-separate for several (default: temp_c:tiff,pseudo:png). See the product list above")
    proc.add_argument("--mosaic", action="store_true", help="Also stitch the nadir frames into one georeferenced mosaic: <output-dir>/mosaic_temp_c.tif (temperature, source frame and view angle bands; UTM), a .json sidecar with every frame's position, attitude and measurement parameters plus a seam/misalignment report, and mosaic_preview.png. Per-frame files are then written only for the --output options given")
    proc.add_argument("--mosaic-max-pitch", type=float, default=mosaic.DEFAULT_MAX_PITCH, metavar="DEG", help="Frames whose gimbal pitch is above this (less steeply down) are left out of the mosaic (default: %(default)g)")
    proc.add_argument("--mosaic-align", action=argparse.BooleanOptionalAction, default=True, help="Correct the frames' placement from their overlaps before compositing: a flight-wide camera boresight plus a small shift and rotation per frame (default: on)")
    proc.add_argument("--mosaic-normalize", action="store_true", help="Make the frames' temperatures consistent before compositing: measure every frame with the flight's median embedded ambient/reflected temperature (unless --ambient-temp/--reflected-temp are given), then remove each frame's offset against its overlapping neighbours (camera drift between shutter recalibrations). Changes the measured values; the offsets are always reported")
    proc.add_argument("--mosaic-unit", action="append", type=_units, dest="mosaic_units", metavar="UNIT[,UNIT]", help="Temperature unit(s) of the mosaic, C, F and/or K; repeat or comma-separate for one mosaic_temp_<unit>.tif each (default: C)")
    proc.add_argument("--mosaic-heights", choices=("auto", "plane"), default="auto", help="Surface the frames are projected onto: auto = heights estimated from the frames' parallax on a grid of cells, anchored to the rangefinder, written to mosaic_heights.tif (the plane when too little overlaps); plane = a horizontal ground plane (default: %(default)s)")
    proc.add_argument("--mosaic-dsm", type=Path, metavar="PATH", help="Project the frames onto this surface model instead of estimating heights: a GeoTIFF in WGS84 UTM or lat/lon, altitudes in metres (e.g. ODM's dsm.tif)")
    proc.add_argument("--mosaic-dsm-offset", type=float, metavar="METRES", help="Metres added to the DSM's altitudes to match the frames' (default: estimated from the rangefinder)")
    proc.add_argument("--mosaic-height-cell", type=int, default=mosaic.HEIGHT_CELL_PX, metavar="PIXELS", help="Size of the height cells, in mosaic pixels (default: %(default)d)")
    proc.add_argument("--mosaic-gsd", type=float, default=None, metavar="METRES", help="Mosaic pixel size (default: the frames' median ground sample distance)")
    proc.add_argument("--mosaic-visual", action="store_true", help="Also mosaic the visual (RGB) frames the drone took with the thermal ones (M4T, M30T, M3T, H20T, H30T wide camera; zoom frames are left out): <output-dir>/mosaic_visual.tif (red, green, blue, alpha; UTM) with a .json sidecar and mosaic_visual_preview.png. Each visual frame is placed as its thermal frame is in the thermal mosaic, the two cameras' offset measured and corrected, so the two mosaics line up")
    proc.add_argument("--mosaic-visual-gsd", type=float, default=None, metavar="METRES", help="The visual mosaic's pixel size (default: the visual frames' median ground sample distance)")
    proc.add_argument("--rjpeg-picture", choices=api.RJPEG_PICTURES, default="palette", help="The rjpeg output's visible picture: palette = coloured as the pseudo output (--palette, --range-min/--range-max), the default; original = DJI's own. DJI's tools colour it from the measurements either way")
    proc.add_argument("--compression", choices=outputs.COMPRESSIONS, default="deflate", help="TIFF/EXR compression: deflate = lossless Deflate (EXR: ZIP), the default; none = uncompressed. PNG is always compressed")
    proc.add_argument("--jobs", "-j", type=_positive_int, default=None, help="Worker processes, and the mosaic's matching threads (default: min(CPU count, 8); 1 = in this process)")
    proc.add_argument("--quiet", "-q", action="store_true", help="Don't show the mosaic's progress (--mosaic, --normalize) on stderr; results, skipped files and errors are still printed")

    lst = sub.add_parser("list", help="Describe every thermal R-JPEG: thermal parameters, temperature stats, position, attitude, camera, EXIF",
                         description="Describe every thermal R-JPEG in a dataset without writing anything. Groups: file, camera, thermal (measurement parameters and ranges, palette, colour bar, ND filter), temperature (stats), geospatial (position, RTK, rangefinder), attitude (gimbal, aircraft), and with --exif all every raw EXIF/XMP tag. Files are found by content, not extension.")
    lst.add_argument("--input-dir", required=True, type=Path, help="Directory (searched recursively) or single R-JPEG file")
    lst.add_argument("--format", choices=["table", "json", "csv"], default="table", help="table: a summary per file; json: every group nested; csv: every group, one row per file with dotted column names (default: table)")
    lst.add_argument("--stats", action=argparse.BooleanOptionalAction, default=True, help="Measure each image for its min/max/mean/median/std temperature with the file's own parameters (default: on; --no-stats is faster)")
    lst.add_argument("--exif", choices=["curated", "all"], default="curated", help="curated: the grouped fields (default); all: also every EXIF tag and XMP property, raw (json/csv)")
    lst.add_argument("--unit", type=str.upper, choices=list(units.UNITS), default="C", help="Unit to display the reflected/ambient temperatures and color bar in: C, F or K (default: C)")
    lst.add_argument("--jobs", "-j", type=_positive_int, default=None, help="Worker processes (default: min(CPU count, 8); 1 = in this process)")

    chk = sub.add_parser("check", help="Find a dataset's errors and anomalies, with the options that address each",
                         formatter_class=argparse.RawDescriptionHelpFormatter,
                         description="Check a dataset before processing it, without writing anything.\n\n"
                                     "errors:   files that can't be used (named like images but not JPEGs, unreadable, damaged,\n"
                                     "          unsupported formats) and duplicate files\n"
                                     "warnings: parameters that differ between files, outlying temperatures, gimbal pitch,\n"
                                     "          altitude or rangefinder readings, missing pose, mixed cameras, RTK quality\n"
                                     "info:     ordinary JPEGs mixed in, oblique frames a mosaic leaves out\n\n"
                                     "Files are identified by content and fully decoded. Outliers use the dataset's median and\n"
                                     "median absolute deviation. Exit status: 1 when there are errors (or warnings, with --strict).")
    chk.add_argument("--input-dir", required=True, type=Path, help="Directory (searched recursively) or single file")
    chk.add_argument("--format", choices=["table", "json"], default="table")
    chk.add_argument("--unit", type=str.upper, choices=list(units.UNITS), default="C", help="Unit for the temperatures reported (default: C)")
    chk.add_argument("--mosaic-max-pitch", type=float, default=mosaic.DEFAULT_MAX_PITCH, metavar="DEG", help="The mosaic's pitch limit, to report which frames it would leave out (default: %(default)g)")
    chk.add_argument("--webodm", action="store_true", help="Also plan a WebODM / ODM task: which frames to use, coverage (frames per ground point, equivalent overlap), the `process` command that prepares the images, and ODM task options with the reason for each")
    chk.add_argument("--strict", action="store_true", help="Exit with status 1 on warnings too")
    chk.add_argument("--all-files", action="store_true", help="List every file of each finding (the table shows the first five)")
    chk.add_argument("--jobs", "-j", type=_positive_int, default=None, help="Worker processes (default: min(CPU count, 8); 1 = in this process)")

    return p


# Settings fields -> the options that set them, for error messages
_FLAGS = {"replace_above_value": "--replace-above-value", "replace_below_value": "--replace-below-value",
          "replace_above": "--replace-above", "replace_below": "--replace-below", "reflected_temp": "--reflected-temp",
          "ambient_temp": "--ambient-temp", "range_min": "--range-min", "range_max": "--range-max",
          "unit": "--input-unit", "units": "--mosaic-unit", "gsd": "--mosaic-gsd", "palette": "--palette",
          "heights": "--mosaic-heights", "height_cell": "--mosaic-height-cell",
          "dsm": "--mosaic-dsm", "dsm_offset": "--mosaic-dsm-offset",
          "distance": "--distance"}


def _fail(e: Exception) -> None:
    """Exit with the error, naming command-line options rather than API fields."""
    message = str(e)
    for field, flag in _FLAGS.items():
        message = re.sub(rf"(?<![\w-]){field}(?!\w)", flag, message)
    sys.exit(message)


def _settings(args: argparse.Namespace) -> api.Settings:
    try:
        return api.Settings(
            distance=args.distance, humidity=args.humidity, emissivity=args.emissivity,
            reflected_temp=args.reflected_temp, ambient_temp=args.ambient_temp, unit=args.input_unit,
            range_min=args.range_min, range_max=args.range_max, palette=args.palette,
            gimbal_yaw=args.gimbal_yaw, gimbal_pitch=args.gimbal_pitch, gimbal_roll=args.gimbal_roll,
            replace_above=args.replace_above, replace_above_value=args.replace_above_value,
            replace_below=args.replace_below, replace_below_value=args.replace_below_value,
            strip_location=getattr(args, "strip_location", False))
    except api.SettingsError as e:
        _fail(e)


def _printer(out=None, mosaic_only: bool = False):
    """A progress callback printing each file's outcome as the CLI always has."""
    out = out or sys.stdout

    def show(event: api.Event) -> None:
        if event.kind == "unreadable":
            print(f"[skip, unreadable: {event.detail}] {event.path}", file=out, flush=True)
        elif event.kind == "non_jpeg":
            print(f"Ignored {event.data['count']} non-JPEG file(s) (identified by content, not extension).",
                  file=out, flush=True)
        elif event.kind == "file" and "result" in event.data:
            r = event.data["result"]
            if r.ok:
                print(f"[ok] {event.path}", file=out, flush=True)
                for note in r.notes:
                    print(f"     note: {note}", file=out, flush=True)
            else:
                print(f"[skip, {r.reason}] {event.path}", file=out, flush=True)
        elif event.kind == "normalize":
            print(f"Normalize: every frame measured with ambient {event.data['ambient_c']:.2f} C, "
                  f"reflected {event.data['reflected_c']:.2f} C (flight medians unless given)", file=out, flush=True)
        elif event.kind == "note":
            print(f"     note: {event.detail} ({event.path.name})", file=out, flush=True)
        elif event.kind == "mosaic_skip":
            print(f"[mosaic skip, {event.detail}] {event.path}", file=out, flush=True)

    return show


class _MosaicProgress:
    """The mosaic's progress on stderr: on a terminal one line, rewritten as the work goes
    ("Mosaic step 3 of 6: aligning (round 2/3), 812/1840 pairs"); otherwise (a log, a pipe)
    one line as each stage starts. Other events go to show, clearing the line first."""

    STAGES = {"ambient": "reading ambient temperatures", "frames": "reading frames", "align": "aligning",
              "composite": "compositing", "check": "checking alignment", "visual_frames": "reading visual frames",
              "visual_offset": "measuring the visual camera's offset", "visual_composite": "compositing visual frames",
              "write": "writing"}

    def __init__(self, show, out=None, quiet: bool = False):
        self.show, self.out, self.quiet = show, out or sys.stderr, quiet
        self.tty = self.out.isatty()
        self.width = 0
        self.started = None

    def __call__(self, event: api.Event) -> None:
        if event.kind != "mosaic":
            self.clear()
            self.show(event)
            return
        if self.quiet:
            return
        d = event.data
        head = f"Mosaic step {d['step']} of {d['steps']}: {self.STAGES.get(d['stage'], d['stage'])}"
        if "round" in d:
            head += f" (round {d['round']}/{d['rounds']})"
        if self.tty:
            line = f"{head}, {d['done']}/{d['total']} {d['unit']}"
            print("\r" + line.ljust(self.width), end="", file=self.out, flush=True)
            self.width = len(line)
        elif (d["stage"], d.get("round"), d["unit"]) != self.started:
            self.started = (d["stage"], d.get("round"), d["unit"])
            print(f"{head}, {d['total']} {d['unit']}", file=self.out, flush=True)

    def clear(self) -> None:
        """Remove the terminal's progress line, before anything else is printed."""
        if self.width:
            print("\r" + " " * self.width + "\r", end="", file=self.out, flush=True)
            self.width = 0


def _cmd_process(args: argparse.Namespace) -> None:
    settings = _settings(args)
    if args.outputs:
        wanted = tuple(dict.fromkeys(o for group in args.outputs for o in group))
    else:
        wanted = () if args.mosaic else outputs.DEFAULT_OUTPUTS
    try:
        options = api.MosaicOptions(max_pitch=args.mosaic_max_pitch, align=args.mosaic_align,
                                    normalize=args.mosaic_normalize, gsd=args.mosaic_gsd,
                                    heights=args.mosaic_heights, height_cell=args.mosaic_height_cell,
                                    dsm=args.mosaic_dsm, dsm_offset=args.mosaic_dsm_offset,
                                    visual=args.mosaic_visual, visual_gsd=args.mosaic_visual_gsd,
                                    units=tuple(u for group in (args.mosaic_units or [["C"]]) for u in group))
    except api.SettingsError as e:
        _fail(e)
    show = _MosaicProgress(_printer(), quiet=args.quiet)
    report = None
    try:
        if wanted:
            try:
                report = api.process(args.input_dir, args.output_dir, settings, wanted, args.compression, args.jobs,
                                     show, normalize=args.normalize, max_pitch=args.max_pitch, mosaic_options=options,
                                     rjpeg_picture=args.rjpeg_picture)
            finally:
                show.clear()
            print(f"\nDone. {report.processed}/{len(report.results)} candidate files processed.")
            if report.normalization:
                n = report.normalization
                values = list(n.offsets_c.values())
                print(f"Normalized: {n.frames} frame(s) with overlap offsets "
                      f"({min(values):+.2f} to {max(values):+.2f} C removed), {n.without_offset} without "
                      f"(oblique, unplaceable or not overlapping)" if values else "Normalized: no frame offsets found")
        if args.mosaic:
            # the scan was reported by process() already
            unreported = show if not wanted else (lambda e: e.kind in ("unreadable", "non_jpeg") or show(e))
            try:
                rep = api.make_mosaic(args.input_dir, args.output_dir, settings, options, args.compression,
                                      args.jobs, unreported)
            finally:
                show.clear()
            _print_mosaic(rep, settings, args.mosaic_max_pitch)
    except (FileNotFoundError, api.MosaicError) as e:
        if isinstance(e, api.MosaicError) and e.counts:
            _print_mosaic_counts(e.counts, args.mosaic_max_pitch)
        _fail(e)
    if not wanted:
        _location_notice(settings)
        return
    if report.processed:
        labels = {"option": "--distance", "rangefinder": "laser rangefinder", "embedded": "embedded"}
        used = ", ".join(f"{n} {labels[k]}" for k, n in report.distance_sources().items() if n)
        print(f"Object distance source: {used}")
    if settings.range_min is not None:
        lo, hi = settings.range_c()
        u = settings.unit
        print(f"Applied fixed color-bar range: {settings.range_min:g} to {settings.range_max:g} {u}"
              + (f" ({lo:.2f} C to {hi:.2f} C for the SDK)" if u != "C" else ""))
    else:
        print("No fixed color-bar range set (--range-min/--range-max omitted); pseudo-color used auto range per image.")
    _location_notice(settings)


def _location_notice(settings: api.Settings) -> None:
    if settings.strip_location:
        print("Location removed (--strip-location): no GPS, position or serial numbers in any output. "
              "ODM/WebODM can't use these frames; process without it for them.")


def _print_mosaic_counts(counts: dict, max_pitch: float, visual: bool = False) -> None:
    others = "for the visual mosaic" if visual else "ignored"
    print(f"\nMosaic: {counts['frames']} nadir frames, {counts['oblique']} oblique left out (pitch above {max_pitch:g} deg), "
          f"{counts['unusable']} unusable"
          + (f", {counts['not_thermal']} other JPEGs (not thermal R-JPEGs) {others}." if counts["not_thermal"] else "."))


def _print_mosaic(rep: api.MosaicReport, settings: api.Settings, max_pitch: float) -> None:
    m = rep.mosaic
    g, r = m.grid, m.report
    _print_mosaic_counts({"frames": rep.frames, "oblique": rep.oblique, "unusable": rep.unusable,
                          "not_thermal": rep.not_thermal}, max_pitch, visual=bool(rep.visual or rep.visual_note))
    crs = "" if settings.strip_location else f", EPSG:{g.epsg}"       # the UTM zone tells roughly where
    for tif, sidecar in rep.files:
        print(f"  {tif}  ({g.width}x{g.height} px, {g.gsd * 100:.1f} cm/px{crs})\n  {sidecar}")
    span = "--range-min/--range-max" if settings.range_min is not None else "2nd-98th percentile"
    low, high = rep.preview_range_c
    print(f"  {rep.preview}  ({rep.palette}, {low:.1f} to {high:.1f} C, {span})")
    if rep.heights:
        hr = r["heights"]
        low_h, high_h = hr["range_m"]
        print(f"  {rep.heights}  ({hr['cell_m'] * 100:.0f} cm cells, {low_h:+.1f} to {high_h:+.1f} m above the ground plane)")
    ground = f"from the {m.ground_alt_source}" if settings.strip_location else f"{m.ground_alt:.2f} m ({m.ground_alt_source})"
    print(f"  ground plane {ground}; {r['frames_contributing']}/{r['frames_used']} frames contribute")
    seams, mis, ali = r["radiometric_seams"], r["misalignment"], r["alignment"]
    if ali:
        b, before = ali["boresight_deg"], ali["before"]
        print(f"  alignment: boresight pitch {b['pitch']:+.2f} deg, yaw {b['yaw']:+.2f} deg; per-frame shift "
              f"{ali['frame_shift_m']['median'] * 100:.0f} cm median, rotation {ali['frame_rotation_deg']['median']:.1f} deg median"
              + (f"; offsets before: {before['median_m'] * 100:.0f} cm median" if before.get("pairs") else ""))
    fo = r["frame_offsets"]
    if fo.get("offset_c"):
        diff = fo["overlap_difference_c"]
        print(f"  frame temperature offsets: {fo['offset_c']['min']:+.2f} to {fo['offset_c']['max']:+.2f} C; overlapping frames differ "
              f"by {diff['median_abs']:.2f} C median, {diff['median_abs_after_offsets']:.2f} C with the offsets removed"
              + (" (removed: --mosaic-normalize)" if fo["normalized"] else " (not removed; see --mosaic-normalize)"))
    if seams.get("pixels"):
        print(f"  seams: best vs second-best frame differ by {seams['median_abs_c']:.2f} C median, {seams['p95_abs_c']:.2f} C p95")
    if mis and mis.get("pairs"):
        print(f"  misalignment between overlapping frames: {mis['median_m'] * 100:.0f} cm median, "
              f"{mis['p90_m'] * 100:.0f} cm p90 ({mis['pairs']} pairs); objects above the ground plane add parallax")
    if rep.visual_note:
        print(f"\nVisual mosaic: none, {rep.visual_note}.")
    elif rep.visual:
        _print_visual(rep.visual, settings)


def _print_visual(v: api.VisualReport, settings: api.Settings) -> None:
    m = v.mosaic
    g, r = m.grid, m.report
    print(f"\nVisual mosaic: {r['frames_used']} visual frames paired with the thermal mosaic's"
          + (f", {len(r['thermal_frames_without_visual'])} thermal frames without one" if r["thermal_frames_without_visual"] else "")
          + (f", {len(r['visual_frames_unused'])} visual frames unused (oblique or unpaired)" if r["visual_frames_unused"] else "")
          + (f", {v.skipped} other JPEGs ignored" if v.skipped else "") + ".")
    crs = "" if settings.strip_location else f", EPSG:{g.epsg}"
    print(f"  {v.file}  ({g.width}x{g.height} px, {g.gsd * 100:.1f} cm/px{crs})\n  {v.sidecar}")
    print(f"  {v.preview}" + (f"  (1/{v.preview_reduction} size)" if v.preview_reduction > 1 else ""))
    off = r["camera_offset"]
    if off.get("applied"):
        b = off["boresight_deg"]
        print(f"  cameras' offset: pitch {b['pitch']:+.2f} deg, yaw {b['yaw']:+.2f} deg from {off['pairs_used']} pairs; "
              f"visual vs thermal {off['offset_before_m']['median'] * 100:.0f} cm median before, "
              f"{off['residual_m']['median'] * 100:.0f} cm after")
    else:
        print(f"  cameras' offset: not corrected, {off['reason']}")


TABLE_COLUMNS = (   # (header, flattened key, format); {u} is the temperature unit
    ("file", "file.name", "{}"),
    ("size", None, None),
    ("utc", "camera.utc_at_exposure", "{}"),
    ("lat", "geospatial.latitude", "{:.6f}"),
    ("lon", "geospatial.longitude", "{:.6f}"),
    ("rel_alt_m", "geospatial.relative_altitude_m", "{:.1f}"),
    ("gimbal_pitch", "attitude.gimbal_pitch", "{:.1f}"),
    ("lrf_m", "geospatial.rangefinder.distance_m", "{:.2f}"),
    ("distance_m", "thermal.measurement_params.distance", "{:.2f}"),
    ("humidity_%", "thermal.measurement_params.humidity", "{:.0f}"),
    ("emissivity", "thermal.measurement_params.emissivity", "{:.2f}"),
    ("reflection_{u}", "thermal.measurement_params.reflection", "{:.1f}"),
    ("ambient_{u}", "thermal.measurement_params.ambient_temp", "{:.1f}"),
    ("min_{u}", "temperature.min", "{:.2f}"),
    ("max_{u}", "temperature.max", "{:.2f}"),
    ("mean_{u}", "temperature.mean", "{:.2f}"),
    ("palette", "thermal.palette.name", "{}"),
)


def _format_table(rows: list[dict], unit: str) -> str:
    if not rows:
        return "No thermal R-JPEGs found."
    u = unit.lower()
    columns = [c for c in TABLE_COLUMNS if c[1] is None or any(listing.flatten(r).get(c[1]) is not None for r in rows)]
    lines = [[h.format(u=u) for h, _, _ in columns]]
    for row in rows:
        flat = listing.flatten(row)
        cells = []
        for _, key, fmt in columns:
            if key is None:
                cells.append(f"{flat['camera.width']}x{flat['camera.height']}")
                continue
            v = flat.get(key)
            cells.append("-" if v is None else fmt.format(v))
        lines.append(cells)
    widths = [max(len(r[i]) for r in lines) for i in range(len(columns))]
    out = []
    for i, row in enumerate(lines):
        out.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)).rstrip())
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def _format_csv(rows: list[dict]) -> str:
    flat = [listing.flatten(r) for r in rows]
    header = list(dict.fromkeys(k for f in flat for k in f))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flat)
    return buf.getvalue().rstrip("\n")


def _cmd_list(args: argparse.Namespace) -> None:
    try:
        res = api.list_dataset(args.input_dir, args.unit, args.stats, args.exif == "all", args.jobs,
                               _printer(sys.stderr))
    except FileNotFoundError as e:
        _fail(e)
    for status in res.problems:
        print(f"[skip, {status.reason}] {status.path}", file=sys.stderr)
    if args.format == "json":
        print(json.dumps(res.rows, indent=2))
    elif args.format == "csv":
        print(_format_csv(res.rows))
    else:
        print(_format_table(res.rows, args.unit))
    print(f"{len(res.rows)} thermal R-JPEG(s) listed"
          + (f", {res.not_thermal} ordinary JPEG(s) without thermal data skipped" if res.not_thermal else "")
          + (f", {len(res.problems)} problem file(s) skipped (see above; `check` for details)" if res.problems else "")
          + ".", file=sys.stderr)


def _cmd_check(args: argparse.Namespace) -> None:
    try:
        res = api.check(args.input_dir, args.unit, args.mosaic_max_pitch, args.jobs, webodm=args.webodm)
    except FileNotFoundError as e:
        _fail(e)
    if args.format == "json":
        out = {"summary": res.summary, "findings": [dataclasses.asdict(f) for f in res.findings]}
        if res.webodm:
            plan = dataclasses.asdict(res.webodm)
            del plan["findings"]                     # already in findings
            plan["odm_arguments"] = res.webodm.odm_arguments()
            out["webodm"] = plan
        print(json.dumps(out, indent=2))
    else:
        print(_format_findings(res.findings, res.summary, args.all_files))
        if res.webodm:
            print("\n" + _format_webodm(res.webodm))
    sys.exit(1 if res.failed(args.strict) else 0)


def _format_findings(findings: list, summary: dict, all_files: bool) -> str:
    files = ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in sorted(summary["files"].items()))
    lines = [f"Checked {sum(summary['files'].values())} JPEG file(s): {files or 'none'}.",
             f"{summary['error']} error(s), {summary['warning']} warning(s), {summary['info']} note(s).", ""]
    for f in findings:
        lines.append(f"{f.severity.upper():7}  {f.message}")
        shown = f.files if all_files else f.files[:5]
        for path in shown:
            lines.append(f"         - {path}")
        if len(f.files) > len(shown):
            lines.append(f"         ... and {len(f.files) - len(shown)} more (--all-files, or --format json)")
        if f.suggestion:
            lines.append(f"         -> {f.suggestion}")
        lines.append("")
    if not findings:
        lines.append("No problems found.")
    return "\n".join(lines).rstrip()


def _format_webodm(plan) -> str:
    lines = ["WebODM / ODM", "============", ""]
    if plan.coverage:
        c = plan.coverage
        lines += [f"Coverage: {c['frames']} nadir frames over {c['area_m2']:.0f} m2 at {c['gsd_m'] * 100:.1f} cm/px; each "
                  f"ground point seen by {c['images_per_point_median']:g} frames (median, 10th percentile "
                  f"{c['images_per_point_p10']:g}), about {c['equivalent_overlap_pct']:.0f} % front and side overlap.", ""]
    lines += ["1. Prepare the images:", "", f"   {plan.process_command}", ""]
    lines += [f"   - {note}" for note in plan.notes]
    lines += ["", "2. Upload OUTPUT's *_temp_c.tiff files to WebODM, with these task options:", ""]
    for name, opt in plan.recommended.items():
        lines.append(f"   {name} = {opt.value}".replace(" = True", "") + f"    ({opt.reason})")
    if plan.consider:
        lines += ["", "   Consider, depending on the site:", ""]
        for name, opt in plan.consider.items():
            lines.append(f"   {name} = {opt.value}".replace(" = True", "") + f"    ({opt.reason})")
    lines += ["", "   As ODM arguments: " + " ".join(plan.odm_arguments())]
    return "\n".join(lines)


def main() -> None:
    try:
        _main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


def _main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "process":
        _cmd_process(args)
    elif args.command == "list":
        _cmd_list(args)
    elif args.command == "check":
        _cmd_check(args)
