"""
dji-thermal-cli

Recursively walks a directory (or reads a single file) for DJI R-JPEG
thermal images.

  `process` produces, for each R-JPEG, two normalized derivative outputs:
    1. A float32 temperature array (Celsius) as a .tiff.
    2. A pseudo-color PNG.
  Any of --distance/--humidity/--emissivity/--reflected-temp/--ambient-temp
  you omit is left as whatever value is already embedded in that R-JPEG
  (read via dirp_get_measurement_params) -- you only need to pass the ones
  you want to change. This does NOT rewrite the original R-JPEG: DJI bakes
  its environmental parameters into a binary APP4 block at capture time that
  is not writable by this SDK.

  `list` prints each R-JPEG's embedded measurement parameters (and the
  valid range for each, resolution, R-JPEG version, and current color-bar
  /palette settings) without writing anything.

Units: humidity is a percent, e.g. 50 for 50%. Temperatures passed to the
SDK are Celsius; this CLI accepts Fahrenheit or Celsius input via --unit and
converts.

Backends (--backend): temperatures come from a pure-Python model of the DJI
R-JPEG radiometry ("python", no DJI SDK needed; supports emissivity 0.4-1.0
on Matrice 4T data) or from the DJI Thermal SDK ("native"). "auto" (default) uses
the Python model when it applies and the native SDK otherwise. The pseudo-color
PNG is only available through the native SDK: by default this looks for it under
./lib/tsdk-core/lib; override with --dll-dir or DJI_THERMAL_SDK_LIB_DIR.

Native SDK calls run in crash-isolated worker processes: libdirp has been
observed to segfault natively on a handful of R-JPEGs (even some of DJI's own
SDK sample files) while succeeding on the rest of a batch, so one crashing file
is reported and skipped rather than taking the whole run down. Pass
--no-isolate to run in-process instead (faster, but a crash then kills the batch).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from . import core, isolate, sdk


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dji-thermal-cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    proc = sub.add_parser("process", help="Write temperature TIFF / pseudo-color PNG derivatives")
    proc.add_argument("--input-dir", required=True, type=Path, help="Directory (searched recursively) or single R-JPEG file")
    proc.add_argument("--output-dir", required=True, type=Path)
    proc.add_argument("--distance", type=float, default=None, help="Object distance in meters (default: keep each file's embedded value)")
    proc.add_argument("--humidity", type=float, default=None, help="Relative humidity as a percent, e.g. 50 for 50%% (default: keep embedded)")
    proc.add_argument("--emissivity", type=float, default=None, help="Emissivity, 0.0-1.0 (default: keep embedded)")
    proc.add_argument("--reflected-temp", type=float, default=None, dest="reflected_temp", help="Reflected apparent temperature (default: keep embedded)")
    proc.add_argument("--ambient-temp", type=float, default=None, dest="ambient_temp", help="Ambient temperature (default: keep embedded)")
    proc.add_argument("--unit", choices=["C", "F"], default="C", help="Unit for --reflected-temp/--ambient-temp/--range-min/--range-max")
    proc.add_argument("--range-min", type=float, default=None, help="Fixed color-bar minimum temperature (enables manual color bar)")
    proc.add_argument("--range-max", type=float, default=None, help="Fixed color-bar maximum temperature (enables manual color bar)")
    proc.add_argument("--palette", type=int, default=None, choices=range(10), metavar="0-9", help="DIRP pseudo-color palette index (default: keep current, 0=whitehot)")
    proc.add_argument("--backend", choices=core.BACKENDS, default="auto", help="Temperature source: pure-Python model, native DJI SDK, or auto (Python where it applies, else native)")
    proc.add_argument("--allow-unknown-sdk-version", action="store_true", help="Use a DJI Thermal SDK whose API revision this tool has not been built for (struct layouts may differ; results are unchecked)")
    proc.add_argument("--dll-dir", type=str, default=None, help="Directory containing libdirp; see module docstring for default")
    proc.add_argument("--skip-temperature-tiff", action="store_true", help="Don't write the float32 temperature TIFF")
    proc.add_argument("--skip-pseudo-color-png", action="store_true", help="Don't write the normalized pseudo-color PNG")
    proc.add_argument("--jobs", "-j", type=int, default=None, help="Concurrent worker processes (default: min(CPU count, 8)). Each worker is crash-isolated; ignored with --no-isolate")
    proc.add_argument("--no-isolate", action="store_true", help="Run in-process instead of one subprocess per file (faster, but a native SDK crash on one file kills the whole batch)")

    lst = sub.add_parser("list", help="Print each R-JPEG's embedded parameters without writing anything")
    lst.add_argument("--input-dir", required=True, type=Path, help="Directory (searched recursively) or single R-JPEG file")
    lst.add_argument("--backend", choices=core.BACKENDS, default="auto", help="Temperature source: pure-Python model, native DJI SDK, or auto (Python where it applies, else native)")
    lst.add_argument("--allow-unknown-sdk-version", action="store_true", help="Use a DJI Thermal SDK whose API revision this tool has not been built for (struct layouts may differ; results are unchecked)")
    lst.add_argument("--dll-dir", type=str, default=None)
    lst.add_argument("--format", choices=["table", "json"], default="table")
    lst.add_argument("--unit", choices=["C", "F"], default="C", help="Unit to display reflection/ambient_temp in")
    lst.add_argument("--jobs", "-j", type=int, default=None, help="Concurrent worker processes (default: min(CPU count, 8)). Each worker is crash-isolated; ignored with --no-isolate")
    lst.add_argument("--no-isolate", action="store_true", help="Run in-process instead of one subprocess per file (faster, but a native SDK crash on one file kills the whole batch)")

    return p


def _resolve_input(input_dir: Path) -> Path:
    if not input_dir.exists():
        sys.exit(f"Input path does not exist: {input_dir}")
    return input_dir


DEFAULT_MAX_WORKERS = 8  # past this, worker startup and disk I/O outweigh the parallelism


def _worker_count(requested: int | None, n_jobs: int) -> int:
    n = requested if requested else min(os.cpu_count() or 1, DEFAULT_MAX_WORKERS)
    return max(1, min(n, n_jobs))


def _apply_sdk_version_override(args: argparse.Namespace) -> None:
    """Worker processes inherit the environment, so the flag reaches them through the variable."""
    if args.allow_unknown_sdk_version:
        os.environ[sdk.ALLOW_UNKNOWN_ENV] = "1"


def _check_sdk_available(dll_dir: str | None) -> None:
    """Fail fast with a clear message if libdirp can't be found, before
    spawning any per-file subprocesses."""
    try:
        sdk.resolve_dll_path(dll_dir)
    except FileNotFoundError as e:
        sys.exit(str(e))


def _cmd_process(args: argparse.Namespace) -> None:
    input_dir = _resolve_input(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _apply_sdk_version_override(args)
    if args.backend == "native":
        _check_sdk_available(args.dll_dir)

    to_c = core.fahrenheit_to_celsius if args.unit == "F" else (lambda v: v)
    overrides = core.MeasurementOverrides(
        distance=args.distance,
        humidity=args.humidity,
        emissivity=args.emissivity,
        reflection=None if args.reflected_temp is None else to_c(args.reflected_temp),
        ambient_temp=None if args.ambient_temp is None else to_c(args.ambient_temp),
    )
    range_min_c = None if args.range_min is None else to_c(args.range_min)
    range_max_c = None if args.range_max is None else to_c(args.range_max)
    manual_range = range_min_c is not None and range_max_c is not None
    src_root = input_dir if input_dir.is_dir() else input_dir.parent
    dirp = core.optional_sdk(args.dll_dir, args.backend) if args.no_isolate else None

    candidates = list(core.find_thermal_candidates(input_dir))
    total = len(candidates)
    processed = 0

    def report(src, ok, payload) -> None:
        nonlocal processed
        if ok and payload["ok"]:
            processed += 1
            tag = f" ({payload['temperature_backend']})" if payload["temperature_backend"] else ""
            print(f"[ok{tag}] {src}", flush=True)
            for note in payload["notes"]:
                print(f"     note: {note}", flush=True)
        elif ok:
            print(f"[skip, {payload['reason']}] {src}", flush=True)
        else:
            print(f"[skip, {payload}] {src}", flush=True)

    def call_args(src):
        return (
            src, args.output_dir, src_root, overrides, range_min_c, range_max_c, args.palette,
            not args.skip_pseudo_color_png, not args.skip_temperature_tiff, args.backend,
        )

    if args.no_isolate:
        for src in candidates:
            report(src, True, dataclasses.asdict(core.process_file(dirp, *call_args(src))))
    else:
        pool = isolate.WorkerPool(_worker_count(args.jobs, total))
        arglist = [(args.dll_dir, *call_args(src)) for src in candidates]
        for i, ok, payload in pool.imap_unordered(core.process_file_isolated, arglist):
            report(candidates[i], ok, payload)

    print(f"\nDone. {processed}/{total} candidate files processed.")
    if manual_range:
        print(f"Applied fixed color-bar range: {range_min_c:.1f} C to {range_max_c:.1f} C")
    else:
        print("No fixed color-bar range set (--range-min/--range-max omitted); pseudo-color used auto range per image.")


def _format_table(rows: list[dict]) -> str:
    if not rows:
        return "No R-JPEG candidates found."
    headers = ["file", "size", "distance_m", "humidity_%", "emissivity", "reflection_c", "ambient_c", "palette", "color_bar"]
    lines = [headers]
    for row in rows:
        mp = row["measurement_params"]
        cb = row["color_bar"]
        if cb is None:
            cb_str = "n/a"
        else:
            cb_str = f"{cb['low']:.1f}..{cb['high']:.1f}" if cb["manual_enable"] else "auto"
        lines.append([
            Path(row["path"]).name,
            f"{row['width']}x{row['height']}",
            f"{mp['distance']:.2f}",
            f"{mp['humidity']:.0f}",
            f"{mp['emissivity']:.2f}",
            f"{mp['reflection']:.1f}",
            f"{mp['ambient_temp']:.1f}",
            row["palette"]["name"] if row["palette"] else "n/a",
            cb_str,
        ])
    widths = [max(len(r[i]) for r in lines) for i in range(len(headers))]
    out = []
    for i, row in enumerate(lines):
        out.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def _cmd_list(args: argparse.Namespace) -> None:
    input_dir = _resolve_input(args.input_dir)
    _apply_sdk_version_override(args)
    if args.backend == "native":
        _check_sdk_available(args.dll_dir)
    dirp = core.optional_sdk(args.dll_dir, args.backend) if args.no_isolate else None

    to_display = core.celsius_to_fahrenheit if args.unit == "F" else (lambda v: v)

    candidates = list(core.find_thermal_candidates(input_dir))

    def results():
        if args.no_isolate:
            for src in candidates:
                yield src, True, core.describe_file(dirp, src, args.backend)
        else:
            pool = isolate.WorkerPool(_worker_count(args.jobs, len(candidates)))
            arglist = [(args.dll_dir, src, args.backend) for src in candidates]
            for i, ok, info in pool.imap_unordered(core.describe_file_isolated, arglist):
                yield candidates[i], ok, info

    rows = []
    for src, ok, info in results():
        if not ok:
            print(f"[skip, {info}] {src}", file=sys.stderr)
            continue
        if info is None:
            print(f"[skip, not an R-JPEG / unsupported] {src}", file=sys.stderr)
            continue
        info["measurement_params"]["reflection"] = to_display(info["measurement_params"]["reflection"])
        info["measurement_params"]["ambient_temp"] = to_display(info["measurement_params"]["ambient_temp"])
        color_bar = info["color_bar"]
        if color_bar is not None:
            if not color_bar["manual_enable"]:
                color_bar["low"] = color_bar["high"] = 0.0
            else:
                color_bar["low"] = to_display(color_bar["low"])
                color_bar["high"] = to_display(color_bar["high"])
        rows.append(info)
    rows.sort(key=lambda r: r["path"])

    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        print(_format_table(rows))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "process":
        _cmd_process(args)
    elif args.command == "list":
        _cmd_list(args)
