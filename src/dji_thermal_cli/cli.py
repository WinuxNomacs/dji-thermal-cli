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

Requires the DJI Thermal SDK binaries. By default this looks for them under
./lib/tsdk-core/lib (vendored in this repo); override with --dll-dir or the
DJI_THERMAL_SDK_LIB_DIR environment variable.

Each file's SDK calls run in their own subprocess by default: libdirp has
been observed to segfault natively on a handful of R-JPEGs (even some of
DJI's own SDK sample files) while succeeding on the rest of a batch, and
isolating per-file means one crashing file gets reported and skipped rather
than taking the whole run down. Pass --no-isolate to run in-process instead
(faster, but a crash then kills the batch).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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
    proc.add_argument("--dll-dir", type=str, default=None, help="Directory containing libdirp; see module docstring for default")
    proc.add_argument("--skip-temperature-tiff", action="store_true", help="Don't write the float32 temperature TIFF")
    proc.add_argument("--skip-pseudo-color-png", action="store_true", help="Don't write the normalized pseudo-color PNG")
    proc.add_argument("--no-isolate", action="store_true", help="Run in-process instead of one subprocess per file (faster, but a native SDK crash on one file kills the whole batch)")

    lst = sub.add_parser("list", help="Print each R-JPEG's embedded parameters without writing anything")
    lst.add_argument("--input-dir", required=True, type=Path, help="Directory (searched recursively) or single R-JPEG file")
    lst.add_argument("--dll-dir", type=str, default=None)
    lst.add_argument("--format", choices=["table", "json"], default="table")
    lst.add_argument("--unit", choices=["C", "F"], default="C", help="Unit to display reflection/ambient_temp in")
    lst.add_argument("--no-isolate", action="store_true", help="Run in-process instead of one subprocess per file (faster, but a native SDK crash on one file kills the whole batch)")

    return p


def _resolve_input(input_dir: Path) -> Path:
    if not input_dir.exists():
        sys.exit(f"Input path does not exist: {input_dir}")
    return input_dir


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
    dirp = None if not args.no_isolate else sdk.DirpSDK(args.dll_dir)

    total = 0
    processed = 0
    for src in core.find_thermal_candidates(input_dir):
        total += 1
        call_args = (
            src, args.output_dir, src_root, overrides, range_min_c, range_max_c, args.palette,
            not args.skip_pseudo_color_png, not args.skip_temperature_tiff,
        )
        if args.no_isolate:
            ok, payload = True, dataclasses.asdict(core.process_file(dirp, *call_args))
        else:
            ok, payload = isolate.run_isolated(core.process_file_isolated, args.dll_dir, *call_args)

        if ok and payload["ok"]:
            processed += 1
            print(f"[ok] {src}")
        elif ok:
            print(f"[skip, {payload['reason']}] {src}")
        else:
            print(f"[skip, {payload}] {src}")

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
        cb_str = f"{cb['low']:.1f}..{cb['high']:.1f}" if cb["manual_enable"] else "auto"
        lines.append([
            Path(row["path"]).name,
            f"{row['width']}x{row['height']}",
            f"{mp['distance']:.2f}",
            f"{mp['humidity']:.0f}",
            f"{mp['emissivity']:.2f}",
            f"{mp['reflection']:.1f}",
            f"{mp['ambient_temp']:.1f}",
            row["palette"]["name"],
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
    _check_sdk_available(args.dll_dir)
    dirp = sdk.DirpSDK(args.dll_dir) if args.no_isolate else None

    to_display = core.celsius_to_fahrenheit if args.unit == "F" else (lambda v: v)

    rows = []
    for src in core.find_thermal_candidates(input_dir):
        if args.no_isolate:
            ok, info = True, core.describe_file(dirp, src)
        else:
            ok, info = isolate.run_isolated(core.describe_file_isolated, args.dll_dir, src)
        if not ok:
            print(f"[skip, {info}] {src}", file=sys.stderr)
            continue
        if info is None:
            print(f"[skip, not an R-JPEG / unsupported] {src}", file=sys.stderr)
            continue
        info["measurement_params"]["reflection"] = to_display(info["measurement_params"]["reflection"])
        info["measurement_params"]["ambient_temp"] = to_display(info["measurement_params"]["ambient_temp"])
        if not info["color_bar"]["manual_enable"]:
            info["color_bar"]["low"] = info["color_bar"]["high"] = 0.0
        else:
            info["color_bar"]["low"] = to_display(info["color_bar"]["low"])
            info["color_bar"]["high"] = to_display(info["color_bar"]["high"])
        rows.append(info)
        from pprint import pprint
        pprint(info)

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
