#!/usr/bin/env python3
"""
normalize_dji_thermal.py

Recursively walks a directory for DJI R-JPEG thermal images and produces,
for each one, two normalized derivative outputs using fixed environmental
and display parameters across the whole batch:

  1. A float32 temperature array (Celsius) as a .tiff, computed with your
     specified distance / humidity / emissivity / reflected-temperature.
  2. A pseudo-color PNG rendered with a fixed color-bar (temperature scale)
     min/max, so every image in the batch uses the same visual range.

IMPORTANT, read before relying on this:
  - This does NOT rewrite the original R-JPEG. DJI's R-JPEG embeds its
    environmental parameters in a binary APP4 block that is fixed at
    capture and is not writable by ExifTool or this SDK. The outputs
    below are new derived files, not edited originals.
  - The measurement-parameter struct fields (distance/humidity/emissivity/
    reflection) were confirmed by reading the dji_thermal_sdk==0.0.2
    Python source directly. That wrapper targets DJI Thermal SDK v1.3.
    If you're running a newer official SDK (M3T/M30T/M4T-era), its
    dirp_measurement_params_t may include a 5th field (separate ambient/
    atmospheric temperature, distinct from reflected temperature) --
    check the dirp_api.h shipped with your SDK download before trusting
    the struct layout here. If it has 5 fields, add "ambient" as another
    c_float after "reflection" and pass --ambient-temp-c.
  - Units: humidity is a 0-1 fraction (not percent) per the SDK's own
    PrintConv (val*100 -> %). Temperatures passed to the SDK are Celsius,
    per the dirp_measure_ex docstring ("FLOAT32 pixel value ... real
    temperature in Celsius"). This script accepts Fahrenheit or Celsius
    input and converts.

Requires:
  - The DJI Thermal SDK binaries (libdirp.dll / libdirp.so) downloaded
    from https://www.dji.com/downloads/softwares/dji-thermal-sdk
    (not included in the pip package -- that package is ctypes bindings
    only, no compiled library).
  - pip install dji_thermal_sdk numpy pillow tifffile exif

Usage (uv):
  uv run normalize_dji_thermal.py \
      --input-dir /path/to/mission \
      --output-dir /path/to/normalized_out \
      --distance 5.0 --humidity 50 --emissivity 0.95 \
      --reflected-temp 20 --unit C \
      --range-min 15 --range-max 45 \
      --dll-dir /path/to/sdk/libs

The --dll-dir should contain libdirp.dll (Windows) or libdirp.so (Linux)
directly, or a windows/ or linux/ subfolder containing it -- matches
dji_init()'s default lookup behavior in the SDK wrapper.
"""

import argparse
import ctypes as CT
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tifffile
    HAVE_TIFFFILE = True
except ImportError:
    HAVE_TIFFFILE = False

from dji_thermal_sdk.dji_sdk import (
    DIRP_HANDLE,
    DIRP_SUCCESS,
    dirp_color_bar_t,
    dirp_create_from_rjpeg,
    dirp_destroy,
    dirp_get_rjpeg_resolution,
    dirp_measure_ex,
    dirp_measurement_params_t,
    dirp_process,
    dirp_resolution_t,
    dirp_set_color_bar,
    dirp_set_measurement_params,
    dirp_set_pseudo_color,
    dji_init,
)

THERMAL_SUFFIXES = (".jpg", ".jpeg")


def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def find_thermal_candidates(root: Path):
    """
    Recursively yield candidate R-JPEG files. DJI thermal frames are
    conventionally named with a _T suffix (e.g. DJI_0001_T.JPG), but not
    every fleet follows that -- we filter by suffix as a fast pre-check,
    then let dirp_create_from_rjpeg be the actual source of truth by
    skipping files it rejects.
    """
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in THERMAL_SUFFIXES:
            yield path


def process_one(
    src: Path,
    dst_root: Path,
    src_root: Path,
    measurement: dirp_measurement_params_t,
    color_bar: dirp_color_bar_t,
    palette: int,
    save_pseudo_color: bool,
    save_temperature: bool,
) -> bool:
    data = src.read_bytes()
    buf = CT.create_string_buffer(data)
    size = CT.c_int32(len(data))

    ret = dirp_create_from_rjpeg(buf, size, CT.byref(DIRP_HANDLE))
    if ret != DIRP_SUCCESS:
        return False  # not a DJI R-JPEG, or unsupported -- skip quietly

    try:
        ret = dirp_set_measurement_params(DIRP_HANDLE, CT.byref(measurement))
        if ret != DIRP_SUCCESS:
            print(f"  [warn] set_measurement_params failed ({ret}): {src}")

        resolution = dirp_resolution_t()
        dirp_get_rjpeg_resolution(DIRP_HANDLE, CT.byref(resolution))
        w, h = resolution.width, resolution.height

        rel = src.relative_to(src_root)
        out_dir = dst_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = rel.stem

        if save_temperature:
            temp_size = w * h * CT.sizeof(CT.c_float)
            temp_buf = CT.create_string_buffer(temp_size)
            ret = dirp_measure_ex(DIRP_HANDLE, CT.byref(temp_buf), temp_size)
            if ret != DIRP_SUCCESS:
                print(f"  [warn] dirp_measure_ex failed ({ret}): {src}")
            else:
                temp_arr = np.frombuffer(temp_buf.raw, dtype=np.float32).reshape(h, w)
                out_tiff = out_dir / f"{stem}_temp_c.tiff"
                if HAVE_TIFFFILE:
                    tifffile.imwrite(out_tiff, temp_arr)
                else:
                    Image.fromarray(temp_arr, mode="F").save(out_tiff)

        if save_pseudo_color:
            ret = dirp_set_color_bar(DIRP_HANDLE, CT.byref(color_bar))
            if ret != DIRP_SUCCESS:
                print(f"  [warn] set_color_bar failed ({ret}): {src}")
            ret = dirp_set_pseudo_color(DIRP_HANDLE, CT.c_int(palette))
            if ret != DIRP_SUCCESS:
                print(f"  [warn] set_pseudo_color failed ({ret}): {src}")

            color_size = w * h * 3 * CT.sizeof(CT.c_uint8)
            color_buf = CT.create_string_buffer(color_size)
            ret = dirp_process(DIRP_HANDLE, CT.byref(color_buf), color_size)
            if ret != DIRP_SUCCESS:
                print(f"  [warn] dirp_process failed ({ret}): {src}")
            else:
                rgb = np.frombuffer(color_buf.raw, dtype=np.uint8).reshape(h, w, 3)
                out_png = out_dir / f"{stem}_normalized.png"
                Image.fromarray(rgb, mode="RGB").save(out_png)

        return True
    finally:
        dirp_destroy(DIRP_HANDLE)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--distance", required=True, type=float, help="Object distance in meters")
    p.add_argument("--humidity", required=True, type=float, help="Relative humidity as a percent, e.g. 50 for 50%%")
    p.add_argument("--emissivity", required=True, type=float, help="Emissivity, 0.0-1.0")
    p.add_argument("--reflected-temp", required=True, type=float, help="Reflected apparent temperature")
    p.add_argument("--unit", choices=["C", "F"], default="C", help="Unit for --reflected-temp, --range-min, --range-max")
    p.add_argument("--range-min", type=float, default=None, help="Fixed color-bar minimum temperature (enables manual color bar)")
    p.add_argument("--range-max", type=float, default=None, help="Fixed color-bar maximum temperature (enables manual color bar)")
    p.add_argument("--palette", type=int, default=0, help="DIRP pseudo-color palette index, 0=whitehot (see SDK enum)")
    p.add_argument("--dll-dir", type=str, default=None, help="Directory containing libdirp.dll/.so, or its windows/linux subfolder")
    p.add_argument("--skip-temperature-tiff", action="store_true", help="Don't write the float32 temperature TIFF")
    p.add_argument("--skip-pseudo-color-png", action="store_true", help="Don't write the normalized pseudo-color PNG")
    args = p.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    osname = "windows" if os.name == "nt" else "linux"
    if args.dll_dir:
        libname = "libdirp.dll" if osname == "windows" else "libdirp.so"
        candidate = Path(args.dll_dir) / libname
        dllpath = str(candidate) if candidate.exists() else str(Path(args.dll_dir) / osname / libname)
        dji_init(dllpath=dllpath, osname=osname)
    else:
        dji_init(osname=osname)

    reflected_c = args.reflected_temp if args.unit == "C" else fahrenheit_to_celsius(args.reflected_temp)
    measurement = dirp_measurement_params_t(
        distance=args.distance,
        humidity=args.humidity / 100.0,
        emissivity=args.emissivity,
        reflection=reflected_c,
    )

    manual_range = args.range_min is not None and args.range_max is not None
    if manual_range:
        rmin = args.range_min if args.unit == "C" else fahrenheit_to_celsius(args.range_min)
        rmax = args.range_max if args.unit == "C" else fahrenheit_to_celsius(args.range_max)
        color_bar = dirp_color_bar_t(manual_enable=True, high=rmax, low=rmin)
    else:
        color_bar = dirp_color_bar_t(manual_enable=False, high=0.0, low=0.0)

    total = 0
    processed = 0
    for src in find_thermal_candidates(args.input_dir):
        total += 1
        ok = process_one(
            src,
            args.output_dir,
            args.input_dir,
            measurement,
            color_bar,
            args.palette,
            save_pseudo_color=not args.skip_pseudo_color_png,
            save_temperature=not args.skip_temperature_tiff,
        )
        if ok:
            processed += 1
            print(f"[ok] {src}")
        else:
            print(f"[skip, not R-JPEG] {src}")

    print(f"\nDone. {processed}/{total} candidate files processed.")
    if manual_range:
        print(f"Applied fixed color-bar range: {color_bar.low:.1f} C to {color_bar.high:.1f} C")
    else:
        print("No fixed color-bar range set (--range-min/--range-max omitted); pseudo-color used auto range per image.")


if __name__ == "__main__":
    main()
