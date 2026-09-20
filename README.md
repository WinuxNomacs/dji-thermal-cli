# dji-thermal-cli

Batch-process DJI R-JPEG thermal images: a float32 temperature TIFF and a
pseudo-color PNG per file.

Temperatures are computed by a pure-Python model, so the temperature TIFF and
`list` work with no DJI SDK installed. The pseudo-color PNG (and anything the
model doesn't cover) uses the DJI Thermal SDK v1.8 when you have it.

This does **not** rewrite the original R-JPEG: DJI bakes its environmental
parameters into a binary APP4 block at capture time that isn't writable by
the SDK. Every command below only reads originals and/or writes new
derivative files.

## Using the output with WebODM / ODX

Each `*_temp_c.tiff` is an uncompressed single-band float32 TIFF in degrees Celsius, at the
sensor's true resolution (640x512 for the Matrice 4T). It carries the source R-JPEG's EXIF
(including GPS and the Exif sub-IFD), its DJI XMP, and `Camera:BandName="LWIR"`, which is what
ODM/ODX read to georeference each frame and to recognise it as thermal. Upload the TIFFs as
your images and leave `--radiometric-calibration` off: the values are already temperatures.

The per-frame TIFFs are not GeoTIFFs; the GeoTIFF is the orthophoto ODX builds. Pixels the
model cannot represent (below about -45 C) are written as -273.15, which marks them as having
no valid temperature.

## Temperature backends

`--backend auto` (default) uses the pure-Python model where it applies and falls
back to the native SDK otherwise. `--backend python` never touches the SDK;
`--backend native` always uses it.

The pure-Python model covers the Matrice 4T (640x512):

- **Range:** emissivity 0.4-1.0. Lower values are refused; in `auto` mode they fall
  back to the native SDK when it is installed.
- **Format:** only the `iirp` R-JPEG header variant is supported; other DJI variants
  are rejected with a clear message. Pixels below about -45 C are written as -273.15.
- **Not covered:** the pseudo-color PNG needs the native SDK. `list` reports the
  color bar and palette only when the SDK is available.
- If no pixel has a valid temperature for the chosen parameters, the file is reported as
  skipped instead of writing an empty image.

## Setup

The native SDK is optional. To use it (for the pseudo-color PNG):

1. Use static sdk version in libs or download the DJI Thermal SDK yourself from
   https://www.dji.com/downloads/softwares/dji-thermal-sdk and unpack it so
   its contents land under `lib/` here, i.e. `lib/tsdk-core/lib/<windows|linux>/...`
   should contain `libdirp.dll`/`libdirp.so` and its companion libraries.
   Only `lib/tsdk-core/api/*.h` (already checked in) is MIT-licensed per the
   SDK's own `License.txt`; the compiled binaries are covered by DJI's
   separate [SDK EULA](https://developer.dji.com/policies/eula/), which is
   why they aren't committed to this repo -- `lib/` beyond the headers is
   gitignored.
2. Nothing else: install the package (`uv sync`) with or without the SDK.

The tool checks the SDK's API revision the first time it opens a file, because struct layouts
differ between SDK releases and a mismatch can silently return wrong values. Revision 20 (DJI
Thermal SDK v1.8) is supported; any other release is refused with a message, and in `auto` mode
the temperature TIFF is still produced by the pure-Python model. To try a different release
anyway, pass `--allow-unknown-sdk-version` (or set `DJI_THERMAL_SDK_ALLOW_UNKNOWN_VERSION=1`);
its results are unchecked.

The SDK binaries under `lib/tsdk-core/lib/` are picked up automatically once
present. To point at a different SDK install, pass `--dll-dir` or set
`DJI_THERMAL_SDK_LIB_DIR`.

## Usage

Inspect what's embedded in a batch of files without writing anything:

```
uv run dji-thermal-cli list --input-dir /path/to/mission
uv run dji-thermal-cli list --input-dir /path/to/mission --format json
```

Produce normalized derivatives. Any of `--distance/--humidity/--emissivity/
--reflected-temp/--ambient-temp` you omit keeps that file's own embedded
value -- pass only the ones you want to change for the whole batch:

```
uv run dji-thermal-cli process \
    --input-dir /path/to/mission \
    --output-dir /path/to/normalized_out \
    --emissivity 0.95 --reflected-temp 20 --unit C \
    --range-min 15 --range-max 45
```

Files are processed concurrently by a pool of crash-isolated worker processes
(`--jobs N`, default `min(CPU count, 8)`). If the native SDK segfaults or hangs
on a file, only that file is reported as skipped; the worker is replaced and
the rest of the batch continues.

Run `uv run dji-thermal-cli process --help` / `... list --help` for the full
flag list.

## License

GNU Affero General Public License v3.0 (`AGPL-3.0-only`), the same license as
[WebODM](https://github.com/WebODM/WebODM) and [ODX](https://github.com/WebODM/ODX). See `LICENSE`.

The pure-Python temperature model was built by observing the DJI Thermal SDK's
output; none of the SDK's code or binaries are included. The native backend loads
the DJI SDK binaries only if you install them yourself, under DJI's own
[SDK EULA](https://developer.dji.com/policies/eula/).
