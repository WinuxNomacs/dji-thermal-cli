# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

## [1.1.0] - 2026-10-18

- Python 3.12 and 3.13 are supported, with numpy 1.26 or newer, so the package installs into
  existing environments such as WebODM's. Results are bit-for-bit the same on every supported
  version.
- `list` and the mosaic no longer fail on an EXIF rational with a zero denominator before
  Python 3.14; such a value is reported as missing, as on 3.14.

## [1.0.0] - 2026-10-18

First release.

- Pure-Python SDK tested against DJI's Thermal SDK v1.8 measurement and image processing,
  bit-for-bit identical to the SDK's `dirp_measure_ex` and `dirp_process`, with no native
  library.
- `process`: temperature images in Celsius, Fahrenheit or Kelvin (float32 TIFF or OpenEXR,
  16-bit TIFF or PNG), raw sensor frames, 16-bit grayscale and the SDK's pseudo-colour
  palettes; EXIF, GPS and DJI XMP carried over.
- `process --strip-location` (API: `Settings(strip_location=True)`): outputs and mosaic
  without location or identifiers, for sharing.
- Dataset-wide adjustments: gimbal override, per-pixel out-of-range replacement and
  per-frame normalization, each recorded in the output's XMP.
- Mosaic: a georeferenced float32 GeoTIFF and previews from a flight's nadir frames, with
  optional alignment and normalization.
- `list`: camera, measurement, temperature, position and attitude for every thermal image,
  as a table, JSON or CSV.
- `check`: content-based validation of a dataset, with `--webodm` to plan a WebODM / ODM
  task; it warns about inputs whose outputs would share a name.
- Safe outputs: inputs whose outputs would share a name (`a.jpg` and `a.jpeg`, or `A.jpg`
  and `a.jpg` on Windows and macOS) never overwrite each other; every file is written
  all-or-nothing; Ctrl+C stops a run with "Interrupted." (exit code 130).
- Python API (`import dji_thermal_cli as dtc`) with everything the command line does.
- Linux, Windows and macOS.

## [0.1.0rc1] - 2026-10-18

Release candidate of 1.0.0, with the same contents.

[Unreleased]: https://github.com/WinuxNomacs/dji-thermal-cli/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/WinuxNomacs/dji-thermal-cli/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/WinuxNomacs/dji-thermal-cli/releases/tag/v1.0.0
[0.1.0rc1]: https://github.com/WinuxNomacs/dji-thermal-cli/releases/tag/v0.1.0rc1
