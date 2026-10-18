# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

### Added

- `process --strip-location` (API: `Settings(strip_location=True)`): outputs and mosaic
  without location or identifiers, for sharing.

### Fixed

- Two inputs whose outputs would have the same names (`a.jpg` and `a.jpeg`, or `A.jpg` and
  `a.jpg` on Windows and macOS) no longer overwrite each other: `process` keeps the first and
  reports the other, and `check` warns about them.
- Outputs are written all-or-nothing: an interrupted or failed run never leaves a
  half-written file.
- Ctrl+C stops a run promptly with "Interrupted." (exit code 130) instead of a traceback.
- Temperatures from files the SDK reads with its iirp plugin (M2EA, M30T, M3T, M3TD, M4T,
  H20N, H30T) now match the SDK in rare cases where they differed by one transmission step (up to 0.03 °C, and only
  under some measurement parameters): the port rounded the Kelvin conversion and the
  humidity coefficient in single precision where the SDK uses double.

## [0.1.0]

First release.

- Pure-Python SDK tested against DJI's Thermal SDK v1.8 measurement and image processing, bit-for-bit
  identical to the SDK's `dirp_measure_ex` and `dirp_process`, with no native library.
- `process`: temperature images in Celsius, Fahrenheit or Kelvin (float32 TIFF or OpenEXR,
  16-bit TIFF or PNG), raw sensor frames, 16-bit grayscale and the SDK's pseudo-colour
  palettes; EXIF, GPS and DJI XMP carried over.
- Dataset-wide adjustments: gimbal override, per-pixel out-of-range replacement and
  per-frame normalization, each recorded in the output's XMP.
- Mosaic: a georeferenced float32 GeoTIFF and previews from a flight's nadir frames, with
  optional alignment and normalization.
- `list`: camera, measurement, temperature, position and attitude for every thermal image,
  as a table, JSON or CSV.
- `check`: content-based validation of a dataset, with `--webodm` to plan a WebODM / ODM task.
- Python API (`import dji_thermal_cli as dtc`) with everything the command line does.

[Unreleased]: https://github.com/WinuxNomacs/dji-thermal-cli/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/WinuxNomacs/dji-thermal-cli/releases/tag/v0.1.0
