# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

- The mosaic's JSON sidecar holds each frame's model (`frame_models`): its camera, lens and
  rotation, the alignment's correction, the surface it was projected onto, and its
  footprint's corners in lon/lat. `read_frame_models` reads them back, and the `FrameModels`
  it returns finds a ground point (lon/lat) in every frame that saw it, not only the one the
  mosaic took it from (`pixels`, `view_angles`, `covering`), and lists the frames whose
  footprint meets an area (`meeting`). On a 220-frame M4T flight a point was seen by 15
  frames (median). Sidecars written with `--strip-location` have no frame models, and older
  sidecars read as None.

## [1.4.0] - 2026-10-25

- The mosaic GeoTIFF has two more bands, `source_u` and `source_v`: the column and row of
  the source frame's pixel each mosaic pixel was read from, so a point of the mosaic can be
  shown in its source image (`Mosaic.source_px` in the API). Readers of the first three
  bands are unaffected.

## [1.3.0] - 2026-10-23

- The mosaic projects its frames onto the heights it estimates from the parallax between
  overlapping frames, anchored to the rangefinder, instead of a flat ground plane
  (`--mosaic-heights auto`, the default; `plane` keeps the old projection). Roofs and trees
  no longer bend or step at the seams: on a 327-frame M4T flight with oblique frames the
  house's straight roof is straight again. The surface is written to `mosaic_heights.tif`,
  and the report's `heights` entry says which surface was used and how well it fits. With
  too few observed cells, or a surface too near a camera, the mosaic falls back to the plane
  and says why.
- A DSM can replace the estimated heights (`--mosaic-dsm PATH`, `--mosaic-dsm-offset METRES`;
  `MosaicOptions(dsm=..., dsm_offset=...)`). It is optional: the mosaic never needs one.
- Per-file jobs run in the calling process when it is daemonic, such as a Celery worker's
  (which cannot start processes), so callers there no longer need `jobs=1`; the mosaic's
  alignment still uses its threads.
- The mosaic builds faster: each frame is rendered once per matching pass instead of once per
  overlapping pair, blocks are matched on a gradient computed once per rendering instead of
  once per block, a pair's blocks are matched in one batch of real FFTs, the overlaps are
  matched in `--jobs` threads, and the frame placements are solved as a sparse system. On
  327 M4T frames the build takes 137 s instead of 752 s (8 threads), with the same
  alignment quality (median misalignment 0.087 m instead of 0.086 m).
- `make_mosaic(..., as_converted_dir=...)` (with `MosaicOptions(normalize=True)`) also writes
  the mosaic without the normalization, from the same alignment and misalignment check, in
  about the time of one mosaic instead of two. Its report is `MosaicReport.as_converted`.

## [1.2.0] - 2026-10-19

dji-thermal-cli is now in maintenance. Releases from here on bring bug fixes and quality-of-life
improvements to the command line and the Python SDK, kept in step with each other, as they are
needed. No support for newer camera models or newer DJI Thermal SDK versions is planned: the
cameras and the SDK v1.8 behaviour supported in 1.1.0 are what the package covers.

- The mosaic no longer fails on super-resolution frames (M4T: a 1280x1024 JPEG over 640x512
  measurements). Frames are placed at the size they measure, and `check --webodm` gives
  their ground sample distance at that size.
- A mosaic reports its progress. The Python API sends `"mosaic"` events, one per stage
  (reading frames, aligning, compositing, checking, writing) with a count in the stage's own
  unit and its place among the stages; the command line shows them on stderr
  (`Mosaic step 2 of 5: aligning (round 2/3), 812/1840 pairs`), updated in place on a terminal.
- `process --quiet` (`-q`) hides the progress display; results, skips and errors still print.

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

[Unreleased]: https://github.com/WinuxNomacs/dji-thermal-cli/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/WinuxNomacs/dji-thermal-cli/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/WinuxNomacs/dji-thermal-cli/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/WinuxNomacs/dji-thermal-cli/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/WinuxNomacs/dji-thermal-cli/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/WinuxNomacs/dji-thermal-cli/releases/tag/v1.0.0
[0.1.0rc1]: https://github.com/WinuxNomacs/dji-thermal-cli/releases/tag/v0.1.0rc1
