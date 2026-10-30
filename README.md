<p align="center">
  <img src="https://raw.githubusercontent.com/WinuxNomacs/dji-thermal-cli/main/img/dji-thermal-cli-sub5.gif"
       alt="dji-thermal-cli: DJI thermal images to temperatures" width="100%">
</p>

# dji-thermal-cli

Batch-process DJI R-JPEG thermal images into temperature images (float32 TIFF or OpenEXR,
16-bit TIFF or PNG), raw sensor frames, and 16-bit grayscale or pseudo-color views.

Temperatures and pseudo-color images come from a pure-Python port of the DJI Thermal SDK
v1.8. Its results are bit-for-bit identical to the SDK's `dirp_measure_ex` and `dirp_process`,
and it needs neither the SDK nor any native library. If you have DJI's SDK, `--sdk` measures
with DJI's own library instead (see [DJI's own SDK](#djis-own-sdk)).

This does **not** rewrite the original R-JPEG: DJI bakes its environmental
parameters into a binary APP4 block at capture time that isn't writable by
the SDK. Every command below only reads originals and/or writes new
derivative files.

## Install

```
pip install dji-thermal-cli
```

or as a stand-alone command with `uv tool install dji-thermal-cli` or
`pipx install dji-thermal-cli`. It needs Python 3.9 or newer; numpy and Pillow are its only
dependencies. Neither the DJI Thermal SDK nor any native library is needed. It runs on
Linux, Windows and macOS, and gives the same results on each.

From a clone, for development: `uv sync`, then `uv run dji-thermal-cli ...`.

## Using the output with WebODM / ODX

Each `*_temp_c.tiff` (the default `temp_c` output) is a Deflate-compressed single-band float32 TIFF in degrees Celsius, at the
sensor's true resolution (640x512 for most cameras, 1280x1024 for the H30T). It carries the
source R-JPEG's EXIF (including GPS and the Exif sub-IFD), its DJI XMP, and
`Camera:BandName="LWIR"`, which is what ODM/ODX read to georeference each frame and to
recognise it as thermal. Upload the TIFFs as your images and leave `--radiometric-calibration`
off: the values are already temperatures.

The per-frame TIFFs are not GeoTIFFs; the GeoTIFF is the orthophoto ODX builds. Pixels the
SDK cannot give a temperature come out at its floor of about -273 C.

`check --webodm` plans the whole task for a dataset:

```
dji-thermal-cli check --input-dir /path/to/flight --webodm
```

- **Frames:** nadir frames for an orthophoto (`--max-pitch`, since obliques mix viewing
  angles and sky into it), and none without a position.
- **Coverage:** how many nadir frames see each ground point, from every frame's footprint,
  and the equivalent front and side overlap. Fewer than 9 frames per point (about 67 %)
  is flagged, because small, low-texture thermal images need a lot of overlap to match.
- **The `process` command** that prepares the images: `-o temp_c:tiff`, one value for any
  measurement parameter that differs between files, and `--normalize`.
- **ODM task options** with the reason for each: `radiometric-calibration none`,
  `orthophoto-resolution` from the ground sample distance, `gps-accuracy` from the RTK
  accuracy; and, depending on the site, `feature-quality`, `min-num-features`,
  `fast-orthophoto`, `skip-3dmodel`. With `--format json` they come as data and as ODM
  arguments.

`process --normalize` makes the per-frame outputs consistent with each other, which matters
because WebODM blends overlapping frames itself. Every frame is measured with the flight's
median embedded ambient and reflected temperature (unless you give them), and each frame's
offset against its overlapping neighbours (camera drift between shutter recalibrations) is
removed from the temperature outputs. The offsets come from the frames' overlaps, found as the
mosaic finds them (`--mosaic-max-pitch`, `--mosaic-align`); frames it can't place get none.
Each output records its offset as `dtc:NormalizationOffsetC`. `process --max-pitch DEG` keeps
only frames looking at least that steeply down.

## Outputs

Choose what to write with `--output PRODUCT[:FORMAT]` (`-o`), repeated or comma-separated.
The default is `temp_c:tiff,pseudo:png`. Each output is named `<stem>_<product>.<format>`,
where `tif` and `tiff` are both accepted and the extension is written as you typed it.

| Product | Pixels | Formats (first = default) |
|---|---|---|
| `temp_c` | float32 temperature, °C (no data: -273.15) | tiff, exr |
| `temp_f` | float32 temperature, °F (no data: -459.67) | tiff, exr |
| `temp_k` | float32 temperature, K (no data: 0) | tiff, exr |
| `temp_ck` | uint16 temperature in centi-kelvin, K×100: 0.01 K steps up to 382.2 °C (no data: 0) | tiff, png |
| `temp_dc` | int16 temperature in tenths of a °C: the SDK's `dirp_measure` output | tiff |
| `raw` | uint16 raw sensor counts, uncalibrated (`dirp_get_original_raw`) | tiff, png |
| `gray16` | uint16 view: `--range-min..--range-max`, or each image's own range, stretched to 0..65535 | png, tiff |
| `pseudo` | 8-bit RGB pseudo-color (`--palette`, `--range-min/--range-max`) | png, tiff |
| `rjpeg` | the source R-JPEG written again, for DJI's own tools (see [R-JPEG copies](#r-jpeg-copies)) | jpg |

```
dji-thermal-cli process --input-dir mission --output-dir out -o temp_c:exr,temp_ck:png,gray16
```

`--palette` picks the colours of `pseudo` images and of the mosaic preview, by number or
by name: `--palette 3` and `--palette hotiron` are the same. Without it, each file keeps the
palette it was saved with. These are the DJI SDK's own colour tables:

| # | Name | Looks like |
|---|---|---|
| 0 | `whitehot` | grayscale, hot = white |
| 1 | `fulgurite` | black, red, yellow, white |
| 2 | `ironred` | DJI's ironbow: dark blue, purple, orange, yellow-white |
| 3 | `hotiron` | black, green, red |
| 4 | `medical` | multi-colour, for small temperature differences |
| 5 | `arctic` | blues for cold, gold for hot |
| 6 | `rainbow1` | full spectrum |
| 7 | `rainbow2` | full spectrum, a different ramp |
| 8 | `tint` | grayscale with hot areas tinted red |
| 9 | `blackhot` | grayscale, hot = black |

TIFF and EXR outputs are compressed losslessly by default: Deflate for TIFF (with the
horizontal predictor on 16-bit integers), ZIP for EXR. That makes them 2-4x smaller.
`--compression none` writes them uncompressed, for tools that can't read compressed files.
PNG is always compressed.

Every TIFF and PNG carries the source EXIF (GPS included) and XMP. The single-band thermal
products also carry `Camera:BandName="LWIR"`. PNGs record their unit (`units`), and a
`gray16` PNG records the range it was stretched over (`range_min_c` and `range_max_c`), as
text chunks. EXR files are a single float channel `Y` with a `units` attribute. `gray16` is
for viewing only: use `temp_c`, `temp_f`, `temp_k`, `temp_ck` or `temp_dc` for measurement.

### Temperature units

The SDK works in degrees Celsius only. The CLI converts at both ends and leaves the SDK's own
behaviour untouched:

- **Inputs:** `--input-unit C|F|K` (default C) sets the unit of `--reflected-temp`,
  `--ambient-temp`, `--range-min` and `--range-max`. They are converted to Celsius in double
  precision before the SDK sees them, so `--input-unit K --reflected-temp 293.15` and
  `--input-unit F --reflected-temp 68` measure exactly like `--reflected-temp 20`. Values
  below absolute zero are rejected.
- **Outputs:** `temp_c` is the SDK's own float32 Celsius. `temp_f` and `temp_k` convert it in
  double precision and round once, so each value is the float32 nearest the exact
  conversion. The SDK's no-data floor maps to absolute zero in each unit. `temp_ck` (uint16
  centi-kelvin) and `temp_dc` (the SDK's int16 tenths of a degree C) are unchanged. Ask for
  any mix, e.g. `-o temp_c,temp_f,temp_k:exr`.
- **Mosaic:** `--mosaic-unit C,F,K` (default C) writes one `mosaic_temp_<unit>.tif` (and
  `.json`) per unit. The report's seam and offset figures stay in °C, labelled `_c`.
- **`list --unit C|F|K`** shows every temperature (parameters, colour bar, statistics) in
  that unit.

## Mosaic

`--mosaic` stitches a flight's nadir thermal frames into one georeferenced image:

```
dji-thermal-cli process --input-dir /path/to/flight --output-dir out --mosaic
```

While it works, it shows its progress on stderr, one stage at a time
(`Mosaic step 2 of 5: aligning (round 2/3), 812/1840 pairs`). On a terminal that is one line
updated in place; in a log or a pipe, a line as each stage starts. `--quiet` (`-q`) hides it;
results, skipped files and errors still print.

It writes these files to the output directory:

- **`mosaic_temp_c.tif`:** a GeoTIFF in the flight's UTM zone (WGS84) with five float32
  bands. NaN marks ground no frame covers.
  1. temperature in °C (`--mosaic-unit F` or `K` for `mosaic_temp_f.tif`/`mosaic_temp_k.tif`);
  2. the frame each pixel came from;
  3. the off-nadir angle at which that frame saw it;
  4. and 5. the pixel of that frame it was read from (column, row), so a point of the mosaic
     can be found in its source image.
- **`mosaic_temp_c.json`:** the grid, a seam and misalignment report, and a table of every
  frame (also embedded in the GeoTIFF). Each frame's entry holds its file name, capture
  time, RTK position and accuracy, gimbal and aircraft attitude, rangefinder reading,
  height above the ground plane, and the measurement parameters used (with where its
  distance came from). Its `frame_models` entry (in the sidecar only) holds how each frame
  saw the ground: its camera, lens and rotation, the alignment's correction, the surface it
  was projected onto, and its footprint's corners in lon/lat. With it a point of the mosaic
  can be found in every frame that saw it, not only the one the mosaic used (see
  `read_frame_models` below). It is left out with `--strip-location`.
- **`mosaic_preview.png`:** a quick look in the files' own palette (or `--palette`), with
  one colour scale for the whole mosaic: `--range-min`/`--range-max`, or else the 2nd to
  98th percentile of its temperatures. Uncovered areas are transparent.
- **`mosaic_heights.tif`:** the surface the frames were projected onto (see Heights below),
  as one float32 band of altitudes in metres on the height cells, NaN where no frame covers
  a cell. Not written when the frames were projected onto the ground plane.

Per-frame files are written as well only for the `--output` options you give.

How it works:

- **Placement (direct georeferencing):** each frame is placed from its own metadata: RTK
  position, gimbal attitude, and a pinhole lens. The lens's focal length in pixels is the
  EXIF focal length divided by the sensor's pixel pitch from DJI's specs (12 µm for every
  supported camera but the XT S, 17 µm). For other cameras it falls back to the EXIF 35
  mm-equivalent focal length, which is rounded to a whole millimetre. Its rays are
  intersected with the surface (see Heights below): a horizontal ground plane at the median
  rangefinder target altitude (or the take-off altitude), plus the heights above it.
- **Alignment (on by default; `--no-mosaic-align` turns it off):**
  - The frames' placement is corrected from their overlaps. Image offsets between
    overlapping frames are measured block by block, by phase correlation of their thermal
    edges.
  - A robust least-squares fit then solves for one camera boresight for the whole flight
    (the angle between where the gimbal says the camera points and where it does) and a
    small shift and rotation per frame.
  - Each correction is held near zero by the RTK and attitude data.
  - Blocks whose offsets disagree, usually parallax from roofs and trees, are down-weighted.
  - The fitted corrections are in the per-frame table, and the boresight and before/after
    offsets are in the report.
- **Heights (`--mosaic-heights auto`, the default):**
  - A plane can't represent roofs and trees: a frame seeing a roof from the side places it
    off where the frame beside it does, so a straight roof bends into a crescent or steps at
    the seams.
  - While aligning, the mosaic is divided into cells (`--mosaic-height-cell`, 64 mosaic
    pixels, about 1.3 m on a 21 m flight) and a height above the plane is fitted per cell
    from the parallax between the overlapping frames, anchored to each frame's rangefinder
    reading and kept smooth between neighbouring cells.
  - Cells that no pair of frames sees are left on the plane. With fewer than 4 observed
    cells, or a surface that comes within 1 m of a camera, the whole mosaic falls back to the
    plane, and the report says why.
  - `--mosaic-heights plane` projects onto the plane, as before 1.3.0.
  - Heights need the alignment: with `--no-mosaic-align` the frames go onto the plane.
- **Surface model (`--mosaic-dsm PATH`, optional):**
  - A DSM you already have, e.g. ODM's `dsm.tif`, replaces the estimated heights. It is a
    GeoTIFF in WGS84 UTM or latitude/longitude, with altitudes in metres. The mosaic never
    needs one.
  - It is read without a TIFF library: uncompressed, Deflate or LZW, with or without a
    predictor, as ODM and GDAL write them (other compressions: rewrite it with
    `gdal_translate -co COMPRESS=DEFLATE`). A DSM that can't be used stops the run before
    any frame is converted.
  - It is resampled onto the height cells (the median of the DSM pixels in each). Cells it
    doesn't cover stay on the plane.
  - A DSM's altitudes are often on another datum than the drone's. They are shifted to
    match by the median difference between the rangefinder targets and the DSM, or by
    `--mosaic-dsm-offset METRES` when you know it.
  - The alignment then corrects only the frames' placement, not the surface. It also works
    with `--no-mosaic-align`.
- The report's `heights` entry says which surface was used, its range, the cell size, how
  many cells were observed, and how well it matches the rangefinder (`anchor_residual_m`).
- **Frame temperature offsets (`--mosaic-normalize`, off by default):**
  - Uncooled thermal cameras drift between their periodic shutter recalibrations: a frame's
    readings creep up, then snap back at each recalibration.
  - DJI's distance correction also stretches temperatures about each frame's embedded
    ambient temperature, which jitters from frame to frame.
  - The report always gives each frame's offset against its overlapping neighbours.
  - `--mosaic-normalize` measures every frame with the flight's median ambient and
    reflected temperature (unless you set them), then removes the offsets. This changes the
    measured values.
- **One frame per pixel:** each mosaic pixel takes the temperature of the single frame that
  saw that spot most directly, by nearest-neighbour sampling, so every value is one of the
  SDK's own measurements. Overlapping frames are never averaged.
- **Excluded frames:** frames whose gimbal pitch is above `--mosaic-max-pitch` (default
  -80°), i.e. oblique shots, are left out.
- **Pixel size:** `--mosaic-gsd` sets it; the default is the frames' ground sample
  distance.

Accuracy on a 21 m M4T flight over a house (42 nadir frames):

| | Median offset between overlapping frames | 90th percentile | Seam temperature difference (median) |
|---|---|---|---|
| Placement only | 38 cm | 71 cm | 0.44 °C |
| With alignment, on the plane | 6 cm | 60 cm | 0.37 °C |
| With alignment and heights (default) | 1 cm | 5 cm | 0.32 °C |
| With heights and `--mosaic-normalize` | 1 cm | 4 cm | 0.25 °C |

The estimated heights range from -0.5 to 5.9 m, which matches the roof. On a 327-frame M4T
flight with oblique frames, the plane bent the house's straight roof into a crescent; the
heights straighten it (median offset 0.4 cm, 90th percentile 1.5 cm).

Taking the focal length from the pixel pitch (1000 px) rather than the rounded 35 mm
equivalent (985 px) improved the unfitted placement (39 → 38 cm median, 77 → 71 cm p90) and
the large offsets left after alignment (p90 54 → 48 cm, worst 133 → 99 cm); the median after alignment
moved from 6.5 to 7.5 cm, within this one flight's roof parallax.

Offsets between frames are measured on bilinear renderings. Nearest-neighbour renderings
carry a staircase pattern tied to each frame's pixel grid, which biases the matching by
several centimetres. Two things were tried and left out:
- **Lens distortion:** not measurable from the overlaps.
- **Oblique frames as extra tie points:** made held-out misalignment worse, because their
  stretched, parallax-heavy views of the ground disagree with the nadir frames.

On that flight, the frame offsets followed a recalibration sawtooth, from -0.8 to +2.2 °C.
Removing them takes the median temperature difference between overlapping frames from
0.31 to 0.24 °C.

What the heights leave is mostly at roof edges and walls, which a cell of about a metre
can't follow, and lens distortion, which isn't modelled. The whole mosaic takes about 20 s
for 42 frames (14 s on the plane) and about 2.5 minutes for 327.

The frames must carry DJI XMP position and attitude (e.g. the M4T); the older cameras'
samples don't.

### Visual mosaic

DJI's thermal drones (M4T, M30T, M3T, H20T, H30T) can save the wide camera's visual
(RGB) image with each thermal one. `--mosaic-visual` mosaics those too, alongside the
thermal mosaic, so the two can be compared:

```
dji-thermal-cli process --input-dir /path/to/flight --output-dir out --mosaic --mosaic-visual
```

It writes, besides the thermal mosaic's files:

- **`mosaic_visual.tif`:** a GeoTIFF on the same UTM zone with four 8-bit bands: red,
  green, blue, and alpha (0 where no frame covers).
- **`mosaic_visual.json`:** the grid, a report, the table of visual frames (each with its
  thermal frame) and their `frame_models`, as the thermal mosaic's sidecar.
- **`mosaic_visual_preview.png`:** a quick look, at most 2048 pixels on its longer side.

How it works:

- **Pairing:** the JPEGs that aren't thermal R-JPEGs are read as visual frames when they
  carry a DJI camera pose and aren't a zoom camera's (`ImageSource`). Each is paired with
  the thermal frame exposed with it: DJI's `UTCAtExposure` within 0.25 s (on an M4T the two
  are 10 to 100 ms apart, the captures at least 0.5 s), or, when either has no exposure
  time, the same capture number in the same folder. A visual frame is used only when its
  thermal frame is in the thermal mosaic, so the oblique ones are left out with theirs.
- **Lens:** the focal length is the EXIF 35 mm equivalent (24 mm on the M4T's wide camera,
  2795 px at 4032×3024), over the JPEG's own size. Lens distortion isn't modelled.
- **Placement:** each visual frame is placed as its thermal frame is in the thermal mosaic:
  the same attitude with the alignment's boresight, the same shift and rotation, the same
  surface. Only its own position and lens are its own.
- **The cameras' offset:** the two cameras don't point exactly the same way. For each pair,
  the offset between where the visual and the thermal frame show the same ground is measured
  by phase correlation of their edges (gradient magnitudes), on the largest square inside
  the thermal footprint. A robust least-squares fit turns these offsets into one small
  rotation between the cameras (pitch and yaw), applied to every visual frame. With fewer
  than 5 pairs measured it isn't corrected, and the report says so.
- **One frame per pixel:** each pixel takes its colour from the frame that saw it most
  directly, interpolated bilinearly (colours are a picture, not measurements).
- **Pixel size:** the visual frames' own ground sample distance, or
  `--mosaic-visual-gsd METRES`. At two or more times the frames' own, the JPEGs are decoded
  at a half, a quarter or an eighth of their size, which is faster.

On a 327-frame M4T flight at 21 m (151 nadir pairs), the visual mosaic is 5764×6256 px at
0.6 cm/px (the thermal one: 1.6 cm/px), 86 MB, built in about a minute after the thermal
mosaic. 94 pairs were measured; the cameras' offset (pitch +0.25°, yaw −0.29°) moved the
visual frames by 10 cm (median), and the pairs then agree to 2.6 cm (median, 4.5 cm p90).
With `--mosaic-normalize` and the as-converted mosaic (the Python API's
`as_converted_dir`), the visual mosaic is written once, in the output directory.

## Supported cameras

| Camera | R-JPEG format |
|---|---|
| Zenmuse H20N, M3T, M30T, M3TD, H30T, M4T, Mavic 2 Enterprise Advanced | `iirp` |
| M30T (some captures) | `girp` |
| Zenmuse H20T, Zenmuse XT S | `dirp` |
| M3TD, M4T (newer firmware) | `hirp` |

- **Accuracy:** for every format, temperatures (float and the SDK's integer tenths), the raw
  sensor frame, pseudo-color images (every palette, brightness, color bar and isotherm
  setting, including the isotherm a `hirp` capture saved with the temperature alarm on starts
  with), accepted parameter ranges and refusals match the SDK exactly. The whole dirp API is
  available from Python (`dji_thermal_cli.tsdk`). The test suite checks this against DJI's
  own sample images, at their embedded parameters and at random ones.
- **Parameters:** the port applies the same per-camera rules as the SDK. On iirp files from
  before the v0300 header (everything except the H30T and M4T), the SDK ignores the ambient
  temperature you pass and uses the reflected temperature in its place. The H20T and XT S keep
  distance in whole metres and start from fixed defaults (5 m, 70 %, emissivity 1.0, 23 C)
  rather than the values stored in the file.
- **ND filter:** the infrared density-filter correction the SDK applies to captures taken
  through the H30T's ND filter is applied as well.
- **Refusals:** parameters the SDK refuses are refused here too, with the reason (for
  example, the H20T and XT S rebuild a color LUT whenever the parameters change and refuse
  some unusual combinations, such as very low emissivity with a hot reflected temperature).
  The file is reported as skipped and the rest of the batch goes on.
- **Not covered:** other R-JPEG formats, which are reported as unsupported (FLIR's
  radiometric JPEG, as DJI's Zenmuse XT2 writes, among them), and a few
  color settings where the SDK itself reads memory it never initialised, so its output isn't
  defined (they need a color bar or isotherm far outside the frame's temperatures). For
  those the temperature TIFF is still written and the PNG is skipped with a note.

## Built-in tables

The SDK's constant tables are built in (`tsdk/tables.py`): the ones DJI computes from a
formula (fixed-point exp/log/square-root tables, the BT.601 colour matrix, the table axes) are
generated from that formula, short tuning constants are written out, and the measured
calibration data and colour palettes are stored in `tsdk/data/dji_tsdk_v18.npz`. The test
suite checks every table byte for byte against the SDK v1.8's own.

## DJI's own SDK

If you have the [DJI Thermal SDK](https://www.dji.com/downloads/softwares/dji-thermal-sdk)
v1.8, `process --sdk DIR` and `list --sdk DIR` (in Python, `Settings(sdk=DIR)`,
`describe(..., sdk=DIR)`, `list_dataset(..., sdk=DIR)`) measure with DJI's own library
instead of the port; DIR is the unpacked SDK, as DJI's zip unpacks. The results are the same,
bit for bit (the test suite checks it on DJI's sample images).

- **DJI's library never runs in the tool's own process:** it is loaded in a child process,
  started on first use. If it crashes there or stops answering (two minutes for one call),
  the file it was working on is skipped with the reason (for example "DJI's Thermal SDK
  crashed (dirp_process): killed by SIGSEGV", or on Windows "ended by exception 0xC0000005
  (access violation)"), and the next file starts a new child.
- **Only SDK v1.8** (API revisions 19 and 20): another version is refused with the version it
  reports, since its data structures may differ. `check_sdk(DIR, samples)` says whether an SDK
  can be used, measuring the samples with it.
- DJI ships its library for Linux and Windows (x64 and x86), not macOS; `--sdk` is checked on
  Linux x64. Without `--sdk` the tool still needs no native library.

## Usage

Describe every thermal image in a dataset without writing anything:

```
dji-thermal-cli list --input-dir /path/to/mission
dji-thermal-cli list --input-dir /path/to/mission --format json --unit F
dji-thermal-cli list --input-dir /path/to/mission --format csv --exif all > dataset.csv
```

Files are found by their content, not their extension: a thermal image named `.bin` is
listed, a text file named `.jpg` is ignored, and ordinary JPEGs (e.g. the `_V` visual
images) are counted and skipped. Each file is described in groups:

| Group | Contents |
|---|---|
| `file` | path, name, size, warnings |
| `camera` | make, model, firmware, camera and drone serials, resolution, focal length (mm, 35 mm, pixels), capture time and UTC exposure time |
| `thermal` | SDK plugin and R-JPEG version, the embedded measurement parameters and their valid ranges, palette, colour bar, ND filter |
| `temperature` | min, max, mean, median and standard deviation of the measured image, with the file's own parameters (`--no-stats` skips the measurement) |
| `geospatial` | latitude, longitude, absolute and relative altitude, GPS/RTK status and accuracy, laser rangefinder reading and target |
| `attitude` | gimbal and aircraft yaw, pitch and roll |
| `exif` | with `--exif all`: every EXIF tag (IFD0, Exif, GPS, Interop) and XMP property, raw |

`--format table` (default) prints one summary line per file; `json` nests the groups; `csv`
writes one row per file with dotted column names (`attitude.gimbal_pitch`). Temperatures are
in `--unit C|F|K`, at the float32 precision the SDK measures in. A field the file doesn't
record is empty (`null`), and so is a parameter the camera doesn't have (the XT S and H20T
have no ambient temperature).

Check a dataset before processing it (nothing is written):

```
dji-thermal-cli check --input-dir /path/to/mission
dji-thermal-cli check --input-dir /path/to/mission --format json --unit F
```

Every file is identified by content and fully decoded, then compared with the rest of the
dataset. Each finding names the files and the option that addresses it:

- **Errors** (exit status 1): files named like images that aren't JPEGs, unreadable or
  damaged files (including files cut short), DJI thermal formats this tool can't measure,
  and byte-for-byte duplicates.
- **Warnings** (exit status 1 with `--strict`): measurement parameters that differ between
  files (emissivity, humidity, embedded distance), embedded ambient/reflected temperatures
  that drift (`--mosaic-normalize`), images whose minimum, maximum or mean temperature stands
  out (noting when they're oblique frames, e.g. with sky in view), mosaic frames with an
  unusual gimbal pitch or altitude, missing position or attitude, invalid rangefinder
  readings, mixed cameras, mixed GPS fix types or poor RTK accuracy, and damaged previews.
- **Notes:** ordinary JPEGs mixed in, oblique frames a mosaic leaves out
  (`--mosaic-max-pitch`), and an embedded distance that disagrees with the rangefinder.

Outliers are values more than 3.5 robust standard deviations (from the median absolute
deviation) from the dataset's median and also far enough apart to matter (5 °C for image
temperatures, 5° of pitch, 3 m of altitude or rangefinder distance). The table lists the
first five files of each finding (`--all-files` for all); JSON has everything.

Produce derivatives. Any of `--distance/--humidity/--emissivity/
--reflected-temp/--ambient-temp` you omit keeps that file's own embedded
value -- pass only the ones you want to change for the whole batch:

```
dji-thermal-cli process \
    --input-dir /path/to/mission \
    --output-dir /path/to/derivatives \
    --emissivity 0.95 --reflected-temp 68 --input-unit F \
    --range-min 59 --range-max 113
```

### Dataset-wide adjustments

Two adjustments act on top of the SDK's measurement, for every file of a run:

- **Gimbal override:** `--gimbal-pitch DEG` (and `--gimbal-yaw`, `--gimbal-roll`) replaces
  the recorded angle when you know it's wrong. It places the frames in the mosaic (and
  decides `--mosaic-max-pitch`), and it replaces `drone-dji:Gimbal*Degree` in every output's
  XMP. The recorded angle is kept as `dtc:OriginalGimbal*Degree`.
- **Out-of-range replacement, per pixel:** `--replace-above TEMP --replace-above-value
  VALUE` gives every pixel measuring above TEMP the value VALUE, or `nodata`; likewise
  `--replace-below`/`--replace-below-value`. Temperatures are in `--input-unit`. It applies to
  `temp_c`, `temp_f`, `temp_k`, `temp_ck`, `gray16` and the mosaic, where a `nodata` pixel
  leaves that spot to the next-best frame. `temp_dc`, `raw` and `pseudo` are the SDK's own
  output and stay unchanged; rendering is left to your workflow.

```
dji-thermal-cli process --input-dir mission --output-dir out --input-unit F \
    --replace-above 105 --replace-above-value 100 --gimbal-pitch -90
```

Every TIFF and PNG output records what was done in its XMP under the `dtc:` namespace
(`urn:dji-thermal-cli:xmp:1.0`): the measurement parameters actually used and where the
distance came from, any gimbal override with the original angle, and the replacement
thresholds, values and pixel counts. EXR and PNG outputs also carry the adjustments as text
attributes, and the mosaic report lists them under `adjustments`.

Object distance is the exception to "keep the embedded value". Cameras with a laser
rangefinder (e.g. the M4T) record the measured distance to the target in their DJI XMP
(`LRFStatus`, `LRFTargetDistance`), while the R-JPEG's embedded distance is often just the
5 m default. When a file has a valid reading (status `Normal`), that distance is used,
clamped to the range the camera accepts (with a note when clamped). This is
`--distance lrf`, the default; `--distance embedded` keeps each file's embedded distance,
and `--distance 12.5` sets one distance (metres) for every file.
The run summary says where each file's distance came from, and `list` shows the
rangefinder reading in its `lrf_m` column.

Files are processed in parallel by a pool of worker processes (`--jobs N`, default
`min(CPU count, 8)`; `--jobs 1` runs them one at a time in the same process, as does a daemonic process such as
a Celery worker's, which cannot start processes of its own); a mosaic
matches its frames' overlaps in as many threads. Files that
can't be converted are listed as `[skip, <reason>]` and the rest of the batch continues.

Run `dji-thermal-cli process --help` / `... list --help` for the full
flag list.

### R-JPEG copies

The `rjpeg` output is the source R-JPEG written again, so that DJI's own tools (DJI Thermal
Analysis Tool, the Thermal SDK) measure it as this tool does:

- **The measurement parameters** used (`--emissivity`, `--distance` or the rangefinder's,
  `--humidity`, `--reflected-temp`, `--ambient-temp`) are written where DJI's SDK reads them,
  when they differ from the file's own. The Zenmuse H20T and XT S store none (DJI's SDK starts
  them from fixed values), so their copies only keep the file's own parameters; asked for
  others, the output is skipped with the reason.
- **The picture** photo viewers show is the `pseudo` output (`--palette`, `--range-min/--range-max`),
  at DJI's size, and so is the preview image inside the file. `--rjpeg-picture original`
  keeps DJI's picture. DJI's tools colour the image from the measurements either way. The small
  EXIF thumbnail stays DJI's.
- **With `--strip-location`** the copy is private (see below), and still radiometric.
- **Not carried:** `--normalize` and `--replace-*` change the temperatures themselves, which an
  R-JPEG can't hold: `rjpeg` is then skipped with the reason (`temp_c` carries them).

Every change is made in place, at the same bytes DJI's SDK reads, and the tests check each
copy of DJI's sample images with the native SDK: it measures the copy exactly as it measures
the source with the same parameters set.

### Sharing without location

`process --strip-location` writes outputs that can be shared, e.g. online, without revealing
where or with which aircraft they were taken:

```
dji-thermal-cli process --input-dir flight/ --output-dir share/ --strip-location --mosaic
```

- Removed from every output's metadata: the EXIF GPS block, DJI's position, altitude, RTK
  and rangefinder-target fields, any other GPS property in the XMP, the drone, camera, body
  and sensor serial numbers, and the free-text fields where a name or an address could be
  (EXIF ImageDescription, Artist, Copyright, UserComment and the Windows XP fields; XMP
  `dc:creator`, `dc:rights`, `dc:description`, `dc:title`, `dc:subject`).
- An `rjpeg` copy also has the sensor's and the thermal module's serial numbers zeroed
  wherever else DJI wrote them (its debug and padding blocks, the calibration block).
- Kept: capture time, camera model, gimbal and flight attitude, the rangefinder distance,
  the measurement parameters, and the `dtc:` record, which notes `LocationRemoved="True"`.
- The mosaic (and the visual mosaic) is still placed using the GPS, then written as a plain
  TIFF without its georeference. Its JSON sidecar gives each frame's position in metres from the mosaic's
  top-left corner (`x_m`, `y_m`) instead of coordinates.

ODM and WebODM need the GPS to place frames, so give them ordinary outputs and make a
separate `--strip-location` copy for sharing. Source files are never modified.

## Python API

Everything the command line does is also a Python API, for programs that use it in-process
instead of running the command:

```python
import dji_thermal_cli as dtc

report = dtc.check("flight/")                       # errors and anomalies
for finding in report.findings:
    print(finding.severity, finding.message, finding.suggestion)

rows = dtc.list_dataset("flight/", unit="F").rows   # every file's groups, as dicts

settings = dtc.Settings(unit="F", emissivity=0.95, reflected_temp=68,
                        replace_above=105, replace_above_value=100)
m = dtc.measure("flight/DJI_0001_T.JPG", settings, unit="K")
m.temperature                                       # float32 kelvin array, nothing written

dtc.process("flight/", "out/", settings, outputs=["temp_c:tiff", "temp_f:exr"])
dtc.make_mosaic("flight/", "out/", settings, dtc.MosaicOptions(normalize=True, units=("C", "K")))
dtc.make_mosaic("flight/", "out-dsm/", settings, dtc.MosaicOptions(dsm="odm/dsm.tif"))
data = dtc.rjpeg_copy("flight/DJI_0001_T.JPG", settings, picture=rgb)   # an R-JPEG, as bytes
```

- **Settings** holds what the `process` options set: measurement parameters, `unit` (the unit
  of every temperature in it), colour range and palette, gimbal override, replacement,
  `strip_location` and `sdk` (DJI's own SDK, see [DJI's own SDK](#djis-own-sdk)).
  It validates itself and raises `SettingsError` naming the field.
- **Functions:** `scan`, `classify`, `describe`, `list_dataset`, `check`, `measure`, `process`,
  `make_mosaic`, and `rjpeg_copy` (one R-JPEG copy as bytes, with the settings' parameters,
  private with `strip_location`, and an RGB array as its picture or DJI's; `RJpegCopyError`
  says why one can't be made). The others return a result object (`ListResult`, `CheckResult`, `Measurement`,
  `ProcessReport`, `MosaicReport`).
- **MosaicOptions** holds what the `--mosaic-*` options set: `heights` (`"auto"` or
  `"plane"`), `height_cell`, `dsm`, `dsm_offset`, `visual` and `visual_gsd` among them.
- **The visual mosaic:** with `MosaicOptions(visual=True)`, `report.visual` is a
  `VisualReport` (its `mosaic` holds the colours as `rgb` and each pixel's frame as
  `source`, -1 where none; `file`, `sidecar`, `preview`), or None with
  `report.visual_note` saying why there is none. `read_frame_models` reads its sidecar too.
- **Both mosaics from one build:** `make_mosaic(..., as_converted_dir="out-raw/")`, with
  `normalize=True`, also writes the mosaic without the normalization there, laid out and
  checked once for both (`report.as_converted` is its report). It is laid out from the
  normalized frames, so it can differ slightly from a separate `normalize=False` mosaic.
- **Every frame that saw a point:** `read_frame_models("out/mosaic_temp_c.json")` reads a
  mosaic's frame models (None for a sidecar written before 1.5.0, or without location).
  `models.pixels(lon, lat)` gives each frame's pixel (column, row) for points, NaN where the
  frame did not see them, and `models.view_angles(lon, lat)` the angle it saw them at;
  `models.covering(lon, lat)` lists the frames that saw a point, and `models.meeting(outline)`
  those whose footprint meets an area (a polygon of lon/lat vertices). Frames are numbered as
  in the sidecar's `frames` table and the mosaic's source band.

  ```python
  models = dtc.read_frame_models("out/mosaic_temp_c.json")
  for k in models.covering(-88.0767, 42.1034):
      print(models.files[k], models.pixels(-88.0767, 42.1034)[k, 0])
  ```
- **Errors:** a missing path raises `FileNotFoundError`, a mosaic that can't be built raises
  `MosaicError`, and `describe` of a non-thermal file raises `FileProblem`. Within a dataset a
  bad file never raises: it is reported in the result and the rest carries on.
- **Progress:** pass `progress=callback` to receive an `Event` per file as the work happens.
  A mosaic (`make_mosaic`, or `process(normalize=True)`) also sends `"mosaic"` events, one
  stage at a time, each counted in its own unit:

  ```python
  def progress(event):
      if event.kind == "mosaic":
          d = event.data   # {"stage": "align", "step": 2, "steps": 5, "done": 812, "total": 1840,
                           #  "unit": "pairs", "round": 1, "rounds": 3}
          print(f"step {d['step']}/{d['steps']} {d['stage']}: {d['done']}/{d['total']} {d['unit']}")
  ```

  The stages, in order and each only when it runs: `ambient` (files), `frames` (files),
  `align` (per round: `pairs`, then `solve steps`), `composite` (frames), `check` (pairs),
  with `visual=True` `visual_frames` (files), `visual_offset` (pairs) and `visual_composite`
  (frames), then `write` (files). Each starts at `done = 0` and ends at its total. There is no overall
  percentage, as the stages' share of the time varies with the flight.
- **Parallelism:** `jobs` as on the command line (`1` runs in the calling process, as does
  any job in a daemonic process, such as a Celery worker's; a mosaic still aligns in threads).
- The API never prints or exits. The command line is a thin layer over it and writes
  byte-identical files.

## Tests

```
uv run pytest
uv run ruff check src tests scripts
```

The suite runs with nothing configured. Sample images, a flight's images and the native DJI
Thermal SDK are each optional (environment variables `DJI_THERMAL_SAMPLES_DIR`,
`DJI_THERMAL_FLIGHT_DIR` and `DJI_THERMAL_SDK_DIR`) and add the tests that need them; with
the SDK, the port is compared bit for bit with DJI's own libraries (Linux x64). Without them
those tests are skipped, the built-in tables are still checked against the SDK's recorded
SHA-256 digests, and, with the sample images, the port is checked against digests of every
result that was compared with the SDK. The tool loads DJI's libraries only when asked to
(`--sdk`), and then in a child process. See
[CONTRIBUTING.md](https://github.com/WinuxNomacs/dji-thermal-cli/blob/main/CONTRIBUTING.md).

## License

GNU Affero General Public License v3.0 (`AGPL-3.0-only`), the same license as
[WebODM](https://github.com/WebODM/WebODM) and [ODX](https://github.com/WebODM/ODX). See `LICENSE`, and `LICENSES/` for the third-party notices.

This tool is an independent reimplementation of the DJI Thermal SDK's measurement and
image-processing algorithms, verified against the SDK's outputs. No DJI code or binaries are
included.
`tsdk/data/dji_tsdk_v18.npz` holds the SDK v1.8's calibration tables and colour palettes. The
`powf` implementation in `tsdk/powf32.py` follows ARM optimized-routines (MIT OR Apache-2.0
WITH LLVM-exception). The test suite loads the DJI SDK binaries from your own SDK copy, under
DJI's own EULA; `tests/native_sdk.py` transcribes struct layouts from DJI's MIT-licensed
`dirp_api.h` and carries its notice.
