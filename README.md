# dji-thermal-cli

Batch-process DJI R-JPEG thermal images (temperature TIFF + pseudo-color PNG)
on top of the DJI Thermal SDK v1.8.

This does **not** rewrite the original R-JPEG: DJI bakes its environmental
parameters into a binary APP4 block at capture time that isn't writable by
the SDK. Every command below only reads originals and/or writes new
derivative files.

## Setup

1. Use static sdk version in libs or download the DJI Thermal SDK yourself from
   https://www.dji.com/downloads/softwares/dji-thermal-sdk and unpack it so
   its contents land under `lib/` here, i.e. `lib/tsdk-core/lib/<windows|linux>/...`
   should contain `libdirp.dll`/`libdirp.so` and its companion libraries.
   Only `lib/tsdk-core/api/*.h` (already checked in) is MIT-licensed per the
   SDK's own `License.txt`; the compiled binaries are covered by DJI's
   separate [SDK EULA](https://developer.dji.com/policies/eula/), which is
   why they aren't committed to this repo -- `lib/` beyond the headers is
   gitignored.
2. ```
   uv sync
   ```

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

Run `uv run dji-thermal-cli process --help` / `... list --help` for the full
flag list.
