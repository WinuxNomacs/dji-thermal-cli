# Contributing

## Setup

```
git clone https://github.com/WinuxNomacs/dji-thermal-cli
cd dji-thermal-cli
uv sync
uv run pytest
uv run ruff check src tests scripts
```

That is all you need to build and test new functionality. The DJI Thermal SDK is optional.

## What the tests need

The suite runs with nothing configured. Three optional settings, all environment variables,
add tests that need real data or DJI's native SDK:

| Variable | What it points at | Adds |
|---|---|---|
| `DJI_THERMAL_SAMPLES_DIR` | sample R-JPEGs, one folder per camera (`<dir>/M4T/DJI_0001_R.JPG`, ...), laid out like the SDK's `dataset/` | tests on real files from each camera |
| `DJI_THERMAL_FLIGHT_DIR` | the images of one flight: overlapping nadir thermal frames with GPS, as the drone wrote them (visual images may be mixed in, in any subfolders) | mosaic, normalization, `check` and WebODM tests |
| `DJI_THERMAL_SDK_DIR` | an unpacked [DJI Thermal SDK](https://www.dji.com/downloads/softwares/dji-thermal-sdk) | the oracle: the port compared with DJI's own Linux x64 libraries, on Linux. Its `dataset/` also supplies the samples when `DJI_THERMAL_SAMPLES_DIR` isn't set |

The pytest header shows what is configured:

```
samples: 49 file(s) in /data/dji-thermal-sdk/dataset
flight: 217 thermal frame(s) in /data/flights/m4t
native SDK: /data/dji-thermal-sdk (API revisions 19, 20)
```

Each test that needs one of these carries a marker, `samples`, `flight` or `native`, and is
skipped without it, with the variable to set as the reason. To leave out the native
comparisons when the SDK is configured: `uv run pytest -m "not native"`.

The native comparisons run on Linux only. The port reproduces DJI's Linux build, and DJI's
Windows build differs from it where results aren't fully defined (H20T float rounding,
out-of-range settings). On Windows and macOS, `test_port_digest.py` instead checks that the
port reproduces, bit for bit, digests of its results recorded on Linux, so with the samples
configured every system is held to the same results.

`test_oracle_digest.py` keeps the native comparisons' checks without the native SDK: every
call sequence they replay (`tests/oracle_cases.py`: parameter changes, palettes, colour bar,
isotherm, brightness, ND filter, gain modes, the hirp alarm, refusals, and each flight frame)
is replayed on the port alone and compared with digests recorded right after the native
comparisons passed on the same sequences.

Expectations in the flight tests are derived from the flight itself (frame counts, gimbal
pitches, its own temperatures), so any flight that `check` reports no errors for will do.

## Writing tests

- **New functionality** that doesn't touch the SDK port needs no native SDK. Prefer
  synthetic data (most of `test_checks.py`, `test_mosaic.py` and `test_webodm.py` build
  their inputs in code); those tests run everywhere, including CI.
- **Real files:** use the helpers in `tests/conftest.py` and mark the test:
  ```python
  @pytest.mark.samples
  def test_something():
      src = sample("M4T")              # skips if there is no M4T sample

  @pytest.mark.flight
  def test_on_a_flight():
      frames = flight_files()          # the flight's thermal R-JPEGs
      nadir, oblique = flight_counts()
  ```
  `--strict-markers` is on, so a misspelled marker fails.
- **Native comparisons** live in the `*_vs_native.py` modules and `test_cli.py`. The native
  SDK always runs in a subprocess (`native_measure`, `native_process`, `native_sequence` in
  `conftest.py`), because it can crash. Sequences go in `tests/oracle_cases.py`, so the
  oracle digests cover them too; rerecord them after adding one (below).
- Warnings are errors: close files you open (`with Image.open(...) as im:`).

## The SDK port

`src/dji_thermal_cli/tsdk` reproduces the DJI Thermal SDK v1.8 bit for bit. Its behaviour
is never changed, except where it is found to differ from the SDK's: new functions may be
added, and everything else belongs in the CLI layer (`dji_thermal_cli` outside `tsdk`). The native tests are what holds it to the SDK, so run
them when you touch `tsdk`.

## Following a new DJI Thermal SDK release

1. Point `DJI_THERMAL_SDK_DIR` at the new release. If its API revision differs, the native
   tests are skipped with a reason that says so.
2. Check `dirp_api.h` against the ctypes bindings in `tests/native_sdk.py`, update them,
   and set `SDK_VERSION` and `SUPPORTED_API_VERSIONS` there.
3. Run the native tests (`DJI_THERMAL_SDK_ALLOW_UNKNOWN_VERSION=1` runs them before step 2
   is done) and look at every difference: each is a change in DJI's results that the port
   would have to follow, as new functions next to the existing ones.
4. `tests/sdk_tables.py` records the digest of every built-in table; a release with new
   tables needs new entries.
5. After any intended change to the port's results, with the native tests passing on Linux,
   rerecord the cross-system digests: `uv run python -m tests.test_port_digest`, and, with
   the samples and the flight configured, the oracle digests:
   `uv run --env-file .env python -m tests.test_oracle_digest` (it runs the native
   comparisons first and records nothing unless they pass).

## Pull requests

Work on a branch, keep `uv run pytest` and `uv run ruff check src tests scripts` passing,
and add a line under `[Unreleased]` in `CHANGELOG.md` for anything a user would notice.
