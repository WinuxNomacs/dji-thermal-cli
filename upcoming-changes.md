# Upcoming changes

Deferred work, tracked here so it isn't lost between sessions.

## Phases (current and upcoming)

The order of work, top first. Each phase is planned with the user before it is built; the details are in the
sections below.

**Up next**: nothing planned yet. The analyzer's side of RGB-only flights is planned in dji-thermal-ui.

**Then**: nothing planned yet.

**Done:** P1 (performance, 2026-11-03), S1 (the eave smear, 2026-11-03: finer height cells for RGB-only mosaics), V1
(RGB-only mosaics in the API and on the command line, 2026-11-03), and their release as 1.13.0 (2026-11-03).

All of it is built on the branch `rgb-mosaic-trial` (decided 2026-11-03), which already holds the trial:
`visual.build_alone` (d753692), and the lens distortion and rangefinder prior (8e5acef), both off unless asked for, so
thermal mosaics are unchanged by them.

## The release of P1, S1 and V1 (done: 1.13.0)

**Decided (2026-11-03):**
- **The version is 1.13.0**, a minor bump: new features (RGB-only mosaics, `lens_k1`) and faster mosaics, with
  nothing removed. `height_cell` defaulting to None gives the same cells as before for thermal mosaics.
- **A script without `if __name__ == "__main__":`** (the user's choice: fall back): where workers would import it and
  run it again (spawn, forkserver), the work runs in the calling process with a `RuntimeWarning`
  (`mosaic.processes_unavailable`, used by `_measure_blocks` and `api._run_all`). The script's module-level line
  that is running is looked up in its source, inside or outside the guard. Before, such a script crashed with
  forkserver on Linux's Python 3.14, and with spawn on Windows and macOS.
- **The worker processes are checked on Windows** before the release (the user gave access). macOS uses spawn, as
  Windows does, and the same shared memory API, so Windows covers it; nothing is run on the Mac.

**Steps:** the fallback on `rgb-mosaic-trial`; the full suite; Windows; merge `rgb-mosaic-trial` into `main`;
`release/1.13.0` (version, lock file, changelog), its release checks, and the merge into `main`. The user pushes and
tags.

## The RGB-only trial (built, on `rgb-mosaic-trial`)

Requested 2026-11-03: mosaics of RGB-only flights, for comparison and analysis in the DJI Thermal Analyzer. The user
saw the trial's mosaics and decided they become a feature.

- **`visual.build_alone`:** frames steeper than the pitch limit, decoded at most 1280 px wide (`WORK_PX`), laid out
  by `mosaic._layout` with `distortion` and `rangefinder_prior` on, checked, then composited at full size.
- **`distortion`:** one radial k1 for every frame, solved with the boresight (0.047 on the M4T's wide camera).
- **`rangefinder_prior`:** height cells the images barely see are pulled toward the median of the 5 nearest
  rangefinder heights, not toward the ground plane (which on the test flight is roof height). Tried on thermal: no
  clear gain, so it stays off there.
- **Results on `m4t-rgb`** (151 frames, 6 mm pixels, about 13 minutes):
  - frames agree to a median of 1.4 cm (p90 4.9 cm);
  - held-out rangefinder error 0.38 m.
- **Not WebODM's DSM:** the user doesn't want to depend on it, as it is specific to WebODM.
- **Tests:** six in `tests/test_visual.py`.

## P1: performance (done)

Requested 2026-11-03, before S1: S1 needs many full runs, and its fix will likely add work.

**Where the time goes** (the RGB run, 802 s):
- block matching between overlapping frames (`_measure_blocks`, 4 alignment rounds of about 10,000 pairs): 600 s;
- the misalignment check (the same matching once more): 100 s;
- the full-size composite: 70 s;
- the solve: 25 s.

Inside `_measure_blocks`, numpy's double-precision FFTs take about half the time. Threads barely help: on 40
frames, 1 thread takes 12.5 s, 4 threads 5.9 s and 20 threads 5.6 s, with about 6 cores busy even on one thread.
Each pair cuts its blocks from its own overlap, so a frame's same ground is transformed again for every neighbour.

**Plan:**
1. **A fixed accuracy benchmark first**, on the RGB and thermal test flights. It records:
   - the frames' agreement (median, p90);
   - the held-out rangefinder error by ring;
   - the frames' shifts and rotations, and k1;
   - the thermal mosaic itself, compared to the pixel with today's.
2. **Speed with results unchanged:**
   - the pairs matched in worker processes rather than threads;
   - whatever competes for the cores found and fixed;
   - the composite in parallel.
3. **Speed that changes results slightly,** each kept only if the benchmark holds:
   - single-precision FFTs;
   - blocks on a fixed grid per frame, each transformed once and reused by all its pairs;
   - pairs that barely touch skipped;
   - later rounds re-measuring only where the solution moved.

numpy only, no new dependency.

**Decided (2026-11-03):** tier 2 leaves the output identical to what this branch makes today for the same flight and
settings; tier 3 may change thermal mosaics within a tolerance: the frames agreeing no worse, and no mosaic pixel
moving more than a fraction of a thermal pixel. The same holds for RGB-only mosaics, in their own pixels.

**Done (2026-11-03), tier 2 only:** 8f2213d and ca5f6d8 on `rgb-mosaic-trial`, in `mosaic.py`:
- pairs matched in worker processes (`_match_in_processes`: the frames in shared memory, each rendered once), 8
  blocks at a time (`MATCH_BATCH`), and in this process when it is daemonic or jobs is 1;
- `Heights.at` interpolating without BLAS;
- glibc keeping freed memory in the workers (`_keep_freed_memory`).

The mosaics are identical to the pixel on both test flights. Thermal went from 96 s to 35 s, and an RGB matching
round from 109 s to 28 s, on 8 workers; the laptop's timings vary by up to 2× under a long load. Tests:
`test_build_in_several_processes_gives_the_same_mosaic` and `test_a_daemonic_process_matches_the_overlaps_itself` in
`tests/test_mosaic.py`. The checks it left for the release (Windows, the `__main__` guard) are in "The release of
P1, S1 and V1 (done: 1.13.0)".

**Tried and dropped:**
- the RGB composite in processes, by bands of rows: 39 s against 30 s with threads, since each band decodes its frames
  again;
- glibc's settings in the calling process: no gain for the composite.

**Tier 3 not built:** the user found the speed enough for now (2026-11-03). What was learnt, in case it comes back:
- Today's alignment, with its blocks only shifted by 16 or 32 px, moves mosaic content by a median of 0.5 thermal
  pixels and 6 at the 99th percentile. That is the method's own spread, so no result-changing step can keep every
  pixel within a fraction of a pixel. A tolerance would need to be restated as: frames agreeing no worse, held-out
  rangefinder error no worse, content moving less than that spread.
- Sparse early rounds (a quarter of the blocks before the last round) took thermal from 96 s to 24 s, moving content
  by a median of 0.39 px and 6 px at the 99th percentile.
- The composite's remaining cost is projecting every full-size pixel; a coarse grid interpolated would cut it.

## S1: the eave smear (done)

At the house's eave the surface drops about 5 m within one 1.4 m height cell, and the RGB-only mosaic smears there.

**Why:** `Heights.at` blends the cells' heights linearly, so a step is spread over a whole cell: the patio within about
a metre of the eave is projected as if up to 5 m higher, and moves by (distance from nadir) × (height error) / (flying
height), about 1.2 m at 10 m from nadir. Each pixel already comes from the frame nearest nadir, so only the heights can
fix it.

**Plan (2026-11-03):**
1. After the alignment, find the cells where the surface steps sharply next to a neighbour.
2. Inside them, decide each pixel between the upper and the lower surface by where the overlapping frames agree when
   projected at each height, giving a fine height map along the edges only.
3. Everywhere else, the heights and the composite stay as they are.

**Decided (2026-11-03):**
- RGB-only mosaics only (`build_alone`); thermal mosaics and the Visual tab's mosaic unchanged.
- Up to about +10–20% on an RGB-only build's time.
- Done when it is better than now and works for any flight, repeatably, not only for this one:
  - no threshold tuned to this flight: each follows from the flight's own cell size, height and height noise;
  - the same input gives the same mosaic;
  - a synthetic test (a box building on flat ground, rendered from known cameras) shows the edge sharp;
  - on `m4t-rgb`: the eave and garage crops better, the benchmark elsewhere no worse.

**First attempt (2026-11-03), on the branch `s1-steps-wip` (5222a39), not merged:**
- Each step pixel chooses between the blend and the lowest or highest of the 4 × 4 cells around it, by the frames'
  agreement with the frame nearest nadir, smoothed by semi-global matching. A second pass uses only the frames that
  see the pixel past the first pass's surface (depth buffers), and so does the composite.
- On a synthetic box (5 m high, 8 m wide, sharp edges) it works: pixels more than 1 m off in the edge band go from 16%
  to 1.3%.
- On `m4t-rgb` the crops get worse (doubled objects, roof spread over the ground, holes where no frame is judged to see
  a pixel), and it takes up to 194 s.
- What a height sweep showed: on open ground and on the roof the frames agree best at the right height (lawn −5.4 m,
  as now; roof 2.4 m); beside walls and in tall plants they don't, and the cells around a step are often too blurred
  to hold the step's true heights. The 4 × 4 test flags 36–70% of the mosaic as steps.

**Done (2026-11-03), the user's option 1:** `build_alone` estimates its heights on cells of 32 work pixels
(`visual.ALONE_HEIGHT_CELL_PX`), half the thermal mosaics' 64; thermal mosaics and the Visual tab's are unchanged, and
`height_cell` still sets it.
- On `m4t-rgb` (cells 1.41 → 0.70 m): the band bent beside the eave about half as wide, the patio's paving nearly
  straight; the garage, tree and roof-ridge crops no worse (the tree's and ridge's seams gone). Cells seen by two
  frames 380 → 1213; the frames' misalignment 4 → 3 mm median, 17 → 12 mm p90; rangefinder misfit 0.262 → 0.240 m.
- On synthetic flights the gain depends on overlap: with 36 frames the frames agree better (16 → 12 mm median), with
  9 slightly worse; a box's edge isn't sharper on either. So the smear is smaller, not gone: placing the steps
  themselves needs dense reconstruction near them (option 2: many heights per pixel), beyond the time agreed.
- Test: `test_a_flight_of_its_own_estimates_its_heights_on_cells_half_the_thermal_mosaics` (`tests/test_visual.py`).
- The first attempt stays on `s1-steps-wip`, unmerged.

## V1: RGB-only mosaics as a feature (done)

`visual.build_alone` reachable from the API and the command line.

**Decided (2026-11-03):** automatic, plus a flag. `--mosaic` on a flight with no thermal frames makes the RGB-only
mosaic by itself; `--mosaic-visual-only` makes it on a flight with thermal frames too, in place of the thermal
mosaic. The API has the same two ways.

**Design:**
- **API:** `MosaicOptions.visual_only`. `make_mosaic` makes the RGB-only mosaic when it is set, or when the flight has
  no thermal frames but has visual ones (today: "No frames to mosaic."). Its `MosaicReport` then has `mosaic` and
  `preview` None, no `files`, and `visual` set, as the paired visual mosaic's; the counts as for a thermal mosaic.
- **Options that apply:** `max_pitch`, `visual_gsd` (its pixel size), `heights`, `height_cell`, the location removed
  (`Settings.strip_location`), compression, jobs, progress. `height_cell` defaults to None: 64 for thermal mosaics, 32
  for RGB-only (S1). The thermal-only ones (units, palette, normalize, gsd) are not used; `dsm`, `alignment`,
  `as_converted_dir` and `align=False` with `visual_only` are refused (SettingsError), and on a flight without
  thermal frames give MosaicError saying why.
- **Files:** the same as the paired visual mosaic's (`mosaic_visual.tif`, its JSON sidecar and
  `mosaic_visual_preview.png`); the sidecar's `laid_out` says how it was laid out.
- **Progress:** "visual_frames" (files read), then build_alone's "visual_decode" (renamed from "visual_frames", not
  yet released), "align", "check" and "visual_composite", then "write".
- **k1:** written in the sidecar's frame models (each frame's `lens_k1`, read as 0 when absent), so
  `read_frame_models(...).pixels` is exact for an RGB-only mosaic. (The backlog said `visual_pairs.json`: that is the
  analyzer's file of thermal-to-visual pairs, which an RGB-only mosaic doesn't have, and the paired mosaic has no k1.)
- **Command line:** `--mosaic-visual-only`, and `--mosaic` falling back to it; the summary printed says the RGB
  mosaic was laid out by its own frames (frames used, alignment, heights, misalignment).
- The README and the changelog.

**Done (2026-11-03)** as designed, on `rgb-mosaic-trial`: `api.py` (`MosaicOptions.visual_only`, `_visual_only`,
`_read_visual`, `_write_visual`), `cli.py` (`--mosaic-visual-only`, `_print_visual_alone`), `mosaic.py` (`lens_k1`),
the README and the changelog. On `m4t-rgb`, `process --mosaic` makes it in 7 minutes, with the S1 figures (0.3 cm
median, 1.2 cm p90). Tests in `tests/test_visual.py`: the automatic case, `visual_only`'s height cells, the refused
options, k1 through `read_frame_models`, and the command line's two cases. The progress steps start again at 1 when
the RGB-only mosaic follows the thermal frames' reading (the automatic case).

The analyzer's side (RGB-only flights, a dataset without temperatures, the viewer and compare page) is to be planned in
dji-thermal-ui's own `upcoming-changes.md` once this is released.
