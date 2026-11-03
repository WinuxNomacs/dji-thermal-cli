# Upcoming changes

Deferred work, tracked here so it isn't lost between sessions.

## Phases (current and upcoming)

The order of work, top first. Each phase is planned with the user before it is built; the details are in the
sections below.

**Up next**:
1. **P1, performance:** the mosaic pipeline faster, accuracy held by a fixed benchmark.

**Then**:
2. **S1, the eave smear:** RGB-only mosaics smear where the surface drops several metres within one height cell.
3. **V1, RGB-only mosaics as a feature:** `visual.build_alone` reachable from the API and the command line.
4. **The release** of P1, S1 and V1.

All of it is built on the branch `rgb-mosaic-trial` (decided 2026-11-03), which already holds the trial:
`visual.build_alone` (d753692), and the lens distortion and rangefinder prior (8e5acef), both off unless asked for, so
thermal mosaics are unchanged by them.

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

## P1: performance

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

**Progress (2026-11-03):**
- **Tier 2 built** (8f2213d, ca5f6d8), the mosaics identical to the pixel on both test flights:
  - pairs matched in worker processes, 8 blocks at a time;
  - heights interpolated without BLAS;
  - glibc keeping freed memory in the workers.
  
  The thermal mosaic went from 96 s to 35 s, and an RGB matching round from 109 s to 28 s, on 8 workers. The
  laptop's timings vary by up to 2× under a long load, so whole RGB runs aren't comparable yet.
- **Tried and dropped:**
  - the RGB composite in processes, by bands of rows: 39 s against 30 s with threads, since each band decodes its
    frames again;
  - glibc's settings in the calling process: no gain for the composite.
- **Tier 3 is waiting on the tolerance.** Today's alignment, with its blocks only shifted by 16 or 32 px, moves
  mosaic content by a median of 0.5 thermal pixels and 6 at the 99th percentile. That is the method's own spread,
  so no result-changing step can keep every pixel within a fraction of a pixel.
  
  Proposed instead: frames agreeing no worse, held-out rangefinder error no worse, and content moving less than
  that spread. Sparse early rounds (a quarter of the blocks before the last round) take thermal from 96 s to 24 s
  and move content by a median of 0.39 px and 6 px at the 99th percentile. They're kept as a patch until this is
  decided.
- **Next tier 3 candidate:** the composite's and renders' projection computed on a coarse grid and interpolated.

## S1: the eave smear

At the house's eave the surface drops about 5 m within one 1.4 m height cell, and the RGB-only mosaic smears there.
To be planned after P1.

## V1: RGB-only mosaics as a feature

To be planned after S1. Known so far:
- an option in the API and a command-line flag;
- k1 in the visual frame models (`visual_pairs.json`), so that a pixel lookup in the analyzer is exact;
- the README, docs and changelog.

The analyzer's side (RGB-only flights, a dataset without temperatures, the viewer and compare page) is planned in
dji-thermal-ui's own `upcoming-changes.md`.
