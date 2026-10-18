# Releasing

Releases are published by `.github/workflows/release.yml` when a `vX.Y.Z` tag is pushed:
tests, build, the distribution check, TestPyPI, then PyPI after approval. No API token is
stored anywhere: PyPI trusts the workflow itself (Trusted Publishing).

## One-time setup

0. **The README image for PyPI.** While the repository is private, the README shows its
   animation by relative path (`img/dji-thermal-cli-sub5.gif`), which GitHub resolves but
   PyPI can't. Once the repository is public, before the first tag, change the `src` to
   `https://raw.githubusercontent.com/WinuxNomacs/dji-thermal-cli/main/img/dji-thermal-cli-sub5.gif`
   and check it loads: `curl -sI <that URL> | head -1` must say `200` (it says 404 while
   the repository is private). Images over 5 MB may not show on GitHub or PyPI.
1. **PyPI and TestPyPI accounts** with two-factor authentication (required by PyPI).
2. **Trusted publishers.** On pypi.org (Account settings > Publishing > Add a new pending
   publisher) and on test.pypi.org, add:
   - PyPI project name: `dji-thermal-cli`
   - Owner / repository: `WinuxNomacs` / `dji-thermal-cli`
   - Workflow name: `release.yml`
   - Environment name: `pypi` on PyPI, `testpypi` on TestPyPI

   A pending publisher also reserves the name until the first upload.
3. **GitHub environments** (repository Settings > Environments):
   - `testpypi`: deployment tags `v*`.
   - `pypi`: deployment tags `v*` and a required reviewer (you), so nothing reaches PyPI
     without an explicit approval after TestPyPI succeeds.
4. **Repository settings:** enable private vulnerability reporting (Security > Advisories),
   protect `main`, and protect tags `v*` (Settings > Rules) so only you can push them.

## Each release

1. On a branch: set `version` in `pyproject.toml`, run `uv lock`, and move the
   `[Unreleased]` notes in `CHANGELOG.md` under the new version. Merge to `main`.
2. Check locally:
   ```
   uv run pytest
   rm -rf dist && uv build --no-sources
   uv run --no-project python scripts/check_dist.py dist
   uvx twine@7.0.0 check --strict dist/*
   ```
3. On macOS, from a fresh clone of `main` (CI covers macOS only without data; this runs the
   sample and flight tests too, DJI ships no macOS libraries for the native ones):
   ```
   git clone <repository> && cd dji-thermal-cli
   uv sync --locked
   DJI_THERMAL_SDK_DIR=... DJI_THERMAL_FLIGHT_DIR=... uv run pytest
   uv run dji-thermal-cli process --input-dir <flight> --output-dir /tmp/out --mosaic
   ```
   Both pass, the outputs open in your usual viewer, and Ctrl+C during a `process` run stops
   it with "Interrupted.". On Linux, run the same with the native comparisons too (they run
   on Linux only). On Windows, run the same without them (`set` the variables in `cmd`, or
   `$env:` in PowerShell).
4. Tag and push: `git tag -s vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`.
5. Watch the Release workflow; check the TestPyPI page and
   `uv tool run --index-url https://test.pypi.org/simple/ --index-strategy unsafe-best-match dji-thermal-cli --version`
   if you want, then approve the `pypi` deployment.
6. Create the GitHub release from the tag with the changelog notes.

A version can never be uploaded to PyPI twice. If something is wrong after publishing,
yank the release on PyPI and publish a new version.
