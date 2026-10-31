# Releasing

Releases are published by `.github/workflows/release.yml` when a `vX.Y.Z` tag is pushed:
CI (`.github/workflows/ci.yml`: lint, the tests on Linux, Windows and macOS, the build and
its smoke test), then the build, the distribution check, TestPyPI, then PyPI. Nothing else
runs GitHub Actions: pushing a branch runs nothing, nor do Dependabot's pull requests; CI
also runs when started by hand (Actions > CI > Run workflow) and on pull requests from forks. No API token is
stored anywhere: PyPI trusts the workflow itself (Trusted Publishing). Only the maintainer
can push a `v*` tag (a tag rule), so pushing the tag is the release decision: nothing waits
for a further approval.

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
   - `pypi`: deployment tags `v*`. (A required reviewer would pause the run before PyPI
     for an approval; it is left off, because only you can push `v*` tags.)
4. **Repository settings:** enable private vulnerability reporting (Security > Advisories),
   protect `main`, and protect tags `v*` (Settings > Rules) so only you can push them: this
   tag rule is what keeps anyone else from publishing.

## Each release

Versions follow PEP 440. A tag is pushed only for a version that is meant to be published,
once the local checks below pass: each tag runs the whole pipeline. When the workflows
themselves changed, a release candidate `X.Y.ZrcN` (tag `vX.Y.ZrcN`) can go first to try them
(PyPI marks it as a pre-release, so `pip install` skips it unless asked). A version number
is used once, on TestPyPI and PyPI alike.

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
4. Push `main` (this runs nothing), then tag and push the tag:
   `git tag -s vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`. This runs CI on all three
   systems and, if it passes, publishes: TestPyPI, then PyPI, without stopping. If CI fails,
   nothing is published: fix it and release a new version (the tag's version is used once
   only if it reached TestPyPI; otherwise delete the tag and push it again).
5. Watch the Release workflow, then check the release installs:
   `uvx dji-thermal-cli@X.Y.Z --version`.
6. Create the GitHub release from the tag with the changelog notes.

A version can never be uploaded to PyPI twice. If something is wrong after publishing,
yank the release on PyPI and publish a new version.
