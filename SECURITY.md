# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub's
[private vulnerability reporting](https://github.com/WinuxNomacs/dji-thermal-cli/security/advisories/new),
not in a public issue. Include the version (`dji-thermal-cli --version`), what you ran, and,
if you can, a file that triggers it.

You should get a reply within a week. Once a fix is released, the advisory is published
with credit to you unless you prefer otherwise.

## Scope

dji-thermal-cli parses image files it is given (JPEG, EXIF, XMP and DJI's binary blocks).
Problems such as a crafted file that makes it hang, use unbounded memory, write outside the
output directory or run code are in scope. A file that is merely rejected or reported as
damaged is the intended behaviour.

Only the latest release receives fixes.

## Verifying a release

Releases are built and published from GitHub Actions with PyPI Trusted Publishing, and
each file carries a digital attestation (PEP 740) linking it to the workflow run that
built it. PyPI shows these under each file's "Provenance".
