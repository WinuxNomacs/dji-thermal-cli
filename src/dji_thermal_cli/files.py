"""Writing output files safely: complete or not at all."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def replaced_when_done(path: Path) -> Iterator[Path]:
    """Yield a temporary path beside `path` to write to, then rename it to `path`, so an
    interrupted or failed write never leaves a half-written file under the real name. The
    temporary name keeps the extension (`.partial-<name>`), which image writers use to pick
    the format. The rename replaces an existing file on every system."""
    path = Path(path)
    tmp = path.with_name(f".partial-{path.name}")
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
