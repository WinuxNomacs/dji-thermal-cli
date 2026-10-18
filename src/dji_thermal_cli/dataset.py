"""Finding and classifying a dataset's files by their content, never by their name.

scan() walks a directory and keeps every file whose first bytes are a JPEG signature,
whatever its extension (DJI_0001_T.JPG, a renamed .jpeg, or no extension at all); the R-JPEG
parser then decides whether each one is a thermal image it supports. classify() gives one
file's status for reporting, optionally decoding the image data to catch truncated or damaged
files that still parse.
"""

from __future__ import annotations

import dataclasses
import io
import struct
from pathlib import Path

from PIL import Image

from . import tsdk

JPEG_SIGNATURE = b"\xff\xd8\xff"        # SOI followed by the next marker's 0xFF

# classify() kinds, from best to worst
THERMAL = "thermal"                     # an R-JPEG the SDK port measures
UNSUPPORTED = "unsupported"             # a DJI thermal R-JPEG of a format the SDK port can't read
NOT_THERMAL = "not_thermal"             # a valid JPEG without DJI thermal data (e.g. a visual _V image)
CORRUPT = "corrupt"                     # a JPEG whose structure or image data is damaged
NOT_JPEG = "not_jpeg"                   # not a JPEG at all, whatever its name
UNREADABLE = "unreadable"               # the file couldn't be read (permissions, I/O error)


@dataclasses.dataclass
class Scan:
    jpegs: list[Path]                   # candidates, in path order
    others: list[Path]                  # files skipped because they aren't JPEGs
    unreadable: list[tuple[Path, str]]  # files that couldn't be opened, with the reason


@dataclasses.dataclass
class FileStatus:
    path: Path
    kind: str
    reason: str = ""
    plugin: str = ""                    # the SDK plugin for thermal and unsupported files
    warnings: list[str] = dataclasses.field(default_factory=list)


def has_jpeg_signature(path: Path) -> bool:
    """True when the file starts with a JPEG signature. Raises OSError if it can't be read."""
    with path.open("rb") as f:
        return f.read(len(JPEG_SIGNATURE)) == JPEG_SIGNATURE


def scan(root: Path) -> Scan:
    """The JPEG files under root (searched recursively), or root itself when it is a file.
    A single file is always a candidate, so a wrong file given explicitly is reported."""
    if root.is_file():
        return Scan([root], [], [])
    out = Scan([], [], [])
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            (out.jpegs if has_jpeg_signature(path) else out.others).append(path)
        except OSError as e:
            out.unreadable.append((path, e.strerror or str(e)))
    return out


def classify(path: Path, decode: bool = False) -> FileStatus:
    """What the file is, from its content. decode: also decode the JPEG image data (slower: a
    full decode), which finds files cut short or damaged inside their image data. In a thermal
    R-JPEG that image is only the camera's preview, so damage there is a warning while the
    thermal data still measures; in any other JPEG it makes the file corrupt."""
    try:
        data = path.read_bytes()
    except OSError as e:
        return FileStatus(path, UNREADABLE, e.strerror or str(e))
    if not data.startswith(JPEG_SIGNATURE):
        return FileStatus(path, NOT_JPEG, "not a JPEG file (by content)")
    if problem := structure_problem(data):
        return FileStatus(path, CORRUPT, problem)
    segments = tsdk.jpeg.segments(data)
    decode_problem = _decode_problem(data) if decode else ""
    plugin = tsdk.plugin_name(segments)
    try:
        tsdk.open(data)
    except tsdk.UnsupportedRJpeg as e:
        return FileStatus(path, UNSUPPORTED, str(e), plugin)
    except tsdk.RJpegError as e:
        if plugin in ("none", "unknown"):
            if decode_problem:
                return FileStatus(path, CORRUPT, decode_problem)
            return FileStatus(path, NOT_THERMAL, str(e))
        return FileStatus(path, CORRUPT, f"thermal data damaged: {e}", plugin)
    except Exception as e:  # noqa: BLE001 - a malformed file must never stop a dataset scan
        return FileStatus(path, CORRUPT, f"thermal data unreadable: {type(e).__name__}: {e}", plugin)
    warnings = [f"preview {decode_problem}; the thermal data is intact"] if decode_problem else []
    return FileStatus(path, THERMAL, "", plugin, warnings)


def structure_problem(data: bytes) -> str:
    """Why the JPEG's marker structure is broken, or '': every segment before the image data
    must fit in the file, and the image data (start of scan) must be present. A file cut short
    fails this even where the cut removed the DJI thermal blocks."""
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return f"JPEG structure broken at byte {i} (no marker)"
        marker = data[i + 1]
        if marker == 0xFF:                         # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9:
            return "JPEG ends before its image data"
        (length,) = struct.unpack_from(">H", data, i + 2)
        if length < 2 or i + 2 + length > len(data):
            return f"file cut short: segment {marker:#04x} at byte {i} runs past the end of the file"
        if marker == 0xDA:
            return ""
        i += 2 + length
    return "file cut short: no image data"


def _decode_problem(data: bytes) -> str:
    """Why the JPEG image data doesn't decode, or ''."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
    except Exception as e:  # noqa: BLE001 - PIL raises many types for damaged data
        return f"image data damaged: {e}"
    return ""


def output_clashes(paths: list[Path], root: Path, is_thermal=None) -> dict[Path, Path]:
    """Thermal files whose outputs would take the same names as another's. Outputs are named
    after the file's folder (relative to root) and stem, so `a.jpg` and `a.jpeg` clash, and so
    do `A.jpg` and `a.jpg` on Windows and macOS, whose file systems ignore case; names are
    compared ignoring case everywhere so a dataset behaves the same on every system.
    Returns {file: the file that keeps the names}: the first in path order keeps them.
    is_thermal(path): whether a file makes outputs (default: classify it); only files in a
    clashing group are classified."""
    groups: dict[str, list[Path]] = {}
    for p in sorted(paths):
        rel = p.relative_to(root)
        groups.setdefault((rel.parent / rel.stem).as_posix().casefold(), []).append(p)
    is_thermal = is_thermal or (lambda p: classify(p).kind == THERMAL)
    clashes = {}
    for group in groups.values():
        if len(group) > 1:
            makers = [p for p in group if is_thermal(p)]
            clashes.update({p: makers[0] for p in makers[1:]})
    return clashes
