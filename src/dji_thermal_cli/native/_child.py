"""The native child process: loads DJI's libdirp and answers the parent's requests (see client).

    python -c "...; _child.main()" <libdirp path>

Requests and replies are pickled frames, each a 4-byte big-endian length then the pickle, on
stdin and on a copy of stdout taken at start; stdout itself then goes to the null device, so
whatever the library prints can't corrupt the frames (or a CLI's output). The first frame sent
is ("ready", library path) or ("failed", reason). Requests: (op, handles to close first, *args):

    ("open", closes, data)              -> ("ok", {"id", "width", "height", "api"})
    ("call", closes, id, method, args)  -> ("ok", result)
    ("exit", closes)                    -> the child exits

A failed call replies ("dirp", function, code) for a dirp error, or ("error", type, message).
A crash in the library ends the child: on Windows, where ctypes turns it into an OSError, the
child exits with the exception's code, as the crash would have ended it.
"""

from __future__ import annotations

import os
import pickle
import re
import struct
import sys
from typing import BinaryIO, Callable

_HEADER = struct.Struct(">I")
_ACCESS_VIOLATION = 0xC0000005


def read_frame(stream: BinaryIO):
    header = stream.read(_HEADER.size)
    if len(header) < _HEADER.size:
        raise EOFError
    size = _HEADER.unpack(header)[0]
    body = stream.read(size)
    if len(body) < size:
        raise EOFError
    return pickle.loads(body)


def write_frame(stream: BinaryIO, obj) -> None:
    body = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(body)) + body)
    stream.flush()


def _streams() -> tuple[BinaryIO, BinaryIO]:
    replies = os.fdopen(os.dup(1), "wb")
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.close(null)
    sys.stdout = open(os.devnull, "w")        # noqa: SIM115 - for the child's lifetime
    return sys.stdin.buffer, replies


def serve(load: Callable[[str], object], argv: list[str]) -> None:
    """Answer requests until "exit" or the end of stdin. load(argv[0]) returns the library:
    an object whose open(data) returns a handle with an id-less method per request."""
    requests, replies = _streams()
    try:
        library = load(argv[0])
    except Exception as e:  # noqa: BLE001 - reported to the parent, which raises it
        write_frame(replies, ("failed", f"{type(e).__name__}: {e}"))
        return
    write_frame(replies, ("ready", str(getattr(library, "path", argv[0]))))
    handles: dict[int, object] = {}
    next_id = 1
    while True:
        try:
            request = read_frame(requests)
        except EOFError:
            return
        op, closes, args = request[0], request[1], request[2:]
        for hid in closes:
            h = handles.pop(hid, None)
            if h is not None:
                h.close()
        if op == "exit":
            return
        try:
            if op == "open":
                h = library.open(args[0])
                try:
                    info = {"id": next_id, "width": h.width, "height": h.height, "api": h.api_version()}
                except BaseException:
                    h.close()
                    raise
                handles[next_id] = h
                next_id += 1
                reply = ("ok", info)
            elif op == "call":
                hid, method, call_args = args
                if hid not in handles:
                    raise LookupError(f"no open handle {hid}")
                reply = ("ok", getattr(handles[hid], method)(*call_args))
            else:
                raise ValueError(f"unknown request {op!r}")
        except Exception as e:  # noqa: BLE001 - every failure is the parent's to raise
            if os.name == "nt" and isinstance(e, OSError) and str(e).startswith("exception"):
                _crash(str(e))
            func, code = getattr(e, "func", None), getattr(e, "code", None)
            reply = ("dirp", func, code) if func is not None and isinstance(code, int) else \
                ("error", type(e).__name__, str(e))
        write_frame(replies, reply)


def _crash(message: str) -> None:
    """End as a crash in the library would have: ctypes on Windows reports one as an OSError
    ("exception: access violation ...", "exception code 0x...") and the library's state can't
    be trusted after it."""
    found = re.search(r"exception code (0x[0-9a-fA-F]+)", message)
    code = int(found.group(1), 16) if found else _ACCESS_VIOLATION
    sys.stderr.write(message + "\n")
    sys.stderr.flush()
    os._exit(code - (1 << 32) if code >= 1 << 31 else code)


def main() -> None:
    from . import bindings
    serve(lambda library: bindings.Dirp(bindings.Path(library)), sys.argv[1:])
