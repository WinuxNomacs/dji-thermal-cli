"""Runs one file's dirp work in a child process.

Observed in practice: libdirp.so segfaults (not a Python exception -- a real
SIGSEGV) inside dirp_create_from_rjpeg for some R-JPEGs (e.g. several of the
DJI SDK's own M3TD/M4T sample files), while the exact same call succeeds for
the great majority of files. Running each file in its own subprocess means
one crashing file is reported and skipped instead of taking the whole batch
down.
"""

from __future__ import annotations

import multiprocessing as mp
import signal
from typing import Any, Callable

_CTX = mp.get_context("spawn")
DEFAULT_TIMEOUT_S = 120


def _run_and_send(fn: Callable[..., Any], args: tuple, kwargs: dict, conn) -> None:
    try:
        conn.send(("ok", fn(*args, **kwargs)))
    except Exception as e:  # noqa: BLE001 -- report any exception back to the parent instead of losing it
        conn.send(("error", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def run_isolated(fn: Callable[..., Any], *args, timeout: float = DEFAULT_TIMEOUT_S, **kwargs) -> tuple[bool, Any]:
    """Run fn(*args, **kwargs) in a child process. Returns (ok, result_or_reason).

    fn, and everything in args/kwargs, must be picklable (top-level
    functions and plain data -- not an already-open ctypes/DirpSDK handle).
    """
    parent_conn, child_conn = _CTX.Pipe(duplex=False)
    proc = _CTX.Process(target=_run_and_send, args=(fn, args, kwargs, child_conn))
    proc.start()
    child_conn.close()

    if parent_conn.poll(timeout):
        try:
            status, payload = parent_conn.recv()
        except EOFError:
            # Child closed the pipe (typically: it crashed) without sending
            # a result. proc.exitcode below turns this into a signal name.
            status, payload = "error", "child process exited without a result"
    else:
        status, payload = "error", f"timed out after {timeout:.0f}s"
        proc.terminate()

    proc.join(timeout=5)
    parent_conn.close()

    if proc.exitcode is not None and proc.exitcode < 0:
        try:
            sig_name = signal.Signals(-proc.exitcode).name
        except ValueError:
            sig_name = f"signal {-proc.exitcode}"
        return False, f"native SDK crashed ({sig_name})"

    if status == "ok":
        return True, payload
    return False, payload
