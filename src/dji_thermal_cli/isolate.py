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
import queue
import signal
import threading
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


def _crash_reason(proc) -> str:
    if proc.exitcode is not None and proc.exitcode < 0:
        try:
            sig_name = signal.Signals(-proc.exitcode).name
        except ValueError:
            sig_name = f"signal {-proc.exitcode}"
        return f"native SDK crashed ({sig_name})"
    return "worker process exited without a result"


def _worker_main(conn) -> None:
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        fn, args, kwargs = msg
        try:
            conn.send(("ok", fn(*args, **kwargs)))
        except Exception as e:  # noqa: BLE001 -- report back instead of killing the worker
            conn.send(("error", f"{type(e).__name__}: {e}"))


class _Worker:
    """One long-lived child process; a crash in it only ever costs the job it was running."""

    def __init__(self) -> None:
        self._conn, child = _CTX.Pipe()
        self.proc = _CTX.Process(target=_worker_main, args=(child,), daemon=True)
        self.proc.start()
        child.close()

    def alive(self) -> bool:
        return self.proc.is_alive()

    def run(self, fn: Callable[..., Any], args: tuple, timeout: float) -> tuple[bool, Any]:
        try:
            self._conn.send((fn, args, {}))
        except (BrokenPipeError, OSError):
            self.proc.join(timeout=5)
            return False, _crash_reason(self.proc)
        if not self._conn.poll(timeout):
            self.proc.terminate()
            self.proc.join(timeout=5)
            return False, f"timed out after {timeout:.0f}s"
        try:
            status, payload = self._conn.recv()
        except EOFError:
            self.proc.join(timeout=5)
            return False, _crash_reason(self.proc)
        return status == "ok", payload

    def close(self) -> None:
        try:
            if self.proc.is_alive():
                self._conn.send(None)
                self.proc.join(timeout=2)
        except (BrokenPipeError, OSError):
            pass
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5)
        self._conn.close()


class WorkerPool:
    """Runs jobs concurrently across N crash-isolated worker processes.

    Each worker is reused across jobs (so per-process setup like loading libdirp is paid
    once). If a job segfaults or times out, that job is reported as failed and a fresh
    worker replaces the dead one; the rest of the batch is unaffected.
    """

    def __init__(self, workers: int, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self.workers = max(1, workers)
        self.timeout = timeout

    def imap_unordered(self, fn: Callable[..., Any], arglist: list[tuple]):
        """Yield (index, ok, result_or_reason) as each job finishes, in completion order."""
        if not arglist:
            return
        job_q: queue.SimpleQueue = queue.SimpleQueue()
        for i, a in enumerate(arglist):
            job_q.put((i, a))
        res_q: queue.SimpleQueue = queue.SimpleQueue()
        live: set[_Worker] = set()
        lock = threading.Lock()

        def loop() -> None:
            w: _Worker | None = None
            try:
                while True:
                    try:
                        i, a = job_q.get_nowait()
                    except queue.Empty:
                        return
                    if w is None:
                        w = _Worker()
                        with lock:
                            live.add(w)
                    ok, payload = w.run(fn, a, self.timeout)
                    if not w.alive():
                        with lock:
                            live.discard(w)
                        w.close()
                        w = None
                    res_q.put((i, ok, payload))
            finally:
                if w is not None:
                    with lock:
                        live.discard(w)
                    w.close()

        threads = [threading.Thread(target=loop, daemon=True) for _ in range(min(self.workers, len(arglist)))]
        for t in threads:
            t.start()
        try:
            for _ in range(len(arglist)):
                yield res_q.get()
        finally:
            with lock:
                stragglers = list(live)
            for w in stragglers:
                w.proc.terminate()


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
