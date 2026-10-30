"""The parent side of the native child process (see _child): DJI's library is loaded only there,
so a crash or a hang in it fails one request, never the caller's process.

One child per library and per process, started on first use, again after a crash, and again
when the library file changes (an SDK replaced in place); requests are serialized. A request that
gets no reply within its timeout kills the child.
"""

from __future__ import annotations

import atexit
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from .. import tsdk
from ._child import read_frame, write_frame

START_TIMEOUT = 30.0        # seconds for the child to load the library
REQUEST_TIMEOUT = 120.0     # seconds for one call (a measurement takes milliseconds)
_WINDOWS_EXCEPTIONS = {0xC0000005: " (access violation)", 0xC00000FD: " (stack overflow)",
                       0xC0000409: " (stack buffer overrun)", 0xC000001D: " (illegal instruction)"}

_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
# The child runs this package's own copy (also when it was put on sys.path at run time, as
# WebODM's plugins are), without the current directory on its path
_BOOT = ("import sys; sys.path = [p for p in sys.path if p not in ('', '.')]; sys.path.insert(0, {!r}); "
         "from dji_thermal_cli.native._child import main; main()")


class NativeError(tsdk.TsdkError):
    """DJI's native SDK failed (other than refusing a file or a parameter)."""

    def __init__(self, message: str, code: int = -1):
        super().__init__(message)
        self.code = code


class NativeSdkError(NativeError):
    """DJI's SDK can't be used: not found, not loading, or a version these bindings don't support."""


class NativeCrash(NativeError):
    """DJI's library crashed, or stopped answering, in the child process; the caller goes on."""


_EOF = object()


class NativeProcess:
    """The child process serving one libdirp."""

    def __init__(self, library: Path, command: list[str] | None = None,
                 request_timeout: float = REQUEST_TIMEOUT, start_timeout: float = START_TIMEOUT):
        self.library = Path(library)
        self.command = command or [sys.executable, "-c", _BOOT.format(_PACKAGE_PARENT), str(self.library)]
        self.request_timeout = request_timeout
        self.start_timeout = start_timeout
        self.generation = 0             # increases with each child: handles of an earlier one are gone
        self._proc: subprocess.Popen | None = None
        self._replies: queue.Queue = queue.Queue()
        self._stderr = None
        self._closes: list[int] = []
        self._loaded = None             # the library file the child loaded (see _identity)
        self._lock = threading.RLock()

    # --- the child's life -------------------------------------------------------------------

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _identity(self):
        """What tells the library file apart from one put in its place."""
        try:
            st = self.library.stat()
        except OSError:
            return None
        return st.st_ino, st.st_size, st.st_mtime_ns

    def _start(self) -> None:
        env = dict(os.environ)
        if os.name != "nt":             # the camera plugins' own dependencies, found by soname
            env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(self.library.parent), env.get("LD_LIBRARY_PATH")]))
        self._stderr = tempfile.TemporaryFile()
        self._replies = queue.Queue()
        self._closes = []
        self._loaded = self._identity()
        self._proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=self._stderr, cwd=str(self.library.parent), env=env)
        self.generation += 1
        threading.Thread(target=self._read, args=(self._proc.stdout, self._replies), daemon=True).start()
        first = self._reply(self.start_timeout, "loading the library")
        if first[0] != "ready":
            self._stop()
            raise NativeSdkError(f"DJI's Thermal SDK at {self.library} does not load: {first[1]}")

    @staticmethod
    def _read(stream, replies: queue.Queue) -> None:
        try:
            while True:
                replies.put(read_frame(stream))
        except Exception:  # noqa: BLE001 - EOF or a broken pipe: the child is gone
            replies.put(_EOF)

    def _reply(self, timeout: float, doing: str):
        try:
            reply = self._replies.get(timeout=timeout)
        except queue.Empty:
            self._stop()
            raise NativeCrash(f"DJI's Thermal SDK stopped answering ({doing}, {timeout:g} s); its process was stopped") from None
        if reply is _EOF:
            raise NativeCrash(f"DJI's Thermal SDK crashed ({doing}): {self._exit_reason()}")
        return reply

    def _exit_reason(self) -> str:
        try:
            code = self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            code = None
        if code is None:
            why = "its process stopped answering"
        elif code < 0:
            try:
                why = f"killed by {signal.Signals(-code).name}"
            except ValueError:
                why = f"killed by signal {-code}"
        elif code >= 0xC0000000:                  # a Windows exception, as a crash ends a process there
            why = f"ended by exception 0x{code:08X}" + _WINDOWS_EXCEPTIONS.get(code, "")
        else:
            why = f"its process exited with status {code}"
        tail = self._stderr_tail()
        self._stop()
        return why + (f"; it wrote: {tail}" if tail else "")

    def _stderr_tail(self, limit: int = 500) -> str:
        try:
            self._stderr.seek(0)
            text = self._stderr.read().decode("utf-8", "replace").strip()
        except (OSError, ValueError, AttributeError):
            return ""
        return " ".join(text.split())[-limit:]

    def _stop(self) -> None:
        """End the child (killed if still running) and close its pipes and error file."""
        proc, self._proc = self._proc, None
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            for stream in (proc.stdin, proc.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def shutdown(self) -> None:
        with self._lock:
            if self.alive():
                try:
                    write_frame(self._proc.stdin, ("exit", []))
                    self._proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._stop()

    # --- requests ---------------------------------------------------------------------------

    def request(self, op: str, *args, doing: str | None = None):
        """The child's reply to (op, *args), started first if needed: ("ok", result), ("dirp",
        function, code) or ("error", type, message). Raises NativeCrash if it crashes or hangs."""
        with self._lock:
            if self.alive() and self._identity() != self._loaded:
                self.shutdown()                 # the SDK was replaced: load the new one
            if not self.alive():
                self._start()
            closes, self._closes = self._closes, []
            try:
                write_frame(self._proc.stdin, (op, closes, *args))
            except OSError:
                raise NativeCrash(f"DJI's Thermal SDK crashed ({doing or op}): {self._exit_reason()}") from None
            return self._reply(self.request_timeout, doing or op)

    def close_handle(self, hid: int, generation: int) -> None:
        """Close a handle with the next request (handles of an earlier child are already gone)."""
        if generation == self.generation:
            self._closes.append(hid)


_processes: dict[Path, NativeProcess] = {}
_owner = os.getpid()
_registry_lock = threading.Lock()


def process_for(library: Path) -> NativeProcess:
    """This process's child for the library. A forked process (a worker pool's) starts its own:
    the pipes it inherited belong to its parent."""
    global _owner
    library = Path(library).resolve()
    with _registry_lock:
        if _owner != os.getpid():
            _processes.clear()
            _owner = os.getpid()
        if library not in _processes:
            _processes[library] = NativeProcess(library)
        return _processes[library]


@atexit.register
def _shutdown_all() -> None:
    if _owner == os.getpid():
        for p in list(_processes.values()):
            p.shutdown()
