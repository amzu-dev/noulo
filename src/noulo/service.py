"""Background service management: start (detached), stop, restart, status.

A pidfile in the state directory records the running server's pid, address
and UI mode so `restart` can bring it back exactly as it was (e.g. after the
model is changed).
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

PIDFILE = "noulo.pid"
LOGFILE = "noulo.log"

_children: dict[int, subprocess.Popen] = {}


@dataclass(frozen=True)
class ServiceState:
    pid: int | None
    host: str
    port: int
    ui: bool
    log: str
    running: bool = True

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("running")
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ServiceState:
        return cls(
            pid=data["pid"],
            host=data["host"],
            port=int(data["port"]),
            ui=bool(data["ui"]),
            log=data.get("log", ""),
        )


NOT_RUNNING = ServiceState(pid=None, host="", port=0, ui=False, log="", running=False)


# ---------------------------------------------------------------------- OS process helpers


def spawn_detached(argv: list[str], log_path: Path, cwd: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    kwargs: dict = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(argv, cwd=cwd, **kwargs)
    _children[proc.pid] = proc
    return proc.pid


def process_alive(pid: int) -> bool:
    child = _children.get(pid)
    if child is not None:  # our own child: poll() also reaps zombies
        return child.poll() is None
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, OSError):
        os.kill(pid, signal.SIGTERM)  # graceful: uvicorn drains requests, engine shuts down


def kill_process(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, OSError):
        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))


def http_healthy(url: str) -> bool:
    import httpx

    try:
        return httpx.get(f"{url}/health", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


# ---------------------------------------------------------------------- manager


class ServiceManager:
    def __init__(
        self,
        state_dir: str | Path,
        *,
        spawn: Callable[[list[str], Path], int] | None = None,
        cwd: str | Path | None = None,
        terminate: Callable[[int], None] = terminate_process,
        kill: Callable[[int], None] = kill_process,
        alive: Callable[[int], bool] = process_alive,
        healthy: Callable[[str], bool] = http_healthy,
        timeout: float = 180.0,
        stop_timeout: float = 20.0,
        poll_interval: float = 0.25,
        python: str = sys.executable,
    ):
        self.state_dir = Path(state_dir)
        self._spawn = spawn or (lambda argv, log: spawn_detached(argv, log, cwd=cwd))
        self._terminate, self._kill = terminate, kill
        self._alive, self._healthy = alive, healthy
        self._timeout, self._stop_timeout = timeout, stop_timeout
        self._poll = poll_interval
        self._python = python

    @property
    def pidfile(self) -> Path:
        return self.state_dir / PIDFILE

    @property
    def logfile(self) -> Path:
        return self.state_dir / LOGFILE

    def status(self) -> ServiceState:
        if not self.pidfile.exists():
            return NOT_RUNNING
        try:
            state = ServiceState.from_dict(json.loads(self.pidfile.read_text()))
        except (ValueError, KeyError):
            return NOT_RUNNING
        if state.pid is None or not self._alive(state.pid):
            return NOT_RUNNING
        return state

    def start(
        self,
        *,
        host: str,
        port: int,
        ui: bool,
        extra_args: list[str] | None = None,
        global_args: list[str] | None = None,
    ) -> ServiceState:
        if self.status().running:
            raise RuntimeError(f"noulo is already running ({self.status().url}).")
        argv = [
            self._python,
            "-m",
            "noulo.cli",
            *(global_args or []),
            "serve",
            "--host",
            host,
            "--port",
            str(port),
            *(extra_args or []),
        ]
        if ui:
            argv.append("--ui")
        pid = self._spawn(argv, self.logfile)
        state = ServiceState(pid=pid, host=host, port=port, ui=ui, log=str(self.logfile))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.pidfile.write_text(json.dumps(state.to_dict()) + "\n")
        self._saved_args = (extra_args, global_args)

        deadline = time.monotonic() + self._timeout
        while True:
            if self._healthy(state.url):
                return state
            if not self._alive(pid) or time.monotonic() >= deadline:
                break
            time.sleep(self._poll)
        self._terminate(pid)
        self.pidfile.unlink(missing_ok=True)
        raise RuntimeError(f"noulo did not become ready; see the log at {self.logfile}.")

    def stop(self) -> bool:
        state = self.status()
        if not state.running or state.pid is None:
            self.pidfile.unlink(missing_ok=True)
            return False
        self._terminate(state.pid)
        deadline = time.monotonic() + self._stop_timeout
        while self._alive(state.pid) and time.monotonic() < deadline:
            time.sleep(self._poll or 0.05)
        if self._alive(state.pid):
            self._kill(state.pid)
        self.pidfile.unlink(missing_ok=True)
        return True

    def restart(
        self, *, extra_args: list[str] | None = None, global_args: list[str] | None = None
    ) -> ServiceState:
        state = self.status()
        if not state.running:
            raise RuntimeError("noulo is not running; start it with `noulo start`.")
        self.stop()
        return self.start(
            host=state.host,
            port=state.port,
            ui=state.ui,
            extra_args=extra_args,
            global_args=global_args,
        )
