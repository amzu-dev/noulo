import json

import pytest

from noulo.service import ServiceManager, ServiceState


class FakeProcesses:
    """Stands in for OS process control: spawn/kill/alive."""

    def __init__(self):
        self.next_pid = 1000
        self.alive_pids: set[int] = set()
        self.spawned: list[list[str]] = []
        self.signalled: list[int] = []

    def spawn(self, argv, log_path):
        self.next_pid += 1
        self.alive_pids.add(self.next_pid)
        self.spawned.append(argv)
        return self.next_pid

    def terminate(self, pid):
        self.signalled.append(pid)
        self.alive_pids.discard(pid)

    def alive(self, pid):
        return pid in self.alive_pids


@pytest.fixture
def procs():
    return FakeProcesses()


def manager(tmp_path, procs, healthy=True):
    return ServiceManager(
        tmp_path / "state",
        spawn=procs.spawn,
        terminate=procs.terminate,
        alive=procs.alive,
        healthy=lambda url: healthy,
        poll_interval=0.0,
        timeout=0.05,
    )


def test_status_when_nothing_running(tmp_path, procs):
    assert manager(tmp_path, procs).status().running is False


def test_start_spawns_detached_server_and_writes_pidfile(tmp_path, procs):
    svc = manager(tmp_path, procs)
    state = svc.start(host="127.0.0.1", port=8787, ui=True, extra_args=["--model", "m1"])
    assert state.running and state.pid == 1001 and state.url == "http://127.0.0.1:8787"
    argv = procs.spawned[0]
    assert argv[-5:] == ["--host", "127.0.0.1", "--port", "8787", "--ui"][-5:] or "--ui" in argv
    assert "serve" in argv and "--model" in argv
    saved = json.loads((tmp_path / "state" / "noulo.pid").read_text())
    assert saved["pid"] == 1001 and saved["ui"] is True


def test_start_refuses_when_already_running(tmp_path, procs):
    svc = manager(tmp_path, procs)
    svc.start(host="127.0.0.1", port=8787, ui=False)
    with pytest.raises(RuntimeError, match="already running"):
        svc.start(host="127.0.0.1", port=8787, ui=False)


def test_start_reports_failure_when_server_never_becomes_healthy(tmp_path, procs):
    svc = manager(tmp_path, procs, healthy=False)
    with pytest.raises(RuntimeError, match="did not become ready"):
        svc.start(host="127.0.0.1", port=8787, ui=False)
    assert svc.status().running is False
    assert procs.signalled == [1001]  # the half-started process is cleaned up


def test_stop_terminates_and_removes_pidfile(tmp_path, procs):
    svc = manager(tmp_path, procs)
    svc.start(host="127.0.0.1", port=8787, ui=False)
    assert svc.stop() is True
    assert procs.signalled == [1001]
    assert not (tmp_path / "state" / "noulo.pid").exists()


def test_stop_when_not_running_is_noop(tmp_path, procs):
    assert manager(tmp_path, procs).stop() is False


def test_stale_pidfile_is_ignored(tmp_path, procs):
    svc = manager(tmp_path, procs)
    svc.start(host="127.0.0.1", port=8787, ui=False)
    procs.alive_pids.clear()  # process died without cleaning up
    assert svc.status().running is False
    svc.start(host="127.0.0.1", port=8787, ui=False)  # allowed again


def test_restart_keeps_host_port_and_ui_mode(tmp_path, procs):
    svc = manager(tmp_path, procs)
    svc.start(host="127.0.0.1", port=9001, ui=True)
    state = svc.restart()
    assert state.pid == 1002 and state.port == 9001 and state.ui is True
    assert procs.signalled == [1001]


def test_restart_when_not_running_raises(tmp_path, procs):
    with pytest.raises(RuntimeError, match="not running"):
        manager(tmp_path, procs).restart()


def test_state_roundtrip():
    state = ServiceState(pid=5, host="127.0.0.1", port=1, ui=False, log="x.log")
    assert ServiceState.from_dict(state.to_dict()) == state
