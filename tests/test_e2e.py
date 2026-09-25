"""End-to-end: real CLI + real background server + real model, via subprocesses."""

import json
import socket
import subprocess
import sys
import time

import httpx
import pytest

from tests.conftest import MODELS_DIR, model_available

pytestmark = [pytest.mark.slow, pytest.mark.model]
SECOND_MODEL = "nli-minilm2-l6-int8"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("noulo-e2e")
    port = free_port()
    (root / ".env").write_text(
        "\n".join(
            [
                f"NOULO_PORT={port}",
                f"NOULO_MODELS_DIR={MODELS_DIR}",
                f"NOULO_MODELS_FILE={root / 'models.json'}",
                f"NOULO_MEMORY_LOCATION={root / 'memory.sqlite3'}",
                f"NOULO_RUN_DIR={root / 'run'}",
            ]
        )
        + "\n"
    )
    yield root, f"http://127.0.0.1:{port}"
    subprocess.run(
        [sys.executable, "-m", "noulo.cli", "--env-file", str(root / ".env"), "stop"],
        capture_output=True,
        timeout=60,
    )


def cli(project, *args, stdin=None, timeout=240):
    root, _ = project
    proc = subprocess.run(
        [sys.executable, "-m", "noulo.cli", "--env-file", str(root / ".env"), *args],
        capture_output=True,
        text=True,
        input=stdin,
        timeout=timeout,
        cwd=root,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_full_lifecycle(project):
    root, url = project

    # --- start in the background with the frontend
    code, out, err = cli(project, "start", "--no-browser")
    assert code == 0, err
    assert f"{url}/ui/" in out
    code, out, _ = cli(project, "status")
    assert "is running" in out and "nli-deberta-v3-xsmall-int8" in out
    assert httpx.get(f"{url}/health").json() == {"status": "ok", "modelLoaded": True}
    assert httpx.get(f"{url}/ui/").status_code == 200

    # --- use the endpoints through the CLI
    code, out, _ = cli(
        project,
        "noul",
        "-i",
        "I checked my account and you have taken the subscription payment twice.",
        "-p",
        "The customer reports being charged more than once.",
    )
    assert code == 0 and float(out) > 0.8
    code, out, _ = cli(
        project,
        "choice",
        "-i",
        "The customer says their subscription payment was taken twice.",
        "-q",
        "Which department should handle this?",
        "-o",
        "A=Billing",
        "-o",
        "B=Technical Support",
        "-o",
        "C=Sales",
    )
    assert out.strip() == "A"
    code, out, _ = cli(
        project,
        "score",
        "-i",
        "The production system is unavailable for every customer.",
        "-q",
        "How severe is this incident?",
        "-r",
        "insignificant,low,medium,high,critical",
    )
    assert 0.0 <= float(out) <= 1.0
    code, out, _ = cli(
        project,
        "evaluate",
        "-",
        stdin=json.dumps(
            {
                "type": "noul",
                "input": "The invoice has remained unpaid for 120 days.",
                "proposition": "The customer has an overdue payment.",
            }
        ),
    )
    assert float(out) > 0.8

    # --- validation error surfaces cleanly
    code, _, err = cli(project, "noul", "-i", "x", "-p", " ")
    assert code == 1 and "Noul requires a proposition." in err

    # --- learning + feedback over real HTTP
    code, out, _ = cli(
        project,
        "noul",
        "-i",
        "My parcel never arrived.",
        "-p",
        "The customer is satisfied.",
        "--json",
    )
    record_id = json.loads(out)["recordId"]
    assert cli(project, "feedback", record_id, "false")[0] == 0

    # --- RAM regression guard: default config measures ~500 MiB RSS on an M1 Pro
    #     (model session ~290 + DeBERTa tokenizer ~100 + embedder + runtime).
    import psutil

    pid = json.loads((root / "run" / "noulo.pid").read_text())["pid"]
    rss_mb = psutil.Process(pid).memory_info().rss / 1024 / 1024
    print(f"server RSS: {rss_mb:.0f} MiB")
    assert rss_mb < 560

    # --- switching models saves the config and restarts the service
    if model_available(MODELS_DIR / SECOND_MODEL):
        code, out, err = cli(project, "model", "use", SECOND_MODEL)
        assert code == 0, err
        assert "Restarting" in out
        assert f"NOULO_MODEL={SECOND_MODEL}" in (root / ".env").read_text()
        assert httpx.get(f"{url}/api/v1/info").json()["model"] == SECOND_MODEL
        assert json.loads((root / "run" / "noulo.pid").read_text())["pid"] != pid
        assert httpx.get(f"{url}/ui/").status_code == 200  # frontend still served

    # --- graceful stop
    code, out, _ = cli(project, "stop")
    assert "Stopped" in out
    time.sleep(0.5)
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{url}/health", timeout=2)
    log = (root / "run" / "noulo.log").read_text()
    assert "Application shutdown complete" in log
    assert "Traceback" not in log and "libc++abi" not in log


def test_low_memory_server_stays_well_under_the_500_mb_budget(tmp_path):
    import os

    import psutil

    port = free_port()
    env = {
        **os.environ,
        "NOULO_PORT": str(port),
        "NOULO_LOW_MEMORY": "true",
        "NOULO_MODELS_DIR": str(MODELS_DIR),
        "NOULO_MODELS_FILE": "",
        "NOULO_MEMORY_LOCATION": str(tmp_path / "memory.sqlite3"),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "noulo.cli", "--env-file", str(tmp_path / ".env"), "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(240):
            try:
                if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.25)
        body = {
            "input": "The production system is unavailable for every customer. " * 4,
            "question": "How severe is this incident?",
            "rubric": ["insignificant", "low", "medium", "high", "critical"],
        }
        assert httpx.post(f"{url}/api/v1/score", json=body, timeout=30).status_code == 200
        rss_mb = psutil.Process(proc.pid).memory_info().rss / 1024 / 1024
        print(f"low-memory server RSS: {rss_mb:.0f} MiB")
        assert rss_mb < 450
    finally:
        proc.terminate()
        proc.wait(30)
