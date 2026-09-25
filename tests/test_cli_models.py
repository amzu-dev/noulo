import io
import json

import pytest
from fastapi.testclient import TestClient

from noulo.api.server import create_app
from noulo.cli import EnvFile, main
from noulo.config import Settings
from noulo.inference.engine import DecisionEngine
from noulo.registry import BUNDLED_MODEL, CURATED_MODELS, ModelRegistry
from noulo.service import ServiceState
from tests.fakes import FakeBackend, FakeLoader


class FakeService:
    def __init__(self, running=False, ui=True):
        self.state = ServiceState(
            pid=42, host="127.0.0.1", port=8787, ui=ui, log="noulo.log", running=running
        )
        self.calls: list[str] = []

    def status(self):
        return self.state if self.state.running else ServiceState(None, "", 0, False, "", False)

    def start(self, *, host, port, ui, extra_args=None, global_args=None):
        self.calls.append(f"start ui={ui}")
        self.state = ServiceState(pid=43, host=host, port=port, ui=ui, log="noulo.log")
        return self.state

    def stop(self):
        self.calls.append("stop")
        was = self.state.running
        self.state = ServiceState(None, "", 0, False, "", False)
        return was

    def restart(self, *, extra_args=None, global_args=None):
        self.calls.append("restart")
        return self.state


class FakeRegistry:
    def __init__(self, installed=()):
        self.installed = set(installed)
        self.downloads: list[str] = []
        self.registered: list[dict] = []

    def list(self):
        return [
            {
                "id": m.id,
                "backend": "onnx-nli",
                "model": m.id,
                "quantization": "INT8",
                "local": True,
                "installed": m.id in self.installed,
                "description": m.label,
            }
            for m in CURATED_MODELS
        ]

    def is_installed(self, model_id):
        return model_id in self.installed

    def is_downloadable(self, model_id):
        return any(m.id == model_id for m in CURATED_MODELS)

    def download(self, model_id, progress=None):
        self.downloads.append(model_id)
        self.installed.add(model_id)

    def register_onnx(self, **kwargs):
        self.registered.append(kwargs)


def run(argv, *, answers=(), api=None, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    replies = iter(answers)
    code = main(
        argv,
        http_client=api,
        stdout=out,
        stderr=err,
        stdin=io.StringIO(""),
        ask=lambda prompt: next(replies),
        browser=lambda url: None,
        **kwargs,
    )
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def env_file(tmp_path):
    return tmp_path / ".env"


UNREACHABLE = "http://127.0.0.1:9"


def test_three_curated_models_with_bundled_default():
    assert len(CURATED_MODELS) == 3
    assert CURATED_MODELS[0].id == BUNDLED_MODEL


def test_picker_lists_three_models_and_plugin_option(env_file):
    code, out, _ = run(
        ["--url", UNREACHABLE, "model"],
        answers=[""],
        env_file=env_file,
        registry=FakeRegistry(installed=[BUNDLED_MODEL]),
        service=FakeService(),
    )
    assert code == 0
    for m in CURATED_MODELS:
        assert m.id in out
    assert "4)" in out and "your own model" in out.lower()


def test_picker_plugin_option_prints_instructions(env_file):
    code, out, _ = run(
        ["--url", UNREACHABLE, "model"],
        answers=["4"],
        env_file=env_file,
        registry=FakeRegistry(),
        service=FakeService(),
    )
    assert code == 0
    assert "add-endpoint" in out and "add-onnx" in out and "models.json" in out


def test_picking_non_bundled_model_downloads_saves_and_restarts_service(env_file):
    registry, service = FakeRegistry(installed=[BUNDLED_MODEL]), FakeService(running=True)
    target = CURATED_MODELS[1].id
    code, out, _ = run(
        ["--url", UNREACHABLE, "model"],
        answers=["2"],
        env_file=env_file,
        registry=registry,
        service=service,
    )
    assert code == 0
    assert registry.downloads == [target]
    assert EnvFile(env_file).get("NOULO_MODEL") == target
    assert service.calls == ["restart"]
    assert "/ui/" in out  # frontend URL shown after restart


def test_picking_installed_bundled_model_does_not_download(env_file):
    registry = FakeRegistry(installed=[BUNDLED_MODEL])
    run(
        ["--url", UNREACHABLE, "model"],
        answers=["1", "n"],
        env_file=env_file,
        registry=registry,
        service=FakeService(),
    )
    assert registry.downloads == []
    assert EnvFile(env_file).get("NOULO_MODEL") == BUNDLED_MODEL


def test_picker_offers_to_start_when_not_running(env_file):
    service = FakeService(running=False)
    code, out, _ = run(
        ["--url", UNREACHABLE, "model"],
        answers=["1", "y"],
        env_file=env_file,
        registry=FakeRegistry(installed=[BUNDLED_MODEL]),
        service=service,
    )
    assert service.calls == ["start ui=True"]


def test_invalid_picker_answer_is_rejected(env_file):
    code, _, err = run(
        ["--url", UNREACHABLE, "model"],
        answers=["9"],
        env_file=env_file,
        registry=FakeRegistry(),
        service=FakeService(),
    )
    assert code == 2 and "1-4" in err


def test_model_use_switches_unmanaged_running_server_live(env_file, tmp_path):
    loader = FakeLoader(m1=FakeBackend("m1"), m2=FakeBackend("m2"))
    engine = DecisionEngine(load_backend=loader, model_id="m1")
    app = create_app(
        Settings(_env_file=None, ui_enabled=False, learning_enabled=False, models_file=None),
        engine=engine,
        registry=ModelRegistry(tmp_path / "models", None),
    )
    with TestClient(app) as api:
        code, out, _ = run(
            ["model", "use", "m2"],
            api=api,
            env_file=env_file,
            registry=FakeRegistry(installed=["m2"]),
            service=FakeService(running=False),
        )
        assert code == 0
        assert api.get("/api/v1/info").json()["model"] == "m2"


def test_model_add_onnx_registers_custom_model(env_file, tmp_path):
    registry = FakeRegistry()
    code, out, _ = run(
        ["model", "add-onnx", "--id", "my-nli", "--path", str(tmp_path)],
        env_file=env_file,
        registry=registry,
        service=FakeService(),
    )
    assert code == 0 and registry.registered[0]["id"] == "my-nli"


def test_models_alias_still_works(env_file):
    code, out, _ = run(
        ["--url", UNREACHABLE, "models", "list"],
        env_file=env_file,
        registry=FakeRegistry(),
        service=FakeService(),
    )
    assert code == 0


# ------------------------------------------------------------------ service commands


def test_start_runs_in_background_with_frontend_by_default(env_file):
    service = FakeService()
    code, out, _ = run(["start", "--no-browser"], env_file=env_file, service=service)
    assert code == 0 and service.calls == ["start ui=True"]
    assert "http://127.0.0.1:8787/ui/" in out


def test_start_headless_background(env_file):
    service = FakeService()
    run(["start", "--headless"], env_file=env_file, service=service)
    assert service.calls == ["start ui=False"]


def test_start_foreground_uses_server_runner(env_file):
    seen = []
    run(
        ["start", "--foreground", "--no-browser"],
        env_file=env_file,
        service=FakeService(),
        run_server=lambda app, s: seen.append(s),
        app_factory=lambda s: object(),
    )
    assert seen and seen[0].ui_enabled is True


def test_stop_restart_status(env_file):
    service = FakeService(running=True)
    assert run(["status"], env_file=env_file, service=service)[1].count("running") >= 1
    assert run(["restart"], env_file=env_file, service=service)[0] == 0
    assert run(["stop"], env_file=env_file, service=service)[0] == 0
    assert service.calls == ["restart", "stop"]
    assert "not running" in run(["status"], env_file=env_file, service=service)[1]


def test_restart_when_stopped_explains(env_file):
    code, _, err = run(["restart"], env_file=env_file, service=FakeService(running=False))
    assert code == 1 and "noulo start" in err


# ------------------------------------------------------------------ learning / feedback commands


@pytest.fixture
def learning_api(tmp_path):
    from noulo.inference.embedder import HashingEmbedder
    from noulo.inference.memory import LearningMemory
    from noulo.inference.stores import open_store

    engine = DecisionEngine(
        load_backend=FakeLoader(m=FakeBackend("m", noul_value=0.9)),
        model_id="m",
        learning=True,
        open_memory=lambda: LearningMemory(
            open_store("sqlite", location=":memory:"), HashingEmbedder()
        ),
    )
    app = create_app(
        Settings(_env_file=None, ui_enabled=False, models_file=None),
        engine=engine,
        registry=ModelRegistry(tmp_path / "models", None),
    )
    with TestClient(app) as client:
        yield client


def test_learning_off_on_persist_and_apply_live(learning_api, env_file):
    assert run(["learning", "off"], api=learning_api, env_file=env_file)[0] == 0
    assert EnvFile(env_file).get("NOULO_LEARNING_ENABLED") == "false"
    assert learning_api.get("/api/v1/learning").json()["enabled"] is False
    run(["learning", "on"], api=learning_api, env_file=env_file)
    assert learning_api.get("/api/v1/learning").json()["enabled"] is True


def test_json_output_includes_record_id_and_feedback_corrects(learning_api, env_file):
    code, out, _ = run(
        ["noul", "-i", "charged twice", "-p", "happy", "--json"],
        api=learning_api,
        env_file=env_file,
    )
    record_id = json.loads(out)["recordId"]
    assert run(["feedback", record_id, "false"], api=learning_api, env_file=env_file)[0] == 0
    code, out, _ = run(
        ["noul", "-i", "charged twice", "-p", "happy"], api=learning_api, env_file=env_file
    )
    assert float(out) < 0.5


def test_learning_records_and_clear(learning_api, env_file):
    run(["noul", "-i", "x", "-p", "y"], api=learning_api, env_file=env_file)
    code, out, _ = run(["learning", "records"], api=learning_api, env_file=env_file)
    assert json.loads(out)["records"][0]["primitive"] == "noul"
    assert run(["learning", "clear"], api=learning_api, env_file=env_file)[0] == 2  # needs --yes
    code, out, _ = run(["learning", "clear", "--yes"], api=learning_api, env_file=env_file)
    assert "Deleted 1" in out


def test_download_missing_only_fetches_absent_default_models(env_file):
    env_file.write_text("NOULO_MODEL=nli-minilm2-l6-int8\nNOULO_EMBEDDER=minilm-l6-v2-int8\n")
    registry = FakeRegistry(installed=["minilm-l6-v2-int8"])
    code, out, _ = run(
        ["model", "download", "--missing"],
        env_file=env_file,
        registry=registry,
        service=FakeService(),
    )
    assert code == 0 and registry.downloads == ["nli-minilm2-l6-int8"]
    assert "already installed: minilm-l6-v2-int8" in out


def test_model_add_onnx_llm_kind(env_file, tmp_path):
    registry = FakeRegistry()
    code, _, _ = run(
        [
            "model",
            "add-onnx",
            "--id",
            "my-llm",
            "--path",
            str(tmp_path),
            "--kind",
            "llm",
            "--quantization",
            "INT4",
        ],
        env_file=env_file,
        registry=registry,
        service=FakeService(),
    )
    assert code == 0 and registry.registered[0]["kind"] == "llm"


def test_status_shows_model_and_device(learning_api, env_file):
    code, out, _ = run(
        ["status"], api=learning_api, env_file=env_file, service=FakeService(running=True)
    )
    assert code == 0 and "device: cpu" in out


# ---------------------------------------------------------------- no terminal (agents, scripts, CI)


def _no_input(prompt):
    raise EOFError


def test_prompt_without_a_terminal_fails_cleanly_with_the_non_interactive_alternative(env_file):
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["--url", UNREACHABLE, "model"],
        stdout=out,
        stderr=err,
        stdin=io.StringIO(""),
        env_file=env_file,
        registry=FakeRegistry(),
        service=FakeService(),
        ask=_no_input,
    )
    assert code == 2
    assert "interactive terminal" in err.getvalue() and "noulo model use" in err.getvalue()
    assert "Traceback" not in err.getvalue()


class TtyOut(io.StringIO):
    def isatty(self):
        return True


def _start(stdout, env_file):
    opened = []
    code = main(
        ["start"],
        stdout=stdout,
        stderr=io.StringIO(),
        stdin=io.StringIO(""),
        env_file=env_file,
        service=FakeService(),
        browser=opened.append,
    )
    return code, opened


def test_start_does_not_open_a_browser_without_a_terminal(env_file):
    code, opened = _start(io.StringIO(), env_file)
    assert code == 0 and opened == []


def test_start_opens_the_frontend_for_a_person_at_a_terminal(env_file):
    code, opened = _start(TtyOut(), env_file)
    assert code == 0 and opened == ["http://127.0.0.1:8787/ui/"]


# ---------------------------------------------------------------- noulo teach FILE


def _write_examples(path, *items, comment=True):
    lines = ["# labelled examples"] if comment else []
    path.write_text("\n".join([*lines, *(json.dumps(i) for i in items)]) + "\n")
    return path


GOOD_NOUL = {"type": "noul", "input": "charged twice", "proposition": "happy", "expected": False}
GOOD_CHOICE = {
    "type": "choice",
    "input": "charged twice",
    "question": "Which team?",
    "choices": [{"id": "A", "text": "Billing"}, {"id": "B", "text": "Sales"}],
    "expected": "B",
}


def test_teach_imports_a_file_and_changes_answers(learning_api, env_file, tmp_path):
    path = _write_examples(tmp_path / "teach.jsonl", GOOD_NOUL, GOOD_CHOICE)
    code, out, err = run(["teach", str(path)], api=learning_api, env_file=env_file)
    assert code == 0, err
    assert "Taught 2 examples" in out
    assert (
        float(
            run(
                ["noul", "-i", "charged twice", "-p", "happy"], api=learning_api, env_file=env_file
            )[1]
        )
        < 0.5
    )


def test_teach_reports_bad_lines_by_number_and_exits_1(learning_api, env_file, tmp_path):
    bad = {"type": "noul", "input": "x", "expected": True}
    path = _write_examples(tmp_path / "teach.jsonl", GOOD_NOUL, bad)
    code, out, err = run(["teach", str(path)], api=learning_api, env_file=env_file)
    assert code == 1
    assert "Taught 1 example" in out
    assert "line 3" in err and "Noul requires a proposition." in err


def test_teach_dry_run_validates_without_teaching(learning_api, env_file, tmp_path):
    path = _write_examples(tmp_path / "teach.jsonl", GOOD_NOUL, GOOD_CHOICE)
    code, out, _ = run(["teach", str(path), "--dry-run"], api=learning_api, env_file=env_file)
    assert code == 0 and "2 examples are valid" in out
    assert learning_api.get("/api/v1/learning").json()["stats"]["records"] == 0


def test_teach_splits_large_files_into_batches(learning_api, env_file, tmp_path):
    items = [dict(GOOD_NOUL, input=f"case {n} " + "x" * 400) for n in range(300)]
    path = _write_examples(tmp_path / "big.jsonl", *items)  # ~140 KB > 64 KB body limit
    code, out, err = run(["teach", str(path)], api=learning_api, env_file=env_file)
    assert code == 0, err and "Taught 300 examples" in out


def test_teach_with_learning_off_says_how_to_turn_it_on(learning_api, env_file, tmp_path):
    learning_api.put("/api/v1/learning", json={"enabled": False})
    path = _write_examples(tmp_path / "teach.jsonl", GOOD_NOUL)
    code, _, err = run(["teach", str(path)], api=learning_api, env_file=env_file)
    assert code == 1 and "noulo learning on" in err


def test_teach_missing_file(env_file, tmp_path):
    code, _, err = run(["teach", str(tmp_path / "nope.jsonl")], env_file=env_file)
    assert code == 2 and "not found" in err
