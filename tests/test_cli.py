import io
import json

import pytest
from fastapi.testclient import TestClient

from noulo.api.server import create_app
from noulo.cli import EnvFile, main
from noulo.config import Settings
from noulo.inference.engine import DecisionEngine
from noulo.registry import ModelRegistry
from tests.fakes import FakeBackend, FakeLoader


@pytest.fixture
def env_file(tmp_path):
    return tmp_path / ".env"


@pytest.fixture
def api(tmp_path):
    loader = FakeLoader(
        **{
            "m1": FakeBackend(
                "m1", noul_value=0.9, choice_probs=[0.2, 0.8], score_probs=[0.0, 0.0, 1.0]
            ),
            "m2": FakeBackend("m2", noul_value=0.1),
        }
    )
    engine = DecisionEngine(load_backend=loader, model_id="m1")
    cfg = Settings(_env_file=None, ui_enabled=False, learning_enabled=False, models_file=None)
    app = create_app(cfg, engine=engine, registry=ModelRegistry(tmp_path / "models", None))
    with TestClient(app) as client:
        yield client


class InstalledRegistry:
    def __init__(self, *ids):
        self.ids = set(ids)

    def is_installed(self, model_id):
        return model_id in self.ids


class StoppedService:
    def status(self):
        from noulo.service import NOT_RUNNING

        return NOT_RUNNING


def run(argv, *, api=None, env_file=None, stdin=None, **kwargs):
    """Run the CLI, returning (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = main(
        argv,
        http_client=api,
        env_file=env_file,
        stdout=out,
        stderr=err,
        stdin=io.StringIO(stdin or ""),
        **kwargs,
    )
    return code, out.getvalue(), err.getvalue()


# ------------------------------------------------------------------ client commands


def test_noul_prints_bare_value(api):
    code, out, _ = run(["noul", "--input", "Unpaid 120 days", "--proposition", "Overdue"], api=api)
    assert code == 0 and float(out) == pytest.approx(0.9)


def test_noul_json_output(api):
    code, out, _ = run(["noul", "--input", "x", "--proposition", "y", "--json"], api=api)
    assert json.loads(out) == {"type": "noul", "value": pytest.approx(0.9)}


def test_input_can_be_read_from_stdin(api):
    code, out, _ = run(["noul", "--input", "-", "--proposition", "y"], api=api, stdin="piped text")
    assert code == 0


def test_choice_accepts_id_equals_text_options(api):
    code, out, _ = run(
        [
            "choice",
            "--input",
            "x",
            "--question",
            "Which?",
            "--option",
            "A=Billing",
            "--option",
            "B=Sales",
        ],
        api=api,
    )
    assert (code, out.strip()) == (0, "B")


def test_score_accepts_comma_separated_rubric(api):
    code, out, _ = run(
        ["score", "--input", "x", "--question", "How bad?", "--rubric", "low,medium,high"], api=api
    )
    assert code == 0 and float(out) == pytest.approx(1.0)


def test_evaluate_reads_json_request_from_stdin(api):
    request = {"type": "noul", "input": "x", "proposition": "y"}
    code, out, _ = run(["evaluate", "-"], api=api, stdin=json.dumps(request))
    assert code == 0 and float(out) == pytest.approx(0.9)


def test_evaluate_reads_json_request_from_file(api, tmp_path):
    path = tmp_path / "req.json"
    path.write_text(
        json.dumps(
            {
                "type": "choice",
                "input": "x",
                "question": "q?",
                "choices": [{"id": "A", "text": "a"}, {"id": "B", "text": "b"}],
            }
        )
    )
    code, out, _ = run(["evaluate", str(path)], api=api)
    assert out.strip() == "B"


def test_api_errors_go_to_stderr_with_exit_code_1(api):
    code, out, err = run(["noul", "--input", "x", "--proposition", " "], api=api)
    assert code == 1 and out == ""
    assert "INVALID_REQUEST" in err and "Noul requires a proposition." in err


def test_health_and_info(api):
    assert run(["health"], api=api)[1].strip() == "ok"
    code, out, _ = run(["info", "--json"], api=api)
    assert json.loads(out)["model"] == "m1"


def test_unreachable_server_explains_how_to_start(env_file):
    code, _, err = run(["--url", "http://127.0.0.1:9", "health"], env_file=env_file)
    assert code == 3
    assert "noulo serve" in err


# ------------------------------------------------------------------ models


def test_models_list_marks_active_model(api):
    code, out, _ = run(["models", "list"], api=api)
    assert code == 0 and "nli-deberta-v3-xsmall-int8" in out


def test_models_use_switches_live_and_persists(api, env_file):
    code, out, _ = run(
        ["models", "use", "m2"],
        api=api,
        env_file=env_file,
        registry=InstalledRegistry("m2"),
        service=StoppedService(),
    )
    assert code == 0
    assert EnvFile(env_file).get("NOULO_MODEL") == "m2"
    assert json.loads(run(["info", "--json"], api=api)[1])["model"] == "m2"


def test_models_use_without_running_server_still_persists(env_file):
    code, out, _ = run(
        ["--url", "http://127.0.0.1:9", "models", "use", "nli-mobilebert-int8"],
        env_file=env_file,
        registry=InstalledRegistry("nli-mobilebert-int8"),
        service=StoppedService(),
    )
    assert code == 0
    assert EnvFile(env_file).get("NOULO_MODEL") == "nli-mobilebert-int8"
    assert "not running" in out


def test_models_download_uses_registry(env_file, tmp_path):
    downloaded = []

    class FakeRegistry:
        def download(self, model_id, progress=None):
            downloaded.append(model_id)
            return tmp_path / model_id

    code, out, _ = run(
        ["models", "download", "nli-mobilebert-int8"], env_file=env_file, registry=FakeRegistry()
    )
    assert code == 0 and downloaded == ["nli-mobilebert-int8"]


def test_models_add_endpoint_persists_openai_model(env_file, tmp_path):
    env_file.write_text(f"NOULO_MODELS_FILE={tmp_path / 'models.json'}\n")
    code, _, _ = run(
        [
            "models",
            "add-endpoint",
            "--id",
            "ollama",
            "--base-url",
            "http://127.0.0.1:11434/v1",
            "--model",
            "qwen2.5:0.5b",
        ],
        env_file=env_file,
    )
    assert code == 0
    saved = json.loads((tmp_path / "models.json").read_text())
    assert saved["models"][0]["id"] == "ollama"


# ------------------------------------------------------------------ config


def test_config_set_and_get_normalise_keys(env_file):
    assert run(["config", "set", "port", "9000"], env_file=env_file)[0] == 0
    assert EnvFile(env_file).get("NOULO_PORT") == "9000"
    assert run(["config", "get", "NOULO_PORT"], env_file=env_file)[1].strip() == "9000"


def test_config_set_rejects_unknown_keys_and_bad_values(env_file):
    code, _, err = run(["config", "set", "nonsense", "1"], env_file=env_file)
    assert code == 2 and "Unknown setting" in err
    code, _, err = run(["config", "set", "port", "not-a-number"], env_file=env_file)
    assert code == 2


def test_config_set_preserves_other_lines_and_comments(env_file):
    env_file.write_text("# my settings\nNOULO_HOST=127.0.0.1\nOTHER=keep\n")
    run(["config", "set", "port", "9001"], env_file=env_file)
    text = env_file.read_text()
    assert "# my settings" in text and "OTHER=keep" in text and "NOULO_PORT=9001" in text


def test_config_unset_removes_key(env_file):
    env_file.write_text("NOULO_PORT=9001\n")
    run(["config", "unset", "port"], env_file=env_file)
    assert EnvFile(env_file).get("NOULO_PORT") is None


def test_config_show_redacts_secrets(env_file):
    env_file.write_text("NOULO_API_KEY=very-secret\n")
    code, out, _ = run(["config", "show", "--json"], env_file=env_file)
    data = json.loads(out)
    assert data["api_key"] == "***" and "very-secret" not in out


# ------------------------------------------------------------------ serve


def test_serve_builds_checked_settings_from_flags(env_file):
    captured = {}

    def fake_run(app, settings):
        captured["settings"] = settings

    code, _, _ = run(
        ["serve", "--port", "9123", "--model", "m9", "--no-learning"],
        env_file=env_file,
        run_server=fake_run,
        app_factory=lambda s: object(),
    )
    assert code == 0
    s = captured["settings"]
    assert (s.port, s.model, s.learning_enabled, s.ui_enabled) == (9123, "m9", False, False)


def test_serve_refuses_network_bind_without_opt_in(env_file):
    code, _, err = run(
        ["serve", "--host", "0.0.0.0"],
        env_file=env_file,
        run_server=lambda app, s: None,
        app_factory=lambda s: object(),
    )
    assert code == 2 and "NOULO_ALLOW_NETWORK" in err


def test_start_foreground_enables_ui_and_browser_unless_headless(env_file):
    seen = []
    run(
        ["start", "--foreground"],
        env_file=env_file,
        run_server=lambda app, s: seen.append(s),
        app_factory=lambda s: object(),
    )
    run(
        ["start", "--foreground", "--headless"],
        env_file=env_file,
        run_server=lambda app, s: seen.append(s),
        app_factory=lambda s: object(),
    )
    assert (seen[0].ui_enabled, seen[0].open_browser) == (True, True)
    assert (seen[1].ui_enabled, seen[1].open_browser) == (False, False)


def test_benchmark_passes_its_options_through(monkeypatch, env_file):
    seen = []
    monkeypatch.setattr("noulo.benchmark.run.main_and_exit", lambda argv: seen.append(argv))
    code, _, err = run(
        ["benchmark", "--model", "m1", "--tune", "--compare", "out.md"], env_file=env_file
    )
    assert code == 0, err
    assert seen == [["--model", "m1", "--tune", "--compare", "out.md"]]
