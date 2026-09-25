import io
import json

import pytest
from fastapi.testclient import TestClient
from rich.console import Console

from noulo.api.server import create_app
from noulo.cli import ApiClient, EnvFile
from noulo.config import Settings
from noulo.inference.embedder import HashingEmbedder
from noulo.inference.engine import DecisionEngine
from noulo.inference.memory import LearningMemory
from noulo.inference.stores import open_store
from noulo.registry import CURATED_MODELS, ModelRegistry
from noulo.repl.session import ReplSession
from tests.fakes import FakeBackend, FakeLoader


class FakePrompter:
    """Scripted answers for ask/confirm/select; records what was asked."""

    def __init__(self, asks=(), confirms=(), selects=()):
        self.asks, self.confirms, self.selects = list(asks), list(confirms), list(selects)
        self.questions: list[str] = []
        self.menus: list[list] = []

    def ask(self, message, default=""):
        self.questions.append(message)
        return self.asks.pop(0)

    def confirm(self, message, default=True):
        self.questions.append(message)
        return self.confirms.pop(0) if self.confirms else default

    def select(self, title, options, default=None):
        self.menus.append(options)
        return self.selects.pop(0)


@pytest.fixture
def api(tmp_path):
    engine = DecisionEngine(
        load_backend=FakeLoader(
            m=FakeBackend("m", noul_value=0.9, choice_probs=[0.2, 0.8], score_probs=[0.0, 0.0, 1.0])
        ),
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


def make(api, tmp_path, prompter=None, cli_calls=None, **kwargs):
    out = io.StringIO()
    console = Console(file=out, width=110, force_terminal=False, color_system=None)
    calls = cli_calls if cli_calls is not None else []
    session = ReplSession(
        api=ApiClient("http://testserver", None, api)
        if api is not None
        else ApiClient("http://127.0.0.1:9", None),
        console=console,
        prompter=prompter or FakePrompter(),
        run_cli=lambda argv: calls.append(argv) or 0,
        env_file=tmp_path / ".env",
        **kwargs,
    )
    return session, out


# ---------------------------------------------------------------- simple requests


def test_plain_text_without_mode_walks_through_setup_then_evaluates(api, tmp_path):
    prompter = FakePrompter(selects=["noul"], asks=["The customer is unhappy."])
    session, out = make(api, tmp_path, prompter)
    session.handle("My order arrived broken and nobody answers my emails.")
    assert session.mode == "noul" and session.proposition == "The customer is unhappy."
    assert "Yes" in out.getvalue() and "0.90" in out.getvalue()


def test_mode_is_sticky_for_following_inputs(api, tmp_path):
    session, out = make(api, tmp_path)
    session.handle("/noul The customer has an overdue payment.")
    session.handle("The invoice has been unpaid for 120 days.")
    session.handle("Another invoice is late.")
    assert out.getvalue().count("0.90") == 2


def test_noul_without_argument_prompts_for_proposition(api, tmp_path):
    prompter = FakePrompter(asks=["The shipment is late."])
    session, _ = make(api, tmp_path, prompter)
    session.handle("/noul")
    assert session.mode == "noul" and session.proposition == "The shipment is late."


def test_choice_mode_prompts_for_options_and_shows_winner(api, tmp_path):
    prompter = FakePrompter(asks=["A=Sales, B=Billing"])
    session, out = make(api, tmp_path, prompter)
    session.handle("/choice Which department should handle this?")
    session.handle("I was charged twice.")
    text = out.getvalue()
    assert "B" in text and "Billing" in text
    assert session.choices == [{"id": "A", "text": "Sales"}, {"id": "B", "text": "Billing"}]


def test_score_mode_shows_value_and_nearest_level(api, tmp_path):
    prompter = FakePrompter(asks=["low, medium, high"])
    session, out = make(api, tmp_path, prompter)
    session.handle("/score How severe is this incident?")
    session.handle("Everything is down.")
    assert "1.00" in out.getvalue() and "high" in out.getvalue()


def test_context_can_be_edited_in_place(api, tmp_path):
    session, _ = make(api, tmp_path, FakePrompter(asks=["A=x, B=y"]))
    session.handle("/choice Which?")
    session.handle("/options A=Billing, B=Sales, C=Support")
    session.handle("/question Which team?")
    assert [c["text"] for c in session.choices] == ["Billing", "Sales", "Support"]
    assert session.question == "Which team?"


def test_context_command_shows_current_mode(api, tmp_path):
    session, out = make(api, tmp_path)
    session.handle("/noul The customer is happy.")
    session.handle("/context")
    assert "noul" in out.getvalue() and "The customer is happy." in out.getvalue()


def test_api_errors_are_shown_without_crashing(api, tmp_path):
    session, out = make(api, tmp_path)
    session.handle("/noul " + "p" * 2000)  # proposition over the 1000-char limit
    session.handle("some input")
    assert "INPUT_TOO_LARGE" in out.getvalue()


def test_unreachable_server_suggests_start(tmp_path):
    session, out = make(None, tmp_path)
    session.handle("/noul p")
    session.handle("text")
    assert "/start" in out.getvalue()


def test_json_toggle_prints_raw_response(api, tmp_path):
    session, out = make(api, tmp_path)
    session.handle("/json")
    session.handle("/noul p")
    session.handle("text")
    last_json = out.getvalue()[out.getvalue().index("{") :]
    assert json.loads(last_json)["type"] == "noul"


def test_verbose_toggle_shows_probabilities(api, tmp_path):
    session, out = make(api, tmp_path, FakePrompter(asks=["A=Sales, B=Billing"]))
    session.handle("/verbose")
    session.handle("/choice Which?")
    session.handle("text")
    assert "0.80" in out.getvalue() and "0.20" in out.getvalue()


# ---------------------------------------------------------------- feedback


def records(api):
    return api.get("/api/v1/learning/records").json()["records"]


def test_good_confirms_last_result(api, tmp_path):
    session, out = make(api, tmp_path)
    session.handle("/noul p")
    session.handle("text")
    session.handle("/good")
    assert records(api)[0]["verifiedValue"] == 1.0
    assert "Thanks" in out.getvalue()


def test_bad_on_noul_records_the_opposite(api, tmp_path):
    session, _ = make(api, tmp_path)
    session.handle("/noul p")
    session.handle("text")
    session.handle("/bad")
    assert records(api)[0]["verifiedValue"] == 0.0


def test_bad_on_choice_asks_for_the_right_option(api, tmp_path):
    prompter = FakePrompter(asks=["A=Sales, B=Billing"], selects=["A"])
    session, _ = make(api, tmp_path, prompter)
    session.handle("/choice Which?")
    session.handle("text")
    session.handle("/bad")
    assert records(api)[0]["verifiedChoice"] == "A"
    assert [o[0] for o in prompter.menus[-1]] == ["A"]  # only the options it didn't pick


def test_correct_with_explicit_level_for_score(api, tmp_path):
    session, _ = make(api, tmp_path, FakePrompter(asks=["low, medium, high"]))
    session.handle("/score How bad?")
    session.handle("text")
    session.handle("/correct medium")
    assert records(api)[0]["verifiedValue"] == pytest.approx(0.5)


def test_feedback_before_any_result_explains(api, tmp_path):
    session, out = make(api, tmp_path)
    session.handle("/good")
    assert "Nothing to give feedback on yet" in out.getvalue()


# ---------------------------------------------------------------- learning / memory


def test_learning_status_and_toggle_delegate_to_cli(api, tmp_path):
    calls = []
    session, out = make(api, tmp_path, cli_calls=calls)
    session.handle("/learning")
    assert "on" in out.getvalue().lower()
    session.handle("/learning off")
    assert calls == [["learning", "off"]]


def test_memory_lists_records(api, tmp_path):
    session, out = make(api, tmp_path)
    session.handle("/noul The customer is happy.")
    session.handle("I love this product")
    session.handle("/memory")
    assert "I love this product" in out.getvalue()


def test_memory_clear_asks_for_confirmation(api, tmp_path):
    session, _ = make(api, tmp_path, FakePrompter(confirms=[False, True]))
    session.handle("/noul p")
    session.handle("text")
    session.handle("/memory clear")
    assert len(records(api)) == 1
    session.handle("/memory clear")
    assert records(api) == []


# ---------------------------------------------------------------- configuration


def test_config_show_lists_settings_with_secrets_redacted(api, tmp_path):
    (tmp_path / ".env").write_text("NOULO_API_KEY=super-secret\nNOULO_PORT=9123\n")
    session, out = make(api, tmp_path)
    session.handle("/config show")
    text = out.getvalue()
    assert "port" in text and "9123" in text and "super-secret" not in text


def test_config_set_saves_and_asks_to_restart(api, tmp_path):
    calls = []
    session, out = make(api, tmp_path, FakePrompter(confirms=[True]), cli_calls=calls)
    session.handle("/config set max_queue 64")
    assert EnvFile(tmp_path / ".env").get("NOULO_MAX_QUEUE") == "64"
    assert calls == [["restart"]]


def test_config_set_rejects_invalid_values(api, tmp_path):
    session, out = make(api, tmp_path)
    session.handle("/config set port not-a-number")
    assert EnvFile(tmp_path / ".env").get("NOULO_PORT") is None
    assert "Invalid" in out.getvalue()


def test_config_set_live_setting_applies_without_restart(api, tmp_path):
    calls = []
    session, _ = make(api, tmp_path, cli_calls=calls)
    session.handle("/config set learning_enabled false")
    assert calls == [["learning", "off"]]


def test_config_interactive_editor(api, tmp_path):
    prompter = FakePrompter(selects=["max_choices"], asks=["12"], confirms=[False])
    session, _ = make(api, tmp_path, prompter)
    session.handle("/config")
    assert EnvFile(tmp_path / ".env").get("NOULO_MAX_CHOICES") == "12"
    labels = [label for _, label, _ in prompter.menus[0]]
    assert any(label.startswith("port") for label in labels)


def test_config_unset(api, tmp_path):
    (tmp_path / ".env").write_text("NOULO_MAX_QUEUE=64\n")
    session, _ = make(api, tmp_path, FakePrompter(confirms=[False]))
    session.handle("/config unset max_queue")
    assert EnvFile(tmp_path / ".env").get("NOULO_MAX_QUEUE") is None


# ---------------------------------------------------------------- models / service


def test_model_picker_offers_curated_models_and_switches(api, tmp_path):
    calls = []
    target = CURATED_MODELS[1].id
    prompter = FakePrompter(selects=[target])
    session, _ = make(api, tmp_path, prompter, cli_calls=calls)
    session.handle("/model")
    values = [value for value, _, _ in prompter.menus[0]]
    assert values[:3] == [m.id for m in CURATED_MODELS] and values[-1] == "__own__"
    assert calls == [["model", "use", target]]


def test_model_picker_own_model_prints_guide(api, tmp_path):
    session, out = make(api, tmp_path, FakePrompter(selects=["__own__"]))
    session.handle("/model")
    assert "add-endpoint" in out.getvalue()


def test_model_with_id_switches_directly(api, tmp_path):
    calls = []
    session, _ = make(api, tmp_path, cli_calls=calls)
    session.handle("/model nli-minilm2-l6-int8")
    assert calls == [["model", "use", "nli-minilm2-l6-int8"]]


@pytest.mark.parametrize(
    "command,argv",
    [
        ("/start", ["start", "--no-browser"]),
        ("/stop", ["stop"]),
        ("/restart", ["restart"]),
        ("/status", ["status"]),
        ("/logs", ["logs"]),
    ],
)
def test_service_commands_delegate_to_cli(api, tmp_path, command, argv):
    calls = []
    session, _ = make(api, tmp_path, cli_calls=calls)
    session.handle(command)
    assert calls == [argv]


# ---------------------------------------------------------------- shell basics


def test_help_lists_commands_with_descriptions(api, tmp_path):
    session, out = make(api, tmp_path)
    session.handle("/help")
    for command in ("/noul", "/choice", "/score", "/config", "/model", "/learning", "/good"):
        assert command in out.getvalue()


def test_unknown_command(api, tmp_path):
    session, out = make(api, tmp_path)
    session.handle("/nope")
    assert "Unknown command /nope" in out.getvalue()


def test_exit_stops_the_session(api, tmp_path):
    session, _ = make(api, tmp_path)
    session.handle("/exit")
    assert session.running is False


def test_startup_banner_shows_server_and_model(api, tmp_path):
    session, out = make(api, tmp_path)
    session.startup()
    assert "noulo" in out.getvalue() and "m" in out.getvalue() and "ready" in out.getvalue()


def test_startup_offers_to_start_when_not_running(tmp_path):
    calls = []
    session, out = make(None, tmp_path, FakePrompter(confirms=[True]), cli_calls=calls)
    session.startup()
    assert calls == [["start", "--no-browser"]]


def test_toolbar_summarises_state(api, tmp_path):
    session, _ = make(api, tmp_path)
    session.startup()
    session.handle("/noul The customer is happy.")
    bar = session.toolbar()
    assert "ready" in bar and "m" in bar and "learning on" in bar and "noul" in bar
