import io

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from noulo.repl.completion import SlashCompleter
from noulo.repl.prompter import TerminalPrompter
from tests.test_repl import api, make  # noqa: F401  (fixture re-export)


def completions(completer, text):
    return [c.text for c in completer.get_completions(Document(text), CompleteEvent())]


@pytest.fixture
def session(api, tmp_path):  # noqa: F811
    s, _ = make(api, tmp_path)
    return s


def test_slash_lists_every_command_with_help(session):
    completer = SlashCompleter(session)
    found = list(completer.get_completions(Document("/"), CompleteEvent()))
    names = [c.text for c in found]
    assert "noul" in names and "config" in names and "model" in names
    assert all(c.display_meta_text for c in found)


def test_command_prefix_completion(session):
    assert set(completions(SlashCompleter(session), "/co")) == {"config", "context", "correct"}


def test_config_subcommands_and_setting_names(session):
    completer = SlashCompleter(session)
    assert {"show", "get", "set", "unset"} <= set(completions(completer, "/config "))
    assert "port" in completions(completer, "/config set po")
    assert "max_queue" in completions(completer, "/config get max_q")


def test_model_ids_complete_from_listing(session):
    session.models()
    ids = completions(SlashCompleter(session), "/model nli-mini")
    assert ids == ["nli-minilm2-l6-int8"]


def test_learning_and_memory_arguments(session):
    completer = SlashCompleter(session)
    assert completions(completer, "/learning ") == ["on", "off"]
    assert completions(completer, "/memory ") == ["clear"]


def test_correct_completes_current_options(session):
    session.mode, session.choices = "choice", [{"id": "A", "text": "x"}, {"id": "B", "text": "y"}]
    session.last = {"mode": "choice", "body": {"choices": session.choices}}
    assert completions(SlashCompleter(session), "/correct ") == ["A", "B"]


def test_plain_text_gets_no_completions(session):
    assert completions(SlashCompleter(session), "hello") == []


# ---------------------------------------------------------------- terminal prompter


@pytest.fixture
def keys():
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        yield pipe


OPTIONS = [("a", "Alpha", "first"), ("b", "Beta", "second"), ("c", "Gamma", "third")]


def test_select_moves_with_arrows_and_confirms_with_enter(keys):
    keys.send_text("\x1b[B\x1b[B\r")  # down, down, enter
    assert TerminalPrompter().select("Pick", OPTIONS) == "c"


def test_select_starts_on_default(keys):
    keys.send_text("\r")
    assert TerminalPrompter().select("Pick", OPTIONS, default="b") == "b"


def test_select_number_key_jumps(keys):
    keys.send_text("2\r")
    assert TerminalPrompter().select("Pick", OPTIONS) == "b"


def test_select_ctrl_c_cancels(keys):
    keys.send_text("\x03")
    assert TerminalPrompter().select("Pick", OPTIONS) is None


def test_ask_returns_typed_text_or_default(keys):
    keys.send_text("hello\r")
    assert TerminalPrompter().ask("Name") == "hello"
    keys.send_text("\r")
    assert TerminalPrompter().ask("Name", default="world") == "world"


def test_confirm_defaults_and_answers(keys):
    keys.send_text("\r")
    assert TerminalPrompter().confirm("Go?", default=True) is True
    keys.send_text("n\r")
    assert TerminalPrompter().confirm("Go?", default=True) is False


# ---------------------------------------------------------------- non-interactive session


def test_piped_session_processes_lines(api, tmp_path):  # noqa: F811
    from noulo.repl.app import run_repl

    stdin = io.StringIO("/noul The customer is happy.\nI love it!\n/exit\nnot reached\n")
    stdout = io.StringIO()
    code = run_repl(
        env_file=tmp_path / ".env",
        url="http://testserver",
        stdin=stdin,
        stdout=stdout,
        http_client=api,
    )
    assert code == 0
    assert "0.90" in stdout.getvalue()
    assert "not reached" not in stdout.getvalue()


def test_noulo_without_command_opens_the_session(monkeypatch, tmp_path):
    from noulo.cli import main

    seen = {}
    monkeypatch.setattr("noulo.repl.app.run_repl", lambda **kwargs: seen.update(kwargs) or 0)
    assert main([], env_file=tmp_path / ".env", stdout=io.StringIO()) == 0
    assert seen["env_file"] == tmp_path / ".env"
