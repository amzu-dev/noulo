"""Run the interactive `noulo` session (prompt_toolkit front end)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import IO

import httpx
from rich.console import Console

from ..cli import ApiClient
from ..config import Settings
from .session import ReplSession

HISTORY_FILE = Path(os.environ.get("NOULO_HISTORY_FILE", "~/.noulo/history")).expanduser()


def run_repl(
    *,
    env_file: Path,
    url: str | None = None,
    api_key: str | None = None,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    http_client: httpx.Client | None = None,
) -> int:
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    interactive = stdin.isatty() and stdout.isatty()
    settings = Settings(_env_file=env_file if env_file.exists() else None)
    base_url = url or os.environ.get("NOULO_URL") or settings.base_url
    key = api_key or (settings.api_key.get_secret_value() if settings.api_key else None)
    console = Console(file=stdout, highlight=False, soft_wrap=False)

    if interactive:
        from .prompter import TerminalPrompter

        prompter = TerminalPrompter()
    else:
        from .prompter import LinePrompter

        prompter = LinePrompter(stdin, stdout)

    def run_cli(argv: list[str]) -> int:
        from ..cli import main

        prefix = ["--url", base_url, *(["--api-key", key] if key else [])]
        return main(
            [*prefix, *argv],
            env_file=env_file,
            stdout=stdout,
            http_client=http_client,
            ask=lambda message: prompter.ask(message.strip().rstrip(":")),
        )

    session = ReplSession(
        api=ApiClient(base_url, key, http_client),
        console=console,
        prompter=prompter,
        run_cli=run_cli,
        env_file=env_file,
    )

    if not interactive:
        session.refresh_state()
        for line in stdin:
            session.handle(line)
            if not session.running:
                break
        return 0

    session.startup()
    return _interactive_loop(session)


def _interactive_loop(session: ReplSession) -> int:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style

    from .completion import SlashCompleter

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    session.models()  # warm the model-id cache for completion
    prompt = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        completer=SlashCompleter(session),
        complete_while_typing=True,
        bottom_toolbar=lambda: session.toolbar(),
        style=Style.from_dict(
            {"bottom-toolbar": "noreverse #888888 bg:default", "prompt": "#d670d6 bold"}
        ),
    )
    while session.running:
        try:
            line = prompt.prompt([("class:prompt", session.prompt_label())])
        except KeyboardInterrupt:
            continue  # Ctrl+C clears the line, like a shell
        except EOFError:
            break
        try:
            session.handle(line)
        except KeyboardInterrupt:
            session.say("  Cancelled.", "dim")
        except EOFError:
            break
    session.say("  Bye.", "dim")
    return 0
