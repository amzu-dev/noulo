"""Interactive `noulo` shell with slash commands (/model, /noul, /learning, ...).

Every slash command maps onto the regular CLI, so the shell and the
non-interactive commands share one implementation.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from typing import IO

from . import __version__

HELP = """\
Commands:
  /model                 choose a model (3 curated options or plug in your own)
  /model use <id>        switch model (downloads and restarts if needed)
  /start  /stop  /restart  /status  /logs
  /noul                  is a proposition true for an input? (0-1)
  /choice                pick one supplied option
  /score                 place input on an ordered rubric (0-1)
  /learning on|off|status|records|clear --yes
  /feedback <recordId> <expected>
  /info  /health  /config show|get|set|unset
  /help  /quit"""

PASS_THROUGH = {
    "model",
    "models",
    "start",
    "stop",
    "restart",
    "status",
    "logs",
    "learning",
    "feedback",
    "info",
    "health",
    "config",
    "benchmark",
}


class Shell:
    def __init__(
        self, *, dispatch: Callable[[list[str]], int], ask: Callable[[str], str], stdout: IO[str]
    ):
        self._dispatch, self._ask, self._out = dispatch, ask, stdout

    def _print(self, text: str = "") -> None:
        print(text, file=self._out)

    def _prompt(self, label: str) -> str:
        return self._ask(f"  {label}: ").strip()

    def _evaluate(self, name: str, args: list[str]) -> list[str] | None:
        if args:
            return [name, *args]
        text = self._prompt("Input")
        if name == "noul":
            return ["noul", "--input", text, "--proposition", self._prompt("Proposition")]
        question = self._prompt("Question")
        if name == "choice":
            raw = self._prompt("Options (ID=text, comma-separated)")
            argv = ["choice", "--input", text, "--question", question]
            for option in (o.strip() for o in raw.split(",") if o.strip()):
                argv += ["--option", option]
            return argv
        rubric = ",".join(
            level.strip()
            for level in self._prompt("Rubric (lowest to highest, comma-separated)").split(",")
            if level.strip()
        )
        return ["score", "--input", text, "--question", question, "--rubric", rubric]

    def handle(self, line: str) -> bool:
        """Run one line; returns False when the shell should exit."""
        line = line.strip()
        if not line:
            return True
        if not line.startswith("/"):
            self._print("Commands start with '/'. Type /help to see them.")
            return True
        try:
            name, *args = shlex.split(line[1:])
        except ValueError as exc:
            self._print(f"Could not parse command: {exc}")
            return True
        if name in ("quit", "exit", "q"):
            return False
        if name == "help":
            self._print(HELP)
        elif name in ("noul", "choice", "score"):
            argv = self._evaluate(name, args)
            if argv:
                self._dispatch(argv)
        elif name in PASS_THROUGH:
            self._dispatch([name, *args])
        else:
            self._print(f"Unknown command /{name}. Type /help.")
        return True

    def loop(self) -> int:
        self._print(
            f"noulo {__version__} - local decision engine. Type /help for commands, /quit to exit."
        )
        while True:
            try:
                line = self._ask("noulo> ")
            except (EOFError, KeyboardInterrupt):
                self._print()
                return 0
            if not self.handle(line):
                return 0
