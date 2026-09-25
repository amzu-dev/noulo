"""Tab / as-you-type completion for slash commands and their arguments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from ..config import SETTING_HELP
from .commands import ALIASES, COMMANDS

CONFIG_ACTIONS = {
    "show": "List all settings",
    "get": "Show one setting",
    "set": "Change a setting",
    "unset": "Back to the default",
}


class SlashCompleter(Completer):
    def __init__(self, session: Any):
        self.session = session

    @staticmethod
    def _matching(word: str, items: Iterable[tuple[str, str]]) -> Iterable[Completion]:
        for value, meta in items:
            if value.startswith(word):
                yield Completion(value, start_position=-len(word), display_meta=meta)

    def _arguments(self, command: str, args: list[str], word: str) -> Iterable[tuple[str, str]]:
        position = len(args)  # index of the argument being typed
        if command == "config":
            if position == 0:
                return CONFIG_ACTIONS.items()
            if position == 1 and args[0] in ("get", "set", "unset"):
                return SETTING_HELP.items()
        elif command in ("learning",) and position == 0:
            return [("on", "Learn from inputs"), ("off", "Stop learning")]
        elif command == "memory" and position == 0:
            return [("clear", "Delete all remembered cases")]
        elif command == "model" and position == 0:
            models = self.session._models_cache
            return [
                ("list", "All known models"),
                *((m["id"], m.get("description", "")) for m in models),
            ]
        elif command == "correct" and position == 0 and self.session.last:
            body = self.session.last.get("body", {})
            if self.session.last.get("mode") == "choice":
                return [(c["id"], c["text"]) for c in body.get("choices", [])]
            if self.session.last.get("mode") == "score":
                return [(level, "") for level in body.get("rubric", [])]
            return [("true", ""), ("false", "")]
        return []

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        head, *rest = text[1:].split(" ")
        if not rest:
            yield from self._matching(head, ((c.name, c.help) for c in COMMANDS))
            return
        command = ALIASES.get(head, head)
        *args, word = rest
        yield from self._matching(word, self._arguments(command, args, word))
