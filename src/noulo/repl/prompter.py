"""Terminal prompts: free text, yes/no, and an arrow-key selection menu."""

from __future__ import annotations

from typing import IO

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style

STYLE = Style.from_dict(
    {
        "title": "bold",
        "hint": "#888888",
        "pointer": "#d670d6 bold",
        "selected": "#d670d6 bold",
        "option": "",
        "desc": "#888888",
        "question": "#5fafd7",
    }
)
MAX_ROWS = 14


class TerminalPrompter:
    """prompt_toolkit-backed prompts used by the interactive session."""

    def ask(self, message: str, default: str = "") -> str:
        return pt_prompt([("class:question", f"  {message}: ")], default=default, style=STYLE)

    def confirm(self, message: str, default: bool = True) -> bool:
        suffix = " [Y/n] " if default else " [y/N] "
        answer = pt_prompt([("class:question", f"  {message}{suffix}")], style=STYLE)
        answer = answer.strip().lower()
        return default if not answer else answer in ("y", "yes")

    def select(
        self, title: str, options: list[tuple[str, str, str]], default: str | None = None
    ) -> str | None:
        if not options:
            return None
        values = [value for value, _, _ in options]
        state = {"index": values.index(default) if default in values else 0}

        def fragments():
            lines = [
                ("class:title", f"  {title}  "),
                ("class:hint", "↑/↓ move · 1-9 jump · enter select · esc cancel\n"),
            ]
            for index, (_, label, description) in enumerate(options):
                current = index == state["index"]
                if current:
                    lines.append(("[SetCursorPosition]", ""))
                lines.append(("class:pointer", "  ❯ ") if current else ("", "    "))
                lines.append(("class:selected" if current else "class:option", label))
                if description:
                    lines.append(("class:desc", f"   {description}"))
                lines.append(("", "\n"))
            return lines

        keys = KeyBindings()

        def move(step: int) -> None:
            state["index"] = (state["index"] + step) % len(options)

        keys.add("up")(lambda event: move(-1))
        keys.add("k")(lambda event: move(-1))
        keys.add("down")(lambda event: move(1))
        keys.add("j")(lambda event: move(1))
        keys.add("tab")(lambda event: move(1))

        @keys.add("enter")
        def _choose(event):
            event.app.exit(result=values[state["index"]])

        @keys.add("escape", eager=True)
        @keys.add("c-c")
        @keys.add("q")
        def _cancel(event):
            event.app.exit(result=None)

        for digit in range(1, 10):
            if digit <= len(options):

                @keys.add(str(digit))
                def _jump(event, index=digit - 1):
                    state["index"] = index

        window = Window(
            FormattedTextControl(fragments, focusable=True, show_cursor=False),
            height=Dimension(max=min(len(options) + 1, MAX_ROWS + 1)),
            wrap_lines=False,
        )
        app: Application = Application(
            layout=Layout(window),
            key_bindings=keys,
            style=STYLE,
            full_screen=False,
            erase_when_done=True,
        )
        return app.run()


class LinePrompter:
    """Plain line-based prompts for piped (non-TTY) sessions."""

    def __init__(self, stdin: IO[str], stdout: IO[str]):
        self.stdin, self.stdout = stdin, stdout

    def _line(self, message: str) -> str:
        self.stdout.write(f"  {message}: ")
        line = self.stdin.readline()
        if not line:
            raise EOFError
        return line.rstrip("\n")

    def ask(self, message: str, default: str = "") -> str:
        return self._line(message) or default

    def confirm(self, message: str, default: bool = True) -> bool:
        answer = self._line(f"{message} [y/n]").strip().lower()
        return default if not answer else answer in ("y", "yes")

    def select(
        self, title: str, options: list[tuple[str, str, str]], default: str | None = None
    ) -> str | None:
        for index, (_, label, description) in enumerate(options, 1):
            self.stdout.write(f"  {index}) {label}  {description}\n")
        answer = self._line(f"{title} [1-{len(options)}]").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        return default
