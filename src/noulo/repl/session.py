"""The interactive session core: state, slash commands and evaluation.

Terminal concerns (line editing, menus, completion) live in `app.py` /
`prompter.py`; this class only talks to an API client, a rich Console, a
Prompter (ask/confirm/select) and the regular CLI (`run_cli`), so it can be
tested without a terminal.
"""

from __future__ import annotations

import json
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .. import __version__
from ..cli import PLUGIN_GUIDE, ApiClient, ApiError, CliError, EnvFile, Unreachable, setting_field
from ..config import LIVE_SETTINGS, SECRET_FIELDS, SETTING_HELP, Settings
from ..model_menu import OWN_MODEL, menu_items
from . import render
from .commands import BY_NAME, COMMANDS, resolve

PRIMITIVE_MENU = [
    ("noul", "Noul", "Is a statement true for the input? (0-1)"),
    ("choice", "Choice", "Which of your options fits the input?"),
    ("score", "Score", "Where does the input sit on your rubric? (0-1)"),
]


class Prompter(Protocol):
    def ask(self, message: str, default: str = "") -> str: ...

    def confirm(self, message: str, default: bool = True) -> bool: ...

    def select(
        self, title: str, options: list[tuple[str, str, str]], default: str | None = None
    ) -> str | None: ...


def parse_options(raw: str) -> list[dict[str, str]]:
    """'A=Billing, B=Sales' or 'Billing, Sales' (ids A, B, ... assigned)."""
    options = []
    for index, part in enumerate(p.strip() for p in raw.split(",") if p.strip()):
        if "=" in part:
            option_id, text = (s.strip() for s in part.split("=", 1))
        else:
            option_id, text = chr(ord("A") + index), part
        options.append({"id": option_id, "text": text})
    return options


def parse_list(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


class ReplSession:
    def __init__(
        self,
        *,
        api: ApiClient,
        console: Console,
        prompter: Prompter,
        run_cli: Callable[[list[str]], int],
        env_file: Path,
        browser: Callable[[str], Any] = webbrowser.open,
    ):
        self.api, self.console, self.prompter = api, console, prompter
        self.run_cli, self.env_file, self.browser = run_cli, Path(env_file), browser
        self.mode: str | None = None
        self.proposition: str | None = None
        self.question: str | None = None
        self.choices: list[dict[str, str]] = []
        self.rubric: list[str] = []
        self.last: dict[str, Any] | None = None
        self.json_output = False
        self.verbose = False
        self.running = True
        self.state: dict[str, Any] = {"server": "unknown", "model": None, "learning": None}
        self._models_cache: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ output helpers

    def say(self, message: str, style: str = "") -> None:
        self.console.print(Text(message, style=style))

    def error(self, message: str) -> None:
        self.console.print(Text("  ✖ ", style="bold red") + Text(message, style="red"))

    def ok(self, message: str) -> None:
        self.console.print(Text("  ✔ ", style="bold green") + Text(message))

    # ------------------------------------------------------------------ state

    def refresh_state(self) -> None:
        try:
            info, _ = self.api.call("GET", "/api/v1/info")
            self.state = {
                "server": "ready",
                "model": info["model"],
                "learning": info["learning"],
                "quantization": info["quantization"],
                "backend": info["backend"],
                "device": info.get("device", "cpu"),
            }
        except Unreachable:
            self.state = {"server": "offline", "model": None, "learning": None}
        except ApiError as exc:
            self.state = {"server": exc.api_code.lower(), "model": None, "learning": None}

    def context_summary(self) -> str:
        if self.mode == "noul":
            return f"noul: {self.proposition}"
        if self.mode == "choice":
            ids = ", ".join(c["id"] for c in self.choices)
            return f"choice: {self.question} [{ids}]"
        if self.mode == "score":
            return f"score: {self.question} [{' < '.join(self.rubric)}]"
        return "no mode (type text to choose)"

    def toolbar(self) -> str:
        server = self.state["server"]
        parts = [f"● {server}"]
        if self.state.get("model"):
            parts.append(f"{self.state['model']} on {self.state.get('device', 'cpu')}")
        if self.state.get("learning") is not None:
            parts.append(f"learning {'on' if self.state['learning'] else 'off'}")
        summary = self.context_summary()
        parts.append(summary if len(summary) <= 60 else summary[:57] + "...")
        parts.append("/help")
        return "  " + "  │  ".join(parts)

    def prompt_label(self) -> str:
        return f"{self.mode} ❯ " if self.mode else "❯ "

    # ------------------------------------------------------------------ lifecycle

    def startup(self) -> None:
        self.refresh_state()
        if self.state["server"] == "offline":
            self.say(f"noulo isn't running at {self.api.base_url}.", "yellow")
            if self.prompter.confirm("Start it now?", default=True):
                self.run_cli(["start", "--no-browser"])
                self.refresh_state()
        self.banner()

    def banner(self) -> None:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim")
        grid.add_column()
        server = self.state["server"]
        color = "green" if server == "ready" else "yellow"
        grid.add_row("server", Text(f"● {server}", style=color) + Text(f"  {self.api.base_url}"))
        if self.state.get("model"):
            grid.add_row("model", f"{self.state['model']} ({self.state.get('quantization', '')})")
            grid.add_row("device", self.state.get("device", "cpu"))
            grid.add_row("learning", "on" if self.state["learning"] else "off")
        title = (
            Text("✻ ", style="bold magenta")
            + Text(f"noulo {__version__}", style="bold")
            + Text(" - local decision engine", style="dim")
        )
        self.console.print(
            Panel(
                grid,
                title=title,
                title_align="left",
                border_style="magenta",
                expand=False,
                padding=(1, 2),
            )
        )
        self.say(
            "  Type a sentence to evaluate it · /noul, /choice, /score set the mode · "
            "/model · /config · /help · Ctrl+D exits",
            "dim",
        )

    # ------------------------------------------------------------------ dispatch

    def handle(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            if line.startswith("/"):
                name, _, rest = line[1:].partition(" ")
                command = resolve(name.lower())
                if command is None:
                    self.error(f"Unknown command /{name}. Type /help to see what's available.")
                    return
                getattr(self, f"cmd_{command}")(rest.strip())
            else:
                self.evaluate(line)
        except Unreachable:
            self.error(f"Can't reach noulo at {self.api.base_url}. Start it with /start.")
            self.state["server"] = "offline"
        except CliError as exc:
            self.error(str(exc))

    # ------------------------------------------------------------------ evaluation

    def _choose_mode(self) -> bool:
        kind = self.prompter.select("What do you want to check?", PRIMITIVE_MENU, default="noul")
        if kind is None:
            return False
        getattr(self, f"cmd_{kind}")("")
        return self.mode is not None

    def _body(self, text: str) -> dict[str, Any]:
        if self.mode == "noul":
            return {"input": text, "proposition": self.proposition}
        if self.mode == "choice":
            return {"input": text, "question": self.question, "choices": self.choices}
        return {"input": text, "question": self.question, "rubric": self.rubric}

    def evaluate(self, text: str) -> None:
        if self.mode is None and not self._choose_mode():
            return
        body = self._body(text)
        started = time.perf_counter()
        with self.console.status("thinking...", spinner="dots"):
            data, headers = self.api.call("POST", f"/api/v1/{self.mode}?diagnostics=true", body)
        latency = (time.perf_counter() - started) * 1000
        record_id = headers.get("X-Record-Id")
        self.last = {"mode": self.mode, "body": body, "data": data, "record_id": record_id}
        self.state["server"] = "ready"
        if self.json_output:
            payload = {k: v for k, v in data.items() if k != "diagnostics" or self.verbose}
            if record_id:
                payload["recordId"] = record_id
            self.console.print_json(json.dumps(payload))
            return
        self._render(data, latency)

    def _render(self, data: dict[str, Any], latency: float) -> None:
        diagnostics = data.get("diagnostics")
        value = data["value"]
        if self.mode == "noul":
            self.console.print(render.noul(value))
        elif self.mode == "choice":
            labels = {c["id"]: c["text"] for c in self.choices}
            self.console.print(render.choice(value, labels.get(value, "")))
        else:
            self.console.print(render.score(value, self.rubric))
        if self.verbose:
            labels = {c["id"]: c["text"] for c in self.choices} if self.mode == "choice" else None
            for row in render.probabilities(diagnostics, labels):
                self.console.print(row)
            if diagnostics and "raw" in diagnostics and self.mode == "noul":
                self.say(f"    raw (uncalibrated) {diagnostics['raw']:.2f}", "dim")
        self.console.print(render.footer(latency, diagnostics))

    # ------------------------------------------------------------------ modes

    def _ask(self, message: str, default: str = "") -> str:
        return self.prompter.ask(message, default).strip()

    def cmd_noul(self, rest: str) -> None:
        proposition = rest or self._ask("Statement to check (proposition)")
        if not proposition:
            return
        self.mode, self.proposition = "noul", proposition
        self.say(f"  Noul mode - every line you type is checked against: “{proposition}”", "cyan")

    def cmd_choice(self, rest: str) -> None:
        question = rest or self._ask("Question")
        if not question:
            return
        options = parse_options(self._ask("Options (A=Billing, B=Sales - or just Billing, Sales)"))
        if not options:
            self.error("Choice needs at least one option.")
            return
        self.mode, self.question, self.choices = "choice", question, options
        listing = ", ".join(f"{o['id']}={o['text']}" for o in options)
        self.say(f"  Choice mode - “{question}” options: {listing}", "cyan")

    def cmd_score(self, rest: str) -> None:
        question = rest or self._ask("Question")
        if not question:
            return
        rubric = parse_list(self._ask("Rubric, lowest to highest (low, medium, high)"))
        if len(rubric) < 2:
            self.error("A rubric needs at least two levels.")
            return
        self.mode, self.question, self.rubric = "score", question, rubric
        self.say(f"  Score mode - “{question}” on {' < '.join(rubric)}", "cyan")

    def cmd_context(self, rest: str) -> None:
        self.say(f"  {self.context_summary()}", "cyan")

    def cmd_proposition(self, rest: str) -> None:
        self.cmd_noul(rest)

    def cmd_question(self, rest: str) -> None:
        if self.mode not in ("choice", "score"):
            self.error("Set a Choice or Score mode first (/choice or /score).")
            return
        if rest:
            self.question = rest
            self.cmd_context("")

    def cmd_options(self, rest: str) -> None:
        options = parse_options(rest or self._ask("Options"))
        if options:
            self.mode, self.choices = "choice", options
            self.question = self.question or self._ask("Question")
            self.cmd_context("")

    def cmd_rubric(self, rest: str) -> None:
        rubric = parse_list(rest or self._ask("Rubric, lowest to highest"))
        if len(rubric) >= 2:
            self.mode, self.rubric = "score", rubric
            self.question = self.question or self._ask("Question")
            self.cmd_context("")

    # ------------------------------------------------------------------ feedback

    def _send_feedback(self, expected: Any, verdict: str) -> None:
        if self.last is None:
            self.say("  Nothing to give feedback on yet - evaluate something first.", "yellow")
            return
        if not self.last["record_id"]:
            self.say(
                "  Learning is off, so there's nothing to teach. Turn it on with /learning on.",
                "yellow",
            )
            return
        self.api.call(
            "POST", "/api/v1/feedback", {"recordId": self.last["record_id"], "expected": expected}
        )
        self.ok(f"Thanks - remembered as {verdict}. Similar inputs will lean this way.")

    def cmd_good(self, rest: str) -> None:
        if self.last is None:
            return self._send_feedback(None, "")
        value = self.last["data"]["value"]
        expected = value >= 0.5 if self.last["mode"] == "noul" else value
        self._send_feedback(expected, "correct")

    def cmd_bad(self, rest: str) -> None:
        if self.last is None:
            return self._send_feedback(None, "")
        mode, value = self.last["mode"], self.last["data"]["value"]
        if mode == "noul":
            return self._send_feedback(not value >= 0.5, "wrong (opposite recorded)")
        if mode == "choice":
            others = [
                (c["id"], f"{c['id']}  {c['text']}", "")
                for c in self.last["body"]["choices"]
                if c["id"] != value
            ]
            picked = self.prompter.select("Which option was right?", others)
            if picked is not None:
                self._send_feedback(picked, f"'{picked}'")
            return
        rubric = self.last["body"]["rubric"]
        levels = [(str(i), level, "") for i, level in enumerate(rubric)]
        picked = self.prompter.select("Which level was right?", levels)
        if picked is not None:
            index = int(picked)
            self._send_feedback(index / (len(rubric) - 1), f"'{rubric[index]}'")

    def cmd_correct(self, rest: str) -> None:
        if self.last is None:
            return self._send_feedback(None, "")
        mode = self.last["mode"]
        answer = rest or self._ask("Right answer")
        if mode == "noul":
            lowered = answer.lower()
            expected: Any = {"true": True, "yes": True, "false": False, "no": False}.get(lowered)
            if expected is None:
                expected = float(answer)
        elif mode == "choice":
            ids = {c["id"]: c["id"] for c in self.last["body"]["choices"]}
            ids.update({c["text"].lower(): c["id"] for c in self.last["body"]["choices"]})
            expected = ids.get(answer) or ids.get(answer.lower())
            if expected is None:
                self.error(f"'{answer}' is not one of the options.")
                return
        else:
            rubric = [level.lower() for level in self.last["body"]["rubric"]]
            if answer.lower() in rubric:
                expected = rubric.index(answer.lower()) / (len(rubric) - 1)
            else:
                expected = float(answer)
        self._send_feedback(expected, f"'{answer}'")

    # ------------------------------------------------------------------ learning

    def cmd_teach(self, rest: str) -> None:
        path = rest or self._ask("File of labelled examples (.jsonl)")
        if path:
            self.run_cli(["teach", path])

    def cmd_learning(self, rest: str) -> None:
        if rest in ("on", "off"):
            self.run_cli(["learning", rest])
            self.refresh_state()
            return
        data, _ = self.api.call("GET", "/api/v1/learning")
        stats = data.get("stats") or {}
        state = "on" if data["enabled"] else "off"
        details = (
            (
                f" · {stats.get('records', 0)} cases ({stats.get('verified', 0)} verified) · "
                f"embedder {stats.get('embedder')}"
            )
            if stats
            else ""
        )
        self.say(f"  Learning is {state}{details}. Switch with /learning on|off.", "cyan")

    def cmd_memory(self, rest: str) -> None:
        if rest == "clear":
            data, _ = self.api.call("GET", "/api/v1/learning")
            count = (data.get("stats") or {}).get("records", 0)
            if self.prompter.confirm(f"Delete all {count} remembered cases?", default=False):
                deleted, _ = self.api.call("DELETE", "/api/v1/learning/records")
                self.ok(f"Deleted {deleted['deleted']} cases.")
            return
        data, _ = self.api.call("GET", "/api/v1/learning/records?limit=20")
        records = data["records"]
        if not records:
            self.say("  No remembered cases yet.", "dim")
            return
        table = Table(box=None, header_style="dim", pad_edge=False)
        for column in ("type", "task", "input", "answer", "taught"):
            table.add_column(
                column,
                overflow="ellipsis",
                no_wrap=True,
                max_width=40 if column in ("task", "input") else None,
            )
        for r in records:
            observed = (
                r["observedChoice"] if r["observedValue"] is None else f"{r['observedValue']:.2f}"
            )
            verified = (
                r["verifiedChoice"] if r["verifiedValue"] is None else f"{r['verifiedValue']:.2f}"
            )
            table.add_row(
                r["primitive"],
                escape(r["task"].split("|", 1)[-1]),
                escape(r["input"]),
                str(observed),
                str(verified) if verified is not None else "",
            )
        self.console.print(table)

    # ------------------------------------------------------------------ configuration

    def _settings(self) -> Settings:
        return Settings(_env_file=self.env_file if self.env_file.exists() else None)

    def cmd_config(self, rest: str) -> None:
        action, _, args = rest.partition(" ")
        args = args.strip()
        if action == "":
            return self._config_editor()
        if action == "show":
            return self._config_table()
        if action == "get":
            field = setting_field(args)
            self.say(f"  {field} = {self._settings().public_dict()[field]}")
            return
        if action == "set":
            key, _, value = args.partition(" ")
            return self._set_setting(setting_field(key), value.strip())
        if action == "unset":
            field = setting_field(args)
            EnvFile(self.env_file).unset(f"NOULO_{field.upper()}")
            self.ok(f"Removed NOULO_{field.upper()} (back to the default).")
            self._offer_restart()
            return
        self.error("Usage: /config [show | get <key> | set <key> <value> | unset <key>]")

    def _config_table(self) -> None:
        table = Table(box=None, header_style="dim", pad_edge=False)
        table.add_column("setting")
        table.add_column("value", overflow="fold")
        table.add_column("", style="dim")
        for field, value in self._settings().public_dict().items():
            table.add_row(field, escape(str(value)), SETTING_HELP.get(field, ""))
        self.console.print(table)
        self.say(
            "  Change one with /config set <key> <value>, or just /config for the editor.", "dim"
        )

    def _config_editor(self) -> None:
        values = self._settings().public_dict()
        options = [
            (field, f"{field} = {values[field]}", SETTING_HELP.get(field, "")) for field in values
        ]
        field = self.prompter.select("Which setting?", options)
        if field is None:
            return
        current = "" if field in SECRET_FIELDS or values[field] is None else str(values[field])
        value = self._ask(f"{field} ({SETTING_HELP.get(field, '')})", current)
        if value == current:
            self.say("  Unchanged.", "dim")
            return
        self._set_setting(field, value)

    def _set_setting(self, field: str, value: str) -> None:
        try:
            parsed = Settings(_env_file=None, **{field: value})
        except ValidationError as exc:
            self.error(f"Invalid value for {field}: {exc.errors()[0]['msg']}")
            return
        if field in LIVE_SETTINGS:
            if field == "model":
                self.run_cli(["model", "use", value])
            else:
                self.run_cli(["learning", "on" if parsed.learning_enabled else "off"])
            self.refresh_state()
            return
        EnvFile(self.env_file).set(f"NOULO_{field.upper()}", value)
        shown = "***" if field in SECRET_FIELDS else value
        self.ok(f"Saved NOULO_{field.upper()}={shown} to {self.env_file.name}.")
        self._offer_restart()

    def _offer_restart(self) -> None:
        self.refresh_state()
        if self.state["server"] == "offline":
            self.say("  It will apply the next time noulo starts.", "dim")
        elif self.prompter.confirm("Restart noulo now to apply it?", default=True):
            self.run_cli(["restart"])
            self.refresh_state()
        else:
            self.say("  Apply it later with /restart.", "dim")

    # ------------------------------------------------------------------ models

    def models(self) -> list[dict[str, Any]]:
        try:
            data, _ = self.api.call("GET", "/api/v1/models")
            self._models_cache = data["models"]
            self.state["model"] = data["active"]
        except CliError:
            from ..runtime import build_registry

            self._models_cache = build_registry(self._settings()).list()
        return self._models_cache

    def model_options(self) -> list[tuple[str, str, str]]:
        return menu_items(self.models(), self.state.get("model"))

    def cmd_model(self, rest: str) -> None:
        if rest == "list":
            self.run_cli(["model", "list"])
            return
        if rest:
            model_id = rest.removeprefix("use ").strip()
            self.run_cli(["model", "use", model_id])
            self.refresh_state()
            return
        options = self.model_options()
        choice = self.prompter.select(
            "Choose a model (✓ installed · ↓ downloads on select)",
            options,
            default=self.state.get("model"),
        )
        if choice is None:
            return
        if choice == OWN_MODEL:
            self.console.print(Text(PLUGIN_GUIDE))
            return
        if choice == self.state.get("model"):
            self.say(f"  Already using {choice}.", "dim")
            return
        self.run_cli(["model", "use", choice])
        self.refresh_state()

    # ------------------------------------------------------------------ service

    def cmd_start(self, rest: str) -> None:
        self.run_cli(["start", "--no-browser"])
        self.refresh_state()

    def cmd_stop(self, rest: str) -> None:
        self.run_cli(["stop"])
        self.refresh_state()

    def cmd_restart(self, rest: str) -> None:
        self.run_cli(["restart"])
        self.refresh_state()

    def cmd_status(self, rest: str) -> None:
        self.run_cli(["status"])

    def cmd_logs(self, rest: str) -> None:
        self.run_cli(["logs"])

    def cmd_open(self, rest: str) -> None:
        url = f"{self.api.base_url}/ui/"
        self.browser(url)
        self.say(f"  Opened {url}", "dim")

    # ------------------------------------------------------------------ view

    def cmd_json(self, rest: str) -> None:
        self.json_output = not self.json_output
        self.say(f"  JSON output {'on' if self.json_output else 'off'}.", "dim")

    def cmd_verbose(self, rest: str) -> None:
        self.verbose = not self.verbose
        self.say(f"  Details {'on' if self.verbose else 'off'}.", "dim")

    def cmd_clear(self, rest: str) -> None:
        self.console.clear()

    def cmd_help(self, rest: str) -> None:
        table = Table(box=None, show_header=False, pad_edge=False)
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="dim", no_wrap=True)
        table.add_column()
        group = None
        for command in COMMANDS:
            if command.group != group:
                group = command.group
                table.add_row(Text(group, style="bold"), "", "")
            table.add_row(f"  /{command.name}", command.args, command.help)
        self.console.print(table)
        self.say(
            "  Anything that doesn't start with / is evaluated as input in the current mode.", "dim"
        )

    def cmd_exit(self, rest: str) -> None:
        self.running = False


assert set(BY_NAME) == {
    name.removeprefix("cmd_") for name in dir(ReplSession) if name.startswith("cmd_")
}, "every command needs a handler"
