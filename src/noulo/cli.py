"""`noulo` command line: configure, run the service, call the API.

    noulo                            # interactive session (type inputs, /commands)
    noulo start [--headless]         # background service + frontend at /ui/
    noulo stop | restart | status | logs
    noulo serve                      # foreground, headless (systemd/containers)
    noulo model                      # pick one of 3 curated models or plug in your own
    noulo model list|use|download|add-endpoint|add-onnx|remove
    noulo noul --input ... --proposition ...
    noulo choice --input ... --question ... --option A=Billing --option B=Sales
    noulo score --input ... --question ... --rubric low,medium,high
    noulo learning status|on|off|records|clear
    noulo config show|get|set|unset|path

Exit codes: 0 ok, 1 API/service error, 2 usage/config error, 3 server unreachable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import httpx
from pydantic import ValidationError

from .config import ConfigError, Settings

EXIT_OK, EXIT_API, EXIT_USAGE, EXIT_UNREACHABLE = 0, 1, 2, 3

PLUGIN_GUIDE = """\
Plug in your own model
======================

A) Any OpenAI-compatible endpoint (OpenAI, Ollama, LM Studio, vLLM, llama.cpp):

   noulo model add-endpoint --id my-llm --base-url http://127.0.0.1:11434/v1 \\
       --model qwen2.5:1.5b [--api-key-env OPENAI_API_KEY]
   noulo model use my-llm

   noulo reads next-token log-probabilities (never generated numbers), so the
   server must support `logprobs`; otherwise it falls back to strict label parsing.

B) Your own ONNX NLI / zero-shot classifier (runs fully local on CPU):

   # export any Hugging Face NLI model (needs: pip install "optimum[onnxruntime]")
   optimum-cli export onnx --model <hf-model-id> --task text-classification my-nli/
   # optional INT8 quantisation
   optimum-cli onnxruntime quantize --onnx_model my-nli/ --avx2 -o my-nli-int8/
   mv my-nli-int8/model_quantized.onnx my-nli-int8/model.onnx
   cp my-nli/tokenizer.json my-nli/config.json my-nli-int8/

   The directory needs model.onnx, tokenizer.json and config.json (with an
   'entailment' label in id2label). Then:

   noulo model add-onnx --id my-nli --path ./my-nli-int8 --quantization INT8
   # a causal LLM export (model.onnx + tokenizer.json + tokenizer_config.json with a chat
   # template, e.g. from onnx-community) works too:
   noulo model add-onnx --id my-llm --path ./my-llm-q4 --kind llm --quantization INT4
   noulo benchmark --model my-nli --tune      # fits templates + calibration
   noulo model use my-nli

Both commands write to models.json (NOULO_MODELS_FILE), which you can also edit
by hand; see docs/models.md.
"""


class CliError(Exception):
    def __init__(self, message: str, code: int = EXIT_USAGE):
        super().__init__(message)
        self.code = code


class ApiError(CliError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{code} ({status}): {message}", EXIT_API)
        self.status, self.api_code = status, code


class Unreachable(CliError):
    def __init__(self, url: str):
        super().__init__(
            f"Cannot reach the decision engine at {url}. "
            "Start it with `noulo serve` (or `noulo start`).",
            EXIT_UNREACHABLE,
        )


# ---------------------------------------------------------------------- .env file


class EnvFile:
    """Minimal .env editor that preserves comments and unrelated lines."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _lines(self) -> list[str]:
        return self.path.read_text().splitlines() if self.path.exists() else []

    @staticmethod
    def _split(line: str) -> tuple[str, str] | None:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            return None
        key, value = stripped.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].replace('\\"', '"')
        return key, value

    def get(self, key: str) -> str | None:
        for line in self._lines():
            parsed = self._split(line)
            if parsed and parsed[0] == key:
                return parsed[1]
        return None

    def set(self, key: str, value: str) -> None:
        if any(c in value for c in " #\"'") or value == "":
            value = '"' + value.replace('"', '\\"') + '"'
        lines, replaced = [], False
        for line in self._lines():
            parsed = self._split(line)
            if parsed and parsed[0] == key:
                if not replaced:
                    lines.append(f"{key}={value}")
                    replaced = True
                continue
            lines.append(line)
        if not replaced:
            lines.append(f"{key}={value}")
        self._write(lines)

    def unset(self, key: str) -> bool:
        lines = self._lines()
        kept = [line for line in lines if (self._split(line) or ("",))[0] != key]
        self._write(kept)
        return len(kept) != len(lines)

    def _write(self, lines: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(lines) + "\n")
        os.chmod(self.path, 0o600)  # may contain API keys


def setting_field(key: str) -> str:
    field = key.lower().removeprefix("noulo_").replace("-", "_")
    if field not in Settings.model_fields:
        raise CliError(f"Unknown setting {key!r}. See `noulo config show`.")
    return field


# ---------------------------------------------------------------------- HTTP client


class ApiClient:
    def __init__(self, base_url: str, api_key: str | None, client: httpx.Client | None = None):
        self.base_url = base_url
        self._client = client or httpx.Client(base_url=base_url, timeout=120.0)
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def call(self, method: str, path: str, body: Any = None) -> tuple[Any, httpx.Headers]:
        try:
            response = self._client.request(method, path, json=body, headers=self._headers)
        except httpx.TransportError:
            raise Unreachable(self.base_url) from None
        data = response.json() if response.content else None
        if response.status_code >= 400:
            error = (data or {}).get("error") if isinstance(data, dict) else None
            if error:
                raise ApiError(
                    response.status_code, error.get("code", "ERROR"), error.get("message", "")
                )
            if path == "/health" and isinstance(data, dict):
                return data, response.headers
            raise ApiError(response.status_code, "HTTP_ERROR", response.text[:200])
        return data, response.headers


# ---------------------------------------------------------------------- server


def _wait_and_open(settings: Settings) -> None:
    url = settings.base_url
    for _ in range(600):
        try:
            if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                webbrowser.open(f"{url}/ui/")
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)


def run_uvicorn(app: Any, settings: Settings) -> None:
    import uvicorn

    if settings.open_browser:
        threading.Thread(target=_wait_and_open, args=(settings,), daemon=True).start()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        timeout_graceful_shutdown=int(settings.shutdown_timeout),
    )


def default_app_factory(settings: Settings) -> Any:
    from .api.server import create_app

    return create_app(settings)


# ---------------------------------------------------------------------- parser


def _add_output_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="print the full JSON response")
    p.add_argument(
        "--diagnostics", action="store_true", help="include probabilities (implies --json)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="noulo",
        description="Local Choice / Score / Noul decision engine.",
    )
    parser.add_argument("--url", help="server URL (default: from NOULO_HOST/NOULO_PORT)")
    parser.add_argument("--api-key", help="API key for a server started with NOULO_API_KEY")
    parser.add_argument("--env-file", help="settings file (default: ./.env or $NOULO_ENV_FILE)")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    for name, help_text in (
        ("serve", "run the REST API in the foreground (headless)"),
        ("start", "start the background service with the frontend"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--host")
        p.add_argument("--port", type=int)
        p.add_argument("--model", help="model id to load at startup")
        p.add_argument(
            "--allow-network",
            action="store_true",
            default=None,
            help="permit binding to a non-loopback host",
        )
        p.add_argument("--learning", dest="learning", action="store_true", default=None)
        p.add_argument("--no-learning", dest="learning", action="store_false")
        p.add_argument("--threads", type=int, help="ONNX Runtime intra-op threads")
        p.add_argument("--max-concurrency", type=int)
        p.add_argument("--log-level")
        if name == "serve":
            p.add_argument("--ui", action="store_true", help="also serve the frontend at /ui")
        else:
            p.add_argument("--headless", action="store_true", help="API only, no frontend")
            p.add_argument("--foreground", action="store_true", help="run in this terminal")
            p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    sub.add_parser("stop", help="stop the background service")
    sub.add_parser("restart", help="restart the background service (e.g. after config changes)")
    sub.add_parser("status", help="background service status")
    p = sub.add_parser("logs", help="show the background service log")
    p.add_argument("-n", "--lines", type=int, default=50)

    p = sub.add_parser("noul", help="P(proposition is true | input)")
    p.add_argument("--input", "-i", required=True, help="input text, or - for stdin")
    p.add_argument("--proposition", "-p", required=True)
    _add_output_flag(p)

    p = sub.add_parser("choice", help="select one of the supplied options")
    p.add_argument("--input", "-i", required=True, help="input text, or - for stdin")
    p.add_argument("--question", "-q", required=True)
    p.add_argument("--option", "-o", action="append", required=True, metavar="ID=TEXT")
    _add_output_flag(p)

    p = sub.add_parser("score", help="place the input on an ordered rubric")
    p.add_argument("--input", "-i", required=True, help="input text, or - for stdin")
    p.add_argument("--question", "-q", required=True)
    p.add_argument(
        "--rubric",
        "-r",
        action="append",
        required=True,
        help="comma-separated levels lowest→highest (or repeat the flag)",
    )
    _add_output_flag(p)

    p = sub.add_parser("evaluate", help="send a JSON /api/v1/evaluate request")
    p.add_argument("file", nargs="?", default="-", help="JSON file, or - for stdin")
    _add_output_flag(p)

    sub.add_parser("health", help="server readiness")
    p = sub.add_parser("info", help="engine/model information")
    p.add_argument("--json", action="store_true")

    models = sub.add_parser(
        "model", aliases=["models"], help="choose, download, switch and register models"
    )
    msub = models.add_subparsers(dest="models_command", metavar="ACTION")
    msub.add_parser("list", help="list known models")
    p = msub.add_parser("download", help="download catalog models (default: model + embedder)")
    p.add_argument("ids", nargs="*")
    p.add_argument("--missing", action="store_true", help="skip models that are already installed")
    p = msub.add_parser("use", help="switch model: download if needed, save, restart service")
    p.add_argument("id")
    p = msub.add_parser("add-endpoint", help="register an OpenAI-compatible endpoint")
    p.add_argument("--id", required=True)
    p.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:11434/v1")
    p.add_argument("--model", required=True, help="model name at the endpoint")
    p.add_argument("--api-key-env", help="env var that holds the endpoint API key")
    p = msub.add_parser("add-onnx", help="register your own ONNX NLI model directory")
    p.add_argument("--id", required=True)
    p.add_argument("--path", required=True, help="dir with model.onnx, tokenizer.json, config.json")
    p.add_argument("--quantization", default="unknown")
    p.add_argument(
        "--kind",
        choices=["nli", "llm"],
        default="nli",
        help="nli: NLI/zero-shot classifier; llm: causal LLM export with a chat template",
    )
    p = msub.add_parser("remove", help="remove a registered endpoint or custom model")
    p.add_argument("id")

    learning = sub.add_parser("learning", help="learning memory controls")
    lsub = learning.add_subparsers(dest="learning_command", required=True, metavar="ACTION")
    lsub.add_parser("status")
    lsub.add_parser("on", help="enable learning (live + saved)")
    lsub.add_parser("off", help="disable learning (live + saved)")
    p = lsub.add_parser("records", help="show stored cases")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--type", choices=["noul", "choice", "score"])
    p = lsub.add_parser("clear", help="delete all stored cases")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")

    p = sub.add_parser("feedback", help="teach the correct outcome for a past evaluation")
    p.add_argument("record_id", help="X-Record-Id from an evaluation")
    p.add_argument("expected", help="Choice ID, or a Noul/Score value in [0,1], or true/false")

    config = sub.add_parser("config", help="view and edit settings in the .env file")
    csub = config.add_subparsers(dest="config_command", required=True, metavar="ACTION")
    p = csub.add_parser("show")
    p.add_argument("--json", action="store_true")
    p = csub.add_parser("get")
    p.add_argument("key")
    p = csub.add_parser("set")
    p.add_argument("key")
    p.add_argument("value")
    p = csub.add_parser("unset")
    p.add_argument("key")
    csub.add_parser("path")

    p = sub.add_parser("openapi", help="write the OpenAPI spec (no model needed)")
    p.add_argument("--output", "-o", default="-", help="file path, or - for stdout")

    sub.add_parser(
        "benchmark",
        add_help=False,
        help="measure accuracy, calibration, latency and memory (see --help)",
    )
    return parser


# ---------------------------------------------------------------------- command runner


class Cli:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        env_file: Path,
        http_client: httpx.Client | None,
        stdout: IO[str],
        stderr: IO[str],
        stdin: IO[str],
        run_server: Callable[[Any, Settings], None],
        app_factory: Callable[[Settings], Any],
        registry: Any = None,
        service: Any = None,
        ask: Callable[[str], str] = input,
        browser: Callable[[str], Any] = webbrowser.open,
    ):
        self.args, self.env_file = args, env_file
        self.out, self.err, self.stdin = stdout, stderr, stdin
        self.run_server, self.app_factory = run_server, app_factory
        self._http_client, self._registry, self._service = http_client, registry, service
        self.ask, self.browser = ask, browser

    # -- helpers
    def settings(self, **overrides: Any) -> Settings:
        env = self.env_file if self.env_file.exists() else None
        try:
            return Settings(_env_file=env, **overrides)
        except ValidationError as exc:
            raise CliError(f"Invalid setting: {exc.errors()[0]['msg']}") from None

    def api(self) -> ApiClient:
        settings = self.settings()
        url = self.args.url or os.environ.get("NOULO_URL") or settings.base_url
        key = self.args.api_key or (
            settings.api_key.get_secret_value() if settings.api_key else None
        )
        return ApiClient(url, key, self._http_client)

    def registry(self) -> Any:
        if self._registry is None:
            from .runtime import build_registry

            self._registry = build_registry(self.settings())
        return self._registry

    def service(self) -> Any:
        if self._service is None:
            from .service import ServiceManager

            root = self.env_file.resolve().parent
            run_dir = self.settings().run_dir
            self._service = ServiceManager(
                run_dir if run_dir.is_absolute() else root / run_dir, cwd=root
            )
        return self._service

    def _interactive(self) -> bool:
        """A person at a terminal (not an agent, script or CI job)."""
        isatty = getattr(self.out, "isatty", None)
        return bool(isatty and isatty())

    def global_args(self) -> list[str]:
        return ["--env-file", str(self.env_file.resolve())]

    def print(self, *parts: Any) -> None:
        print(*parts, file=self.out)

    def read_text(self, value: str) -> str:
        return self.stdin.read() if value == "-" else value

    def emit_result(self, data: dict, headers: httpx.Headers) -> None:
        if self.args.json or self.args.diagnostics:
            if headers.get("X-Record-Id"):
                data = {**data, "recordId": headers["X-Record-Id"]}
            self.print(json.dumps(data, indent=2))
        else:
            self.print(data["value"])

    def evaluate(self, path: str, body: dict) -> int:
        query = "?diagnostics=true" if self.args.diagnostics else ""
        data, headers = self.api().call("POST", f"/api/v1/{path}{query}", body)
        self.emit_result(data, headers)
        return EXIT_OK

    # -- commands
    def _server_overrides(self) -> dict[str, Any]:
        a = self.args
        return {
            k: v
            for k, v in {
                "host": a.host,
                "port": a.port,
                "model": a.model,
                "allow_network": a.allow_network,
                "learning_enabled": a.learning,
                "threads": a.threads,
                "max_concurrency": a.max_concurrency,
                "log_level": a.log_level,
            }.items()
            if v is not None
        }

    def _checked_settings(self, **overrides: Any) -> Settings:
        settings = self.settings(**overrides)
        try:
            return settings.check()
        except ConfigError as exc:
            raise CliError(str(exc)) from None

    def cmd_serve(self) -> int:
        """Foreground server (headless unless --ui)."""
        overrides = self._server_overrides()
        if self.args.command == "serve":
            overrides.update(ui_enabled=bool(self.args.ui), open_browser=False)
        else:
            overrides.update(
                ui_enabled=not self.args.headless,
                open_browser=not (self.args.headless or self.args.no_browser),
            )
        settings = self._checked_settings(**overrides)
        self.run_server(self.app_factory(settings), settings)
        return EXIT_OK

    def _start_background(
        self, *, ui: bool, open_browser: bool, extra_args: list[str] | None = None, **overrides: Any
    ) -> int:
        settings = self._checked_settings(**overrides)
        self.print(f"Starting noulo with model {settings.model} ...")
        try:
            state = self.service().start(
                host=settings.host,
                port=settings.port,
                ui=ui,
                extra_args=extra_args,
                global_args=self.global_args(),
            )
        except RuntimeError as exc:
            raise CliError(str(exc), EXIT_API) from None
        return self._announce(state, open_browser=open_browser)

    def _announce(self, state: Any, *, open_browser: bool = False) -> int:
        self.print(f"noulo is running at {state.url} (pid {state.pid})")
        if state.ui:
            self.print(f"Frontend: {state.url}/ui/")
            if open_browser:
                self.browser(f"{state.url}/ui/")
        self.print(f"Logs: {state.log}")
        return EXIT_OK

    def cmd_start(self) -> int:
        if self.args.foreground:
            return self.cmd_serve()
        overrides = self._server_overrides()
        extra: list[str] = []
        for flag, key in (
            ("--host", "host"),
            ("--port", "port"),
            ("--model", "model"),
            ("--threads", "threads"),
            ("--max-concurrency", "max_concurrency"),
            ("--log-level", "log_level"),
        ):
            if key in overrides:
                extra += [flag, str(overrides[key])]
        if overrides.get("allow_network"):
            extra.append("--allow-network")
        if "learning_enabled" in overrides:
            extra.append("--learning" if overrides["learning_enabled"] else "--no-learning")
        ui = not self.args.headless
        return self._start_background(
            ui=ui,
            open_browser=ui and not self.args.no_browser and self._interactive(),
            extra_args=extra,
            **overrides,
        )

    def cmd_stop(self) -> int:
        stopped = self.service().stop()
        self.print("Stopped noulo." if stopped else "noulo is not running.")
        return EXIT_OK

    def cmd_restart(self) -> int:
        if not self.service().status().running:
            raise CliError("noulo is not running; start it with `noulo start`.", EXIT_API)
        self.print("Restarting noulo ...")
        try:
            state = self.service().restart(global_args=self.global_args())
        except RuntimeError as exc:
            raise CliError(str(exc), EXIT_API) from None
        return self._announce(state)

    def cmd_status(self) -> int:
        state = self.service().status()
        if state.running:
            frontend = f", frontend {state.url}/ui/" if state.ui else ", headless"
            self.print(f"noulo is running at {state.url} (pid {state.pid}{frontend})")
            try:
                data, _ = self.api().call("GET", "/api/v1/info")
                self.print(
                    f"model: {data['model']} ({data['backend']}, {data['quantization']}), "
                    f"learning: {'on' if data['learning'] else 'off'}, "
                    f"device: {data.get('device', 'cpu')}"
                )
            except CliError:
                pass
        else:
            self.print("noulo is not running. Start it with `noulo start`.")
        return EXIT_OK

    def cmd_logs(self) -> int:
        logfile = Path(getattr(self.service(), "logfile", ""))
        if not logfile.is_file():
            self.print("No log yet.")
            return EXIT_OK
        for line in logfile.read_text(errors="replace").splitlines()[-self.args.lines :]:
            self.print(line)
        return EXIT_OK

    def cmd_noul(self) -> int:
        return self.evaluate(
            "noul", {"input": self.read_text(self.args.input), "proposition": self.args.proposition}
        )

    def cmd_choice(self) -> int:
        choices = []
        for spec in self.args.option:
            if "=" not in spec:
                raise CliError(f"Option {spec!r} must look like ID=TEXT.")
            option_id, text = spec.split("=", 1)
            choices.append({"id": option_id.strip(), "text": text.strip()})
        return self.evaluate(
            "choice",
            {
                "input": self.read_text(self.args.input),
                "question": self.args.question,
                "choices": choices,
            },
        )

    def cmd_score(self) -> int:
        rubric = [level.strip() for group in self.args.rubric for level in group.split(",")]
        return self.evaluate(
            "score",
            {
                "input": self.read_text(self.args.input),
                "question": self.args.question,
                "rubric": rubric,
            },
        )

    def cmd_evaluate(self) -> int:
        raw = self.stdin.read() if self.args.file == "-" else Path(self.args.file).read_text()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CliError(f"Request is not valid JSON: {exc.msg}") from None
        return self.evaluate("evaluate", body)

    def cmd_health(self) -> int:
        data, _ = self.api().call("GET", "/health")
        self.print(data["status"])
        return EXIT_OK if data.get("modelLoaded") else EXIT_API

    def cmd_info(self) -> int:
        data, _ = self.api().call("GET", "/api/v1/info")
        if self.args.json:
            self.print(json.dumps(data, indent=2))
        else:
            for key in ("name", "version", "model", "backend", "quantization", "local", "learning"):
                self.print(f"{key:13s} {data.get(key)}")
        return EXIT_OK

    def pick_model(self) -> int:
        from .model_menu import OWN_MODEL, menu_items

        try:
            data, _ = self.api().call("GET", "/api/v1/models")
            active, listing = data["active"], data["models"]
        except CliError:
            active, listing = self.settings().model, self.registry().list()
        items = menu_items(listing, active)
        self.print(f"Current model: {active}\n")
        self.print("Choose a model (✓ installed · ↓ downloads when selected):")
        for number, (_, label, facts) in enumerate(items, 1):
            self.print(f"  {number:>2}) {label} {facts}")
        answer = self.ask(f"Select 1-{len(items)} (Enter keeps the current model): ").strip()
        if not answer:
            self.print(f"Keeping {active}.")
            return EXIT_OK
        if not answer.isdigit() or not 1 <= int(answer) <= len(items):
            raise CliError(f"Please choose 1-{len(items)}.")
        choice = items[int(answer) - 1][0]
        if choice == OWN_MODEL:
            self.print(PLUGIN_GUIDE)
            return EXIT_OK
        return self.use_model(choice, interactive=True)

    def use_model(self, model_id: str, *, interactive: bool = False) -> int:
        """Download if needed, save as default, then restart/switch the running service."""
        registry = self.registry()
        if not registry.is_installed(model_id):
            if not registry.is_downloadable(model_id):
                raise CliError(
                    f"Model {model_id!r} is not installed or registered; see `noulo model list`."
                )
            self.print(f"Downloading {model_id} ...")
            registry.download(model_id, progress=self.print)
        EnvFile(self.env_file).set("NOULO_MODEL", model_id)
        self.print(f"Saved NOULO_MODEL={model_id} to {self.env_file}")

        service = self.service()
        if service.status().running:
            self.print(f"Restarting noulo with {model_id} ...")
            try:
                state = service.restart(global_args=self.global_args())
            except RuntimeError as exc:
                raise CliError(str(exc), EXIT_API) from None
            return self._announce(state)
        try:
            data, _ = self.api().call("PUT", "/api/v1/models/active", {"id": model_id})
            self.print(f"Switched the running server to {data['active']['id']} (live).")
            return EXIT_OK
        except Unreachable:
            pass
        if interactive:
            reply = self.ask("noulo is not running. Start it now with the frontend? [Y/n] ")
            if reply.strip().lower() in ("", "y", "yes"):
                return self._start_background(ui=True, open_browser=self._interactive())
        self.print("noulo is not running; start it with `noulo start`.")
        return EXIT_OK

    def cmd_model(self) -> int:
        action = self.args.models_command
        if action is None:
            return self.pick_model()
        if action == "list":
            try:
                data, _ = self.api().call("GET", "/api/v1/models")
                active, models = data["active"], data["models"]
            except Unreachable:
                active, models = self.settings().model, self.registry().list()
                self.print("(server not running; showing local registry)")
            for m in models:
                mark = "*" if m["id"] == active else " "
                state = "installed" if m["installed"] else "not installed"
                self.print(
                    f"{mark} {m['id']:34s} {m['backend']:8s} {m['quantization']:6s} "
                    f"{state:13s} {m['description']}"
                )
            return EXIT_OK
        if action == "download":
            settings = self.settings()
            for model_id in self.args.ids or [settings.model, settings.embedder]:
                if self.args.missing and self.registry().is_installed(model_id):
                    self.print(f"already installed: {model_id}")
                    continue
                path = self.registry().download(model_id, progress=self.print)
                self.print(f"Installed {model_id} -> {path}")
            return EXIT_OK
        if action == "use":
            return self.use_model(self.args.id)
        try:
            if action == "add-endpoint":
                self.registry().register_openai(
                    id=self.args.id,
                    base_url=self.args.base_url,
                    model=self.args.model,
                    api_key_env=self.args.api_key_env,
                )
            elif action == "add-onnx":
                self.registry().register_onnx(
                    id=self.args.id,
                    path=self.args.path,
                    quantization=self.args.quantization,
                    kind=self.args.kind,
                )
            elif action == "remove":
                self.registry().unregister(self.args.id)
                self.print(f"Removed {self.args.id}")
                return EXIT_OK
        except ValueError as exc:
            raise CliError(str(exc)) from None
        self.print(f"Registered {self.args.id}; activate with `noulo model use {self.args.id}`")
        return EXIT_OK

    cmd_models = cmd_model

    def cmd_learning(self) -> int:
        action = self.args.learning_command
        api = self.api()
        if action in ("on", "off"):
            enabled = action == "on"
            try:
                api.call("PUT", "/api/v1/learning", {"enabled": enabled})
                self.print(
                    f"Learning {'enabled' if enabled else 'disabled'} on the running server."
                )
            except Unreachable:
                self.print("Server not running; the setting applies on next start.")
            EnvFile(self.env_file).set("NOULO_LEARNING_ENABLED", "true" if enabled else "false")
            return EXIT_OK
        if action == "status":
            data, _ = api.call("GET", "/api/v1/learning")
            self.print(json.dumps(data, indent=2))
            return EXIT_OK
        if action == "records":
            query = f"?limit={self.args.limit}" + (
                f"&type={self.args.type}" if self.args.type else ""
            )
            data, _ = api.call("GET", f"/api/v1/learning/records{query}")
            self.print(json.dumps(data, indent=2))
            return EXIT_OK
        if action == "clear":
            if not self.args.yes:
                self.err.write("Delete all learning records? Re-run with --yes to confirm.\n")
                return EXIT_USAGE
            data, _ = api.call("DELETE", "/api/v1/learning/records")
            self.print(f"Deleted {data['deleted']} records.")
            return EXIT_OK
        raise CliError(f"Unknown learning action {action!r}")

    def cmd_feedback(self) -> int:
        raw = self.args.expected
        expected: Any = raw
        if raw.lower() in ("true", "false"):
            expected = raw.lower() == "true"
        else:
            try:
                expected = float(raw)
            except ValueError:
                expected = raw
        data, _ = self.api().call(
            "POST", "/api/v1/feedback", {"recordId": self.args.record_id, "expected": expected}
        )
        self.print(f"Recorded feedback for {data['recordId']}")
        return EXIT_OK

    def cmd_config(self) -> int:
        action = self.args.config_command
        env = EnvFile(self.env_file)
        if action == "path":
            self.print(self.env_file)
        elif action == "show":
            data = self.settings().public_dict()
            if self.args.json:
                self.print(json.dumps(data, indent=2, default=str))
            else:
                for key, value in data.items():
                    self.print(f"NOULO_{key.upper():28s} {value}")
        elif action == "get":
            field = setting_field(self.args.key)
            self.print(self.settings().public_dict()[field])
        elif action == "set":
            field = setting_field(self.args.key)
            try:
                Settings(_env_file=None, **{field: self.args.value})
            except ValidationError as exc:
                raise CliError(f"Invalid value for {field}: {exc.errors()[0]['msg']}") from None
            env.set(f"NOULO_{field.upper()}", self.args.value)
            self.print(f"Saved NOULO_{field.upper()} to {self.env_file}")
        elif action == "unset":
            field = setting_field(self.args.key)
            env.unset(f"NOULO_{field.upper()}")
            self.print(f"Removed NOULO_{field.upper()} from {self.env_file}")
        return EXIT_OK

    def cmd_openapi(self) -> int:
        from .api.server import create_app

        settings = Settings(_env_file=None, ui_enabled=False, models_file=None)
        text = json.dumps(create_app(settings).openapi(), indent=2) + "\n"
        if self.args.output == "-":
            self.out.write(text)
        else:
            Path(self.args.output).write_text(text)
            self.print(f"Wrote {self.args.output}")
        return EXIT_OK

    def cmd_benchmark(self) -> int:
        from .benchmark.run import main_and_exit

        main_and_exit(self.args.benchmark_args)
        return EXIT_OK  # not reached

    def run(self) -> int:
        handler = getattr(self, f"cmd_{self.args.command}")
        return handler()


def main(
    argv: list[str] | None = None,
    *,
    http_client: httpx.Client | None = None,
    env_file: str | Path | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    stdin: IO[str] | None = None,
    run_server: Callable[[Any, Settings], None] = run_uvicorn,
    app_factory: Callable[[Settings], Any] = default_app_factory,
    registry: Any = None,
    service: Any = None,
    ask: Callable[[str], str] = input,
    browser: Callable[[str], Any] = webbrowser.open,
) -> int:
    stdout, stderr, stdin = stdout or sys.stdout, stderr or sys.stderr, stdin or sys.stdin
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if args.command == "benchmark":
        args.benchmark_args = extra  # options belong to the benchmark runner
    elif extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    env_path = Path(args.env_file or env_file or os.environ.get("NOULO_ENV_FILE", ".env"))
    injected = dict(
        http_client=http_client,
        stdout=stdout,
        stderr=stderr,
        stdin=stdin,
        run_server=run_server,
        app_factory=app_factory,
        registry=registry,
        service=service,
        ask=ask,
        browser=browser,
    )

    if args.command is None:
        from .repl import app

        return app.run_repl(
            env_file=env_path,
            url=args.url,
            api_key=args.api_key,
            stdin=stdin,
            stdout=stdout,
            http_client=http_client,
        )

    cli = Cli(args, env_file=env_path, **injected)
    try:
        return cli.run()
    except CliError as exc:
        print(f"error: {exc}", file=stderr)
        return exc.code
    except EOFError:
        print(
            "error: this command asks a question and needs an interactive terminal. "
            "Pass the answer as arguments instead, e.g. `noulo model use <id>` "
            "(see `noulo model list`) or `noulo learning clear --yes`.",
            file=stderr,
        )
        return EXIT_USAGE
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
