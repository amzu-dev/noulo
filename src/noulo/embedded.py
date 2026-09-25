"""Direct Python API — the same engine and validation as the REST API, in-process.

from noulo import evaluate
evaluate({"type": "noul", "input": "...", "proposition": "..."})
# {"type": "noul", "value": 0.96}

from noulo import Noulo
with Noulo(model="nli-minilm2-l6-int8", learning_enabled=False) as engine:
    engine.choice("Charged twice", "Which team?", {"A": "Billing", "B": "Sales"})
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .api.validation import parse_request
from .config import Settings
from .inference.engine import DecisionEngine


class Noulo:
    """An in-process decision engine. The model is loaded once, on construction."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        engine: DecisionEngine | None = None,
        **overrides: Any,
    ):
        from .runtime import build_engine, build_registry

        self.settings = settings or Settings(**overrides)
        self.engine = engine or build_engine(self.settings, build_registry(self.settings))
        if self.engine.state == "created":
            self.engine.start()

    def _run(self, primitive: str, request: Mapping[str, Any], diagnostics: bool) -> dict[str, Any]:
        parsed = parse_request(primitive, dict(request), self.settings.limits)
        result = self.engine.evaluate(parsed, diagnostics=diagnostics)
        data = result.to_dict()
        if diagnostics:
            data["diagnostics"] = result.diagnostics
            data["recordId"] = result.record_id
        return data

    def evaluate(self, request: Mapping[str, Any], *, diagnostics: bool = False) -> dict[str, Any]:
        return self._run("evaluate", request, diagnostics)

    async def aevaluate(
        self, request: Mapping[str, Any], *, diagnostics: bool = False
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self.evaluate, request, diagnostics=diagnostics)

    def noul(self, input: str, proposition: str) -> float:
        return self._run("noul", {"input": input, "proposition": proposition}, False)["value"]

    def choice(
        self, input: str, question: str, choices: Mapping[str, str] | Sequence[Mapping[str, str]]
    ) -> str:
        if isinstance(choices, Mapping):
            choices = [{"id": key, "text": text} for key, text in choices.items()]
        payload = {"input": input, "question": question, "choices": list(choices)}
        return self._run("choice", payload, False)["value"]

    def score(self, input: str, question: str, rubric: Sequence[str]) -> float:
        payload = {"input": input, "question": question, "rubric": list(rubric)}
        return self._run("score", payload, False)["value"]

    def teach_file(self, path: str | Path) -> dict[str, Any]:
        """Teach every labelled example in a .jsonl/.json file (see noulo.teaching)."""
        from .api.validation import RequestError
        from .teaching import TeachingError, load_examples, to_request

        imported, failed = 0, []
        for line, item in load_examples(path):
            try:
                request, expected = to_request(item)
                self.engine.teach(
                    parse_request("evaluate", request, self.settings.limits), expected
                )
                imported += 1
            except RequestError as exc:
                failed.append({"line": line, "message": exc.message})
            except (TeachingError, ValueError, TypeError) as exc:
                failed.append({"line": line, "message": str(exc)})
        return {"imported": imported, "failed": failed}

    def feedback(self, record_id: str, expected: Any) -> str:
        return self.engine.feedback(record_id, expected)

    def switch_model(self, model_id: str):
        return self.engine.switch_model(model_id)

    def close(self) -> None:
        self.engine.shutdown(self.settings.shutdown_timeout)

    def __enter__(self) -> Noulo:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


_default: Noulo | None = None
_default_lock = threading.Lock()


def _make_default() -> Noulo:
    return Noulo()


def _get_default() -> Noulo:
    global _default
    with _default_lock:
        if _default is None:
            _default = _make_default()
        return _default


def evaluate(request: Mapping[str, Any], *, diagnostics: bool = False) -> dict[str, Any]:
    """Evaluate with a shared default engine configured from NOULO_* env / .env."""
    return _get_default().evaluate(request, diagnostics=diagnostics)


async def aevaluate(request: Mapping[str, Any], *, diagnostics: bool = False) -> dict[str, Any]:
    return await asyncio.to_thread(evaluate, request, diagnostics=diagnostics)


def close() -> None:
    """Release the shared default engine (it is re-created on next use)."""
    global _default
    with _default_lock:
        if _default is not None:
            _default.close()
            _default = None
