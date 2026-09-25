"""Decision backend for OpenAI-compatible ``/chat/completions`` endpoints.

The model is never asked to generate a number. Each primitive is posed as a
single-token classification: candidates are presented as labels (``yes`` /
``no`` / ``unknown`` for Noul, letters ``A``, ``B``, ... for Choice and Score)
and the backend reads the log-probabilities of the *first generated token*
(``max_tokens=1``, ``temperature=0``, ``logprobs=true``). Token variants such
as ``" Yes"`` or ``"yes."`` are folded onto their label and the label masses
are renormalised into a distribution.

Servers that do not return logprobs fall back to strict parsing of the one
generated label (one-hot distribution, logged once per backend). Every other
failure (transport, HTTP status, malformed payload, no recognisable label)
surfaces as :class:`BackendError` with a short message that never contains
the API key.

Works with OpenAI, Ollama, LM Studio, vLLM and llama.cpp server.
"""

from __future__ import annotations

import logging
import math
import string
from collections.abc import Sequence
from typing import Any

import httpx

from noulo.inference.llm_prompts import (
    NOUL_LABELS,
    SYSTEM_PROMPT,
    choice_task,
    messages,
    noul_task,
    noul_value,
    score_task,
)
from noulo.inference.types import BackendError, BackendInfo, ModelLoadError

__all__ = ["NOUL_LABELS", "SYSTEM_PROMPT", "OpenAIBackend"]

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

logger = logging.getLogger(__name__)


class OpenAIBackend:
    """:class:`~noulo.inference.types.DecisionBackend` over an OpenAI-compatible API.

    ``client`` is for dependency injection (tests, shared pools); an injected
    client is used as-is (its own timeout applies) and is not closed by
    :meth:`close`. Otherwise the backend owns an ``httpx.Client(timeout=timeout)``.
    ``extra_body`` is merged last into every request payload (e.g. ``{"seed": 0}``).
    """

    def __init__(
        self,
        *,
        id: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        top_logprobs: int = 20,
        extra_body: dict[str, Any] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.info = BackendInfo(
            id=id,
            backend="openai",
            model=model,
            quantization="n/a",
            local=httpx.URL(base_url).host in _LOCAL_HOSTS,
        )
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._top_logprobs = top_logprobs
        self._extra_body = dict(extra_body or {})
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._closed = False
        self._warned_fallback = False

    # ------------------------------------------------------------------ primitives

    def noul(self, input: str, proposition: str) -> float:
        task = noul_task(input, proposition)
        return noul_value(self._classify(task.prompt, task.labels))

    def choice(self, input: str, question: str, options: Sequence[str]) -> list[float]:
        task = choice_task(input, question, options)
        return self._classify(task.prompt, task.labels)

    def score(self, input: str, question: str, rubric: Sequence[str]) -> list[float]:
        task = score_task(input, question, rubric)
        return self._classify(task.prompt, task.labels)

    # ------------------------------------------------------------------ lifecycle

    def warmup(self) -> None:
        """One tiny Noul round-trip; raises :class:`ModelLoadError` if it fails."""
        try:
            self.noul("ok", "ok")
        except Exception as exc:
            raise ModelLoadError(f"Backend {self.info.id!r} failed its readiness check.") from exc

    def close(self) -> None:
        """Close the HTTP client if this backend created it. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------------ internals

    def _classify(self, prompt: str, labels: Sequence[str]) -> list[float]:
        choice = self._complete(prompt)
        probs = _distribution_from_logprobs(_logprob_candidates(choice), labels)
        if probs is not None:
            return probs
        probs = _distribution_from_content(choice, labels)
        if probs is None:
            raise self._error("model did not answer with an allowed label")
        self._warn_fallback()
        return probs

    def _complete(self, prompt: str) -> dict[str, Any]:
        """POST the prompt and return ``choices[0]`` of the completion."""
        if self._closed:
            raise self._error("backend is closed")
        try:
            response = self._client.post(
                self._url, json=self._payload(prompt), headers=self._headers
            )
        except httpx.TimeoutException as exc:
            raise self._error("request timed out") from exc
        except httpx.HTTPError as exc:
            # The exception text is deliberately left out: it may echo request details.
            raise self._error(f"request failed ({type(exc).__name__})") from exc
        if not response.is_success:
            # The body is left out too: e.g. OpenAI's 401 echoes the (partial) API key.
            raise self._error(
                f"upstream returned HTTP {response.status_code} {response.reason_phrase}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise self._error("response is not valid JSON") from exc
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise self._error("response has no completion choices")
        return choices[0]

    def _payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": messages(prompt),
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": self._top_logprobs,
            **self._extra_body,
        }

    def _error(self, reason: str) -> BackendError:
        return BackendError(f"OpenAI backend {self.info.id!r} ({self._model}): {reason}.")

    def _warn_fallback(self) -> None:
        if not self._warned_fallback:
            self._warned_fallback = True
            logger.warning(
                "OpenAI backend %r (%s) returned no usable logprobs; falling back to parsing "
                "the generated label (one-hot probabilities).",
                self.info.id,
                self._model,
            )


# ---------------------------------------------------------------------- probability extraction


def _logprob_candidates(choice: dict[str, Any]) -> dict[str, float]:
    """Raw token -> logprob for the first generated position; empty if absent or malformed.

    Includes the chosen token itself, which is usually also listed in
    ``top_logprobs``; each distinct token string is counted once. Non-finite
    logprobs carry no usable evidence and are dropped.
    """
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        return {}
    first = content[0]
    top = first.get("top_logprobs")
    candidates: dict[str, float] = {}
    for entry in [*(top if isinstance(top, list) else []), first]:
        if not isinstance(entry, dict):
            continue
        token, logprob = entry.get("token"), entry.get("logprob")
        if isinstance(token, str) and isinstance(logprob, (int, float)) and math.isfinite(logprob):
            candidates.setdefault(token, float(logprob))
    return candidates


def _distribution_from_logprobs(
    candidates: dict[str, float], labels: Sequence[str]
) -> list[float] | None:
    """Sum the probability of every token variant per label, renormalised over ``labels``.

    Returns ``None`` when no candidate token maps to an allowed label.
    """
    index = {label.lower(): i for i, label in enumerate(labels)}
    per_label: list[list[float]] = [[] for _ in labels]
    for raw_token, logprob in candidates.items():
        i = index.get(_normalise_token(raw_token))
        if i is not None:
            per_label[i].append(logprob)
    if not any(per_label):
        return None
    # Shift by the largest logprob so exp() can neither overflow nor underflow to all-zero.
    peak = max(lp for logprobs in per_label for lp in logprobs)
    mass = [sum(math.exp(lp - peak) for lp in logprobs) for logprobs in per_label]
    total = sum(mass)
    return [m / total for m in mass]


def _distribution_from_content(choice: dict[str, Any], labels: Sequence[str]) -> list[float] | None:
    """One-hot distribution from the generated text, or ``None`` if it is not a label."""
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return None
    token = _normalise_token(content)
    matches = [label.lower() == token for label in labels]
    return [float(match) for match in matches] if any(matches) else None


def _normalise_token(token: str) -> str:
    """Map a raw token such as ``' Yes.'`` or ``'"no"'`` to its bare lowercase form."""
    return token.strip().strip(string.punctuation).strip().lower()
