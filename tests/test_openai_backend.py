"""Tests for the OpenAI-compatible decision backend (no network: httpx.MockTransport)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import Any

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from noulo.inference.openai_backend import OpenAIBackend
from noulo.inference.types import (
    BackendError,
    BackendInfo,
    DecisionBackend,
    ModelLoadError,
)

LOCAL_URL = "http://127.0.0.1:11434/v1"
Handler = Callable[[httpx.Request], httpx.Response]


def completion(
    top: Mapping[str, float] | None,
    *,
    chosen: str | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    """Build a /chat/completions response.

    ``top`` maps token -> probability for ``top_logprobs``; ``None`` omits logprobs.
    The chosen token defaults to the most likely token in ``top``.
    """
    if top is None:
        logprobs = None
        chosen = chosen if chosen is not None else (content or "")
    else:
        chosen = chosen if chosen is not None else max(top, key=lambda t: top[t])
        top_logprobs = [{"token": t, "logprob": math.log(p)} for t, p in top.items()]
        chosen_logprob = math.log(top[chosen]) if chosen in top else math.log(0.5)
        logprobs = {
            "content": [{"token": chosen, "logprob": chosen_logprob, "top_logprobs": top_logprobs}]
        }
    message = {"role": "assistant", "content": content if content is not None else chosen}
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "logprobs": logprobs}],
    }


def _choice_body(message: Any, logprobs: Any) -> dict[str, Any]:
    return {"choices": [{"message": message, "logprobs": logprobs}]}


class Recorder:
    """MockTransport handler that records requests and replies with a fixed response."""

    def __init__(self, body: Any = None, status: int = 200) -> None:
        self.body = completion({"yes": 0.6, "no": 0.3, "unknown": 0.1}) if body is None else body
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body)

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


def make_backend(handler: Handler, **kwargs: Any) -> OpenAIBackend:
    options: dict[str, Any] = {"id": "test-llm", "base_url": LOCAL_URL, "model": "llama3.2"}
    options.update(kwargs)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAIBackend(client=client, **options)


# --------------------------------------------------------------------------- payload


def test_noul_posts_chat_completion_with_single_token_logprob_params():
    recorder = Recorder()
    backend = make_backend(recorder, base_url=LOCAL_URL + "/")

    backend.noul("The sky is blue.", "The sky has a colour.")

    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
    payload = recorder.payload
    assert payload["model"] == "llama3.2"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 1
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 20
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user"]


def test_sends_bearer_token_when_api_key_is_set():
    recorder = Recorder()
    backend = make_backend(recorder, api_key="sk-secret-123")

    backend.noul("x", "y")

    assert recorder.requests[0].headers["Authorization"] == "Bearer sk-secret-123"


def test_omits_authorization_header_without_api_key():
    recorder = Recorder()
    backend = make_backend(recorder)

    backend.noul("x", "y")

    assert "Authorization" not in recorder.requests[0].headers


def test_payload_uses_configured_top_logprobs_and_merges_extra_body():
    recorder = Recorder()
    backend = make_backend(recorder, top_logprobs=5, extra_body={"seed": 7, "keep_alive": "5m"})

    backend.noul("x", "y")

    payload = recorder.payload
    assert payload["top_logprobs"] == 5
    assert payload["seed"] == 7
    assert payload["keep_alive"] == "5m"


# --------------------------------------------------------------------------- noul


def test_noul_is_p_yes_plus_half_p_unknown_from_logprobs():
    backend = make_backend(Recorder(completion({"yes": 0.7, "no": 0.2, "unknown": 0.1})))

    assert backend.noul("It rained all day.", "The ground is wet.") == pytest.approx(0.75)


def test_noul_aggregates_case_whitespace_and_punctuation_variants():
    top = {
        " Yes": 0.4,
        "yes.": 0.2,
        "No": 0.1,
        " no)": 0.1,
        '"unknown":': 0.1,
        "maybe": 0.1,  # not a label: excluded before renormalising
    }
    backend = make_backend(Recorder(completion(top)))

    # yes = 0.6, no = 0.2, unknown = 0.1 -> renormalised over 0.9
    assert backend.noul("x", "y") == pytest.approx((0.6 + 0.5 * 0.1) / 0.9)


def test_chosen_token_counts_when_missing_from_top_logprobs():
    backend = make_backend(Recorder(completion({}, chosen=" No")))

    assert backend.noul("x", "y") == pytest.approx(0.0)


def test_chosen_token_is_not_double_counted_when_also_in_top_logprobs():
    body = completion({"yes": 0.5, "no": 0.3, "unknown": 0.2}, chosen="yes")
    backend = make_backend(Recorder(body))

    assert backend.noul("x", "y") == pytest.approx(0.6)


def test_noul_prompt_fences_input_and_defines_the_three_labels():
    recorder = Recorder()
    backend = make_backend(recorder)

    backend.noul("The invoice was paid on 3 May.", "The invoice is outstanding.")

    system, user = (m["content"] for m in recorder.payload["messages"])
    assert "classifier" in system.lower()
    assert "exactly one label" in system.lower()
    assert '"""\nThe invoice was paid on 3 May.\n"""' in user
    assert "The invoice is outstanding." in user
    for label in ("yes", "no", "unknown"):
        assert f'"{label}"' in user


def test_very_small_logprobs_keep_their_relative_weights():
    first = {
        "token": "Sure",
        "logprob": -0.01,
        "top_logprobs": [
            {"token": "yes", "logprob": -800 + math.log(0.75)},
            {"token": "no", "logprob": -800 + math.log(0.25)},
        ],
    }
    backend = make_backend(Recorder(_choice_body({"content": "Sure"}, {"content": [first]})))

    assert backend.noul("x", "y") == pytest.approx(0.75)


def test_non_finite_logprobs_are_ignored():
    first = {
        "token": "yes",
        "logprob": math.log(0.6),
        "top_logprobs": [
            {"token": "yes", "logprob": math.log(0.6)},
            {"token": "no", "logprob": math.log(0.4)},
            {"token": "unknown", "logprob": float("nan")},
            {"token": " Unknown", "logprob": float("inf")},
            {"token": "No.", "logprob": float("-inf")},
        ],
    }
    body = json.dumps(_choice_body({"content": "yes"}, {"content": [first]})).encode()
    backend = make_backend(lambda request: httpx.Response(200, content=body))

    assert backend.noul("x", "y") == pytest.approx(0.6)


# --------------------------------------------------------------------------- choice


def test_choice_maps_letters_to_options_in_order_and_ignores_extra_letters():
    top = {"B": 0.5, " A": 0.2, "D": 0.2, "c.": 0.1}  # D is beyond the 3 options
    backend = make_backend(Recorder(completion(top)))

    probs = backend.choice("I love this!", "What is the sentiment?", ["neg", "pos", "neutral"])

    assert probs == pytest.approx([0.2 / 0.8, 0.5 / 0.8, 0.1 / 0.8])


def test_choice_prompt_lists_options_with_letters():
    recorder = Recorder(completion({"A": 0.9, "B": 0.1}))
    backend = make_backend(recorder)

    backend.choice("I love this!", "What is the sentiment?", ["negative", "positive"])

    user = recorder.payload["messages"][1]["content"]
    assert '"""\nI love this!\n"""' in user
    assert "What is the sentiment?" in user
    assert "A. negative\nB. positive" in user


@pytest.mark.parametrize("count", [0, 27])
def test_choice_rejects_unsupported_option_counts_without_calling_the_server(count):
    recorder = Recorder()
    backend = make_backend(recorder)

    with pytest.raises(ValueError):
        backend.choice("x", "q", [f"option {i}" for i in range(count)])
    assert recorder.requests == []


# --------------------------------------------------------------------------- score

RUBRIC = ["poor", "fair", "good", "excellent"]


def test_score_returns_distribution_over_rubric_levels_in_order():
    top = {"C": 0.6, "D": 0.25, "b": 0.1, "A)": 0.05}
    backend = make_backend(Recorder(completion(top)))

    probs = backend.score("Great answer, minor typo.", "How good is the answer?", RUBRIC)

    assert probs == pytest.approx([0.05, 0.1, 0.6, 0.25])


def test_score_prompt_lists_levels_from_lowest_to_highest():
    recorder = Recorder(completion({"A": 1.0}))
    backend = make_backend(recorder)

    backend.score("Great answer.", "How good is the answer?", RUBRIC)

    user = recorder.payload["messages"][1]["content"]
    assert '"""\nGreat answer.\n"""' in user
    assert "How good is the answer?" in user
    assert "A. poor\nB. fair\nC. good\nD. excellent" in user
    assert "lowest (A)" in user
    assert "highest (D)" in user


# --------------------------------------------------------------------------- fallback


@pytest.mark.parametrize(("content", "expected"), [("Yes", 1.0), (" no.", 0.0), ("Unknown", 0.5)])
def test_noul_falls_back_to_content_when_logprobs_are_missing(content, expected):
    backend = make_backend(Recorder(completion(None, content=content)))

    assert backend.noul("x", "y") == expected


def test_choice_falls_back_to_one_hot_content_when_logprobs_are_missing():
    backend = make_backend(Recorder(completion(None, content="B")))

    assert backend.choice("x", "q", ["a", "b", "c"]) == [0.0, 1.0, 0.0]


def test_content_fallback_logs_a_warning_once_per_backend(caplog):
    backend = make_backend(Recorder(completion(None, content="yes")))

    with caplog.at_level("WARNING", logger="noulo.inference.openai_backend"):
        backend.noul("x", "y")
        backend.noul("x", "y")

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "logprobs" in warnings[0].getMessage()


@pytest.mark.parametrize("content", ["Maybe", "yes no", "", "D"])
def test_unparseable_content_without_logprobs_raises_backend_error(content):
    backend = make_backend(Recorder(completion(None, content=content)))

    with pytest.raises(BackendError):
        backend.choice("x", "q", ["a", "b", "c"])


def test_logprobs_without_any_label_token_and_unparseable_content_raise_backend_error():
    backend = make_backend(Recorder(completion({"Sure": 0.8, "The": 0.2})))

    with pytest.raises(BackendError):
        backend.noul("x", "y")


# --------------------------------------------------------------------------- errors

API_KEY = "sk-test-DO-NOT-LEAK-4f9a"


@pytest.mark.parametrize("status", [401, 500])
def test_http_error_status_raises_backend_error_with_status_but_not_api_key(status):
    body = {"error": {"message": f"Incorrect API key provided: {API_KEY}"}}
    backend = make_backend(Recorder(body, status=status), api_key=API_KEY)

    with pytest.raises(BackendError) as excinfo:
        backend.noul("x", "y")

    message = str(excinfo.value)
    assert str(status) in message
    assert API_KEY not in message


@pytest.mark.parametrize(
    ("exc_type", "expected"),
    [(httpx.ReadTimeout, "timed out"), (httpx.ConnectError, "ConnectError")],
)
def test_transport_failures_raise_backend_error_without_api_key(exc_type, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        # Even if a lower layer echoed the credentials, they must not reach our message.
        raise exc_type(f"failed with {request.headers['Authorization']}", request=request)

    backend = make_backend(handler, api_key=API_KEY)

    with pytest.raises(BackendError) as excinfo:
        backend.noul("x", "y")

    message = str(excinfo.value)
    assert expected in message
    assert API_KEY not in message


def test_non_json_response_raises_backend_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>Bad gateway</html>")

    with pytest.raises(BackendError):
        make_backend(handler).noul("x", "y")


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"choices": None},
        {"choices": []},
        {"choices": ["yes"]},
        {"choices": [{}]},
        _choice_body(None, None),
        _choice_body({"content": None}, None),
        _choice_body({"content": 42}, None),
        _choice_body({"content": "?"}, {"content": [{"token": 1, "logprob": 0.0}]}),
        _choice_body({"content": "?"}, {"content": [{"token": "yes", "logprob": "high"}]}),
        _choice_body({"content": "?"}, {"content": [{"token": "yes"}]}),
        _choice_body({"content": "?"}, {"content": "yes"}),
        _choice_body({"content": "?"}, {"content": [{"top_logprobs": [None, 3]}]}),
    ],
)
def test_malformed_completion_raises_backend_error(body):
    backend = make_backend(Recorder(body))

    with pytest.raises(BackendError):
        backend.noul("x", "y")


def test_empty_logprobs_content_falls_back_to_message_content():
    backend = make_backend(Recorder(_choice_body({"content": "yes"}, {"content": []})))

    assert backend.noul("x", "y") == 1.0


# --------------------------------------------------------------------------- info


def test_info_describes_backend_without_secrets():
    backend = make_backend(Recorder(), id="ollama-llama", model="llama3.2", api_key=API_KEY)

    assert backend.info == BackendInfo(
        id="ollama-llama", backend="openai", model="llama3.2", quantization="n/a", local=True
    )
    assert API_KEY not in repr(backend.info)


@pytest.mark.parametrize(
    ("base_url", "local"),
    [
        ("http://127.0.0.1:11434/v1", True),
        ("http://localhost:1234/v1/", True),
        ("http://LOCALHOST:8000/v1", True),
        ("http://[::1]:8080/v1", True),
        ("https://api.openai.com/v1", False),
        ("http://192.168.1.20:11434/v1", False),
        ("http://localhost.example.com/v1", False),
    ],
)
def test_info_local_is_true_only_for_loopback_hosts(base_url, local):
    assert make_backend(Recorder(), base_url=base_url).info.local is local


# --------------------------------------------------------------------------- lifecycle


def test_warmup_performs_one_tiny_noul_request():
    recorder = Recorder()
    backend = make_backend(recorder)

    backend.warmup()

    assert len(recorder.requests) == 1
    assert '"""\nok\n"""' in recorder.payload["messages"][1]["content"]


@pytest.mark.parametrize(
    "recorder", [Recorder({"error": "boom"}, status=503), Recorder(completion(None, content="?"))]
)
def test_warmup_failure_raises_model_load_error_wrapping_the_cause(recorder):
    backend = make_backend(recorder, api_key=API_KEY)

    with pytest.raises(ModelLoadError) as excinfo:
        backend.warmup()

    assert isinstance(excinfo.value.__cause__, BackendError)
    assert API_KEY not in str(excinfo.value)


def test_close_is_idempotent_and_later_calls_raise_backend_error():
    backend = OpenAIBackend(id="x", base_url=LOCAL_URL, model="m")  # owns its client

    backend.close()
    backend.close()

    with pytest.raises(BackendError):
        backend.noul("x", "y")


def test_close_leaves_an_injected_client_open():
    client = httpx.Client(transport=httpx.MockTransport(Recorder()))
    backend = OpenAIBackend(id="x", base_url=LOCAL_URL, model="m", client=client)

    backend.close()

    assert not client.is_closed
    client.close()


def test_implements_the_decision_backend_protocol():
    assert isinstance(make_backend(Recorder()), DecisionBackend)


def test_identical_inputs_produce_identical_requests():
    recorder = Recorder(completion({"A": 0.5, "B": 0.5}))
    backend = make_backend(recorder, api_key=API_KEY, extra_body={"seed": 1})

    backend.score("same input", "same question", RUBRIC)
    backend.score("same input", "same question", RUBRIC)

    first, second = recorder.requests
    assert first.content == second.content
    assert first.headers["Authorization"] == second.headers["Authorization"]


# --------------------------------------------------------------------------- properties

TOKENS = [
    "yes", " Yes", "YES.", "no", " No", "no)", "unknown", '"Unknown":',
    "A", " b", "C.", "(D)", "e", "maybe", "The", "", " ",
]  # fmt: skip
logprob_values = st.floats(min_value=-40.0, max_value=0.0) | st.floats()  # incl. nan/inf/huge
top_entries = st.lists(st.tuples(st.sampled_from(TOKENS), logprob_values), max_size=20)


def _lenient_json_handler(body: Any) -> Handler:
    """Encode like Python servers do, allowing NaN/Infinity literals."""
    content = json.dumps(body).encode()
    return lambda request: httpx.Response(
        200, content=content, headers={"content-type": "application/json"}
    )


@settings(max_examples=300, deadline=None)
@given(
    entries=top_entries,
    chosen=st.sampled_from(TOKENS),
    chosen_logprob=logprob_values,
    primitive=st.sampled_from(["noul", "choice", "score"]),
    count=st.integers(min_value=1, max_value=6),
)
def test_outputs_obey_probability_invariants(entries, chosen, chosen_logprob, primitive, count):
    first = {
        "token": chosen,
        "logprob": chosen_logprob,
        "top_logprobs": [{"token": t, "logprob": lp} for t, lp in entries],
    }
    body = _choice_body({"content": chosen}, {"content": [first]})
    backend = make_backend(_lenient_json_handler(body))
    candidates = [f"candidate {i}" for i in range(count)]

    try:
        if primitive == "noul":
            values = [backend.noul("x", "y")]
        else:
            values = getattr(backend, primitive)("x", "q", candidates)
    except BackendError:
        return  # no allowed label anywhere: a clean error is acceptable

    assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values)
    if primitive != "noul":
        assert len(values) == count
        assert sum(values) == pytest.approx(1.0, abs=1e-6)
