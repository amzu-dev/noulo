"""Request schemas and validation shared by the REST API and the Python API.

The Pydantic models double as the OpenAPI schemas. `parse_request` turns any
payload into a typed request or raises `RequestError`, whose messages are the
stable, human-readable strings documented for clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

NonEmpty = Annotated[str, Field(min_length=1)]


class RequestError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


@dataclass(frozen=True)
class Limits:
    max_input_chars: int = 5000
    max_text_chars: int = 1000  # proposition, question, choice text, rubric level
    max_choices: int = 20
    max_rubric_levels: int = 11


class _Strict(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, strict=True)


class ChoiceOption(_Strict):
    id: NonEmpty = Field(description="Caller-supplied identifier returned when this option wins.")
    text: NonEmpty = Field(description="Option text evaluated against the input.")


class NoulRequest(_Strict):
    input: NonEmpty = Field(description="Natural-language input to evaluate.")
    proposition: NonEmpty = Field(description="Statement whose truth is evaluated.")


class ChoiceRequest(_Strict):
    input: NonEmpty
    question: NonEmpty = Field(description="Question the options answer.")
    choices: list[ChoiceOption] = Field(
        min_length=1, description="Options; exactly one is returned."
    )

    @field_validator("choices")
    @classmethod
    def _unique_ids(cls, choices: list[ChoiceOption]) -> list[ChoiceOption]:
        seen: set[str] = set()
        for option in choices:
            if option.id in seen:
                raise ValueError(f"Choice IDs must be unique (duplicate: {option.id!r}).")
            seen.add(option.id)
        return choices


class ScoreRequest(_Strict):
    input: NonEmpty
    question: NonEmpty = Field(description="Question the rubric answers.")
    rubric: list[NonEmpty] = Field(
        min_length=2,
        description="Ordered levels, lowest first. Score = expected level / max level.",
    )

    @field_validator("rubric")
    @classmethod
    def _unique_levels(cls, rubric: list[str]) -> list[str]:
        seen: set[str] = set()
        for level in rubric:
            if level.lower() in seen:
                raise ValueError(f"Score rubric levels must be unique (duplicate: {level!r}).")
            seen.add(level.lower())
        return rubric


class NoulEvaluateRequest(NoulRequest):
    type: Literal["noul"]


class ChoiceEvaluateRequest(ChoiceRequest):
    type: Literal["choice"]


class ScoreEvaluateRequest(ScoreRequest):
    type: Literal["score"]


EvaluateRequest = Annotated[
    NoulEvaluateRequest | ChoiceEvaluateRequest | ScoreEvaluateRequest,
    Field(discriminator="type"),
]
AnyRequest = NoulRequest | ChoiceRequest | ScoreRequest

_ADAPTERS: dict[str, Any] = {
    "noul": TypeAdapter(NoulRequest),
    "choice": TypeAdapter(ChoiceRequest),
    "score": TypeAdapter(ScoreRequest),
    "evaluate": TypeAdapter(EvaluateRequest),
}

_FIELD_MESSAGES = {
    "input": "Input is required.",
    "proposition": "Noul requires a proposition.",
    "choices": "Choice requires at least one choice.",
    "rubric": "Score requires a rubric with at least two levels.",
}
_UNSUPPORTED = "Unsupported primitive. Use one of: choice, score, noul."


def _message_for(error: dict, primitive: str) -> tuple[str, str]:
    """Translate one Pydantic error into (code, message)."""
    etype = error["type"]
    loc = list(error["loc"])
    if etype in ("union_tag_invalid", "union_tag_not_found"):
        return "UNSUPPORTED_PRIMITIVE", _UNSUPPORTED
    if loc and loc[0] in ("noul", "choice", "score"):
        primitive, loc = str(loc[0]), loc[1:]
    if not loc:
        return "INVALID_REQUEST", "Request body must be a JSON object."
    field = str(loc[0])
    if etype == "value_error":
        return "INVALID_REQUEST", str(error["ctx"]["error"])
    if field == "choices" and len(loc) > 1:
        return "INVALID_REQUEST", "Each choice needs a non-empty string 'id' and 'text'."
    if field == "rubric" and len(loc) > 1:
        return "INVALID_REQUEST", "Score rubric levels must be non-empty strings."
    if field == "rubric" and etype == "list_type":
        return "INVALID_REQUEST", "Score rubric must be a list of at least two levels."
    if field == "question":
        return "INVALID_REQUEST", f"{primitive.capitalize()} requires a question."
    if field in _FIELD_MESSAGES and etype in ("missing", "too_short", "string_too_short"):
        return "INVALID_REQUEST", _FIELD_MESSAGES[field]
    return "INVALID_REQUEST", f"Invalid '{field}': {error['msg']}."


def validation_error_to_request_error(exc: ValidationError, primitive: str) -> RequestError:
    code, message = _message_for(exc.errors()[0], primitive)
    return RequestError(422, code, message)


def _check_limits(request: AnyRequest, limits: Limits) -> None:
    def too_large(what: str, limit: int) -> RequestError:
        return RequestError(
            413, "INPUT_TOO_LARGE", f"{what} exceeds the limit of {limit} characters."
        )

    if len(request.input) > limits.max_input_chars:
        raise too_large("Input", limits.max_input_chars)
    texts: list[str] = []
    if isinstance(request, NoulRequest):
        texts.append(request.proposition)
    if isinstance(request, ChoiceRequest):
        if len(request.choices) > limits.max_choices:
            raise RequestError(
                422, "INVALID_REQUEST", f"Choice supports at most {limits.max_choices} choices."
            )
        texts += [request.question, *(c.text for c in request.choices)]
    if isinstance(request, ScoreRequest):
        if len(request.rubric) > limits.max_rubric_levels:
            raise RequestError(
                422,
                "INVALID_REQUEST",
                f"Score rubric supports at most {limits.max_rubric_levels} levels.",
            )
        texts += [request.question, *request.rubric]
    if any(len(t) > limits.max_text_chars for t in texts):
        raise too_large("A proposition, question, choice or rubric level", limits.max_text_chars)


def parse_request(primitive: str, payload: Any, limits: Limits) -> AnyRequest:
    """Validate a decoded JSON payload for "noul", "choice", "score" or "evaluate"."""
    adapter = _ADAPTERS.get(primitive)
    if adapter is None:
        raise RequestError(422, "UNSUPPORTED_PRIMITIVE", _UNSUPPORTED)
    if primitive == "evaluate" and isinstance(payload, dict):
        kind = payload.get("type")
        if not isinstance(kind, str) or kind not in ("noul", "choice", "score"):
            raise RequestError(422, "UNSUPPORTED_PRIMITIVE", _UNSUPPORTED)
    try:
        request = adapter.validate_python(payload)
    except ValidationError as exc:
        raise validation_error_to_request_error(exc, primitive) from None
    _check_limits(request, limits)
    return request


def primitive_of(request: AnyRequest) -> str:
    if isinstance(request, NoulRequest):
        return "noul"
    if isinstance(request, ChoiceRequest):
        return "choice"
    return "score"
