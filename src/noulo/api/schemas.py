"""Response and small request schemas for the REST API (documented in OpenAPI)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str = Field(examples=["INVALID_REQUEST"])
    message: str = Field(examples=["Noul requires a proposition."])


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    modelLoaded: bool


class InfoResponse(BaseModel):
    name: str
    version: str
    model: str
    quantization: str
    capabilities: list[str]
    backend: str
    local: bool
    learning: bool
    device: str = Field(
        description="Where the model runs: cpu, coreml, cuda, directml, rocm, remote"
    )


class NoulResponse(BaseModel):
    type: Literal["noul"]
    value: float = Field(ge=0.0, le=1.0, description="P(proposition is true | input).")
    diagnostics: dict[str, Any] | None = None


class ChoiceResponse(BaseModel):
    type: Literal["choice"]
    value: str = Field(description="One of the supplied choice IDs.")
    diagnostics: dict[str, Any] | None = None


class ScoreResponse(BaseModel):
    type: Literal["score"]
    value: float = Field(ge=0.0, le=1.0, description="Expected rubric level / highest level.")
    diagnostics: dict[str, Any] | None = None


EvaluateResponse = NoulResponse | ChoiceResponse | ScoreResponse


class ModelEntry(BaseModel):
    id: str
    backend: str
    model: str
    quantization: str
    local: bool
    installed: bool
    description: str
    tier: str | None = Field(
        None,
        description="basic (< 200 MB), large (0.5-1 GB), llm (small 4-bit LLM) or experimental",
    )
    label: str | None = Field(None, description="Short name used in model menus")
    sizeMB: int | None = Field(None, description="Download size of the model file")
    ramMB: int | None = Field(None, description="Measured peak RAM while benchmarking")
    noulAccuracy: float | None = None
    choiceAccuracy: float | None = None
    scoreMae: float | None = None
    p50Ms: float | None = None


class ModelsResponse(BaseModel):
    active: str
    models: list[ModelEntry]


class ActiveModel(BaseModel):
    id: str
    backend: str
    model: str
    quantization: str
    local: bool


class SwitchModelRequest(BaseModel):
    id: str = Field(min_length=1, description="Registry id of the model to activate.")


class SwitchModelResponse(BaseModel):
    active: ActiveModel


class RegisterEndpointRequest(BaseModel):
    id: str = Field(min_length=1)
    baseUrl: str = Field(description="OpenAI-compatible base URL, e.g. http://127.0.0.1:11434/v1")
    model: str = Field(min_length=1)
    apiKeyEnv: str | None = Field(None, description="Name of the env var holding the API key.")


class LearningToggle(BaseModel):
    enabled: bool


class LearningStatus(BaseModel):
    enabled: bool
    stats: dict[str, Any] | None = None


class FeedbackRequest(BaseModel):
    recordId: str | None = Field(None, description="X-Record-Id returned by an evaluation.")
    request: dict[str, Any] | None = Field(
        None, description="Alternatively, a full /api/v1/evaluate request to teach directly."
    )
    expected: float | bool | str = Field(
        description="Correct outcome: Noul/Score value in [0,1] (or true/false), or a Choice ID."
    )


class FeedbackResponse(BaseModel):
    recordId: str
    verified: bool = True


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse, "description": description}
    for code, description in {
        400: "Malformed JSON",
        401: "Missing or invalid API key",
        413: "Payload or input too large",
        422: "Invalid request",
        500: "Inference error",
        502: "Upstream model endpoint error",
        503: "Model not ready, busy or shutting down",
    }.items()
}
