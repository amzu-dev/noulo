"""REST routes. Every primitive endpoint funnels into the same DecisionEngine."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..inference.engine import EngineResult
from ..inference.types import PRIMITIVES
from . import schemas
from .validation import RequestError, parse_request


def _body_schema(name: str) -> dict[str, Any]:
    return {
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{name}"}}},
        }
    }


async def _parse(request: Request, primitive: str):
    settings = request.app.state.settings
    body = await request.body()
    if len(body) > settings.max_body_bytes:
        raise RequestError(
            413, "PAYLOAD_TOO_LARGE", f"Request body exceeds {settings.max_body_bytes} bytes."
        )
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise RequestError(400, "MALFORMED_JSON", "Request body is not valid JSON.") from None
    return parse_request(primitive, payload, settings.limits)


def _respond(result: EngineResult, response: Response) -> dict[str, Any]:
    if result.record_id:
        response.headers["X-Record-Id"] = result.record_id
    body = result.to_dict()
    if result.diagnostics is not None:
        body["diagnostics"] = result.diagnostics
    return body


async def _evaluate(request: Request, response: Response, primitive: str, diagnostics: bool):
    parsed = await _parse(request, primitive)
    engine = request.app.state.engine
    result = await run_in_threadpool(engine.evaluate, parsed, diagnostics=diagnostics)
    return _respond(result, response)


DiagnosticsFlag = Query(False, description="Include model probabilities and learning influence.")

health_router = APIRouter(tags=["health"])
router = APIRouter(prefix="/api/v1", responses=schemas.ERROR_RESPONSES)


@health_router.get(
    "/health",
    response_model=schemas.HealthResponse,
    responses={503: {"model": schemas.HealthResponse, "description": "Not ready"}},
)
async def health(request: Request):
    engine = request.app.state.engine
    if engine.ready:
        return {"status": "ok", "modelLoaded": True}
    return JSONResponse({"status": engine.state, "modelLoaded": False}, status_code=503)


@router.get("/info", response_model=schemas.InfoResponse, tags=["health"])
async def info(request: Request):
    engine = request.app.state.engine
    model = engine.model_info
    return {
        "name": "noulo",
        "version": __version__,
        "model": model.id,
        "quantization": model.quantization,
        "capabilities": list(PRIMITIVES),
        "backend": model.backend,
        "local": model.local,
        "learning": engine.learning_enabled,
        "device": model.device,
    }


@router.post(
    "/evaluate",
    response_model=schemas.EvaluateResponse,
    response_model_exclude_none=True,
    openapi_extra=_body_schema("EvaluateRequest"),
    tags=["decisions"],
    summary="Evaluate any primitive (dispatch on `type`)",
)
async def evaluate(request: Request, response: Response, diagnostics: bool = DiagnosticsFlag):
    return await _evaluate(request, response, "evaluate", diagnostics)


@router.post(
    "/noul",
    response_model=schemas.NoulResponse,
    response_model_exclude_none=True,
    openapi_extra=_body_schema("NoulRequest"),
    tags=["decisions"],
    summary="Noul: P(proposition is true | input)",
)
async def noul(request: Request, response: Response, diagnostics: bool = DiagnosticsFlag):
    return await _evaluate(request, response, "noul", diagnostics)


@router.post(
    "/choice",
    response_model=schemas.ChoiceResponse,
    response_model_exclude_none=True,
    openapi_extra=_body_schema("ChoiceRequest"),
    tags=["decisions"],
    summary="Choice: select exactly one supplied option",
)
async def choice(request: Request, response: Response, diagnostics: bool = DiagnosticsFlag):
    return await _evaluate(request, response, "choice", diagnostics)


@router.post(
    "/score",
    response_model=schemas.ScoreResponse,
    response_model_exclude_none=True,
    openapi_extra=_body_schema("ScoreRequest"),
    tags=["decisions"],
    summary="Score: place the input on an ordered rubric (0-1)",
)
async def score(request: Request, response: Response, diagnostics: bool = DiagnosticsFlag):
    return await _evaluate(request, response, "score", diagnostics)


# ---------------------------------------------------------------------- models


@router.get("/models", response_model=schemas.ModelsResponse, tags=["models"])
async def list_models(request: Request):
    engine, registry = request.app.state.engine, request.app.state.registry
    return {"active": engine.model_info.id, "models": registry.list()}


@router.put(
    "/models/active",
    response_model=schemas.SwitchModelResponse,
    tags=["models"],
    summary="Switch the active model at runtime",
)
async def switch_model(body: schemas.SwitchModelRequest, request: Request):
    info = await run_in_threadpool(request.app.state.engine.switch_model, body.id)
    return {
        "active": {
            "id": info.id,
            "backend": info.backend,
            "model": info.model,
            "quantization": info.quantization,
            "local": info.local,
        }
    }


@router.post(
    "/models",
    status_code=201,
    response_model=schemas.ModelEntry,
    tags=["models"],
    summary="Register an OpenAI-compatible endpoint",
)
async def register_model(body: schemas.RegisterEndpointRequest, request: Request):
    registry = request.app.state.registry
    try:
        registry.register_openai(
            id=body.id, base_url=body.baseUrl, model=body.model, api_key_env=body.apiKeyEnv
        )
    except ValueError as exc:
        raise RequestError(422, "INVALID_REQUEST", str(exc)) from None
    return next(m for m in registry.list() if m["id"] == body.id)


@router.delete(
    "/models/{model_id}",
    status_code=204,
    tags=["models"],
    summary="Remove a user-registered endpoint",
)
async def unregister_model(model_id: str, request: Request):
    if model_id == request.app.state.engine.model_info.id:
        raise RequestError(409, "MODEL_IN_USE", "Switch to another model before removing this one.")
    request.app.state.registry.unregister(model_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------- learning


def _learning_status(engine) -> dict[str, Any]:
    memory = engine.memory
    return {"enabled": engine.learning_enabled, "stats": memory.stats() if memory else None}


@router.get("/learning", response_model=schemas.LearningStatus, tags=["learning"])
async def learning_status(request: Request):
    return await run_in_threadpool(_learning_status, request.app.state.engine)


@router.put(
    "/learning",
    response_model=schemas.LearningStatus,
    tags=["learning"],
    summary="Switch learning from inputs on or off at runtime",
)
async def set_learning(body: schemas.LearningToggle, request: Request):
    engine = request.app.state.engine
    await run_in_threadpool(engine.set_learning, body.enabled)
    return await run_in_threadpool(_learning_status, engine)


@router.get("/learning/records", tags=["learning"], summary="Stored cases, newest first")
async def learning_records(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    type: str | None = Query(None, pattern="^(noul|choice|score)$"),
):
    memory = request.app.state.engine.memory
    if memory is None:
        return {"records": []}
    records = await run_in_threadpool(memory.records, limit=limit, offset=offset, primitive=type)
    return {"records": records}


@router.delete("/learning/records", tags=["learning"], summary="Delete all stored cases")
async def clear_learning_records(request: Request):
    memory = request.app.state.engine.memory
    deleted = await run_in_threadpool(memory.clear) if memory is not None else 0
    return {"deleted": deleted}


@router.post(
    "/feedback",
    response_model=schemas.FeedbackResponse,
    tags=["learning"],
    summary="Teach the correct outcome (by X-Record-Id or with a full request)",
)
async def feedback(body: schemas.FeedbackRequest, request: Request):
    engine = request.app.state.engine
    if (body.recordId is None) == (body.request is None):
        raise RequestError(
            422, "INVALID_REQUEST", "Provide exactly one of 'recordId' or 'request'."
        )
    try:
        if body.recordId is not None:
            record_id = await run_in_threadpool(engine.feedback, body.recordId, body.expected)
        else:
            parsed = parse_request("evaluate", body.request, request.app.state.settings.limits)
            record_id = await run_in_threadpool(engine.teach, parsed, body.expected)
    except ValueError as exc:
        raise RequestError(422, "INVALID_REQUEST", str(exc)) from None
    return {"recordId": record_id, "verified": True}
