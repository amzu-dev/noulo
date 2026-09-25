"""FastAPI application factory: lifespan, security, CORS, errors, OpenAPI."""

from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter
from pydantic.json_schema import models_json_schema
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..config import Settings
from ..inference.engine import (
    DecisionEngine,
    EngineBusy,
    EngineUnavailable,
    InferenceError,
    RecordNotFound,
)
from ..inference.types import ModelLoadError
from ..registry import ModelRegistry, UnknownModelError
from . import routes, schemas
from .validation import ChoiceRequest, EvaluateRequest, NoulRequest, RequestError, ScoreRequest

log = logging.getLogger(__name__)
UI_DIR = Path(__file__).resolve().parent.parent / "ui" / "static"
REF = "#/components/schemas/{model}"

_bearer = HTTPBearer(auto_error=False, description="Required only when NOULO_API_KEY is set.")


def _error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}}, status_code=status, headers=headers
    )


def _require_api_key(
    request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)
) -> None:
    expected = request.app.state.settings.api_key
    if expected is None:
        return
    supplied = credentials.credentials if credentials else ""
    if not hmac.compare_digest(supplied.encode(), expected.get_secret_value().encode()):
        raise RequestError(
            401, "UNAUTHORIZED", "A valid 'Authorization: Bearer <key>' is required."
        )


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestError)
    async def _request_error(_, exc: RequestError):
        headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
        return _error(exc.status, exc.code, exc.message, headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_, exc: RequestValidationError):
        first = exc.errors()[0]
        if first.get("type") == "json_invalid":
            return _error(400, "MALFORMED_JSON", "Request body is not valid JSON.")
        field = ".".join(str(p) for p in first.get("loc", ()) if p != "body") or "body"
        return _error(422, "INVALID_REQUEST", f"Invalid '{field}': {first.get('msg')}.")

    @app.exception_handler(EngineUnavailable)
    async def _unavailable(_, exc: EngineUnavailable):
        status = 409 if exc.code == "LEARNING_DISABLED" else 503
        return _error(status, exc.code, exc.message)

    @app.exception_handler(EngineBusy)
    async def _busy(_, exc: EngineBusy):
        return _error(503, "ENGINE_BUSY", str(exc), {"Retry-After": "1"})

    @app.exception_handler(InferenceError)
    async def _inference(_, exc: InferenceError):
        if exc.remote:
            return _error(502, "UPSTREAM_ERROR", "The model endpoint failed to answer.")
        return _error(500, "INFERENCE_ERROR", "Inference failed.")

    @app.exception_handler(RecordNotFound)
    async def _record_not_found(_, exc: RecordNotFound):
        return _error(404, "RECORD_NOT_FOUND", f"No learning record {exc.args[0]!r}.")

    @app.exception_handler(UnknownModelError)
    async def _unknown_model(_, exc: UnknownModelError):
        return _error(404, "MODEL_NOT_FOUND", str(exc))

    @app.exception_handler(ModelLoadError)
    async def _load_failed(_, exc: ModelLoadError):
        return _error(500, "MODEL_LOAD_FAILED", str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_, exc: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return _error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(_, exc: Exception):
        log.exception("Unhandled error", exc_info=exc)
        return _error(500, "INTERNAL_ERROR", "Internal server error.")


def _install_openapi(app: FastAPI) -> None:
    def openapi():
        if app.openapi_schema:
            return app.openapi_schema
        spec = get_openapi(
            title=app.title, version=app.version, description=app.description, routes=app.routes
        )
        components = spec.setdefault("components", {}).setdefault("schemas", {})
        _, defs = models_json_schema(
            [
                (NoulRequest, "validation"),
                (ChoiceRequest, "validation"),
                (ScoreRequest, "validation"),
                (schemas.ErrorResponse, "validation"),
            ],
            ref_template=REF,
        )
        components.update(defs.get("$defs", {}))
        evaluate = TypeAdapter(EvaluateRequest).json_schema(ref_template=REF)
        components.update(evaluate.pop("$defs", {}))
        components["EvaluateRequest"] = evaluate
        app.openapi_schema = spec
        return spec

    app.openapi = openapi


def create_app(
    settings: Settings,
    *,
    engine: DecisionEngine | None = None,
    registry: ModelRegistry | None = None,
) -> FastAPI:
    from ..runtime import build_engine, build_registry

    registry = registry or build_registry(settings)
    engine = engine or build_engine(settings, registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine.state == "created":
            await run_in_threadpool(engine.start)
        yield
        await run_in_threadpool(engine.shutdown, settings.shutdown_timeout)

    app = FastAPI(
        title="noulo",
        version=__version__,
        description="Local semantic decision service: Choice, Score and Noul primitives.",
        lifespan=lifespan,
    )
    app.state.settings, app.state.engine, app.state.registry = settings, engine, registry

    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_origin_regex=settings.cors_origin_regex,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
            expose_headers=["X-Record-Id", "Retry-After"],
        )

    _install_error_handlers(app)
    app.include_router(routes.health_router)
    app.include_router(routes.router, dependencies=[Depends(_require_api_key)])
    _install_openapi(app)

    if settings.ui_enabled and UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def root():
            return RedirectResponse("/ui/")

    return app
