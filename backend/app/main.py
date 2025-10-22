from __future__ import annotations

import asyncio
import json
import logging
import math
import secrets
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, Dict, List, Tuple, cast

from fastapi import Depends, FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import (
    AuthenticationError,
    LLMClient,
    MockLLMClient,
    ProviderConfigError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from app.providers.openai_provider import OpenAIProvider
from app.task_manager import RequestTaskRegistry, provide_request_task_registry


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds the configured size."""

    def __init__(self, app: FastAPI, *, max_body_size: int) -> None:
        super().__init__(app)
        self.max_body_size = max_body_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            with suppress(ValueError):
                if int(content_length) > self.max_body_size:
                    return JSONResponse(status_code=413, content={"detail": "Request body too large"})

        body = await request.body()
        if len(body) > self.max_body_size:
            return JSONResponse(status_code=413, content={"detail": "Request body too large"})

        async def receive() -> Dict[str, Any]:
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive)  # type: ignore[arg-type]
        return await call_next(request)


MAX_BODY_BYTES = 16_384
QueueItem = Tuple[str, Any]
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
_LOGGING_CONFIGURED = False


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    reset_time = getattr(exc, "reset_time", time.time())
    retry_after = max(1, math.ceil(reset_time - time.time()))
    return JSONResponse(
        status_code=429,
        content={"detail": "Too Many Requests"},
        headers={"Retry-After": str(retry_after)},
    )


class JSONLogFormatter(logging.Formatter):
    def __init__(self, *, secrets_to_redact: list[str]):
        super().__init__()
        self._secrets = [s for s in secrets_to_redact if s]

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            redacted = value
            for secret in self._secrets:
                if secret and secret in redacted:
                    redacted = redacted.replace(secret, "***")
            return redacted
        return value

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": self._redact(record.getMessage()),
        }
        req_id = getattr(record, "request_id", None)
        if req_id:
            payload["request_id"] = req_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                continue
            if key not in payload and value is not None:
                payload[key] = self._redact(value)
        return json.dumps(payload, default=str)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def configure_logging(settings: Settings) -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    secrets_to_redact = [
        settings.openai_api_key or "",
        settings.anthropic_api_key or "",
        settings.auth_bearer_token or "",
    ]
    formatter = JSONLogFormatter(secrets_to_redact=secrets_to_redact)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    for logger_name in ("uvicorn.access", "uvicorn.error", "app", "app.requests"):
        logger = logging.getLogger(logger_name)
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)

    _LOGGING_CONFIGURED = True


app = FastAPI(title="Rephraser API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
    )

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    default_provider: str = Field(default="auto")
    request_timeout_seconds: float = Field(default=10.0, gt=0)
    allowed_origins_env: str | None = Field(default=None, validation_alias="ALLOWED_ORIGINS")
    allowed_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    auth_bearer_token: str | None = Field(default=None, validation_alias="AUTH_BEARER_TOKEN")

    @staticmethod
    def _is_empty(value: str | None) -> bool:
        return value is None or value.strip() == ""

    def validate(self) -> None:
        """Ensure our runtime configuration is coherent before use."""
        provider = self.default_provider.lower()
        if provider not in {"auto", "openai", "anthropic", "mock"}:
            raise ValueError("default_provider must be one of: auto, openai, anthropic, mock")
        if provider == "openai" and self._is_empty(self.openai_api_key):
            raise ValueError("OPENAI_API_KEY is required when default_provider=openai")
        if provider == "anthropic" and self._is_empty(self.anthropic_api_key):
            raise ValueError("ANTHROPIC_API_KEY is required when default_provider=anthropic")
        if self.allowed_origins_env:
            parsed = [
                origin.strip()
                for origin in self.allowed_origins_env.split(",")
                if origin and origin.strip()
            ]
            if parsed:
                self.allowed_origins = parsed
        if not self.allowed_origins:
            self.allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]


class HealthResponse(BaseModel):
    status: str = "ok"


class RephraseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    styles: List[str] = Field(..., min_length=1, max_length=4)

    @staticmethod
    def _allowed_styles() -> set[str]:
        return {"professional", "casual", "polite", "social"}

    def model_post_init(self, __context: Any) -> None:
        """Reject any style that is not in the supported list."""
        allowed = self._allowed_styles()
        validated_styles: List[str] = []
        seen = set()
        for style in self.styles:
            if style not in allowed:
                raise ValueError(f"style must be one of {sorted(allowed)}")
            if style in seen:
                continue
            seen.add(style)
            validated_styles.append(style)
        self.styles = validated_styles


class RephraseResponse(BaseModel):
    results: Dict[str, str]


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process; FastAPI reuses this cached instance."""
    settings = Settings()
    settings.validate()
    configure_logging(settings)
    return settings


def build_llm_client(settings: Settings) -> LLMClient:
    """Pick the most suitable provider implementation based on configuration."""
    provider_choice = settings.default_provider.lower()

    if provider_choice == "mock":
        return MockLLMClient()

    if provider_choice in {"openai", "auto"} and settings.openai_api_key:
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout_seconds,
        )

    if provider_choice in {"anthropic", "auto"} and settings.anthropic_api_key:
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            timeout=settings.request_timeout_seconds,
        )

    # No matching provider is configured; fall back to mock implementation.
    return MockLLMClient()


def configure_app(app: FastAPI) -> None:
    settings = get_settings()
    app.add_middleware(BodySizeLimitMiddleware, max_body_size=MAX_BODY_BYTES)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SlowAPIMiddleware)


request_logger = logging.getLogger("app.requests")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    token = request_id_ctx.set(request_id)
    start = time.perf_counter()
    try:
        request_logger.info(
            "request.start",
            extra={"method": request.method, "path": request.url.path},
        )
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        request_logger.info(
            "request.end",
            extra={
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response
    except Exception:
        request_logger.exception(
            "request.error",
            extra={"path": request.url.path, "method": request.method},
        )
        raise
    finally:
        request_id_ctx.reset(token)


configure_app(app)


def require_bearer_token(
    authorization: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    expected = settings.auth_bearer_token
    if expected is None:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    provided = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


def get_llm_client(settings: Settings = Depends(get_settings)) -> LLMClient:
    try:
        return build_llm_client(settings)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ProviderConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


def _handle_provider_error(exc: ProviderError) -> HTTPException:
    if isinstance(exc, ProviderTimeoutError):
        return HTTPException(status_code=504, detail="Upstream provider timed out")
    if isinstance(exc, RateLimitError):
        return HTTPException(status_code=429, detail="Upstream provider rate limited the request")
    if isinstance(exc, AuthenticationError):
        return HTTPException(status_code=401, detail="Invalid or missing provider credentials")
    return HTTPException(status_code=502, detail="Upstream provider error")


@limiter.limit("30/minute")
@app.post("/rephrase", response_model=RephraseResponse)
async def rephrase(
    payload: RephraseRequest,
    request: Request,
    _: None = Depends(require_bearer_token),
    client: LLMClient = Depends(get_llm_client),
    tasks: RequestTaskRegistry = Depends(provide_request_task_registry),
) -> RephraseResponse:
    # We independently call the provider for each requested style so partial failures
    # map cleanly to HTTP errors without returning partial results.
    results: Dict[str, str] = {}

    style_tasks = {
        style: await tasks.create_task(client.rephrase(payload.text, style))
        for style in payload.styles
    }

    for style, task in style_tasks.items():
        try:
            rewritten = await task
        except asyncio.CancelledError as exc:
            raise HTTPException(status_code=499, detail="Client cancelled the request") from exc
        except ProviderError as exc:
            raise _handle_provider_error(exc) from exc
        results[style] = rewritten

    return RephraseResponse(results=results)


def _sse_message(event: str, data: Dict[str, Any], *, event_id: str | None = None) -> str:
    body = [f"event: {event}"]
    if event_id is not None:
        body.append(f"id: {event_id}")
    body.append(f"data: {json.dumps(data)}")
    return "\n".join(body) + "\n\n"


@limiter.limit("30/minute")
@app.post("/rephrase/stream")
async def rephrase_stream(
    payload: RephraseRequest,
    request: Request,
    _: None = Depends(require_bearer_token),
    client: LLMClient = Depends(get_llm_client),
    tasks: RequestTaskRegistry = Depends(provide_request_task_registry),
) -> StreamingResponse:
    message_queue: asyncio.Queue[QueueItem] = asyncio.Queue()
    total_styles = len(payload.styles)

    async def stream_style(style: str) -> None:
        style_id = str(uuid.uuid4())
        try:
            async for chunk in client.rephrase_stream(payload.text, style):
                if await request.is_disconnected():
                    return
                await message_queue.put(
                    (
                        "data",
                        _sse_message(
                            "chunk",
                            {"style": style, "delta": chunk, "done": False},
                            event_id=style_id,
                        ),
                    )
                )
            if await request.is_disconnected():
                return
            await message_queue.put(
                (
                    "data",
                    _sse_message(
                        "chunk",
                        {"style": style, "delta": "", "done": True},
                        event_id=style_id,
                    ),
                )
            )
        except ProviderError as exc:
            await message_queue.put(("error", exc))
        finally:
            await message_queue.put(("complete", style))

    for style in payload.styles:
        await tasks.create_task(stream_style(style))

    async def event_stream() -> AsyncGenerator[str, None]:
        current = asyncio.current_task()
        await tasks.register_task(current)
        completed = 0
        try:
            while completed < total_styles:
                kind, payload_item = await message_queue.get()
                if kind == "data":
                    yield cast(str, payload_item)
                elif kind == "error":
                    raise _handle_provider_error(cast(ProviderError, payload_item))
                elif kind == "complete":
                    completed += 1
            if not await request.is_disconnected():
                yield _sse_message("done", {"done": True})
        finally:
            await tasks.cancel_all(exclude={current} if current else None)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


__all__ = [
    "app",
    "get_llm_client",
    "get_settings",
]
