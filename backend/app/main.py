from __future__ import annotations

import asyncio
import json
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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

app = FastAPI(title="Rephraser API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class HealthResponse(BaseModel):
    status: str = "ok"


class RephraseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    styles: List[str] = Field(..., min_length=1)

    @staticmethod
    def _allowed_styles() -> set[str]:
        return {"professional", "casual", "polite", "social"}

    def model_post_init(self, __context: Any) -> None:
        """Reject any style that is not in the supported list."""
        allowed = self._allowed_styles()
        validated_styles: List[str] = []
        for style in self.styles:
            if style not in allowed:
                raise ValueError(f"style must be one of {sorted(allowed)}")
            validated_styles.append(style)
        self.styles = validated_styles


class RephraseResponse(BaseModel):
    results: Dict[str, str]


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process; FastAPI reuses this cached instance."""
    settings = Settings()
    settings.validate()
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


@app.post("/rephrase", response_model=RephraseResponse)
async def rephrase(
    payload: RephraseRequest,
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


@app.post("/rephrase/stream")
async def rephrase_stream(
    payload: RephraseRequest,
    request: Request,
    client: LLMClient = Depends(get_llm_client),
    tasks: RequestTaskRegistry = Depends(provide_request_task_registry),
) -> StreamingResponse:
    async def event_stream() -> AsyncGenerator[str, None]:
        await tasks.register_task(asyncio.current_task())
        for style in payload.styles:
            style_id = str(uuid.uuid4())
            try:
                async for chunk in client.rephrase_stream(payload.text, style):
                    if await request.is_disconnected():
                        return
                    yield _sse_message(
                        "chunk",
                        {"style": style, "delta": chunk, "done": False},
                        event_id=style_id,
                    )
                if await request.is_disconnected():
                    return
                yield _sse_message(
                    "chunk",
                    {"style": style, "delta": "", "done": True},
                    event_id=style_id,
                )
            except ProviderError as exc:
                raise _handle_provider_error(exc) from exc
            except asyncio.CancelledError:
                # Disconnection triggered cancellation.
                return
        if not await request.is_disconnected():
            yield _sse_message("done", {"done": True})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


__all__ = [
    "app",
    "get_llm_client",
    "get_settings",
]
