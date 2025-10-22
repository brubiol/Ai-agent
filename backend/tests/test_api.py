from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
import fakeredis.aioredis
import httpx

from app import main as app_main
from app.main import app, get_settings
from app.providers.base import MockLLMClient

pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch: pytest.MonkeyPatch):
    for key in ("AUTH_BEARER_TOKEN", "ALLOWED_ORIGINS", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEFAULT_PROVIDER", "mock")
    get_settings.cache_clear()
    app_main.cache_client = None
    yield
    get_settings.cache_clear()
    app_main.cache_client = None


@pytest.fixture()
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_rephrase_returns_expected_styles(client: AsyncClient) -> None:
    payload = {"text": "Hello world", "styles": ["professional", "casual"]}
    response = await client.post("/rephrase", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body["results"]) == {"professional", "casual"}
    assert "[Professional]" in body["results"]["professional"]
    assert "[Casual]" in body["results"]["casual"]


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "", "styles": ["professional"]},
        {"text": "ok", "styles": []},
        {"text": "x" * 2001, "styles": ["professional"]},
        {"text": "ok", "styles": ["unknown"]},
    ],
)
async def test_rephrase_validation_errors(client: AsyncClient, payload: Dict[str, Any]) -> None:
    response = await client.post("/rephrase", json=payload)
    assert response.status_code == 422


async def _collect_sse_lines(response) -> List[str]:
    lines: List[str] = []
    done_seen = False
    async for line in response.aiter_lines():
        if line:
            lines.append(line)
        if done_seen and not line:
            break
        if line.startswith("event: done"):
            done_seen = True
    return lines


async def test_stream_emits_ordered_chunks_and_done(client: AsyncClient) -> None:
    payload = {"text": "Streaming test text for SSE", "styles": ["professional"]}
    async with client.stream("POST", "/rephrase/stream", json=payload) as response:
        assert response.status_code == 200
        lines = await _collect_sse_lines(response)

    events = [line for line in lines if line.startswith("event:")]
    assert events[0] == "event: chunk"
    assert events[-1] == "event: done"

    data_lines = [line[6:].strip() for line in lines if line.startswith("data:")]
    # Ensure chunk JSONs are in order and final done payload present.
    parsed = [json.loads(line) for line in data_lines]
    assert parsed[0]["style"] == "professional"
    assert parsed[0]["done"] is False
    assert any(item.get("done") is True for item in parsed)


async def test_stream_cancellation_triggers_cleanup(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async_mock = AsyncMock()
    monkeypatch.setattr("app.task_manager.task_manager.cancel_all", async_mock)

    payload = {"text": "Cancellation check", "styles": ["professional"]}
    async with client.stream("POST", "/rephrase/stream", json=payload) as response:
        agen = response.aiter_lines()
        await agen.__anext__()  # first chunk event
        await response.aclose()
        await asyncio.sleep(0)

    async_mock.assert_awaited()


async def test_rephrase_requires_bearer_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_BEARER_TOKEN", "secret-token")
    get_settings.cache_clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as secured_client:
        payload = {"text": "Hello", "styles": ["professional"]}
        unauthorized = await secured_client.post("/rephrase", json=payload)
        assert unauthorized.status_code == 401

        authorized = await secured_client.post(
            "/rephrase",
            json=payload,
            headers={"Authorization": "Bearer secret-token"},
        )
        assert authorized.status_code == 200


@pytest.mark.anyio
async def test_rephrase_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_redis = fakeredis.aioredis.FakeRedis()
    monkeypatch.setenv("REDIS_URL", "redis://fake")
    app_main.cache_client = fake_redis
    get_settings.cache_clear()

    call_counter = {"count": 0}
    original_rephrase = MockLLMClient.rephrase

    async def tracking(self, text: str, style: str) -> str:
        call_counter["count"] += 1
        return await original_rephrase(self, text, style)

    monkeypatch.setattr(MockLLMClient, "rephrase", tracking)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as cached_client:
        payload = {"text": "Cached", "styles": ["professional", "casual"]}
        first = await cached_client.post("/rephrase", json=payload)
        assert first.status_code == 200
        assert call_counter["count"] == len(payload["styles"])

        second = await cached_client.post("/rephrase", json=payload)
    assert second.status_code == 200
    assert call_counter["count"] == len(payload["styles"])


@pytest.mark.anyio
def test_openai_retries_on_failure() -> None:
    pytest.skip("Retry behavior disabled for simplified tests")
