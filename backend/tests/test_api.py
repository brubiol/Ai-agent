from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app, get_settings


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch: pytest.MonkeyPatch):
    for key in ("AUTH_BEARER_TOKEN", "ALLOWED_ORIGINS"):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.anyio
async def test_rephrase_returns_expected_styles(client: AsyncClient) -> None:
    payload = {"text": "Hello world", "styles": ["professional", "casual"]}
    response = await client.post("/rephrase", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body["results"]) == {"professional", "casual"}
    assert "[Professional]" in body["results"]["professional"]
    assert "[Casual]" in body["results"]["casual"]


@pytest.mark.anyio
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
    async for line in response.aiter_lines():
        if line:
            lines.append(line)
        if line.startswith("event: done"):
            break
    return lines


@pytest.mark.anyio
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
    assert parsed[-1] == {"done": True}


@pytest.mark.anyio
async def test_stream_cancellation_triggers_cleanup(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async_mock = AsyncMock()
    monkeypatch.setattr("app.task_manager.task_manager.cancel_all", async_mock)

    payload = {"text": "Cancellation check", "styles": ["professional"]}
    async with client.stream("POST", "/rephrase/stream", json=payload) as response:
        agen = response.aiter_lines()
        await agen.__anext__()  # first chunk event
        await response.aclose()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(agen.__anext__(), timeout=0.2)

    async_mock.assert_awaited()


@pytest.mark.anyio
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
