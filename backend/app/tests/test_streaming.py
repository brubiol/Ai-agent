from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.main import app, get_llm_client, get_settings


class SequencedStreamClient:
    __provider_name__ = "mock"

    async def rephrase(self, text: str, style: str) -> str:
        return "".join(self._chunks(style))

    async def rephrase_stream(self, text: str, style: str):
        for chunk in self._chunks(style):
            await asyncio.sleep(0)
            yield chunk

    @staticmethod
    def _chunks(style: str) -> list[str]:
        return [f"{style}-part-1 ", f"{style}-part-2"]


class CancellingStreamClient:
    __provider_name__ = "mock"

    def __init__(self) -> None:
        self.cancelled = asyncio.Event()

    async def rephrase(self, text: str, style: str) -> str:
        return ""

    async def rephrase_stream(self, text: str, style: str):
        try:
            while True:
                await asyncio.sleep(0)
                yield "chunk"
        finally:
            self.cancelled.set()


@pytest.fixture(autouse=True)
def _reset_streaming(monkeypatch: pytest.MonkeyPatch):
    app.dependency_overrides.clear()
    monkeypatch.setenv("DEFAULT_PROVIDER", "mock")
    get_settings.cache_clear()
    yield
    app.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_stream_emits_done_and_chunks() -> None:
    app.dependency_overrides[get_llm_client] = lambda: SequencedStreamClient()
    payload = {"text": "ignored", "styles": ["professional", "casual"]}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/rephrase/stream", json=payload) as response:
            lines = [line async for line in response.aiter_lines() if line.startswith("data: ")]

    assert json.loads(lines[-1][6:]) == {"done": True}
    events = [json.loads(line[6:]) for line in lines[:-1]]
    for style in payload["styles"]:
        per_style = [event for event in events if event["style"] == style]
        assert per_style == [
            {"style": style, "delta": f"{style}-part-1 ", "done": False},
            {"style": style, "delta": f"{style}-part-2", "done": False},
            {"style": style, "delta": "", "done": True},
        ]


@pytest.mark.asyncio
async def test_stream_disconnect_triggers_cancel() -> None:
    client_instance = CancellingStreamClient()
    app.dependency_overrides[get_llm_client] = lambda: client_instance
    payload = {"text": "ignored", "styles": ["professional"]}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        stream_ctx = client.stream("POST", "/rephrase/stream", json=payload)
        async with stream_ctx as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    break

    await asyncio.wait_for(client_instance.cancelled.wait(), timeout=1)
