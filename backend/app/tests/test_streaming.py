from __future__ import annotations

import asyncio
import json

import pytest
import httpx

from app.main import app, get_llm_client


@pytest.fixture(autouse=True)
def _reset_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


class SequencedStreamClient:
    async def rephrase(self, text: str, style: str) -> str:
        return "".join(self._chunks(style))

    async def rephrase_stream(self, text: str, style: str):
        for chunk in self._chunks(style):
            await asyncio.sleep(0)
            yield chunk

    @staticmethod
    def _chunks(style: str) -> list[str]:
        return [f"{style}-part-1 ", f"{style}-part-2"]


@pytest.mark.asyncio
async def test_stream_endpoint_emits_chunks_in_order():
    app.dependency_overrides[get_llm_client] = lambda: SequencedStreamClient()

    payload = {"text": "ignored", "styles": ["professional", "casual"]}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/rephrase/stream", json=payload) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

            data_events = []
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_events.append(json.loads(line[6:]))

    assert data_events[:-1] == [
        {"style": "professional", "delta": "professional-part-1 ", "done": False},
        {"style": "professional", "delta": "professional-part-2", "done": False},
        {"style": "professional", "delta": "", "done": True},
        {"style": "casual", "delta": "casual-part-1 ", "done": False},
        {"style": "casual", "delta": "casual-part-2", "done": False},
        {"style": "casual", "delta": "", "done": True},
    ]
    assert data_events[-1] == {"done": True}


class CancellingStreamClient:
    def __init__(self) -> None:
        self.cancelled = asyncio.Event()

    async def rephrase(self, text: str, style: str) -> str:
        return ""

    async def rephrase_stream(self, text: str, style: str):
        try:
            while True:
                await asyncio.sleep(0)
                yield "streaming"
        finally:
            self.cancelled.set()


@pytest.mark.asyncio
async def test_stream_disconnect_cancels_provider():
    client_instance = CancellingStreamClient()
    app.dependency_overrides[get_llm_client] = lambda: client_instance

    payload = {"text": "ignored", "styles": ["professional"]}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        stream_ctx = client.stream("POST", "/rephrase/stream", json=payload)
        async with stream_ctx as response:
            line_stream = response.aiter_lines()
            async for line in line_stream:
                if line.startswith("data: "):
                    break
            # Exiting the context manager simulates the client disconnect.

    await asyncio.wait_for(client_instance.cancelled.wait(), timeout=1)
