from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_llm_client, get_settings
from app.providers.base import RateLimitError


class DummyClient:
    __provider_name__ = "mock"

    async def rephrase(self, text: str, style: str) -> str:
        return f"{style}:{text}"

    async def rephrase_stream(self, text: str, style: str):
        yield f"{style}:{text}"



@pytest.fixture(autouse=True)
def _reset_overrides(monkeypatch: pytest.MonkeyPatch):
    app.dependency_overrides.clear()
    monkeypatch.setenv("DEFAULT_PROVIDER", "mock")
    get_settings.cache_clear()
    yield
    app.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rephrase_with_stubbed_client(client: TestClient) -> None:
    app.dependency_overrides[get_llm_client] = lambda: DummyClient()
    payload = {"text": "Hello there", "styles": ["professional", "casual"]}
    response = client.post("/rephrase", json=payload)
    assert response.status_code == 200
    assert response.json()["results"] == {
        "professional": "professional:Hello there",
        "casual": "casual:Hello there",
    }


