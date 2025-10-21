from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_llm_client, get_settings
from app.providers.base import RateLimitError


class DummyClient:
    async def rephrase(self, text: str, style: str) -> str:
        return f"{style}:{text}"

    async def rephrase_stream(self, text: str, style: str):
        yield f"{style}:{text}"


class RateLimitedClient:
    async def rephrase(self, text: str, style: str) -> str:
        raise RateLimitError("Too many requests")

    async def rephrase_stream(self, text: str, style: str):
        raise RateLimitError("Too many requests")


@pytest.fixture(autouse=True)
def _reset_overrides():
    # Make sure each test starts with a clean dependency graph.
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rephrase_returns_styles(client: TestClient):
    # Inject a stub that returns predictable outputs so we can assert the payload.
    app.dependency_overrides[get_llm_client] = lambda: DummyClient()

    payload = {"text": "Hello there", "styles": ["professional", "casual"]}
    response = client.post("/rephrase", json=payload)

    assert response.status_code == 200
    assert response.json()["results"] == {
        "professional": "professional:Hello there",
        "casual": "casual:Hello there",
    }


def test_rephrase_maps_rate_limit_error(client: TestClient):
    # Simulate a 429 from the provider and ensure FastAPI response mirrors it.
    app.dependency_overrides[get_llm_client] = lambda: RateLimitedClient()

    response = client.post("/rephrase", json={"text": "Hi", "styles": ["social"]})

    assert response.status_code == 429
    assert response.json()["detail"] == "Upstream provider rate limited the request"


def test_rephrase_uses_mock_when_no_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()

    with TestClient(app) as client:
        response = client.post("/rephrase", json={"text": "Hi", "styles": ["polite"]})

    assert response.status_code == 200
    assert response.json()["results"]["polite"].startswith("[Polite]")
