from __future__ import annotations

import pytest

from app.providers.base import MockLLMClient, ProviderConfigError
from app.providers.openai_provider import OpenAIProvider


@pytest.mark.asyncio
async def test_mock_llm_client_marks_style():
    client = MockLLMClient()
    # The mock should keep the source text and clearly mark the style.
    result = await client.rephrase("Sample text", "professional")
    assert "Sample text" in result
    assert result.startswith("[Professional]")


def test_openai_provider_requires_key():
    with pytest.raises(ProviderConfigError):
        OpenAIProvider(api_key="")
