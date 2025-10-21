from __future__ import annotations

from typing import Any, AsyncIterator, Dict

import httpx

import asyncio

from app.providers.base import (
    AuthenticationError,
    ProviderConfigError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    chunk_text,
)


class OpenAIProvider:
    """LLM client that uses OpenAI's Chat Completions API."""

    _API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str, *, model: str = "gpt-3.5-turbo", timeout: float = 10.0) -> None:
        if not api_key or not api_key.strip():
            raise ProviderConfigError("OPENAI_API_KEY is required for OpenAIProvider")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def rephrase(self, text: str, style: str) -> str:
        payload = self._build_payload(text=text, style=style)
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # We rely on httpx to raise for non-2xx so we can normalize the errors below.
                response = await client.post(self._API_URL, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("OpenAI request timed out") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                raise RateLimitError("OpenAI rate limit encountered") from exc
            if status in {401, 403}:
                raise AuthenticationError("OpenAI rejected the API key") from exc
            raise ProviderError(f"OpenAI returned an unexpected status {status}") from exc
        except httpx.RequestError as exc:
            raise ProviderError("OpenAI request failed") from exc

        data = response.json()
        return self._extract_text(data)

    async def rephrase_stream(self, text: str, style: str) -> AsyncIterator[str]:
        rewritten = await self.rephrase(text, style)
        for chunk in chunk_text(rewritten):
            await asyncio.sleep(0)
            yield chunk

    def _build_payload(self, *, text: str, style: str) -> Dict[str, Any]:
        instructions = (
            "You rephrase content. Respond with the rephrased text only, "
            "no explanations or markdown. "
            f"Rewrite the text using a {style} tone."
        )
        return {
            "model": self.model,
            "messages": [
                # System prompt keeps the output format consistent regardless of input style.
                {"role": "system", "content": instructions},
                {"role": "user", "content": text},
            ],
            "temperature": 0.7,
        }

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = message["content"]
            if isinstance(content, list):
                # Newer responses may return structured content.
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if not isinstance(content, str):
                raise TypeError
            return content.strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Unexpected response format from OpenAI") from exc


__all__ = ["OpenAIProvider"]
