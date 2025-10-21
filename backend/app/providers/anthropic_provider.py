from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List

import httpx

from app.providers.base import (
    AuthenticationError,
    ProviderConfigError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    chunk_text,
)


class AnthropicProvider:
    """LLM client that uses Anthropic's Messages API."""

    _API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "claude-3-haiku-20240307",
        timeout: float = 10.0,
        max_tokens: int = 512,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ProviderConfigError("ANTHROPIC_API_KEY is required for AnthropicProvider")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    async def rephrase(self, text: str, style: str) -> str:
        payload = self._build_payload(text=text, style=style)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Align Anthropic error handling with our shared ProviderError hierarchy.
                response = await client.post(self._API_URL, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("Anthropic request timed out") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                raise RateLimitError("Anthropic rate limit encountered") from exc
            if status in {401, 403}:
                raise AuthenticationError("Anthropic rejected the API key") from exc
            raise ProviderError(f"Anthropic returned an unexpected status {status}") from exc
        except httpx.RequestError as exc:
            raise ProviderError("Anthropic request failed") from exc

        data = response.json()
        return self._extract_text(data)

    async def rephrase_stream(self, text: str, style: str) -> AsyncIterator[str]:
        rewritten = await self.rephrase(text, style)
        for chunk in chunk_text(rewritten):
            await asyncio.sleep(0)
            yield chunk

    def _build_payload(self, *, text: str, style: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a writing assistant that rewrites user-provided text. "
            "Return only the rewritten text, without explanations, "
            "rewritten in the requested tone."
        )
        user_message: List[Dict[str, str]] = [
            {
                "type": "text",
                "text": f"Rewrite the following text in a {style} tone:\n{text}",
            }
        ]
        return {
            "model": self.model,
            "system": system_prompt,
            "max_tokens": self.max_tokens,
            "messages": [
                # Anthropic expects a list of blocks; we keep to plain text for simplicity.
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
        }

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        try:
            content = data["content"]
            if not isinstance(content, list):
                raise TypeError
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            if not parts:
                raise KeyError("text")
            return "".join(parts).strip()
        except (KeyError, TypeError) as exc:
            raise ProviderError("Unexpected response format from Anthropic") from exc


__all__ = ["AnthropicProvider"]
