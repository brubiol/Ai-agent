from __future__ import annotations

from typing import Any, AsyncIterator, Dict

import httpx

import asyncio
import random

from app.prompts import system_prompt_for_style
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

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        timeout: float = 10.0,
        temperature: float = 0.7,
        max_tokens: int = 256,
        max_retries: int = 3,
        retry_base_delay: float = 0.2,
        retry_jitter: float = 0.1,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ProviderConfigError("OPENAI_API_KEY is required for OpenAIProvider")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_jitter = retry_jitter

    async def rephrase(self, text: str, style: str) -> str:
        payload = self._build_payload(text=text, style=style)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        delay = self.retry_base_delay
        attempt = 0
        last_error: Exception | None = None
        while attempt < self.max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self._API_URL, json=payload, headers=headers)
                    response.raise_for_status()
                data = response.json()
                return self._extract_text(data)
            except httpx.TimeoutException as exc:
                err = ProviderTimeoutError("OpenAI request timed out")
                err.__cause__ = exc
                last_error = err
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 429:
                    err = RateLimitError("OpenAI rate limit encountered")
                    err.__cause__ = exc
                    last_error = err
                elif status in {401, 403}:
                    raise AuthenticationError("OpenAI rejected the API key") from exc
                elif status >= 500:
                    err = ProviderError(f"OpenAI returned an unexpected status {status}")
                    err.__cause__ = exc
                    last_error = err
                else:
                    raise ProviderError(f"OpenAI returned an unexpected status {status}") from exc
            except httpx.RequestError as exc:
                err = ProviderError("OpenAI request failed")
                err.__cause__ = exc
                last_error = err

            attempt += 1
            if attempt >= self.max_retries:
                break
            await asyncio.sleep(delay + random.uniform(0, self.retry_jitter))
            delay *= 2

        assert last_error is not None
        raise last_error

    async def rephrase_stream(self, text: str, style: str) -> AsyncIterator[str]:
        rewritten = await self.rephrase(text, style)
        for chunk in chunk_text(rewritten):
            await asyncio.sleep(0.05)
            yield chunk

    def _build_payload(self, *, text: str, style: str) -> Dict[str, Any]:
        system_prompt = system_prompt_for_style(style)
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
