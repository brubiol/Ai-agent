from __future__ import annotations

import asyncio
from typing import AsyncIterator, Protocol, runtime_checkable


class ProviderError(Exception):
    """Base exception for provider-related failures."""


class ProviderConfigError(ProviderError):
    """Raised when a provider is configured incorrectly."""


class ProviderTimeoutError(ProviderError):
    """Raised when the upstream provider request times out."""


class RateLimitError(ProviderError):
    """Raised when the upstream provider enforces a rate limit."""


class AuthenticationError(ProviderError):
    """Raised when credentials are missing or rejected."""


@runtime_checkable
class LLMClient(Protocol):
    """Protocol describing the rephrase capability required by the app."""

    async def rephrase(self, text: str, style: str) -> str:
        """Rephrase *text* following the requested *style*."""

    async def rephrase_stream(self, text: str, style: str) -> AsyncIterator[str]:
        """Stream chunks of rewritten text as they are produced."""


def chunk_text(text: str, *, size: int = 32) -> list[str]:
    """Return human-friendly chunks so streaming feels smoother."""
    if not text:
        return [""]

    words = text.split()
    if not words:
        return [text]

    chunks: list[str] = []
    buffer = ""

    for word in words:
        if len(word) > size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            for i in range(0, len(word), size):
                chunks.append(word[i : i + size])
            continue

        candidate = f"{buffer} {word}".strip()
        if buffer and len(candidate) > size:
            chunks.append(buffer)
            buffer = word
        else:
            buffer = candidate

    if buffer:
        chunks.append(buffer)

    return chunks


class MockLLMClient:
    """Fallback provider that fabricates responses without external calls."""

    async def rephrase(self, text: str, style: str) -> str:
        # We keep the format deterministic so tests and clients can rely on it.
        style_label = style.replace("_", " ").capitalize()
        return f"[{style_label}] {text}"

    async def rephrase_stream(self, text: str, style: str) -> AsyncIterator[str]:
        rewritten = await self.rephrase(text, style)
        for chunk in chunk_text(rewritten):
            await asyncio.sleep(0.05)
            yield chunk


__all__ = [
    "AuthenticationError",
    "LLMClient",
    "MockLLMClient",
    "ProviderConfigError",
    "ProviderError",
    "ProviderTimeoutError",
    "RateLimitError",
    "chunk_text",
]
