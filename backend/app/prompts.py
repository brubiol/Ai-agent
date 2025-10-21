from __future__ import annotations

from typing import Dict

BASE_PROMPT = (
    "You rewrite user supplied text while preserving its factual meaning. "
    "Return plain text only—no lists, no markdown, no explanations."
)

STYLE_TEMPLATES: Dict[str, str] = {
    "professional": (
        "Adopt a polished, confident business tone. "
        "Use concise sentences, precise vocabulary, and avoid slang or emoji."
    ),
    "casual": (
        "Use a relaxed, conversational tone with friendly phrasing and natural contractions. "
        "Keep sentences short-to-medium and approachable."
    ),
    "polite": (
        "Sound warm, respectful, and considerate. "
        "Include courteous language such as please and thank you when appropriate, and avoid forceful wording."
    ),
    "social": (
        "Write a lively social media style post. "
        "Keep it under 200 characters if possible, allow up to two light emoji, and stay upbeat while remaining clear."
    ),
}


def system_prompt_for_style(style: str) -> str:
    template = STYLE_TEMPLATES.get(style.lower(), STYLE_TEMPLATES["professional"])
    return f"{BASE_PROMPT} {template}"


__all__ = ["system_prompt_for_style"]
