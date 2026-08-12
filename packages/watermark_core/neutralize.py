"""Text watermark neutralization via LLM paraphrase.

Default style: subtle (preserve meaning, structure, tone).
Uses OpenAI-compatible chat completions API (DeepSeek, Grok, OpenAI, …).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import httpx

ParaphraseStyle = Literal["subtle", "strong"]

SUBTLE_SYSTEM_PROMPT = """You are a careful editor. Make only subtle wording changes to the following text.
Keep the exact same meaning, facts, tone, and overall structure.
Prefer simple synonym replacements and minor phrase adjustments.
Do not restructure sentences or add/remove information unless necessary.
Return only the revised text."""

STRONG_SYSTEM_PROMPT = """You are a skilled rewriter. Paraphrase the following text thoroughly while
preserving the exact same meaning and factual content.
Vary vocabulary, sentence openings, and phrasing more aggressively, but do not add or remove information.
Return only the revised text."""

STYLE_PROMPTS: dict[str, str] = {
    "subtle": SUBTLE_SYSTEM_PROMPT,
    "strong": STRONG_SYSTEM_PROMPT,
}


@dataclass
class NeutralizeConfig:
    """OpenAI-compatible API settings."""

    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    style: ParaphraseStyle = "subtle"
    temperature: float = 0.7
    timeout: float = 120.0

    @classmethod
    def from_env(cls, style: ParaphraseStyle = "subtle") -> "NeutralizeConfig":
        """Load from ~/.faku/settings.json first, then environment variables."""
        try:
            from .settings import load_settings

            s = load_settings()
            return cls(
                api_key=s.api_key
                or os.environ.get("FAKU_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("DEEPSEEK_API_KEY")
                or os.environ.get("XAI_API_KEY"),
                base_url=s.base_url
                or os.environ.get("FAKU_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or "https://api.openai.com/v1",
                model=s.model
                or os.environ.get("FAKU_MODEL")
                or os.environ.get("OPENAI_MODEL")
                or "gpt-4o-mini",
                style=style,
                temperature=float(s.temperature or 0.7),
            )
        except Exception:  # noqa: BLE001
            return cls(
                api_key=os.environ.get("FAKU_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("DEEPSEEK_API_KEY")
                or os.environ.get("XAI_API_KEY"),
                base_url=os.environ.get("FAKU_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or "https://api.openai.com/v1",
                model=os.environ.get("FAKU_MODEL")
                or os.environ.get("OPENAI_MODEL")
                or "gpt-4o-mini",
                style=style,
            )


@dataclass
class NeutralizeResult:
    original: str
    cleaned: str
    style: str
    model: str
    success: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "cleaned": self.cleaned,
            "style": self.style,
            "model": self.model,
            "success": self.success,
            "error": self.error,
        }


async def neutralize_async(
    text: str,
    config: NeutralizeConfig | None = None,
) -> NeutralizeResult:
    """Call an OpenAI-compatible chat API to paraphrase (neutralize) text."""
    config = config or NeutralizeConfig.from_env()
    if not text.strip():
        return NeutralizeResult(
            original=text,
            cleaned=text,
            style=config.style,
            model=config.model,
            success=True,
        )
    if not config.api_key:
        return NeutralizeResult(
            original=text,
            cleaned=text,
            style=config.style,
            model=config.model,
            success=False,
            error=(
                "No API key configured. Set FAKU_API_KEY (or OPENAI_API_KEY / "
                "DEEPSEEK_API_KEY / XAI_API_KEY) and optionally FAKU_BASE_URL / FAKU_MODEL."
            ),
        )

    system = STYLE_PROMPTS.get(config.style, SUBTLE_SYSTEM_PROMPT)
    base = config.base_url.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            cleaned = data["choices"][0]["message"]["content"].strip()
            return NeutralizeResult(
                original=text,
                cleaned=cleaned,
                style=config.style,
                model=config.model,
                success=True,
            )
    except Exception as exc:  # noqa: BLE001 — surface to API/UI
        return NeutralizeResult(
            original=text,
            cleaned=text,
            style=config.style,
            model=config.model,
            success=False,
            error=str(exc),
        )


def neutralize_sync(text: str, config: NeutralizeConfig | None = None) -> NeutralizeResult:
    """Synchronous wrapper for Gradio / CLI."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Nested event loop (e.g. Jupyter / some Gradio paths)
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, neutralize_async(text, config)).result()
        return loop.run_until_complete(neutralize_async(text, config))
    except RuntimeError:
        return asyncio.run(neutralize_async(text, config))
