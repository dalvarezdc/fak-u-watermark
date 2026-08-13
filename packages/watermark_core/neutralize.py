"""Text watermark neutralization via LLM paraphrase.

Default style: subtle (preserve meaning, structure, tone).
Uses OpenAI-compatible chat completions API (DeepSeek, Grok, OpenAI, …).
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Literal

import httpx

from .chunking import DEFAULT_MAX_CHUNK_CHARS, TextChunk, join_chunks, split_document

# Entire reply is one markdown fence (optional language tag). Inner text is kept as-is.
_FENCE_RE = re.compile(r"\A\s*```[^\n]*\r?\n(.*)\r?\n[ \t]*```\s*\Z", re.DOTALL)

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
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS
    chunk_concurrency: int = 3

    @classmethod
    def from_env(cls, style: ParaphraseStyle = "subtle") -> NeutralizeConfig:
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
    chunks: int = 1

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "cleaned": self.cleaned,
            "style": self.style,
            "model": self.model,
            "success": self.success,
            "error": self.error,
            "chunks": self.chunks,
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

    from .settings import resolve_chat_model

    model = resolve_chat_model(None, config.base_url, config.model)
    chunks = split_document(text, max_chars=config.max_chunk_chars)
    body_idxs = [i for i, c in enumerate(chunks) if c.kind == "body" and c.text.strip()]
    if len(body_idxs) <= 1:
        cleaned, err = await _paraphrase_one(text, config, model)
        if err:
            return NeutralizeResult(
                original=text,
                cleaned=text,
                style=config.style,
                model=model,
                success=False,
                error=err,
                chunks=1,
            )
        return NeutralizeResult(
            original=text,
            cleaned=cleaned if cleaned is not None else text,
            style=config.style,
            model=model,
            success=True,
            chunks=1,
        )

    sem = asyncio.Semaphore(max(1, int(config.chunk_concurrency or 1)))
    rewritten: dict[int, str] = {}
    errors: list[str] = []

    async def _run(idx: int, piece: str) -> None:
        async with sem:
            cleaned, err = await _paraphrase_one(piece, config, model)
            if err:
                errors.append(err)
                return
            rewritten[idx] = cleaned if cleaned is not None else piece

    await asyncio.gather(*[_run(i, chunks[i].text) for i in body_idxs])
    if not rewritten:
        return NeutralizeResult(
            original=text,
            cleaned=text,
            style=config.style,
            model=model,
            success=False,
            error=errors[0] if errors else "Model returned empty content.",
            chunks=len(body_idxs),
        )

    out: list[TextChunk] = []
    for i, chunk in enumerate(chunks):
        if i in rewritten:
            out.append(TextChunk(rewritten[i], "body"))
        else:
            out.append(chunk)
    return NeutralizeResult(
        original=text,
        cleaned=join_chunks(out),
        style=config.style,
        model=model,
        success=True,
        error=None if len(rewritten) == len(body_idxs) else (
            f"Paraphrased {len(rewritten)}/{len(body_idxs)} sections; "
            f"others kept original. {errors[0] if errors else ''}".strip()
        ),
        chunks=len(body_idxs),
    )


async def _paraphrase_one(
    text: str,
    config: NeutralizeConfig,
    model: str,
) -> tuple[str | None, str | None]:
    """Return (cleaned, error). error set on failure."""
    system = STYLE_PROMPTS.get(config.style, SUBTLE_SYSTEM_PROMPT)
    base = config.base_url.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": config.temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                return None, _http_error_detail(resp, url)
            cleaned = cleaned_from_completion(resp.json())
            if cleaned is None:
                return None, "Model returned empty content."
            return cleaned, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _unwrap_chat_content(raw: object) -> str | None:
    """Pull a string out of chat `message.content` (string or multipart list)."""
    if raw is None:
        return None
    if isinstance(raw, list):
        chunks: list[str] = []
        for part in raw:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and part.get("type", "text") in (
                    "text",
                    "output_text",
                ):
                    chunks.append(text)
        raw = "".join(chunks)
    if not isinstance(raw, str):
        return None
    return raw


def _unwrap_fence(text: str) -> str:
    """If the whole reply is one markdown fence, return the inner text unchanged."""
    match = _FENCE_RE.fullmatch(text)
    if match:
        return match.group(1)
    return text


def cleaned_from_completion(data: object) -> str | None:
    """Extract cleaned text from a chat-completions JSON body. None = unusable."""
    if not isinstance(data, dict):
        return None
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    raw = message.get("content") if isinstance(message, dict) else None
    text = _unwrap_chat_content(raw)
    if text is None:
        return None
    text = _unwrap_fence(text)
    if not text.strip():
        return None
    return text


def _http_error_detail(resp: httpx.Response, url: str) -> str:
    """Prefer the provider's error message over a bare HTTP status."""
    detail = ""
    try:
        data = resp.json()
        err = data.get("error", data)
        if isinstance(err, dict):
            detail = str(err.get("message") or err.get("msg") or err)
        elif err:
            detail = str(err)
    except Exception:  # noqa: BLE001
        detail = (resp.text or "").strip()
    detail = detail.replace("\n", " ").strip()[:400]
    suffix = f" — {detail}" if detail else ""
    return f"HTTP {resp.status_code} from {url}{suffix}"


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
