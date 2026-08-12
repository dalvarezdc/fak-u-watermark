"""Text analysis and neutralization endpoints."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_history_store

router = APIRouter(prefix="/text", tags=["text"])


class AnalyzeRequest(BaseModel):
    text: str
    scheme: str = "kgw"
    gamma: float = Field(default=0.25, gt=0.0, lt=1.0)
    key: str | int | None = None
    tokenizer_name: str = "gpt2"
    window: int = Field(default=1, ge=1)
    threshold: float = 4.0
    preset: str | None = None
    save_history: bool = True


class NeutralizeRequest(BaseModel):
    text: str
    style: Literal["subtle", "strong"] = "subtle"
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    save_history: bool = True


@router.get("/presets")
def list_presets() -> dict[str, Any]:
    from watermark_core.schemes import PRESETS

    return {"presets": PRESETS}


@router.get("/tokenizers")
def list_tokenizers() -> dict[str, Any]:
    from watermark_core.tokenizer import AVAILABLE_TOKENIZERS

    return {"tokenizers": AVAILABLE_TOKENIZERS}


@router.post("/analyze")
def analyze_text(body: AnalyzeRequest) -> dict[str, Any]:
    from watermark_core.analyzer import WatermarkAnalyzer
    from watermark_core.visualization import tokens_to_html

    try:
        analyzer = WatermarkAnalyzer(
            scheme=body.scheme,
            gamma=body.gamma,
            key=body.key,
            tokenizer_name=body.tokenizer_name,
            window=body.window,
            threshold=body.threshold,
            preset=body.preset,
        )
        result = analyzer.analyze(body.text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = result.to_dict()
    payload["highlighted_html"] = tokens_to_html(
        result.tokens, show_highlights=True, wrap=True, include_style=True
    )

    if body.save_history and body.text.strip():
        store = get_history_store()
        title = body.text.strip().replace("\n", " ")[:80]
        store.add(
            kind="text",
            title=f"Analyze: {title}",
            payload={
                "type": "analyze",
                "text": body.text,
                "statistics": result.statistics.to_dict(),
                "scheme": result.scheme,
                "tokenizer_name": result.tokenizer_name,
            },
        )

    return payload


@router.post("/neutralize")
async def neutralize_text(body: NeutralizeRequest) -> dict[str, Any]:
    from watermark_core.neutralize import NeutralizeConfig, neutralize_async

    config = NeutralizeConfig.from_env(style=body.style)
    if body.api_key:
        config.api_key = body.api_key
    if body.base_url:
        config.base_url = body.base_url
    if body.model:
        config.model = body.model

    result = await neutralize_async(body.text, config)
    if not result.success and result.error and "API key" in (result.error or ""):
        raise HTTPException(status_code=400, detail=result.error)

    data = result.to_dict()
    if body.save_history and result.success:
        store = get_history_store()
        title = body.text.strip().replace("\n", " ")[:80]
        store.add(
            kind="text",
            title=f"Neutralize: {title}",
            payload={
                "type": "neutralize",
                "original": result.original,
                "cleaned": result.cleaned,
                "style": result.style,
                "model": result.model,
            },
        )
    return data


@router.get("/history")
def text_history(limit: int = 50) -> dict[str, Any]:
    store = get_history_store()
    entries = store.list(kind="text", limit=limit)
    return {"entries": [e.to_dict() for e in entries]}
