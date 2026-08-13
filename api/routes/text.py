"""Text analysis and neutralization endpoints."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_history_store

router = APIRouter(prefix="/text", tags=["text"])


class AnalyzeRequest(BaseModel):
    text: str
    scheme: str | None = None
    gamma: float | None = None
    key: str | int | None = None
    tokenizer_name: str | None = None
    window: int | None = None
    threshold: float | None = None
    preset: str | None = None
    density_window: int = Field(default=20, ge=3, le=200)
    save_history: bool = True


class NeutralizeRequest(BaseModel):
    text: str
    style: Literal["subtle", "strong"] = "subtle"
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    save_history: bool = True


class TargetedRequest(BaseModel):
    text: str
    scheme: str | None = None
    gamma: float | None = None
    key: str | int | None = None
    tokenizer_name: str | None = None
    preset: str | None = None
    save_history: bool = True


class AdaptiveRequest(BaseModel):
    text: str
    style: Literal["subtle", "strong"] = "subtle"
    max_rounds: int = Field(default=3, ge=1, le=8)
    target_z: float = 4.0
    scheme: str | None = None
    gamma: float | None = None
    key: str | int | None = None
    tokenizer_name: str | None = None
    preset: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    save_history: bool = True


class BatchRequest(BaseModel):
    items: list[dict[str, str]]  # [{name, text}, ...]
    scheme: str | None = None
    gamma: float | None = None
    key: str | int | None = None
    tokenizer_name: str | None = None
    preset: str | None = None
    threshold: float | None = None


class SettingsUpdate(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    inpaint_model: str | None = None
    inpaint_base_url: str | None = None
    provider: str | None = None
    temperature: float | None = None
    clear_key: bool = False


@router.get("/presets")
def list_presets() -> dict[str, Any]:
    from watermark_core.schemes import PRESETS

    return {"presets": PRESETS}


@router.get("/tokenizers")
def list_tokenizers() -> dict[str, Any]:
    from watermark_core.tokenizer import AVAILABLE_TOKENIZERS

    return {"tokenizers": AVAILABLE_TOKENIZERS}


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    from watermark_core.settings import PROVIDER_PRESETS, load_settings, settings_path

    s = load_settings()
    return {
        "settings": s.masked_dict(),
        "path": str(settings_path()),
        "providers": PROVIDER_PRESETS,
    }


@router.put("/settings")
def put_settings(body: SettingsUpdate) -> dict[str, Any]:
    from watermark_core.settings import (
        apply_provider_preset,
        clear_api_key,
        load_settings,
        save_settings,
        settings_path,
    )

    if body.clear_key:
        clear_api_key()
    if body.provider and body.provider != "custom":
        apply_provider_preset(body.provider, keep_key=True)

    s = load_settings()
    if body.api_key is not None and body.api_key.strip() and body.api_key.strip() not in (
        "***",
        "••••",
        "(unchanged)",
    ):
        s.api_key = body.api_key.strip()
    if body.base_url is not None and body.base_url.strip():
        s.base_url = body.base_url.strip()
    if body.model is not None and body.model.strip():
        s.model = body.model.strip()
    if body.inpaint_model is not None:
        s.inpaint_model = body.inpaint_model.strip()
    if body.inpaint_base_url is not None:
        s.inpaint_base_url = body.inpaint_base_url.strip()
    if body.temperature is not None:
        s.temperature = float(body.temperature)
    if body.provider is not None:
        s.provider = body.provider
    save_settings(s)
    return {"ok": True, "settings": s.masked_dict(), "path": str(settings_path())}


@router.post("/analyze")
def analyze_text(body: AnalyzeRequest) -> dict[str, Any]:
    from watermark_core.analyzer import WatermarkAnalyzer
    from watermark_core.density import density_summary, density_to_html, sliding_window_density
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

    points = sliding_window_density(
        result.tokens, window=body.density_window, gamma=result.gamma
    )
    payload = result.to_dict()
    payload["highlighted_html"] = tokens_to_html(
        result.tokens, show_highlights=True, wrap=True, include_style=True
    )
    payload["density"] = [p.to_dict() for p in points]
    payload["density_html"] = density_to_html(points)
    payload["density_summary"] = density_summary(points)

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


@router.post("/neutralize/targeted")
def neutralize_targeted_endpoint(body: TargetedRequest) -> dict[str, Any]:
    from watermark_core.analyzer import WatermarkAnalyzer
    from watermark_core.targeted import neutralize_targeted

    try:
        analyzer = WatermarkAnalyzer(
            scheme=body.scheme,
            gamma=body.gamma,
            key=body.key,
            tokenizer_name=body.tokenizer_name,
            preset=body.preset,
        )
        result = neutralize_targeted(body.text, analyzer=analyzer)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data = result.to_dict()
    if body.save_history:
        store = get_history_store()
        store.add(
            kind="text",
            title=f"Targeted: {body.text.strip().replace(chr(10), ' ')[:60]}",
            payload={
                "type": "targeted",
                "text": body.text,
                "cleaned": result.cleaned,
                "notes": result.notes,
            },
        )
    return data


@router.post("/neutralize/adaptive")
def neutralize_adaptive_endpoint(body: AdaptiveRequest) -> dict[str, Any]:
    from watermark_core.adaptive import neutralize_adaptive
    from watermark_core.analyzer import WatermarkAnalyzer
    from watermark_core.neutralize import NeutralizeConfig

    analyzer = WatermarkAnalyzer(
        scheme=body.scheme,
        gamma=body.gamma,
        key=body.key,
        tokenizer_name=body.tokenizer_name,
        threshold=body.target_z,
        preset=body.preset,
    )
    config = NeutralizeConfig.from_env(style=body.style)
    if body.api_key:
        config.api_key = body.api_key
    if body.base_url:
        config.base_url = body.base_url
    if body.model:
        config.model = body.model

    result = neutralize_adaptive(
        body.text,
        analyzer=analyzer,
        config=config,
        max_rounds=body.max_rounds,
        target_z=body.target_z,
    )
    data = result.to_dict()
    if body.save_history and result.cleaned:
        store = get_history_store()
        store.add(
            kind="text",
            title=f"Adaptive: {body.text.strip().replace(chr(10), ' ')[:60]}",
            payload={
                "type": "adaptive",
                "text": body.text,
                "cleaned": result.cleaned,
                "z_scores": result.z_scores,
            },
        )
    return data


@router.post("/batch")
def batch_analyze(body: BatchRequest) -> dict[str, Any]:
    from watermark_core.batch import analyze_batch_texts, batch_to_markdown

    pairs = [(it.get("name") or f"item-{i}", it.get("text") or "") for i, it in enumerate(body.items)]
    results = analyze_batch_texts(
        pairs,
        scheme=body.scheme,
        gamma=body.gamma,
        key=body.key,
        tokenizer_name=body.tokenizer_name,
        preset=body.preset,
        threshold=body.threshold,
    )
    return {
        "results": [r.to_dict() for r in results],
        "markdown": batch_to_markdown(results),
    }


@router.get("/history")
def text_history(limit: int = 50) -> dict[str, Any]:
    store = get_history_store()
    entries = store.list(kind="text", limit=limit)
    return {"entries": [e.to_dict() for e in entries]}
