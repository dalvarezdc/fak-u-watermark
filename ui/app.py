"""Gradio UI — text + images + settings.

Identify → Show → Remove.
"""

from __future__ import annotations

import base64
import html
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGES = _ROOT / "packages"
for p in (_PACKAGES, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gradio as gr
from image_tools.c2pa_tools import detect_c2pa, strip_c2pa
from image_tools.exif import read_exif, strip_exif, update_exif
from image_tools.inpaint import extract_mask_from_editor, inpaint_region, rectangle_mask
from PIL import Image
from watermark_core.adaptive import neutralize_adaptive
from watermark_core.analyzer import WatermarkAnalyzer
from watermark_core.batch import analyze_batch_files, batch_to_markdown
from watermark_core.density import (
    compress_density_points,
    density_summary,
    density_to_html,
    sliding_window_density,
)
from watermark_core.diffview import compare_document, compare_html
from watermark_core.history import HistoryStore
from watermark_core.neutralize import NeutralizeConfig, neutralize_sync
from watermark_core.schemes import PRESETS
from watermark_core.settings import (
    PROVIDER_PRESETS,
    apply_provider_preset,
    clear_api_key,
    load_settings,
    resolve_chat_model,
    save_settings,
    settings_path,
)
from watermark_core.targeted import neutralize_targeted
from watermark_core.tokenizer import AVAILABLE_TOKENIZERS, tokenizer_error_message

SOFT_YELLOW = "#FEF08A"
HISTORY = HistoryStore()
TMP = Path(tempfile.gettempdir()) / "faku"
TMP.mkdir(parents=True, exist_ok=True)
_MAX_HISTORY_IMAGE_BYTES = 1_500_000

CUSTOM_CSS = f"""
/* ── App chrome ─────────────────────────────────────────── */
.gradio-container {{
  max-width: 1320px !important;
  margin: 0 auto !important;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif !important;
}}
.faku-hero {{
  background: linear-gradient(135deg, #fffbeb 0%, #fafaf9 55%, #f5f5f4 100%);
  border: 1px solid #e7e5e4;
  border-radius: 16px;
  padding: 1.25rem 1.5rem 1rem;
  margin-bottom: 0.75rem;
}}
.faku-hero h1 {{
  margin: 0 0 0.35rem 0 !important;
  font-size: 1.65rem !important;
  font-weight: 750 !important;
  letter-spacing: -0.02em;
  color: #1c1917 !important;
}}
.faku-hero p {{
  margin: 0.25rem 0 0 !important;
  color: #57534e !important;
  font-size: 0.98rem !important;
  line-height: 1.5 !important;
}}
.faku-steps {{
  display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.85rem;
}}
.faku-step {{
  background: #fff; border: 1px solid #e7e5e4; color: #44403c;
  border-radius: 999px; padding: 0.28rem 0.75rem; font-size: 0.82rem; font-weight: 600;
}}
.faku-step em {{ font-style: normal; color: #b45309; }}

/* ── Reading panes (long-form friendly) ─────────────────── */
.watermark-signal {{
  background-color: {SOFT_YELLOW} !important;
  border-radius: 3px !important;
  padding: 0.05em 0.12em !important;
}}
.watermark-output, .density-map {{
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif !important;
  font-size: 1.05rem !important;
  line-height: 1.75 !important;
  color: #1c1917 !important;
  white-space: pre-wrap !important;
  word-break: break-word !important;
  padding: 1.25rem 1.5rem !important;
  border: 1px solid #e7e5e4 !important;
  border-radius: 12px !important;
  background: #fffefb !important;
  max-height: min(28rem, 60vh) !important;
  overflow: auto !important;
}}
.faku-section-label {{
  font-size: 0.78rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  color: #78716c !important;
  margin: 1rem 0 0.4rem !important;
}}

/* ── Verdict + stats cards ──────────────────────────────── */
.verdict-banner {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1.25rem;
  padding: 0.9rem 1.15rem; border-radius: 12px; margin: 0.25rem 0 0.75rem;
  border: 1px solid transparent;
}}
.verdict-banner.detected {{ background: #fef2f2; border-color: #fecaca; }}
.verdict-banner.uncertain {{ background: #fffbeb; border-color: #fde68a; }}
.verdict-banner.none {{ background: #f5f5f4; border-color: #e7e5e4; }}
.verdict-pill {{
  font-weight: 750; font-size: 0.95rem; padding: 0.35rem 0.85rem;
  border-radius: 999px; display: inline-block;
}}
.verdict-detected .verdict-pill, .verdict-banner.detected .verdict-pill {{
  background: #fecaca; color: #7f1d1d;
}}
.verdict-uncertain .verdict-pill, .verdict-banner.uncertain .verdict-pill {{
  background: #fde68a; color: #78350f;
}}
.verdict-none .verdict-pill, .verdict-banner.none .verdict-pill {{
  background: #e7e5e4; color: #44403c;
}}
.verdict-meta {{ color: #57534e; font-size: 0.9rem; line-height: 1.4; }}
.stats-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(7.5rem, 1fr));
  gap: 0.65rem;
  margin: 0.5rem 0 1rem;
}}
.stat-card {{
  background: #fafaf9; border: 1px solid #e7e5e4; border-radius: 12px;
  padding: 0.75rem 0.85rem;
}}
.stat-card .label {{
  font-size: 0.72rem; font-weight: 650; text-transform: uppercase;
  letter-spacing: 0.04em; color: #78716c; margin-bottom: 0.2rem;
}}
.stat-card .value {{
  font-size: 1.2rem; font-weight: 750; color: #1c1917; letter-spacing: -0.02em;
}}
.stat-card .hint {{ font-size: 0.75rem; color: #a8a29e; margin-top: 0.15rem; }}

/* ── Compare editors ────────────────────────────────────── */
.faku-editor-hint {{
  font-size: 0.88rem !important; color: #57534e !important; line-height: 1.45 !important;
}}

/* ── Compact toolbars ───────────────────────────────────── */
#text-toolbar, #export-toolbar, #settings-toolbar, #image-toolbar {{
  gap: 0.4rem !important;
  align-items: center !important;
}}
#text-toolbar > *, #export-toolbar > *, #settings-toolbar > *, #image-toolbar > * {{
  flex: 0 0 auto !important;
  min-width: 0 !important;
  width: auto !important;
}}
#text-toolbar button, #export-toolbar button,
#settings-toolbar button, #image-toolbar button {{
  width: auto !important;
  min-width: 0 !important;
  padding: 0.28rem 0.7rem !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
}}
#text-toolbar label, #export-toolbar label {{
  margin-bottom: 0 !important;
}}
#text-toolbar .block, #export-toolbar .block,
#settings-toolbar .block, #image-toolbar .block {{
  padding: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}}

/* ── Status toast ───────────────────────────────────────── */
.faku-status {{
  background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534;
  border-radius: 10px; padding: 0.65rem 0.9rem; font-size: 0.92rem;
}}

/* ── Soften Gradio density ──────────────────────────────── */
footer {{ display: none !important; }}
"""


def _make_analyzer(preset, scheme, gamma, key, tokenizer, threshold) -> WatermarkAnalyzer:
    """Score with the visible fields. Preset only fills those fields in the UI."""
    del preset
    return WatermarkAnalyzer(
        scheme=scheme,
        gamma=float(gamma),
        key=key.strip() if key else None,
        tokenizer_name=tokenizer,
        threshold=float(threshold),
    )


def apply_text_preset(preset):
    """Copy preset scheme / gamma / key into the editable fields."""
    if not preset or preset == "(none)":
        return gr.update(), gr.update(), gr.update()
    p = PRESETS.get(preset) or {}
    hash_key = p.get("hash_key", "")
    return (
        p.get("scheme", "kgw"),
        float(p.get("gamma", 0.25)),
        "" if hash_key is None else str(hash_key),
    )


def _analyze_pack(text: str, analyzer: WatermarkAnalyzer, density_window: int) -> dict | None:
    if not text or not str(text).strip():
        return None
    result = analyzer.analyze(text)
    points = compress_density_points(
        sliding_window_density(
            result.tokens, window=int(density_window or 20), gamma=result.gamma
        )
    )
    return {
        "text": text,
        "tokens": [t.to_dict() for t in result.tokens if t.is_signal],
        "statistics": result.statistics.to_dict(),
        "density": [p.to_dict() for p in points],
        "gamma": result.gamma,
    }


def _tokens_from_pack(pack: dict | None):
    if not pack or "tokens" not in pack:
        return None
    from watermark_core.schemes.base import TokenInfo

    return [TokenInfo(**t) for t in pack["tokens"]]


def _current_pack(pack: dict | None, text: str) -> dict | None:
    if not pack:
        return None
    if (pack.get("text") or "") != (text or ""):
        return None
    return pack


def _render_compare(
    left: dict | None,
    right: dict | None,
    old_text: str,
    new_text: str,
    show_highlights: bool,
) -> str:
    old_text = old_text or ""
    new_text = new_text or ""
    left_ok = _current_pack(left, old_text)
    right_ok = _current_pack(right, new_text)
    stale = (left is not None and left_ok is None and bool(old_text.strip())) or (
        right is not None and right_ok is None and bool(new_text.strip())
    )
    return compare_html(
        old_text,
        new_text,
        old_tokens=_tokens_from_pack(left_ok),
        new_tokens=_tokens_from_pack(right_ok),
        old_stats=(left_ok or {}).get("statistics"),
        new_stats=(right_ok or {}).get("statistics"),
        show_highlights=bool(show_highlights),
        old_analyzed=left_ok is not None,
        new_analyzed=right_ok is not None,
        preview_unanalyzed=stale,
    )


def _heatmaps_html(left: dict | None, right: dict | None) -> tuple[str, str]:
    from watermark_core.density import DensityPoint

    parts: list[str] = []
    metas: list[str] = []
    for label, pack in (("Original", left), ("Cleaned", right)):
        if not pack or not pack.get("density"):
            continue
        pts = [DensityPoint(**p) for p in pack["density"]]
        parts.append(
            f"<h4 style='margin:0.75rem 0 0.35rem;color:#57534e;font-size:0.85rem'>"
            f"{html.escape(label)}</h4>"
            + density_to_html(pts)
        )
        dsum = density_summary(pts)
        metas.append(
            f"{label}: mean={dsum['mean_fraction']:.1%} · max={dsum['max_fraction']:.1%} · "
            f"hot (z≥4): {dsum['hot_spans']}"
        )
    if not parts:
        return "", "—"
    return "".join(parts), " · ".join(metas)


def _score_note(left: dict | None, right: dict | None) -> str:
    bits: list[str] = []
    if left and left.get("statistics"):
        s = left["statistics"]
        bits.append(f"old z={s.get('z_score', 0):.2f} ({s.get('verdict_label', '')})")
    if right and right.get("statistics"):
        s = right["statistics"]
        bits.append(f"new z={s.get('z_score', 0):.2f} ({s.get('verdict_label', '')})")
    return " · ".join(bits)


def _llm_config(style: str, api_key: str, base_url: str, model: str) -> NeutralizeConfig:
    """Merge UI fields over saved settings / env."""
    config = NeutralizeConfig.from_env(style=style)  # type: ignore[arg-type]
    if api_key and api_key.strip() and api_key.strip() not in ("***", "••••", "(saved)"):
        config.api_key = api_key.strip()
    if base_url and base_url.strip():
        config.base_url = base_url.strip()
    if model and model.strip():
        config.model = model.strip()
    return config


def _pil_to_b64_png(img: Image.Image, max_bytes: int = _MAX_HISTORY_IMAGE_BYTES) -> str | None:
    buf = io.BytesIO()
    rgb = img.convert("RGB") if img.mode not in ("RGB", "L", "RGBA") else img
    rgb.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    if len(data) > max_bytes:
        w, h = rgb.size
        small = rgb.resize((max(1, w // 2), max(1, h // 2)))
        buf = io.BytesIO()
        small.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        if len(data) > max_bytes:
            return None
    return base64.b64encode(data).decode("ascii")


def _b64_to_pil(data: str | None) -> Image.Image | None:
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(data)))
    except Exception:  # noqa: BLE001
        return None


def _save_download(img: Image.Image | None, prefix: str = "cleaned") -> str | None:
    if img is None:
        return None
    path = TMP / f"{prefix}_{int(time.time())}.png"
    out = img.convert("RGB") if img.mode not in ("RGB", "RGBA", "L") else img
    out.save(path, format="PNG")
    return str(path)


# ── Settings ───────────────────────────────────────────────────────────────


def settings_load_ui():
    s = load_settings()
    key_display = "(saved)" if s.api_key else ""
    return (
        s.provider,
        key_display,
        s.base_url,
        s.model,
        s.inpaint_model,
        f"✓ Loaded from `{settings_path()}` — key {'set' if s.api_key else 'not set'}",
    )


def settings_save_ui(provider, api_key, base_url, model, inpaint_model):
    s = load_settings()
    if provider:
        s.provider = provider
        preset = PROVIDER_PRESETS.get(provider)
        if preset and provider != "custom":
            # Only fill empty URL/model from preset if user left defaults-ish
            if base_url:
                s.base_url = base_url.strip()
            else:
                s.base_url = preset["base_url"]
            if model:
                s.model = model.strip()
            else:
                s.model = preset["model"]
            if preset.get("inpaint_model") and not inpaint_model:
                s.inpaint_model = preset["inpaint_model"]
        else:
            if base_url:
                s.base_url = base_url.strip()
            if model:
                s.model = model.strip()
    else:
        if base_url:
            s.base_url = base_url.strip()
        if model:
            s.model = model.strip()
    if api_key and api_key.strip() not in ("", "(saved)", "***", "••••"):
        s.api_key = api_key.strip()
    if inpaint_model is not None and str(inpaint_model).strip():
        s.inpaint_model = str(inpaint_model).strip()
    s.model = resolve_chat_model(s.provider, s.base_url, s.model)
    save_settings(s)
    return (
        "(saved)" if s.api_key else "",
        s.base_url,
        s.model,
        s.inpaint_model,
        f"✓ Saved to `{settings_path()}` (mode 600). Key {'stored' if s.api_key else 'empty'}.",
    )


def settings_apply_provider(provider):
    s = apply_provider_preset(provider or "custom", keep_key=True)
    return (
        "(saved)" if s.api_key else "",
        s.base_url,
        s.model,
        s.inpaint_model,
        f"✓ Applied **{provider}** preset (API key kept).",
    )


def settings_clear_key_ui():
    clear_api_key()
    return "", f"✓ API key cleared in `{settings_path()}`"


# ── Text ───────────────────────────────────────────────────────────────────


def analyze_text(
    text: str,
    cleaned: str,
    preset: str,
    scheme: str,
    gamma: float,
    key: str,
    tokenizer: str,
    threshold: float,
    show_highlights: bool,
    density_window: int,
    *,
    source_label: str = "Analyze",
):
    text = text or ""
    cleaned = cleaned or ""
    if not text.strip() and not cleaned.strip():
        return (
            compare_html("", ""),
            None,
            "",
            "—",
            "Paste text on the left (or both sides), then Analyze.",
        )

    try:
        analyzer = _make_analyzer(preset, scheme, gamma, key, tokenizer, threshold)
        left = _analyze_pack(text, analyzer, density_window)
        right = _analyze_pack(cleaned, analyzer, density_window)
    except Exception as exc:  # noqa: BLE001 — surface tokenizer / scoring errors
        return (
            compare_html(text, cleaned),
            None,
            "",
            "—",
            f"⚠️ {tokenizer_error_message(exc, tokenizer or 'gpt2')}",
        )
    config = {
        "preset": preset,
        "scheme": scheme,
        "gamma": float(gamma),
        "key": key,
        "tokenizer": tokenizer,
        "threshold": float(threshold),
        "density_window": int(density_window or 20),
    }
    state = {"left": left, "right": right, "config": config}
    preview_src = text.strip() or cleaned.strip()
    HISTORY.add(
        kind="text",
        title=f"{source_label}: {preview_src.replace(chr(10), ' ')[:60]}",
        payload={
            "type": "analyze",
            "text": text[:100_000],
            "cleaned": cleaned[:100_000],
            "original": text[:100_000],
            "statistics": (left or {}).get("statistics"),
            "right_statistics": (right or {}).get("statistics"),
            "preview": preview_src[:500],
            "config": config,
        },
    )
    heat, heat_meta = _heatmaps_html(left, right)
    scored = _score_note(left, right)
    note = f"✓ Analyzed. {scored}" if scored else "✓ Analyzed."
    if len(text) + len(cleaned) >= 8_000:
        note += " Large document — green lists are cached; Neutralize splits into sections."
    return (
        _render_compare(left, right, text, cleaned, show_highlights),
        state,
        heat,
        heat_meta,
        note,
    )


def toggle_highlights(text, cleaned, show: bool, state: dict | None):
    left = (state or {}).get("left") if state else None
    right = (state or {}).get("right") if state else None
    return _render_compare(left, right, text or "", cleaned or "", show)


def neutralize_paraphrase(
    text,
    cleaned,
    style,
    api_key,
    base_url,
    model,
    preset,
    scheme,
    gamma,
    key,
    tokenizer,
    threshold,
    show_highlights,
    density_window,
):
    if not text or not text.strip():
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "Provide original text on the left.",
        )
    config = _llm_config(style, api_key, base_url, model)
    result = neutralize_sync(text, config)
    if not result.success:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            f"⚠️ {result.error}",
        )
    HISTORY.add(
        kind="text",
        title=f"Neutralize ({style}): {text.strip().replace(chr(10), ' ')[:50]}",
        payload={
            "type": "neutralize",
            "text": text[:100_000],
            "original": text[:100_000],
            "cleaned": result.cleaned[:100_000],
            "style": style,
            "model": result.model,
        },
    )
    compare, state, heat, heat_meta, note = analyze_text(
        text,
        result.cleaned,
        preset,
        scheme,
        gamma,
        key,
        tokenizer,
        threshold,
        show_highlights,
        density_window,
        source_label=f"Neutralize ({style})",
    )
    return (
        result.cleaned,
        compare,
        state,
        heat,
        heat_meta,
        (
            f"✓ Paraphrased with `{result.model}` ({result.style})"
            + (f", {result.chunks} sections" if result.chunks > 1 else "")
            + f". {note}"
        ),
    )


def neutralize_targeted_ui(
    text,
    cleaned,
    preset,
    scheme,
    gamma,
    key,
    tokenizer,
    threshold,
    show_highlights,
    density_window,
):
    if not text or not text.strip():
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "Provide original text on the left.",
        )
    analyzer = _make_analyzer(preset, scheme, gamma, key, tokenizer, threshold)
    result = neutralize_targeted(text, analyzer=analyzer)
    HISTORY.add(
        kind="text",
        title=f"Targeted: {text.strip().replace(chr(10), ' ')[:50]}",
        payload={
            "type": "targeted",
            "text": text[:100_000],
            "cleaned": result.cleaned[:100_000],
            "notes": result.notes,
        },
    )
    compare, state, heat, heat_meta, note = analyze_text(
        text,
        result.cleaned,
        preset,
        scheme,
        gamma,
        key,
        tokenizer,
        threshold,
        show_highlights,
        density_window,
        source_label="Targeted",
    )
    return (
        result.cleaned,
        compare,
        state,
        heat,
        heat_meta,
        f"✓ Offline targeted: {result.notes} {note}",
    )


def neutralize_adaptive_ui(
    text,
    cleaned,
    style,
    max_rounds,
    target_z,
    api_key,
    base_url,
    model,
    preset,
    scheme,
    gamma,
    key,
    tokenizer,
    threshold,
    show_highlights,
    density_window,
):
    if not text or not text.strip():
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "Provide original text on the left.",
        )
    analyzer = _make_analyzer(preset, scheme, gamma, key, tokenizer, target_z)
    config = _llm_config(style, api_key, base_url, model)
    result = neutralize_adaptive(
        text,
        analyzer=analyzer,
        config=config,
        max_rounds=int(max_rounds),
        target_z=float(target_z),
    )
    note = (
        f"{'✓' if result.success else '⚠️'} Adaptive: rounds={result.rounds} "
        f"z={result.z_scores} model=`{result.model}`"
    )
    if result.error:
        note += f" — {result.error}"
    HISTORY.add(
        kind="text",
        title=f"Adaptive: {text.strip().replace(chr(10), ' ')[:50]}",
        payload={
            "type": "adaptive",
            "text": text[:100_000],
            "cleaned": result.cleaned[:100_000],
            "z_scores": result.z_scores,
        },
    )
    compare, state, heat, heat_meta, scored = analyze_text(
        text,
        result.cleaned,
        preset,
        scheme,
        gamma,
        key,
        tokenizer,
        threshold,
        show_highlights,
        density_window,
        source_label="Adaptive",
    )
    return result.cleaned, compare, state, heat, heat_meta, f"{note} {scored}"


def copy_status(cleaned: str) -> str:
    if not cleaned:
        return "Nothing to copy."
    return f"✓ Copied {len(cleaned)} characters to clipboard."


def export_plain(cleaned: str) -> str | None:
    if not cleaned:
        return None
    path = TMP / f"cleaned_{int(time.time())}.txt"
    path.write_text(cleaned, encoding="utf-8")
    return str(path)


def export_html(state: dict | None, text: str, cleaned: str) -> str | None:
    path = TMP / f"annotated_{int(time.time())}.html"
    text = text or ""
    cleaned = cleaned or ""
    left = (state or {}).get("left") if state else None
    right = (state or {}).get("right") if state else None
    left_ok = _current_pack(left, text)
    right_ok = _current_pack(right, cleaned)
    if not text and not cleaned:
        return None
    doc = compare_document(
        text,
        cleaned,
        old_tokens=_tokens_from_pack(left_ok),
        new_tokens=_tokens_from_pack(right_ok),
        old_stats=(left_ok or {}).get("statistics"),
        new_stats=(right_ok or {}).get("statistics"),
        title="fak-u-watermark compare",
    )
    path.write_text(doc, encoding="utf-8")
    return str(path)


def refresh_text_history_dropdown():
    choices = HISTORY.choices(kind="text", limit=40)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def restore_text_history(label: str | None):
    entry = HISTORY.get_by_choice(label)
    if not entry:
        return gr.update(), gr.update(), "—", "No history entry selected.", "—"
    p = entry.payload or {}
    original = p.get("original") or p.get("text") or p.get("preview") or ""
    cleaned = p.get("cleaned") or ""
    return (
        original,
        cleaned,
        _render_compare(None, None, original, cleaned, True),
        f"✓ Restored **{entry.title}** (`{entry.id[:8]}`). Click Analyze to highlight both sides.",
        "—",
    )


def on_file_upload(file_obj):
    if file_obj is None:
        return gr.update()
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path:
        return gr.update()
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"[Error reading file: {exc}]"


def batch_analyze_ui(files, preset, scheme, gamma, key, tokenizer, threshold):
    if not files:
        return "Upload one or more .txt / .md files."
    paths = []
    for f in files:
        p = f if isinstance(f, str) else getattr(f, "name", None)
        if p:
            paths.append(p)
    preset_arg = preset if preset and preset != "(none)" else None
    try:
        results = analyze_batch_files(
            paths,
            scheme=scheme,
            gamma=float(gamma),
            key=key.strip() if key else None,
            tokenizer_name=tokenizer,
            preset=preset_arg,
            threshold=float(threshold),
        )
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ {tokenizer_error_message(exc, tokenizer or 'gpt2')}"
    return batch_to_markdown(results)


# ── Images ─────────────────────────────────────────────────────────────────


def process_exif(image: Image.Image | None):
    if image is None:
        return "{}", None, None, "Upload an image first."
    meta = read_exif(image)
    display = {k: v for k, v in meta.items() if not str(k).startswith("piexif.")}
    text = json.dumps(display, indent=2, default=str, ensure_ascii=False)
    b64 = _pil_to_b64_png(image)
    HISTORY.add(
        kind="image",
        title="EXIF inspect",
        payload={"type": "exif", "keys": list(meta.keys())[:50], "image_b64": b64},
    )
    return text, image, _save_download(image, "preview"), "✓ Metadata loaded."


def do_strip_exif(image: Image.Image | None):
    if image is None:
        return None, None, "Upload an image first."
    buf = io.BytesIO()
    fmt = image.format or "PNG"
    save_img = image
    if fmt.upper() in ("JPEG", "JPG") and save_img.mode not in ("RGB", "L"):
        save_img = save_img.convert("RGB")
    save_img.save(buf, format=fmt if fmt.upper() != "JPG" else "JPEG")
    cleaned = strip_exif(buf.getvalue())
    out = Image.open(io.BytesIO(cleaned))
    HISTORY.add(
        kind="image",
        title="Strip EXIF",
        payload={"type": "strip_exif", "image_b64": _pil_to_b64_png(out)},
    )
    return out, _save_download(out, "stripped"), "✓ Metadata stripped."


def do_update_exif(image, artist, copyright_, description):
    if image is None:
        return None, None, "Upload an image first."
    buf = io.BytesIO()
    img = image.convert("RGB") if image.mode not in ("RGB", "L") else image
    img.save(buf, format="JPEG", quality=95)
    updates = {}
    if artist.strip():
        updates["Artist"] = artist.strip()
    if copyright_.strip():
        updates["Copyright"] = copyright_.strip()
    if description.strip():
        updates["ImageDescription"] = description.strip()
    if not updates:
        return image, None, "No fields to update."
    data = update_exif(buf.getvalue(), updates)
    out = Image.open(io.BytesIO(data))
    return out, _save_download(out, "meta_updated"), f"✓ Updated: {', '.join(updates.keys())}"


def do_c2pa_detect(image: Image.Image | None):
    if image is None:
        return "Upload an image first.", None, None
    buf = io.BytesIO()
    image.save(buf, format=image.format or "PNG")
    report = detect_c2pa(buf.getvalue())
    return json.dumps(report.to_dict(), indent=2), image, None


def do_c2pa_strip(image: Image.Image | None):
    if image is None:
        return None, None, "Upload an image first.", "{}"
    buf = io.BytesIO()
    fmt = "PNG"
    image.save(buf, format=fmt)
    cleaned, report = strip_c2pa(buf.getvalue())
    out = Image.open(io.BytesIO(cleaned))
    HISTORY.add(
        kind="image",
        title="C2PA strip",
        payload={"type": "c2pa_strip", "report": report.to_dict(), "image_b64": _pil_to_b64_png(out)},
    )
    return (
        out,
        _save_download(out, "c2pa_stripped"),
        f"✓ C2PA strip attempted. present_after={report.present}",
        json.dumps(report.to_dict(), indent=2),
    )


def load_image_into_mask_editor(image: Image.Image | None):
    return image


def do_inpaint_from_editor(
    editor_value,
    fallback_image,
    x1,
    y1,
    x2,
    y2,
    use_rect,
    instruction,
    method,
    api_key,
    base_url,
    model,
):
    base_img, brush_mask = extract_mask_from_editor(editor_value)
    image = base_img or fallback_image
    if image is None:
        return None, None, "Upload an image and paint over the watermark."

    w, h = image.size
    mask = brush_mask
    brush_empty = mask is None or int(mask.sum()) == 0
    if use_rect or brush_empty:
        xa, xb, ya, yb = int(x1), int(x2), int(y1), int(y2)
        if abs(xb - xa) >= 2 and abs(yb - ya) >= 2:
            rect = rectangle_mask(w, h, xa, ya, xb, yb)
            mask = rect if brush_empty else np.maximum(mask, rect)
        elif brush_empty:
            return image, None, "Paint the watermark or enable rectangle coords."

    if mask is None or int(mask.sum()) == 0:
        return image, None, "Mask is empty."

    # Pull defaults from saved settings
    s = load_settings()
    key = api_key.strip() if api_key and api_key.strip() not in ("(saved)",) else s.api_key
    burl = base_url.strip() if base_url else (s.resolved_inpaint_base())
    mod = model.strip() if model else s.inpaint_model

    result = inpaint_region(
        image,
        mask,
        instruction=instruction or None,
        method=method,
        api_key=key or None,
        base_url=burl or None,
        model=mod or None,
    )
    if not result.success or not result.image_bytes:
        return image, None, f"⚠️ Inpaint failed: {result.error}"

    out = Image.open(io.BytesIO(result.image_bytes))
    HISTORY.add(
        kind="image",
        title=f"Inpaint ({result.method})",
        payload={
            "type": "inpaint",
            "instruction": instruction,
            "method": result.method,
            "image_b64": _pil_to_b64_png(out),
        },
    )
    note = f"✓ Region removed with **{result.method}**."
    return out, _save_download(out, "inpainted"), note


def refresh_image_history_dropdown():
    choices = HISTORY.choices(kind="image", limit=40)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def restore_image_history(label: str | None):
    entry = HISTORY.get_by_choice(label)
    if not entry:
        return None, None, "{}", "No history entry selected."
    p = entry.payload or {}
    img = _b64_to_pil(p.get("image_b64"))
    meta_txt = "{}"
    if img is not None:
        try:
            meta = read_exif(img)
            display = {k: v for k, v in meta.items() if not str(k).startswith("piexif.")}
            meta_txt = json.dumps(display, indent=2, default=str)
        except Exception:  # noqa: BLE001
            pass
    note = f"✓ Restored **{entry.title}** (`{entry.id[:8]}`)"
    if img is None:
        note += " — no image stored for this entry."
    return img, img, meta_txt, note


# ── Build UI ───────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    preset_choices = ["(none)"] + list(PRESETS.keys())
    tokenizer_choices = list(AVAILABLE_TOKENIZERS.keys())
    provider_choices = list(PROVIDER_PRESETS.keys())

    with gr.Blocks(title="fak-u-watermark") as demo:
        gr.HTML(
            """
<div class="faku-hero">
  <h1>fak-u-watermark</h1>
  <p><strong>Strip the mark. Keep the meaning.</strong>
  Paste text or upload an image — detect statistical watermarks, see yellow highlights, then neutralize.</p>
  <div class="faku-steps">
    <span class="faku-step"><em>1</em> Paste / upload</span>
    <span class="faku-step"><em>2</em> Analyze</span>
    <span class="faku-step"><em>3</em> Compare old vs new</span>
    <span class="faku-step"><em>4</em> Edit · re-analyze · export</span>
  </div>
</div>
            """
        )

        with gr.Tabs():
            # ── SETTINGS ──────────────────────────────────────────────────
            with gr.Tab("⚙️ Settings"):
                gr.Markdown(
                    f"""
### API keys
Stored **only on this machine** in `{settings_path()}` (permissions 600).  
Also works with env vars: `FAKU_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `XAI_API_KEY`.
                    """
                )
                with gr.Row():
                    set_provider = gr.Dropdown(
                        choices=provider_choices,
                        value="custom",
                        label="Provider preset",
                    )
                    btn_apply_provider = gr.Button(
                        "Apply preset URLs", size="sm", scale=0, min_width=110
                    )
                set_api_key = gr.Textbox(
                    label="API key",
                    type="password",
                    placeholder="sk-...  (paste to set; leave as (saved) to keep)",
                )
                set_base_url = gr.Textbox(
                    label="Base URL",
                    placeholder="https://api.openai.com/v1",
                )
                set_model = gr.Textbox(label="Chat model (paraphrase)", placeholder="gpt-4o-mini")
                set_inpaint_model = gr.Textbox(
                    label="Inpaint model (images/edits)",
                    placeholder="dall-e-2",
                )
                with gr.Row(elem_id="settings-toolbar"):
                    btn_save_settings = gr.Button(
                        "Save settings", variant="primary", size="sm", scale=0, min_width=80
                    )
                    btn_load_settings = gr.Button("Reload", size="sm", scale=0, min_width=60)
                    btn_clear_key = gr.Button("Clear API key", size="sm", scale=0, min_width=80)
                settings_status = gr.Markdown()

                gr.Markdown(
                    """
**Provider base URLs (examples)**

| Provider | Base URL | Example model |
|----------|----------|---------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com` | `deepseek-v4-flash` |
| xAI Grok | `https://api.x.ai/v1` | `grok-2-latest` |

CLI: `faku settings set --api-key sk-... --provider openai`
                    """
                )

                btn_load_settings.click(
                    settings_load_ui,
                    outputs=[
                        set_provider,
                        set_api_key,
                        set_base_url,
                        set_model,
                        set_inpaint_model,
                        settings_status,
                    ],
                )
                btn_save_settings.click(
                    settings_save_ui,
                    inputs=[
                        set_provider,
                        set_api_key,
                        set_base_url,
                        set_model,
                        set_inpaint_model,
                    ],
                    outputs=[
                        set_api_key,
                        set_base_url,
                        set_model,
                        set_inpaint_model,
                        settings_status,
                    ],
                )
                btn_apply_provider.click(
                    settings_apply_provider,
                    inputs=[set_provider],
                    outputs=[
                        set_api_key,
                        set_base_url,
                        set_model,
                        set_inpaint_model,
                        settings_status,
                    ],
                )
                btn_clear_key.click(
                    settings_clear_key_ui,
                    outputs=[set_api_key, settings_status],
                )
                demo.load(
                    settings_load_ui,
                    outputs=[
                        set_provider,
                        set_api_key,
                        set_base_url,
                        set_model,
                        set_inpaint_model,
                        settings_status,
                    ],
                )

            # ── TEXT ──────────────────────────────────────────────────────
            with gr.Tab("📝 Text"):
                analysis_state = gr.State(None)

                gr.Markdown(
                    "### Compare original vs cleaned\n"
                    "Left is **old**, right is **new**. Edit either side, then Analyze. "
                    "Neutralize writes a cleaned draft on the right."
                )
                with gr.Row(elem_id="text-toolbar"):
                    file_in = gr.UploadButton(
                        "Upload .txt / .md",
                        file_types=[".txt", ".md", ".text"],
                        type="filepath",
                        size="sm",
                        scale=0,
                        min_width=90,
                    )
                    btn_analyze = gr.Button(
                        "Analyze", variant="primary", size="sm", scale=0, min_width=70
                    )
                    btn_neu = gr.Button("Neutralize", size="sm", scale=0, min_width=80)
                    btn_targeted = gr.Button("Targeted", size="sm", scale=0, min_width=70)
                    btn_adaptive = gr.Button("Adaptive", size="sm", scale=0, min_width=70)
                    show_hl = gr.Checkbox(
                        value=True, label="Yellow marks", scale=0, min_width=110
                    )

                with gr.Accordion("Watermark settings (optional)", open=False):
                    gr.Markdown(
                        "Preset fills scheme, γ, and key. Override those fields afterward "
                        "(needed for Targeted with a custom key). Default is classic "
                        "Kirchenbauer (γ=0.25)."
                    )
                    with gr.Row():
                        preset = gr.Dropdown(
                            choices=preset_choices,
                            value="kirchenbauer_default",
                            label="Preset",
                        )
                        scheme = gr.Dropdown(
                            choices=["kgw", "unigram"], value="kgw", label="Scheme"
                        )
                        tokenizer = gr.Dropdown(
                            choices=tokenizer_choices, value="gpt2", label="Tokenizer"
                        )
                    with gr.Row():
                        gamma = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Gamma")
                        threshold = gr.Slider(
                            1.0, 10.0, value=4.0, step=0.5, label="Z threshold"
                        )
                        density_window = gr.Slider(
                            5, 80, value=20, step=1, label="Heatmap window"
                        )
                    key = gr.Textbox(label="Secret key / seed", value="15485863")

                with gr.Accordion("Paraphrase / API options", open=False):
                    style = gr.Radio(
                        choices=["subtle", "strong"],
                        value="subtle",
                        label="Paraphrase style (default: subtle wording only)",
                    )
                    with gr.Row():
                        api_key = gr.Textbox(
                            label="API key override",
                            type="password",
                            placeholder="Empty = use Settings tab",
                        )
                        base_url = gr.Textbox(label="Base URL override")
                        model = gr.Textbox(label="Model override")
                    with gr.Row():
                        max_rounds = gr.Slider(
                            1, 6, value=3, step=1, label="Adaptive max rounds"
                        )
                        target_z = gr.Slider(
                            1.0, 8.0, value=4.0, step=0.5, label="Adaptive target z"
                        )

                status_out = gr.Markdown()

                with gr.Row(equal_height=True):
                    with gr.Column():
                        text_in = gr.Textbox(
                            label="Original (old) · editable",
                            placeholder="Paste original text here…",
                            lines=10,
                            max_lines=22,
                            elem_id="input-text",
                        )
                    with gr.Column():
                        cleaned_out = gr.Textbox(
                            label="Cleaned (new) · editable",
                            placeholder="Neutralize, or paste a candidate to compare…",
                            lines=10,
                            max_lines=22,
                            interactive=True,
                            elem_id="cleaned-text",
                        )

                compare_out = gr.HTML(
                    value=compare_html("", ""),
                )

                with gr.Row(elem_id="export-toolbar"):
                    btn_copy = gr.Button("Copy", variant="primary", size="sm", scale=0, min_width=60)
                    btn_export_txt = gr.DownloadButton(
                        "Export .txt", size="sm", scale=0, min_width=80
                    )
                    btn_export_html = gr.DownloadButton(
                        "Export .html", size="sm", scale=0, min_width=80
                    )

                with gr.Accordion("Density heatmap (where the signal clusters)", open=False):
                    heat_meta = gr.Markdown()
                    heat_out = gr.HTML()

                with gr.Accordion("Batch · history", open=False):
                    gr.Markdown("#### Batch analyze")
                    batch_files = gr.File(
                        label="Multiple .txt / .md files",
                        file_count="multiple",
                        type="filepath",
                        height=90,
                    )
                    btn_batch = gr.Button("Run batch", size="sm", scale=0, min_width=80)
                    batch_out = gr.Markdown()
                    gr.Markdown("#### History")
                    hist_dd = gr.Dropdown(label="Past text jobs", choices=[])
                    with gr.Row():
                        btn_refresh_hist = gr.Button("Refresh", size="sm", scale=0, min_width=70)
                        btn_restore_hist = gr.Button(
                            "Restore", variant="primary", size="sm", scale=0, min_width=70
                        )
                    hist_status = gr.Markdown()

                file_in.upload(on_file_upload, inputs=[file_in], outputs=[text_in])
                preset.change(
                    apply_text_preset,
                    inputs=[preset],
                    outputs=[scheme, gamma, key],
                )

                analyze_inputs = [
                    text_in,
                    cleaned_out,
                    preset,
                    scheme,
                    gamma,
                    key,
                    tokenizer,
                    threshold,
                    show_hl,
                    density_window,
                ]
                analyze_outputs = [
                    compare_out,
                    analysis_state,
                    heat_out,
                    heat_meta,
                    status_out,
                ]
                neutralize_outputs = [
                    cleaned_out,
                    compare_out,
                    analysis_state,
                    heat_out,
                    heat_meta,
                    status_out,
                ]

                btn_analyze.click(analyze_text, inputs=analyze_inputs, outputs=analyze_outputs)
                show_hl.change(
                    toggle_highlights,
                    inputs=[text_in, cleaned_out, show_hl, analysis_state],
                    outputs=[compare_out],
                )
                btn_neu.click(
                    neutralize_paraphrase,
                    inputs=[
                        text_in,
                        cleaned_out,
                        style,
                        api_key,
                        base_url,
                        model,
                        preset,
                        scheme,
                        gamma,
                        key,
                        tokenizer,
                        threshold,
                        show_hl,
                        density_window,
                    ],
                    outputs=neutralize_outputs,
                )
                btn_targeted.click(
                    neutralize_targeted_ui,
                    inputs=[
                        text_in,
                        cleaned_out,
                        preset,
                        scheme,
                        gamma,
                        key,
                        tokenizer,
                        threshold,
                        show_hl,
                        density_window,
                    ],
                    outputs=neutralize_outputs,
                )
                btn_adaptive.click(
                    neutralize_adaptive_ui,
                    inputs=[
                        text_in,
                        cleaned_out,
                        style,
                        max_rounds,
                        target_z,
                        api_key,
                        base_url,
                        model,
                        preset,
                        scheme,
                        gamma,
                        key,
                        tokenizer,
                        threshold,
                        show_hl,
                        density_window,
                    ],
                    outputs=neutralize_outputs,
                )
                btn_copy.click(
                    fn=copy_status,
                    inputs=[cleaned_out],
                    outputs=[status_out],
                    js="(text) => { if (text) { navigator.clipboard.writeText(text); } return [text]; }",
                )
                btn_export_txt.click(
                    export_plain, inputs=[cleaned_out], outputs=[btn_export_txt]
                )
                btn_export_html.click(
                    export_html,
                    inputs=[analysis_state, text_in, cleaned_out],
                    outputs=[btn_export_html],
                )
                btn_batch.click(
                    batch_analyze_ui,
                    inputs=[batch_files, preset, scheme, gamma, key, tokenizer, threshold],
                    outputs=[batch_out],
                )
                btn_refresh_hist.click(refresh_text_history_dropdown, outputs=[hist_dd])
                btn_restore_hist.click(
                    restore_text_history,
                    inputs=[hist_dd],
                    outputs=[text_in, cleaned_out, compare_out, hist_status, status_out],
                )
                demo.load(refresh_text_history_dropdown, outputs=[hist_dd])

            # ── IMAGES ────────────────────────────────────────────────────
            with gr.Tab("🖼️ Images"):
                gr.Markdown(
                    "### Image tools\n"
                    "Inspect / strip metadata, check C2PA markers, or paint over a visible watermark."
                )
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(
                            label="Upload", type="pil", sources=["upload", "clipboard"]
                        )
                        with gr.Row(elem_id="image-toolbar"):
                            btn_exif = gr.Button(
                                "Read EXIF", variant="primary", size="sm", scale=0, min_width=70
                            )
                            btn_strip = gr.Button("Strip metadata", size="sm", scale=0, min_width=80)
                            btn_c2pa = gr.Button("Detect C2PA", size="sm", scale=0, min_width=80)
                            btn_c2pa_strip = gr.Button(
                                "Strip C2PA", size="sm", scale=0, min_width=80
                            )
                        with gr.Accordion("Edit metadata fields", open=False):
                            artist = gr.Textbox(label="Artist")
                            copyright_ = gr.Textbox(label="Copyright")
                            description = gr.Textbox(label="ImageDescription")
                            btn_update = gr.Button(
                                "Apply field updates", size="sm", scale=0, min_width=90
                            )
                    with gr.Column():
                        img_out = gr.Image(label="Result", type="pil")
                        img_download = gr.DownloadButton(
                            "Download image", size="sm", scale=0, min_width=90
                        )
                        img_status = gr.Markdown()
                        exif_out = gr.Code(label="Metadata / C2PA report", language="json", lines=12)

                gr.Markdown("### Remove a visible mark\nPaint white over the logo or watermark, then remove.")
                mask_editor = gr.ImageMask(
                    label="Brush mask",
                    type="pil",
                    layers=False,
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=24, color_mode="fixed"),
                    sources=["upload", "clipboard"],
                )
                with gr.Accordion("Rectangle fallback", open=False):
                    use_rect = gr.Checkbox(value=False, label="Use rectangle coords")
                    with gr.Row():
                        x1 = gr.Number(label="x1", value=0, precision=0)
                        y1 = gr.Number(label="y1", value=0, precision=0)
                        x2 = gr.Number(label="x2", value=100, precision=0)
                        y2 = gr.Number(label="y2", value=50, precision=0)
                instruction = gr.Textbox(
                    label="Instruction (API / auto)",
                    value="Remove the watermark or logo and fill the area naturally.",
                )
                method = gr.Radio(
                    choices=["telea", "ns", "api", "auto"],
                    value="telea",
                    label="Inpaint method",
                )
                with gr.Accordion("Inpaint API override (optional)", open=False):
                    inpaint_key = gr.Textbox(label="API key", type="password")
                    inpaint_base = gr.Textbox(label="Base URL")
                    inpaint_model = gr.Textbox(label="Model", placeholder="dall-e-2")
                btn_inpaint = gr.Button(
                    "Remove painted region", variant="primary", size="sm", scale=0, min_width=120
                )

                with gr.Accordion("Image history", open=False):
                    img_hist_dd = gr.Dropdown(label="Past image jobs", choices=[])
                    with gr.Row():
                        btn_img_hist = gr.Button("Refresh", size="sm", scale=0, min_width=70)
                        btn_img_restore = gr.Button(
                            "Restore", variant="primary", size="sm", scale=0, min_width=70
                        )
                    img_hist_status = gr.Markdown()

                img_in.change(load_image_into_mask_editor, inputs=[img_in], outputs=[mask_editor])
                btn_exif.click(
                    process_exif,
                    inputs=[img_in],
                    outputs=[exif_out, img_out, img_download, img_status],
                )
                btn_strip.click(
                    do_strip_exif, inputs=[img_in], outputs=[img_out, img_download, img_status]
                )
                btn_c2pa.click(do_c2pa_detect, inputs=[img_in], outputs=[exif_out, img_out, img_download])
                btn_c2pa_strip.click(
                    do_c2pa_strip,
                    inputs=[img_in],
                    outputs=[img_out, img_download, img_status, exif_out],
                )
                btn_update.click(
                    do_update_exif,
                    inputs=[img_in, artist, copyright_, description],
                    outputs=[img_out, img_download, img_status],
                )
                btn_inpaint.click(
                    do_inpaint_from_editor,
                    inputs=[
                        mask_editor,
                        img_in,
                        x1,
                        y1,
                        x2,
                        y2,
                        use_rect,
                        instruction,
                        method,
                        inpaint_key,
                        inpaint_base,
                        inpaint_model,
                    ],
                    outputs=[img_out, img_download, img_status],
                )
                btn_img_hist.click(refresh_image_history_dropdown, outputs=[img_hist_dd])
                btn_img_restore.click(
                    restore_image_history,
                    inputs=[img_hist_dd],
                    outputs=[img_in, img_out, exif_out, img_hist_status],
                )
                demo.load(refresh_image_history_dropdown, outputs=[img_hist_dd])

            # ── ABOUT ─────────────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown(
                    """
## How it works

1. **Tokenize** (default GPT-2) and reconstruct the green list from previous tokens + key.  
2. **Score** green fraction → z-score and a clear verdict.  
3. **Compare** original (left) vs cleaned (right) with yellow watermark marks and rose/green wording changes.  
4. **Remove** via paraphrase (API), targeted synonyms (offline), or adaptive loops.  
5. **Edit either side**, Analyze again, then copy / export.

**Images:** EXIF · C2PA markers · brush inpaint (local OpenCV or API).

**Keys:** Settings tab, `~/.faku/settings.json`, or env vars. Open the UI at **http://127.0.0.1:7860** (not `0.0.0.0`).

Full docs: `README.md`.
                    """
                )

        gr.HTML(
            "<p style='text-align:center;color:#a8a29e;font-size:0.8rem;margin-top:1.25rem'>"
            "fak-u-watermark · local history &amp; keys in ~/.faku/</p>"
        )

    return demo


def main():
    import os

    demo = build_app()
    # Bind all interfaces for LAN/Docker; open via 127.0.0.1 in the browser
    # (http://0.0.0.0:7860 shows about:blank in Safari and some other browsers).
    host = os.environ.get("FAKU_UI_HOST", "0.0.0.0")
    port = int(os.environ.get("FAKU_UI_PORT", "7860"))
    print(f"\n  fak-u-watermark UI →  http://127.0.0.1:{port}\n", flush=True)
    # Gradio 6: theme/css on launch
    theme = gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="stone",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("DM Sans"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        radius_size="lg",
        spacing_size="md",
        text_size="md",
    ).set(
        body_background_fill="#fafaf9",
        block_background_fill="#ffffff",
        block_border_color="#e7e5e4",
        block_label_text_color="#57534e",
        block_title_text_color="#1c1917",
        button_primary_background_fill="#d97706",
        button_primary_background_fill_hover="#b45309",
        button_primary_text_color="#ffffff",
    )
    demo.launch(
        server_name=host,
        server_port=port,
        css=CUSTOM_CSS,
        theme=theme,
        allowed_paths=[str(TMP)],
        inbrowser=os.environ.get("FAKU_UI_INBROWSER", "1") not in ("0", "false", "False"),
    )


if __name__ == "__main__":
    main()
