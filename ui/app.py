"""Gradio UI — text + images + settings.

Identify → Show → Remove.
"""

from __future__ import annotations

import base64
import html
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGES = _ROOT / "packages"
for p in (_PACKAGES, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gradio as gr
from PIL import Image

from watermark_core.adaptive import neutralize_adaptive
from watermark_core.analyzer import WatermarkAnalyzer
from watermark_core.batch import analyze_batch_files, batch_to_markdown
from watermark_core.density import density_summary, density_to_html, sliding_window_density
from watermark_core.history import HistoryStore
from watermark_core.neutralize import NeutralizeConfig, neutralize_sync
from watermark_core.schemes import PRESETS
from watermark_core.settings import (
    PROVIDER_PRESETS,
    apply_provider_preset,
    clear_api_key,
    load_settings,
    save_settings,
    settings_path,
)
from watermark_core.targeted import neutralize_targeted
from watermark_core.tokenizer import AVAILABLE_TOKENIZERS
from watermark_core.visualization import tokens_to_annotated_document, tokens_to_html

from image_tools.c2pa_tools import detect_c2pa, strip_c2pa
from image_tools.exif import read_exif, strip_exif, update_exif
from image_tools.inpaint import extract_mask_from_editor, inpaint_region, rectangle_mask

SOFT_YELLOW = "#FEF08A"
HISTORY = HistoryStore()
TMP = Path("/tmp/faku")
TMP.mkdir(parents=True, exist_ok=True)
_MAX_HISTORY_IMAGE_BYTES = 1_500_000

CUSTOM_CSS = f"""
.watermark-signal {{
  background-color: {SOFT_YELLOW} !important;
  border-radius: 2px;
  padding: 0 1px;
}}
.watermark-output {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  white-space: pre-wrap; word-break: break-word; line-height: 1.65;
  font-size: 0.95rem; padding: 0.75rem; border: 1px solid #e5e7eb;
  border-radius: 8px; background: #fafafa;
}}
.verdict-detected {{
  background: #fecaca; color: #7f1d1d; padding: 0.35rem 0.75rem;
  border-radius: 999px; font-weight: 600; display: inline-block;
}}
.verdict-uncertain {{
  background: #fde68a; color: #78350f; padding: 0.35rem 0.75rem;
  border-radius: 999px; font-weight: 600; display: inline-block;
}}
.verdict-none {{
  background: #e5e7eb; color: #374151; padding: 0.35rem 0.75rem;
  border-radius: 999px; font-weight: 600; display: inline-block;
}}
.faku-compare {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
}}
@media (max-width: 800px) {{ .faku-compare {{ grid-template-columns: 1fr; }} }}
"""


def _verdict_html(stats: dict) -> str:
    v = stats.get("verdict", "none")
    label = stats.get("verdict_label", v)
    cls = {
        "detected": "verdict-detected",
        "uncertain": "verdict-uncertain",
        "none": "verdict-none",
    }.get(v, "verdict-none")
    return f'<span class="{cls}">{html.escape(label)}</span>'


def _stats_markdown(stats: dict) -> str:
    return f"""
| Metric | Value |
|--------|------:|
| **Verdict** | {stats.get("verdict_label", "—")} |
| Green fraction | {stats.get("green_fraction", 0):.2%} |
| Green / total | {stats.get("green_count", 0)} / {stats.get("total_tokens", 0)} |
| Z-score | {stats.get("z_score", 0):.3f} |
| P-value (approx) | {stats.get("p_value", 1):.2e} |
| Gamma | {stats.get("gamma", "—")} |
| Scheme | {stats.get("scheme", "—")} |
| Threshold | {stats.get("threshold", 4.0)} |
"""


def _compare_html(original: str, cleaned: str) -> str:
    o = html.escape(original or "")
    c = html.escape(cleaned or "")
    return f"""
<div class="faku-compare">
  <div><strong>Original</strong>
    <pre style="white-space:pre-wrap;background:#fafafa;padding:0.75rem;border-radius:8px;border:1px solid #e5e7eb;max-height:20rem;overflow:auto">{o}</pre>
  </div>
  <div><strong>Cleaned</strong>
    <pre style="white-space:pre-wrap;background:#fafafa;padding:0.75rem;border-radius:8px;border:1px solid #e5e7eb;max-height:20rem;overflow:auto">{c}</pre>
  </div>
</div>
"""


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
    preset: str,
    scheme: str,
    gamma: float,
    key: str,
    tokenizer: str,
    threshold: float,
    show_highlights: bool,
    density_window: int,
    *,
    seed_cleaned: bool = True,
    source_label: str = "Analyze",
):
    if not text or not text.strip():
        empty = "<em>Paste or upload text, then click Analyze.</em>"
        cleaned_val = (text or "") if seed_cleaned else gr.update()
        return empty, "—", "", cleaned_val, None, "—", empty, "—"

    preset_arg = preset if preset and preset != "(none)" else None
    analyzer = WatermarkAnalyzer(
        scheme=scheme,
        gamma=float(gamma),
        key=key.strip() if key else None,
        tokenizer_name=tokenizer,
        threshold=float(threshold),
        preset=preset_arg,
    )
    result = analyzer.analyze(text)
    stats = result.statistics.to_dict()
    highlighted = tokens_to_html(
        result.tokens, show_highlights=show_highlights, wrap=True, include_style=True
    )
    points = sliding_window_density(
        result.tokens, window=int(density_window or 20), gamma=result.gamma
    )
    heat = density_to_html(points)
    dsum = density_summary(points)
    heat_meta = (
        f"Density window={int(density_window)} · mean={dsum['mean_fraction']:.1%} · "
        f"max={dsum['max_fraction']:.1%} · hot tokens (local z≥4): {dsum['hot_spans']}"
    )

    state = {
        "tokens": [t.to_dict() for t in result.tokens],
        "statistics": stats,
        "text": text,
        "density": [p.to_dict() for p in points],
        "config": {
            "preset": preset,
            "scheme": scheme,
            "gamma": float(gamma),
            "key": key,
            "tokenizer": tokenizer,
            "threshold": float(threshold),
            "density_window": int(density_window or 20),
        },
    }
    HISTORY.add(
        kind="text",
        title=f"{source_label}: {text.strip().replace(chr(10), ' ')[:60]}",
        payload={
            "type": "analyze",
            "text": text[:100_000],
            "statistics": stats,
            "preview": text[:500],
            "config": state["config"],
        },
    )
    cleaned_val = text if seed_cleaned else gr.update()
    return (
        highlighted,
        _verdict_html(stats),
        _stats_markdown(stats),
        cleaned_val,
        state,
        _compare_html(text, text if seed_cleaned else ""),
        heat,
        heat_meta,
    )


def reanalyze_cleaned(
    cleaned, preset, scheme, gamma, key, tokenizer, threshold, show_highlights, density_window
):
    hl, verdict, stats_md, _, state, _, heat, heat_meta = analyze_text(
        cleaned,
        preset,
        scheme,
        gamma,
        key,
        tokenizer,
        threshold,
        show_highlights,
        density_window,
        seed_cleaned=False,
        source_label="Re-analyze cleaned",
    )
    note = "✓ Re-analyzed cleaned text."
    if state and state.get("statistics"):
        s = state["statistics"]
        note += f" Z={s.get('z_score', 0):.2f} · {s.get('verdict_label', '')}"
    return hl, verdict, stats_md, state, note, heat, heat_meta


def toggle_highlights(show: bool, state: dict | None):
    if not state or "tokens" not in state:
        return "<em>No analysis yet.</em>"
    from watermark_core.schemes.base import TokenInfo

    tokens = [TokenInfo(**t) for t in state["tokens"]]
    return tokens_to_html(tokens, show_highlights=show, wrap=True, include_style=True)


def neutralize_paraphrase(text, style, api_key, base_url, model):
    if not text or not text.strip():
        return "", "Provide text to neutralize.", "—"
    config = _llm_config(style, api_key, base_url, model)
    result = neutralize_sync(text, config)
    if not result.success:
        return text, f"⚠️ {result.error}", _compare_html(text, text)
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
    return (
        result.cleaned,
        f"✓ Paraphrased with `{result.model}` ({result.style}).",
        _compare_html(text, result.cleaned),
    )


def neutralize_targeted_ui(text, preset, scheme, gamma, key, tokenizer):
    if not text or not text.strip():
        return "", "Provide text.", "—"
    preset_arg = preset if preset and preset != "(none)" else None
    analyzer = WatermarkAnalyzer(
        scheme=scheme,
        gamma=float(gamma),
        key=key.strip() if key else None,
        tokenizer_name=tokenizer,
        preset=preset_arg,
    )
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
    return (
        result.cleaned,
        f"✓ Offline targeted: {result.notes}",
        _compare_html(text, result.cleaned),
    )


def neutralize_adaptive_ui(
    text, style, max_rounds, target_z, api_key, base_url, model, preset, scheme, gamma, key, tokenizer
):
    if not text or not text.strip():
        return "", "Provide text.", "—"
    preset_arg = preset if preset and preset != "(none)" else None
    analyzer = WatermarkAnalyzer(
        scheme=scheme,
        gamma=float(gamma),
        key=key.strip() if key else None,
        tokenizer_name=tokenizer,
        preset=preset_arg,
        threshold=float(target_z),
    )
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
    return result.cleaned, note, _compare_html(text, result.cleaned)


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


def export_html(state: dict | None, cleaned: str) -> str | None:
    from watermark_core.schemes.base import TokenInfo

    path = TMP / f"annotated_{int(time.time())}.html"
    if state and "tokens" in state:
        tokens = [TokenInfo(**t) for t in state["tokens"]]
        doc = tokens_to_annotated_document(
            tokens, title="fak-u-watermark export", statistics=state.get("statistics")
        )
        if state.get("density"):
            from watermark_core.density import DensityPoint

            pts = [DensityPoint(**p) for p in state["density"]]
            doc = doc.replace(
                "</body>",
                f"<h2>Density heatmap</h2>{density_to_html(pts)}</body>",
            )
        if cleaned:
            doc = doc.replace(
                "</body>",
                f"<h2>Cleaned text</h2><pre>{html.escape(cleaned)}</pre></body>",
            )
    else:
        doc = f"<!DOCTYPE html><html><body><pre>{html.escape(cleaned or '')}</pre></body></html>"
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
    original = p.get("text") or p.get("original") or p.get("preview") or ""
    cleaned = p.get("cleaned") or original
    return (
        original,
        cleaned,
        _compare_html(original, cleaned),
        f"✓ Restored **{entry.title}** (`{entry.id[:8]}`)",
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
    results = analyze_batch_files(
        paths,
        scheme=scheme,
        gamma=float(gamma),
        key=key.strip() if key else None,
        tokenizer_name=tokenizer,
        preset=preset_arg,
        threshold=float(threshold),
    )
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
        gr.Markdown(
            """
# fak-u-watermark
**Strip the mark. Keep the meaning.**

Detect → highlight (yellow + density heatmap) → neutralize (paraphrase / targeted / adaptive).
Configure **API keys** in the Settings tab (saved under `~/.faku/settings.json`).
            """
        )

        with gr.Tabs():
            # ── SETTINGS ──────────────────────────────────────────────────
            with gr.Tab("Settings"):
                gr.Markdown(
                    f"""
### API keys (manual)
Keys are stored **locally only** in `{settings_path()}` (file mode 600).  
Never committed to git. You can also use env vars (`FAKU_API_KEY`, `OPENAI_API_KEY`, …).

Supported: any **OpenAI-compatible** endpoint (OpenAI, DeepSeek, xAI Grok, proxies).
                    """
                )
                with gr.Row():
                    set_provider = gr.Dropdown(
                        choices=provider_choices,
                        value="custom",
                        label="Provider preset",
                    )
                    btn_apply_provider = gr.Button("Apply preset URLs")
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
                with gr.Row():
                    btn_save_settings = gr.Button("Save settings", variant="primary")
                    btn_load_settings = gr.Button("Reload")
                    btn_clear_key = gr.Button("Clear API key")
                settings_status = gr.Markdown()

                gr.Markdown(
                    """
**Provider base URLs (examples)**

| Provider | Base URL | Example model |
|----------|----------|---------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
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
            with gr.Tab("Text"):
                analysis_state = gr.State(None)

                with gr.Row():
                    with gr.Column(scale=3):
                        text_in = gr.Textbox(
                            label="Input text",
                            placeholder="Paste text to analyze…",
                            lines=12,
                            elem_id="input-text",
                        )
                        file_in = gr.File(
                            label="Or upload .txt / .md",
                            file_types=[".txt", ".md", ".text"],
                            type="filepath",
                        )
                    with gr.Column(scale=2):
                        preset = gr.Dropdown(
                            choices=preset_choices,
                            value="kirchenbauer_default",
                            label="Preset",
                        )
                        scheme = gr.Dropdown(
                            choices=["kgw", "unigram"], value="kgw", label="Scheme"
                        )
                        gamma = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Gamma")
                        key = gr.Textbox(label="Secret key / seed", value="15485863")
                        tokenizer = gr.Dropdown(
                            choices=tokenizer_choices, value="gpt2", label="Tokenizer"
                        )
                        threshold = gr.Slider(
                            1.0, 10.0, value=4.0, step=0.5, label="Z-score threshold"
                        )
                        density_window = gr.Slider(
                            5, 80, value=20, step=1, label="Density window size"
                        )
                        show_hl = gr.Checkbox(value=True, label="Show yellow highlights")

                with gr.Row():
                    btn_analyze = gr.Button("Analyze", variant="primary")
                    btn_neu = gr.Button("Neutralize (paraphrase)")
                    btn_targeted = gr.Button("Targeted green→red (offline)")
                    btn_adaptive = gr.Button("Adaptive neutralize")
                    btn_reanalyze = gr.Button("Re-analyze cleaned")

                with gr.Accordion("LLM options (uses Settings keys if fields empty)", open=False):
                    style = gr.Radio(
                        choices=["subtle", "strong"],
                        value="subtle",
                        label="Paraphrase style",
                    )
                    api_key = gr.Textbox(
                        label="API key override",
                        type="password",
                        placeholder="Leave empty to use Settings / env",
                    )
                    base_url = gr.Textbox(label="Base URL override")
                    model = gr.Textbox(label="Model override")
                    max_rounds = gr.Slider(1, 6, value=3, step=1, label="Adaptive max rounds")
                    target_z = gr.Slider(
                        1.0, 8.0, value=4.0, step=0.5, label="Adaptive target z-score"
                    )

                gr.Markdown("### Original (highlighted) + statistics")
                verdict_out = gr.HTML(value="—")
                highlighted_out = gr.HTML()
                stats_out = gr.Markdown()

                gr.Markdown("### Density heatmap")
                heat_out = gr.HTML()
                heat_meta = gr.Markdown()

                gr.Markdown("### Cleaned (editable)")
                cleaned_out = gr.Textbox(
                    label="Cleaned text — edit freely",
                    lines=12,
                    interactive=True,
                    elem_id="cleaned-text",
                )
                status_out = gr.Markdown()

                with gr.Row():
                    btn_copy = gr.Button("Copy cleaned", variant="primary")
                    btn_export_txt = gr.Button("Export .txt")
                    btn_export_html = gr.Button("Export .html")
                    export_txt = gr.File(label="Download .txt")
                    export_html_f = gr.File(label="Download .html")

                gr.Markdown("### Before / After")
                compare_out = gr.HTML(value="—")

                with gr.Accordion("Batch analyze", open=False):
                    batch_files = gr.File(
                        label="Multiple .txt / .md files",
                        file_count="multiple",
                        type="filepath",
                    )
                    btn_batch = gr.Button("Run batch")
                    batch_out = gr.Markdown()

                with gr.Accordion("History", open=False):
                    hist_dd = gr.Dropdown(label="Past text jobs", choices=[])
                    with gr.Row():
                        btn_refresh_hist = gr.Button("Refresh")
                        btn_restore_hist = gr.Button("Restore selected", variant="primary")
                    hist_status = gr.Markdown()

                file_in.change(on_file_upload, inputs=[file_in], outputs=[text_in])

                analyze_inputs = [
                    text_in,
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
                    highlighted_out,
                    verdict_out,
                    stats_out,
                    cleaned_out,
                    analysis_state,
                    compare_out,
                    heat_out,
                    heat_meta,
                ]

                btn_analyze.click(analyze_text, inputs=analyze_inputs, outputs=analyze_outputs)
                show_hl.change(
                    toggle_highlights,
                    inputs=[show_hl, analysis_state],
                    outputs=[highlighted_out],
                )
                btn_neu.click(
                    neutralize_paraphrase,
                    inputs=[text_in, style, api_key, base_url, model],
                    outputs=[cleaned_out, status_out, compare_out],
                )
                btn_targeted.click(
                    neutralize_targeted_ui,
                    inputs=[text_in, preset, scheme, gamma, key, tokenizer],
                    outputs=[cleaned_out, status_out, compare_out],
                )
                btn_adaptive.click(
                    neutralize_adaptive_ui,
                    inputs=[
                        text_in,
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
                    ],
                    outputs=[cleaned_out, status_out, compare_out],
                )
                btn_reanalyze.click(
                    reanalyze_cleaned,
                    inputs=[
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
                    outputs=[
                        highlighted_out,
                        verdict_out,
                        stats_out,
                        analysis_state,
                        status_out,
                        heat_out,
                        heat_meta,
                    ],
                )
                btn_copy.click(
                    fn=copy_status,
                    inputs=[cleaned_out],
                    outputs=[status_out],
                    js="(text) => { if (text) { navigator.clipboard.writeText(text); } return [text]; }",
                )
                btn_export_txt.click(export_plain, inputs=[cleaned_out], outputs=[export_txt])
                btn_export_html.click(
                    export_html, inputs=[analysis_state, cleaned_out], outputs=[export_html_f]
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
            with gr.Tab("Images"):
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(
                            label="Upload image", type="pil", sources=["upload", "clipboard"]
                        )
                        btn_exif = gr.Button("Read EXIF / metadata", variant="primary")
                        btn_strip = gr.Button("Strip all metadata")
                        btn_c2pa = gr.Button("Detect C2PA / Content Credentials")
                        btn_c2pa_strip = gr.Button("Strip C2PA (re-encode)")
                        with gr.Accordion("Edit metadata fields", open=False):
                            artist = gr.Textbox(label="Artist")
                            copyright_ = gr.Textbox(label="Copyright")
                            description = gr.Textbox(label="ImageDescription")
                            btn_update = gr.Button("Apply field updates")
                    with gr.Column():
                        img_out = gr.Image(label="Result / preview", type="pil")
                        img_download = gr.File(label="Download cleaned image")
                        img_status = gr.Markdown()
                        exif_out = gr.Code(label="Metadata / C2PA report", language="json", lines=14)

                gr.Markdown("### Visual watermark removal — paint the region")
                mask_editor = gr.ImageMask(
                    label="Paint watermark region (brush)",
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
                btn_inpaint = gr.Button("Remove painted region", variant="primary")

                with gr.Accordion("Image history", open=False):
                    img_hist_dd = gr.Dropdown(label="Past image jobs", choices=[])
                    with gr.Row():
                        btn_img_hist = gr.Button("Refresh")
                        btn_img_restore = gr.Button("Restore selected", variant="primary")
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
            with gr.Tab("About"):
                gr.Markdown(
                    """
## How it works

### Text
1. Tokenize (default GPT-2) → reconstruct green list from key → z-score.
2. Soft yellow highlights + **density heatmap** (sliding window).
3. Neutralize:
   - **Paraphrase** (API, subtle/strong)
   - **Targeted green→red** (offline, needs correct key)
   - **Adaptive** (API loop until z drops)
4. Copy / export / re-analyze / history restore / batch files.

### Images
- EXIF view/edit/strip · **C2PA marker detect/strip** · brush inpaint (local or API).

### Config
Settings tab or `~/.faku/settings.json` or env `FAKU_API_KEY` / `FAKU_BASE_URL` / `FAKU_MODEL`.

See **README.md** for deploy and full usage.
                    """
                )

        gr.Markdown(
            "<center><small>fak-u-watermark · ~/.faku/ for history + settings</small></center>"
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
    demo.launch(
        server_name=host,
        server_port=port,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="amber", secondary_hue="stone"),
        inbrowser=os.environ.get("FAKU_UI_INBROWSER", "1") not in ("0", "false", "False"),
    )


if __name__ == "__main__":
    main()
