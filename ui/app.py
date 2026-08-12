"""Gradio MVP UI — text + image tabs.

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

# Path setup before package imports
_ROOT = Path(__file__).resolve().parent.parent
_PACKAGES = _ROOT / "packages"
for p in (_PACKAGES, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gradio as gr
from PIL import Image

from watermark_core.analyzer import WatermarkAnalyzer
from watermark_core.history import HistoryStore
from watermark_core.neutralize import NeutralizeConfig, neutralize_sync
from watermark_core.schemes import PRESETS
from watermark_core.tokenizer import AVAILABLE_TOKENIZERS
from watermark_core.visualization import (
    tokens_to_annotated_document,
    tokens_to_html,
)

from image_tools.exif import read_exif, strip_exif, update_exif
from image_tools.inpaint import extract_mask_from_editor, inpaint_region, rectangle_mask

SOFT_YELLOW = "#FEF08A"
HISTORY = HistoryStore()
TMP = Path("/tmp/faku")
TMP.mkdir(parents=True, exist_ok=True)

# Max base64 payload stored in image history (~1.5 MB decoded)
_MAX_HISTORY_IMAGE_BYTES = 1_500_000

CUSTOM_CSS = f"""
.watermark-signal {{
  background-color: {SOFT_YELLOW} !important;
  border-radius: 2px;
  padding: 0 1px;
}}
.watermark-output {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.65;
  font-size: 0.95rem;
  padding: 0.75rem;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #fafafa;
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
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}}
@media (max-width: 800px) {{
  .faku-compare {{ grid-template-columns: 1fr; }}
}}
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
    """Simple before/after side-by-side panel."""
    o = html.escape(original or "")
    c = html.escape(cleaned or "")
    return f"""
<div class="faku-compare">
  <div>
    <strong>Original</strong>
    <pre style="white-space:pre-wrap;background:#fafafa;padding:0.75rem;border-radius:8px;border:1px solid #e5e7eb;max-height:20rem;overflow:auto">{o}</pre>
  </div>
  <div>
    <strong>Cleaned</strong>
    <pre style="white-space:pre-wrap;background:#fafafa;padding:0.75rem;border-radius:8px;border:1px solid #e5e7eb;max-height:20rem;overflow:auto">{c}</pre>
  </div>
</div>
"""


def _pil_to_b64_png(img: Image.Image, max_bytes: int = _MAX_HISTORY_IMAGE_BYTES) -> str | None:
    buf = io.BytesIO()
    rgb = img.convert("RGB") if img.mode not in ("RGB", "L", "RGBA") else img
    rgb.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    if len(data) > max_bytes:
        # Downscale once
        scale = 0.5
        w, h = rgb.size
        small = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
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


# ── Text handlers ──────────────────────────────────────────────────────────


def analyze_text(
    text: str,
    preset: str,
    scheme: str,
    gamma: float,
    key: str,
    tokenizer: str,
    threshold: float,
    show_highlights: bool,
    *,
    seed_cleaned: bool = True,
    source_label: str = "Analyze",
):
    if not text or not text.strip():
        empty = "<em>Paste or upload text, then click Analyze.</em>"
        cleaned_val = (text or "") if seed_cleaned else gr.update()
        return empty, "—", "", cleaned_val, None, "—"

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
        result.tokens,
        show_highlights=show_highlights,
        wrap=True,
        include_style=True,
    )
    verdict = _verdict_html(stats)
    stats_md = _stats_markdown(stats)

    state = {
        "tokens": [t.to_dict() for t in result.tokens],
        "statistics": stats,
        "text": text,
        "result": result.to_dict(),
        "config": {
            "preset": preset,
            "scheme": scheme,
            "gamma": float(gamma),
            "key": key,
            "tokenizer": tokenizer,
            "threshold": float(threshold),
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
    return highlighted, verdict, stats_md, cleaned_val, state, verdict


def reanalyze_cleaned(
    cleaned: str,
    preset: str,
    scheme: str,
    gamma: float,
    key: str,
    tokenizer: str,
    threshold: float,
    show_highlights: bool,
):
    """Re-run detection on the (possibly edited) cleaned text."""
    hl, verdict, stats_md, _cleaned, state, _ = analyze_text(
        cleaned,
        preset,
        scheme,
        gamma,
        key,
        tokenizer,
        threshold,
        show_highlights,
        seed_cleaned=False,
        source_label="Re-analyze cleaned",
    )
    note = "✓ Re-analyzed cleaned text — stats updated."
    if state and state.get("statistics"):
        s = state["statistics"]
        note += f" Z={s.get('z_score', 0):.2f} · {s.get('verdict_label', '')}"
    return hl, verdict, stats_md, state, note


def toggle_highlights(show: bool, state: dict | None):
    if not state or "tokens" not in state:
        return "<em>No analysis yet.</em>"
    from watermark_core.schemes.base import TokenInfo

    tokens = [TokenInfo(**t) for t in state["tokens"]]
    return tokens_to_html(tokens, show_highlights=show, wrap=True, include_style=True)


def neutralize_text(
    text: str,
    style: str,
    api_key: str,
    base_url: str,
    model: str,
):
    if not text or not text.strip():
        return "", "Provide text to neutralize.", "—"

    config = NeutralizeConfig.from_env(style=style)  # type: ignore[arg-type]
    if api_key and api_key.strip():
        config.api_key = api_key.strip()
    if base_url and base_url.strip():
        config.base_url = base_url.strip()
    if model and model.strip():
        config.model = model.strip()

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
    compare = _compare_html(text, result.cleaned)
    return (
        result.cleaned,
        f"✓ Neutralized with `{result.model}` ({result.style}). Edit freely, then **Re-analyze** or **Copy**.",
        compare,
    )


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
            tokens,
            title="fak-u-watermark export",
            statistics=state.get("statistics"),
        )
        if cleaned:
            doc = doc.replace(
                "</body>",
                f"<h2>Cleaned text</h2><pre>{html.escape(cleaned)}</pre></body>",
            )
    else:
        doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>export</title></head>
<body><pre>{html.escape(cleaned or "")}</pre></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return str(path)


def refresh_text_history_dropdown():
    choices = HISTORY.choices(kind="text", limit=40)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def restore_text_history(label: str | None):
    entry = HISTORY.get_by_choice(label)
    if not entry:
        return (
            gr.update(),
            gr.update(),
            "—",
            "No history entry selected.",
            "—",
        )
    p = entry.payload or {}
    original = p.get("text") or p.get("original") or p.get("preview") or ""
    cleaned = p.get("cleaned") or original
    compare = _compare_html(original, cleaned) if cleaned else "—"
    note = f"✓ Restored **{entry.title}** (`{entry.id[:8]}`)"
    return original, cleaned, compare, note, "—"


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


# ── Image handlers ─────────────────────────────────────────────────────────


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
        payload={
            "type": "exif",
            "keys": list(meta.keys())[:50],
            "image_b64": b64,
        },
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
    b64 = _pil_to_b64_png(out)
    HISTORY.add(
        kind="image",
        title="Strip EXIF",
        payload={"type": "strip_exif", "image_b64": b64},
    )
    return out, _save_download(out, "stripped"), "✓ Metadata stripped — download ready."


def do_update_exif(image: Image.Image | None, artist: str, copyright_: str, description: str):
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


def load_image_into_mask_editor(image: Image.Image | None):
    """Seed the ImageMask editor with the uploaded photo."""
    if image is None:
        return None
    return image


def do_inpaint_from_editor(
    editor_value,
    fallback_image: Image.Image | None,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    use_rect: bool,
    instruction: str,
    method: str,
    api_key: str,
    base_url: str,
    model: str,
):
    base_img, brush_mask = extract_mask_from_editor(editor_value)
    image = base_img or fallback_image
    if image is None:
        return None, None, "Upload an image and paint over the watermark (or set a rectangle)."

    w, h = image.size
    mask = brush_mask
    brush_empty = mask is None or int(mask.sum()) == 0

    if use_rect or brush_empty:
        xa, xb = int(x1), int(x2)
        ya, yb = int(y1), int(y2)
        if abs(xb - xa) >= 2 and abs(yb - ya) >= 2:
            rect = rectangle_mask(w, h, xa, ya, xb, yb)
            if brush_empty:
                mask = rect
            else:
                mask = np.maximum(mask, rect)
        elif brush_empty:
            return (
                image,
                None,
                "Paint the watermark region with the brush, or enable rectangle coords.",
            )

    if mask is None or int(mask.sum()) == 0:
        return image, None, "Mask is empty — paint over the watermark first."

    result = inpaint_region(
        image,
        mask,
        instruction=instruction or None,
        method=method,
        api_key=api_key.strip() if api_key else None,
        base_url=base_url.strip() if base_url else None,
        model=model.strip() if model else None,
    )
    if not result.success or not result.image_bytes:
        return image, None, f"⚠️ Inpaint failed: {result.error}"

    out = Image.open(io.BytesIO(result.image_bytes))
    b64 = _pil_to_b64_png(out)
    HISTORY.add(
        kind="image",
        title=f"Inpaint ({result.method})",
        payload={
            "type": "inpaint",
            "instruction": instruction,
            "method": result.method,
            "image_b64": b64,
        },
    )
    note = f"✓ Region removed with **{result.method}**."
    if instruction and result.method.startswith("opencv"):
        note += " (Local OpenCV ignores the text instruction — use method **api** or **auto** for LLM-guided edits.)"
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
        note += " — no image stored for this entry (older history)."
    return img, img, meta_txt, note


# ── Build UI ────────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    preset_choices = ["(none)"] + list(PRESETS.keys())
    tokenizer_choices = list(AVAILABLE_TOKENIZERS.keys())

    with gr.Blocks(title="fak-u-watermark") as demo:
        gr.Markdown(
            """
# fak-u-watermark
**Strip the mark. Keep the meaning.**

Identify statistical AI watermarks → highlight signal tokens in soft yellow → neutralize.
Same tool for **text** and **images**. Offline detection & EXIF; neutralization / API inpaint need a key.
            """
        )

        with gr.Tabs():
            # ── TEXT TAB ──────────────────────────────────────────────────
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
                            choices=["kgw", "unigram"],
                            value="kgw",
                            label="Scheme",
                        )
                        gamma = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Gamma")
                        key = gr.Textbox(
                            label="Secret key / seed",
                            value="15485863",
                            placeholder="Integer or string secret",
                        )
                        tokenizer = gr.Dropdown(
                            choices=tokenizer_choices,
                            value="gpt2",
                            label="Tokenizer",
                        )
                        threshold = gr.Slider(
                            1.0, 10.0, value=4.0, step=0.5, label="Z-score threshold"
                        )
                        show_hl = gr.Checkbox(value=True, label="Show yellow highlights")

                with gr.Row():
                    btn_analyze = gr.Button("Analyze", variant="primary")
                    btn_neutralize = gr.Button("Neutralize watermark", variant="secondary")
                    btn_reanalyze = gr.Button("Re-analyze cleaned")

                with gr.Accordion("LLM settings (for neutralization)", open=False):
                    style = gr.Radio(
                        choices=["subtle", "strong"],
                        value="subtle",
                        label="Paraphrase style",
                        info="Default is subtle — light wording changes only.",
                    )
                    api_key = gr.Textbox(
                        label="API key",
                        type="password",
                        placeholder="FAKU_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY / XAI_API_KEY",
                    )
                    base_url = gr.Textbox(
                        label="Base URL (OpenAI-compatible)",
                        placeholder="https://api.openai.com/v1  ·  https://api.deepseek.com  ·  https://api.x.ai/v1",
                    )
                    model = gr.Textbox(
                        label="Model",
                        placeholder="gpt-4o-mini / deepseek-chat / grok-2-latest",
                    )

                gr.Markdown("### Original (highlighted) + statistics")
                verdict_out = gr.HTML(value="—")
                highlighted_out = gr.HTML()
                stats_out = gr.Markdown()

                gr.Markdown("### Cleaned (editable) — Copy · Export · Re-analyze")
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

                with gr.Accordion("History (click restore)", open=False):
                    hist_dd = gr.Dropdown(
                        label="Past text jobs",
                        choices=[],
                        interactive=True,
                    )
                    with gr.Row():
                        btn_refresh_hist = gr.Button("Refresh")
                        btn_restore_hist = gr.Button("Restore selected", variant="primary")
                    hist_status = gr.Markdown()

                # Wire text events
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
                ]
                analyze_outputs = [
                    highlighted_out,
                    verdict_out,
                    stats_out,
                    cleaned_out,
                    analysis_state,
                    compare_out,
                ]

                def _analyze_and_clear_compare(*args):
                    hl, v, sm, cleaned, state, _ver = analyze_text(*args)
                    # Seed compare with original only until neutralize
                    cmp = _compare_html(args[0] or "", cleaned if isinstance(cleaned, str) else "")
                    return hl, v, sm, cleaned, state, cmp

                btn_analyze.click(
                    _analyze_and_clear_compare,
                    inputs=analyze_inputs,
                    outputs=analyze_outputs,
                )

                show_hl.change(
                    toggle_highlights,
                    inputs=[show_hl, analysis_state],
                    outputs=[highlighted_out],
                )

                btn_neutralize.click(
                    neutralize_text,
                    inputs=[text_in, style, api_key, base_url, model],
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
                    ],
                    outputs=[
                        highlighted_out,
                        verdict_out,
                        stats_out,
                        analysis_state,
                        status_out,
                    ],
                )

                # Copy: browser clipboard via JS + status feedback
                btn_copy.click(
                    fn=copy_status,
                    inputs=[cleaned_out],
                    outputs=[status_out],
                    js="(text) => { if (text) { navigator.clipboard.writeText(text); } return [text]; }",
                )

                btn_export_txt.click(export_plain, inputs=[cleaned_out], outputs=[export_txt])
                btn_export_html.click(
                    export_html,
                    inputs=[analysis_state, cleaned_out],
                    outputs=[export_html_f],
                )

                btn_refresh_hist.click(refresh_text_history_dropdown, outputs=[hist_dd])
                btn_restore_hist.click(
                    restore_text_history,
                    inputs=[hist_dd],
                    outputs=[text_in, cleaned_out, compare_out, hist_status, status_out],
                )
                # Auto-load history choices when accordion opens is not trivial;
                # refresh on demo load:
                demo.load(refresh_text_history_dropdown, outputs=[hist_dd])

            # ── IMAGE TAB ─────────────────────────────────────────────────
            with gr.Tab("Images"):
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(
                            label="Upload image",
                            type="pil",
                            sources=["upload", "clipboard"],
                        )
                        btn_exif = gr.Button("Read EXIF / metadata", variant="primary")
                        btn_strip = gr.Button("Strip all metadata")
                        with gr.Accordion("Edit metadata fields", open=False):
                            artist = gr.Textbox(label="Artist")
                            copyright_ = gr.Textbox(label="Copyright")
                            description = gr.Textbox(label="ImageDescription")
                            btn_update = gr.Button("Apply field updates")
                    with gr.Column():
                        img_out = gr.Image(label="Result / preview", type="pil")
                        img_download = gr.File(label="Download cleaned image")
                        img_status = gr.Markdown()
                        exif_out = gr.Code(label="Metadata", language="json", lines=14)

                gr.Markdown(
                    """### Visual watermark removal
**Paint** over the watermark with the brush (white strokes), or use a rectangle.
Local OpenCV is offline; choose **api** / **auto** for LLM-guided edits (OpenAI-compatible `images/edits`)."""
                )

                mask_editor = gr.ImageMask(
                    label="Paint watermark region (brush)",
                    type="pil",
                    layers=False,
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=24, color_mode="fixed"),
                    sources=["upload", "clipboard"],
                )

                with gr.Accordion("Rectangle fallback (optional)", open=False):
                    use_rect = gr.Checkbox(
                        value=False,
                        label="Use rectangle coords instead of (or in addition to) brush",
                    )
                    with gr.Row():
                        x1 = gr.Number(label="x1", value=0, precision=0)
                        y1 = gr.Number(label="y1", value=0, precision=0)
                        x2 = gr.Number(label="x2", value=100, precision=0)
                        y2 = gr.Number(label="y2", value=50, precision=0)

                instruction = gr.Textbox(
                    label="Instruction (used by API / auto methods)",
                    placeholder="e.g. remove the logo, keep the sky natural",
                    value="Remove the watermark or logo and fill the area naturally.",
                )
                method = gr.Radio(
                    choices=["telea", "ns", "api", "auto"],
                    value="telea",
                    label="Inpaint method",
                    info="telea/ns = local · api = OpenAI images/edits · auto = API if key+instruction else local",
                )
                with gr.Accordion("Inpaint API settings", open=False):
                    inpaint_key = gr.Textbox(label="API key", type="password")
                    inpaint_base = gr.Textbox(
                        label="Base URL",
                        placeholder="https://api.openai.com/v1",
                    )
                    inpaint_model = gr.Textbox(
                        label="Model",
                        placeholder="dall-e-2",
                    )

                btn_inpaint = gr.Button("Remove painted region", variant="primary")

                with gr.Accordion("Image history (click restore)", open=False):
                    img_hist_dd = gr.Dropdown(label="Past image jobs", choices=[])
                    with gr.Row():
                        btn_img_hist = gr.Button("Refresh")
                        btn_img_restore = gr.Button("Restore selected", variant="primary")
                    img_hist_status = gr.Markdown()

                # Seed mask editor when main image changes
                img_in.change(load_image_into_mask_editor, inputs=[img_in], outputs=[mask_editor])

                btn_exif.click(
                    process_exif,
                    inputs=[img_in],
                    outputs=[exif_out, img_out, img_download, img_status],
                )
                btn_strip.click(
                    do_strip_exif,
                    inputs=[img_in],
                    outputs=[img_out, img_download, img_status],
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

### Text (KGW / green-red list family)
1. **Tokenize** with an open tokenizer (default GPT-2).
2. **Reconstruct** the green list for each position from the previous token + secret key.
3. **Score** how often tokens fall in the green list → z-score vs expected γ.
4. **Highlight** green-list (signal) tokens with a soft yellow background.
5. **Neutralize** via subtle paraphrase (OpenAI-compatible API) so the statistical signal breaks.
6. **Copy / Export / Re-analyze** the cleaned text; restore past jobs from History.

### Images
- Full **EXIF / metadata** view, edit, and strip (offline).
- **Brush** (or rectangle) over a logo → local OpenCV or API inpaint.
- Download cleaned images; restore from history.

### Limitations (honest)
- Without the provider’s key you cannot perfectly color tokens for proprietary schemes.
- Classic KGW detection assumes you know or can guess scheme parameters.
- Paraphrase may slightly change wording; always review cleaned text.
- API inpainting quality depends on the provider; local Telea/NS is basic but offline.

### Config (env)
| Variable | Purpose |
|----------|---------|
| `FAKU_API_KEY` / `OPENAI_API_KEY` / … | Neutralization & optional inpaint |
| `FAKU_BASE_URL` / `OPENAI_BASE_URL` | API base URL |
| `FAKU_MODEL` | Chat model for paraphrase |
| `FAKU_INPAINT_MODEL` | Images-edits model (default `dall-e-2`) |

**Tagline:** *Detect. Highlight. Neutralize.*
                    """
                )

        gr.Markdown(
            "<center><small>fak-u-watermark · offline detection · local history in ~/.faku/</small></center>"
        )

    return demo


def main():
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="amber", secondary_hue="stone"),
    )


if __name__ == "__main__":
    main()
