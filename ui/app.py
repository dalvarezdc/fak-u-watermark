"""Gradio MVP UI — text + image tabs.

Identify → Show → Remove.
"""

from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path

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
from image_tools.inpaint import inpaint_region, rectangle_mask

SOFT_YELLOW = "#FEF08A"
HISTORY = HistoryStore()

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
):
    if not text or not text.strip():
        empty = "<em>Paste or upload text, then click Analyze.</em>"
        return empty, "—", "", text or "", None

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

    # Cache tokens for export via state
    state = {
        "tokens": [t.to_dict() for t in result.tokens],
        "statistics": stats,
        "text": text,
        "result": result.to_dict(),
    }

    HISTORY.add(
        kind="text",
        title=f"Analyze: {text.strip().replace(chr(10), ' ')[:60]}",
        payload={"type": "analyze", "statistics": stats, "preview": text[:500]},
    )
    return highlighted, verdict, stats_md, text, state


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
        return "", "Provide text to neutralize."

    config = NeutralizeConfig.from_env(style=style)  # type: ignore[arg-type]
    if api_key and api_key.strip():
        config.api_key = api_key.strip()
    if base_url and base_url.strip():
        config.base_url = base_url.strip()
    if model and model.strip():
        config.model = model.strip()

    result = neutralize_sync(text, config)
    if not result.success:
        return text, f"⚠️ {result.error}"

    HISTORY.add(
        kind="text",
        title=f"Neutralize ({style}): {text.strip().replace(chr(10), ' ')[:50]}",
        payload={
            "type": "neutralize",
            "original": text[:2000],
            "cleaned": result.cleaned[:2000],
            "style": style,
            "model": result.model,
        },
    )
    return result.cleaned, f"✓ Neutralized with `{result.model}` ({result.style})"


def export_plain(cleaned: str) -> str | None:
    if not cleaned:
        return None
    path = Path("/tmp") / f"faku_cleaned_{int(time.time())}.txt"
    path.write_text(cleaned, encoding="utf-8")
    return str(path)


def export_html(state: dict | None, cleaned: str) -> str | None:
    from watermark_core.schemes.base import TokenInfo

    path = Path("/tmp") / f"faku_annotated_{int(time.time())}.html"
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


def load_history_text():
    entries = HISTORY.list(kind="text", limit=30)
    if not entries:
        return "No history yet."
    lines = []
    for e in entries:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.created_at))
        lines.append(f"- **{ts}** · {e.title} (`{e.id[:8]}`)")
    return "\n".join(lines)


def on_file_upload(file_obj):
    if file_obj is None:
        return gr.update()
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path:
        return gr.update()
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        return content
    except Exception as exc:  # noqa: BLE001
        return f"[Error reading file: {exc}]"


# ── Image handlers ─────────────────────────────────────────────────────────


def process_exif(image: Image.Image | None):
    if image is None:
        return "{}", None
    meta = read_exif(image)
    # Pretty JSON for display
    display = {k: v for k, v in meta.items() if not str(k).startswith("piexif.")}
    # Prefer compact view; include full under piexif if needed
    text = json.dumps(display, indent=2, default=str, ensure_ascii=False)
    HISTORY.add(
        kind="image",
        title="EXIF inspect",
        payload={"type": "exif", "keys": list(meta.keys())[:50]},
    )
    return text, image


def do_strip_exif(image: Image.Image | None):
    if image is None:
        return None, "Upload an image first."
    import io

    buf = io.BytesIO()
    # Ensure we have bytes with original format attempt
    fmt = image.format or "PNG"
    save_img = image
    if fmt.upper() in ("JPEG", "JPG") and save_img.mode not in ("RGB", "L"):
        save_img = save_img.convert("RGB")
    save_img.save(buf, format=fmt if fmt.upper() != "JPG" else "JPEG")
    cleaned = strip_exif(buf.getvalue())
    out = Image.open(io.BytesIO(cleaned))
    HISTORY.add(kind="image", title="Strip EXIF", payload={"type": "strip_exif"})
    return out, "✓ Metadata stripped. Download the cleaned image."


def do_update_exif(image: Image.Image | None, artist: str, copyright_: str, description: str):
    if image is None:
        return None, "Upload an image first."
    import io

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
        return image, "No fields to update."
    data = update_exif(buf.getvalue(), updates)
    out = Image.open(io.BytesIO(data))
    return out, f"✓ Updated fields: {', '.join(updates.keys())}"


def do_inpaint(
    image: Image.Image | None,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    instruction: str,
    method: str,
):
    if image is None:
        return None, "Upload an image first."
    w, h = image.size
    # Clamp
    x1, x2 = int(x1), int(x2)
    y1, y2 = int(y1), int(y2)
    if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
        return image, "Draw a larger rectangle (set x1,y1,x2,y2 covering the watermark)."

    mask = rectangle_mask(w, h, x1, y1, x2, y2)
    result = inpaint_region(
        image,
        mask,
        instruction=instruction or None,
        method="ns" if method == "ns" else "telea",
    )
    if not result.success or not result.image_bytes:
        return image, f"⚠️ Inpaint failed: {result.error}"

    import io

    out = Image.open(io.BytesIO(result.image_bytes))
    HISTORY.add(
        kind="image",
        title=f"Inpaint ({result.method})",
        payload={"type": "inpaint", "instruction": instruction, "rect": [x1, y1, x2, y2]},
    )
    note = f"✓ Region inpainted with {result.method}."
    if instruction:
        note += f" Instruction noted (local model ignores LLM instruction): “{instruction}”"
    return out, note


def load_history_images():
    entries = HISTORY.list(kind="image", limit=30)
    if not entries:
        return "No image history yet."
    lines = []
    for e in entries:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.created_at))
        lines.append(f"- **{ts}** · {e.title} (`{e.id[:8]}`)")
    return "\n".join(lines)


# ── Build UI ────────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    preset_choices = ["(none)"] + list(PRESETS.keys())
    tokenizer_choices = list(AVAILABLE_TOKENIZERS.keys())

    with gr.Blocks(
        title="fak-u-watermark",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="amber", secondary_hue="stone"),
    ) as demo:
        gr.Markdown(
            """
# fak-u-watermark
**Strip the mark. Keep the meaning.**

Identify statistical AI watermarks → highlight signal tokens in soft yellow → neutralize.
Same tool for **text** and **images**. Offline detection & EXIF; neutralization needs an API key.
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
                        placeholder="https://api.openai.com/v1  or  https://api.deepseek.com  or  https://api.x.ai/v1",
                    )
                    model = gr.Textbox(
                        label="Model",
                        placeholder="gpt-4o-mini / deepseek-chat / grok-2-latest",
                    )

                gr.Markdown("### Original (highlighted)")
                with gr.Row():
                    verdict_out = gr.HTML(label="Verdict")
                highlighted_out = gr.HTML(label="Highlighted text")
                stats_out = gr.Markdown(label="Statistics")

                gr.Markdown("### Cleaned (editable)")
                cleaned_out = gr.Textbox(
                    label="Cleaned text — edit freely",
                    lines=12,
                    interactive=True,
                )
                status_out = gr.Markdown()

                with gr.Row():
                    # Gradio clipboard via JS component alternative: download files
                    export_txt = gr.File(label="Export plain text")
                    export_html_f = gr.File(label="Export annotated HTML")
                    btn_export_txt = gr.Button("Export .txt")
                    btn_export_html = gr.Button("Export .html")

                gr.Markdown(
                    "*Tip: select all cleaned text (⌘/Ctrl+A) then copy (⌘/Ctrl+C), "
                    "or use Export. Re-analyze after edits to re-check the z-score.*"
                )

                with gr.Accordion("History", open=False):
                    history_md = gr.Markdown("No history yet.")
                    btn_refresh_hist = gr.Button("Refresh history")

                file_in.change(on_file_upload, inputs=[file_in], outputs=[text_in])

                btn_analyze.click(
                    analyze_text,
                    inputs=[
                        text_in,
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
                        cleaned_out,
                        analysis_state,
                    ],
                )

                show_hl.change(
                    toggle_highlights,
                    inputs=[show_hl, analysis_state],
                    outputs=[highlighted_out],
                )

                btn_neutralize.click(
                    neutralize_text,
                    inputs=[text_in, style, api_key, base_url, model],
                    outputs=[cleaned_out, status_out],
                )

                btn_export_txt.click(export_plain, inputs=[cleaned_out], outputs=[export_txt])
                btn_export_html.click(
                    export_html,
                    inputs=[analysis_state, cleaned_out],
                    outputs=[export_html_f],
                )
                btn_refresh_hist.click(load_history_text, outputs=[history_md])

            # ── IMAGE TAB ─────────────────────────────────────────────────
            with gr.Tab("Images"):
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(label="Upload image", type="pil")
                        btn_exif = gr.Button("Read EXIF / metadata", variant="primary")
                        btn_strip = gr.Button("Strip all metadata")
                        with gr.Accordion("Edit metadata fields", open=False):
                            artist = gr.Textbox(label="Artist")
                            copyright_ = gr.Textbox(label="Copyright")
                            description = gr.Textbox(label="ImageDescription")
                            btn_update = gr.Button("Apply field updates")
                    with gr.Column():
                        img_out = gr.Image(label="Result / preview", type="pil")
                        img_status = gr.Markdown()
                        exif_out = gr.Code(label="Metadata", language="json", lines=16)

                gr.Markdown("### Visual watermark removal (region inpaint)")
                gr.Markdown(
                    "Set a rectangle covering the watermark (pixel coordinates). "
                    "Local OpenCV inpainting runs offline. Optional instruction is stored for future LLM/API backends."
                )
                with gr.Row():
                    x1 = gr.Number(label="x1", value=0, precision=0)
                    y1 = gr.Number(label="y1", value=0, precision=0)
                    x2 = gr.Number(label="x2", value=100, precision=0)
                    y2 = gr.Number(label="y2", value=50, precision=0)
                instruction = gr.Textbox(
                    label="Instruction (optional)",
                    placeholder="e.g. remove the logo, keep the sky natural",
                )
                method = gr.Radio(choices=["telea", "ns"], value="telea", label="Inpaint method")
                btn_inpaint = gr.Button("Remove region", variant="primary")

                with gr.Accordion("Image history", open=False):
                    img_hist = gr.Markdown("No image history yet.")
                    btn_img_hist = gr.Button("Refresh history")

                btn_exif.click(process_exif, inputs=[img_in], outputs=[exif_out, img_out])
                btn_strip.click(do_strip_exif, inputs=[img_in], outputs=[img_out, img_status])
                btn_update.click(
                    do_update_exif,
                    inputs=[img_in, artist, copyright_, description],
                    outputs=[img_out, img_status],
                )
                btn_inpaint.click(
                    do_inpaint,
                    inputs=[img_in, x1, y1, x2, y2, instruction, method],
                    outputs=[img_out, img_status],
                )
                btn_img_hist.click(load_history_images, outputs=[img_hist])

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

### Images
- Full **EXIF / metadata** view, edit, and strip (offline).
- **Region inpainting** for visible logos/watermarks (local OpenCV).

### Limitations (honest)
- Without the provider’s key you cannot perfectly color tokens for proprietary schemes.
- Classic KGW detection assumes you know or can guess scheme parameters.
- Paraphrase may slightly change wording; always review cleaned text.
- Local inpainting is basic; high-quality diffusion inpainting is optional later.

### Config (env)
| Variable | Purpose |
|----------|---------|
| `FAKU_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` / `XAI_API_KEY` | Neutralization |
| `FAKU_BASE_URL` / `OPENAI_BASE_URL` | API base URL |
| `FAKU_MODEL` | Model name |

**Tagline:** *Detect. Highlight. Neutralize.*
                    """
                )

        gr.Markdown(
            "<center><small>fak-u-watermark · offline detection · local history in ~/.faku/</small></center>"
        )

    return demo


def main():
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
