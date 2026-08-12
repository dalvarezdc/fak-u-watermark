# fak-u-watermark

**Strip the mark. Keep the meaning.**

Detect, highlight, and neutralize AI watermarks in **text** and **images**.

- **Text:** KGW / Unigram green-list detection · soft yellow highlights · z-score stats · subtle paraphrase neutralization · copy/export · local history  
- **Images:** EXIF view / edit / strip · region inpainting for visible watermarks · history  

> Identify → Show → Remove. Same tool for everyone. Honest about limitations.

## Quick start

```bash
# Python 3.11+
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Gradio UI (recommended MVP front-end)
python -m ui.app
# → http://127.0.0.1:7860

# FastAPI
uvicorn api.main:app --reload --port 8000
# → http://127.0.0.1:8000/docs

# CLI
python -m cli.main text analyze "Your text here" --preset kirchenbauer_default
python -m cli.main image exif photo.jpg
```

### Neutralization (optional API key)

```bash
export FAKU_API_KEY=sk-...           # or OPENAI_API_KEY / DEEPSEEK_API_KEY / XAI_API_KEY
export FAKU_BASE_URL=https://api.openai.com/v1
export FAKU_MODEL=gpt-4o-mini
```

Works with any OpenAI-compatible endpoint (OpenAI, DeepSeek, xAI Grok, local proxies, …).  
Default paraphrase style is **subtle** (light wording changes only).

## Project layout

```
packages/
  watermark_core/   # detection, z-score, highlights, neutralize, history
  image_tools/      # EXIF + OpenCV inpaint
api/                # FastAPI
ui/                 # Gradio app
cli/                # command-line helpers
docs/architecture.md
design_plan.md
```

## Text detection (offline)

1. Tokenize with **GPT-2** (or other open tokenizers).  
2. Reconstruct the **green list** from previous token + secret key (KGW).  
3. Score green fraction → **z-score** / p-value / verdict.  
4. Highlight signal tokens with soft yellow (`#FEF08A`).

Presets: `kirchenbauer_default` (γ=0.25), `kirchenbauer_hard` (γ=0.5), `unigram_default`.

## Images

- Read full metadata (Pillow + piexif).  
- Strip all EXIF or edit Artist / Copyright / Description.  
- **Brush** (or rectangle) over a logo/watermark → local **OpenCV** inpainting (offline).  
- Optional **API** inpaint (`method=api|auto`) via OpenAI-compatible `images/edits`.  
- Download cleaned images; restore past jobs from history.

## Text UX

- Soft yellow highlights + statistics  
- Neutralize (subtle/strong) · **Copy** · Export `.txt` / annotated HTML  
- **Re-analyze cleaned** text after edits  
- Before/After panel · History restore

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

First run downloads the GPT-2 tokenizer from Hugging Face.

## Docker

```bash
docker compose up --build
# UI :7860  ·  API :8000
```

## Limitations

- Proprietary schemes without a known key cannot be perfectly reconstructed.  
- Neutralization needs a network LLM (or your own compatible endpoint).  
- Local inpainting is basic; instruction text is stored for future API backends.  
- Always review cleaned text before publishing.

## License

MIT
