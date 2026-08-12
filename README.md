# fak-u-watermark

**Strip the mark. Keep the meaning.**

Detect, highlight, and neutralize AI watermarks in **text** and **images**.

| Area | What you get |
|------|----------------|
| **Text** | KGW / Unigram detection · soft yellow highlights · density heatmap · paraphrase / targeted / adaptive neutralize · copy & export · batch · history |
| **Images** | EXIF view/edit/strip · C2PA marker detect/strip · brush inpaint (local or API) · history |
| **Keys** | Manual API keys in the UI **Settings** tab or CLI — stored in `~/.faku/settings.json` |

> Identify → Show → Remove. Offline detection & EXIF; neutralization needs an API key only for paraphrase modes.

---

## Requirements

- **[uv](https://docs.astral.sh/uv/)** (Python package manager)
- Python **3.11+** (uv can install it for you)
- macOS / Linux / Windows
- Optional: API key for paraphrase / adaptive neutralize / API inpaint

Install uv if needed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or Homebrew
brew install uv
```

---

## Install

```bash
git clone https://github.com/<you>/fak-u-watermark.git
cd fak-u-watermark

# Create env + install project + deps from pyproject.toml / uv.lock
uv sync

# Dev tools (pytest, ruff)
uv sync --extra dev
```

`uv` manages its own `.venv` under the project — you do **not** need `python -m venv` or `pip`.

Run commands with `uv run` (no manual activate required):

```bash
uv run python -m ui.app
uv run faku settings show
uv run pytest -q
```

Optional: activate the env if you prefer a classic shell:

```bash
source .venv/bin/activate   # created by uv sync
```

First text analysis downloads the **GPT-2** tokenizer from Hugging Face.

---

## Add API keys (manual)

You can set keys in **three** ways. Priority when calling the LLM:

1. **UI override fields** (per request, optional)
2. **Saved settings** → `~/.faku/settings.json`
3. **Environment variables**

### Option A — UI Settings tab (recommended)

1. Start the UI: `uv run python -m ui.app`
2. Open **http://127.0.0.1:7860**
3. Go to the **Settings** tab
4. Choose a provider preset (OpenAI / DeepSeek / xAI / Custom)
5. Paste your **API key**
6. Confirm **Base URL** and **Model**
7. Click **Save settings**

Keys are written to:

```text
~/.faku/settings.json    # mode 600 when possible
```

Example file:

```json
{
  "api_key": "sk-...",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o-mini",
  "inpaint_model": "dall-e-2",
  "inpaint_base_url": "",
  "temperature": 0.7,
  "provider": "openai",
  "extra": {}
}
```

### Option B — CLI

```bash
# Show (key masked)
uv run faku settings show

# Save key + provider preset
uv run faku settings set \
  --api-key "sk-..." \
  --provider openai

# Custom endpoint (any OpenAI-compatible server)
uv run faku settings set \
  --api-key "sk-..." \
  --base-url "https://api.deepseek.com" \
  --model "deepseek-chat" \
  --provider custom

# Clear key
uv run faku settings set --clear-key

# Reveal full key (careful)
uv run faku settings show --reveal
```

### Option C — Environment variables

```bash
export FAKU_API_KEY="sk-..."                 # preferred
# or: OPENAI_API_KEY / DEEPSEEK_API_KEY / XAI_API_KEY

export FAKU_BASE_URL="https://api.openai.com/v1"
export FAKU_MODEL="gpt-4o-mini"

# Optional inpaint overrides
export FAKU_INPAINT_MODEL="dall-e-2"
export FAKU_INPAINT_BASE_URL="https://api.openai.com/v1"
```

### Provider examples

| Provider | Base URL | Example model |
|----------|----------|---------------|
| **OpenAI** | `https://api.openai.com/v1` | `gpt-4o-mini` |
| **DeepSeek** | `https://api.deepseek.com` | `deepseek-chat` |
| **xAI Grok** | `https://api.x.ai/v1` | `grok-2-latest` |
| **Custom proxy** | your OpenAI-compatible URL | depends |

---

## Run the app

### Gradio UI (easiest)

```bash
uv run python -m ui.app
# Open this URL in the browser (not 0.0.0.0):
# → http://127.0.0.1:7860
```

> **Note:** The server listens on `0.0.0.0` (all interfaces). Use **`http://127.0.0.1:7860`** or **`http://localhost:7860`**. Opening `http://0.0.0.0:7860` often shows a blank page (`about:blank`) in Safari and some other browsers.

**Tabs**

| Tab | Actions |
|-----|---------|
| **Settings** | Save / clear API keys, provider presets |
| **Text** | Analyze, heatmap, paraphrase / targeted / adaptive, copy, export, batch, history |
| **Images** | EXIF, C2PA, brush inpaint, download, history |
| **About** | How it works |

### FastAPI

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Docs: http://127.0.0.1:8000/docs
# Health: http://127.0.0.1:8000/health
```

Useful endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/text/analyze` | Detect + highlights + density |
| `POST` | `/api/text/neutralize` | Paraphrase |
| `POST` | `/api/text/neutralize/targeted` | Offline green→red |
| `POST` | `/api/text/neutralize/adaptive` | Loop until z drops |
| `POST` | `/api/text/batch` | Multi-item analyze |
| `GET/PUT` | `/api/text/settings` | Read/update saved keys |
| `POST` | `/api/images/exif` | Metadata |
| `POST` | `/api/images/strip-exif` | Strip metadata |
| `POST` | `/api/images/c2pa` | Detect C2PA markers |
| `POST` | `/api/images/inpaint` | Region inpaint |

### CLI

```bash
# Analyze
uv run faku text analyze "Your text here" --preset kirchenbauer_default
uv run faku text analyze -f sample.txt --html out.html

# Batch
uv run faku text batch ./notes/*.txt

# Neutralize (uses saved settings / env)
uv run faku text neutralize -f sample.txt -o cleaned.txt --style subtle

# Offline targeted (needs correct watermark key)
uv run faku text targeted -f sample.txt --key 15485863 -o cleaned.txt

# Adaptive paraphrase
uv run faku text adaptive -f sample.txt --max-rounds 3 --target-z 4

# Images
uv run faku image exif photo.jpg
uv run faku image strip photo.jpg -o clean.png
uv run faku image c2pa photo.jpg
uv run faku image c2pa photo.jpg --strip -o noc2pa.png

# Settings
uv run faku settings set --api-key sk-... --provider openai
uv run faku settings show
```

---

## Deploy

### Local (laptop)

```bash
uv sync
uv run python -m ui.app                              # UI :7860
# and/or
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Data lives on the machine:

```text
~/.faku/settings.json   # API keys
~/.faku/history.db      # analysis history
```

### Docker Compose

```bash
# Optional: pass keys at runtime
export FAKU_API_KEY=sk-...
export FAKU_BASE_URL=https://api.openai.com/v1
export FAKU_MODEL=gpt-4o-mini

docker compose up --build
```

| Service | Port |
|---------|------|
| Gradio UI | **7860** |
| FastAPI | **8000** |

Compose mounts a volume for `/root/.faku` so settings/history persist.

### Single container

```bash
docker build -t faku .
docker run --rm -p 7860:7860 \
  -e FAKU_API_KEY=sk-... \
  -e FAKU_BASE_URL=https://api.openai.com/v1 \
  -e FAKU_MODEL=gpt-4o-mini \
  -v faku-data:/root/.faku \
  faku uv run python -m ui.app
```

### VPS / Railway / Fly / Render (outline)

1. Build from this repo (`Dockerfile` or `uv sync` on the host).
2. Set env: `FAKU_API_KEY`, `FAKU_BASE_URL`, `FAKU_MODEL`.
3. Expose port **7860** (UI) and/or **8000** (API).
4. Persist `~/.faku` (or `/root/.faku`) if you want history + saved keys.
5. Put TLS in front (Caddy / nginx / platform HTTPS).
6. **Do not** commit `settings.json` or keys to git.

Example process (systemd-style):

```bash
WorkingDirectory=/opt/fak-u-watermark
Environment=FAKU_API_KEY=sk-...
ExecStart=/usr/local/bin/uv run --directory /opt/fak-u-watermark python -m ui.app
```

---

## Usage walkthrough

### Text — detect & show

1. Paste text (or upload `.txt` / `.md`).
2. Pick preset (e.g. `kirchenbauer_default`) or set scheme / gamma / key.
3. Click **Analyze**.
4. Read verdict + z-score; yellow marks = green-list tokens.
5. Check the **density heatmap** for high-signal regions.

### Text — remove

| Mode | Network? | When to use |
|------|----------|-------------|
| **Neutralize (paraphrase)** | Yes | Unknown or proprietary schemes; default **subtle** |
| **Targeted green→red** | No | You know the watermark **key**; synonym swaps onto red list |
| **Adaptive** | Yes | Keep paraphrasing until z-score &lt; threshold |

Then **Copy**, **Export**, or **Re-analyze cleaned** to verify the signal dropped.

### Images

1. Upload image → **Read EXIF** / **Strip metadata**.
2. **Detect C2PA** / **Strip C2PA** for Content Credentials markers.
3. Paint the logo on the mask editor → **Remove painted region** (`telea` offline, `api`/`auto` with key).

---

## Tests

```bash
uv sync --extra dev
uv run pytest -q
```

---

## Project layout

```text
packages/
  watermark_core/   # detect, density, neutralize, targeted, adaptive, settings, history
  image_tools/      # EXIF, C2PA, inpaint
api/                # FastAPI
ui/                 # Gradio
cli/                # faku CLI
docs/architecture.md
design_plan.md
```

---

## Limitations (honest)

- Without the right secret key you cannot perfectly color proprietary schemes.
- Targeted synonym coverage is a built-in dictionary + red-list check — not perfect prose.
- C2PA support is **marker detect + re-encode strip**, not full cryptographic verification.
- API inpaint depends on provider `images/edits` support.
- Always review cleaned text before publishing.

---

## License

MIT
