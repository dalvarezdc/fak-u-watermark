# Architecture

## Overview

```
UI (Gradio)  ──┐
CLI          ──┼──►  watermark_core  /  image_tools  (pure Python packages)
FastAPI      ──┘
```

Detection, EXIF, history, and basic inpainting run **offline**.  
Text neutralization calls an **OpenAI-compatible** chat API (DeepSeek / Grok / OpenAI).

## Packages

### `watermark_core`

| Module | Role |
|--------|------|
| `analyzer.py` | `WatermarkAnalyzer` — tokenize, score, return `AnalysisResult` |
| `schemes/kgw.py` | Kirchenbauer green-red list (hash prev token → PRNG → green set) |
| `schemes/unigram.py` | Context-independent green list |
| `tokenizer.py` | GPT-2 (default) via Hugging Face |
| `visualization.py` | Soft yellow HTML highlights (`#FEF08A`) |
| `neutralize.py` | Subtle/strong paraphrase via chat completions |
| `history.py` | SQLite store in `~/.faku/history.db` |

### `image_tools`

| Module | Role |
|--------|------|
| `exif.py` | Read / update / strip metadata (Pillow + piexif) |
| `inpaint.py` | OpenCV Telea/NS region inpainting |

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/text/analyze` | Detect + highlight data + stats |
| POST | `/api/text/neutralize` | LLM paraphrase |
| POST | `/api/images/exif` | Upload → metadata JSON |
| POST | `/api/images/strip-exif` | Upload → cleaned file |
| POST | `/api/images/inpaint` | Image + mask/rect → inpainted PNG |

## Z-score

For \(n\) scored tokens, green count \(g\), green fraction \(\gamma\):

\[
z = \frac{g - \gamma n}{\sqrt{n \gamma (1-\gamma)}}
\]

Default threshold \(z \ge 4\) → **Watermark Detected**.
