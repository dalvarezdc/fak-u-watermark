# fak-u-watermark
## Complete Product & Technical Planning Document

**Version:** 1.4  
**Date:** 2026-08-12  
**Status:** Ready for implementation  
**Target:** Single developer / small team → production-ready MVP in 4–6 weeks  

**Changelog 1.4:**  
- Design decisions locked:  
  - Default tokenizer = GPT-2 (open, free, classic for watermark research) + selectable open alternatives  
  - LLM providers: prioritize OpenAI-compatible (DeepSeek, Grok, OpenAI) + Claude/Gemini later  
  - Maximize offline capability; only neutralization & advanced inpainting require network  
  - Default paraphrase style = **subtle** (light wording changes, preserve structure & tone)

**Changelog 1.3:**  
- Locked in **Python-first** architecture (FastAPI + pure Python packages)  
- Recommended starting UI: Gradio or Streamlit for speed, with optional later upgrade to Next.js  
- Clarified project structure around `watermark_core` and `image_tools` packages  

**Changelog 1.2:**  
- Text cleaned view: mandatory Copy + Export options  
- Added full **Image** module: EXIF view/edit/strip + region selection + LLM-guided visual watermark removal  

**Changelog 1.1:**  
- Clarified core loop: Identify → Show → Remove  
- Removed any special-casing or modes for specific user groups  
- Elevated History and Manual editing of cleaned output to P0 MVP features

---

## Branding

| Item | Value |
|------|-------|
| **App name** | fak-u-watermark |
| **Short name** | faku / Faku |
| **Tagline ideas** | "Strip the mark. Keep the meaning." · "Yellow means marked. We make it plain." · "Detect. Highlight. Neutralize." |
| **Repo / package name** | `fak-u-watermark` |
| **Tone** | Direct, slightly irreverent, technically honest |

The name signals the core action (removing/faking out the watermark) without pretending to be a corporate safety tool.

---

## Design Decisions (Locked)

### 1. Tokenizer
- **Default:** GPT-2 (`gpt2` from Hugging Face / transformers)  
  - Open, free, no license issues  
  - Classic choice in original watermark papers (Kirchenbauer et al.)
- User can also select other open tokenizers (e.g. Llama-2/3, Mistral, Qwen) from a short list.
- No closed proprietary tokenizers required for core detection.

### 2. LLM Providers for Neutralization
**Priority order for MVP:**
1. **OpenAI-compatible endpoints** (easiest unified client)  
   - DeepSeek  
   - Grok (xAI)  
   - OpenAI
2. Anthropic Claude
3. Google Gemini

Implementation: single abstraction layer that accepts an OpenAI-compatible base URL + API key. Claude/Gemini can be added via their official SDKs or compatible proxies later.

### 3. Offline Capability
**Make as much as possible offline:**

| Feature                        | Offline? | Notes |
|--------------------------------|----------|-------|
| Text detection & highlighting  | Yes      | Pure Python + local tokenizer |
| Statistics & z-score           | Yes      | Local |
| History                        | Yes      | Local storage / SQLite |
| EXIF read / edit / strip       | Yes      | Local |
| Save cleaned image             | Yes      | Local |
| Basic region inpainting        | Optional | Prefer local model when available |
| Text neutralization (paraphrase) | No*    | Requires LLM (API or local model) |
| High-quality image inpainting  | No*      | Usually needs external model |

*Optional local LLM / local inpainting model support is desirable but not required for MVP.

### 4. Default Paraphrase Style
- **Subtle / light** by default.
- Changes should be minimal: synonym swaps, light rephrasing, preserve sentence structure, tone, and meaning as much as possible.
- Goal is to disrupt the statistical watermark signal without rewriting the text aggressively.

**Default system prompt (subtle):**
```
You are a careful editor. Make only subtle wording changes to the following text. 
Keep the exact same meaning, facts, tone, and overall structure. 
Prefer simple synonym replacements and minor phrase adjustments. 
Do not restructure sentences or add/remove information unless necessary.
Return only the revised text.
```

Users can later choose stronger modes if desired.

---

## 1. Vision & Purpose

Build a clean, focused web application that handles both **text** and **images**:

### Text
1. Accept any text.
2. **Identify** statistical AI watermarks (primarily KGW / green-red list family and compatible schemes).
3. **Show** the tokens that contribute to the watermark signal using a soft **yellow background**.
4. Display clear statistical analysis (z-score, green fraction, confidence).
5. **Remove** the watermark (neutralization) while preserving meaning as much as possible.
6. Let the user **manually edit** the cleaned output.
7. Provide **Copy** and **Export** options for the cleaned text.
8. Keep a **history** of past analyses and cleaned versions.

### Images
1. Upload an image.
2. **Read and show** EXIF / metadata.
3. **Modify or delete** metadata (full strip or selective).
4. **Save** the cleaned image file.
5. Allow the user to **select a region** that contains a visible watermark.
6. **Remove** the selected region using LLM / inpainting instructions.
7. Keep history of processed images.

**Core philosophy:**  
Identify. Show. Remove.  
Same tool for text and images. No special modes for specific user groups.  
Be honest about limitations.

---

## 2. Goals & Non-Goals

### Goals (MVP)

**Text**
- Accurate reconstruction and visualization of green-list tokens when the watermark configuration is known or can be assumed.
- Soft yellow highlighting of watermark-contributing tokens.
- Side-by-side or sequential “Original (highlighted)” → “Cleaned” view.
- User can **manually edit** the cleaned text.
- Prominent **Copy** button and **Export** options (plain text + annotated HTML).
- Local history of past analyses and cleaned versions.
- Local or API-based paraphrasing for neutralization.

**Images**
- Upload image and display full EXIF / metadata in a readable panel.
- Edit individual metadata fields or strip all metadata.
- Save the cleaned image file.
- Interactive region selection (brush / rectangle) for visible watermarks.
- Remove selected region via LLM instruction + inpainting.
- History of processed images.

**General**
- Runs comfortably on a laptop for personal use.
- Same tool for text and images — no special modes for specific user groups.

### Non-Goals (MVP)
- Special modes or UX paths for specific user groups.
- Detecting every proprietary watermark without the provider’s key/detector.
- Perfect semantic / visual reconstruction after removal (we aim for high quality, not magic).
- Real-time collaborative editing.
- Mobile-native apps (responsive web is enough).
- Training new watermark detectors from scratch.
- Automatic detection of invisible pixel-level watermarks (SynthID etc.) in images (can be Phase 2/3).

---

## 3. User Personas

The tool is general-purpose. No special modes per persona.

Typical users include:

1. **Researcher / AI Safety person**  
   Wants to inspect and understand watermark signals in model outputs.

2. **Content creator / editor**  
   Receives text that might be watermarked and wants to clean it before publishing.

3. **Developer / tool builder**  
   Needs a clear reference implementation and API for integration.

4. **Anyone** who simply wants to identify, visualize, and remove watermarks from text or images.

---

## 4. Core Features

### 4.1 MVP Features (Phase 1) — Text

| Feature | Description | Priority |
|---------|-------------|----------|
| Text input | Large textarea + file upload (.txt, .md) | P0 |
| Watermark config | Select scheme + optional secret key / seed + gamma | P0 |
| Detection | Tokenize → reconstruct green lists → label tokens | P0 |
| Yellow highlighting | Soft yellow background on watermark-contributing tokens | P0 |
| Statistics panel | Green fraction, z-score, p-value, verdict badge | P0 |
| Toggle highlights | On/off switch for yellow marks | P0 |
| Neutralize (Paraphrase) | Send text to LLM with strong rewrite prompt | P0 |
| Before / After view | Side-by-side or stacked comparison | P0 |
| Manual edit of cleaned text | User can freely edit the neutralized output | P0 |
| **Copy button** | One-click copy of cleaned text to clipboard | P0 |
| **Export** | Download plain text and/or annotated HTML | P0 |
| History | Local history of past analyses and cleaned versions | P0 |
| Basic presets | Common configs (Kirchenbauer default, Unigram, etc.) | P1 |

### 4.2 MVP Features (Phase 1) — Images

| Feature | Description | Priority |
|---------|-------------|----------|
| Image upload | Drag & drop or file picker (JPG, PNG, WebP, etc.) | P0 |
| EXIF / Metadata viewer | Readable panel showing all available metadata | P0 |
| Metadata edit / strip | Edit individual fields or one-click full strip | P0 |
| Save cleaned image | Download the image with modified/stripped metadata | P0 |
| Region selection | Brush or rectangle tool to mark visible watermark area | P0 |
| LLM-guided removal | Send selected region + instruction to inpainting model | P0 |
| Image history | Local history of processed images | P0 |

### 4.3 Phase 2 Features

**Text**
- Targeted green→red synonym substitution (when key is known)
- Sliding-window density heatmap
- Multiple algorithm support (KGW, Unigram, SWEET, basic SynthID-Text)
- Stronger adaptive neutralization modes

**Images**
- Better automatic detection of common AI watermarks (Gemini sparkle, etc.)
- Support for C2PA / Content Credentials viewing and stripping
- Higher-quality local inpainting models

**General**
- API endpoint for programmatic use
- Dark / light theme
- Optional “How it works” explainer
- Batch processing

### 4.4 Phase 3 (Later)

- Browser extension
- Custom detector training UI (advanced)
- Invisible pixel-level watermark handling (SynthID-style)
- Video support

---

## 5. User Flow (Happy Path)

1. User pastes or uploads text.
2. Selects watermark scheme (or leaves “Auto / Common presets”).
3. Clicks **Analyze**.
4. App shows:
   - Original text with yellow highlights on signal tokens.
   - Statistics card (z-score, verdict, green %).
5. User can toggle highlights.
6. User clicks **Neutralize Watermark**.
7. App shows loading state → returns cleaned version.
8. User compares original vs cleaned and exports either or both.

Secondary flows:
- Change config → re-analyze.
- Hover on yellow token → tooltip “Contributes to watermark signal”.
- Copy cleaned text with one click.

---

## 6. Technical Architecture

**Primary approach: Python-first**

Because the main developer works mostly in Python, the architecture prioritizes pure Python packages + FastAPI. Frontend can start simple (Gradio / Streamlit) and be upgraded later if needed.

### High-level

```
┌──────────────────────────────────────────────────────────────┐
│  Frontend options                                            │
│  A) Gradio or Streamlit (fastest MVP, almost pure Python)    │
│  B) Next.js (higher polish, later)                           │
│                                                              │
│  Text: input, yellow highlights, stats, edit, Copy, Export │
│  Image: upload, EXIF panel, region select, inpaint           │
│  Shared: History                                             │
└────────────────────────────┬─────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼─────────────────────────────────┐
│                 FastAPI Backend (Python)                     │
│  Text: /analyze, /neutralize                               │
│  Image: /exif, /strip-exif, /inpaint                         │
└──────────────┬──────────────────────┬────────────────────────┘
               │                      │
     ┌─────────▼─────────┐   ┌────────▼──────────┐
     │  watermark_core   │   │  image_tools      │
     │  (pure Python)    │   │  (pure Python)    │
     │  - Tokenizer      │   │  - EXIF read/     │
     │  - Green list     │   │    write/strip    │
     │  - Z-test         │   │  - Inpainting     │
     │  - Highlighting   │   │    wrapper        │
     └───────────────────┘   └───────────────────┘
```

### Recommended Stack (Python-first)

**Core language**
- Python 3.11+

**Backend**
- **FastAPI** (main API)
- Uvicorn

**Frontend (choose one to start)**
- **Gradio** or **Streamlit** → fastest path, stays in Python
- Later optional upgrade to Next.js + React for higher polish

**Text detection**
- Custom pure-Python implementation inspired by MarkLLM / Kirchenbauer
- `transformers` tokenizer only when needed

**Image tools**
- EXIF: `exiftool` (most complete) via subprocess, or pure-Python (`piexif`, `Pillow`)
- Visual removal: start with an external inpainting API; later optional local LaMA / diffusion

**Neutralization (Text)**
- Default: **subtle** paraphrase via OpenAI-compatible API (DeepSeek, Grok, OpenAI first)
- Optional stronger mode
- Phase 2: local model support + targeted substitution

**History & storage**
- SQLite or simple JSON/file-based history
- Local file downloads for cleaned outputs

**Deployment**
- Local first (`uvicorn` + Gradio/Streamlit)
- Later: single VPS or Railway / Fly.io / Render
- Optional Docker Compose

---

## 7. Detailed Component Design

### 7.1 Detection Engine (Python)

```python
class WatermarkAnalyzer:
    def __init__(self, scheme: str, gamma: float, key: str | None, tokenizer_name: str):
        ...

    def analyze(self, text: str) -> AnalysisResult:
        """
        Returns:
        - list of TokenInfo (text, is_signal, start, end)
        - Statistics
        - optional density map
        """
```

Key methods:
- `_get_green_list(prev_token_id) -> set[int]`
- `_score_sequence(token_ids) -> (green_count, total, z_score)`
- `get_highlighted_spans(text) -> list[Span]`

### 7.2 Frontend Highlighter

Component: `<WatermarkHighlighter tokens={tokens} showHighlights={bool} />`

- Renders plain text with `<mark class="watermark-signal">` around signal tokens.
- CSS:
  ```css
  .watermark-signal {
    background-color: #FEF08A; /* soft yellow */
    border-radius: 2px;
    padding: 0 1px;
  }
  ```
- Optional stronger yellow for high-density windows.

### 7.3 Statistics Panel

- Verdict badge: “Watermark Detected” (red/orange) / “No significant signal” (gray) / “Uncertain”
- Metrics: Green fraction, Z-score, P-value, Tokens analyzed
- Short explanation tooltip

### 7.4 Neutralization Service (Text)

**Mode 1 – Subtle Paraphrase (MVP default)**
```
System prompt:
You are a careful editor. Make only subtle wording changes to the following text. 
Keep the exact same meaning, facts, tone, and overall structure. 
Prefer simple synonym replacements and minor phrase adjustments. 
Do not restructure sentences or add/remove information unless necessary.
Return only the revised text.
```

**Mode 2 – Stronger paraphrase** (optional user toggle)  
More aggressive rewording while still preserving meaning.

**Mode 3 – Targeted (Phase 2)**  
Only when key is known: replace signal tokens with red-list synonyms while preserving grammar.

### 7.5 Cleaned Text Actions

- Large editable textarea with the neutralized result
- **Copy** button (one-click clipboard)
- **Export** dropdown: Plain text (.txt) / Annotated HTML
- Optional “Re-analyze” after manual edits

### 7.6 Image Metadata Module

- Use `exiftool` (or pure-Python fallback) to extract full metadata
- Display in a clean, searchable key-value panel
- Actions:
  - Edit individual fields
  - “Strip all metadata” one-click
  - Save as new file (or overwrite with confirmation)

### 7.7 Image Visual Watermark Removal

1. User uploads image
2. User draws a mask (brush or rectangle) over the visible watermark
3. Optional short instruction (e.g. “remove the logo, keep the sky natural”)
4. Backend sends image + mask + instruction to an inpainting model (LaMA, diffusion inpainting API, or similar)
5. Return cleaned image for preview + download
6. Store in history

---

## 8. Project Structure (Suggested)

```
fak-u-watermark/
├── packages/
│   ├── watermark_core/             # Text detection (pure Python)
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   ├── schemes/
│   │   │   ├── kgw.py
│   │   │   ├── unigram.py
│   │   │   └── base.py
│   │   ├── visualization.py
│   │   └── tests/
│   └── image_tools/                # EXIF + inpainting helpers
│       ├── __init__.py
│       ├── exif.py
│       ├── inpaint.py
│       └── tests/
├── api/                            # FastAPI application
│   ├── main.py
│   ├── routes/
│   │   ├── text.py
│   │   └── images.py
│   └── deps.py
├── ui/                             # Gradio or Streamlit app (MVP)
│   └── app.py
├── cli/                            # Optional CLI for testing
│   └── main.py
├── docker-compose.yml
├── pyproject.toml / requirements.txt
├── README.md
└── docs/
    └── architecture.md
```

---

## 9. Implementation Roadmap

### Phase 0 – Foundation (3–5 days)
- [ ] Create project structure (`packages/`, `api/`, `ui/`)
- [ ] Implement pure Python `WatermarkAnalyzer` for classic KGW
- [ ] Unit tests for green-list reconstruction and z-score
- [ ] Basic EXIF read / strip helpers
- [ ] Simple CLI for text highlighting and EXIF inspection

### Phase 1 – MVP (FastAPI + Gradio/Streamlit) (2–3 weeks)
**Text**
- [ ] FastAPI `/analyze` and `/neutralize` endpoints
- [ ] Yellow highlighting data + statistics
- [ ] Paraphrase neutralization via LLM API
- [ ] Manual edit + **Copy** + **Export**
- [ ] Before/After view
- [ ] Local history

**Images**
- [ ] Image upload + EXIF viewer / editor
- [ ] Strip metadata + save cleaned image
- [ ] Region selection (brush or rectangle)
- [ ] Basic LLM / inpainting removal of selected region
- [ ] Image history

**UI**
- [ ] Gradio or Streamlit interface covering both text and image tabs
- [ ] Clean, functional layout

### Phase 2 – Robustness & UX (2–3 weeks)
- [ ] Multiple text schemes + stronger neutralization modes
- [ ] Better inpainting quality / model options
- [ ] C2PA / Content Credentials viewing
- [ ] Density heatmap, API keys management, etc.

### Phase 3 – Polish & Distribution
- [ ] Public demo deployment
- [ ] Documentation site
- [ ] Optional browser extension
- [ ] Batch processing
- [ ] Performance optimizations (streaming, large texts)

---

## 10. Key Technical Decisions & Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| Highlight color | Soft yellow | Neutral, accessible, familiar “highlight” metaphor |
| Green/Red internal | Keep | Matches academic literature and existing libraries |
| Primary neutralization | LLM paraphrase | Most robust against unknown or proprietary schemes |
| Core language | Python | Best ecosystem for tokenizers + existing watermark code |
| Frontend | Next.js + Tailwind | Fast development, excellent DX, easy deployment |
| Config exposure | Advanced panel | Power users can experiment; beginners use presets |

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|----------|
| User expects detection of Claude / ChatGPT watermarks | Clear disclaimer + educational mode explaining limitations |
| Paraphrase changes meaning | Strong system prompt + optional “strict” vs “creative” modes + human review suggestion |
| Performance on very long texts | Chunking + progressive analysis |
| Key unknown → cannot color tokens accurately | Fall back to “statistical AI style detection” + warning |
| Color accessibility | Provide high-contrast mode and pattern (underline) alternative |

---

## 12. Success Metrics (MVP)

- User can paste text and see yellow highlights in < 3 seconds for 500-token documents.
- Neutralized text reduces z-score below threshold in > 85% of classic KGW cases.
- Export works reliably.
- Positive feedback on clarity of the yellow highlighting vs old green/red.

---

## 13. Open Questions for Implementation

1. Do we hard-code a default tokenizer (e.g. GPT-2 / Llama) or let user choose?
2. Which LLM provider(s) do we support first for neutralization?
3. Should the app work fully offline (local model) as an optional mode?
4. How aggressive should the default paraphrase prompt be?

---

## 14. Next Immediate Steps for Coding Agent

1. Scaffold the Python `watermark-core` package with KGW analyzer.
2. Write comprehensive unit tests for green-list + z-score.
3. Create a minimal FastAPI or Next.js API route that returns the structured `AnalysisResult`.
4. Build the yellow highlighter React component.
5. Wire the basic UI flow: Input → Analyze → Highlighted view + Stats.

---

**End of Planning Document**

This document is intentionally detailed so that a coding agent (or human developer) can implement the system without needing further high-level clarification. All major architectural and UX decisions are recorded.

