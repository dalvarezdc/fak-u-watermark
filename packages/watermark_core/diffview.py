"""Side-by-side old/new compare: watermark yellow + word-level diff."""

from __future__ import annotations

import difflib
import html
import re
from collections.abc import Iterable, Sequence
from itertools import pairwise

from .schemes.base import TokenInfo
from .visualization import HIGHLIGHT_CSS

_WORD_RE = re.compile(r"\s+|\S+")

PANE_CSS = f"""
{HIGHLIGHT_CSS}
.faku-legend {{
  display: flex; flex-wrap: wrap; gap: 0.65rem 1.1rem;
  font-size: 0.8rem; color: #57534e; margin: 0.15rem 0 0.75rem;
}}
.faku-swatch {{
  display: inline-block; width: 0.75rem; height: 0.75rem;
  border-radius: 2px; margin-right: 0.35rem; vertical-align: -1px;
  border: 1px solid rgba(28, 25, 23, 0.12);
}}
.faku-compare {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.85rem;
}}
@media (max-width: 800px) {{ .faku-compare {{ grid-template-columns: 1fr; }} }}
.faku-compare-col {{
  background: #fffefb; border: 1px solid #e7e5e4; border-radius: 12px;
  overflow: hidden; min-width: 0;
}}
.faku-compare-col header {{
  font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.05em; color: #78716c;
  padding: 0.55rem 0.9rem; background: #fafaf9; border-bottom: 1px solid #e7e5e4;
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.35rem 0.75rem;
}}
.faku-compare-col header .side {{ color: #44403c; }}
.faku-compare-col .watermark-output {{
  border: none; border-radius: 0; box-shadow: none;
  max-height: min(36rem, 58vh); margin: 0;
}}
.diff-old {{
  background: #fecaca;
  border-radius: 2px;
  padding: 0.02em 0.05em;
}}
.diff-new {{
  background: #bbf7d0;
  border-radius: 2px;
  padding: 0.02em 0.05em;
}}
.watermark-signal.diff-old {{
  background: #fed7aa;
  box-shadow: inset 0 -2px 0 #ef4444;
}}
.watermark-signal.diff-new {{
  background: #d9f99d;
  box-shadow: inset 0 -2px 0 #16a34a;
}}
.faku-pane-empty {{
  color: #a8a29e; font-style: italic; padding: 1.1rem 1rem; margin: 0;
}}
.faku-compare-col .verdict-pill {{
  font-weight: 750; font-size: 0.78rem; padding: 0.18rem 0.55rem;
  border-radius: 999px; text-transform: none; letter-spacing: 0;
}}
.faku-compare-col .verdict-detected {{ background: #fecaca; color: #7f1d1d; }}
.faku-compare-col .verdict-uncertain {{ background: #fde68a; color: #78350f; }}
.faku-compare-col .verdict-none {{ background: #e7e5e4; color: #44403c; }}
.faku-compare-col .verdict-meta {{ color: #78716c; font-size: 0.75rem; font-weight: 500; text-transform: none; letter-spacing: 0; }}
""".strip()

_SPAN_TITLES = {
    "watermark-signal": "Watermark signal (green-list token)",
    "diff-old": "Removed or replaced in the cleaned text",
    "diff-new": "Added or replaced versus the original",
}


def word_diff_ranges(old: str, new: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Character ranges that differ: (deleted/replaced in old, inserted/replaced in new)."""
    old = old or ""
    new = new or ""
    if old == new or not old or not new:
        return [], []
    # SequenceMatcher is quadratic in token count — skip on book-length inputs.
    if len(old) + len(new) > 160_000 or old.count(" ") + new.count(" ") > 12_000:
        return [], []

    old_parts = list(_WORD_RE.finditer(old))
    new_parts = list(_WORD_RE.finditer(new))
    old_toks = [m.group(0) for m in old_parts]
    new_toks = [m.group(0) for m in new_parts]
    matcher = difflib.SequenceMatcher(a=old_toks, b=new_toks, autojunk=False)

    left: list[tuple[int, int]] = []
    right: list[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "delete") and i1 < i2:
            left.append((old_parts[i1].start(), old_parts[i2 - 1].end()))
        if tag in ("replace", "insert") and j1 < j2:
            right.append((new_parts[j1].start(), new_parts[j2 - 1].end()))
    return left, right


def signal_ranges(tokens: Sequence[TokenInfo] | None) -> list[tuple[int, int]]:
    if not tokens:
        return []
    return [(t.start, t.end) for t in tokens if t.is_signal and t.end > t.start]


def paint_text_html(
    text: str,
    *,
    signal: Iterable[tuple[int, int]] = (),
    diff: Iterable[tuple[int, int]] = (),
    show_highlights: bool = True,
    diff_class: str = "",
) -> str:
    """Render `text` with overlapping watermark + diff classes."""
    text = text or ""
    if not text:
        return ""

    spans: list[tuple[int, int, str]] = []
    if show_highlights:
        spans.extend((s, e, "watermark-signal") for s, e in signal)
    if diff_class:
        spans.extend((s, e, diff_class) for s, e in diff)

    n = len(text)
    bounds = {0, n}
    cleaned: list[tuple[int, int, str]] = []
    for start, end, cls in spans:
        a = max(0, min(n, int(start)))
        b = max(0, min(n, int(end)))
        if b > a:
            cleaned.append((a, b, cls))
            bounds.add(a)
            bounds.add(b)
    if not cleaned:
        return html.escape(text)

    parts: list[str] = []
    cuts = sorted(bounds)
    for a, b in pairwise(cuts):
        classes = sorted({cls for s, e, cls in cleaned if s < b and e > a})
        chunk = html.escape(text[a:b])
        if not classes:
            parts.append(chunk)
            continue
        titles = [t for c in classes if (t := _SPAN_TITLES.get(c))]
        title_attr = f' title="{html.escape(" · ".join(titles))}"' if titles else ""
        tag = "mark" if "watermark-signal" in classes else "span"
        parts.append(f'<{tag} class="{" ".join(classes)}"{title_attr}>{chunk}</{tag}>')
    return "".join(parts)


def _verdict_line(stats: dict | None) -> str:
    if not stats:
        return '<span class="verdict-meta">Not analyzed yet</span>'
    v = stats.get("verdict", "none")
    label = stats.get("verdict_label", v)
    z = float(stats.get("z_score", 0) or 0)
    gf = float(stats.get("green_fraction", 0) or 0)
    n = stats.get("total_tokens", 0)
    return (
        f'<span class="verdict-pill verdict-{html.escape(str(v))}">{html.escape(str(label))}</span>'
        f'<span class="verdict-meta">z={z:.2f} · green {gf:.1%} · {n} tokens</span>'
    )


def pane_html(
    text: str,
    *,
    side: str,
    tokens: Sequence[TokenInfo] | None = None,
    stats: dict | None = None,
    diff: Iterable[tuple[int, int]] = (),
    show_highlights: bool = True,
    empty_hint: str = "Nothing on this side yet.",
    analyzed: bool = False,
    preview_unanalyzed: bool = False,
) -> str:
    """One compare column: header + annotated body (or a hint if empty / not scored)."""
    label = "Original (old)" if side == "old" else "Cleaned (new)"
    diff_class = "diff-old" if side == "old" else "diff-new"
    body: str
    if not (text or "").strip():
        body = f'<p class="faku-pane-empty">{html.escape(empty_hint)}</p>'
    elif analyzed and tokens is not None:
        painted = paint_text_html(
            text,
            signal=signal_ranges(tokens),
            diff=diff,
            show_highlights=show_highlights,
            diff_class=diff_class,
        )
        body = f'<div class="watermark-output" role="article">{painted}</div>'
    elif preview_unanalyzed:
        painted = paint_text_html(
            text,
            signal=(),
            diff=diff,
            show_highlights=False,
            diff_class=diff_class,
        )
        body = f'<div class="watermark-output" role="article">{painted}</div>'
    else:
        body = f'<p class="faku-pane-empty">{html.escape("Click Analyze to highlight this side.")}</p>'

    return f"""
<div class="faku-compare-col">
  <header>
    <span class="side">{html.escape(label)}</span>
    {_verdict_line(stats if analyzed else None)}
  </header>
  {body}
</div>
"""


def compare_html(
    old_text: str,
    new_text: str,
    *,
    old_tokens: Sequence[TokenInfo] | None = None,
    new_tokens: Sequence[TokenInfo] | None = None,
    old_stats: dict | None = None,
    new_stats: dict | None = None,
    show_highlights: bool = True,
    old_analyzed: bool = False,
    new_analyzed: bool = False,
    preview_unanalyzed: bool = False,
    include_style: bool = True,
) -> str:
    """Left = original, right = cleaned. Yellow = watermark; rose/green = wording changes."""
    left_diff, right_diff = word_diff_ranges(old_text or "", new_text or "")
    left = pane_html(
        old_text or "",
        side="old",
        tokens=old_tokens,
        stats=old_stats,
        diff=left_diff,
        show_highlights=show_highlights,
        empty_hint="Paste original text in the editor above.",
        analyzed=old_analyzed,
        preview_unanalyzed=preview_unanalyzed,
    )
    right = pane_html(
        new_text or "",
        side="new",
        tokens=new_tokens,
        stats=new_stats,
        diff=right_diff,
        show_highlights=show_highlights,
        empty_hint="Neutralize, or paste a candidate here, then Analyze.",
        analyzed=new_analyzed,
        preview_unanalyzed=preview_unanalyzed,
    )
    legend = """
<div class="faku-legend">
  <span><i class="faku-swatch" style="background:#FEF08A"></i>Watermark (green-list)</span>
  <span><i class="faku-swatch" style="background:#fecaca"></i>Removed / replaced (old)</span>
  <span><i class="faku-swatch" style="background:#bbf7d0"></i>Added / replaced (new)</span>
</div>
"""
    body = f'{legend}<div class="faku-compare">{left}{right}</div>'
    if include_style:
        return f"<style>{PANE_CSS}</style>\n{body}"
    return body


def compare_document(
    old_text: str,
    new_text: str,
    *,
    old_tokens: Sequence[TokenInfo] | None = None,
    new_tokens: Sequence[TokenInfo] | None = None,
    old_stats: dict | None = None,
    new_stats: dict | None = None,
    title: str = "fak-u-watermark compare",
) -> str:
    """Standalone HTML export of the side-by-side compare."""
    body = compare_html(
        old_text,
        new_text,
        old_tokens=old_tokens,
        new_tokens=new_tokens,
        old_stats=old_stats,
        new_stats=new_stats,
        show_highlights=True,
        old_analyzed=old_tokens is not None,
        new_analyzed=new_tokens is not None,
        preview_unanalyzed=True,
        include_style=False,
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 72rem; margin: 2rem auto; padding: 0 1.25rem; color: #1c1917; }}
    {PANE_CSS}
    .faku-compare-col .watermark-output {{ max-height: none; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {body}
  <p><small>Generated by fak-u-watermark · Yellow = watermark signal · Rose/green = wording changes.</small></p>
</body>
</html>
"""
