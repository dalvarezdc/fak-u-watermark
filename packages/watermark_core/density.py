"""Sliding-window green-list density heatmap."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Sequence

from .schemes.base import TokenInfo


@dataclass
class DensityPoint:
    index: int
    start: int
    end: int
    text: str
    is_signal: bool
    window_green_fraction: float
    window_z: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "is_signal": self.is_signal,
            "window_green_fraction": round(self.window_green_fraction, 4),
            "window_z": round(self.window_z, 4),
        }


def sliding_window_density(
    tokens: Sequence[TokenInfo],
    *,
    window: int = 20,
    gamma: float = 0.25,
) -> list[DensityPoint]:
    """
    For each scored token, compute green fraction in a centered window of size `window`.

    Tokens that are not signals and have no neighbors still get a local score.
    Uses only tokens that participate in scoring (for KGW, first token has is_signal=False
    and may be excluded from the green_flags stream — here we use all tokens' is_signal).
    """
    n = len(tokens)
    if n == 0:
        return []
    window = max(3, int(window))
    flags = [1.0 if t.is_signal else 0.0 for t in tokens]
    # Prefix sums for O(1) range queries
    prefix = [0.0]
    for f in flags:
        prefix.append(prefix[-1] + f)

    half = window // 2
    points: list[DensityPoint] = []
    for i, t in enumerate(tokens):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)  # exclusive
        length = hi - lo
        green = prefix[hi] - prefix[lo]
        frac = green / length if length else 0.0
        # Local z vs gamma
        if length > 0 and 0 < gamma < 1:
            expected = gamma * length
            var = length * gamma * (1.0 - gamma)
            z = (green - expected) / (var**0.5) if var > 0 else 0.0
        else:
            z = 0.0
        points.append(
            DensityPoint(
                index=t.index,
                start=t.start,
                end=t.end,
                text=t.text,
                is_signal=t.is_signal,
                window_green_fraction=frac,
                window_z=z,
            )
        )
    return points


def density_to_html(
    points: Sequence[DensityPoint],
    *,
    show_values: bool = False,
    wrap: bool = True,
) -> str:
    """
    Render tokens with background intensity from local green density.

    Soft yellow scale: low density nearly transparent, high density strong yellow.
    """
    if not points:
        return "<em>No tokens for heatmap.</em>"

    fracs = [p.window_green_fraction for p in points]
    fmin, fmax = min(fracs), max(fracs)
    span = (fmax - fmin) or 1.0

    parts: list[str] = []
    css = """
.density-map {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 1.05rem; line-height: 1.75; color: #1c1917;
  white-space: pre-wrap; word-break: break-word;
  padding: 1.25rem 1.5rem; border: 1px solid #e7e5e4; border-radius: 12px;
  background: #fffefb; max-height: min(22rem, 50vh); overflow: auto;
  box-shadow: 0 1px 2px rgba(28, 25, 23, 0.04);
}
.density-token { border-radius: 3px; padding: 0.05em 0.1em; }
.density-legend {
  font-family: system-ui, sans-serif; font-size: 0.85rem; color: #78716c;
  margin-top: 0.65rem; line-height: 1.45;
}
""".strip()

    for p in points:
        # Normalize 0..1 within document
        norm = (p.window_green_fraction - fmin) / span
        # Gentler alpha so long prose stays readable
        alpha = 0.06 + 0.55 * norm
        if p.window_z >= 4:
            alpha = min(0.85, alpha + 0.12)
        bg = f"rgba(254, 240, 138, {alpha:.3f})"
        underline = (
            "box-shadow: inset 0 -2px 0 rgba(202, 138, 4, 0.55);" if p.is_signal else ""
        )
        title = (
            f"density={p.window_green_fraction:.0%} z≈{p.window_z:.2f}"
            + (" · green-list token" if p.is_signal else "")
        )
        escaped = html.escape(p.text)
        label = f"{escaped}"
        if show_values:
            label += f"<sup style='font-size:0.6em;opacity:0.7'>{p.window_green_fraction:.0%}</sup>"
        parts.append(
            f'<span class="density-token" style="background:{bg};{underline}" '
            f'title="{html.escape(title)}">{label}</span>'
        )

    body = "".join(parts)
    legend = (
        f'<div class="density-legend">'
        f"<strong>How to read:</strong> warmer yellow = higher local green density "
        f"({fmin:.0%}–{fmax:.0%}). Underlined tokens are on the green list. Hover for details."
        f"</div>"
    )
    if wrap:
        body = f'<div class="density-map">{body}</div>{legend}'
    return f"<style>{css}</style>\n{body}"


def density_summary(points: Sequence[DensityPoint]) -> dict:
    if not points:
        return {"max_fraction": 0.0, "mean_fraction": 0.0, "hot_spans": 0}
    fracs = [p.window_green_fraction for p in points]
    hot = sum(1 for p in points if p.window_z >= 4)
    return {
        "max_fraction": round(max(fracs), 4),
        "mean_fraction": round(sum(fracs) / len(fracs), 4),
        "hot_spans": hot,
        "tokens": len(points),
    }
