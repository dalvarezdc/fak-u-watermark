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
.density-map { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  white-space: pre-wrap; word-break: break-word; line-height: 1.7; font-size: 0.95rem;
  padding: 0.75rem; border: 1px solid #e5e7eb; border-radius: 8px; background: #fafafa; }
.density-token { border-radius: 2px; padding: 0 1px; }
.density-legend { font-size: 0.8rem; color: #6b7280; margin-top: 0.5rem; }
""".strip()

    for p in points:
        # Normalize 0..1 within document
        norm = (p.window_green_fraction - fmin) / span
        # Map to alpha 0.05 .. 0.95 on soft yellow #FEF08A ≈ rgb(254,240,138)
        alpha = 0.08 + 0.87 * norm
        # Stronger when z high
        if p.window_z >= 4:
            alpha = min(0.98, alpha + 0.1)
        bg = f"rgba(254, 240, 138, {alpha:.3f})"
        border = "1px solid rgba(202, 138, 4, 0.35)" if p.is_signal else "none"
        title = (
            f"density={p.window_green_fraction:.0%} z≈{p.window_z:.2f}"
            + (" · green-list token" if p.is_signal else "")
        )
        escaped = html.escape(p.text)
        label = f"{escaped}"
        if show_values:
            label += f"<sup style='font-size:0.6em;opacity:0.7'>{p.window_green_fraction:.0%}</sup>"
        parts.append(
            f'<span class="density-token" style="background:{bg};border-bottom:{border}" '
            f'title="{html.escape(title)}">{label}</span>'
        )

    body = "".join(parts)
    legend = (
        f'<div class="density-legend">Heatmap: lighter = lower local green density · '
        f"darker yellow = higher · range {fmin:.0%}–{fmax:.0%} · hover for details</div>"
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
