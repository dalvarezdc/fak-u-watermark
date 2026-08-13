"""Batch analysis helpers for multiple texts / files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .analyzer import AnalysisResult, WatermarkAnalyzer


@dataclass
class BatchItemResult:
    name: str
    success: bool
    statistics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    char_count: int = 0
    preview: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "success": self.success,
            "statistics": self.statistics,
            "error": self.error,
            "char_count": self.char_count,
            "preview": self.preview,
        }


def analyze_batch_texts(
    items: Iterable[tuple[str, str]],
    *,
    analyzer: WatermarkAnalyzer | None = None,
    scheme: str | None = None,
    gamma: float | None = None,
    key: str | int | None = None,
    tokenizer_name: str | None = None,
    preset: str | None = None,
    threshold: float | None = None,
) -> list[BatchItemResult]:
    """items: iterable of (name, text)."""
    if analyzer is None:
        analyzer = WatermarkAnalyzer(
            scheme=scheme,
            gamma=gamma,
            key=key,
            tokenizer_name=tokenizer_name,
            preset=preset,
            threshold=threshold,
        )
    results: list[BatchItemResult] = []
    for name, text in items:
        try:
            r: AnalysisResult = analyzer.analyze(text or "")
            results.append(
                BatchItemResult(
                    name=name,
                    success=True,
                    statistics=r.statistics.to_dict(),
                    char_count=len(text or ""),
                    preview=(text or "")[:120].replace("\n", " "),
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                BatchItemResult(
                    name=name,
                    success=False,
                    error=str(exc),
                    char_count=len(text or ""),
                )
            )
    return results


def analyze_batch_files(
    paths: Iterable[str | Path],
    **kwargs: Any,
) -> list[BatchItemResult]:
    items: list[tuple[str, str]] = []
    for p in paths:
        path = Path(p)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            items.append((path.name, text))
        except Exception as exc:  # noqa: BLE001
            items.append((path.name, ""))
            # Will still analyze empty; capture error via wrapper
            _ = exc
    return analyze_batch_texts(items, **kwargs)


def batch_to_markdown(results: list[BatchItemResult]) -> str:
    if not results:
        return "No batch results."
    lines = [
        "| File | Verdict | Z-score | Green % | Tokens |",
        "|------|---------|--------:|--------:|-------:|",
    ]
    for r in results:
        if not r.success:
            lines.append(f"| {r.name} | ERROR | — | — | — |")
            continue
        s = r.statistics
        lines.append(
            f"| {r.name} | {s.get('verdict_label', s.get('verdict', '—'))} | "
            f"{s.get('z_score', 0):.2f} | {s.get('green_fraction', 0):.1%} | "
            f"{s.get('total_tokens', 0)} |"
        )
    return "\n".join(lines)
