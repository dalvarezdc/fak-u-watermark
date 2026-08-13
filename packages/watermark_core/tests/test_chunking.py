"""Document splitting + long-text analyze speed."""

from __future__ import annotations

import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark_core.chunking import join_chunks, split_document
from watermark_core.diffview import word_diff_ranges


def test_split_roundtrip_short():
    text = "Hello world.\n\nNext paragraph."
    assert join_chunks(split_document(text, max_chars=20)) == text
    assert join_chunks(split_document(text, max_chars=5000)) == text


def test_split_long_paragraphs():
    paras = [f"Paragraph {i} with enough words to be real text." for i in range(40)]
    text = "\n\n".join(paras)
    chunks = split_document(text, max_chars=250)
    assert join_chunks(chunks) == text
    bodies = [c for c in chunks if c.kind == "body"]
    assert len(bodies) > 1
    assert all(len(c.text) <= 250 or c.kind == "sep" for c in chunks)


def test_split_hard_wraps_giant_token():
    text = "x" * 500
    chunks = split_document(text, max_chars=80)
    assert join_chunks(chunks) == text
    assert all(len(c.text) <= 80 for c in chunks)


def test_word_diff_skips_huge_inputs():
    old = ("word " * 20_000).strip()
    new = ("word " * 20_000).strip() + " extra"
    assert word_diff_ranges(old, new) == ([], [])


def test_analyze_long_text_is_fast():
    from watermark_core.analyzer import WatermarkAnalyzer

    text = (
        "The committee announced a new method to generate watermarked text "
        "about data systems and important results. "
    ) * 250
    analyzer = WatermarkAnalyzer(preset="kirchenbauer_default")
    t0 = time.perf_counter()
    result = analyzer.analyze(text)
    elapsed = time.perf_counter() - t0
    assert result.statistics.total_tokens > 1500
    assert elapsed < 8.0, f"analyze took {elapsed:.2f}s"
