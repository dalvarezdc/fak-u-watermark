"""Side-by-side compare + word-level diff."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark_core.diffview import (
    compare_document,
    compare_html,
    paint_text_html,
    word_diff_ranges,
)
from watermark_core.schemes.base import TokenInfo


def test_identical_texts_have_no_diff_ranges():
    left, right = word_diff_ranges("hello world", "hello world")
    assert left == []
    assert right == []


def test_empty_or_one_sided_has_no_diff_ranges():
    assert word_diff_ranges("", "hello") == ([], [])
    assert word_diff_ranges("hello", "") == ([], [])
    assert word_diff_ranges("", "") == ([], [])


def test_replacement_marks_both_sides():
    old = "The quick brown fox"
    new = "The slow brown fox"
    left, right = word_diff_ranges(old, new)
    assert len(left) == 1
    assert old[left[0][0] : left[0][1]] == "quick"
    assert len(right) == 1
    assert new[right[0][0] : right[0][1]] == "slow"


def test_insertion_only_on_new_side():
    old = "hello world"
    new = "hello brave world"
    left, right = word_diff_ranges(old, new)
    assert left == []
    assert len(right) == 1
    assert "brave" in new[right[0][0] : right[0][1]]


def test_paint_applies_signal_and_diff_classes():
    text = "alpha beta gamma"
    html = paint_text_html(
        text,
        signal=[(0, 5)],
        diff=[(6, 10)],
        show_highlights=True,
        diff_class="diff-old",
    )
    assert 'class="watermark-signal"' in html
    assert "diff-old" in html
    assert "alpha" in html and "beta" in html


def test_paint_hides_signal_when_asked():
    html = paint_text_html(
        "alpha",
        signal=[(0, 5)],
        show_highlights=False,
        diff_class="diff-old",
    )
    assert "watermark-signal" not in html


def test_compare_html_has_both_columns_and_no_body_reprint_before_analyze():
    html = compare_html(
        "original text here",
        "",
        old_analyzed=False,
        new_analyzed=False,
    )
    assert "Original (old)" in html
    assert "Cleaned (new)" in html
    assert "Click Analyze to highlight this side." in html
    # Unanalyzed panes must not dump the raw chapter again
    assert "original text here" not in html


def test_compare_html_paints_after_analyze():
    tokens = [
        TokenInfo(text="Hello", token_id=1, is_signal=True, start=0, end=5, index=0),
        TokenInfo(text=" world", token_id=2, is_signal=False, start=5, end=11, index=1),
    ]
    html = compare_html(
        "Hello world",
        "Hello earth",
        old_tokens=tokens,
        new_tokens=tokens,
        old_stats={"verdict": "detected", "verdict_label": "Watermark Detected", "z_score": 5.2, "green_fraction": 0.6, "total_tokens": 2},
        new_stats={"verdict": "none", "verdict_label": "No significant signal", "z_score": 0.4, "green_fraction": 0.2, "total_tokens": 2},
        old_analyzed=True,
        new_analyzed=True,
    )
    assert "watermark-signal" in html
    assert "diff-old" in html
    assert "diff-new" in html
    assert "Watermark Detected" in html
    assert "No significant signal" in html


def test_compare_document_is_standalone_html():
    doc = compare_document("aaa extra", "bbb extra")
    assert doc.startswith("<!DOCTYPE html>")
    assert "faku-compare" in doc
    # Export always includes the texts (preview_unanalyzed); changed words are wrapped
    assert "aaa" in doc and "bbb" in doc and "extra" in doc
    assert "diff-old" in doc
    assert "diff-new" in doc


def test_preview_unanalyzed_shows_diff_without_watermark():
    html = compare_html(
        "one two three",
        "one TWO three",
        preview_unanalyzed=True,
        old_analyzed=False,
        new_analyzed=False,
    )
    assert ">two<" in html or "two" in html
    assert "diff-old" in html
    assert "diff-new" in html
    body = html.split("</style>", 1)[-1]
    assert "watermark-signal" not in body
