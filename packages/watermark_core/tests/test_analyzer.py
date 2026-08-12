"""Integration-ish tests for WatermarkAnalyzer (requires transformers + gpt2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark_core.analyzer import WatermarkAnalyzer
from watermark_core.schemes.base import Verdict
from watermark_core.schemes.kgw import KGWScheme
from watermark_core.tokenizer import encode_with_offsets, vocab_size
from watermark_core.visualization import tokens_to_html


@pytest.fixture(scope="module")
def gpt2_available():
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("gpt2")
        return True
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gpt2 tokenizer unavailable: {exc}")


def _synthetic_watermarked_ids(seed_key: int = 15485863, n_tokens: int = 80) -> list[int]:
    """
    Generate token ids by always picking from the green list.

    Scoring must use the same ids (decode→re-encode is not lossless for GPT-2).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    scheme = KGWScheme(gamma=0.25, hash_key=seed_key, window=1)
    vsize = tok.vocab_size
    ids = [tok.encode("The", add_special_tokens=False)[0]]
    for _ in range(n_tokens):
        green = scheme.get_green_list(ids, vsize)
        # Prefer mid-range printable tokens for slightly nicer decode
        chosen = min(green, key=lambda t: abs(t - 1000))
        ids.append(chosen)
    return ids


class TestAnalyzer:
    def test_analyze_plain_english(self, gpt2_available):
        analyzer = WatermarkAnalyzer(preset="kirchenbauer_default")
        result = analyzer.analyze(
            "The quick brown fox jumps over the lazy dog. "
            "This is ordinary human-written prose without any watermark."
        )
        assert result.statistics.total_tokens > 5
        # Unwatermarked → z should not be extremely high
        assert result.statistics.z_score < 8
        assert len(result.tokens) >= result.statistics.total_tokens

    def test_synthetic_watermark_detected(self, gpt2_available):
        key = 15485863
        ids = _synthetic_watermarked_ids(seed_key=key, n_tokens=100)
        analyzer = WatermarkAnalyzer(
            scheme="kgw",
            gamma=0.25,
            key=key,
            tokenizer_name="gpt2",
            threshold=4.0,
        )
        result = analyzer.analyze_token_ids(ids)
        assert result.statistics.total_tokens >= 50
        # Every scored token was chosen from green → perfect green fraction
        assert result.statistics.green_fraction == pytest.approx(1.0)
        assert result.statistics.z_score > 4.0
        assert result.statistics.verdict == Verdict.DETECTED
        signal_count = sum(1 for t in result.tokens if t.is_signal)
        assert signal_count >= 50

    def test_wrong_key_weakens_signal(self, gpt2_available):
        key = 15485863
        ids = _synthetic_watermarked_ids(seed_key=key, n_tokens=80)
        good = WatermarkAnalyzer(scheme="kgw", gamma=0.25, key=key).analyze_token_ids(ids)
        bad = WatermarkAnalyzer(scheme="kgw", gamma=0.25, key=999999).analyze_token_ids(ids)
        assert good.statistics.z_score > bad.statistics.z_score
        assert good.statistics.green_fraction > bad.statistics.green_fraction

    def test_html_contains_mark(self, gpt2_available):
        analyzer = WatermarkAnalyzer(preset="kirchenbauer_default")
        result = analyzer.analyze("Hello watermark world, testing highlights.")
        html = tokens_to_html(result.tokens, show_highlights=True)
        assert "watermark-output" in html
        # May or may not have marks depending on randomness
        assert isinstance(html, str)

    def test_token_offsets(self, gpt2_available):
        text = "Hello, world!"
        ids, strings, offsets = encode_with_offsets(text, "gpt2")
        assert len(ids) == len(strings) == len(offsets)
        reconstructed = "".join(text[s:e] for s, e in offsets if e > s)
        # Offsets should cover the text (GPT-2 may skip nothing for this string)
        assert "Hello" in reconstructed or "Hello" in "".join(strings)

    def test_empty_text(self, gpt2_available):
        result = WatermarkAnalyzer().analyze("")
        assert result.statistics.total_tokens == 0
        assert result.tokens == []

    def test_key_from_string(self, gpt2_available):
        a = WatermarkAnalyzer(key="my-secret")
        b = WatermarkAnalyzer(key="my-secret")
        r1 = a.analyze("Some sample text for hashing the key path.")
        r2 = b.analyze("Some sample text for hashing the key path.")
        assert r1.statistics.z_score == r2.statistics.z_score

    def test_to_dict(self, gpt2_available):
        result = WatermarkAnalyzer().analyze("dict export test")
        d = result.to_dict()
        assert "tokens" in d and "statistics" in d
        assert d["statistics"]["verdict"] in ("detected", "uncertain", "none")


class TestVocab:
    def test_vocab_size(self, gpt2_available):
        assert vocab_size("gpt2") > 1000
