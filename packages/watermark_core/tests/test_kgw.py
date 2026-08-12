"""Unit tests for KGW green-list reconstruction and z-score."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# packages/ on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark_core.schemes.base import Verdict, _normal_sf
from watermark_core.schemes.kgw import KGWScheme
from watermark_core.schemes.unigram import UnigramScheme


class TestKGWGreenList:
    def test_deterministic(self):
        scheme = KGWScheme(gamma=0.25, hash_key=42, window=1)
        g1 = scheme.get_green_list([100], vocab_size=1000)
        g2 = scheme.get_green_list([100], vocab_size=1000)
        assert g1 == g2

    def test_green_list_size(self):
        scheme = KGWScheme(gamma=0.25, hash_key=1, window=1)
        vocab = 1000
        green = scheme.get_green_list([7], vocab_size=vocab)
        assert len(green) == int(0.25 * vocab)

    def test_different_context_different_list(self):
        scheme = KGWScheme(gamma=0.5, hash_key=99, window=1)
        g_a = scheme.get_green_list([1], vocab_size=500)
        g_b = scheme.get_green_list([2], vocab_size=500)
        # Extremely unlikely to be identical
        assert g_a != g_b

    def test_different_key_different_list(self):
        a = KGWScheme(gamma=0.5, hash_key=1, window=1)
        b = KGWScheme(gamma=0.5, hash_key=2, window=1)
        assert a.get_green_list([10], 200) != b.get_green_list([10], 200)

    def test_score_token(self):
        scheme = KGWScheme(gamma=0.5, hash_key=7, window=1)
        green = scheme.get_green_list([5], vocab_size=100)
        tid = next(iter(green))
        assert scheme.score_token(tid, [5], 100) is True
        # Find a red token
        red = next(i for i in range(100) if i not in green)
        assert scheme.score_token(red, [5], 100) is False


class TestZScore:
    def test_empty(self):
        scheme = KGWScheme(gamma=0.25)
        stats = scheme.compute_statistics([])
        assert stats.total_tokens == 0
        assert stats.verdict == Verdict.NONE

    def test_random_near_zero(self):
        """~gamma green flags → z near 0."""
        scheme = KGWScheme(gamma=0.5)
        # Exactly 50% green
        flags = [True, False] * 500
        stats = scheme.compute_statistics(flags)
        assert abs(stats.z_score) < 0.01
        assert stats.verdict == Verdict.NONE

    def test_watermarked_high_z(self):
        """Heavy green bias → high z and DETECTED."""
        scheme = KGWScheme(gamma=0.25, hash_key=1)
        # 80% green when gamma=0.25
        n = 400
        green = int(0.8 * n)
        flags = [True] * green + [False] * (n - green)
        stats = scheme.compute_statistics(flags, threshold=4.0)
        assert stats.z_score > 10
        assert stats.verdict == Verdict.DETECTED
        assert stats.green_fraction == pytest.approx(0.8)

    def test_formula(self):
        scheme = KGWScheme(gamma=0.25)
        flags = [True] * 100 + [False] * 100  # 50% green, gamma=0.25
        stats = scheme.compute_statistics(flags)
        n = 200
        expected = 0.25 * n
        var = n * 0.25 * 0.75
        z_expected = (100 - expected) / math.sqrt(var)
        assert stats.z_score == pytest.approx(z_expected)


class TestUnigram:
    def test_context_independent(self):
        scheme = UnigramScheme(gamma=0.5, hash_key=123)
        g1 = scheme.get_green_list([], vocab_size=300)
        g2 = scheme.get_green_list([1, 2, 3], vocab_size=300)
        assert g1 == g2
        assert len(g1) == 150


class TestNormalSF:
    def test_zero(self):
        assert _normal_sf(0.0) == pytest.approx(0.5, abs=0.01)

    def test_large_positive(self):
        assert _normal_sf(5.0) < 1e-5

    def test_negative(self):
        assert _normal_sf(-1.0) > 0.5


class TestCreateScheme:
    def test_factory(self):
        from watermark_core.schemes import create_scheme

        kgw = create_scheme("kgw", gamma=0.3, hash_key=1)
        assert kgw.name == "kgw"
        assert kgw.gamma == 0.3

        uni = create_scheme("unigram", gamma=0.4, hash_key=2)
        assert uni.name == "unigram"

    def test_unknown(self):
        from watermark_core.schemes import create_scheme

        with pytest.raises(ValueError):
            create_scheme("not-a-scheme")
