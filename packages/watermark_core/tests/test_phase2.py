"""Phase 2: density, targeted, adaptive, settings, batch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark_core.analyzer import WatermarkAnalyzer
from watermark_core.batch import analyze_batch_texts, batch_to_markdown
from watermark_core.density import density_summary, density_to_html, sliding_window_density
from watermark_core.schemes.base import TokenInfo
from watermark_core.settings import AppSettings, clear_api_key, load_settings, save_settings
from watermark_core.targeted import neutralize_targeted


class TestDensity:
    def test_sliding_window(self):
        tokens = [
            TokenInfo(text=f"t{i}", token_id=i, is_signal=(i % 2 == 0), start=i, end=i + 1, index=i)
            for i in range(40)
        ]
        points = sliding_window_density(tokens, window=10, gamma=0.5)
        assert len(points) == 40
        assert 0 <= points[0].window_green_fraction <= 1
        summary = density_summary(points)
        assert summary["tokens"] == 40
        html = density_to_html(points)
        assert "density-map" in html


class TestSettings:
    def test_save_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        # Re-import path resolution uses Path.home()
        from watermark_core import settings as st

        monkeypatch.setattr(st, "settings_path", lambda: tmp_path / ".faku" / "settings.json")
        s = AppSettings(api_key="sk-test-12345678", base_url="https://example.com/v1", model="m1")
        st.save_settings(s)
        loaded = st.load_settings()
        assert loaded.api_key == "sk-test-12345678"
        assert loaded.model == "m1"
        masked = loaded.masked_dict()
        assert masked["has_api_key"] is True
        assert "sk-t" in masked["api_key_masked"] or "…" in masked["api_key_masked"]


class TestBatch:
    def test_batch_texts(self):
        items = [
            ("a.txt", "Hello world ordinary prose."),
            ("b.txt", "Another short sample of text here."),
        ]
        # Avoid heavy tokenizer if possible — still needs gpt2 for real analyze
        try:
            results = analyze_batch_texts(items, scheme="kgw", gamma=0.25)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(str(exc))
        assert len(results) == 2
        assert all(r.success for r in results)
        md = batch_to_markdown(results)
        assert "a.txt" in md


class TestTargeted:
    def test_targeted_runs(self):
        try:
            analyzer = WatermarkAnalyzer(preset="kirchenbauer_default")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(str(exc))
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a good method to generate text about data systems."
        )
        result = neutralize_targeted(text, analyzer=analyzer)
        assert result.success
        assert result.cleaned
        # Replacements may be zero if no green synonyms matched — still ok
        assert isinstance(result.replacements, list)


class TestC2PA:
    def test_detect_plain_png(self):
        import io

        from PIL import Image

        from image_tools.c2pa_tools import detect_c2pa, strip_c2pa

        img = Image.new("RGB", (16, 16), (1, 2, 3))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        report = detect_c2pa(raw)
        assert report.present is False
        cleaned, after = strip_c2pa(raw)
        assert len(cleaned) > 0
        assert after.present is False

    def test_detect_marker(self):
        from image_tools.c2pa_tools import detect_c2pa

        # Synthetic bytes with c2pa marker
        raw = b"\xff\xd8\xff\xe0" + b"c2pa" + b"jumb" + b"\x00" * 20
        report = detect_c2pa(raw)
        assert report.present is True
        assert any("c2pa" in m.lower() for m in report.markers_found)
