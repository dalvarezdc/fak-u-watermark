"""Phase 2: density, targeted, adaptive, settings, batch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark_core.adaptive import neutralize_adaptive
from watermark_core.analyzer import WatermarkAnalyzer, resolve_analyzer_config
from watermark_core.batch import analyze_batch_texts, batch_to_markdown
from watermark_core.density import density_summary, density_to_html, sliding_window_density
from watermark_core.neutralize import (
    NeutralizeConfig,
    NeutralizeResult,
    cleaned_from_completion,
)
from watermark_core.schemes.base import TokenInfo
from watermark_core.settings import (
    AppSettings,
    clear_api_key,
    load_settings,
    resolve_chat_model,
    save_settings,
)
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

    def test_resolve_chat_model_rewrites_openai_name_on_deepseek(self):
        assert (
            resolve_chat_model("deepseek", "https://api.deepseek.com", "gpt-4o-mini")
            == "deepseek-v4-flash"
        )
        assert (
            resolve_chat_model("", "https://api.deepseek.com", "deepseek-chat")
            == "deepseek-v4-flash"
        )
        assert (
            resolve_chat_model("deepseek", "https://api.deepseek.com", "deepseek-v4-pro")
            == "deepseek-v4-pro"
        )
        assert resolve_chat_model("openai", "https://api.openai.com/v1", "gpt-4o-mini") == "gpt-4o-mini"

    def test_http_error_includes_provider_message(self):
        import httpx

        from watermark_core.neutralize import _http_error_detail

        resp = httpx.Response(
            400,
            json={"error": {"message": "Model Not Exist"}},
            request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"),
        )
        msg = _http_error_detail(resp, "https://api.deepseek.com/chat/completions")
        assert "400" in msg
        assert "Model Not Exist" in msg


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


class TestAnalyzerConfig:
    def test_preset_alone_applies_hard_gamma(self):
        cfg = resolve_analyzer_config(preset="kirchenbauer_hard")
        assert cfg["gamma"] == 0.5
        assert cfg["scheme"] == "kgw"
        a = WatermarkAnalyzer(preset="kirchenbauer_hard")
        assert a.gamma == 0.5
        assert a.hash_key == 15485863

    def test_explicit_key_wins_over_preset(self):
        a = WatermarkAnalyzer(preset="kirchenbauer_default", key=99999)
        assert a.hash_key == 99999
        assert a.gamma == 0.25

    def test_explicit_gamma_wins_over_preset(self):
        a = WatermarkAnalyzer(preset="kirchenbauer_hard", gamma=0.9)
        assert a.gamma == pytest.approx(0.9)


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

    def test_preserves_markdown_whitespace_without_replacements(self):
        try:
            analyzer = WatermarkAnalyzer(preset="kirchenbauer_default")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(str(exc))
        text = "Paragraph.\n\n    indented code line\n\nTwo  spaces  here."
        result = neutralize_targeted(text, analyzer=analyzer)
        if result.replacements:
            pytest.skip("synonym hit on fixture text; cannot assert identity")
        assert result.cleaned == text

    def test_replacement_keeps_leading_space(self):
        try:
            analyzer = WatermarkAnalyzer(preset="kirchenbauer_default")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(str(exc))
        text = "The however example follows."
        result = neutralize_targeted(text, analyzer=analyzer)
        assert "Theyet" not in result.cleaned
        assert "Therequire" not in result.cleaned
        for rep in result.replacements:
            if rep["original"].startswith(" "):
                assert rep["replacement"].startswith(" "), rep


class TestNeutralizeExtract:
    def test_none_content(self):
        assert cleaned_from_completion({"choices": [{"message": {"content": None}}]}) is None

    def test_empty_content(self):
        assert cleaned_from_completion({"choices": [{"message": {"content": "   "}}]}) is None

    def test_unwraps_single_fence_keeps_inner(self):
        body = {"choices": [{"message": {"content": "```\nHello  world\n```\n"}}]}
        assert cleaned_from_completion(body) == "Hello  world"

    def test_does_not_strip_plain_text(self):
        body = {"choices": [{"message": {"content": "  leading and trailing  \n"}}]}
        assert cleaned_from_completion(body) == "  leading and trailing  \n"

    def test_multipart_content(self):
        body = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": " world"},
                        ]
                    }
                }
            ]
        }
        assert cleaned_from_completion(body) == "Hello world"


class _FakeStats:
    def __init__(self, z: float):
        self.z_score = z


class _FakeAnalysis:
    def __init__(self, z: float):
        self.statistics = _FakeStats(z)


class _ScriptedAnalyzer:
    def __init__(self, scores: list[float]):
        self.scores = list(scores)

    def analyze(self, text: str):
        del text
        z = self.scores.pop(0) if self.scores else 10.0
        return _FakeAnalysis(z)


class TestAdaptive:
    def test_missed_target_is_not_success_and_keeps_best(self, monkeypatch):
        from watermark_core import adaptive as adp

        sample = "original draft"
        drafts = ["better", "worse", "mid"]
        calls = {"n": 0}

        def fake_sync(text, config):
            del text
            i = calls["n"]
            calls["n"] += 1
            return NeutralizeResult(
                original=sample,
                cleaned=drafts[i],
                style=config.style,
                model="fake",
                success=True,
            )

        monkeypatch.setattr(adp, "neutralize_sync", fake_sync)
        # initial 8 → better 5 → worse 7 → mid 6; keep "better"
        analyzer = _ScriptedAnalyzer([8.0, 5.0, 7.0, 6.0])
        cfg = NeutralizeConfig(api_key="x", model="m", style="subtle")
        result = neutralize_adaptive(
            sample,
            analyzer=analyzer,  # type: ignore[arg-type]
            config=cfg,
            max_rounds=3,
            target_z=4.0,
        )
        assert result.success is False
        assert result.error and "Stopped after" in result.error
        assert result.cleaned == "better"
        assert result.z_scores == [8.0, 5.0, 7.0, 6.0]
        assert cfg.style == "subtle"

    def test_stops_when_paraphrase_is_unchanged(self, monkeypatch):
        from watermark_core import adaptive as adp

        sample = "same text"

        def fake_sync(text, config):
            return NeutralizeResult(
                original=text,
                cleaned=text,
                style=config.style,
                model="fake",
                success=True,
            )

        monkeypatch.setattr(adp, "neutralize_sync", fake_sync)
        result = neutralize_adaptive(
            sample,
            analyzer=_ScriptedAnalyzer([8.0]),  # type: ignore[arg-type]
            config=NeutralizeConfig(api_key="x", model="m", style="subtle"),
            max_rounds=3,
            target_z=4.0,
        )
        assert result.success is False
        assert result.rounds == 1
        assert "unchanged" in (result.error or "")

    def test_chunked_rewrites_only_hot_sections(self, monkeypatch):
        from watermark_core import adaptive as adp

        clean = ("Ordinary clean prose about nothing special. " * 40).strip()
        hot = ("Watermarked generated text about data systems. " * 40).strip()
        sample = clean + "\n\n" + hot
        seen: list[str] = []

        def fake_sync(text, config):
            seen.append(text)
            return NeutralizeResult(
                original=text,
                cleaned=text.replace("Watermarked", "Rewritten"),
                style=config.style,
                model="fake",
                success=True,
            )

        monkeypatch.setattr(adp, "neutralize_sync", fake_sync)

        class _An:
            def analyze(self, text: str):
                if "Watermarked" in text:
                    return _FakeAnalysis(6.5)
                return _FakeAnalysis(1.0)

        result = neutralize_adaptive(
            sample,
            analyzer=_An(),  # type: ignore[arg-type]
            config=NeutralizeConfig(
                api_key="x", model="m", style="subtle", max_chunk_chars=200
            ),
            max_rounds=2,
            target_z=4.0,
        )
        assert result.success is True
        assert "Rewritten" in result.cleaned
        assert "Ordinary clean prose" in result.cleaned
        assert seen
        assert all("Ordinary clean prose" not in s or "Watermarked" in s for s in seen)

    def test_already_below_target_skips_llm(self, monkeypatch):
        from watermark_core import adaptive as adp

        def boom(*_a, **_k):
            raise AssertionError("should not call neutralize")

        monkeypatch.setattr(adp, "neutralize_sync", boom)
        result = neutralize_adaptive(
            "hi",
            analyzer=_ScriptedAnalyzer([1.0]),  # type: ignore[arg-type]
            config=NeutralizeConfig(api_key="x", model="m"),
            max_rounds=3,
            target_z=4.0,
        )
        assert result.success is True
        assert result.rounds == 0


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
