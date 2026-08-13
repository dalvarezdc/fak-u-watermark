"""Bundled GPT-2 tokenizer + hub error wording."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark_core.tokenizer import (
    bundled_tokenizer_path,
    encode_with_offsets,
    load_tokenizer,
    tokenizer_error_message,
    vocab_size,
)


def test_gpt2_is_bundled():
    path = bundled_tokenizer_path("gpt2")
    assert path is not None
    assert (path / "tokenizer.json").is_file()


def test_load_bundled_gpt2_and_encode():
    load_tokenizer.cache_clear()
    tok = load_tokenizer("gpt2")
    assert tok is not None
    ids, strings, offsets = encode_with_offsets("Hello world", "gpt2")
    assert ids
    assert len(ids) == len(strings) == len(offsets)
    assert "Hello" in "".join(strings)
    assert vocab_size("gpt2") > 1000


def test_chunked_read_error_is_actionable():
    exc = RuntimeError(
        "peer closed connection without sending complete message body "
        "(incomplete chunked read)"
    )
    msg = tokenizer_error_message(exc, "gpt2")
    assert "Hugging Face" in msg
    assert "Retry" in msg
    assert "incomplete chunked" not in msg.lower() or "download" in msg.lower()


def test_unknown_tokenizer_has_no_bundle():
    assert bundled_tokenizer_path("definitely-not-a-tokenizer") is None
