"""Tokenizer helpers. Default: GPT-2 via a bundled fast tokenizer (offline)."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

# Open tokenizers available for selection in the UI
AVAILABLE_TOKENIZERS: dict[str, str] = {
    "gpt2": "GPT-2 (default, classic watermark research)",
    "gpt2-medium": "GPT-2 Medium",
    "distilgpt2": "DistilGPT-2 (smaller, faster)",
}

# Hugging Face Xet / hf_transfer often abort mid-file:
# "peer closed connection without sending complete message body (incomplete chunked read)"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

_DATA_DIR = Path(__file__).resolve().parent / "data"


def bundled_tokenizer_path(name: str) -> Path | None:
    """Return a local snapshot dir if this package ships that tokenizer."""
    slug = (name or "gpt2").strip() or "gpt2"
    if "/" in slug or slug.startswith("."):
        slug = slug.replace("/", "--").lstrip(".")
    path = _DATA_DIR / slug
    if (path / "tokenizer.json").is_file() or (path / "vocab.json").is_file():
        return path
    return None


def tokenizer_error_message(exc: BaseException | None, name: str = "gpt2") -> str:
    raw = str(exc or "unknown error").replace("\n", " ").strip()
    low = raw.lower()
    if any(
        s in low
        for s in (
            "incomplete chunked",
            "peer closed",
            "remoteprotocol",
            "connection reset",
            "timed out",
            "timeout",
        )
    ):
        return (
            f"Could not download tokenizer '{name}' from Hugging Face "
            f"(connection dropped mid-file). Retry Analyze — after one "
            f"successful download it runs offline."
        )
    if len(raw) > 280:
        raw = raw[:277] + "…"
    return f"Could not load tokenizer '{name}': {raw}"


@lru_cache(maxsize=4)
def load_tokenizer(name: str = "gpt2") -> Any:
    """Load and cache a Hugging Face tokenizer.

    Prefers the bundled GPT-2 snapshot, then the local HF cache, then the Hub
    (with retries). Hub backends that commonly drop chunked downloads are disabled.
    """
    from transformers import AutoTokenizer

    name = (name or "gpt2").strip() or "gpt2"
    last_err: BaseException | None = None

    bundled = bundled_tokenizer_path(name)
    if bundled is not None:
        try:
            tok = AutoTokenizer.from_pretrained(
                str(bundled), use_fast=True, local_files_only=True
            )
            tok.model_max_length = int(1e12)
            return tok
        except Exception as exc:  # noqa: BLE001
            last_err = exc

    try:
        tok = AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=True)
        tok.model_max_length = int(1e12)
        return tok
    except Exception as exc:  # noqa: BLE001
        last_err = exc

    for attempt in range(3):
        try:
            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            tok.model_max_length = int(1e12)
            return tok
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.5 * (attempt + 1))

    raise RuntimeError(tokenizer_error_message(last_err, name)) from last_err


def encode_with_offsets(text: str, tokenizer_name: str = "gpt2") -> tuple[list[int], list[str], list[tuple[int, int]]]:
    """
    Tokenize text and recover character offsets for each token.

    Returns:
        token_ids, token_strings, [(start, end), ...]
    """
    tokenizer = load_tokenizer(tokenizer_name)
    # Prefer offset mapping from fast tokenizers
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_ids: list[int] = list(encoded["input_ids"])
    offsets: list[tuple[int, int]] = list(encoded["offset_mapping"])
    token_strings: list[str] = []

    for tid, (start, end) in zip(token_ids, offsets):
        if start == end == 0 and not text:
            piece = ""
        elif start < end:
            piece = text[start:end]
        else:
            # Fallback decode for edge cases
            piece = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        token_strings.append(piece)

    return token_ids, token_strings, offsets


def vocab_size(tokenizer_name: str = "gpt2") -> int:
    tokenizer = load_tokenizer(tokenizer_name)
    return int(getattr(tokenizer, "vocab_size", len(tokenizer)))
