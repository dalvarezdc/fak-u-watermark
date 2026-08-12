"""Tokenizer helpers. Default: GPT-2 via Hugging Face transformers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# Open tokenizers available for selection in the UI
AVAILABLE_TOKENIZERS: dict[str, str] = {
    "gpt2": "GPT-2 (default, classic watermark research)",
    "gpt2-medium": "GPT-2 Medium",
    "distilgpt2": "DistilGPT-2 (smaller, faster)",
}


@lru_cache(maxsize=4)
def load_tokenizer(name: str = "gpt2") -> Any:
    """Load and cache a Hugging Face tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    return tok


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
