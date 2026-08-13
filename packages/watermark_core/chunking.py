"""Split long documents at paragraph / sentence boundaries for LLM calls."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Keep separators so join(split(text)) == text.
_PARA_SPLIT = re.compile(r"(\n[ \t]*\n+)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])([ \t]+|\n+)")
_WS_SPLIT = re.compile(r"(\s+)")

DEFAULT_MAX_CHUNK_CHARS = 3500


@dataclass
class TextChunk:
    text: str
    kind: str  # "body" | "sep"


def join_chunks(chunks: list[TextChunk]) -> str:
    return "".join(c.text for c in chunks)


def split_document(text: str, *, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[TextChunk]:
    """Split `text` into body chunks of at most `max_chars`, plus original separators.

    Round-trip: ``join_chunks(split_document(text)) == text``.
    """
    if text is None:
        return []
    max_chars = max(80, int(max_chars))
    if len(text) <= max_chars:
        return [TextChunk(text, "body")] if text else []

    raw_parts = _PARA_SPLIT.split(text)
    pieces: list[TextChunk] = []
    for part in raw_parts:
        if not part:
            continue
        if _PARA_SPLIT.fullmatch(part):
            pieces.append(TextChunk(part, "sep"))
        else:
            pieces.extend(_split_body(part, max_chars))
    return _pack(pieces, max_chars)


def _split_body(text: str, max_chars: int) -> list[TextChunk]:
    if len(text) <= max_chars:
        return [TextChunk(text, "body")]
    out: list[TextChunk] = []
    buf = ""
    parts = _SENT_SPLIT.split(text)
    # split keeps delimiters as their own items
    i = 0
    while i < len(parts):
        piece = parts[i]
        i += 1
        if i < len(parts) and _SENT_SPLIT.fullmatch(parts[i]):
            piece += parts[i]
            i += 1
        if not piece:
            continue
        if len(piece) > max_chars:
            if buf:
                out.append(TextChunk(buf, "body"))
                buf = ""
            out.extend(_split_hard(piece, max_chars))
            continue
        if buf and len(buf) + len(piece) > max_chars:
            out.append(TextChunk(buf, "body"))
            buf = piece
        else:
            buf += piece
    if buf:
        out.append(TextChunk(buf, "body"))
    return out


def _split_hard(text: str, max_chars: int) -> list[TextChunk]:
    if len(text) <= max_chars:
        return [TextChunk(text, "body")]
    out: list[TextChunk] = []
    buf = ""
    for part in _WS_SPLIT.split(text):
        if not part:
            continue
        if len(part) > max_chars:
            if buf:
                out.append(TextChunk(buf, "body"))
                buf = ""
            for start in range(0, len(part), max_chars):
                out.append(TextChunk(part[start : start + max_chars], "body"))
            continue
        if buf and len(buf) + len(part) > max_chars:
            out.append(TextChunk(buf, "body"))
            buf = part
        else:
            buf += part
    if buf:
        out.append(TextChunk(buf, "body"))
    return out


def _pack(chunks: list[TextChunk], max_chars: int) -> list[TextChunk]:
    """Merge a body with following seps/bodies while they still fit."""
    packed: list[TextChunk] = []
    buf = ""
    for chunk in chunks:
        if not buf:
            if chunk.kind == "sep":
                packed.append(chunk)
            else:
                buf = chunk.text
            continue
        if len(buf) + len(chunk.text) <= max_chars:
            buf += chunk.text
            continue
        packed.append(TextChunk(buf, "body"))
        if chunk.kind == "sep":
            packed.append(chunk)
            buf = ""
        else:
            buf = chunk.text
    if buf:
        packed.append(TextChunk(buf, "body"))
    return packed
