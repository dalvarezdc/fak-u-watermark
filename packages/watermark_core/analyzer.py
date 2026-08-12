"""High-level watermark analyzer: tokenize → green-list score → highlight data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemes import PRESETS, Statistics, TokenInfo, create_scheme
from .tokenizer import encode_with_offsets, vocab_size


@dataclass
class AnalysisResult:
    """Full analysis output for a piece of text."""

    text: str
    tokens: list[TokenInfo]
    statistics: Statistics
    scheme: str
    tokenizer_name: str
    gamma: float
    hash_key: int
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens": [t.to_dict() for t in self.tokens],
            "statistics": self.statistics.to_dict(),
            "scheme": self.scheme,
            "tokenizer_name": self.tokenizer_name,
            "gamma": self.gamma,
            "hash_key": self.hash_key,
            "config": self.config,
        }


class WatermarkAnalyzer:
    """
    Detect statistical green-list watermarks (KGW / Unigram family).

    Offline: pure Python scoring + local tokenizer.
    """

    def __init__(
        self,
        scheme: str = "kgw",
        gamma: float = 0.25,
        key: str | int | None = None,
        tokenizer_name: str = "gpt2",
        window: int = 1,
        threshold: float = 4.0,
        preset: str | None = None,
    ):
        if preset:
            if preset not in PRESETS:
                raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS)}")
            p = PRESETS[preset]
            scheme = p.get("scheme", scheme)
            gamma = float(p.get("gamma", gamma))
            key = p.get("hash_key", key)
            window = int(p.get("window", window))

        self.scheme_name = scheme
        self.gamma = gamma
        self.hash_key = _parse_key(key)
        self.tokenizer_name = tokenizer_name
        self.window = window
        self.threshold = threshold
        self.scheme = create_scheme(
            scheme,
            gamma=gamma,
            hash_key=self.hash_key,
            window=window,
        )

    def analyze(self, text: str) -> AnalysisResult:
        """Tokenize text, reconstruct green lists, and score each token."""
        if text is None:
            text = ""
        token_ids, token_strings, offsets = encode_with_offsets(text, self.tokenizer_name)
        return self._score(
            text=text,
            token_ids=token_ids,
            token_strings=token_strings,
            offsets=offsets,
        )

    def analyze_token_ids(
        self,
        token_ids: list[int],
        *,
        text: str = "",
        token_strings: list[str] | None = None,
    ) -> AnalysisResult:
        """Score a known token-id sequence (useful for tests and generators)."""
        if token_strings is None:
            from .tokenizer import load_tokenizer

            tok = load_tokenizer(self.tokenizer_name)
            token_strings = [
                tok.decode([tid], clean_up_tokenization_spaces=False) for tid in token_ids
            ]
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for s in token_strings:
            offsets.append((cursor, cursor + len(s)))
            cursor += len(s)
        if not text:
            text = "".join(token_strings)
        return self._score(
            text=text,
            token_ids=token_ids,
            token_strings=token_strings,
            offsets=offsets,
        )

    def _score(
        self,
        *,
        text: str,
        token_ids: list[int],
        token_strings: list[str],
        offsets: list[tuple[int, int]],
    ) -> AnalysisResult:
        vsize = vocab_size(self.tokenizer_name)
        tokens: list[TokenInfo] = []
        green_flags: list[bool] = []

        for i, (tid, tstr, (start, end)) in enumerate(
            zip(token_ids, token_strings, offsets)
        ):
            # First token has no previous context for window-based schemes;
            # we still label it but do not count it in z-score for KGW.
            if i == 0 and self.scheme_name == "kgw":
                is_signal = False
                scored = False
            else:
                prev = token_ids[:i]
                is_signal = self.scheme.score_token(tid, prev, vsize)
                scored = True

            if scored:
                green_flags.append(is_signal)

            tokens.append(
                TokenInfo(
                    text=tstr,
                    token_id=tid,
                    is_signal=is_signal,
                    start=start,
                    end=end,
                    index=i,
                )
            )

        stats = self.scheme.compute_statistics(green_flags, threshold=self.threshold)
        return AnalysisResult(
            text=text,
            tokens=tokens,
            statistics=stats,
            scheme=self.scheme_name,
            tokenizer_name=self.tokenizer_name,
            gamma=self.gamma,
            hash_key=self.hash_key,
            config={
                "window": self.window,
                "threshold": self.threshold,
            },
        )

    def get_highlighted_spans(self, text: str) -> list[dict]:
        """Convenience: return list of {start, end, is_signal, text} spans."""
        result = self.analyze(text)
        return [
            {
                "start": t.start,
                "end": t.end,
                "is_signal": t.is_signal,
                "text": t.text,
            }
            for t in result.tokens
        ]


def _parse_key(key: str | int | None) -> int:
    if key is None:
        return 15485863  # default Kirchenbauer-style prime
    if isinstance(key, int):
        return key
    key = str(key).strip()
    if not key:
        return 15485863
    try:
        return int(key)
    except ValueError:
        # Derive integer from arbitrary secret string
        import hashlib

        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little", signed=False)
