"""High-level watermark analyzer: tokenize → green-list score → highlight data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemes import PRESETS, Statistics, TokenInfo, create_scheme
from .tokenizer import encode_with_offsets, vocab_size


def resolve_analyzer_config(
    *,
    scheme: str | None = None,
    gamma: float | None = None,
    key: str | int | None = None,
    tokenizer_name: str | None = None,
    window: int | None = None,
    threshold: float | None = None,
    preset: str | None = None,
) -> dict[str, Any]:
    """Preset fills defaults; any explicit field wins (so a custom key is not ignored)."""
    cfg: dict[str, Any] = {
        "scheme": "kgw",
        "gamma": 0.25,
        "key": None,
        "tokenizer_name": "gpt2",
        "window": 1,
        "threshold": 4.0,
    }
    if preset and preset != "(none)":
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS)}")
        p = PRESETS[preset]
        cfg["scheme"] = p.get("scheme", cfg["scheme"])
        cfg["gamma"] = float(p.get("gamma", cfg["gamma"]))
        cfg["key"] = p.get("hash_key", cfg["key"])
        if "window" in p:
            cfg["window"] = int(p["window"])
    if scheme is not None and str(scheme).strip():
        cfg["scheme"] = scheme
    if gamma is not None:
        cfg["gamma"] = float(gamma)
    if key is not None and not (isinstance(key, str) and not str(key).strip()):
        cfg["key"] = key
    if tokenizer_name is not None and str(tokenizer_name).strip():
        cfg["tokenizer_name"] = tokenizer_name
    if window is not None:
        cfg["window"] = int(window)
    if threshold is not None:
        cfg["threshold"] = float(threshold)
    return cfg


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
        scheme: str | None = None,
        gamma: float | None = None,
        key: str | int | None = None,
        tokenizer_name: str | None = None,
        window: int | None = None,
        threshold: float | None = None,
        preset: str | None = None,
    ):
        cfg = resolve_analyzer_config(
            scheme=scheme,
            gamma=gamma,
            key=key,
            tokenizer_name=tokenizer_name,
            window=window,
            threshold=threshold,
            preset=preset,
        )
        self.scheme_name = cfg["scheme"]
        self.gamma = cfg["gamma"]
        self.hash_key = _parse_key(cfg["key"])
        self.tokenizer_name = cfg["tokenizer_name"]
        self.window = cfg["window"]
        self.threshold = cfg["threshold"]
        self.scheme = create_scheme(
            self.scheme_name,
            gamma=self.gamma,
            hash_key=self.hash_key,
            window=self.window,
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
        window = max(1, int(self.window))

        for i, (tid, tstr, (start, end)) in enumerate(
            zip(token_ids, token_strings, offsets)
        ):
            # First token has no previous context for window-based schemes;
            # we still label it but do not count it in z-score for KGW.
            if i == 0 and self.scheme_name == "kgw":
                is_signal = False
                scored = False
            else:
                # Only the last `window` tokens affect the seed — avoid O(n²) copies.
                prev = token_ids[max(0, i - window) : i]
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
