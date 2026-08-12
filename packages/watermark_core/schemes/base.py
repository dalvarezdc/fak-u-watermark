"""Base types and interfaces for watermark schemes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class Verdict(str, Enum):
    DETECTED = "detected"
    UNCERTAIN = "uncertain"
    NONE = "none"

    @property
    def label(self) -> str:
        return {
            Verdict.DETECTED: "Watermark Detected",
            Verdict.UNCERTAIN: "Uncertain",
            Verdict.NONE: "No significant signal",
        }[self]


@dataclass
class Statistics:
    """Detection statistics for a scored sequence."""

    green_count: int
    total_tokens: int
    green_fraction: float
    z_score: float
    p_value: float
    gamma: float
    verdict: Verdict
    threshold: float = 4.0
    scheme: str = "kgw"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "green_count": self.green_count,
            "total_tokens": self.total_tokens,
            "green_fraction": round(self.green_fraction, 4),
            "z_score": round(self.z_score, 4),
            "p_value": self.p_value,
            "gamma": self.gamma,
            "verdict": self.verdict.value,
            "verdict_label": self.verdict.label,
            "threshold": self.threshold,
            "scheme": self.scheme,
            "notes": self.notes,
        }


@dataclass
class TokenInfo:
    """A single token with watermark signal metadata."""

    text: str
    token_id: int
    is_signal: bool  # True if token is on the green list
    start: int
    end: int
    index: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "token_id": self.token_id,
            "is_signal": self.is_signal,
            "start": self.start,
            "end": self.end,
            "index": self.index,
        }


class WatermarkScheme(ABC):
    """Abstract watermark scheme: green-list reconstruction + scoring."""

    name: str = "base"
    gamma: float = 0.5

    @abstractmethod
    def get_green_list(self, prev_token_ids: Sequence[int], vocab_size: int) -> set[int]:
        """Return the set of green-list token ids given previous context."""

    def score_token(
        self,
        token_id: int,
        prev_token_ids: Sequence[int],
        vocab_size: int,
    ) -> bool:
        """Return True if token_id is on the green list."""
        green = self.get_green_list(prev_token_ids, vocab_size)
        return token_id in green

    def compute_statistics(
        self,
        green_flags: Sequence[bool],
        *,
        threshold: float = 4.0,
    ) -> Statistics:
        """Compute z-score and verdict from per-token green flags."""
        n = len(green_flags)
        if n == 0:
            return Statistics(
                green_count=0,
                total_tokens=0,
                green_fraction=0.0,
                z_score=0.0,
                p_value=1.0,
                gamma=self.gamma,
                verdict=Verdict.NONE,
                threshold=threshold,
                scheme=self.name,
                notes="No tokens scored (need at least one context token).",
            )

        green_count = sum(1 for g in green_flags if g)
        gamma = self.gamma
        expected = gamma * n
        variance = n * gamma * (1.0 - gamma)
        if variance <= 0:
            z = 0.0
        else:
            z = (green_count - expected) / (variance**0.5)

        p_value = _normal_sf(z)
        green_fraction = green_count / n

        if z >= threshold:
            verdict = Verdict.DETECTED
        elif z >= threshold * 0.5:
            verdict = Verdict.UNCERTAIN
        else:
            verdict = Verdict.NONE

        return Statistics(
            green_count=green_count,
            total_tokens=n,
            green_fraction=green_fraction,
            z_score=z,
            p_value=p_value,
            gamma=gamma,
            verdict=verdict,
            threshold=threshold,
            scheme=self.name,
        )


def _normal_sf(z: float) -> float:
    """One-sided survival function P(Z > z) for standard normal (approx)."""
    # Abramowitz & Stegun approximation via erfc-like formula
    import math

    if z < 0:
        return 1.0 - _normal_sf(-z)
    # Complementary error function approximation
    t = 1.0 / (1.0 + 0.2316419 * z)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    return max(0.0, min(1.0, d * math.exp(-0.5 * z * z) * poly))


# Presets for common configurations
PRESETS: dict[str, dict] = {
    "kirchenbauer_default": {
        "scheme": "kgw",
        "gamma": 0.25,
        "delta": 2.0,
        "hash_key": 15485863,
        "window": 1,
        "description": "Classic Kirchenbauer et al. (gamma=0.25, soft watermark)",
    },
    "kirchenbauer_hard": {
        "scheme": "kgw",
        "gamma": 0.5,
        "delta": 2.0,
        "hash_key": 15485863,
        "window": 1,
        "description": "Harder green list (gamma=0.5)",
    },
    "unigram_default": {
        "scheme": "unigram",
        "gamma": 0.5,
        "hash_key": 15485863,
        "description": "Unigram watermark (context-independent green list)",
    },
}
