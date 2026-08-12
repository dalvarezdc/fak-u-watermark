"""Watermark scheme implementations."""

from .base import PRESETS, Statistics, TokenInfo, Verdict, WatermarkScheme
from .kgw import KGWScheme
from .unigram import UnigramScheme

SCHEME_REGISTRY: dict[str, type[WatermarkScheme]] = {
    "kgw": KGWScheme,
    "unigram": UnigramScheme,
}


def create_scheme(
    scheme: str = "kgw",
    *,
    gamma: float = 0.25,
    hash_key: int = 15485863,
    window: int = 1,
    **_kwargs,
) -> WatermarkScheme:
    """Factory for watermark schemes."""
    scheme = scheme.lower().strip()
    if scheme not in SCHEME_REGISTRY:
        raise ValueError(f"Unknown scheme '{scheme}'. Choose from: {list(SCHEME_REGISTRY)}")
    cls = SCHEME_REGISTRY[scheme]
    if scheme == "kgw":
        return cls(gamma=gamma, hash_key=hash_key, window=window)  # type: ignore[call-arg]
    if scheme == "unigram":
        return cls(gamma=gamma, hash_key=hash_key)  # type: ignore[call-arg]
    return cls()  # type: ignore[call-arg]


__all__ = [
    "PRESETS",
    "SCHEME_REGISTRY",
    "Statistics",
    "TokenInfo",
    "Verdict",
    "WatermarkScheme",
    "KGWScheme",
    "UnigramScheme",
    "create_scheme",
]
