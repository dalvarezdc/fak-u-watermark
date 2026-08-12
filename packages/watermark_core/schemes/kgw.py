"""Kirchenbauer et al. green-red list (KGW) watermark scheme.

Reference: Kirchenbauer et al., "A Watermark for Large Language Models"
(ICML 2023). Uses a seeded PRNG over the vocabulary keyed by previous token(s).
"""

from __future__ import annotations

import hashlib
import struct
from typing import Sequence

from .base import WatermarkScheme


class KGWScheme(WatermarkScheme):
    """Classic KGW: hash previous token → seed → green list of size gamma * V."""

    name = "kgw"

    def __init__(
        self,
        gamma: float = 0.25,
        hash_key: int = 15485863,
        window: int = 1,
    ):
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.gamma = gamma
        self.hash_key = int(hash_key)
        self.window = window

    def _seed_from_context(self, prev_token_ids: Sequence[int]) -> int:
        """Deterministic 64-bit seed from previous token window + secret key."""
        ctx = list(prev_token_ids[-self.window :])
        # Pad if context shorter than window (beginning of sequence)
        while len(ctx) < self.window:
            ctx.insert(0, 0)
        payload = struct.pack(f"<{len(ctx) + 1}Q", self.hash_key, *[int(t) & 0xFFFFFFFFFFFFFFFF for t in ctx])
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "little", signed=False)

    def get_green_list(self, prev_token_ids: Sequence[int], vocab_size: int) -> set[int]:
        """Return green-list token ids for the given previous context."""
        if vocab_size <= 0:
            return set()
        seed = self._seed_from_context(prev_token_ids)
        # Deterministic Fisher-Yates style selection via xorshift PRNG
        rng = _Xorshift64(seed)
        k = max(1, int(self.gamma * vocab_size))
        # Efficient sampling without full shuffle for large vocabs:
        # generate permutation of [0, V) on the fly for first k via partial Fisher-Yates
        indices = list(range(vocab_size))
        green: set[int] = set()
        for i in range(k):
            j = i + rng.randint(vocab_size - i)
            indices[i], indices[j] = indices[j], indices[i]
            green.add(indices[i])
        return green


class _Xorshift64:
    """Simple deterministic PRNG for green-list reconstruction."""

    def __init__(self, seed: int):
        self.state = seed if seed != 0 else 0xDEADBEEFCAFEBABE

    def next_u64(self) -> int:
        x = self.state & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.state = x
        return x

    def randint(self, n: int) -> int:
        """Uniform integer in [0, n)."""
        if n <= 0:
            return 0
        return self.next_u64() % n
