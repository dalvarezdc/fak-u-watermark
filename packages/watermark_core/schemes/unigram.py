"""Unigram watermark scheme — context-independent green list.

A fixed partition of the vocabulary seeded only by the secret key.
Simpler than KGW; green membership does not depend on previous tokens.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Sequence

from .base import WatermarkScheme
from .kgw import _Xorshift64, sample_green_ids


class UnigramScheme(WatermarkScheme):
    """Unigram green list: same green set for every position."""

    name = "unigram"

    def __init__(self, gamma: float = 0.5, hash_key: int = 15485863):
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        self.gamma = gamma
        self.hash_key = int(hash_key)
        self._cache: dict[int, set[int]] = {}

    def get_green_list(self, prev_token_ids: Sequence[int], vocab_size: int) -> set[int]:
        # Context ignored — unigram is position-independent
        del prev_token_ids
        if vocab_size in self._cache:
            return self._cache[vocab_size]

        payload = struct.pack("<Q", self.hash_key)
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        rng = _Xorshift64(seed)
        k = max(1, int(self.gamma * vocab_size))
        green = set(sample_green_ids(rng, vocab_size, k))
        self._cache[vocab_size] = green
        return green
