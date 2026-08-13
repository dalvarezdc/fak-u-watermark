"""Kirchenbauer et al. green-red list (KGW) watermark scheme.

Reference: Kirchenbauer et al., "A Watermark for Large Language Models"
(ICML 2023). Uses a seeded PRNG over the vocabulary keyed by previous token(s).
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from typing import Sequence

from .base import WatermarkScheme

# ~6.3 KB per GPT-2 context. 2048 entries ≈ 13 MB.
_BITSET_CACHE_MAX = 2048


def sample_green_ids(rng: _Xorshift64, vocab_size: int, k: int) -> list[int]:
    """Partial Fisher-Yates with an implicit identity permutation (O(k) memory).

    Same subset as `indices = list(range(V)); swap first k`.
    """
    if k <= 0 or vocab_size <= 0:
        return []
    k = min(k, vocab_size)
    swapped: dict[int, int] = {}

    def at(i: int) -> int:
        return swapped.get(i, i)

    green: list[int] = []
    for i in range(k):
        j = i + rng.randint(vocab_size - i)
        vi, vj = at(i), at(j)
        swapped[i] = vj
        swapped[j] = vi
        green.append(vj)
    return green


def ids_to_bitset(ids: Sequence[int], vocab_size: int) -> bytearray:
    bits = bytearray((max(vocab_size, 1) + 7) // 8)
    limit = vocab_size
    for tid in ids:
        if 0 <= tid < limit:
            bits[tid >> 3] |= 1 << (tid & 7)
    return bits


def bitset_contains(bits: bytearray, token_id: int) -> bool:
    if token_id < 0:
        return False
    idx = token_id >> 3
    if idx >= len(bits):
        return False
    return bool(bits[idx] & (1 << (token_id & 7)))


def bitset_to_set(bits: bytearray, vocab_size: int) -> set[int]:
    out: set[int] = set()
    for tid in range(vocab_size):
        if bits[tid >> 3] & (1 << (tid & 7)):
            out.add(tid)
    return out


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
        self._bits_cache: OrderedDict[tuple[int, int, int], bytearray] = OrderedDict()

    def _seed_from_context(self, prev_token_ids: Sequence[int]) -> int:
        """Deterministic 64-bit seed from previous token window + secret key."""
        w = self.window
        n = len(prev_token_ids)
        if n >= w:
            ctx = prev_token_ids[n - w :]
            payload = struct.pack(
                f"<{w + 1}Q",
                self.hash_key,
                *[int(t) & 0xFFFFFFFFFFFFFFFF for t in ctx],
            )
        else:
            padded = [0] * (w - n) + [int(t) & 0xFFFFFFFFFFFFFFFF for t in prev_token_ids]
            payload = struct.pack(f"<{w + 1}Q", self.hash_key, *padded)
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "little", signed=False)

    def _cache_key(self, prev_token_ids: Sequence[int], vocab_size: int) -> tuple[int, int, int]:
        seed = self._seed_from_context(prev_token_ids)
        k = max(1, int(self.gamma * vocab_size)) if vocab_size > 0 else 0
        return (seed, vocab_size, k)

    def _green_bits(self, prev_token_ids: Sequence[int], vocab_size: int) -> bytearray:
        if vocab_size <= 0:
            return bytearray()
        key = self._cache_key(prev_token_ids, vocab_size)
        cached = self._bits_cache.get(key)
        if cached is not None:
            self._bits_cache.move_to_end(key)
            return cached
        seed = key[0]
        k = key[2]
        rng = _Xorshift64(seed)
        ids = sample_green_ids(rng, vocab_size, k)
        bits = ids_to_bitset(ids, vocab_size)
        self._bits_cache[key] = bits
        if len(self._bits_cache) > _BITSET_CACHE_MAX:
            self._bits_cache.popitem(last=False)
        return bits

    def get_green_list(self, prev_token_ids: Sequence[int], vocab_size: int) -> set[int]:
        """Return green-list token ids for the given previous context."""
        if vocab_size <= 0:
            return set()
        return bitset_to_set(self._green_bits(prev_token_ids, vocab_size), vocab_size)

    def score_token(
        self,
        token_id: int,
        prev_token_ids: Sequence[int],
        vocab_size: int,
    ) -> bool:
        return bitset_contains(self._green_bits(prev_token_ids, vocab_size), token_id)


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
