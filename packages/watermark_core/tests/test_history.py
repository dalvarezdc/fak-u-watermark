"""History store tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from watermark_core.history import HistoryStore


def test_history_roundtrip(tmp_path):
    store = HistoryStore(tmp_path / "t.db")
    e = store.add("text", "Hello", {"foo": 1})
    assert e.id
    listed = store.list(kind="text")
    assert len(listed) == 1
    assert listed[0].payload["foo"] == 1
    got = store.get(e.id)
    assert got is not None
    assert got.title == "Hello"
    assert store.delete(e.id)
    assert store.get(e.id) is None
