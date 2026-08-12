"""Shared dependencies and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure packages/ is importable when running from repo root
_ROOT = Path(__file__).resolve().parent.parent
_PACKAGES = _ROOT / "packages"
if str(_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_PACKAGES))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def get_history_store():
    from watermark_core.history import HistoryStore

    return HistoryStore()
