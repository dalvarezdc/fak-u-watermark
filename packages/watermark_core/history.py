"""Local SQLite history for text analyses and cleaned outputs."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def default_db_path() -> Path:
    root = Path.home() / ".faku"
    root.mkdir(parents=True, exist_ok=True)
    return root / "history.db"


@dataclass
class HistoryEntry:
    id: str
    kind: str  # "text" | "image"
    created_at: float
    title: str
    payload: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "created_at": self.created_at,
            "title": self.title,
            "payload": self.payload,
        }


class HistoryStore:
    """Simple local history backed by SQLite."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    title TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_created ON history(created_at DESC)"
            )

    def add(
        self,
        kind: str,
        title: str,
        payload: dict[str, Any],
        entry_id: str | None = None,
    ) -> HistoryEntry:
        entry = HistoryEntry(
            id=entry_id or str(uuid.uuid4()),
            kind=kind,
            created_at=time.time(),
            title=title[:200] or "Untitled",
            payload=payload,
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO history (id, kind, created_at, title, payload) VALUES (?, ?, ?, ?, ?)",
                (
                    entry.id,
                    entry.kind,
                    entry.created_at,
                    entry.title,
                    json.dumps(entry.payload, ensure_ascii=False),
                ),
            )
        return entry

    def list(
        self,
        kind: str | None = None,
        limit: int = 50,
    ) -> list[HistoryEntry]:
        query = "SELECT * FROM history"
        params: list[Any] = []
        if kind:
            query += " WHERE kind = ?"
            params.append(kind)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_entry(r) for r in rows]

    def choices(self, kind: str | None = None, limit: int = 40) -> list[str]:
        """Dropdown labels: ``{id8} · {timestamp} · {title}``."""
        labels: list[str] = []
        for e in self.list(kind=kind, limit=limit):
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.created_at))
            labels.append(f"{e.id[:8]} · {ts} · {e.title[:70]}")
        return labels

    def get_by_choice(self, label: str | None) -> HistoryEntry | None:
        """Resolve a choices() label (or bare id prefix) to an entry."""
        if not label or not str(label).strip():
            return None
        label = str(label).strip()
        prefix = label.split("·", 1)[0].strip()
        # Prefer full-id match via prefix
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM history WHERE id LIKE ? ORDER BY created_at DESC LIMIT 1",
                (f"{prefix}%",),
            ).fetchone()
        return _row_to_entry(row) if row else None

    def get(self, entry_id: str) -> HistoryEntry | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM history WHERE id = ?", (entry_id,)
            ).fetchone()
        return _row_to_entry(row) if row else None

    def delete(self, entry_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM history WHERE id = ?", (entry_id,))
            return cur.rowcount > 0

    def clear(self, kind: str | None = None) -> int:
        with self._connect() as conn:
            if kind:
                cur = conn.execute("DELETE FROM history WHERE kind = ?", (kind,))
            else:
                cur = conn.execute("DELETE FROM history")
            return cur.rowcount


def _row_to_entry(row: sqlite3.Row) -> HistoryEntry:
    return HistoryEntry(
        id=row["id"],
        kind=row["kind"],
        created_at=row["created_at"],
        title=row["title"],
        payload=json.loads(row["payload"]),
    )
