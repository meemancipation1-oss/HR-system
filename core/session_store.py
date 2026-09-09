"""Durable session storage for API conversations.

SQLite keeps the local prototype restart-safe without coupling the runtime to
an external service. Messages are stored as JSON so the same format can be
passed back to the OpenAI-compatible client.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any


class SessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    messages_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS task_state (
                    session_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT messages_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return []
        try:
            messages = json.loads(row["messages_json"])
            return messages if isinstance(messages, list) else []
        except (TypeError, json.JSONDecodeError):
            return []

    def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        payload = json.dumps(messages[-100:], ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO sessions(session_id, messages_json, updated_at)
                   VALUES (?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(session_id) DO UPDATE SET
                     messages_json = excluded.messages_json,
                     updated_at = CURRENT_TIMESTAMP""",
                (session_id, payload),
            )

    def delete(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM task_state WHERE session_id = ?", (session_id,))

    def load_task_state(self, session_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM task_state WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return {}
        try:
            state = json.loads(row["state_json"])
            return state if isinstance(state, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}

    def save_task_state(self, session_id: str, state: dict[str, Any]) -> None:
        payload = json.dumps(state, ensure_ascii=False, default=str)
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO task_state(session_id, state_json, updated_at)
                   VALUES (?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(session_id) DO UPDATE SET
                     state_json = excluded.state_json,
                     updated_at = CURRENT_TIMESTAMP""",
                (session_id, payload),
            )


__all__ = ["SessionStore"]
