from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | None = None) -> None:
    db_path = path or settings.database_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              email TEXT UNIQUE NOT NULL,
              password_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_history (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              mode TEXT NOT NULL,
              detected_rasa TEXT,
              detected_bhava TEXT,
              confidence REAL,
              prahara INTEGER,
              ritu TEXT,
              weather_condition TEXT,
              recommendations_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()


def get_user_by_email(email: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, email, password_hash, created_at FROM users WHERE email = ?",
            (email.lower().strip(),),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, email, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def create_user(email: str, password_hash: str) -> dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email.lower().strip(), password_hash, utc_now()),
        )
        user_id = int(cursor.lastrowid)
        row = conn.execute(
            "SELECT id, email, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row)


def save_recommendation_history(
    *,
    user_id: int,
    mode: str,
    detected_rasa: str | None,
    detected_bhava: str | None,
    confidence: float | None,
    prahara: int | None,
    ritu: str | None,
    weather_condition: str | None,
    recommendations: list[dict[str, Any]],
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO recommendation_history (
              user_id, mode, detected_rasa, detected_bhava, confidence,
              prahara, ritu, weather_condition, recommendations_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                mode,
                detected_rasa,
                detected_bhava,
                confidence,
                prahara,
                ritu,
                weather_condition,
                json.dumps(recommendations),
                utc_now(),
            ),
        )


def list_recommendation_history(user_id: int, limit: int = 20) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, mode, detected_rasa, detected_bhava, confidence,
                   prahara, ritu, weather_condition, recommendations_json, created_at
            FROM recommendation_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["recommendations"] = json.loads(item.pop("recommendations_json"))
        except json.JSONDecodeError:
            item["recommendations"] = []
        items.append(item)
    return items

