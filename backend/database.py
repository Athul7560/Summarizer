from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from backend.config import settings


class StudentRepository:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.sqlite_path
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS students (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS quiz_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    score REAL NOT NULL,
                    attempted_at TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY(student_id) REFERENCES students(id)
                );

                CREATE TABLE IF NOT EXISTS agent_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    suggestion_type TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(student_id) REFERENCES students(id)
                );
                """
            )
            conn.commit()

    def upsert_student(self, student_id: str, name: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO students(id, name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name=COALESCE(excluded.name, students.name)
                """,
                (student_id, name, now),
            )
            conn.commit()

    def record_quiz_attempt(self, *, student_id: str, topic: str, score: float, metadata: str = "{}") -> int:
        self.upsert_student(student_id)
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO quiz_attempts(student_id, topic, score, attempted_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_id, topic, score, now, metadata),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def record_feedback(self, *, student_id: str, topic: str, suggestion_type: str, accepted: bool) -> int:
        self.upsert_student(student_id)
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO agent_feedback(student_id, topic, suggestion_type, accepted, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_id, topic, suggestion_type, int(accepted), now),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_recent_attempts(self, student_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, student_id, topic, score, attempted_at, metadata
                FROM quiz_attempts
                WHERE student_id = ?
                ORDER BY attempted_at DESC
                LIMIT ?
                """,
                (student_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_topic_scores(self, student_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT topic,
                       AVG(score) AS avg_score,
                       COUNT(*) AS attempts
                FROM quiz_attempts
                WHERE student_id = ?
                GROUP BY topic
                ORDER BY avg_score ASC
                """,
                (student_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_trend(self, student_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT DATE(attempted_at) AS date, AVG(score) AS avg_score, COUNT(*) AS attempts
                FROM quiz_attempts
                WHERE student_id = ?
                GROUP BY DATE(attempted_at)
                ORDER BY DATE(attempted_at)
                """,
                (student_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_declined_suggestions(self, student_id: str) -> set[tuple[str, str]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT topic, suggestion_type
                FROM agent_feedback
                WHERE student_id = ? AND accepted = 0
                """,
                (student_id,),
            ).fetchall()
        return {(row["topic"], row["suggestion_type"]) for row in rows}


student_repo = StudentRepository()
