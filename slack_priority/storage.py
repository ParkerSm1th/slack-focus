from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import DB_PATH, LABEL_SCORES, ensure_dirs


@dataclass(frozen=True)
class Message:
    message_id: str
    channel_id: str | None
    channel_name: str | None
    user_id: str | None
    text: str
    ts: str | None
    permalink: str | None
    score: float | None
    level: str | None
    label: str | None
    created_at: float


class Store:
    def __init__(self, path: Path = DB_PATH):
        ensure_dirs()
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.migrate()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def migrate(self) -> None:
        with self._lock:
            self.conn.execute("pragma journal_mode=wal")
            self.conn.executescript(
                """
                create table if not exists messages (
                    message_id text primary key,
                    channel_id text,
                    channel_name text,
                    user_id text,
                    text text not null,
                    ts text,
                    permalink text,
                    raw_json text,
                    created_at real not null,
                    updated_at real not null
                );

                create table if not exists labels (
                    message_id text primary key references messages(message_id),
                    label text not null,
                    score real not null,
                    updated_at real not null
                );

                create table if not exists predictions (
                    message_id text primary key references messages(message_id),
                    score real not null,
                    level text not null,
                    model_version text not null,
                    created_at real not null
                );

                create table if not exists settings (
                    key text primary key,
                    value text not null
                );
                """
            )
            self.conn.commit()

    def upsert_message(
        self,
        *,
        message_id: str,
        text: str,
        channel_id: str | None = None,
        channel_name: str | None = None,
        user_id: str | None = None,
        ts: str | None = None,
        permalink: str | None = None,
        raw: dict | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                insert into messages (
                    message_id, channel_id, channel_name, user_id, text, ts, permalink,
                    raw_json, created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(message_id) do update set
                    channel_id = excluded.channel_id,
                    channel_name = excluded.channel_name,
                    user_id = excluded.user_id,
                    text = excluded.text,
                    ts = excluded.ts,
                    permalink = coalesce(excluded.permalink, messages.permalink),
                    raw_json = excluded.raw_json,
                    updated_at = excluded.updated_at
                """,
                (
                    message_id,
                    channel_id,
                    channel_name,
                    user_id,
                    text,
                    ts,
                    permalink,
                    json.dumps(raw or {}),
                    now,
                    now,
                ),
            )
            self.conn.commit()

    def message_exists(self, message_id: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "select 1 from messages where message_id = ? limit 1",
                (message_id,),
            ).fetchone()
        return row is not None

    def set_label(self, message_id: str, label: str) -> None:
        if label not in LABEL_SCORES:
            raise ValueError(f"Unknown label: {label}")
        with self._lock:
            self.conn.execute(
                """
                insert into labels (message_id, label, score, updated_at)
                values (?, ?, ?, ?)
                on conflict(message_id) do update set
                    label = excluded.label,
                    score = excluded.score,
                    updated_at = excluded.updated_at
                """,
                (message_id, label, LABEL_SCORES[label], time.time()),
            )
            self.conn.commit()

    def upsert_prediction(
        self,
        *,
        message_id: str,
        score: float,
        level: str,
        model_version: str,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                insert into predictions (message_id, score, level, model_version, created_at)
                values (?, ?, ?, ?, ?)
                on conflict(message_id) do update set
                    score = excluded.score,
                    level = excluded.level,
                    model_version = excluded.model_version,
                    created_at = excluded.created_at
                """,
                (message_id, score, level, model_version, time.time()),
            )
            self.conn.commit()

    def list_labeled_examples(self) -> list[tuple[str, float]]:
        with self._lock:
            rows = self.conn.execute(
                """
                select m.text, l.score
                from labels l
                join messages m on m.message_id = l.message_id
                order by l.updated_at desc
                """
            ).fetchall()
        return [(row["text"], float(row["score"])) for row in rows]

    def review_messages(self, limit: int = 100) -> list[Message]:
        with self._lock:
            rows = self.conn.execute(
                """
                select
                    m.message_id, m.channel_id, m.channel_name, m.user_id, m.text, m.ts,
                    m.permalink, m.created_at,
                    p.score, p.level,
                    l.label
                from messages m
                left join predictions p on p.message_id = m.message_id
                left join labels l on l.message_id = m.message_id
                order by
                    case when l.label is null then 0 else 1 end,
                    abs(coalesce(p.score, 0.5) - 0.75) asc,
                    m.created_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [self._message_from_row(row) for row in rows]

    def recent_messages(self, limit: int = 1000) -> list[Message]:
        with self._lock:
            rows = self.conn.execute(
                """
                select
                    m.message_id, m.channel_id, m.channel_name, m.user_id, m.text, m.ts,
                    m.permalink, m.created_at,
                    p.score, p.level,
                    l.label
                from messages m
                left join predictions p on p.message_id = m.message_id
                left join labels l on l.message_id = m.message_id
                order by coalesce(p.created_at, m.created_at) desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [self._message_from_row(row) for row in rows]

    def unlabeled_count(self) -> int:
        with self._lock:
            row = self.conn.execute(
                """
                select count(*) as count
                from messages m
                left join labels l on l.message_id = m.message_id
                where l.message_id is null
                """
            ).fetchone()
        return int(row["count"])

    def _message_from_row(self, row: sqlite3.Row) -> Message:
        return Message(
            message_id=row["message_id"],
            channel_id=row["channel_id"],
            channel_name=row["channel_name"],
            user_id=row["user_id"],
            text=row["text"],
            ts=row["ts"],
            permalink=row["permalink"],
            score=row["score"],
            level=row["level"],
            label=row["label"],
            created_at=row["created_at"],
        )
