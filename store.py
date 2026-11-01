"""SQLite log of every chat completion, so whole conversations can be replayed in the simulator.

We store token ids, not activations: the forward pass is a pure function of the ids, so the
simulator regenerates every layer's values on demand by re-running them (see tracer.py).

    sessions     one row per conversation (the chat page's "New chat" starts a new one)
    generations  one row per /v1/chat/completions call: request, prompt ids, sampled ids, result
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "virtualcuda.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id       TEXT PRIMARY KEY,
    created  REAL NOT NULL,
    updated  REAL NOT NULL,
    title    TEXT NOT NULL,
    model    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL REFERENCES sessions(id),
    idx               INTEGER NOT NULL,
    created           REAL NOT NULL,
    request           TEXT NOT NULL,  -- json: messages, tools, temperature, top_p, max_tokens
    prompt_ids        TEXT NOT NULL,  -- json list
    completion_ids    TEXT NOT NULL,  -- json list
    content           TEXT NOT NULL,
    tool_calls        TEXT,           -- json list or NULL
    finish_reason     TEXT NOT NULL,
    duration          REAL NOT NULL,
    UNIQUE (session_id, idx)
);
"""


def connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def title_of(content):
    """A chat's title: its first user message, with media shown as 🖼 / 🔊."""
    if isinstance(content, str):
        return content
    marks = {"image_url": "🖼", "input_audio": "🔊"}
    text = " ".join(p.get("text", "") for p in content or [] if p.get("type") == "text").strip()
    media = "".join(marks.get(p.get("type"), "") for p in content or [])
    return f"{media} {text}".strip()


def record(session_id, model, request, prompt_ids, completion_ids, content, tool_calls,
           finish_reason, duration):
    """Append a generation to the session (creating it if needed). Returns its index."""
    now = time.time()
    with connect() as conn:
        if not conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone():
            first_user = next((m.get("content") for m in request["messages"] if m["role"] == "user"), "")
            conn.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
                         (session_id, now, now, (title_of(first_user) or "untitled")[:80], model))
        idx = conn.execute("SELECT COUNT(*) FROM generations WHERE session_id = ?",
                           (session_id,)).fetchone()[0]
        conn.execute(
            "INSERT INTO generations (session_id, idx, created, request, prompt_ids, completion_ids,"
            " content, tool_calls, finish_reason, duration) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, idx, now, json.dumps(request), json.dumps(prompt_ids),
             json.dumps(completion_ids), content, json.dumps(tool_calls) if tool_calls else None,
             finish_reason, duration))
        conn.execute("UPDATE sessions SET updated = ? WHERE id = ?", (now, session_id))
    return idx


def list_sessions(limit=100, q=""):
    """Newest first. q: only chats whose title or any message / reply contains it."""
    where, args = "", []
    if q:
        like = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        where = (" WHERE s.title LIKE ? ESCAPE '\\' OR EXISTS (SELECT 1 FROM generations x"
                 " WHERE x.session_id = s.id AND (x.content LIKE ? ESCAPE '\\' OR x.request LIKE ? ESCAPE '\\'))")
        args = [like] * 3
    with connect() as conn:
        rows = conn.execute(
            "SELECT s.*, COUNT(g.id) AS generations FROM sessions s"
            " LEFT JOIN generations g ON g.session_id = s.id" + where +
            " GROUP BY s.id ORDER BY s.updated DESC LIMIT ?", (*args, limit)).fetchall()
    return [dict(r) for r in rows]


def rename_session(session_id, title):
    with connect() as conn:
        return conn.execute("UPDATE sessions SET title = ? WHERE id = ?",
                            (title[:80], session_id)).rowcount > 0


def _generation(row, with_ids):
    g = dict(row)
    g["request"] = json.loads(g["request"])
    g["tool_calls"] = json.loads(g["tool_calls"]) if g["tool_calls"] else None
    prompt_ids, completion_ids = json.loads(g["prompt_ids"]), json.loads(g["completion_ids"])
    g["prompt_tokens"], g["completion_tokens"] = len(prompt_ids), len(completion_ids)
    if with_ids:
        g["prompt_ids"], g["completion_ids"] = prompt_ids, completion_ids
    else:
        del g["prompt_ids"], g["completion_ids"]
    return g


def get_session(session_id):
    with connect() as conn:
        s = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not s:
            return None
        rows = conn.execute("SELECT * FROM generations WHERE session_id = ? ORDER BY idx",
                            (session_id,)).fetchall()
    return {**dict(s), "generations": [_generation(r, with_ids=False) for r in rows]}


def delete_session(session_id):
    with connect() as conn:
        conn.execute("DELETE FROM generations WHERE session_id = ?", (session_id,))
        return conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount > 0


def delete_sessions(ids=None):
    """Delete these chats, or every chat when ids is None. Returns how many were deleted."""
    with connect() as conn:
        if ids is None:
            conn.execute("DELETE FROM generations")
            return conn.execute("DELETE FROM sessions").rowcount
        marks = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM generations WHERE session_id IN ({marks})", ids)
        return conn.execute(f"DELETE FROM sessions WHERE id IN ({marks})", ids).rowcount


def get_generation(session_id, idx):
    with connect() as conn:
        row = conn.execute("SELECT * FROM generations WHERE session_id = ? AND idx = ?",
                           (session_id, idx)).fetchone()
    return _generation(row, with_ids=True) if row else None
