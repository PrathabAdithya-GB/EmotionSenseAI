import sqlite3
import datetime
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "emotion.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key TEXT UNIQUE,
            candidate   TEXT DEFAULT 'Candidate',
            started_at  TEXT,
            ended_at    TEXT,
            duration_s  INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS emotion_records (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key  TEXT,
            ts           TEXT,
            dominant     TEXT,
            scores_json  TEXT,
            confidence   REAL,
            stress_score REAL,
            stress_level TEXT,
            attention    REAL,
            eye_contact  REAL,
            head_stability REAL,
            smile_score  REAL
        );
        """)


def new_session(session_key: str, candidate: str = "Candidate") -> None:
    with get_conn() as conn:
        now = datetime.datetime.now().isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_key, candidate, started_at) VALUES (?,?,?)",
            (session_key, candidate, now)
        )


def save_record(session_key: str, data: dict) -> None:
    with get_conn() as conn:
        now = datetime.datetime.now().isoformat()
        conn.execute("""
            INSERT INTO emotion_records
            (session_key, ts, dominant, scores_json, confidence, stress_score,
             stress_level, attention, eye_contact, head_stability, smile_score)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            session_key, now,
            data.get("dominant", "Neutral"),
            json.dumps(data.get("scores", {})),
            data.get("confidence_score", 0),
            data.get("stress_score", 0),
            data.get("stress_level", "Low"),
            data.get("attention_score", 0),
            data.get("eye_contact_score", 0),
            data.get("head_stability", 0),
            data.get("smile_score", 0),
        ))


def close_session(session_key: str) -> None:
    with get_conn() as conn:
        now = datetime.datetime.now().isoformat()
        # compute duration
        row = conn.execute(
            "SELECT started_at FROM sessions WHERE session_key=?", (session_key,)
        ).fetchone()
        dur = 0
        if row:
            try:
                start = datetime.datetime.fromisoformat(row["started_at"])
                dur   = int((datetime.datetime.now() - start).total_seconds())
            except Exception:
                pass
        conn.execute(
            "UPDATE sessions SET ended_at=?, duration_s=? WHERE session_key=?",
            (now, dur, session_key)
        )


def get_records(session_key: str) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM emotion_records WHERE session_key=? ORDER BY ts",
            (session_key,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["scores"] = json.loads(d.get("scores_json", "{}"))
            out.append(d)
        return out


def get_session(session_key: str) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_key=?", (session_key,)
        ).fetchone()
        return dict(row) if row else {}


def list_sessions() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT 50"
        ).fetchall()
        return [dict(r) for r in rows]
