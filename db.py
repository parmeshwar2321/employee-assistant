import sqlite3
from contextlib import contextmanager
from .config import settings

APP_DB = str(settings.data_path / "app.db")
CHECKPOINT_DB = str(settings.data_path / "checkpoints.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender TEXT NOT NULL,
    recipients TEXT DEFAULT '',
    message_id TEXT,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    received_at TEXT DEFAULT CURRENT_TIMESTAMP,
    analysis_json TEXT,
    analyzed_at TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def conn():
    c = sqlite3.connect(APP_DB)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db() -> None:
    with conn() as c:
        c.executescript(SCHEMA)
        cols = [r["name"] for r in c.execute("PRAGMA table_info(emails)")]
        if "message_id" not in cols:  # migrate older databases
            c.execute("ALTER TABLE emails ADD COLUMN message_id TEXT")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_emails_msgid ON emails(message_id)")
