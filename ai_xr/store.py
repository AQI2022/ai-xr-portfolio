import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class Store:
    """Small persistent store. Each operation owns its connection; no shared async transaction."""
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                  id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, title TEXT NOT NULL,
                  page INTEGER, start INTEGER, end INTEGER, text TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
                CREATE TABLE IF NOT EXISTS memory (
                  id INTEGER PRIMARY KEY, session TEXT NOT NULL, role TEXT NOT NULL,
                  content TEXT NOT NULL, created REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS memory_session ON memory(session, id);
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS pending_actions (
                  id TEXT PRIMARY KEY, session TEXT NOT NULL, body TEXT NOT NULL,
                  created REAL NOT NULL, used INTEGER NOT NULL DEFAULT 0);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def all_chunks(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM chunks ORDER BY id")]

    def replace_document(self, doc_id, chunks):
        with self.lock, self.connect() as db:
            db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            db.executemany("INSERT INTO chunks VALUES (:id,:doc_id,:title,:page,:start,:end,:text)", chunks)

    def append_memory(self, session, role, content):
        with self.connect() as db:
            db.execute("INSERT INTO memory(session,role,content,created) VALUES (?,?,?,?)",
                       (session, role, content[:12000], time.time()))
            db.execute("DELETE FROM memory WHERE session=? AND id NOT IN "
                       "(SELECT id FROM memory WHERE session=? ORDER BY id DESC LIMIT 24)", (session, session))

    def history(self, session, limit=8):
        with self.connect() as db:
            rows = db.execute("SELECT role,content FROM memory WHERE session=? ORDER BY id DESC LIMIT ?",
                              (session, limit)).fetchall()
            return [dict(row) for row in reversed(rows)]

    def delete_memory(self, session):
        with self.connect() as db:
            db.execute("DELETE FROM memory WHERE session=?", (session,))
            db.execute("DELETE FROM pending_actions WHERE session=?", (session,))

    def add_pending(self, action_id, session, action):
        with self.connect() as db:
            db.execute("INSERT INTO pending_actions(id,session,body,created) VALUES (?,?,?,?)",
                       (action_id, session, json.dumps(action), time.time()))

    def consume_pending(self, action_id, session):
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM pending_actions WHERE id=? AND session=? AND used=0",
                             (action_id, session)).fetchone()
            if row is None or time.time() - row["created"] > 300:
                return None
            db.execute("UPDATE pending_actions SET used=1 WHERE id=?", (action_id,))
            return json.loads(row["body"])

    def save_jobs(self, jobs):
        with self.connect() as db:
            db.executemany("INSERT OR REPLACE INTO jobs VALUES (?,?)",
                           [(job["id"], json.dumps(job, ensure_ascii=False)) for job in jobs])

    def list_jobs(self):
        with self.connect() as db:
            return [json.loads(row["body"]) for row in db.execute("SELECT body FROM jobs ORDER BY id")]
