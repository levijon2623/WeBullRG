"""SQLite state: orders seen, positions (for P&L), the outbox, and meta.

One file, stdlib only. The OUTBOX is the important table: every (event, sink)
pair is written there first and sent from there, so a configured delay, a
sink that is down, or a restart in the middle of a send never loses a post and
never sends one twice (UNIQUE(sink, event_key)).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    source TEXT, order_id TEXT, state TEXT, filled REAL, seen_at REAL,
    PRIMARY KEY (source, order_id));
CREATE TABLE IF NOT EXISTS positions (
    source TEXT, inst TEXT, qty REAL, avg_cost REAL,
    PRIMARY KEY (source, inst));
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT, sink TEXT, event_key TEXT,
    payload TEXT, ready_at REAL, attempts INTEGER DEFAULT 0,
    sent_at REAL, last_error TEXT, UNIQUE (sink, event_key));
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


class Store:
    def __init__(self, path: str):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---- meta
    def get_meta(self, k, default=None):
        r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_meta(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (k, json.dumps(v)))
        self.db.commit()

    # ---- orders
    def get_order(self, source, oid):
        r = self.db.execute("SELECT state, filled FROM orders WHERE source=? AND order_id=?",
                            (source, oid)).fetchone()
        return (r[0], r[1]) if r else None

    def put_order(self, source, oid, state, filled):
        self.db.execute("INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?)",
                        (source, oid, state, float(filled), time.time()))

    # ---- positions (average cost, long side)
    def get_position(self, source, inst):
        r = self.db.execute("SELECT qty, avg_cost FROM positions WHERE source=? AND inst=?",
                            (source, inst)).fetchone()
        return (r[0], r[1]) if r else (0.0, None)

    def put_position(self, source, inst, qty, avg_cost):
        if qty <= 1e-9:
            self.db.execute("DELETE FROM positions WHERE source=? AND inst=?", (source, inst))
        else:
            self.db.execute("INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?)",
                            (source, inst, qty, avg_cost))

    # ---- outbox
    def enqueue(self, sink, event_key, payload: dict, ready_at: float) -> bool:
        cur = self.db.execute(
            "INSERT OR IGNORE INTO outbox (sink, event_key, payload, ready_at) VALUES (?, ?, ?, ?)",
            (sink, event_key, json.dumps(payload), ready_at))
        return cur.rowcount == 1

    def due(self, now: float, max_attempts: int):
        return self.db.execute(
            "SELECT id, sink, event_key, payload, attempts FROM outbox "
            "WHERE sent_at IS NULL AND ready_at <= ? AND attempts < ? ORDER BY id",
            (now, max_attempts)).fetchall()

    def mark_sent(self, row_id):
        self.db.execute("UPDATE outbox SET sent_at=? WHERE id=?", (time.time(), row_id))
        self.db.commit()

    def mark_failed(self, row_id, err: str, retry_at: float):
        self.db.execute("UPDATE outbox SET attempts=attempts+1, last_error=?, ready_at=? WHERE id=?",
                        (err[:200], retry_at, row_id))
        self.db.commit()

    def commit(self):
        self.db.commit()
