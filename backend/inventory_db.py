"""Lightweight SQLite inventory store for shelf scans (Module 4 groundwork).

Every analyzed shelf image is persisted as one `scans` row plus one `items` row per detected
product. This turns the per-image pipeline into a queryable inventory over time, which is what
the Business-Intelligence layer (bi_interface/bi_engine.py) answers questions against.

Dependency-free (stdlib sqlite3 + pandas) so it can later be lifted behind a FastAPI backend
with minimal change.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

# Kept out of git (see .gitignore: *.db). Override with $INVENTORY_DB.
DB_PATH = str(Path(__file__).resolve().parents[1] / "inventory.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    image_name TEXT,
    num_items INTEGER,
    distinct_categories INTEGER,
    empty_pct REAL,
    shelf_type TEXT,
    review_count INTEGER
);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL,
    crop_id INTEGER,
    category TEXT,
    subcategory TEXT,
    score REAL,
    area INTEGER,
    box TEXT,
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_items_scan ON items(scan_id);
CREATE INDEX IF NOT EXISTS idx_items_category ON items(category);
"""


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def save_scan(result, image_name: str, records: list[dict], db_path: str = DB_PATH) -> int:
    """Persist one analysis result. Returns the new scan id."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO scans (ts, image_name, num_items, distinct_categories,
                                  empty_pct, shelf_type, review_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(timespec="seconds"), image_name, result.num_items,
             result.distinct_categories, result.empty_pct, result.shelf_type,
             result.review_count),
        )
        scan_id = int(cur.lastrowid)
        conn.executemany(
            """INSERT INTO items (scan_id, crop_id, category, subcategory, score, area, box)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(scan_id, r["crop_id"], r["category"], r["subcategory"], r["score"], r["area"],
              json.dumps(r["box"])) for r in records],
        )
    return scan_id


def get_scans_df(db_path: str = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM scans ORDER BY id DESC", conn)


def get_items_df(scan_id: int | None = None, db_path: str = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with _connect(db_path) as conn:
        if scan_id is None:
            return pd.read_sql_query("SELECT * FROM items", conn)
        return pd.read_sql_query("SELECT * FROM items WHERE scan_id = ?", conn, params=(scan_id,))


def clear_all(db_path: str = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript("DELETE FROM items; DELETE FROM scans;")


def stats(db_path: str = DB_PATH) -> dict:
    scans = get_scans_df(db_path)
    items = get_items_df(db_path=db_path)
    known = items[items["category"].str.lower() != "unknown"] if not items.empty else items
    return {
        "total_scans": int(len(scans)),
        "total_items": int(len(items)),
        "distinct_categories": int(known["category"].nunique()) if not known.empty else 0,
    }
